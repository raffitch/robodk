"""SAM (point-prompted) work-surface boundary — learned segmentation, host-side.

The classical colour segmenter (``color_boundary.color_work_boundary``) draws a clean live
rectangle on distinct/contrasty objects but *abstains* on genuinely low-contrast scenes —
measured on the real cell, a dark green cutting mat on a gray table is only ~5 Lab units
apart, so no single threshold isolates it (it grabbed the whole frame). A learned,
point-prompted model segments by object-ness / edges instead of one threshold, so it nails
that scene: EdgeSAM keyed off the reticle centre hugged the mat's true edges (score 0.98,
no frame-border touch) where colour could only fall back to depth.

This module runs such a model on the **Windows host** via ONNXRuntime (the Jetson never runs
SAM — it just streams frames). It is written **model-agnostic**: it reads the encoder input
size and the decoder's input/output names off the ONNX graph, so both EdgeSAM's simplified
decoder (``image_embeddings, point_coords, point_labels`` -> ``scores, masks``) and the
standard SAM / MobileSAM decoder (extra ``mask_input, has_mask_input, orig_im_size`` ->
``masks, iou_predictions``) drop in by just pointing the config at different weight files.

Two entry points:

- :func:`sam_work_boundary` — one frame in, a normalized-uv boundary out (or ``None``);
  reuses ``color_boundary.mask_to_boundary`` for the mask -> rectangle tail so SAM and colour
  share identical abstain-safe geometry.
- :class:`SamBoundaryWorker` — a background thread the live loop feeds the latest frame;
  it runs SAM (~450 ms/frame on CPU) off the video thread so the ~6 fps preview never
  hitches, publishing the boundary at whatever rate SAM sustains (~2 fps) with an optional
  colour fallback when SAM abstains or its weights are absent.

**Licensing:** EdgeSAM's weights are S-Lab License 1.0 (**non-commercial research only**);
MobileSAM's are Apache-2.0. Weights are NOT vendored in the repo (see ``models/README.md`` /
``tools/download_sam.py``) — pick the one your use permits.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import cv2

from .color_boundary import mask_to_boundary

# SAM preprocessing constants (RGB, 0-255). EdgeSAM and MobileSAM both inherit SAM's
# image normalization; the ONNX encoders here take the *preprocessed* tensor (they do not
# bake normalization), validated empirically against EdgeSAM.
_PIXEL_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
_PIXEL_STD = np.array([58.395, 57.12, 57.375], np.float32)


class SamUnavailable(RuntimeError):
    """Raised when SAM cannot run: onnxruntime missing, or weight files absent/invalid.

    The caller (dispatch in the live loop) catches this to fall back to the colour
    segmenter and log it once, so a missing model degrades gracefully instead of killing
    the video or the boundary layer.
    """


class _Sessions:
    """Lazy, cached ONNXRuntime encoder+decoder pair, adaptive to the graph signature."""

    def __init__(self, enc_path: Path, dec_path: Path):
        try:
            import onnxruntime as ort
        except Exception as e:  # pragma: no cover - environment dependent
            raise SamUnavailable(f"onnxruntime not installed ({e}); pip install -e .[sam]")
        if not enc_path.exists() or not dec_path.exists():
            raise SamUnavailable(
                f"SAM weights missing ({enc_path.name}/{dec_path.name}); run "
                f"`py -3.10 tools/download_sam.py`")
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, (os_cpu() - 1))
            prov = ["CPUExecutionProvider"]
            self.enc = ort.InferenceSession(str(enc_path), sess_options=opts, providers=prov)
            self.dec = ort.InferenceSession(str(dec_path), sess_options=opts, providers=prov)
        except Exception as e:  # pragma: no cover
            raise SamUnavailable(f"failed to load SAM ONNX ({e})")
        self.enc_in = self.enc.get_inputs()[0].name
        # Encoder square input size (e.g. 1024); fall back to 1024 if the graph is dynamic.
        shp = self.enc.get_inputs()[0].shape
        self.enc_size = int(shp[-1]) if isinstance(shp[-1], int) and shp[-1] > 0 else 1024
        self.dec_in = {i.name for i in self.dec.get_inputs()}
        self.dec_out = [o.name for o in self.dec.get_outputs()]


def os_cpu() -> int:
    import os
    return os.cpu_count() or 2


# module-level cache so the ~40 MB sessions load once, not per frame
_CACHE: dict[tuple[str, str], _Sessions] = {}
_CACHE_LOCK = threading.Lock()


def _sessions(model_dir: str, enc_file: str, dec_file: str) -> _Sessions:
    enc_path = Path(model_dir) / enc_file
    dec_path = Path(model_dir) / dec_file
    key = (str(enc_path.resolve()), str(dec_path.resolve()))
    with _CACHE_LOCK:
        s = _CACHE.get(key)
        if s is None:
            s = _Sessions(enc_path, dec_path)
            _CACHE[key] = s
        return s


def _preprocess(bgr, size: int):
    """RGB, ResizeLongestSide(size), SAM-normalize, pad to size x size -> (1,3,S,S)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    H, W = rgb.shape[:2]
    scale = float(size) / float(max(H, W))
    nH, nW = int(round(H * scale)), int(round(W * scale))
    r = cv2.resize(rgb, (nW, nH), interpolation=cv2.INTER_LINEAR)
    r = (r - _PIXEL_MEAN) / _PIXEL_STD
    canvas = np.zeros((size, size, 3), np.float32)
    canvas[:nH, :nW] = r
    return canvas.transpose(2, 0, 1)[None], scale, (H, W), (nH, nW)


def _decode(s: _Sessions, emb, point_xy_resized, orig_hw):
    """Run the decoder adaptively for the point; return (scores[N], masks[N,mh,mw])."""
    H, W = orig_hw
    px, py = point_xy_resized
    # Standard SAM decoders expect a padding point (label -1) when there is no box.
    standard = "orig_im_size" in s.dec_in or "mask_input" in s.dec_in
    if standard:
        pc = np.array([[[px, py], [0.0, 0.0]]], np.float32)   # (1,2,2)
        pl = np.array([[1.0, -1.0]], np.float32)              # (1,2)
    else:
        pc = np.array([[[px, py]]], np.float32)               # (1,1,2)
        pl = np.array([[1.0]], np.float32)                    # (1,1)
    feed = {"image_embeddings": emb, "point_coords": pc, "point_labels": pl}
    if "mask_input" in s.dec_in:
        feed["mask_input"] = np.zeros((1, 1, 256, 256), np.float32)
    if "has_mask_input" in s.dec_in:
        feed["has_mask_input"] = np.zeros((1,), np.float32)
    if "orig_im_size" in s.dec_in:
        feed["orig_im_size"] = np.array([H, W], np.float32)
    outs = s.dec.run(None, feed)
    named = dict(zip(s.dec_out, outs))
    masks = None
    scores = None
    for name, arr in named.items():
        low = name.lower()
        if "mask" in low and masks is None:
            masks = np.asarray(arr)
        elif ("score" in low or "iou" in low) and scores is None:
            scores = np.asarray(arr)
    if masks is None:
        # last resort: the biggest 4-D output is the masks
        masks = max((np.asarray(a) for a in outs if np.asarray(a).ndim == 4),
                    key=lambda a: a.size, default=None)
    if masks is None:
        raise SamUnavailable("decoder produced no mask output")
    m = np.asarray(masks)
    if m.ndim == 4:
        m = m[0]                      # (N, mh, mw)
    elif m.ndim == 3:
        pass                          # already (N, mh, mw)
    else:
        m = m[None]
    if scores is None:
        sc = np.ones((m.shape[0],), np.float32)
    else:
        sc = np.asarray(scores).ravel().astype(np.float32)
        if sc.size < m.shape[0]:
            sc = np.pad(sc, (0, m.shape[0] - sc.size), constant_values=0.0)
    return sc[: m.shape[0]], m


def sam_work_boundary(
    color_bgr,
    *,
    point_uv=(0.5, 0.5),
    model_dir: str = "models",
    encoder_file: str = "edge_sam_encoder.onnx",
    decoder_file: str = "edge_sam_decoder.onnx",
    min_score: float = 0.80,
    min_fill_frac: float = 0.01,
    max_fill_frac: float = 0.92,
    border_touch_frac: float = 0.60,
) -> dict | None:
    """Segment the object under ``point_uv`` with SAM; return its boundary as normalized uv.

    Returns the same dict shape as ``color_work_boundary`` —
    ``{outline_uv, polygon_uv, contrast, fill_frac, border_touch, overruns}`` — with
    ``contrast`` = the model's confidence (IoU/stability score), so the HUD and the
    ``boundary`` event consume it identically. ``None`` = abstain (low score, or the mask
    is implausibly small / fills the frame); the HUD then falls back to the depth outline.

    ``point_uv`` is the prompt in normalized (0-1) coords over the colour frame — the reticle
    centre (0.5, 0.5) by default. Raises :class:`SamUnavailable` if onnxruntime or the weight
    files are absent (the dispatcher catches it and falls back to colour).
    """
    if color_bgr is None:
        return None
    img = np.asarray(color_bgr)
    if img.ndim != 3 or img.shape[2] < 3 or img.shape[0] < 8 or img.shape[1] < 8:
        return None
    s = _sessions(model_dir, encoder_file, decoder_file)
    H, W = img.shape[:2]
    chw, scale, _, (nH, nW) = _preprocess(img, s.enc_size)
    emb = s.enc.run(None, {s.enc_in: chw})[0]
    px = float(point_uv[0]) * W * scale
    py = float(point_uv[1]) * H * scale
    scores, masks = _decode(s, emb, (px, py), (H, W))
    if scores.size == 0:
        return None
    best = int(np.argmax(scores))
    score = float(scores[best])
    if score < float(min_score):
        return None                          # model is unsure -> abstain
    mm = masks[best].astype(np.float32)       # logits, > 0 = object
    mh, mw = mm.shape
    if (mh, mw) == (H, W):
        binm = (mm > 0).astype(np.uint8)      # standard SAM already upsampled to orig
    else:
        # low-res mask (e.g. 256): it maps to the padded encoder frame; undo the
        # letterbox (crop to the resized region) then resize to the original frame.
        up = cv2.resize(mm, (s.enc_size, s.enc_size), interpolation=cv2.INTER_LINEAR)
        up = up[:nH, :nW]
        up = cv2.resize(up, (W, H), interpolation=cv2.INTER_LINEAR)
        binm = (up > 0).astype(np.uint8)

    rx, ry = float(point_uv[0]) * W, float(point_uv[1]) * H
    out = mask_to_boundary(
        binm, (rx, ry),
        min_fill_frac=min_fill_frac,
        max_fill_frac=max_fill_frac,
        border_touch_frac=border_touch_frac)
    if out is None:
        return None
    out["contrast"] = score
    return out


class SamBoundaryWorker:
    """Run SAM off the video thread: feed it the latest frame, it publishes the boundary.

    SAM is ~450 ms/frame on host CPU — inline in the ~6 fps live loop it would hitch the
    video (the operator's "laggy" complaint). This worker takes the newest frame (dropping
    any it couldn't keep up with), runs SAM (+ optional colour fallback), and publishes a
    ``boundary`` event, so the video stays full-rate and the box updates at whatever SAM
    sustains. Weights load once in the worker thread; if they're missing it flips to
    colour-only (or idle) and logs once — it never raises into the video path.
    """

    def __init__(self, publish, *, model_dir, encoder_file, decoder_file,
                 min_score, max_fill_frac, point_uv=(0.5, 0.5), fallback=None, log=None):
        self._publish = publish                # (dict) -> None ; builds+emits the event
        self._fallback = fallback              # (bgr) -> dict|None ; colour segmenter or None
        self._log = log or (lambda m: None)
        self._kw = dict(model_dir=model_dir, encoder_file=encoder_file,
                        decoder_file=decoder_file, min_score=min_score,
                        max_fill_frac=max_fill_frac, point_uv=point_uv)
        self._latest = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._sam_dead = False                 # SAM unavailable -> use fallback only
        self._warned = False
        self._thread = threading.Thread(target=self._loop, name="sam-boundary", daemon=True)
        self._thread.start()

    def submit(self, color_bgr) -> None:
        """Hand the worker the latest colour frame (cheap; overwrites any unprocessed one)."""
        if self._stop.is_set():
            return
        with self._lock:
            self._latest = color_bgr
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _warn_once(self, msg: str) -> None:
        if not self._warned:
            self._warned = True
            self._log(msg)

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(timeout=1.0):
                continue
            self._wake.clear()
            with self._lock:
                color = self._latest
                self._latest = None
            if color is None or self._stop.is_set():
                continue
            cb = None
            if not self._sam_dead:
                try:
                    cb = sam_work_boundary(color, **self._kw)
                except SamUnavailable as e:
                    self._sam_dead = True
                    self._warn_once(
                        f"[scan] SAM boundary unavailable, "
                        f"{'falling back to colour' if self._fallback else 'disabled'}: {e}")
                except Exception as e:  # never kill the boundary layer on a bad frame
                    self._warn_once(f"[scan] SAM boundary error (continuing): {e}")
                    cb = None
            if cb is None and self._fallback is not None:
                try:
                    cb = self._fallback(color)
                except Exception:
                    cb = None
            if cb is not None and not self._stop.is_set():
                try:
                    self._publish(cb)
                except Exception:
                    pass
