# Handoff: add SAM (point-prompted) for the live scan work boundary

> **✅ IMPLEMENTED (2026-07-06).** SAM is wired end-to-end and verified on the real
> green-mat cell (EdgeSAM hugged the mat, score 0.98, where colour abstained). Details:
> - Engine: `tasni/modules/scan/sam_boundary.py` — model-agnostic ONNX (reads the graph
>   signature, so EdgeSAM's simplified decoder AND MobileSAM's standard SAM decoder drop
>   in). Runs in a **background worker** (`SamBoundaryWorker`) off the video thread, so
>   ~450 ms/frame inference never hitches the ~6 fps preview; boundary updates at ~2 fps.
> - Shared tail factored to `color_boundary.mask_to_boundary` (colour + SAM use identical
>   abstain-safe geometry). Config: `scan.boundary_engine` (default `sam_then_color`) +
>   `sam_*` knobs. Dispatch + worker lifecycle in `module.py` (`live_start`/`live_stop`).
> - **Default model = EdgeSAM** (S-Lab **non-commercial** license — see below). MobileSAM
>   (Apache-2.0) drops in via config for commercial use.
> - **Windows gotcha (fixed):** onnxruntime's DLL init fails if PySide2/Qt (pulled by
>   RoboDK `robolink`) loads first → onnxruntime is pre-loaded in `tasni/__init__.py`
>   (and `tests/conftest.py`) before any robolink import.
> - Weights are **not** committed (`.gitignore`); fetch with `tools/download_sam.py`
>   (`pip install -e .[sam]` first). Frontend unchanged — same `boundary` event.
>
> The sections below are the original design notes (kept for reasoning history).

**Goal:** make the live blue work-rectangle reliable on *any* scene — including
low-contrast ones the classical color layer can't handle — by segmenting the object
under the reticle with a learned, point-prompted model (SAM family) on the host.

**Status when this was written (2026-07-06):** the *plumbing* is done and shipped
(commit `1dd8d21` on `calibration-improvements`). A classical color segmenter
(`color_work_boundary`) already produces a video-rate boundary and the whole
event/render path is live. SAM is a **drop-in replacement/augmentation of one function** —
if it emits the same payload, **no frontend or transport changes are needed.** The user
explicitly chose to add SAM; they'll do it in a fresh session.

Read `docs/agent-debug-map.md` and `docs/live-robot-testing.md` first if you're new to
the cell. This doc assumes that context.

---

## 1. Why SAM (the finding that motivated this)

The live boundary was depth-only: fitted from noisy RealSense depth, throttled to the
server's 1 Hz `SCAN_TELEMETRY_PERIOD_S` telemetry, and frozen by the anti-jitter hold —
so it flickered, lagged, and "only updated on Refresh."

We moved the boundary to the **color** image (host decodes it at ~6 fps) and segment it
there. The classical segmenter (`color_work_boundary`, Lab colour distance, reticle-
seeded) works well on **distinct / contrasty objects** but **abstains on genuinely
low-contrast scenes**. Measured on the actual cell: the object is a **dark green cutting
mat (grid + plaster stains) on a gray metal table with a bright reflective strip at top**.
Mat vs table are ~equal luminance and only **~5 Lab units** apart in chroma, saturation
overlaps — no single threshold isolates it (it grabbed 91% of the frame). That is exactly
where a learned model wins: SAM segments by learned object-ness/edges, not one threshold.

So: **real-time half solved (plumbing + classical), reliability half needs SAM for hard
scenes.**

---

## 2. The seam — what SAM must produce (the contract)

The whole HUD path already consumes a **`boundary`** event. Match this payload and you are
done end-to-end. Producer: `tasni/modules/scan/module.py`, the `analyze(frame)` closure in
the `/live/start` handler. It currently does (paraphrased):

```python
if sc.color_boundary_enabled:
    cb = color_work_boundary(frame.color, reticle_frac=sc.center_patch_frac,
                             min_color_dist=sc.color_boundary_min_color_dist,
                             seg_width=sc.color_boundary_seg_width)
    if cb is not None:
        services.bus.publish(JobEvent("boundary", {
            "outline_uv": cb["outline_uv"],   # 4 min-area-rect corners, normalized 0-1
            "polygon_uv": cb["polygon_uv"],   # simplified contour, normalized 0-1
            "overruns": cb["overruns"],       # object hugs/exceeds the frame edge
            "contrast": cb["contrast"],       # confidence (color: Lab dist; SAM: use score)
        }))
```

**`color_work_boundary` (in `tasni/modules/scan/color_boundary.py`) is the function to
replace or fall back from.** Its interface:

```
color_work_boundary(color_bgr, *, reticle_frac, min_color_dist, min_fill_frac,
                    max_fill_frac, border_touch_frac, seg_width) -> dict | None
# returns {outline_uv, polygon_uv, contrast, fill_frac, border_touch, overruns} or None
```

`frame.color` is a **BGR** `uint8` numpy image (full res, e.g. 1280x720). All `*_uv`
outputs are **normalized 0-1** over the color frame (resolution-independent — that's why
downscaling for inference is lossless for the output). `None` = abstain (the HUD falls
back to the depth outline; never draws a wrong box).

### Reuse the mask→boundary tail
`color_work_boundary` already turns a binary **mask** into
`{outline_uv, polygon_uv, fill_frac, border_touch, overruns}` (connected component at the
reticle → largest contour → `cv2.minAreaRect` → normalize; abstain if the blob is tiny or
fills ≥ `max_fill_frac`). **Factor that tail into a shared helper**, e.g.
`_mask_to_boundary(mask, reticle_xy) -> dict | None`, and have both the color segmenter and
the new `sam_work_boundary` call it. SAM's job is only to produce the mask; the rest is
identical (including the abstain-safe guards — keep them).

### The point prompt
The reticle is drawn at the **image center** (`analyze` centers it via
`center_patch_frac`), so the SAM point prompt = **(0.5, 0.5)** in normalized coords → pixel
`(W/2, H/2)`. If the reticle ever moves, thread its uv through.

---

## 3. Recommended model + dependency

Run on the **Windows host** (it already decodes the color stream; the Jetson Nano cannot
run SAM real-time — and this is host-side, so **no Jetson deploy**).

Prefer **ONNXRuntime** over full PyTorch (lighter, CPU-friendly on the host, no CUDA
required):

| Model | Notes | Latency (host CPU) |
|---|---|---|
| **MobileSAM** | ViT-tiny encoder + SAM decoder; ONNX exports exist; point-promptable. Good default. | ~100-300 ms |
| **EdgeSAM** | Distilled for on-device; point-promptable; smaller/faster. | ~50-150 ms |
| **FastSAM** | YOLOv8-seg "everything" then pick the mask under the point. Ultralytics dep (heavier), different prompt flow. | ~50-150 ms GPU |
| **SAM2** | Heavier, but can *track* a mask across frames (temporal) — nice-to-have later. | slower |

**Recommendation:** MobileSAM (or EdgeSAM) via `onnxruntime`. Two-part inference: run the
**image encoder once per frame** (the expensive part) → embedding; run the **prompt
decoder** with the reticle point (cheap) → mask + IoU score. Bundle the ONNX weights in the
repo (a `models/` dir; ~10-40 MB — consider Git LFS or a `tools/` download-on-setup step
so the main clone stays light). Keep it **offline** (no runtime download): a headless/cron
run must work.

Add the dep as an **optional extra** in `pyproject.toml` (mirror the `[scan]`/open3d
pattern) so the core install stays lean: `pip install -e .[sam]`.

---

## 4. Cadence / latency (don't block the video)

SAM at ~100-300 ms would throttle the 6 fps video if run inline in `analyze`. Options,
cheapest first:
1. **Throttle**: run SAM every ~300-400 ms (skip frames), publish `boundary` when ready.
   Simple; boundary updates at ~2-3 fps (still far better than 1 Hz, and steady when
   parked).
2. **Background worker thread**: `analyze` hands the latest color frame to a worker; the
   worker runs SAM and publishes `boundary`. Keeps video at full fps. More code.
3. **Cache the encoder embedding** while the camera is parked (pose gate says static — see
   `camera_pose_moved`) and only re-run the cheap decoder; re-encode on motion. Best
   perceived latency, more logic.

Start with **(1)**. The `boundary` event already has a 1.5 s frontend staleness timeout,
so a slower cadence degrades gracefully.

---

## 5. Integration checklist

- [ ] `pyproject.toml`: add `onnxruntime` (+ model helper) under a `[sam]` extra.
- [ ] Bundle model weights offline (`models/mobile_sam.onnx` or split encoder/decoder).
- [ ] New `tasni/modules/scan/sam_boundary.py`: `sam_work_boundary(color_bgr, *,
      point_uv=(0.5,0.5), seg_width, min_score, ...) -> dict | None`. Encode → decode at the
      point → mask → **reuse `_mask_to_boundary`** (factor it out of `color_boundary.py`).
      Set `contrast` = the model's IoU/stability score; keep the abstain-safe guards
      (`min_score`, `max_fill_frac`).
- [ ] `tasni/core/config.py` (`ScanConfig`): add `boundary_engine: str = "color"` (`"color"`
      | `"sam"` | `"sam_then_color"`) + `sam_*` knobs (model path, min_score, throttle_ms).
      The existing `color_boundary_*` knobs stay.
- [ ] `tasni/modules/scan/module.py` `analyze`: dispatch on `boundary_engine`. Recommended:
      **SAM primary, color fallback** (if SAM abstains/errors, try `color_work_boundary`),
      then publish the same `boundary` event. Wrap in try/except (never kill the video).
      Apply the throttle here (see §4).
- [ ] **Frontend: no changes needed** — `Scan.tsx` (`boundary` handler → `liveBoundary`) and
      `AimHud.tsx` (draws `liveBoundary` as the blue rectangle) already consume the payload.
- [ ] Tests: `tests/test_sam_boundary.py`. Gate the model-dependent test on the weights +
      onnxruntime being present and **skip otherwise** (mirror the `open3d` skip in
      `tests/test_scan_job.py`). Always-run tests can cover `_mask_to_boundary` with a
      synthetic mask (pure, no model).
- [ ] Keep `Lock` **depth-authoritative** (do NOT let SAM drive the 3D work rectangle in
      v1 — it's a visual aiming aid). `lock_scan_surface` in `service.py` is unchanged.
      (A later step could use the SAM mask to *refine* the depth rectangle at lock.)

---

## 6. How to test on the cell (reuse this session's scripts)

Everything is host-side; `.\start.ps1` / restart the backend to load new code. Grab a
frame and eyeball the mask/rectangle (the scratchpad scripts below are ephemeral — recreate
them):

```python
# grab one live color frame (frees the camera lease first, restarts preview after)
import urllib.request, cv2, sys; sys.path.insert(0, r"<repo>")
def post(p): 
    import urllib.request
    return urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000"+p, method="POST"), timeout=10).read().decode()
post("/api/modules/scan/live/stop")
from tasni.core.config import load_config; from tasni.core.camera import CameraClient
cam = CameraClient(load_config().camera); fr = cam.grab(with_depth=False, timeout=15)
cv2.imwrite("frame.png", fr.color); post("/api/modules/scan/live/start")
```

Then run `sam_work_boundary("frame.png")`, draw `outline_uv`/`polygon_uv` scaled by
`(W,H)`, `imwrite`, and view it. Live end-to-end: tap `ws://127.0.0.1:8000/ws` and count
`boundary` events (see `docs/live-robot-testing.md` §4 for the /ws pattern) — confirm they
flow with a plausible outline that tracks the object.

**The current cell object** = a **dark green cutting mat on a gray table** (grab a fresh
frame to confirm; it may change). This is the hard case SAM must nail — verify the mask
hugs the mat, not the whole frame.

---

## 7. Gotchas (learned this session)

- **BGR, not RGB.** `frame.color` is a cv2 BGR image. SAM preprocessing usually wants RGB —
  convert (`cv2.cvtColor(..., COLOR_BGR2RGB)`) before the encoder.
- **Normalized uv out.** Everything downstream expects `outline_uv`/`polygon_uv` in 0-1 over
  the color frame. Downscale for inference freely; normalize the output.
- **Abstain-safe is non-negotiable.** Return `None` when unsure (low score, mask fills the
  frame). The HUD falls back to the depth outline; a wrong full-frame box is worse than
  nothing. Keep `max_fill_frac`.
- **Anti-jitter hold vs the boundary.** The `boundary` event is published **directly from
  `analyze`** (`services.bus.publish`), *outside* the depth gate + `stabilize_live_scan_
  payload` hold — so it is NOT frozen. Keep it that way (that's what makes it live). The
  depth gate/numbers still ride their own 1 Hz + hold path; leave them alone.
- **Lock is depth.** The blue rectangle you aim with (SAM/color) is a visual aid. Target
  creation uses the locked depth snapshot. Don't conflate them in v1.
- **No Jetson deploy.** SAM runs on the host. `server/` is untouched. (Don't try to run SAM
  on the Nano.)
- **Offline weights.** Headless/cron runs have no interactive network — bundle weights, no
  runtime download.
- **Model licensing.** MobileSAM/EdgeSAM/FastSAM/SAM2 have different licenses — check before
  vendoring weights into a private repo shipped to the Jetson clone.

---

## 8. Files & reference commits

- `tasni/modules/scan/color_boundary.py` — classical segmenter + the mask→boundary tail to
  factor out. **The interface SAM should match.**
- `tasni/modules/scan/module.py` — `analyze` closure: the `boundary`-event publish seam.
- `tasni/core/config.py` — `ScanConfig.color_boundary_*` (add `boundary_engine` + `sam_*`).
- `tasni/webui/src/pages/Scan.tsx` — `boundary` handler → `liveBoundary` (no change needed).
- `tasni/webui/src/pages/AimHud.tsx` — draws `liveBoundary` (no change needed).
- `tests/test_color_boundary.py` — pattern for the segmenter tests.
- Commit `1dd8d21` — "Add a live COLOR work boundary (video-rate, host-side)" — the plumbing.
- Prior scan context: `docs/agent-debug-map.md`, memory `scan-module-status.md`.
