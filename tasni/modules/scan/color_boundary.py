"""Color-image work-surface boundary (host-side, every frame).

RealSense DEPTH is noisy on low-texture objects, so a depth-only work rectangle flickers
and lags — and it also rides a ~1 Hz telemetry channel plus the anti-jitter freeze, so the
blue box barely tracks the object while aiming. The object's outline, though, is usually
crisp in the COLOR image the host already decodes at video rate. This module segments the
object under the reticle from a single color frame and returns its boundary as
normalized-uv corners, so the live HUD can draw a steady blue rectangle that tracks the
object in real time — independent of depth.

Deliberately simple, deterministic classical CV (reticle-seeded Otsu -> largest central
component -> min-area rectangle). It is a VISUAL aid for aiming; the authoritative 3D work
rectangle is still measured from depth at Lock. Pure numpy + cv2 (no RealSense, no RoboDK)
so it unit-tests on synthetic images.
"""
from __future__ import annotations

import numpy as np
import cv2


def color_work_boundary(
    color_bgr,
    *,
    reticle_frac: float = 0.25,
    min_color_dist: float = 14.0,
    min_fill_frac: float = 0.02,
    max_fill_frac: float = 0.85,
    border_touch_frac: float = 0.30,
    seg_width: int = 480,
) -> dict | None:
    """Segment the object under the reticle; return its boundary as normalized uv.

    Returns ``None`` when there is no trustworthy boundary (the object looks like the
    background, or the segmented blob is implausibly small). Otherwise a dict with:

    - ``outline_uv``  – 4 min-area-rectangle corners, normalized 0-1 (the blue box).
    - ``polygon_uv``  – simplified contour of the object, normalized 0-1 (hugs the shape).
    - ``contrast``    – median Lab colour distance object<->background; the confidence.
    - ``fill_frac``   – object area / image area.
    - ``border_touch``– fraction of the object contour within ~1 px of the frame edge.
    - ``overruns``    – the object runs past the view (fills it or hugs the border), so
      the rectangle is not the true object extent; the caller may prefer a reticle square.

    Segments in **Lab colour space**, keyed off the reticle sample: each pixel's distance
    to the object's colour separates it from the background. This is what makes it work on
    real scenes a grayscale threshold misses — e.g. a green cutting mat on a gray table has
    nearly the same *luminance* but very different *colour* (chroma), while a white block on
    a dark table differs in *luminance*; Lab distance captures both. Object-agnostic.
    """
    if color_bgr is None:
        return None
    img = np.asarray(color_bgr)
    if img.ndim != 3 or img.shape[2] < 3 or img.shape[0] < 8 or img.shape[1] < 8:
        return None
    H0, W0 = img.shape[:2]
    # Downscale for speed; normalized uv is resolution-independent so this is lossless
    # for the output.
    scale = float(seg_width) / float(W0) if W0 > seg_width else 1.0
    if scale < 1.0:
        img = cv2.resize(img, (int(round(W0 * scale)), int(round(H0 * scale))),
                         interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    cx, cy = W // 2, H // 2
    rw = max(2, int(W * reticle_frac / 2))
    rh = max(2, int(H * reticle_frac / 2))
    patch = lab[max(0, cy - rh):cy + rh, max(0, cx - rw):cx + rw].reshape(-1, 3)
    if patch.size == 0:
        return None
    obj_lab = np.median(patch, axis=0)                  # the object's colour (L, a, b)
    dist = np.linalg.norm(lab - obj_lab, axis=2)        # per-pixel colour distance

    # Background separation = how different the frame border (the table) looks from the
    # object colour. Small -> the object looks like the table, or it overruns the view
    # (the border IS the object): no trustworthy boundary -> abstain.
    border = np.concatenate([dist[0, :], dist[-1, :], dist[:, 0], dist[:, -1]])
    contrast = float(np.median(border))
    if contrast < float(min_color_dist):
        return None

    # Object = pixels close to the reticle colour. Otsu on the distance image finds the
    # split between the object cluster (near 0) and the background cluster (far).
    dmax = float(dist.max())
    d8 = np.clip(dist / max(dmax, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (d8 <= thr).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    lbl = int(labels[cy, cx])
    if lbl == 0:
        # Reticle sits on background: fall back to the largest foreground component.
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size == 0:
            return None
        lbl = 1 + int(np.argmax(areas))
    comp = (labels == lbl).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    fill_frac = area / float(H * W)
    # Abstain (return None -> the HUD falls back to the depth outline) when the blob is
    # implausibly small OR fills most of the frame. A near-full blob means either the
    # object genuinely overruns the view (depth's reticle square is the right answer) or
    # the segmentation failed to separate a low-contrast scene — in BOTH cases the color
    # rectangle is untrustworthy, so it must never be drawn. This is what keeps the layer
    # "never worse than depth": it shows a boundary only when it confidently isolated the
    # object, and stays out of the way otherwise.
    if fill_frac < float(min_fill_frac) or fill_frac >= float(max_fill_frac):
        return None

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(float)  # 4 corners in the downscaled frame

    pts = cnt.reshape(-1, 2).astype(float)
    edge = 1.5
    on_border = ((pts[:, 0] <= edge) | (pts[:, 0] >= W - 1 - edge)
                 | (pts[:, 1] <= edge) | (pts[:, 1] >= H - 1 - edge))
    border_touch = float(np.mean(on_border)) if len(pts) else 0.0
    overruns = bool(border_touch >= float(border_touch_frac))

    eps = 0.01 * cv2.arcLength(cnt, True)
    poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(float)

    norm = np.array([W, H], dtype=float)
    return {
        "outline_uv": (box / norm).tolist(),
        "polygon_uv": (poly / norm).tolist(),
        "contrast": contrast,
        "fill_frac": fill_frac,
        "border_touch": border_touch,
        "overruns": overruns,
    }
