"""Extract base-frame corner/edge evidence from one corner capture (spec §7).

The boundary polygon (colour/SAM) proposes WHERE the physical edge is; depth +
the calibrated camera pose provide METRIC geometry (spec §11). Samples are
inset a few pixels toward the surface interior so depth is read on the
platform, not in the discontinuity at its edge.

Units: ``depth`` is a ``(H, W)`` array in **millimetres** (RealSense uint16
convention); ``polygon_uv`` is normalized ``(N, 2)`` image coordinates;
``T_base_cam`` is the 4x4 base<-camera pose in millimetres. All outputs
(``corner_base_mm``, ``edge_points_base``) are millimetres in the robot base
frame.

This module deliberately does NOT reuse ``service._backproject_depth`` (which
has ambiguous units) -- it is self-contained so its mm convention is
unambiguous end to end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CornerEvidence:
    corner_uv: tuple[float, float]
    corner_base_mm: tuple[float, float, float] | None
    edge_points_base: np.ndarray  # (N, 3) pooled from both arms


def _median_depth(depth, px, py, window_px: int) -> float:
    h, w = depth.shape
    x0, x1 = max(0, px - window_px), min(w, px + window_px + 1)
    y0, y1 = max(0, py - window_px), min(h, py + window_px + 1)
    patch = np.asarray(depth[y0:y1, x0:x1], dtype=float)
    vals = patch[patch > 0]  # NaN and <=0 both excluded (NaN > 0 is False)
    if len(vals) == 0:
        return 0.0
    z = float(np.median(vals))
    return z if np.isfinite(z) else 0.0


def _deproject_base(u_px, v_px, z_mm, K, T_base_cam) -> np.ndarray:
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    p_cam = np.array([(u_px - cx) / fx * z_mm, (v_px - cy) / fy * z_mm, z_mm, 1.0])
    return (np.asarray(T_base_cam, dtype=float) @ p_cam)[:3]


def _walk_arm(poly_px, start_idx, step, arm_len_px, n_samples):
    """Sample points along the open polyline from start_idx in direction `step`.

    Terminates as soon as the next vertex index would leave [0, len(poly_px)),
    so it never indexes out of range and never wraps around on an open
    polyline. If the polyline is shorter than `arm_len_px`, fewer than
    `n_samples` points are returned.
    """
    out, travelled = [], 0.0
    i = start_idx
    target = np.linspace(arm_len_px / n_samples, arm_len_px, n_samples)
    ti = 0
    while ti < len(target) and 0 <= i + step < len(poly_px):
        a, b = poly_px[i], poly_px[i + step]
        seg = float(np.linalg.norm(b - a))
        while ti < len(target) and travelled + seg >= target[ti]:
            t = (target[ti] - travelled) / max(seg, 1e-9)
            out.append(a + t * (b - a))
            ti += 1
        travelled += seg
        i += step
    return np.asarray(out, dtype=float).reshape(-1, 2)


def extract_corner_evidence(depth, K, polygon_uv, T_base_cam, *,
                             corner_hint_uv=(0.5, 0.5), arm_frac: float = 0.35,
                             samples_per_arm: int = 40, inset_px: float = 4.0,
                             window_px: int = 2, min_valid_frac: float = 0.3):
    depth = np.asarray(depth, dtype=float)
    if depth.ndim != 2:
        return None
    h, w = depth.shape
    poly = np.asarray(polygon_uv, dtype=float).reshape(-1, 2)
    if len(poly) < 3 or not np.all(np.isfinite(poly)):
        return None
    poly_px = poly * [w, h]
    # Degenerate polygon guard: if every vertex sits within a couple of
    # pixels of the first one, there is no usable arm length to walk.
    if float(np.ptp(poly_px, axis=0).max()) < 2.0:
        return None

    hint_px = np.asarray(corner_hint_uv, dtype=float) * [w, h]
    corner_idx = int(np.argmin(np.linalg.norm(poly_px - hint_px, axis=1)))
    corner_px = poly_px[corner_idx]

    arm_len_px = arm_frac * float(np.hypot(w, h))
    arms = [_walk_arm(poly_px, corner_idx, +1, arm_len_px, samples_per_arm),
            _walk_arm(poly_px, corner_idx, -1, arm_len_px, samples_per_arm)]
    interior = poly_px.mean(axis=0)

    pts_base = []
    n_requested = 0
    for arm in arms:
        for p in arm:
            n_requested += 1
            direction = interior - p
            norm = float(np.linalg.norm(direction))
            sample = p + (direction / norm * inset_px if norm > 1e-6 else 0.0)
            px, py = int(round(sample[0])), int(round(sample[1]))
            if not (0 <= px < w and 0 <= py < h):
                continue
            z = _median_depth(depth, px, py, window_px)
            if z <= 0 or not np.isfinite(z):
                continue
            point = _deproject_base(sample[0], sample[1], z, K, T_base_cam)
            if not np.all(np.isfinite(point)):
                continue
            pts_base.append(point)
    if n_requested == 0 or len(pts_base) < max(4, int(min_valid_frac * n_requested)):
        return None

    corner_base = None
    zc = _median_depth(depth, int(round(corner_px[0])), int(round(corner_px[1])),
                        window_px * 3)
    if zc > 0 and np.isfinite(zc):
        cb = _deproject_base(corner_px[0], corner_px[1], zc, K, T_base_cam)
        if np.all(np.isfinite(cb)):
            corner_base = tuple(float(v) for v in cb)

    edge_points_base = np.asarray(pts_base, dtype=float)
    if edge_points_base.size and not np.all(np.isfinite(edge_points_base)):
        return None

    return CornerEvidence(corner_uv=(float(corner_px[0] / w), float(corner_px[1] / h)),
                           corner_base_mm=corner_base,
                           edge_points_base=edge_points_base)
