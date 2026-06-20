"""Diagnostic intrinsic-parameter check — verify-and-warn, never alters the solve.

The captured ChArUco views are themselves an intrinsic-calibration dataset: each
view carries detected corner pixels + their board-frame 3D coordinates. So we can
re-estimate the camera matrix K and lens distortion straight from the calibration
captures and compare them to the configured (factory) intrinsics. This surfaces
the one error the hand-eye metrics cannot see — a wrong K (or the assumed-zero
distortion) silently absorbed into the solve.

Per the project decision this is **diagnostic only**: it produces a warning, not
a new K. The solve keeps using the configured intrinsics (review-then-apply).
"""
from __future__ import annotations

import cv2
import numpy as np

from .handeye import CalibrationView

# Warn bands: relative focal drift, absolute principal-point drift, |k1| radial.
_FOCAL_WARN_FRAC = 0.02      # 2% of fx / fy
_PRINCIPAL_WARN_PX = 8.0
_DIST_WARN = 0.05            # assumed-zero distortion this large is suspicious


def verify_intrinsics(views: list[CalibrationView], K: np.ndarray, dist: np.ndarray,
                      image_size: tuple[int, int]) -> dict | None:
    """Re-estimate K/distortion from the captured corners and compare to ``K``.

    ``image_size`` is ``(width, height)``. Returns a dict of deltas + a ``warn``
    flag, or ``None`` if the estimate could not be computed. Diagnostic only — the
    result is reported, never fed back into the calibration.
    """
    obj = [v.obj_points.astype(np.float32) for v in views]
    img = [v.corners.reshape(-1, 1, 2).astype(np.float32) for v in views]
    if len(obj) < 4:
        return None
    K0 = np.asarray(K, dtype=np.float64).copy()
    try:
        rms, K_est, dist_est, _, _ = cv2.calibrateCamera(
            obj, img, (int(image_size[0]), int(image_size[1])), K0.copy(),
            np.zeros((5, 1)), flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    except cv2.error:
        return None

    fx0, fy0, cx0, cy0 = K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]
    fx, fy, cx, cy = K_est[0, 0], K_est[1, 1], K_est[0, 2], K_est[1, 2]
    d_est = np.asarray(dist_est, dtype=float).reshape(-1)
    dfx, dfy = abs(fx - fx0), abs(fy - fy0)
    dcx, dcy = abs(cx - cx0), abs(cy - cy0)

    notes: list[str] = []
    if dfx > _FOCAL_WARN_FRAC * fx0 or dfy > _FOCAL_WARN_FRAC * fy0:
        notes.append(f"focal length off by {dfx:.1f}/{dfy:.1f} px")
    if dcx > _PRINCIPAL_WARN_PX or dcy > _PRINCIPAL_WARN_PX:
        notes.append(f"principal point off by {dcx:.1f}/{dcy:.1f} px")
    if abs(float(d_est[0])) > _DIST_WARN:
        notes.append(f"non-zero lens distortion (k1={d_est[0]:.3f})")

    return {
        "warn": bool(notes),
        "fit_rms_px": float(rms),
        "delta_fx_px": float(fx - fx0), "delta_fy_px": float(fy - fy0),
        "delta_cx_px": float(cx - cx0), "delta_cy_px": float(cy - cy0),
        "dist_recovered": [float(x) for x in d_est[:5]],
        "note": "; ".join(notes) if notes else "consistent with configured intrinsics",
    }
