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

# Warn bands: relative focal drift, absolute principal-point drift, radial-coeff drift.
_FOCAL_WARN_FRAC = 0.02      # 2% of fx / fy
_PRINCIPAL_WARN_PX = 8.0
_DIST_WARN = 0.05            # |estimated - configured| k1/k2 this large = disagreement


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
    d_cfg = np.concatenate([np.asarray(dist, dtype=float).reshape(-1), np.zeros(5)])[:5]
    # Re-estimate with the SAME distortion model the config uses: if the configured
    # k3 is 0 (our calibrated default fixes it), fix k3 here too. A free k3 trades off
    # against k2, so a free-k3 estimate would look divergent from a fixed-k3 config
    # even when they agree — the cause of the false "non-zero distortion" warning that
    # persisted after the correct distortion was already applied.
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    if abs(float(d_cfg[4])) < 1e-9:
        flags |= cv2.CALIB_FIX_K3
    try:
        rms, K_est, dist_est, _, _ = cv2.calibrateCamera(
            obj, img, (int(image_size[0]), int(image_size[1])), K0.copy(),
            np.zeros((5, 1)), flags=flags)
    except cv2.error:
        return None

    fx0, fy0, cx0, cy0 = K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]
    fx, fy, cx, cy = K_est[0, 0], K_est[1, 1], K_est[0, 2], K_est[1, 2]
    d_est = np.concatenate([np.asarray(dist_est, dtype=float).reshape(-1), np.zeros(5)])[:5]
    dfx, dfy = abs(fx - fx0), abs(fy - fy0)
    dcx, dcy = abs(cx - cx0), abs(cy - cy0)

    notes: list[str] = []
    if dfx > _FOCAL_WARN_FRAC * fx0 or dfy > _FOCAL_WARN_FRAC * fy0:
        notes.append(f"focal length off by {dfx:.1f}/{dfy:.1f} px")
    if dcx > _PRINCIPAL_WARN_PX or dcy > _PRINCIPAL_WARN_PX:
        notes.append(f"principal point off by {dcx:.1f}/{dcy:.1f} px")
    # Warn only when the recovered distortion DISAGREES with the configured coeffs —
    # not merely because the lens has distortion (k1/k2 are the dominant, stable
    # radial terms; k3/p1/p2 are noisy on a centred board, so they aren't gated).
    dk1, dk2 = abs(float(d_est[0] - d_cfg[0])), abs(float(d_est[1] - d_cfg[1]))
    if dk1 > _DIST_WARN or dk2 > 2 * _DIST_WARN:
        notes.append(f"configured distortion disagrees (est k1={d_est[0]:.3f}/k2={d_est[1]:.3f} "
                     f"vs cfg k1={d_cfg[0]:.3f}/k2={d_cfg[1]:.3f})")

    return {
        "warn": bool(notes),
        "fit_rms_px": float(rms),
        "delta_fx_px": float(fx - fx0), "delta_fy_px": float(fy - fy0),
        "delta_cx_px": float(cx - cx0), "delta_cy_px": float(cy - cy0),
        "dist_recovered": [float(x) for x in d_est[:5]],
        "dist_configured": [float(x) for x in d_cfg[:5]],
        "note": "; ".join(notes) if notes else "consistent with configured intrinsics",
    }
