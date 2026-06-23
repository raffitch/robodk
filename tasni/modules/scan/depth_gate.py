"""The scan standoff gate — turn one depth frame into HUD readiness, no marker.

Calibration's gate keys off a printed ChArUco board; a bare table has none, so the
scan reads the **depth** of a central image patch instead and lights the same three
HUD lamps:

    detected   enough valid depth in the centre (a surface is actually there)
    distance   patch median depth within ``ideal_distance_mm ± distance_tol_mm``
    angle      surface tilt off fronto-parallel within ``max_tilt_deg``

The operator jogs the camera to look down at the table until all three are green,
then Create targets seeds pose generation from that pose. Pure numpy (no RoboDK / no
live camera) so it is a reusable core-style service and unit-testable — and it is the
seam the future camera-feedback loop reads realtime depth through.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScanGateThresholds:
    ideal_distance_mm: float = 500.0
    distance_tol_mm: float = 150.0
    max_tilt_deg: float = 35.0
    center_patch_frac: float = 0.25      # central image fraction sampled
    min_valid_depth_frac: float = 0.5    # >= this fraction of the patch must be valid


@dataclass
class ScanGateReading:
    detected: bool
    distance_mm: float | None            # central-patch median depth (surface<->camera)
    tilt_deg: float | None               # 0 = fronto-parallel, 90 = edge-on
    valid_frac: float                    # fraction of the patch with valid depth
    gates: dict                          # {"detected", "distance", "angle"}
    ok: bool
    ideal_distance_mm: float
    distance_tol_mm: float
    max_tilt_deg: float
    move_cam: tuple[float, float, float] | None = None  # jog hint (Z = distance error)
    # How to correct the tilt, as TOOL-frame rotations (KUKA A/B/C convention:
    # A = rotate about Z, B = about Y, C = about X). A surface tilt is fixed by B + C
    # (a rotation about Z / A doesn't change the tilt). Signed degrees the operator
    # should rotate the TOOL to make the surface fronto-parallel; None if no surface.
    tilt_b_deg: float | None = None      # rotate about camera/TOOL Y (KUKA B): left/right
    tilt_c_deg: float | None = None      # rotate about camera/TOOL X (KUKA C): fwd/back

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "distance_mm": self.distance_mm,
            "tilt_deg": self.tilt_deg,
            "valid_frac": self.valid_frac,
            "gates": self.gates,
            "ok": self.ok,
            "ideal_distance_mm": self.ideal_distance_mm,
            "distance_tol_mm": self.distance_tol_mm,
            "max_tilt_deg": self.max_tilt_deg,
            "move_cam": list(self.move_cam) if self.move_cam is not None else None,
            "tilt_b_deg": self.tilt_b_deg,
            "tilt_c_deg": self.tilt_c_deg,
        }


def _not_detected(th: ScanGateThresholds, valid_frac: float) -> ScanGateReading:
    return ScanGateReading(
        detected=False, distance_mm=None, tilt_deg=None, valid_frac=valid_frac,
        gates={"detected": False, "distance": False, "angle": False}, ok=False,
        ideal_distance_mm=th.ideal_distance_mm, distance_tol_mm=th.distance_tol_mm,
        max_tilt_deg=th.max_tilt_deg)


def evaluate_depth_gate(depth: np.ndarray | None, K: np.ndarray,
                        th: ScanGateThresholds, *, depth_scale: float = 1000.0,
                        max_samples: int = 3000) -> ScanGateReading:
    """Build the :class:`ScanGateReading` for one depth frame.

    ``depth`` is the raw depth image (uint16 mm for the D435i, i.e. raw value == mm
    when ``depth_scale == 1000``); ``None`` (or all-invalid) ⇒ not detected. ``K`` is
    the camera matrix for that frame (back-projects the patch to 3D to estimate the
    surface tilt).
    """
    if depth is None or np.asarray(depth).size == 0:
        return _not_detected(th, 0.0)
    d = np.asarray(depth)
    h, w = d.shape[:2]
    pf = float(np.clip(th.center_patch_frac, 0.05, 1.0))
    cw, ch = max(2, int(w * pf)), max(2, int(h * pf))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    patch = d[y0:y0 + ch, x0:x0 + cw].astype(np.float64)
    valid = patch > 0
    valid_frac = float(valid.mean())
    if valid_frac < th.min_valid_depth_frac:
        return _not_detected(th, valid_frac)

    # Distance = median valid depth (raw mm at depth_scale 1000); scale-general.
    distance_mm = float(np.median(patch[valid]) / float(depth_scale) * 1000.0)

    # Tilt: back-project the valid patch pixels (absolute image coords) to camera 3D
    # and fit a plane; tilt = angle between its normal and the optical axis.
    K = np.asarray(K, dtype=float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ys, xs = np.nonzero(valid)
    z = patch[valid]
    if len(z) > max_samples:                 # subsample for speed (deterministic stride)
        step = int(np.ceil(len(z) / max_samples))
        ys, xs, z = ys[::step], xs[::step], z[::step]
    u, v = xs + x0, ys + y0
    pts = np.column_stack([(u - cx) / fx * z, (v - cy) / fy * z, z])
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vt[2] / max(float(np.linalg.norm(vt[2])), 1e-9)
    # Orient the normal to face the camera (surface is in front at +Z, so the face
    # toward the optical axis has a negative Z component).
    if normal[2] > 0:
        normal = -normal
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(nz), 0.0, 1.0))))
    # Tilt-correction rotations to bring the normal to (0,0,-1) — i.e. face the
    # surface squarely. About the optical-frame Y axis (KUKA B, left/right) and the
    # X axis (KUKA C, fwd/back). A rotation about Z (KUKA A) doesn't change tilt.
    denom = max(-nz, 1e-9)
    tilt_b_deg = float(np.degrees(np.arctan2(nx, denom)))
    tilt_c_deg = float(np.degrees(np.arctan2(ny, denom)))

    gates = {
        "detected": True,
        "distance": abs(distance_mm - th.ideal_distance_mm) <= th.distance_tol_mm,
        "angle": tilt_deg <= th.max_tilt_deg,
    }
    move_cam = (0.0, 0.0, float(distance_mm - th.ideal_distance_mm))
    return ScanGateReading(
        detected=True, distance_mm=distance_mm, tilt_deg=tilt_deg,
        valid_frac=valid_frac, gates=gates, ok=all(gates.values()),
        ideal_distance_mm=th.ideal_distance_mm, distance_tol_mm=th.distance_tol_mm,
        max_tilt_deg=th.max_tilt_deg, move_cam=move_cam,
        tilt_b_deg=tilt_b_deg, tilt_c_deg=tilt_c_deg)
