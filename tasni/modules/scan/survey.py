"""survey.py — full-frame surface survey for the scan planner (pure numpy).

Where ``depth_gate.py`` reads one central depth patch (a coarse readiness lamp), this
module surveys the **whole** depth frame: it RANSACs the dominant plane from the entire
image, then measures the surface's standoff, tilt, real-world extent, centroid and
whether it is fully framed — and emits vector overlays (an outline + an adaptive metric
grid) for the browser HUD to draw over the live camera.

It reuses the scan module's plane fit (``plane.fit_plane`` + ``_oriented_rectangle``),
so the same RANSAC/refine that turns a fused cloud into a work rectangle also drives the
live aiming survey. Pure numpy (no RoboDK / no live camera) so it is a reusable, unit-
testable core-style service.

Conventions match ``depth_gate.py``:

  * depth is raw uint16; mm = raw / depth_scale * 1000 (so raw == mm at depth_scale 1000)
  * the surface normal is oriented to FACE the camera (Z component < 0)
  * tilt = angle between the normal and the optical axis (0 = fronto-parallel)
  * tilt_b / tilt_c are KUKA B/C corrections (rotate about Y / X) — same math as the gate
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .plane import fit_plane, _oriented_rectangle, _plane_basis


@dataclass
class SurveyThresholds:
    accurate_min_mm: float = 300.0        # near edge of camera's accurate band
    accurate_max_mm: float = 800.0        # far edge; beyond → reference mode
    survey_max_tilt_deg: float = 6.0      # survey squareness requirement
    border_margin_px: int = 10            # pixel margin for fully_framed test
    min_valid_depth_frac: float = 0.3     # fraction of frame that must have valid depth
    ransac_distance_mm: float = 6.0       # RANSAC plane inlier band (mm)
    max_samples: int = 8000               # max points passed to RANSAC (stride-subsample)
    grid_target_px: int = 64              # desired on-screen grid cell size (px)


@dataclass
class SurveyMeasurement:
    detected: bool
    standoff_mm: float | None            # median depth of inlier points
    tilt_deg: float | None               # angle between surface normal and optical axis (0=fronto-parallel)
    tilt_b_deg: float | None             # KUKA B correction (rotate about Y; left/right)
    tilt_c_deg: float | None             # KUKA C correction (rotate about X; fwd/back)
    normal_cam: np.ndarray | None        # unit surface normal in camera frame (Z component < 0 = faces camera)
    centroid_cam_mm: np.ndarray | None   # inlier centroid in CAMERA frame (mm)
    extent_mm: tuple[float, float] | None  # (longer, shorter) bounding rect (real-world mm)
    shape: str                           # "rect" | "unknown"
    fully_framed: bool                   # inlier pixels do NOT touch the image border
    fov_deg: tuple[float, float]         # (hfov, vfov) computed from K + image size
    outline_uv: list[tuple[float, float]] | None   # 4 projected corners, normalized 0-1
    grid_uv: list[tuple[tuple[float, float], tuple[float, float]]] | None  # grid line segments
    grid_spacing_mm: float | None        # chosen grid spacing
    ok: bool                             # all gates pass: detected + in-range + tilt + framed
    gates: dict                          # {"detected": bool, "distance": bool, "angle": bool, "framed": bool}
    accurate_min_mm: float               # threshold used (for to_dict serialization)
    accurate_max_mm: float
    survey_max_tilt_deg: float
    corners_cam_mm: np.ndarray | None = None  # oriented-rectangle corners (4,3) in CAMERA frame (mm)
    points_uv: list | None = None             # decimated plane-inlier pixels, normalized 0-1, for the HUD dot overlay

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "standoff_mm": self.standoff_mm,
            "tilt_deg": self.tilt_deg,
            "tilt_b_deg": self.tilt_b_deg,
            "tilt_c_deg": self.tilt_c_deg,
            "normal_cam": self.normal_cam.tolist() if self.normal_cam is not None else None,
            "centroid_cam_mm": (self.centroid_cam_mm.tolist()
                                if self.centroid_cam_mm is not None else None),
            "extent_mm": list(self.extent_mm) if self.extent_mm is not None else None,
            "shape": self.shape,
            "fully_framed": self.fully_framed,
            "fov_deg": list(self.fov_deg),
            "outline_uv": self.outline_uv,
            "grid_uv": self.grid_uv,
            "grid_spacing_mm": self.grid_spacing_mm,
            "ok": self.ok,
            "gates": self.gates,
            "accurate_min_mm": self.accurate_min_mm,
            "accurate_max_mm": self.accurate_max_mm,
            "survey_max_tilt_deg": self.survey_max_tilt_deg,
            "corners_cam_mm": (np.asarray(self.corners_cam_mm, float).tolist()
                               if self.corners_cam_mm is not None else None),
            "points_uv": self.points_uv,
            # Backward-compatible fields for the frontend that expects the old
            # ScanGateReading shape (so the HUD can render either reading).
            "ideal_distance_mm": (self.accurate_min_mm + self.accurate_max_mm) / 2,
            "distance_tol_mm": (self.accurate_max_mm - self.accurate_min_mm) / 2,
            "max_tilt_deg": self.survey_max_tilt_deg,
            "distance_mm": self.standoff_mm,
            "valid_frac": 1.0 if self.detected else 0.0,
            "move_cam": None,
        }


def _fov_deg(K: np.ndarray, W: int, H: int) -> tuple[float, float]:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    hfov = float(np.degrees(2.0 * np.arctan(W / (2.0 * fx))))
    vfov = float(np.degrees(2.0 * np.arctan(H / (2.0 * fy))))
    return (hfov, vfov)


def _not_detected(th: SurveyThresholds, fov_deg: tuple[float, float]) -> SurveyMeasurement:
    return SurveyMeasurement(
        detected=False, standoff_mm=None, tilt_deg=None, tilt_b_deg=None,
        tilt_c_deg=None, normal_cam=None, centroid_cam_mm=None, extent_mm=None,
        shape="unknown", fully_framed=False, fov_deg=fov_deg, outline_uv=None,
        grid_uv=None, grid_spacing_mm=None, ok=False,
        gates={"detected": False, "distance": False, "angle": False, "framed": False},
        accurate_min_mm=th.accurate_min_mm, accurate_max_mm=th.accurate_max_mm,
        survey_max_tilt_deg=th.survey_max_tilt_deg)


def _snap_125(rough_mm: float) -> float:
    """Snap a length to the nearest >= value in the 1-2-5 (decade) series, min 1 mm."""
    rough = max(float(rough_mm), 1.0)
    e = int(np.floor(np.log10(rough)))
    base = 10.0 ** e
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * base >= rough:
            return max(m * base, 1.0)
    return max(10.0 * base, 1.0)


def survey_surface(
    depth: np.ndarray | None,
    K: np.ndarray,
    thresholds: SurveyThresholds,
    *,
    depth_scale: float = 1000.0,
) -> SurveyMeasurement:
    """Survey the dominant surface across a full depth frame for the aiming HUD.

    ``depth`` is the raw uint16 depth image (mm = raw / depth_scale * 1000; raw == mm at
    depth_scale 1000; 0 = invalid). ``K`` is the camera matrix for that frame. Returns a
    :class:`SurveyMeasurement` with standoff/tilt/extent + outline & grid overlays in
    normalized image coords. ``None``/all-invalid/too-little-depth ⇒ not detected.
    """
    K = np.asarray(K, dtype=float)
    th = thresholds

    # 1. FOV (always computable from K + image size; default to a sane size if no depth).
    if depth is None or np.asarray(depth).size == 0:
        # No image to size from — use K's principal point as a rough size proxy.
        W = int(round(2.0 * float(K[0, 2]))) or 1
        H = int(round(2.0 * float(K[1, 2]))) or 1
        return _not_detected(th, _fov_deg(K, W, H))

    d = np.asarray(depth)
    H, W = int(d.shape[0]), int(d.shape[1])
    fov_deg = _fov_deg(K, W, H)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # 2. Back-project valid pixels to camera 3D (mm).
    valid = d > 0
    valid_frac = float(valid.mean())
    if valid_frac < th.min_valid_depth_frac:
        return _not_detected(th, fov_deg)

    ys, xs = np.nonzero(valid)
    z_mm = d[ys, xs].astype(np.float64) / float(depth_scale) * 1000.0

    # Deterministic stride-subsample to at most max_samples points.
    n = len(z_mm)
    if n > th.max_samples:
        step = int(np.ceil(n / th.max_samples))
        ys, xs, z_mm = ys[::step], xs[::step], z_mm[::step]

    X = (xs - cx) / fx * z_mm
    Y = (ys - cy) / fy * z_mm
    pts_mm = np.column_stack([X, Y, z_mm])

    # 3. RANSAC plane fit (camera frame), then re-orient the normal to face the camera.
    try:
        normal, centroid, _ = fit_plane(pts_mm, distance=th.ransac_distance_mm)
    except ValueError:
        return _not_detected(th, fov_deg)

    normal = np.asarray(normal, float)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
    if normal[2] > 0:                      # face the camera (surface at +Z faces it w/ -Z)
        normal = -normal

    # Re-select inliers against the re-oriented normal (clean, same distance formula).
    dist = np.abs((pts_mm - centroid) @ normal)
    inlier_mask = dist < th.ransac_distance_mm
    if int(inlier_mask.sum()) < 8:
        return _not_detected(th, fov_deg)

    inlier_pts = pts_mm[inlier_mask]
    inlier_xs = xs[inlier_mask]
    inlier_ys = ys[inlier_mask]

    # 4. Measurements from inliers (same tilt math as depth_gate.py lines 121-127).
    standoff_mm = float(np.median(inlier_pts[:, 2]))
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(nz), 0.0, 1.0))))
    denom = max(-nz, 1e-9)                 # nz < 0 since the normal faces the camera
    tilt_b_deg = float(np.degrees(np.arctan2(nx, denom)))
    tilt_c_deg = float(np.degrees(np.arctan2(ny, denom)))

    # 5. Oriented rectangle (camera mm). ax1 is the longer edge direction.
    corners3d, ax1, ax2, len1, len2 = _oriented_rectangle(inlier_pts, normal, centroid)
    extent_mm = (float(len1), float(len2))

    # 6. Border test (fully_framed) — do any inlier pixels touch the image border?
    margin = th.border_margin_px
    fully_framed = not (
        bool(np.any(inlier_xs < margin)) or bool(np.any(inlier_xs > W - 1 - margin)) or
        bool(np.any(inlier_ys < margin)) or bool(np.any(inlier_ys > H - 1 - margin))
    )

    # 7. Gates and ok.
    gates = {
        "detected": True,
        "distance": th.accurate_min_mm <= standoff_mm <= th.accurate_max_mm,
        "angle": tilt_deg <= th.survey_max_tilt_deg,
        "framed": fully_framed,
    }
    ok = all(gates.values())

    # 8. Overlay — outline_uv (project the 4 rectangle corners to normalized image coords).
    def _project(p: np.ndarray):
        Zc = float(p[2])
        if Zc <= 0:
            return None
        u_norm = (float(p[0]) * fx / Zc + cx) / W
        v_norm = (float(p[1]) * fy / Zc + cy) / H
        return (u_norm, v_norm)

    outline_uv: list[tuple[float, float]] = []
    for c in corners3d:
        uv = _project(c)
        if uv is not None:
            outline_uv.append(uv)
    if not outline_uv:
        outline_uv = None  # type: ignore[assignment]

    # 9. Overlay — adaptive 1-2-5 metric grid aligned to the rectangle axes.
    rough_spacing_mm = th.grid_target_px * standoff_mm / fx
    spacing_mm = _snap_125(rough_spacing_mm)

    rel = inlier_pts - centroid
    proj1 = rel @ ax1
    proj2 = rel @ ax2
    lo1, hi1 = float(proj1.min()), float(proj1.max())
    lo2, hi2 = float(proj2.min()), float(proj2.max())

    s = spacing_mm
    s1 = np.arange(np.ceil(lo1 / s) * s, hi1 + 0.5 * s, s)
    s2 = np.arange(np.ceil(lo2 / s) * s, hi2 + 0.5 * s, s)

    grid_uv: list[tuple[tuple[float, float], tuple[float, float]]] = []

    # Lines PARALLEL to ax2 (vary along ax1).
    for t1 in s1:
        p_start = centroid + t1 * ax1 + lo2 * ax2
        p_end = centroid + t1 * ax1 + hi2 * ax2
        uv_s, uv_e = _project(p_start), _project(p_end)
        if uv_s is not None and uv_e is not None:
            grid_uv.append((uv_s, uv_e))

    # Lines PARALLEL to ax1 (vary along ax2).
    for t2 in s2:
        p_start = centroid + lo1 * ax1 + t2 * ax2
        p_end = centroid + hi1 * ax1 + t2 * ax2
        uv_s, uv_e = _project(p_start), _project(p_end)
        if uv_s is not None and uv_e is not None:
            grid_uv.append((uv_s, uv_e))

    # 10. Decimated plane-inlier pixels (normalized) so the HUD can draw the detected
    # surface as DOTS over the live RGB — not just the outline + grid. Capped so the
    # SVG overlay stays light; these are diagnostic, not used for any measurement.
    points_uv = None
    if len(inlier_xs) > 0:
        step = max(1, int(np.ceil(len(inlier_xs) / 700)))
        pu = inlier_xs[::step].astype(float) / float(W)
        pv = inlier_ys[::step].astype(float) / float(H)
        points_uv = np.round(np.column_stack([pu, pv]), 4).tolist()

    return SurveyMeasurement(
        detected=True,
        standoff_mm=standoff_mm,
        tilt_deg=tilt_deg,
        tilt_b_deg=tilt_b_deg,
        tilt_c_deg=tilt_c_deg,
        normal_cam=normal,
        centroid_cam_mm=np.asarray(centroid, float),
        extent_mm=extent_mm,
        shape="rect",
        fully_framed=fully_framed,
        fov_deg=fov_deg,
        outline_uv=outline_uv,
        grid_uv=grid_uv if grid_uv else None,
        grid_spacing_mm=float(spacing_mm),
        ok=ok,
        gates=gates,
        accurate_min_mm=th.accurate_min_mm,
        accurate_max_mm=th.accurate_max_mm,
        survey_max_tilt_deg=th.survey_max_tilt_deg,
        corners_cam_mm=np.asarray(corners3d, float),
        points_uv=points_uv,
    )
