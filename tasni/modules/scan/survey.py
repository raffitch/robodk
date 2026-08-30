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

  * depth is native raw (uint16); pixel -> mm goes through ``geometry.depth_unit_mm``
    via ``backproject`` (Task 7), which also registers the points into the COLOUR
    camera frame — every point/overlay this module produces is already in that frame
  * the surface normal is oriented to FACE the camera (Z component < 0)
  * tilt = angle between the normal and the optical axis (0 = fronto-parallel)
  * tilt_b / tilt_c are KUKA B/C corrections (rotate about Y / X) — same math as the gate
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ...core.depth_geometry import CameraGeometry, backproject, project_to_color
from .plane import (fit_plane, reticle_plane_square, _oriented_rectangle,
                    _plane_basis)


@dataclass
class SurveyThresholds:
    accurate_min_mm: float = 300.0        # near edge of camera's accurate band
    accurate_max_mm: float = 800.0        # far edge; beyond → reference mode
    survey_max_tilt_deg: float = 6.0      # survey squareness requirement
    min_valid_depth_frac: float = 0.3     # fraction of frame that must have valid depth
    ransac_distance_mm: float = 6.0       # RANSAC plane inlier band (mm)
    max_samples: int = 8000               # max points passed to RANSAC (stride-subsample)
    grid_target_px: int = 64              # desired on-screen grid cell size (px)
    frame_margin_uv: float = 0.02         # fitted-rect corners this far inside the frame
    #                                       => object bounded in view (keep the rectangle)
    work_crop_mm: tuple[float, float] = (1000.0, 1000.0)  # generic work square when the
    #                                       surface overruns the view (edges untrustworthy)
    center_patch_frac: float = 0.25       # aiming-reticle region (same convention as
    #     ScanGateThresholds.center_patch_frac) that SEEDS the plane fit below, so
    #     RANSAC locks onto the surface the operator is aiming at rather than
    #     whichever coplanar-ish cluster happens to have the most points anywhere
    #     in the wider native depth frame (e.g. an adjoining floor).


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
    # -- provenance of the plane selection (2026-08-30 silent-fallback fix) -----
    # ``plane_seed_used`` is False whenever the reticle seed could NOT drive the
    # fit and ``fit_plane`` degraded to whole-cloud maximal-consensus RANSAC.
    # That degradation is exactly the pre-seeding behaviour the seeding was added
    # to replace -- measured: an empty seed region put the centroid at 1070 mm
    # (the floor) instead of 450 mm (the platform) -- so it must never again be
    # invisible in the record. ``plane_seed_status`` names WHICH way it degraded
    # (one of plane.SEED_*); ``plane_seed_points`` is how many sampled points the
    # reticle box actually contained. Defaults describe "no seed was requested",
    # which is what the unseeded callers/archived records mean.
    plane_seed_used: bool = False
    plane_seed_status: str = "not_requested"
    plane_seed_points: int = 0

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
            # Plane-selection provenance. Additive keys: every existing consumer
            # reads by name, and an archived record that predates them simply
            # lacks them -- readers must treat a MISSING key as "unknown", not as
            # "seeded" (the dataclass defaults say "not_requested" for the same
            # reason). See the field comments above.
            "plane_seed_used": bool(self.plane_seed_used),
            "plane_seed_status": str(self.plane_seed_status),
            "plane_seed_points": int(self.plane_seed_points),
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


# --------------------------------------------------------------------------
# "Are the fitted rectangle's corners inside the colour frame?" -- ONE predicate
#
# There used to be two independent copies of this question with two DIFFERENT
# lens models: this module's own ``_corners_in_frame`` (pinhole, since 04e5760)
# and the live-HUD copy in ``scan/service.py`` (calibrated). Measured on
# 04e5760's own fold-back fixture they DISAGREED -- pinhole said "not framed",
# calibrated said "framed" -- so the HUD showed FRAMED and the subsequent lock
# raised ``LargeSurfaceRequired``: exactly the false-refusal symptom 04e5760 was
# written to kill. Both call sites now go through ``corners_in_color_frame``
# below, so they cannot disagree again.
# --------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _radial_monotonic_limit(dist_key: tuple, r_max: float) -> float:
    """First pinhole normalized radius at which the calibrated radial map stops
    increasing (``inf`` if it never does within ``r_max``).

    ``dist_key`` is an OpenCV distortion vector as a tuple
    (``[k1, k2, p1, p2, k3, k4, k5, k6, ...]``). The radial term is evaluated in
    its general rational form, so both the 5-coefficient polynomial model this
    cell uses and a rational (8-coefficient) one are handled by the same scan;
    tangential/thin-prism terms are ignored deliberately -- they are tiny
    (p1 = -1.8e-3, p2 = 4.2e-4 on this D435i) and do not create the fold-back.
    Cached because the coefficients change only when calibration re-solves."""
    k = np.zeros(14, dtype=float)
    n = min(len(dist_key), 14)
    k[:n] = np.asarray(dist_key, dtype=float)[:n]
    r = np.linspace(0.0, float(r_max), 4001)
    r2 = r * r
    num = 1.0 + k[0] * r2 + k[1] * r2 ** 2 + k[4] * r2 ** 3
    den = 1.0 + k[5] * r2 + k[6] * r2 ** 2 + k[7] * r2 ** 3
    with np.errstate(divide="ignore", invalid="ignore"):
        rd = r * num / den
    bad = ~np.isfinite(rd)
    drop = np.nonzero(bad[1:] | (np.diff(rd) <= 0.0))[0]
    return float(r[int(drop[0])]) if len(drop) else float("inf")


def color_model_domain_radius(K_color, dist_color, image_size) -> float:
    """Largest pinhole normalized radius at which ``dist_color`` may be trusted.

    cv2's FORWARD Brown-Conrady polynomial is only well-behaved inside the domain
    it was fit on. Past the radius where it stops being monotonic it FOLDS BACK,
    mapping a point that is genuinely far outside the frame to a pixel that lands
    back inside it. Measured with this workstation's real calibration
    (k1=0.1148, k2=-0.2386): the map turns non-monotonic at normalized radius
    **1.035** and re-enters the image's own (distorted) corner radius at **1.200**
    -- against an image corner radius of only **0.835** at 1280x720. So anything
    past the monotonic limit is unambiguously outside the frame, and the
    calibrated projection's answer for it is meaningless.

    Clamped to be no smaller than the image's own corner radius, so this guard is
    only ever a REJECTOR of points that the pinhole model already places outside
    the frame -- a pathological calibration that goes non-monotonic inside the
    image itself can never make a genuinely in-frame corner read "not framed".
    Returns ``inf`` for a zero/absent distortion vector (nothing to fold back)."""
    K = np.asarray(K_color, dtype=float)
    W, H = float(image_size[0]), float(image_size[1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    r_corner = float(np.hypot(max(cx, W - cx) / float(K[0, 0]),
                              max(cy, H - cy) / float(K[1, 1])))
    if dist_color is None:
        return float("inf")
    d = np.asarray(dist_color, dtype=float).reshape(-1)
    if d.size == 0 or not np.any(np.abs(d) > 1e-12):
        return float("inf")
    limit = _radial_monotonic_limit(tuple(float(v) for v in d),
                                    round(4.0 * max(r_corner, 1e-6), 6))
    return max(limit, r_corner)


def corners_in_color_frame(corners_cam_mm, K_color, dist_color, image_size,
                           *, margin_uv: float = 0.02) -> bool:
    """Do all four rectangle corners (COLOUR-camera mm) sit inside the colour
    image with a ``margin_uv`` border? The single framing predicate.

    Projection uses the **calibrated** model, which is the physically correct
    answer to "where does this 3D point appear in the image the operator is
    looking at" -- guarded by :func:`color_model_domain_radius` so the one way
    that model can lie (fold-back, far outside its fitted domain) is rejected up
    front instead of silently reported as "framed".

    Why not simply project with the pinhole model everywhere (04e5760's fix)?
    Measured with this repo's real K/dist at 1280x720 and the default
    ``frame_margin_uv`` = 0.02 (a 25.6 px / 14.4 px border):

      * along the frame boundary the two models disagree by up to 40.6 px in u
        and 23.5 px in v -- larger than the margin itself, so classifications
        genuinely flip near the edge;
      * an **821 x 410 mm rectangle at 600 mm standoff** reads framed under the
        calibrated model and NOT framed under pinhole. The largest still-framed
        half-width for that aspect at 600 mm is 414.0 mm calibrated vs 408.2 mm
        pinhole -- i.e. pinhole *widens* the false-refusal band that 04e5760 set
        out to close, by declaring genuinely in-view platforms "too large";
      * the guard reproduces the calibrated answer exactly in that boundary case
        (414.0 mm, framed) while still rejecting 04e5760's fold-back fixture,
        whose corners sit at pinhole radius 1.404 -- well past the 1.035
        monotonic limit.

    So: calibrated inside the fitted domain, hard reject outside it."""
    cc = np.asarray(corners_cam_mm, dtype=float).reshape(-1, 3)
    if cc.shape[0] < 4 or not np.all(np.isfinite(cc)):
        return False
    if bool(np.any(cc[:, 2] <= 0)):        # behind (or on) the image plane
        return False
    r_pinhole = np.hypot(cc[:, 0] / cc[:, 2], cc[:, 1] / cc[:, 2])
    if float(np.max(r_pinhole)) > color_model_domain_radius(
            K_color, dist_color, image_size):
        return False
    W, H = float(image_size[0]), float(image_size[1])
    uv = project_to_color(cc, K_color, dist_color) / np.array([W, H], float)
    if not np.all(np.isfinite(uv)):
        return False
    frac = float(margin_uv)
    return bool(np.all((uv[:, 0] >= frac) & (uv[:, 0] <= 1.0 - frac)
                       & (uv[:, 1] >= frac) & (uv[:, 1] <= 1.0 - frac)))


def survey_surface(
    depth: np.ndarray | None,
    geometry: CameraGeometry | None,
    K_color: np.ndarray,
    dist_color,
    thresholds: SurveyThresholds,
) -> SurveyMeasurement:
    """Survey the dominant surface across a full NATIVE depth frame for the aiming HUD.

    ``depth`` is the raw depth image (0 = invalid); ``geometry`` (Task 7's
    :class:`CameraGeometry`, from the connection's greeting) carries the unit and the
    depth->colour registration. ``K_color``/``dist_color`` are the calibrated colour
    model — the image the HUD actually shows, and what every overlay below is
    projected into. Returns a :class:`SurveyMeasurement` with standoff/tilt/extent +
    outline & grid overlays in normalized COLOUR-image coords.
    ``None``/no geometry/all-invalid/too-little-depth ⇒ not detected.
    """
    K_color = np.asarray(K_color, dtype=float)
    th = thresholds

    # 1. FOV (of the COLOUR image the HUD shows -- always computable from K_color +
    # geometry.color_size; falls back to K's own principal point as a size proxy when
    # there is nothing to survey at all).
    if geometry is not None:
        W, H = geometry.color_size
    else:
        W = int(round(2.0 * float(K_color[0, 2]))) or 1
        H = int(round(2.0 * float(K_color[1, 2]))) or 1
    fov_deg = _fov_deg(K_color, W, H)

    if depth is None or np.asarray(depth).size == 0 or geometry is None:
        return _not_detected(th, fov_deg)

    d = np.asarray(depth)
    # Fraction of the DEPTH frame with valid depth -- same meaning as before (the
    # depth image, not the colour image, is what the sensor actually filled in).
    valid_frac = float(np.count_nonzero(d)) / float(d.size) if d.size else 0.0
    if valid_frac < th.min_valid_depth_frac:
        return _not_detected(th, fov_deg)

    # 2. Back-project valid depth pixels to COLOUR-camera-frame 3D (mm), already
    # registered by backproject() -- deterministic stride-subsample (2D grid, so
    # roughly stride**2 fewer points) to at most max_samples.
    n_valid = int(np.count_nonzero(d))
    stride = 1
    if n_valid > th.max_samples:
        stride = int(np.ceil((n_valid / float(th.max_samples)) ** 0.5))
    pts_mm, _uv_depth = backproject(d, geometry, stride=stride)
    if len(pts_mm) > th.max_samples:
        # ascontiguousarray, not just the slice: a strided view makes
        # cv2.projectPoints (via project_to_color, two lines below) raise
        # "npoints >= 0 && (depth == CV_32F || CV_64F)" and take the whole survey
        # down. Reachable whenever the 2D stride-subsample above still overshoots
        # max_samples -- e.g. 72,000 valid depth pixels in a 320x240 frame -> the
        # grid stride lands on 8,027 points, one decimation pass short of the
        # 8,000 cap. Found by the seed-fallback fixture below; pre-existing.
        pts_mm = np.ascontiguousarray(
            pts_mm[::int(np.ceil(len(pts_mm) / th.max_samples))])

    # 2b. Seed region (2026-08-30 false-refusal fix): the colour-image centre
    # patch, same box convention as ColorRegistered._center_patch_bounds / the
    # depth gate's own centre-patch aiming region -- what the operator is
    # actually pointed at. Passed to fit_plane as seed_mask so the RANSAC locks
    # onto the plane THROUGH this region rather than whichever cluster anywhere
    # in the (wider-than-colour) native depth frame happens to have the most
    # points -- see fit_plane's own docstring for the live-cell evidence.
    seed_uv = project_to_color(pts_mm, K_color, dist_color)
    pf = float(np.clip(th.center_patch_frac, 0.05, 1.0))
    cw, ch = max(2, int(W * pf)), max(2, int(H * pf))
    sx0, sy0 = (W - cw) // 2, (H - ch) // 2
    sx1, sy1 = sx0 + cw, sy0 + ch
    seed_mask = ((seed_uv[:, 0] >= sx0) & (seed_uv[:, 0] < sx1)
                & (seed_uv[:, 1] >= sy0) & (seed_uv[:, 1] < sy1))

    # 3. RANSAC plane fit (COLOUR camera frame), then re-orient the normal to face it.
    # ``seed_report`` captures whether the reticle seed actually drove the fit: when
    # it did not (empty/too-sparse seed region, or a seed plane that failed on its
    # own), fit_plane degrades to whole-cloud maximal-consensus RANSAC -- the
    # pre-seeding behaviour, which measured a 1070 mm floor instead of a 450 mm
    # platform. That used to happen silently; it is now carried on the measurement.
    seed_report: dict = {}
    try:
        normal, centroid, _ = fit_plane(pts_mm, distance=th.ransac_distance_mm,
                                        seed_mask=seed_mask, report=seed_report)
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

    # 4. Measurements from inliers (same tilt math as depth_gate.py lines 121-127).
    standoff_mm = float(np.median(inlier_pts[:, 2]))
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(nz), 0.0, 1.0))))
    denom = max(-nz, 1e-9)                 # nz < 0 since the normal faces the camera
    tilt_b_deg = float(np.degrees(np.arctan2(nx, denom)))
    tilt_c_deg = float(np.degrees(np.arctan2(ny, denom)))

    # 5. Oriented rectangle (camera mm). ax1 is the longer edge direction. The raw
    # extent (len1, len2) is kept for the "is the surface too large?" decision even
    # when the overlay below switches to a generic square.
    corners3d, ax1, ax2, len1, len2 = _oriented_rectangle(inlier_pts, normal, centroid)
    extent_mm = (float(len1), float(len2))

    # 6. Framed test. The old test asked "do any raw inlier PIXELS touch the image
    # border?" — too strict: a few stray coplanar fringe points near an edge made a
    # well-margined object read as an overrun and fall back to the generic square.
    # Instead TRUST THE FITTED RECTANGLE: if its projected corners sit inside the
    # frame with a margin, the object is bounded in view and we keep its rectangle.
    # (That old test's ``border_margin_px`` threshold went with it — it has now been
    # deleted from SurveyThresholds entirely. Native depth pixels no longer correspond
    # 1:1 to colour pixels, so a border test in depth-pixel space cannot answer "is this
    # inside the COLOUR frame?" at all; the fitted-rectangle test below is now the only
    # framing decision.)
    # Containment goes through the ONE shared predicate (see
    # ``corners_in_color_frame`` above for the model choice and the measured
    # numbers behind it) -- the live-HUD copy in ``scan/service.py`` calls the
    # same function, so the HUD's "FRAMED" lamp and this lock-time test can
    # never disagree the way they did before 2026-08-30.
    fully_framed = corners_in_color_frame(
        corners3d, K_color, dist_color, (W, H), margin_uv=float(th.frame_margin_uv))

    # When the surface overruns the view, its real edges are not in frame, so the
    # board rectangle above would over-run the table. Replace the operator overlay +
    # work corners with a GENERIC fixed square on the plane, centred on the reticle
    # (the aim point). The plane fit (standoff/tilt/normal) is unchanged; only the
    # programmable footprint becomes the generic crop. Fully-framed surfaces keep the
    # measured board rectangle (the user's "edges clear -> use that" rule).
    if not fully_framed:
        corners3d, _ax_u, _ax_v, _reticle = reticle_plane_square(
            normal, centroid, th.work_crop_mm)

    # 7. Gates and ok.
    gates = {
        "detected": True,
        "distance": th.accurate_min_mm <= standoff_mm <= th.accurate_max_mm,
        "angle": tilt_deg <= th.survey_max_tilt_deg,
        "framed": fully_framed,
    }
    ok = all(gates.values())

    # 8. Overlay — outline_uv (project the 4 rectangle corners to normalized COLOUR
    # image coords, through the calibrated colour model — CameraGeometry.color_size,
    # not the depth image, is "the image the HUD shows").
    def _project(p: np.ndarray):
        if float(p[2]) <= 0:
            return None
        uv = project_to_color(np.asarray(p, float).reshape(1, 3), K_color, dist_color)[0]
        return (float(uv[0]) / W, float(uv[1]) / H)

    outline_uv: list[tuple[float, float]] = []
    for c in corners3d:
        uv = _project(c)
        if uv is not None:
            outline_uv.append(uv)
    if not outline_uv:
        outline_uv = None  # type: ignore[assignment]

    # 9. Overlay — adaptive 1-2-5 metric grid aligned to the rectangle axes. Sized in
    # COLOUR pixels (grid_target_px is an on-screen size in the image the HUD draws).
    rough_spacing_mm = th.grid_target_px * standoff_mm / float(K_color[0, 0])
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

    # 10. Detected-surface DOTS for the HUD overlay: the ACTUAL measured surface
    # points (the plane inliers, where depth truly landed), snapped to a FIXED image
    # grid and emitted as the occupied cells — NOT idealized cell centers derived
    # from the surface estimate. The grid is fixed in the IMAGE, so the dots hold
    # still; a cell appears only where a real point fell in it, so an empty cell is a
    # genuine coverage hole. Matches the live server's coverage dots (same GRID), so
    # the locked snapshot and the live aiming stream show the same kind of marker.
    points_uv = None
    if len(inlier_pts) > 0:
        Zc = inlier_pts[:, 2]
        in_front = Zc > 0
        real_uv = (project_to_color(inlier_pts[in_front], K_color, dist_color)
                  / np.array([W, H], float))
        in_frame = np.all((real_uv >= 0.0) & (real_uv <= 1.0), axis=1)
        real_uv = real_uv[in_frame]
        if len(real_uv):
            GRID = 180  # matches the live server / frontend coverage-dedupe resolution
            cells = np.unique(np.floor(real_uv * GRID).astype(int), axis=0)
            if len(cells) > 4000:
                cells = cells[:: int(np.ceil(len(cells) / 4000.0))]
            dot_uv = (cells + 0.5) / float(GRID)
            points_uv = np.round(dot_uv, 4).tolist()

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
        plane_seed_used=bool(seed_report.get("seed_used", False)),
        plane_seed_status=str(seed_report.get("seed_status", "not_requested")),
        plane_seed_points=int(seed_report.get("seed_points", 0)),
    )
