"""From a full-frame surface measurement to a scan plan — pure numpy.

Phase 2 of the surface-aware scan planner. ``plan_scan`` takes a
:class:`~tasni.modules.scan.survey.SurveyMeasurement` (the surface the camera is
currently looking at, measured from a single full-frame view) and decides:

  * **mode** — ``"quality"`` (close enough for a tour + fused mesh) or
    ``"reference"`` (too far / too big — just a single-shot rectangle).
  * **standoff** — the distance that frames the whole surface with margin,
    clamped into the D435i's accurate depth band.
  * **voxel size** — the TSDF/fusion resolution, scaled to the standoff.
  * **cone + view count** — the orbit spread and number of views, by surface type
    (flat plates need a tight cone; raised objects need a wide one).
  * **aim** — for quality mode, the look-at point + desired camera-forward
    direction, expressed in the base frame when a camera→base transform is given.

Pure function: no hardware, no RoboDK, no sockets — only numpy + the local survey
contract. So it is unit-testable on any machine and runs under ``py -3.10``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .survey import SurveyMeasurement


@dataclass
class AimPoint:
    """A single look-at target for the pose generator (quality mode)."""

    point_base_mm: np.ndarray      # look-at target (centroid) in base frame (or camera frame if cam_to_base_T=None)
    view_dir_base: np.ndarray      # desired camera forward direction (toward surface)
    standoff_mm: float
    min_perpendicular_mm: float
    cone_half_angle_deg: float
    roll_max_deg: float
    n_views: int

    def to_dict(self) -> dict:
        return {
            "point_base_mm": np.asarray(self.point_base_mm, float).tolist(),
            "view_dir_base": np.asarray(self.view_dir_base, float).tolist(),
            "standoff_mm": float(self.standoff_mm),
            "min_perpendicular_mm": float(self.min_perpendicular_mm),
            "cone_half_angle_deg": float(self.cone_half_angle_deg),
            "roll_max_deg": float(self.roll_max_deg),
            "n_views": int(self.n_views),
        }


@dataclass
class ScanPlan:
    """The plan a survey yields: what to scan, how close, how finely, how widely."""

    mode: str                      # "quality" | "reference"
    aims: list[AimPoint]           # 1 for quality mode, 0 for reference mode
    standoff_mm: float             # planned standoff (even for reference mode, for logging)
    voxel_size_m: float
    cone_half_angle_deg: float     # from aim if quality, from preset if reference
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "aims": [a.to_dict() for a in self.aims],
            "standoff_mm": float(self.standoff_mm),
            "voxel_size_m": float(self.voxel_size_m),
            "cone_half_angle_deg": float(self.cone_half_angle_deg),
            "warnings": list(self.warnings),
        }


def plan_scan(
    survey: SurveyMeasurement,
    K: np.ndarray,
    size: tuple[int, int],          # (W, H) image size
    scan_cfg,                        # ScanConfig object (duck-typed; reads attributes below)
    *,
    cam_to_base_T: np.ndarray | None = None,  # 4x4 camera-to-base transform
) -> ScanPlan:
    """Decide standoff / mode / cone / view-count / voxel from a surface survey.

    See the module docstring for the contract. ``cam_to_base_T`` (4x4) expresses
    the aim in the robot base frame; pass ``None`` to keep it in the camera frame.
    """
    # 1. No surface detected → reference mode, far standoff, coarsest voxel.
    if not survey.detected:
        return ScanPlan(
            mode="reference",
            aims=[],
            standoff_mm=scan_cfg.accurate_max_mm,
            voxel_size_m=scan_cfg.voxel_max_m,
            cone_half_angle_deg=scan_cfg.flat_cone_deg,
            warnings=["no surface detected"],
        )

    # 2. Standoff that frames the surface in both axes (pinhole FOV math).
    Sx, Sy = survey.extent_mm            # (longer, shorter) in mm
    W, H = size
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    m = scan_cfg.frame_margin
    d_fit = max(m * Sx * fx / W, m * Sy * fy / H)

    # 3. Mode: too far to frame within the accurate band → reference; else quality.
    if d_fit > scan_cfg.accurate_max_mm:
        mode = "reference"
    else:
        mode = "quality"
    standoff_mm = float(np.clip(d_fit, scan_cfg.accurate_min_mm, scan_cfg.accurate_max_mm))

    # 4. Voxel size scales with standoff (further → coarser), clamped.
    # standoff_mm / 1000 converts to metres before multiplying voxel_k so the
    # result is in metres (k=0.008 → 4 mm @ 500 mm standoff).
    voxel_size_m = float(
        np.clip(standoff_mm / 1000.0 * scan_cfg.voxel_k,
                scan_cfg.voxel_min_m, scan_cfg.voxel_max_m)
    )

    # 5. Cone + view count by surface type.
    if scan_cfg.surface_type == "raised":
        cone = scan_cfg.raised_cone_deg
        n_views = scan_cfg.raised_views
    else:                                  # "flat" default
        cone = scan_cfg.flat_cone_deg
        n_views = scan_cfg.flat_views

    # 6. Reference mode: no aim, return early (with a framing warning if needed).
    if mode == "reference":
        warnings: list[str] = []
        if not survey.fully_framed:
            warnings.append(
                "surface is not fully framed — back the camera off for a "
                "trustworthy rectangle"
            )
        return ScanPlan(
            mode="reference",
            aims=[],
            standoff_mm=standoff_mm,
            voxel_size_m=voxel_size_m,
            cone_half_angle_deg=cone,
            warnings=warnings,
        )

    # 7. Quality mode: build the aim (centroid look-at + camera-forward direction).
    centroid_cam = np.asarray(survey.centroid_cam_mm, float)
    # normal_cam faces the camera (Z < 0); camera forward = toward the surface
    # = -(normal facing camera) = -normal_cam, normalized.
    fwd_cam = -np.asarray(survey.normal_cam, float)
    fwd_cam = fwd_cam / np.linalg.norm(fwd_cam)

    if cam_to_base_T is not None:
        R = np.asarray(cam_to_base_T[:3, :3], float)
        t = np.asarray(cam_to_base_T[:3, 3], float)
        point_mm = R @ centroid_cam + t
        view_dir = R @ fwd_cam
    else:
        point_mm = centroid_cam
        view_dir = fwd_cam

    view_dir = view_dir / np.linalg.norm(view_dir)
    aim = AimPoint(
        point_base_mm=point_mm,
        view_dir_base=view_dir,
        standoff_mm=standoff_mm,
        min_perpendicular_mm=float(d_fit),
        cone_half_angle_deg=cone,
        roll_max_deg=scan_cfg.roll_max_deg,
        n_views=n_views,
    )
    return ScanPlan(
        mode="quality",
        aims=[aim],
        standoff_mm=standoff_mm,
        voxel_size_m=voxel_size_m,
        cone_half_angle_deg=cone,
        warnings=[],
    )


def _tile_grid_dims(
    corners_base: np.ndarray, K: np.ndarray, size_px: tuple[int, int], scan_cfg,
) -> tuple[int, int, float, float]:
    """The tile GRID SHAPE ``plan_rect_tour`` lays out: ``(nx, ny, foot_w, foot_h)``.

    Factored out of ``plan_rect_tour`` (post-review fix, Task 12) for two
    reasons: (1) it is the single point that validates ``survey_tour_overlap``
    — see below — so ``plan_rect_tour`` and any other caller that needs the
    grid shape (the tiled-tour target generator maps a linear tile index back
    to ``(i, j)`` for the contiguous-empty-tile check) can never drift out of
    sync or duplicate/skip the check; (2) recovering ``(i, j)`` from a linear
    tile index needs ``ny`` specifically, which isn't otherwise recoverable
    from ``ScanPlan``/``AimPoint`` (frozen interfaces from earlier tasks) —
    ``ScanPlan.aims`` is a flat list, so its length alone cannot tell you
    ``nx`` vs ``ny`` separately (e.g. 64 could be 8x8 or 4x16 or 2x32).

    ``survey_tour_overlap`` is a live config knob (``ScanConfig``, additive,
    no built-in bound) that this whole tiling scheme's coverage argument
    depends on staying in ``[0.0, 1.0)`` — ``step = 1 - overlap`` must stay in
    ``(0.0, 1.0]`` for "tile spacing never exceeds the footprint" (the no-gap
    proof) to hold at all. Measured what happens if it doesn't (on the brief's
    own 2000x1200 mm example): ``overlap=1.0`` -> ``step=0`` ->
    ZeroDivisionError inside the tile-count math, which ``module.py`` then
    turns into a misleading HTTP 503 "RoboDK/camera unavailable" for what is
    really a config typo; ``overlap=1.3`` -> ``step=-0.3``, and
    ``ceil(negative)`` collapses through ``max(1, ...)`` to a SINGLE tile for
    the whole 2 m platform, while the tile-completeness coverage gate (which
    assumes one tile per patch) reports 100% — the robot would scan one
    381x214 mm patch and the gate would wave it through; ``overlap=-0.5`` -> a
    spacing of 1.5x the footprint, opening a real gap between every tile,
    again silently reported as 100% by a fraction gate that only knows "tile
    present/absent," not "footprints touch." Chosen to FAIL LOUDLY (raise
    ``ValueError``) here rather than silently clamp: a clamp would let a typo
    quietly change the actual tiling density (and thus scan time/coverage
    margin) with no operator-visible signal, on a path that is about to drive
    the real KUKA. This does not touch the config LOAD path (JSON files don't
    set this key today; the default 0.30 is valid) — it only fires the moment
    a five-position survey actually asks to be tiled with a nonsensical value.
    """
    fx, fy = float(K[0][0]), float(K[1][1])
    W, H = int(size_px[0]), int(size_px[1])
    d = float(scan_cfg.accurate_min_mm)
    margin = float(scan_cfg.frame_margin)
    foot_w = d * W / fx / margin
    foot_h = d * H / fy / margin
    overlap = float(scan_cfg.survey_tour_overlap)
    if not (0.0 <= overlap < 1.0):
        raise ValueError(
            f"scan.survey_tour_overlap must be in [0.0, 1.0) (got {overlap!r}). "
            "step = 1 - overlap must stay in (0.0, 1.0] so tile spacing never "
            "exceeds the tile footprint: overlap=1.0 divides by zero; overlap>1.0 "
            "makes spacing negative and collapses the whole platform to one tile "
            "while still reporting full coverage; overlap<0.0 makes spacing wider "
            "than the footprint and opens a gap between every tile.")
    step = 1.0 - overlap

    corners = np.asarray(corners_base, dtype=float).reshape(4, 3)
    ex, ey = corners[1] - corners[0], corners[3] - corners[0]
    Lx, Ly = float(np.linalg.norm(ex)), float(np.linalg.norm(ey))
    # Epsilon-guarded ceil: when Lx (or Ly) divides the step EXACTLY, float
    # rounding in the /fx//margin chain above can land the ratio a hair over
    # the integer (measured: 150.0 / 149.99999999999997 = 1.0000000000000002),
    # which ceil() then rounds up to the NEXT integer — silently doubling a
    # single-tile case (200x150 mm -> 1x2 instead of 1x1, measured). The eps is
    # far below any real geometric difference (ratios here are O(1)-O(1e2)).
    # Independently re-verified (post-review) that this guard cannot instead
    # cause UNDER-tiling: a gap would need the tile-centre spacing Lx/nx to
    # exceed the footprint, i.e. nx smaller by a factor of >= 1/step (~1.43x
    # at the default 0.30 overlap) than what the eps-free ratio would give —
    # 1e-9 is nowhere near that, confirmed by sweeping L from 200 mm to 1e6 mm
    # (``tests/test_scan_planner.py::test_rect_tour_epsilon_guard_never_undertiles``).
    _EPS = 1e-9
    nx = max(1, int(math.ceil(Lx / (foot_w * step) - _EPS)))
    ny = max(1, int(math.ceil(Ly / (foot_h * step) - _EPS)))
    return nx, ny, foot_w, foot_h


def _largest_contiguous_empty_block(empty_ij: set[tuple[int, int]]) -> int:
    """Size (tile count) of the largest 4-connected contiguous block within
    ``empty_ij`` (a set of ``(i, j)`` tile-grid coordinates that got zero
    poses) — post-review fix (Task 12 Finding 2): the tile-completeness
    FRACTION alone tolerates one large contiguous hole (e.g. a 3x3 block, 9 of
    64 tiles, still passes a 0.85 fraction threshold) exactly as easily as the
    same count of tiles scattered across the grid, even though a contiguous
    hole is a single, real, unscanned patch of the surface while a scattered
    miss is mostly absorbed by neighbouring tiles' overlap. Pure combinatorics,
    no geometry — the caller decides the acceptable block size.
    """
    seen: set[tuple[int, int]] = set()
    best = 0
    for start in empty_ij:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            i, j = stack.pop()
            size += 1
            for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if nb in empty_ij and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        best = max(best, size)
    return best


def plan_rect_tour(
    corners_base: np.ndarray,
    normal_base: np.ndarray,
    K: np.ndarray,
    size_px: tuple[int, int],
    scan_cfg,
) -> ScanPlan:
    """Tile a large (five-position-surveyed) rectangle with close-range views.

    Phase 2b of the surface-aware planner (two-path plan §7/§12, Task 12): when
    the platform is too large for one camera view to frame at an accurate
    standoff, ``plan_scan`` alone cannot help — the only way to frame it in a
    single shot is to back the camera off, which destroys depth quality. This
    instead keeps the camera at ``accurate_min_mm`` (near edge of the D435i's
    accurate depth band) and tiles the rectangle with overlapping close-range
    views, one :class:`AimPoint` per tile.

    ``corners_base`` (4,3) are the rectangle's own corners in the robot base
    frame, in the same cyclic order :func:`survey_contract.order_corners_clockwise`
    / :func:`service._densify_quad` use (edges ``c0->c1`` and ``c0->c3``) — NOT
    assumed axis-aligned, and generally NOT at z=0. Every aim is built as an
    affine combination of the (coplanar) corners, so it lands exactly on the
    rectangle's own plane regardless of its orientation.

    Tile footprint (mm) and grid shape come from :func:`_tile_grid_dims` (the
    pinhole model at the fixed standoff ``d = accurate_min_mm``, shrunk by
    ``frame_margin``, matching ``plan_scan``'s FOV math — see
    ``min_perpendicular_mm`` below — with ``survey_tour_overlap`` validated
    there). Tile *centres* are spaced evenly across each axis at
    ``Lx/nx`` / ``Ly/ny`` where ``nx``/``ny`` are sized so that spacing never
    exceeds ``footprint * (1 - survey_tour_overlap)`` — i.e. adjacent tile
    footprints always overlap by *at least* the configured fraction (the
    `ceil()` in ``_tile_grid_dims`` can only add overlap, never remove it),
    and the outermost tile centres sit half a footprint in from the
    rectangle's own edges, so their footprints always reach past those edges.
    Together this guarantees the tiling has no gap and covers the rectangle
    including its edge bands (verified by construction in
    ``tests/test_scan_planner.py``).
    """
    corners = np.asarray(corners_base, dtype=float).reshape(4, 3)
    n = np.asarray(normal_base, dtype=float)
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    d = float(scan_cfg.accurate_min_mm)
    nx, ny, foot_w, foot_h = _tile_grid_dims(corners, K, size_px, scan_cfg)

    center = corners.mean(axis=0)
    ex, ey = corners[1] - corners[0], corners[3] - corners[0]
    Lx, Ly = float(np.linalg.norm(ex)), float(np.linalg.norm(ey))
    ux = ex / Lx if Lx > 1e-9 else np.zeros(3)
    uy = ey / Ly if Ly > 1e-9 else np.zeros(3)

    voxel = float(np.clip(d / 1000.0 * scan_cfg.voxel_k,
                          scan_cfg.voxel_min_m, scan_cfg.voxel_max_m))
    aims = []
    for i in range(nx):
        for j in range(ny):
            p = (center + ux * ((i + 0.5) / nx - 0.5) * Lx
                        + uy * ((j + 0.5) / ny - 0.5) * Ly)
            aims.append(AimPoint(
                point_base_mm=p, view_dir_base=-n, standoff_mm=d,
                # Equals plan_scan's own d_fit = max(m*Sx*fx/W, m*Sy*fy/H)
                # (planner.py:105) when the "surface" is this tile's own
                # footprint: substituting Sx=foot_w=d*W/fx/m cancels the
                # fx/W and the margin m exactly (and likewise for Sy/foot_h),
                # leaving d_fit == d for every tile — not an invented value.
                min_perpendicular_mm=d,
                cone_half_angle_deg=float(scan_cfg.flat_cone_deg),
                roll_max_deg=float(scan_cfg.roll_max_deg),
                n_views=int(scan_cfg.survey_tour_views_per_tile)))
    return ScanPlan(mode="large_survey", aims=aims, standoff_mm=d, voxel_size_m=voxel,
                    cone_half_angle_deg=float(scan_cfg.flat_cone_deg), warnings=[])
