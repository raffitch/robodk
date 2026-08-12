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

    Tile footprint (mm) comes from the pinhole model at the fixed standoff
    ``d = accurate_min_mm``, shrunk by ``frame_margin`` for a comfortable
    border, matching ``plan_scan``'s FOV math (see ``min_perpendicular_mm``
    below). Tile *centres* are spaced evenly across each axis at
    ``Lx/nx`` / ``Ly/ny`` where ``nx``/``ny`` are sized so that spacing never
    exceeds ``footprint * (1 - survey_tour_overlap)`` — i.e. adjacent tile
    footprints always overlap by *at least* the configured fraction (the
    `ceil()` below can only add overlap, never remove it), and the outermost
    tile centres sit half a footprint in from the rectangle's own edges, so
    their footprints always reach past those edges. Together this guarantees
    the tiling has no gap and covers the rectangle including its edge bands
    (verified by construction in ``tests/test_scan_planner.py``).
    """
    corners = np.asarray(corners_base, dtype=float).reshape(4, 3)
    n = np.asarray(normal_base, dtype=float)
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    fx, fy = float(K[0][0]), float(K[1][1])
    W, H = int(size_px[0]), int(size_px[1])
    d = float(scan_cfg.accurate_min_mm)
    margin = float(scan_cfg.frame_margin)
    foot_w = d * W / fx / margin
    foot_h = d * H / fy / margin
    step = 1.0 - float(scan_cfg.survey_tour_overlap)

    center = corners.mean(axis=0)
    ex, ey = corners[1] - corners[0], corners[3] - corners[0]
    Lx, Ly = float(np.linalg.norm(ex)), float(np.linalg.norm(ey))
    ux = ex / Lx if Lx > 1e-9 else np.zeros(3)
    uy = ey / Ly if Ly > 1e-9 else np.zeros(3)
    # Epsilon-guarded ceil: when Lx (or Ly) divides the step EXACTLY, float
    # rounding in the /fx//margin chain above can land the ratio a hair over
    # the integer (measured: 150.0 / 149.99999999999997 = 1.0000000000000002),
    # which ceil() then rounds up to the NEXT integer — silently doubling a
    # single-tile case (200x150 mm -> 1x2 instead of 1x1, measured). The eps is
    # far below any real geometric difference (ratios here are O(1)-O(1e2)).
    _EPS = 1e-9
    nx = max(1, int(math.ceil(Lx / (foot_w * step) - _EPS)))
    ny = max(1, int(math.ceil(Ly / (foot_h * step) - _EPS)))

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
