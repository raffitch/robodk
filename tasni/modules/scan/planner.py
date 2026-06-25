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

from dataclasses import dataclass, field

import numpy as np

from .survey import SurveyMeasurement


@dataclass
class AimPoint:
    """A single look-at target for the pose generator (quality mode)."""

    point_base_mm: np.ndarray      # look-at target (centroid) in base frame (or camera frame if cam_to_base_T=None)
    view_dir_base: np.ndarray      # desired camera forward direction (toward surface)
    standoff_mm: float
    cone_half_angle_deg: float
    roll_max_deg: float
    n_views: int

    def to_dict(self) -> dict:
        return {
            "point_base_mm": np.asarray(self.point_base_mm, float).tolist(),
            "view_dir_base": np.asarray(self.view_dir_base, float).tolist(),
            "standoff_mm": float(self.standoff_mm),
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
