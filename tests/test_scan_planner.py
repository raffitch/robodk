"""planner.py — surface-aware scan-plan math (pure numpy).

Synthetic surveys (small/large, near/far, flat/raised) -> assert the planner's
mode selection, FOV-derived standoff (with clamping), voxel scaling, cone/count
presets, and the aim transform. No RoboDK / open3d / cv2 / hardware.

    py -3.10 tests/test_scan_planner.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.modules.scan.planner import plan_scan  # noqa: E402
from tasni.modules.scan.survey import SurveyMeasurement  # noqa: E402


@dataclass
class MockScanCfg:
    accurate_min_mm: float = 300.0
    accurate_max_mm: float = 800.0
    frame_margin: float = 1.3
    voxel_k: float = 0.008
    voxel_min_m: float = 0.002
    voxel_max_m: float = 0.006
    surface_type: str = "flat"
    flat_cone_deg: float = 18.0
    flat_views: int = 8
    raised_cone_deg: float = 38.0
    raised_views: int = 13
    roll_max_deg: float = 30.0


# K with fx=fy=300, principal point (160,120) — pairs with a 320x240 image.
K_TEST = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])
SIZE_TEST = (320, 240)


def _survey(
    *,
    detected: bool = True,
    extent_mm: tuple[float, float] = (300.0, 200.0),
    standoff_mm: float = 500.0,
    centroid_cam_mm: np.ndarray | None = None,
    normal_cam: np.ndarray | None = None,
    fully_framed: bool = True,
) -> SurveyMeasurement:
    """Build a SurveyMeasurement with sensible defaults for planner tests."""
    if centroid_cam_mm is None:
        centroid_cam_mm = np.array([0.0, 0.0, standoff_mm])
    if normal_cam is None:
        normal_cam = np.array([0.0, 0.0, -1.0])
    return SurveyMeasurement(
        detected=detected,
        standoff_mm=standoff_mm,
        tilt_deg=1.5,
        tilt_b_deg=1.0,
        tilt_c_deg=1.0,
        normal_cam=np.asarray(normal_cam, float),
        centroid_cam_mm=np.asarray(centroid_cam_mm, float),
        extent_mm=extent_mm,
        shape="rect",
        fully_framed=fully_framed,
        fov_deg=(69.4, 43.6),
        outline_uv=None,
        grid_uv=None,
        grid_spacing_mm=None,
        ok=True,
        gates={"detected": True, "distance": True, "angle": True, "framed": True},
        accurate_min_mm=300.0,
        accurate_max_mm=800.0,
        survey_max_tilt_deg=6.0,
    )


def test_small_surface_quality_mode():
    cfg = MockScanCfg()
    survey = _survey(extent_mm=(200.0, 150.0), standoff_mm=500.0)
    plan = plan_scan(survey, K_TEST, SIZE_TEST, cfg, cam_to_base_T=np.eye(4))

    assert plan.mode == "quality", plan.mode
    assert len(plan.aims) == 1, plan.aims
    assert cfg.accurate_min_mm <= plan.standoff_mm <= cfg.accurate_max_mm, plan.standoff_mm
    assert cfg.voxel_min_m <= plan.voxel_size_m <= cfg.voxel_max_m, plan.voxel_size_m
    print("[small/quality] standoff", round(plan.standoff_mm, 1),
          "voxel", round(plan.voxel_size_m, 4))


def test_large_surface_reference_mode():
    cfg = MockScanCfg()
    survey = _survey(extent_mm=(1500.0, 1200.0))
    fx, fy = K_TEST[0, 0], K_TEST[1, 1]
    W, H = SIZE_TEST
    d_fit = max(cfg.frame_margin * 1500.0 * fx / W, cfg.frame_margin * 1200.0 * fy / H)
    assert d_fit > cfg.accurate_max_mm, d_fit   # sanity: this really is too far

    plan = plan_scan(survey, K_TEST, SIZE_TEST, cfg, cam_to_base_T=np.eye(4))
    assert plan.mode == "reference", plan.mode
    assert len(plan.aims) == 0, plan.aims
    print("[large/reference] d_fit", round(d_fit, 1), "standoff", round(plan.standoff_mm, 1))


def test_standoff_clamped_below():
    cfg = MockScanCfg()
    # A tiny 10x10 mm surface frames far closer than the accurate minimum.
    survey = _survey(extent_mm=(10.0, 10.0))
    plan = plan_scan(survey, K_TEST, SIZE_TEST, cfg, cam_to_base_T=np.eye(4))
    assert plan.standoff_mm == cfg.accurate_min_mm, plan.standoff_mm
    print("[clamp-below] standoff", plan.standoff_mm, "==", cfg.accurate_min_mm)


def test_standoff_clamped_above_quality_boundary():
    cfg = MockScanCfg()
    # Pick an extent so d_fit lands right at accurate_max_mm (still quality, not
    # reference — the boundary is "> max" => reference, so == max stays quality).
    # d_fit = margin * Sx * fx / W  ->  Sx = d_fit * W / (margin * fx). Nudge Sx a
    # hair below so float round-trip error can't push d_fit just over the boundary.
    fx = K_TEST[0, 0]
    W = SIZE_TEST[0]
    Sx = cfg.accurate_max_mm * W / (cfg.frame_margin * fx) * (1.0 - 1e-9)
    d_fit = cfg.frame_margin * Sx * fx / W
    assert d_fit <= cfg.accurate_max_mm, d_fit   # sanity: lands at/below the boundary
    # shorter axis must not dominate: keep it small.
    survey = _survey(extent_mm=(Sx, 10.0))
    plan = plan_scan(survey, K_TEST, SIZE_TEST, cfg, cam_to_base_T=np.eye(4))
    assert plan.mode == "quality", plan.mode
    assert abs(plan.standoff_mm - cfg.accurate_max_mm) < 1e-3, plan.standoff_mm
    print("[clamp-above-boundary] standoff", round(plan.standoff_mm, 3),
          "mode", plan.mode)


def test_voxel_scales_with_standoff():
    # Use a voxel_k that keeps (standoff_mm / 1000) * voxel_k INSIDE [voxel_min, voxel_max]
    # for the standoffs under test, so the proportional regime is exercised (not the clamp).
    # voxel_k=0.01: 300/1000*0.01=0.003, 500/1000*0.01=0.005 — both inside [0.002, 0.006].
    cfg = MockScanCfg(voxel_k=0.01)
    fx = K_TEST[0, 0]
    W = SIZE_TEST[0]

    def standoff_for(d_target: float) -> float:
        # extent that frames at d_target along the longer axis (shorter axis tiny)
        Sx = d_target * W / (cfg.frame_margin * fx)
        survey = _survey(extent_mm=(Sx, 5.0))
        plan = plan_scan(survey, K_TEST, SIZE_TEST, cfg, cam_to_base_T=np.eye(4))
        assert plan.mode == "quality", (d_target, plan.mode)
        return plan

    # Both standoffs stay in the proportional band (300/1000*0.01=0.003, 500/1000*0.01=0.005,
    # both inside [0.002, 0.006]).
    near = standoff_for(300.0)
    far = standoff_for(500.0)
    assert near.standoff_mm < far.standoff_mm, (near.standoff_mm, far.standoff_mm)
    # closer standoff -> smaller (finer) voxel, both inside the clamp band here
    assert near.voxel_size_m < far.voxel_size_m, (near.voxel_size_m, far.voxel_size_m)
    # voxel is proportional to standoff (voxel_k) within the band
    ratio_voxel = near.voxel_size_m / far.voxel_size_m
    ratio_standoff = near.standoff_mm / far.standoff_mm
    assert abs(ratio_voxel - ratio_standoff) < 1e-6, (ratio_voxel, ratio_standoff)
    print("[voxel-scaling] near", round(near.standoff_mm, 1), round(near.voxel_size_m, 4),
          "far", round(far.standoff_mm, 1), round(far.voxel_size_m, 4))


def test_fov_math():
    cfg = MockScanCfg()
    # 300 mm extent along width -> d_fit = 1.3 * 300 * 300 / 320 ~= 365.6 mm
    survey = _survey(extent_mm=(300.0, 5.0))
    expected = cfg.frame_margin * 300.0 * K_TEST[0, 0] / SIZE_TEST[0]
    assert abs(expected - 365.625) < 0.5, expected
    plan = plan_scan(survey, K_TEST, SIZE_TEST, cfg, cam_to_base_T=np.eye(4))
    assert plan.standoff_mm >= cfg.accurate_min_mm, plan.standoff_mm
    # this d_fit is within the band, so standoff equals d_fit
    assert abs(plan.standoff_mm - expected) < 1e-6, (plan.standoff_mm, expected)
    print("[fov-math] expected", round(expected, 2), "standoff", round(plan.standoff_mm, 2))


def test_raised_preset():
    cfg = MockScanCfg(surface_type="raised")
    survey = _survey(extent_mm=(200.0, 150.0))
    plan = plan_scan(survey, K_TEST, SIZE_TEST, cfg, cam_to_base_T=np.eye(4))
    assert plan.mode == "quality", plan.mode
    assert plan.cone_half_angle_deg == 38.0, plan.cone_half_angle_deg
    assert plan.aims[0].n_views == 13, plan.aims[0].n_views
    print("[raised] cone", plan.cone_half_angle_deg, "views", plan.aims[0].n_views)


def test_flat_preset():
    cfg = MockScanCfg(surface_type="flat")
    survey = _survey(extent_mm=(200.0, 150.0))
    plan = plan_scan(survey, K_TEST, SIZE_TEST, cfg, cam_to_base_T=np.eye(4))
    assert plan.mode == "quality", plan.mode
    assert plan.cone_half_angle_deg == 18.0, plan.cone_half_angle_deg
    assert plan.aims[0].n_views == 8, plan.aims[0].n_views
    print("[flat] cone", plan.cone_half_angle_deg, "views", plan.aims[0].n_views)


def test_aim_point_coords_identity_transform():
    cfg = MockScanCfg()
    centroid = np.array([12.0, -34.0, 500.0])
    normal = np.array([0.1, -0.2, -0.95])
    survey = _survey(extent_mm=(200.0, 150.0), centroid_cam_mm=centroid, normal_cam=normal)
    plan = plan_scan(survey, K_TEST, SIZE_TEST, cfg, cam_to_base_T=np.eye(4))
    aim = plan.aims[0]
    # identity transform: point unchanged, view_dir = -normal normalized
    assert np.allclose(aim.point_base_mm, centroid, atol=1e-9), aim.point_base_mm
    expected_dir = -normal / np.linalg.norm(normal)
    assert np.allclose(aim.view_dir_base, expected_dir, atol=1e-9), aim.view_dir_base
    assert abs(np.linalg.norm(aim.view_dir_base) - 1.0) < 1e-9
    print("[aim-identity] point", aim.point_base_mm.round(2),
          "dir", aim.view_dir_base.round(3))


def test_aim_point_coords_no_transform():
    cfg = MockScanCfg()
    centroid = np.array([5.0, 6.0, 450.0])
    survey = _survey(extent_mm=(200.0, 150.0), centroid_cam_mm=centroid)
    plan = plan_scan(survey, K_TEST, SIZE_TEST, cfg, cam_to_base_T=None)
    aim = plan.aims[0]
    # no transform: point stays in the camera frame == centroid_cam_mm
    assert np.allclose(aim.point_base_mm, centroid, atol=1e-9), aim.point_base_mm
    print("[aim-no-transform] point", aim.point_base_mm.round(2))


def test_reference_warning_not_framed():
    cfg = MockScanCfg()
    # large surface (reference mode) that is NOT fully framed
    survey = _survey(extent_mm=(1500.0, 1200.0), fully_framed=False)
    plan = plan_scan(survey, K_TEST, SIZE_TEST, cfg, cam_to_base_T=np.eye(4))
    assert plan.mode == "reference", plan.mode
    assert any("not fully framed" in w for w in plan.warnings), plan.warnings
    print("[ref-warning]", plan.warnings)


def test_not_detected_returns_reference():
    cfg = MockScanCfg()
    survey = _survey(detected=False)
    plan = plan_scan(survey, K_TEST, SIZE_TEST, cfg, cam_to_base_T=np.eye(4))
    assert plan.mode == "reference", plan.mode
    assert len(plan.aims) == 0, plan.aims
    assert any("no surface detected" in w for w in plan.warnings), plan.warnings
    print("[not-detected] mode", plan.mode, "warnings", plan.warnings)


if __name__ == "__main__":
    test_small_surface_quality_mode()
    test_large_surface_reference_mode()
    test_standoff_clamped_below()
    test_standoff_clamped_above_quality_boundary()
    test_voxel_scales_with_standoff()
    test_fov_math()
    test_raised_preset()
    test_flat_preset()
    test_aim_point_coords_identity_transform()
    test_aim_point_coords_no_transform()
    test_reference_warning_not_framed()
    test_not_detected_returns_reference()
    print("\nplanner.py scan-plan tests passed.")
