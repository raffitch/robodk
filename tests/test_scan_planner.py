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
import pytest

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


# -- plan_rect_tour: tiled close-range tour over a surveyed large rectangle ---
# (Task 12, two-path plan §7 hand-off — the five-position survey measures a
# platform too large for one camera view; plan_rect_tour tiles it with
# overlapping close-range views instead of backing the camera off.)

def test_rect_tour_tiles_cover_a_large_rectangle():
    import numpy as np
    from tasni.modules.scan.planner import plan_rect_tour
    from tasni.core.config import ScanConfig
    corners = np.array([[0, 0, 0], [2000, 0, 0], [2000, 1200, 0], [0, 1200, 0]], float)
    K = [[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]
    plan = plan_rect_tour(corners, [0, 0, 1], K, (1280, 720), ScanConfig())
    assert plan.mode == "large_survey"
    assert len(plan.aims) >= 4                      # 2000x1200 needs a grid at ~350 mm
    pts = np.array([a.point_base_mm for a in plan.aims])
    assert pts[:, 0].min() > 0 and pts[:, 0].max() < 2000    # aims inside the rectangle
    assert np.allclose(pts[:, 2], 0.0, atol=1e-6)            # aims on the plane
    for a in plan.aims:
        assert a.standoff_mm == ScanConfig().accurate_min_mm
        assert np.allclose(a.view_dir_base, [0, 0, -1])
    print("[rect-tour] large rectangle ->", len(plan.aims), "tiles")


def test_rect_tour_small_rectangle_is_single_tile():
    import numpy as np
    from tasni.modules.scan.planner import plan_rect_tour
    from tasni.core.config import ScanConfig
    corners = np.array([[0, 0, 0], [200, 0, 0], [200, 150, 0], [0, 150, 0]], float)
    K = [[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]
    plan = plan_rect_tour(corners, [0, 0, 1], K, (1280, 720), ScanConfig())
    assert len(plan.aims) == 1
    print("[rect-tour] small rectangle -> single tile")


def test_rect_tour_min_perpendicular_matches_plan_scan_convention():
    """min_perpendicular_mm must equal plan_scan's d_fit for a tile: d_fit =
    max(m*Sx*fx/W, m*Sy*fy/H); substituting the tile's own footprint
    (Sx=foot_w=d*W/fx/m, Sy=foot_h=d*H/fy/m) makes both branches collapse to
    exactly d, independent of K/margin — this is the pre-flight resolution
    from the controller, verified numerically here with an ASYMMETRIC K/size
    so fx != fy and W != H (a symmetric camera would hide a mismatched axis)."""
    import numpy as np
    from tasni.modules.scan.planner import plan_rect_tour
    from tasni.core.config import ScanConfig
    corners = np.array([[0, 0, 0], [3000, 0, 0], [3000, 2000, 0], [0, 2000, 0]], float)
    K = [[850.0, 0, 640], [0, 720.0, 360], [0, 0, 1]]      # fx != fy
    cfg = ScanConfig(frame_margin=1.2)
    plan = plan_rect_tour(corners, [0, 0, 1], K, (1280, 800), cfg)  # W != H
    for a in plan.aims:
        assert a.min_perpendicular_mm == pytest.approx(cfg.accurate_min_mm, abs=1e-9)
    print("[rect-tour] min_perpendicular_mm == accurate_min_mm for every tile",
          "(asymmetric K/size)")


def test_rect_tour_tiles_overlap_and_cover_rectangle_by_construction():
    """Ambiguity resolution #2: verify BY CONSTRUCTION (not just aim count) that
    adjacent tile footprints overlap by >= survey_tour_overlap and the union of
    footprints spans the full rectangle including its edges — a formula that
    left a gap or missed an edge band would silently under-scan the surface."""
    import numpy as np
    from tasni.modules.scan.planner import plan_rect_tour
    from tasni.core.config import ScanConfig
    corners = np.array([[0, 0, 0], [2000, 0, 0], [2000, 1200, 0], [0, 1200, 0]], float)
    K = [[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]
    W, H = 1280, 720
    cfg = ScanConfig()
    plan = plan_rect_tour(corners, [0, 0, 1], K, (W, H), cfg)

    d = cfg.accurate_min_mm
    fx, fy = K[0][0], K[1][1]
    foot_w = d * W / fx / cfg.frame_margin
    foot_h = d * H / fy / cfg.frame_margin

    pts = np.array([a.point_base_mm for a in plan.aims])
    xs = sorted(set(round(float(x), 3) for x in pts[:, 0]))
    ys = sorted(set(round(float(y), 3) for y in pts[:, 1]))
    assert len(xs) >= 2 and len(ys) >= 2, (xs, ys)   # sanity: this really is a grid

    # 1) No gap: consecutive tile-centre spacing must not exceed the footprint
    #    (otherwise a strip between two tiles would see no view at all).
    step_x = xs[1] - xs[0]
    step_y = ys[1] - ys[0]
    assert max(np.diff(xs)) <= foot_w + 1e-6, (xs, foot_w)
    assert max(np.diff(ys)) <= foot_h + 1e-6, (ys, foot_h)

    # 2) Overlap: adjacent footprints overlap by at least the configured
    #    fraction (the ceil() in tile-count math can only ADD overlap, never
    #    remove it below the target).
    overlap_x = 1.0 - step_x / foot_w
    overlap_y = 1.0 - step_y / foot_h
    assert overlap_x >= cfg.survey_tour_overlap - 1e-6, overlap_x
    assert overlap_y >= cfg.survey_tour_overlap - 1e-6, overlap_y

    # 3) Full coverage including edges: the outermost tiles' footprints must
    #    reach past the rectangle's own edges (0 and Lx/Ly here).
    x_lo, x_hi = xs[0] - foot_w / 2, xs[-1] + foot_w / 2
    y_lo, y_hi = ys[0] - foot_h / 2, ys[-1] + foot_h / 2
    assert x_lo <= 0.0 and x_hi >= 2000.0, (x_lo, x_hi)
    assert y_lo <= 0.0 and y_hi >= 1200.0, (y_lo, y_hi)

    print(f"[rect-tour] {len(plan.aims)} tiles; step=({step_x:.1f},{step_y:.1f}) mm "
          f"footprint=({foot_w:.1f},{foot_h:.1f}) mm "
          f"measured overlap=({overlap_x:.1%},{overlap_y:.1%}) vs configured "
          f"{cfg.survey_tour_overlap:.0%}; x span [{x_lo:.0f},{x_hi:.0f}] "
          f"y span [{y_lo:.0f},{y_hi:.0f}] (rect is [0,2000]x[0,1200])")


def test_rect_tour_aims_lie_on_a_tilted_offset_plane():
    """Ambiguity resolution #3: the rectangle is generally NOT axis-aligned in
    the base frame and its plane is generally NOT z=0. Rotate+translate the
    test rectangle off-axis and confirm every aim still lies exactly on the
    rectangle's own plane (not silently projected to z=0)."""
    import numpy as np
    from tasni.modules.scan.planner import plan_rect_tour
    from tasni.core.config import ScanConfig
    base = np.array([[0, 0, 0], [2000, 0, 0], [2000, 1200, 0], [0, 1200, 0]], float)
    ax, ay = np.radians(20.0), np.radians(15.0)
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    R = Ry @ Rx
    t = np.array([500.0, -300.0, 800.0])
    corners = (R @ base.T).T + t
    normal_base = R @ np.array([0.0, 0.0, 1.0])

    K = [[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]
    plan = plan_rect_tour(corners, normal_base, K, (1280, 720), ScanConfig())
    assert len(plan.aims) >= 4

    plane_point = corners.mean(axis=0)
    n = normal_base / np.linalg.norm(normal_base)
    if n[2] < 0:            # plan_rect_tour up-orients the normal the same way
        n = -n
    residuals = [abs(float((np.asarray(a.point_base_mm) - plane_point) @ n))
                 for a in plan.aims]
    assert max(residuals) < 1e-6, residuals
    for a in plan.aims:
        assert np.allclose(a.view_dir_base, -n, atol=1e-9)
    print(f"[rect-tour] tilted/offset plane -> {len(plan.aims)} aims, "
          f"max plane residual {max(residuals):.2e} mm")


# -- post-review fixes (Findings 1/2 + follow-ups 3/4) -----------------------

def test_rect_tour_rejects_overlap_equal_to_one():
    """Finding 1 (post-review): overlap=1.0 -> step=0 -> would divide by zero
    inside the tile-count math. Must fail loudly with a clear ValueError, not
    a raw ZeroDivisionError (which module.py used to turn into a misleading
    HTTP 503 'RoboDK/camera unavailable' for what is really a config typo)."""
    import numpy as np
    from tasni.modules.scan.planner import plan_rect_tour
    from tasni.core.config import ScanConfig
    corners = np.array([[0, 0, 0], [2000, 0, 0], [2000, 1200, 0], [0, 1200, 0]], float)
    K = [[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]
    cfg = ScanConfig(survey_tour_overlap=1.0)
    with pytest.raises(ValueError, match="survey_tour_overlap"):
        plan_rect_tour(corners, [0, 0, 1], K, (1280, 720), cfg)
    print("[rect-tour] overlap=1.0 correctly rejected (no ZeroDivisionError)")


def test_rect_tour_rejects_overlap_above_one():
    """overlap=1.3 -> step=-0.3 -> ceil(negative) collapses through max(1, ...)
    to a SINGLE tile for the whole platform while the coverage gate (which
    assumes one tile per patch) would report 100% -- must be rejected."""
    import numpy as np
    from tasni.modules.scan.planner import plan_rect_tour
    from tasni.core.config import ScanConfig
    corners = np.array([[0, 0, 0], [2000, 0, 0], [2000, 1200, 0], [0, 1200, 0]], float)
    K = [[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]
    cfg = ScanConfig(survey_tour_overlap=1.3)
    with pytest.raises(ValueError, match="survey_tour_overlap"):
        plan_rect_tour(corners, [0, 0, 1], K, (1280, 720), cfg)
    print("[rect-tour] overlap=1.3 correctly rejected")


def test_rect_tour_rejects_negative_overlap():
    """overlap=-0.5 -> step=1.5 -> spacing exceeds the footprint, opening a
    gap between every tile -- must be rejected, not silently tiled with holes."""
    import numpy as np
    from tasni.modules.scan.planner import plan_rect_tour
    from tasni.core.config import ScanConfig
    corners = np.array([[0, 0, 0], [2000, 0, 0], [2000, 1200, 0], [0, 1200, 0]], float)
    K = [[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]
    cfg = ScanConfig(survey_tour_overlap=-0.5)
    with pytest.raises(ValueError, match="survey_tour_overlap"):
        plan_rect_tour(corners, [0, 0, 1], K, (1280, 720), cfg)
    print("[rect-tour] overlap=-0.5 correctly rejected")


def test_rect_tour_valid_overlap_boundary_zero_is_allowed():
    """overlap=0.0 (step=1.0, tiles exactly touching, no overlap) is the
    minimum VALID value and must NOT be rejected -- confirms the guard's
    boundary is [0.0, 1.0), not (0.0, 1.0)."""
    import numpy as np
    from tasni.modules.scan.planner import plan_rect_tour
    from tasni.core.config import ScanConfig
    corners = np.array([[0, 0, 0], [2000, 0, 0], [2000, 1200, 0], [0, 1200, 0]], float)
    K = [[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]
    cfg = ScanConfig(survey_tour_overlap=0.0)
    plan = plan_rect_tour(corners, [0, 0, 1], K, (1280, 720), cfg)
    assert len(plan.aims) >= 1
    print("[rect-tour] overlap=0.0 (touching tiles) accepted,", len(plan.aims), "tiles")


def test_tile_grid_dims_matches_plan_rect_tour_aim_count():
    """_tile_grid_dims (factored out for the contiguous-empty-tile check,
    Finding 2) must describe EXACTLY the grid plan_rect_tour actually built:
    nx*ny == len(plan.aims), for both a multi-tile and a single-tile case."""
    import numpy as np
    from tasni.modules.scan.planner import _tile_grid_dims, plan_rect_tour
    from tasni.core.config import ScanConfig
    K = [[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]
    cfg = ScanConfig()
    for corners in (
        np.array([[0, 0, 0], [2000, 0, 0], [2000, 1200, 0], [0, 1200, 0]], float),
        np.array([[0, 0, 0], [200, 0, 0], [200, 150, 0], [0, 150, 0]], float),
    ):
        plan = plan_rect_tour(corners, [0, 0, 1], K, (1280, 720), cfg)
        nx, ny, foot_w, foot_h = _tile_grid_dims(corners, K, (1280, 720), cfg)
        assert nx * ny == len(plan.aims), (nx, ny, len(plan.aims))
        assert foot_w > 0 and foot_h > 0
    print("[rect-tour] _tile_grid_dims grid shape matches plan_rect_tour aim count")


def test_largest_contiguous_empty_block_basic_shapes():
    """Pure combinatorics sanity check for the Finding 2 helper: an empty
    grid, isolated singletons, an L-shape, and a solid 3x3 block."""
    from tasni.modules.scan.planner import _largest_contiguous_empty_block
    assert _largest_contiguous_empty_block(set()) == 0
    assert _largest_contiguous_empty_block({(0, 0)}) == 1
    # two DIAGONAL (not 4-connected) singletons -> largest block is still 1
    assert _largest_contiguous_empty_block({(0, 0), (1, 1)}) == 1
    # a 2-tile horizontal run
    assert _largest_contiguous_empty_block({(0, 0), (1, 0)}) == 2
    # an L-shape (3 tiles, 4-connected)
    assert _largest_contiguous_empty_block({(0, 0), (1, 0), (1, 1)}) == 3
    # a solid 3x3 block among scattered singleton noise elsewhere
    block = {(i, j) for i in range(3, 6) for j in range(3, 6)}
    noise = {(0, 0), (7, 7)}
    assert _largest_contiguous_empty_block(block | noise) == 9
    print("[contiguous-block] basic shapes verified")


def test_rect_tour_epsilon_guard_never_undertiles():
    """Independent re-verification (post-review) that the epsilon subtracted
    before ceil() (added to fix the float-boundary bug) cannot instead cause
    UNDER-tiling: a gap needs tile-centre spacing (Lx/nx) to exceed the
    footprint, which needs nx to be too SMALL by a factor of >= 1/step (~1.43x
    at the default 0.30 overlap) versus the eps-free ratio -- 1e-9 is nowhere
    near that. Swept across 5 orders of magnitude (200 mm to 1e6 mm, i.e. a
    tabletop up to a kilometre-scale platform)."""
    import numpy as np
    from tasni.modules.scan.planner import _tile_grid_dims
    from tasni.core.config import ScanConfig
    K = [[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]
    cfg = ScanConfig()
    for L in (200.0, 381.0, 500.0, 1000.0, 2000.0, 12345.678, 1e5, 1e6):
        corners = np.array([[0, 0, 0], [L, 0, 0], [L, L * 0.6, 0], [0, L * 0.6, 0]], float)
        nx, ny, foot_w, foot_h = _tile_grid_dims(corners, K, (1280, 720), cfg)
        Lx, Ly = L, L * 0.6
        step_x, step_y = Lx / nx, Ly / ny
        assert step_x <= foot_w + 1e-6, (L, step_x, foot_w)
        assert step_y <= foot_h + 1e-6, (L, step_y, foot_h)
    print("[rect-tour] epsilon guard never under-tiles across L=200 mm..1e6 mm")


def _pinhole_ground_footprint(T, K, size_px, plane_point, plane_normal):
    """Ray-plane intersection of the four image corners -> the camera's
    actual observed ground footprint (a quadrilateral, generally a trapezoid
    under tilt). Test-only verification tool (not production code) -- pure
    numpy, no matplotlib (not a declared project dependency)."""
    K = np.asarray(K, dtype=float)
    W, H = size_px
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    R, origin = np.asarray(T)[:3, :3], np.asarray(T)[:3, 3]
    pts = []
    for u, v in ((0, 0), (W, 0), (W, H), (0, H)):
        dir_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        dir_base = R @ dir_cam
        denom = float(np.dot(dir_base, plane_normal))
        if abs(denom) < 1e-9:
            return None
        s = float(np.dot(plane_point - origin, plane_normal)) / denom
        if s <= 0:
            return None
        pts.append(origin + s * dir_base)
    return np.array(pts)


def _points_in_convex_quad(pts, quad):
    """Vectorized point-in-convex-polygon via consistent edge cross-product
    sign (test-only; avoids an undeclared matplotlib dependency — verified
    against matplotlib.path.Path.contains_points during development, same
    result to 10+ significant figures)."""
    n = quad.shape[0]
    signs = None
    inside = np.ones(len(pts), dtype=bool)
    for i in range(n):
        a, b = quad[i], quad[(i + 1) % n]
        edge = b - a
        rel = pts - a
        cross = edge[0] * rel[:, 1] - edge[1] * rel[:, 0]
        s = cross >= -1e-6
        if signs is None:
            signs = s
        inside &= (s == signs)
    return inside


def test_rect_tour_worst_tilt_pose_per_tile_covers_the_rectangle():
    """Follow-up 4 (post-review): pins the premise the tile-completeness gate
    depends on -- "at least one surviving pose for a tile implies that
    tile's patch is actually imaged." Holds today by measurement (world-space
    union coverage >= 0.999 even keeping only the WIDEST-TILT candidate per
    tile, the worst case for ground-footprint size/shape), but a later change
    to flat_cone_deg / frame_margin / distance_jitter could break it silently
    -- this test pins it so such a change fails loudly HERE instead of only
    showing up as a mysterious hole in a real scan."""
    import numpy as np
    from tasni.modules.calibration.poses import generate_calibration_poses
    from tasni.modules.scan.planner import plan_rect_tour
    from tasni.core.config import ScanConfig

    corners = np.array([[0, 0, 0], [2000, 0, 0], [2000, 1200, 0], [0, 1200, 0]], float)
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    W, H = 1280, 720
    cfg = ScanConfig()
    plan = plan_rect_tour(corners, [0, 0, 1], K, (W, H), cfg)
    seed_T = np.eye(4)
    seed_T[2, 3] = 900.0
    plane_point = corners.mean(axis=0)
    plane_normal = np.array([0.0, 0.0, 1.0])

    worst_quads = []
    for aim in plan.aims:
        cands = generate_calibration_poses(
            seed_T, count=aim.n_views, look_distance_mm=aim.standoff_mm,
            cone_half_angle_deg=aim.cone_half_angle_deg, roll_max_deg=aim.roll_max_deg,
            distance_jitter=cfg.distance_jitter,
            target_center=np.asarray(aim.point_base_mm, float),
            target_normal=-np.asarray(aim.view_dir_base, float),
            min_perpendicular_mm=aim.min_perpendicular_mm)
        # Widest-tilt candidate: largest angle of its forward axis from
        # straight down -- the worst case for ground-footprint size/shape.
        angles = [float(np.degrees(np.arccos(np.clip(
            float(np.dot(T[:3, 2], [0, 0, -1])), -1.0, 1.0)))) for T in cands]
        worst = cands[int(np.argmax(angles))]
        q = _pinhole_ground_footprint(worst, K, (W, H), plane_point, plane_normal)
        assert q is not None, "worst-tilt candidate must still see the plane"
        worst_quads.append(q)

    res = 4.0
    xs = np.arange(0, 2000 + res, res)
    ys = np.arange(0, 1200 + res, res)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    covered = np.zeros(len(pts), dtype=bool)
    for q in worst_quads:
        covered |= _points_in_convex_quad(pts, q[:, :2])
    coverage = float(covered.mean())
    assert coverage >= 0.999, coverage
    print(f"[rect-tour] worst-tilt-pose-per-tile world-space union coverage "
          f"= {coverage:.4%} ({len(plan.aims)} tiles)")


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
    test_rect_tour_tiles_cover_a_large_rectangle()
    test_rect_tour_small_rectangle_is_single_tile()
    test_rect_tour_min_perpendicular_matches_plan_scan_convention()
    test_rect_tour_tiles_overlap_and_cover_rectangle_by_construction()
    test_rect_tour_aims_lie_on_a_tilted_offset_plane()
    test_rect_tour_rejects_overlap_equal_to_one()
    test_rect_tour_rejects_overlap_above_one()
    test_rect_tour_rejects_negative_overlap()
    test_rect_tour_valid_overlap_boundary_zero_is_allowed()
    test_tile_grid_dims_matches_plan_rect_tour_aim_count()
    test_largest_contiguous_empty_block_basic_shapes()
    test_rect_tour_epsilon_guard_never_undertiles()
    test_rect_tour_worst_tilt_pose_per_tile_covers_the_rectangle()
    print("\nplanner.py scan-plan tests passed.")
