"""Ring-stack measure-only experiment: synthetic proof, processing, jobs, API."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import extrusion_synthetic as syn
from tasni.modules.extrusion.processing import depth_to_work_points


def test_renderer_puts_a_ring_where_it_says_at_the_height_it_says():
    center = (200.0, 150.0)
    T = syn.inspection_camera_T([center[0], center[1], 6.0], 300.0)
    depth = syn.render_scene([syn.RingSpec(60.0, 8.0, center, height_fn=syn.flat(6.0))], T,
                             plane_center_xy_mm=center, noise_mm=0.0)
    assert depth.dtype == np.uint16 and depth.shape == (720, 1280)
    points, raw = depth_to_work_points(depth, syn.K_720P, T)
    assert raw > 100_000                                  # plane + ring both rendered
    ring = points[points[:, 2] > 3.0]
    radii = np.linalg.norm(ring[:, :2] - np.array(center), axis=1)
    assert 55.0 < radii.min() and radii.max() < 65.0     # 60 +/- bead/2 (+ rounding)
    assert 5.0 < ring[:, 2].max() < 7.0                   # crest at 6 mm
    plane = points[points[:, 2] <= 1.0]
    assert len(plane) > 50_000


# ---------------------------------------------------------------- circle metrics

from tasni.modules.extrusion.comparison import compare_circle


def test_shifted_circle_reports_its_offset_and_zero_shape_error():
    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    shifted = np.column_stack((40 * np.cos(theta) + 10, 40 * np.sin(theta), np.full(360, 5.0)))
    m = compare_circle(shifted, 40.0, nominal_center_mm=(0.0, 0.0))
    assert m.center_offset_mm == pytest.approx((10.0, 0.0), abs=1e-6)
    assert m.center_offset_norm_mm == pytest.approx(10.0, abs=1e-6)
    assert m.shape_rms_mm < 1e-6 and m.shape_max_mm < 1e-6
    # Deviation is still measured from the NOMINAL centre (the paper's number).
    assert m.mean_absolute_mm == pytest.approx(6.35, abs=0.05)
    assert m.rms_mm == pytest.approx(7.06, abs=0.05)
    assert m.maximum_mm == pytest.approx(10.0, abs=0.05)


def test_old_metrics_payload_without_offset_fields_still_validates():
    from tasni.modules.extrusion.models import DeviationMetrics
    old = DeviationMetrics(mean_absolute_mm=1, rms_mm=1, maximum_mm=1,
                           measured_center_mm=(0, 0), measured_radius_mm=40,
                           path_completeness=1, maximum_angular_gap_deg=2, valid=True)
    assert old.center_offset_norm_mm == 0.0 and old.shape_rms_mm == 0.0


# ------------------------------------------------- end-to-end synthetic processing

from tasni.core.config import ExtrusionConfig
from tasni.modules.extrusion.inspection import aim_point_mm
from tasni.modules.extrusion.models import CylinderRecipe, CylinderSetup
from tasni.modules.extrusion.processing import process_observation
from tasni.modules.extrusion.toolpath import generate_cylinder_plan

CENTER = (200.0, 150.0)


def scene_plan(*, radius=60.0, bead=8.0, layers=1, layer_height=6.0, center=CENTER):
    recipe = CylinderRecipe(radius_mm=radius, layer_count=layers, layer_height_mm=layer_height,
                            bead_diameter_mm=bead, robot_speed_mm_s=75, extrusion_rate_pct=0,
                            points_per_circle=180)
    setup = CylinderSetup(print_tool="LongCalibTool", work_frame="Tasni Work Frame",
                          inspection_tool="Realsense", inspection_auto=True,
                          center_x_mm=center[0], center_y_mm=center[1])
    return generate_cylinder_plan(recipe, setup)


def observe(plan, layer_index, rings, *, config=None, floor_profile=None, seed=0):
    """Render the rings from the derived inspection pose and process that frame."""
    layer = plan.layers[layer_index - 1]
    T = syn.inspection_camera_T(aim_point_mm(plan.recipe, plan.setup, layer_index), 300.0)
    depth = syn.render_scene(rings, T, plane_center_xy_mm=(plan.setup.center_x_mm,
                                                           plan.setup.center_y_mm), seed=seed)
    color = np.zeros((syn.SIZE_720P[1], syn.SIZE_720P[0], 3), np.uint8)
    kwargs = {} if floor_profile is None else {"floor_profile": floor_profile}
    return process_observation(color=color, depth=depth, T_work_camera=T, K=syn.K_720P,
                               plan=plan, layer=layer, config=config or ExtrusionConfig(),
                               **kwargs)


def test_true_ring_measures_as_true():
    pytest.importorskip("open3d")
    plan = scene_plan()
    out = observe(plan, 1, [syn.RingSpec(60.0, 8.0, CENTER, height_fn=syn.flat(6.0))])
    m = out.metrics
    assert m.valid, m.warnings
    assert m.mean_absolute_mm < 1.0 and m.rms_mm < 1.0
    assert abs(m.measured_radius_mm - 60.0) < 1.0
    assert m.center_offset_norm_mm < 1.0
    assert m.path_completeness >= 0.95
    assert out.report["timings_ms"]["total_ms"] > 0


def test_ring_shifted_10mm_reports_the_shift():
    pytest.importorskip("open3d")
    plan = scene_plan()
    out = observe(plan, 1, [syn.RingSpec(60.0, 8.0, (CENTER[0] + 10.0, CENTER[1]),
                                          height_fn=syn.flat(6.0))])
    m = out.metrics
    assert m.valid, m.warnings
    assert m.center_offset_mm[0] == pytest.approx(10.0, abs=1.0)
    assert abs(m.center_offset_mm[1]) < 1.0
    assert m.maximum_mm == pytest.approx(10.0, abs=1.5)
    assert m.mean_absolute_mm == pytest.approx(6.36, abs=1.0)
    assert m.rms_mm == pytest.approx(7.06, abs=1.0)
    assert m.shape_rms_mm < 1.0


def test_floor_from_previous_layer_keeps_the_ring_below_out_of_the_measurement():
    pytest.importorskip("open3d")
    plan = scene_plan(layers=2, layer_height=6.0)
    ring1 = syn.RingSpec(60.0, 8.0, CENTER, height_fn=syn.flat(6.0))
    first = observe(plan, 1, [ring1])
    assert first.metrics.valid and first.report["floor"]["source"] == "build_plane"

    ring2 = syn.RingSpec(60.0, 8.0, (CENTER[0] + 10.0, CENTER[1]), z_base_mm=6.0,
                         height_fn=syn.flat(6.0))
    floored = observe(plan, 2, [ring1, ring2], floor_profile=first.measured_xyz)
    assert floored.metrics.valid, floored.metrics.warnings
    assert floored.report["floor"]["source"] == "previous_layer_measured"
    assert floored.metrics.center_offset_norm_mm == pytest.approx(10.0, abs=1.5)

    # Without the floor the exposed crescent of ring 1 contaminates the answer:
    # either the branch guard rejects it, or the offset is pulled well under 10.
    try:
        blended = observe(plan, 2, [ring1, ring2])
    except RuntimeError:
        return
    assert (abs(blended.metrics.center_offset_norm_mm - 10.0)
            > abs(floored.metrics.center_offset_norm_mm - 10.0) + 1.0)


# ------------------------------------------------------ ring geometry (Task 5)

def test_wavy_ring_height_profile_is_measured():
    pytest.importorskip("open3d")
    plan = scene_plan(layer_height=7.5)
    out = observe(plan, 1, [syn.RingSpec(60.0, 8.0, CENTER, height_fn=syn.wavy(7.5, 2.5, lobes=2))])
    g = out.geometry
    assert g is not None and g.height_reference == "build_plane"
    assert g.top_z_min_mm == pytest.approx(5.0, abs=1.5)
    assert g.top_z_max_mm == pytest.approx(10.0, abs=1.5)
    assert g.top_z_std_mm > 1.0
    assert g.height_mean_mm == pytest.approx(7.5, abs=1.0)


def test_bead_width_is_the_rings_radial_footprint():
    pytest.importorskip("open3d")
    plan = scene_plan()
    out = observe(plan, 1, [syn.RingSpec(60.0, 8.0, CENTER, height_fn=syn.flat(6.0))])
    g = out.geometry
    assert g.bead_width_mean_mm == pytest.approx(8.0, rel=0.25)
    assert g.bead_width_bins == 36


def test_bead_width_profile_on_an_ideal_annulus():
    from tasni.modules.extrusion.processing import bead_width_profile
    rng = np.random.default_rng(1)
    theta = rng.uniform(0, 2 * np.pi, 20000)
    r = rng.uniform(36.0, 44.0, 20000)                # annulus 40 +/- 4 -> width 8
    pts = np.column_stack((r * np.cos(theta), r * np.sin(theta), np.zeros(20000)))
    w = bead_width_profile(pts, (0.0, 0.0), bins=36)
    assert w["bins_with_data"] == 36
    assert w["mean_mm"] == pytest.approx(8.0, abs=0.6)   # p97.5 - p2.5 of a uniform 8 mm band
