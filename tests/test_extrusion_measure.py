"""Ring-stack measure-only experiment: synthetic proof, processing, jobs, API."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extrusion_synthetic as syn  # noqa: E402
import geometry_fixtures as gf  # noqa: E402
from tasni.modules.extrusion.processing import depth_to_work_points  # noqa: E402


def test_renderer_puts_a_ring_where_it_says_at_the_height_it_says():
    center = (200.0, 150.0)
    T = syn.inspection_camera_T([center[0], center[1], 6.0], 300.0)
    depth = syn.render_scene([syn.RingSpec(60.0, 8.0, center, height_fn=syn.flat(6.0))], T,
                             plane_center_xy_mm=center, noise_mm=0.0)
    assert depth.dtype == np.uint16 and depth.shape == (720, 1280)
    points, raw = depth_to_work_points(depth, gf.aligned(syn.K_720P, syn.SIZE_720P), T)
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
from tasni.modules.extrusion.toolpath import generate_cylinder_plan, points_array

CENTER = (200.0, 150.0)


def scene_plan(*, radius=60.0, bead=8.0, layers=1, layer_height=6.0, center=CENTER):
    recipe = CylinderRecipe(radius_mm=radius, layer_count=layers, layer_height_mm=layer_height,
                            bead_diameter_mm=bead, robot_speed_mm_s=75, extrusion_rate_pct=0,
                            points_per_circle=180)
    setup = CylinderSetup(print_tool="LongCalibTool", work_frame="Tasni Work Frame",
                          inspection_tool="Realsense", inspection_auto=True,
                          center_x_mm=center[0], center_y_mm=center[1])
    return generate_cylinder_plan(recipe, setup)


def observe(plan, layer_index, rings, *, config=None, floor_profile=None, seed=0,
            stages=None):
    """Render the rings from the derived inspection pose and process that frame."""
    layer = plan.layers[layer_index - 1]
    T = syn.inspection_camera_T(aim_point_mm(plan.recipe, plan.setup, layer_index), 300.0)
    depth = syn.render_scene(rings, T, plane_center_xy_mm=(plan.setup.center_x_mm,
                                                           plan.setup.center_y_mm), seed=seed)
    color = np.zeros((syn.SIZE_720P[1], syn.SIZE_720P[0], 3), np.uint8)
    kwargs = {} if floor_profile is None else {"floor_profile": floor_profile}
    if stages is not None:
        kwargs["stages"] = stages
    return process_observation(color=color, depth=depth,
                               geometry=gf.aligned(syn.K_720P, syn.SIZE_720P),
                               T_work_camera=T, K=syn.K_720P, dist=None,
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


def _board_bias_patch(center, *, r_from: float, r_to: float, half_height_mm: float,
                      z_mm: float, step_mm: float = 1.0) -> np.ndarray:
    """A patch of the build plane reading a few mm HIGH, touching the ring's outer flank.

    What the D435i does to the ChArUco board at 300 mm: broad patches biased by
    2-5 mm (measured 2026-08-28: bare board z p50 0.8 / p99 4.8 mm, 22.7% above the
    2.5 mm deposit floor). Flat, so its normals face straight up, and fused to the
    ring, so it lands in the ring's DBSCAN cluster.
    """
    xs = np.arange(center[0] + r_from, center[0] + r_to + step_mm, step_mm)
    ys = np.arange(center[1] - half_height_mm, center[1] + half_height_mm + step_mm, step_mm)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack((X.ravel(), Y.ravel(), np.full(X.size, float(z_mm))))


def observe_with_board_bias(plan, layer_index, rings, patch, *, seed=0):
    layer = plan.layers[layer_index - 1]
    T = syn.inspection_camera_T(aim_point_mm(plan.recipe, plan.setup, layer_index), 300.0)
    centre = (plan.setup.center_x_mm, plan.setup.center_y_mm)
    parts = [syn.plane_points(center_xy_mm=centre), patch]
    parts += [ring.surface_points() for ring in rings]
    depth = syn.render_depth(np.vstack(parts), T, seed=seed)
    color = np.zeros((syn.SIZE_720P[1], syn.SIZE_720P[0], 3), np.uint8)
    return process_observation(color=color, depth=depth,
                               geometry=gf.aligned(syn.K_720P, syn.SIZE_720P),
                               T_work_camera=T, K=syn.K_720P, dist=None,
                               plan=plan, layer=layer, config=ExtrusionConfig())


def test_board_depth_bias_fused_to_the_ring_does_not_break_the_measurement():
    """Cell 2026-08-28 20:48: a board patch at ~3.5 mm, touching the ring, dilated
    into a lobe with a 37 mm skeleton arm and exhausted the branch guard.

    The fix has to work by SHAPE -- the patch is inside the bead's own height band
    (bead z p25 1.8 / p50 3.8 mm), so no floor separates them without also
    destroying the ring (a 3 mm floor read r 36.7 for a 42.6 mm ring, and passed).
    """
    pytest.importorskip("open3d")
    plan = scene_plan()
    ring = syn.RingSpec(60.0, 8.0, CENTER, height_fn=syn.flat(6.0))
    patch = _board_bias_patch(CENTER, r_from=62.0, r_to=100.0, half_height_mm=14.0, z_mm=3.5)

    out = observe_with_board_bias(plan, 1, [ring], patch)

    m = out.metrics
    assert m.valid, m.warnings
    assert m.measured_radius_mm == pytest.approx(60.0, abs=1.0)
    assert m.center_offset_norm_mm < 1.0
    assert m.path_completeness >= 0.95
    r = np.linalg.norm(np.asarray(out.filtered_xyz)[:, :2] - np.asarray(CENTER), axis=1)
    assert r.max() < 60.0 + 8.0 + 4.0, "board points beyond the bead must not reach the raster"
    assert out.report["counts"]["after_radial_trim"] < out.report["counts"]["after_largest_cluster"]


def test_radial_trim_follows_a_displaced_ring_not_the_nominal():
    """The trim is about the FITTED circle: a ring shifted 12 mm must still read 12."""
    pytest.importorskip("open3d")
    plan = scene_plan()
    moved = (CENTER[0] + 12.0, CENTER[1])
    ring = syn.RingSpec(60.0, 8.0, moved, height_fn=syn.flat(6.0))
    patch = _board_bias_patch(moved, r_from=62.0, r_to=100.0, half_height_mm=14.0, z_mm=3.5)

    out = observe_with_board_bias(plan, 1, [ring], patch)

    m = out.metrics
    assert m.valid, m.warnings
    assert m.center_offset_mm[0] == pytest.approx(12.0, abs=1.0)
    assert m.measured_radius_mm == pytest.approx(60.0, abs=1.0)


def test_real_frame_with_board_noise_measures_the_applied_ring():
    """The 2026-08-28 20:48 cell frame that failed with `branch guard exhausted`."""
    pytest.importorskip("open3d")
    fixture = np.load(Path(__file__).parent / "fixtures" / "extrusion" / "ring2"
                      / "ring2_board_noise_20260828.npz")
    centre = tuple(float(v) for v in fixture["nominal_center_mm"])
    plan = scene_plan(radius=float(fixture["recipe_radius_mm"]),
                      bead=float(fixture["recipe_bead_mm"]),
                      layer_height=float(fixture["recipe_layer_height_mm"]),
                      center=centre)
    depth = fixture["depth"]
    color = np.zeros((*depth.shape, 3), np.uint8)

    out = process_observation(
        color=color, depth=depth,
        geometry=gf.aligned(fixture["K"], (depth.shape[1], depth.shape[0])),
        T_work_camera=fixture["T_work_camera"], K=fixture["K"], dist=None,
        plan=plan, layer=plan.layers[0], config=ExtrusionConfig())

    m = out.metrics
    assert m.valid, m.warnings
    # Characterize, minutes earlier on the same ring: r 42.60, centre (214.64, 146.69).
    assert m.measured_radius_mm == pytest.approx(42.6, abs=1.0)
    assert m.measured_center_mm == pytest.approx(centre, abs=2.0)
    assert m.path_completeness >= 0.95
    assert m.shape_rms_mm < 2.5
    r = np.linalg.norm(np.asarray(out.filtered_xyz)[:, :2] - np.asarray(centre), axis=1)
    assert not (r > 55.0).any(), "the board lobe at r 55-72 mm must be gone (was 21% of points)"


# ------------------------------------------- chroma gate: bead vs board by colour

from tasni.core.depth_geometry import ColorRegistered
from tasni.modules.extrusion.processing import chroma_gate_mask, deposit_floor_mm

RING1_TAKE04 = (Path(__file__).parent / "fixtures" / "extrusion" / "ring1"
                / "ring1_take04_branchguard_20260829.npz")


def _ring1_take04() -> dict:
    """Cell trial 20260829-165938, layer 1 take 4: the frame that crashed."""
    import cv2
    fixture = np.load(RING1_TAKE04)
    centre = tuple(float(v) for v in fixture["nominal_center_mm"])
    return {
        "depth": fixture["depth"], "K": fixture["K"],
        "T_work_camera": fixture["T_work_camera"], "centre": centre,
        "color": cv2.imdecode(fixture["color_jpeg"], cv2.IMREAD_COLOR),
        "plan": scene_plan(radius=float(fixture["recipe_radius_mm"]),
                           bead=float(fixture["recipe_bead_mm"]),
                           layer_height=float(fixture["recipe_layer_height_mm"]),
                           center=centre),
    }


def test_chroma_gate_clears_the_board_lobe_that_exhausted_the_branch_guard():
    """The 2026-08-29 13:03 cell frame. Board patch welded to the ring's +X flank."""
    pytest.importorskip("open3d")
    f = _ring1_take04()

    # This fixture is a pre-protocol-2 capture with 1 mm depth WORDS, so it is
    # processed at the 2 mm voxel it was captured under. The new 1 mm default
    # (spec 4.4) sits at this archive's quantisation floor, where it merges
    # nothing and lets noise through -- it flips this frame's branch-guard
    # outcome in BOTH directions across the two archived takes. On protocol-2
    # depth (0.1 mm words) 1 mm spans ten quantisation steps, which is the point.
    # Pinned here too (Ruling R23): this is the gate-ENABLED half of the pair
    # below (test_the_same_frame_still_fails_with_the_chroma_gate_disabled) --
    # the pair's whole claim, "crashes with the gate off, survives with it on",
    # only holds if both halves measure the SAME frame at the SAME voxel.
    out = process_observation(color=f["color"], depth=f["depth"],
                              geometry=gf.aligned(f["K"], (1280, 720)), K=f["K"], dist=None,
                              T_work_camera=f["T_work_camera"], plan=f["plan"],
                              layer=f["plan"].layers[0],
                              config=ExtrusionConfig(voxel_size_m=0.002))

    m = out.metrics
    assert m.valid, m.warnings
    assert out.report["counts"]["chroma_gate_applied"] == 1
    # Takes 1-3 of the same trial measured this ring as an arc with a 41-46 deg
    # hole; with the board gone and the floor it earns, the ring closes.
    assert out.report["closed"]
    assert m.path_completeness >= 0.98
    assert m.maximum_angular_gap_deg < 5.0
    # Characterization of this ring, minutes earlier: r 42.2 mm.
    assert m.measured_radius_mm == pytest.approx(42.2, abs=1.0)
    cluster = np.asarray(out.filtered_xyz)
    r = np.linalg.norm(cluster[:, :2] - np.asarray(f["centre"]), axis=1)
    # The patch sat on the ring's +X flank at r 50-54 mm (raw, out to r 71 mm);
    # 285 of its points reached the deposit cluster and 22 the crest, where they
    # dilated into the 17 and 22 px skeleton arms the guard refused.
    assert not ((cluster[:, 0] > 254.0) & (r > 47.0)).any(), "the +X board patch must be gone"
    assert r.max() < 54.0, "nothing may survive past the bead's own outer flank"


def test_the_same_frame_still_fails_with_the_chroma_gate_disabled():
    """Locks the CAUSE. Without colour the frame reproduces the cell crash exactly.

    Guards against 'fixing' this by loosening the branch guard instead: the guard
    was right, and takes 1-3 carried the same contamination into radii biased
    0.6-0.7 mm large precisely because their topology let them through.
    """
    pytest.importorskip("open3d")
    f = _ring1_take04()
    # This fixture is a pre-protocol-2 capture with 1 mm depth WORDS, so it is
    # processed at the 2 mm voxel it was captured under. The new 1 mm default
    # (spec 4.4) sits at this archive's quantisation floor, where it merges
    # nothing and lets noise through -- it flips this frame's branch-guard
    # outcome in BOTH directions across the two archived takes. On protocol-2
    # depth (0.1 mm words) 1 mm spans ten quantisation steps, which is the point.
    with pytest.raises(RuntimeError, match="branch guard exhausted"):
        process_observation(color=f["color"], depth=f["depth"],
                            geometry=gf.aligned(f["K"], (1280, 720)), K=f["K"], dist=None,
                            T_work_camera=f["T_work_camera"], plan=f["plan"],
                            layer=f["plan"].layers[0],
                            config=ExtrusionConfig(deposit_min_saturation=0,
                                                   voxel_size_m=0.002))


# ------------------------------- branch-guard spur limit: measured, not nominal

def test_spur_guard_uses_the_deposits_own_footprint_not_an_inflated_recipe_bead():
    """A caller's ``recipe.bead_diameter_mm`` can be stale or simply wrong -- a
    fresh trial's generic default, an operator's guess before any ring has been
    characterized. The spur-pruning tolerance must not inherit that number
    uncritically: an OVERSTATED bead inflates the tolerance, which prunes a
    LONGER twig without asking whether it is real (see the next test for what
    that costs). Two calls on the SAME clean ring, one with the true bead and
    one with it more than doubled, must (a) both measure the ring the same way
    -- this is a topology guard, not a measurement stage -- and (b) both report
    a ``spur_guard_bead_mm`` clamped near the true footprint, not the inflated
    recipe value.
    """
    pytest.importorskip("open3d")
    T = syn.inspection_camera_T([CENTER[0], CENTER[1], 6.0], 300.0)
    ring = syn.RingSpec(40.0, 10.0, CENTER, height_fn=syn.flat(6.0))
    scene = np.vstack((syn.plane_points(center_xy_mm=CENTER), ring.surface_points()))
    depth = syn.render_depth(scene, T, noise_mm=0.3)
    color = np.zeros((720, 1280, 3), np.uint8)

    def run(bead_mm):
        plan = scene_plan(radius=40.0, bead=bead_mm, layer_height=6.0, center=CENTER)
        return process_observation(
            color=color, depth=depth, geometry=gf.aligned(syn.K_720P, syn.SIZE_720P),
            T_work_camera=T, K=syn.K_720P, dist=None, plan=plan, layer=plan.layers[0],
            config=ExtrusionConfig())

    true_bead = run(10.0)          # recipe already matches the physical ring
    inflated = run(25.0)           # recipe overstates the bead by 2.5x

    assert true_bead.report["valid"] and inflated.report["valid"]
    true_guard_mm = true_bead.report["counts"]["spur_guard_bead_mm"]
    inflated_guard_mm = inflated.report["counts"]["spur_guard_bead_mm"]
    assert true_guard_mm == pytest.approx(9.5, abs=1.5)
    # The whole point: NOT the inflated 25 mm the caller supplied.
    assert inflated_guard_mm < 15.0
    assert inflated_guard_mm == pytest.approx(true_guard_mm, abs=1.0)
    # The measurement itself is unmoved by the recipe's (wrong) bead guess.
    assert inflated.metrics.measured_radius_mm == pytest.approx(
        true_bead.metrics.measured_radius_mm, abs=1.0)
    assert (inflated.report["geometry"]["bead_width_mean_mm"]
            == pytest.approx(true_bead.report["geometry"]["bead_width_mean_mm"], abs=1.0))


def test_spur_guard_still_catches_real_contamination_regardless_of_recipe_bead():
    """The guard must not go silent just because a caller's recipe happens to be
    wrong in the SAFE direction either. A ring with a genuine tangential shelf
    of extra material welded to its outer edge (the synthetic stand-in for the
    2026-08-29 cell's board-patch-on-the-flank failure) has to keep tripping
    the branch guard whether the recipe's bead assumption is accurate or
    grossly inflated -- clamping the tolerance to the measured footprint must
    never let a genuinely contaminated topology through just because the
    caller supplied a bad number.
    """
    pytest.importorskip("open3d")
    T = syn.inspection_camera_T([CENTER[0], CENTER[1], 6.0], 300.0)
    ring = syn.RingSpec(40.0, 10.0, CENTER, height_fn=syn.flat(6.0))
    # A tangential shelf just proud of the ring's own outer edge, spanning a
    # 20 degree arc -- long enough that neither an accurate nor a doubled+
    # dilation/spur tolerance can absorb it into the clean loop.
    r0 = 40.0 + 10.0 / 2.0 + 3.0
    thetas = np.deg2rad(np.arange(0.0, 20.0, 0.5))
    radii = np.arange(r0 - 2.5, r0 + 2.5, 0.5)
    Th, R = np.meshgrid(thetas, radii, indexing="ij")
    shelf = np.column_stack((
        CENTER[0] + R.ravel() * np.cos(Th.ravel()),
        CENTER[1] + R.ravel() * np.sin(Th.ravel()),
        np.full(Th.size, 4.0)))
    scene = np.vstack((syn.plane_points(center_xy_mm=CENTER), ring.surface_points(), shelf))
    depth = syn.render_depth(scene, T, noise_mm=0.3)
    color = np.zeros((720, 1280, 3), np.uint8)

    for bead_mm in (10.0, 30.0):
        plan = scene_plan(radius=40.0, bead=bead_mm, layer_height=6.0, center=CENTER)
        with pytest.raises(RuntimeError, match="branch guard exhausted"):
            process_observation(
                color=color, depth=depth, geometry=gf.aligned(syn.K_720P, syn.SIZE_720P),
                T_work_camera=T, K=syn.K_720P, dist=None, plan=plan, layer=plan.layers[0],
                config=ExtrusionConfig())


def test_branch_guard_exhaustion_names_the_tolerance_it_gave_up_with():
    """An abort has to say WHICH tolerance it gave up with, and where it came from.

    Since the spur tolerance was clamped to the frame's own measured bead
    (``a0fabca``) there are two very different reasons this can raise, with
    opposite remedies: a genuinely contaminated frame (leave the guard alone --
    it is doing its job) or a clean ring whose measured bead came in narrow
    enough to drop ``spur_limit`` a pixel below what this frame needed. The
    number that separates them is ``spur_bead_mm``, and it was computed and then
    thrown away: it is recorded in ``counts["spur_guard_bead_mm"]``, but
    ``counts`` only reaches a caller through the report and this path raises
    before a report exists. So on the one occasion the number decides what to do
    next -- a take that aborted at the cell -- nobody could see it.

    Guarded by a test because the message is the entire diagnostic: the raw
    RGB-D is archived and the take can be reprocessed, so what an abort costs is
    the operator's time working out which of the two cases they are in.
    """
    pytest.importorskip("open3d")
    T = syn.inspection_camera_T([CENTER[0], CENTER[1], 6.0], 300.0)
    ring = syn.RingSpec(40.0, 10.0, CENTER, height_fn=syn.flat(6.0))
    r0 = 40.0 + 10.0 / 2.0 + 3.0
    thetas = np.deg2rad(np.arange(0.0, 20.0, 0.5))
    radii = np.arange(r0 - 2.5, r0 + 2.5, 0.5)
    Th, R = np.meshgrid(thetas, radii, indexing="ij")
    shelf = np.column_stack((
        CENTER[0] + R.ravel() * np.cos(Th.ravel()),
        CENTER[1] + R.ravel() * np.sin(Th.ravel()),
        np.full(Th.size, 4.0)))
    scene = np.vstack((syn.plane_points(center_xy_mm=CENTER), ring.surface_points(), shelf))
    depth = syn.render_depth(scene, T, noise_mm=0.3)
    color = np.zeros((720, 1280, 3), np.uint8)

    # The recipe deliberately overstates the bead 3x, so the clamped value and the
    # recipe value are far apart and the message cannot pass by quoting one twice.
    plan = scene_plan(radius=40.0, bead=30.0, layer_height=6.0, center=CENTER)
    cfg = ExtrusionConfig()
    with pytest.raises(RuntimeError, match="branch guard exhausted") as excinfo:
        process_observation(
            color=color, depth=depth, geometry=gf.aligned(syn.K_720P, syn.SIZE_720P),
            T_work_camera=T, K=syn.K_720P, dist=None, plan=plan, layer=plan.layers[0],
            config=cfg)

    message = str(excinfo.value)
    limit = re.search(r"spur_limit (\d+) px", message)
    assert limit, message
    clamped = re.search(r"1\.5 x ([\d.]+) mm bead", message)
    assert clamped, message

    # The tolerance quoted must be the one actually applied -- the same formula the
    # loop prunes with, on the CLAMPED bead, not the recipe's inflated 30 mm.
    spur_bead_mm = float(clamped.group(1))
    assert int(limit.group(1)) == max(2, int(math.ceil(
        1.5 * spur_bead_mm / cfg.raster_mm_per_pixel)))
    assert spur_bead_mm < 20.0, message           # clamped, not the recipe's 30 mm

    # And the two inputs that decide which remedy applies are both named, so the
    # reader can see the clamp fired rather than having to infer it.
    assert "recipe bead 30.000 mm" in message, message
    assert re.search(r"this frame measured [\d.]+ mm", message), message


def test_chroma_gate_keeps_the_chromatic_bead_and_blanks_the_achromatic_board():
    K = np.array([[400.0, 0, 20.0], [0, 400.0, 20.0], [0, 0, 1.0]])
    depth = np.full((40, 40), 300, np.uint16)
    color = np.dstack([np.full((40, 40), 40, np.uint8)] * 3)      # black checker
    color[10:30, 10:30] = (40, 110, 190)                          # tan clay
    reg = ColorRegistered.build(depth, gf.aligned(K, (40, 40)), K, None)

    keep, applied = chroma_gate_mask(color, reg, ExtrusionConfig())

    assert applied and keep.shape == (len(reg),)
    # Aligned/identity registration: uv_depth (the depth pixel that made the
    # point) and its projected colour pixel are the same (v, u) -- the region
    # selectors below are just the pixel test's crops, restated on points.
    on_clay = ((reg.uv_depth[:, 1] >= 15) & (reg.uv_depth[:, 1] < 25)
              & (reg.uv_depth[:, 0] >= 15) & (reg.uv_depth[:, 0] < 25))
    off_clay = (reg.uv_depth[:, 1] < 5) & (reg.uv_depth[:, 0] < 5)
    assert keep[on_clay].all()
    assert not keep[off_clay].any()


def test_chroma_gate_abstains_on_an_achromatic_frame_and_restores_the_floor():
    """An RGB dropout must not erase the deposit, nor drop the floor it earned."""
    config = ExtrusionConfig()
    K = np.array([[400.0, 0, 4.0], [0, 400.0, 4.0], [0, 0, 1.0]])
    depth = np.full((8, 8), 300, np.uint16)
    reg = ColorRegistered.build(depth, gf.aligned(K, (8, 8)), K, None)

    keep, applied = chroma_gate_mask(np.zeros((8, 8, 3), np.uint8), reg, config)
    assert not applied and keep.all()
    assert deposit_floor_mm(config, False) == pytest.approx(2.5)

    saturated = np.zeros((8, 8, 3), np.uint8)
    saturated[:, :, 2] = 200
    keep, applied = chroma_gate_mask(saturated, reg, config)
    assert applied and keep.all()
    assert deposit_floor_mm(config, True) == pytest.approx(1.5)


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


# ------------------------------------------------- characterize a ring (Task 6)

from tasni.modules.extrusion.processing import characterize_ring, fit_circle_xy


def test_characterize_recovers_a_ring_the_recipe_got_wrong():
    pytest.importorskip("open3d")
    # The recipe/plan says 75 mm radius, 6 mm bead. The physical ring is 60 / 8,
    # 6 mm tall, and sits 15 mm off the table centre.
    plan = scene_plan(radius=75.0, bead=6.0, layer_height=5.0)
    true_center = (CENTER[0] + 15.0, CENTER[1] - 10.0)
    T = syn.inspection_camera_T(aim_point_mm(plan.recipe, plan.setup, 1), 300.0)
    depth = syn.render_scene([syn.RingSpec(60.0, 8.0, true_center, height_fn=syn.flat(6.0))], T,
                             plane_center_xy_mm=CENTER)
    color = np.zeros((720, 1280, 3), np.uint8)
    found = characterize_ring(color=color, depth=depth,
                              geometry=gf.aligned(syn.K_720P, syn.SIZE_720P),
                              T_work_camera=T, K=syn.K_720P, dist=None,
                              search_center_mm=CENTER, work_frame="Tasni Work Frame",
                              config=ExtrusionConfig())
    assert found.radius_mm == pytest.approx(60.0, abs=1.0)
    assert found.center_mm[0] == pytest.approx(true_center[0], abs=1.0)
    assert found.center_mm[1] == pytest.approx(true_center[1], abs=1.0)
    assert found.bead_width_mm == pytest.approx(8.0, abs=2.0)
    assert found.top_z_mean_mm == pytest.approx(6.0, abs=1.5)
    assert found.report["coarse"]["radius_mm"] == pytest.approx(60.0, abs=3.0)
    assert found.measured_xyz.shape[1] == 3


def test_characterize_selects_ring_instead_of_larger_raised_patch():
    pytest.importorskip("open3d")
    true_center = (CENTER[0] + 8.0, CENTER[1] - 5.0)
    T = syn.inspection_camera_T([CENTER[0], CENTER[1], 6.0], 300.0)
    ring = syn.RingSpec(40.0, 14.0, true_center, height_fn=syn.flat(8.0))
    # A broad 3 mm-high residual, separated from the ring, models the real
    # checkerboard depth bias that used to win solely because it was largest.
    patch_x = np.arange(CENTER[0] + 80.0, CENTER[0] + 151.0, 1.0)
    patch_y = np.arange(CENTER[1] - 110.0, CENTER[1] + 111.0, 1.0)
    X, Y = np.meshgrid(patch_x, patch_y, indexing="ij")
    patch = np.column_stack((X.ravel(), Y.ravel(), np.full(X.size, 3.0)))
    scene = np.vstack((
        syn.plane_points(center_xy_mm=CENTER), ring.surface_points(), patch))
    depth = syn.render_depth(scene, T)
    color = np.zeros((720, 1280, 3), np.uint8)

    found = characterize_ring(
        color=color, depth=depth, geometry=gf.aligned(syn.K_720P, syn.SIZE_720P),
        T_work_camera=T, K=syn.K_720P, dist=None,
        search_center_mm=CENTER, work_frame="Tasni Work Frame",
        config=ExtrusionConfig())

    assert found.radius_mm == pytest.approx(40.0, abs=1.5)
    assert found.center_mm == pytest.approx(true_center, abs=1.5)
    candidates = found.report["ring_selector"]["candidates"]
    selected = next(candidate for candidate in candidates if candidate.get("selected"))
    assert selected["points"] < max(candidate["points"] for candidate in candidates)
    assert selected["angular_coverage"] >= 0.95
    assert selected["radial_span_ratio"] < 0.8


def test_characterize_real_checkerboard_capture_selects_the_visible_ring():
    pytest.importorskip("open3d")
    fixture = np.load(Path(__file__).parent / "fixtures" / "extrusion" / "ring1"
                      / "ring1_checkerboard_20260828.npz")
    depth = fixture["depth"]
    color = np.zeros((*depth.shape, 3), np.uint8)

    # This fixture is a pre-protocol-2 capture with 1 mm depth WORDS, so it is
    # processed at the 2 mm voxel it was captured under. The new 1 mm default
    # (spec 4.4) sits at this archive's quantisation floor, where it merges
    # nothing and lets noise through -- it flips this frame's branch-guard
    # outcome in BOTH directions across the two archived takes. On protocol-2
    # depth (0.1 mm words) 1 mm spans ten quantisation steps, which is the point.
    found = characterize_ring(
        color=color, depth=depth,
        geometry=gf.aligned(fixture["K"], (depth.shape[1], depth.shape[0])),
        T_work_camera=fixture["T_work_camera"], K=fixture["K"], dist=None,
        search_center_mm=fixture["search_center_mm"],
        work_frame="Tasni Work Frame", config=ExtrusionConfig(voxel_size_m=0.002))

    assert found.radius_mm == pytest.approx(39.17, abs=0.5)
    assert found.center_mm == pytest.approx((217.94, 150.44), abs=0.5)
    assert found.bead_width_mm == pytest.approx(13.26, abs=0.75)
    assert found.top_z_mean_mm == pytest.approx(6.14, abs=0.75)
    selector = found.report["ring_selector"]
    selected = next(candidate for candidate in selector["candidates"]
                    if candidate.get("selected"))
    largest = max(selector["candidates"], key=lambda candidate: candidate["points"])
    assert selected["points"] < largest["points"]
    assert selected["radius_mm"] == pytest.approx(41.12, abs=0.5)
    assert selected["angular_coverage"] >= 0.95
    assert not largest["eligible"]


def _thin_at(mean_mm: float, dips_deg, width_deg: float = 12.0, floor_mm: float = 1.0):
    """A ring that all but vanishes at ``dips_deg`` -- the real low-relief failure.

    The 2026-08-29 capture had a hand-placed dried ring 2-11 mm tall whose two
    thinnest arcs fell under the ROI height floor, so one loop reached DBSCAN as
    two disconnected arcs and the per-cluster shape gate rejected both.
    """
    def height(theta):
        h = np.full_like(theta, float(mean_mm), dtype=float)
        for dip in dips_deg:
            d = np.abs(np.mod(np.degrees(theta) - dip + 180.0, 360.0) - 180.0)
            h = np.where(d <= width_deg, float(floor_mm), h)
        return h
    return height


def test_characterize_assembles_one_ring_from_arcs_the_height_floor_broke():
    pytest.importorskip("open3d")
    # One physical ring, thinned to nothing at 125 deg and 245 deg: the ROI floor
    # erases those arcs, so DBSCAN yields two clusters neither of which spans
    # 70% of the circumference on its own.
    center = (CENTER[0] + 3.0, CENTER[1] - 1.0)
    T = syn.inspection_camera_T([CENTER[0], CENTER[1], 6.0], 300.0)
    ring = syn.RingSpec(40.0, 10.0, center,
                        height_fn=_thin_at(7.0, (125.0, 245.0)))
    depth = syn.render_scene([ring], T, plane_center_xy_mm=CENTER)
    color = np.zeros((720, 1280, 3), np.uint8)

    found = characterize_ring(
        color=color, depth=depth, geometry=gf.aligned(syn.K_720P, syn.SIZE_720P),
        T_work_camera=T, K=syn.K_720P, dist=None,
        search_center_mm=CENTER, work_frame="Tasni Work Frame",
        config=ExtrusionConfig())

    selector = found.report["ring_selector"]
    selected = next(c for c in selector["candidates"] if c.get("selected"))
    # The winner must be the ASSEMBLED ring, not either arc on its own.
    assert selected["cluster_count"] >= 2
    # 2 x 24 deg of the ring is genuinely erased, so ~0.87 is the honest ceiling;
    # what matters is that it clears the 0.70 gate no single arc could.
    assert selected["angular_coverage"] >= 0.85
    assert max(c["angular_coverage"] for c in selector["candidates"]
               if c["cluster_count"] == 1) < 0.70
    assert found.radius_mm == pytest.approx(40.0, abs=1.5)
    assert found.center_mm == pytest.approx(center, abs=1.5)


def test_characterize_real_low_relief_capture_is_not_rejected():
    pytest.importorskip("open3d")
    # trial 20260829-151445-acb42814/characterize-01: a 2-11 mm dried ring whose
    # thin arcs fell under the floor. Every cluster failed the old per-cluster
    # gate (best 48/72 bins = 0.667) though together they cover 71/72.
    fixture = np.load(Path(__file__).parent / "fixtures" / "extrusion" / "ring1"
                      / "ring1_low_relief_20260829.npz")
    depth = fixture["depth"]
    color = np.zeros((*depth.shape, 3), np.uint8)

    found = characterize_ring(
        color=color, depth=depth,
        geometry=gf.aligned(fixture["K"], (depth.shape[1], depth.shape[0])),
        T_work_camera=fixture["T_work_camera"], K=fixture["K"], dist=None,
        search_center_mm=fixture["search_center_mm"],
        work_frame="Tasni Work Frame", config=ExtrusionConfig())

    assert found.radius_mm == pytest.approx(42.0, abs=2.0)
    selected = next(c for c in found.report["ring_selector"]["candidates"]
                    if c.get("selected"))
    assert selected["cluster_count"] == 2
    assert selected["angular_coverage"] >= 0.90


def test_a_ring_measured_only_in_part_is_not_closed_into_a_full_one():
    pytest.importorskip("open3d")
    # Same broken ring, but through the layer pipeline: the centreline must cover
    # only what was actually measured, and the report must say so.
    center = (CENTER[0], CENTER[1])
    plan = scene_plan(radius=40.0, bead=10.0, layer_height=7.0)
    T = syn.inspection_camera_T(aim_point_mm(plan.recipe, plan.setup, 1), 300.0)
    ring = syn.RingSpec(40.0, 10.0, center, height_fn=_thin_at(7.0, (200.0,), width_deg=35.0))
    depth = syn.render_scene([ring], T, plane_center_xy_mm=CENTER)
    color = np.zeros((720, 1280, 3), np.uint8)

    result = process_observation(color=color, depth=depth,
                                 geometry=gf.aligned(syn.K_720P, syn.SIZE_720P),
                                 T_work_camera=T, K=syn.K_720P, dist=None,
                                 plan=plan, layer=plan.layers[0],
                                 config=ExtrusionConfig())

    measured = np.asarray(result.measured_xyz, dtype=float)
    fitted, _ = fit_circle_xy(measured)
    theta = np.mod(np.arctan2(measured[:, 1] - fitted[1],
                              measured[:, 0] - fitted[0]), 2 * np.pi)
    occupied = len(set((theta / (2 * np.pi) * 72).astype(int).tolist()))
    assert occupied < 68                       # the gap must survive into the output
    # compare_circle already guards completeness; forcing the spline closed made
    # it blind, because a periodic curve is always 100% complete.
    assert result.report["closed"] is False
    assert result.metrics.path_completeness < 0.95
    assert result.metrics.maximum_angular_gap_deg > 30.0
    assert result.metrics.valid is False
    assert any("completeness" in w or "angular gap" in w for w in result.metrics.warnings)

# ------------------------------------------------------------- archive (Task 7)


from tasni.modules.extrusion.archive import ExtrusionArchive
from tasni.modules.extrusion.models import LayerManifest


def test_archive_keeps_every_take_of_a_layer_and_records_the_mode(tmp_path):
    plan = scene_plan()
    archive = ExtrusionArchive(tmp_path)
    trial = archive.create_trial("t1", plan, mode="MEASURE_ONLY",
                                 experiment={"note": "dried rings, hand-placed"})
    data = json.loads((trial / "trial.json").read_text())
    assert data["mode"] == "MEASURE_ONLY" and data["experiment"]["note"].startswith("dried")
    nominal = np.zeros((4, 3))
    for take in (1, 2):
        manifest = LayerManifest(trial_id="t1", layer_index=2, take=take, mode="MEASURE_ONLY",
                                 recipe=plan.recipe, toolpath_fingerprint=plan.fingerprint,
                                 annotation={"introduced_offset_mm": [10, 0]})
        archive.write_layer(manifest, nominal_xyz=nominal, commanded_xyz=nominal)
    assert (tmp_path / "t1" / "layer-002" / "manifest.json").is_file()
    assert (tmp_path / "t1" / "layer-002-take02" / "manifest.json").is_file()
    assert archive.layer_dir("t1", 2, take=2).name == "layer-002-take02"
    loaded = json.loads((tmp_path / "t1" / "layer-002-take02" / "manifest.json").read_text())
    assert loaded["take"] == 2 and loaded["annotation"]["introduced_offset_mm"] == [10, 0]


def test_archive_writes_a_characterization_directory(tmp_path):
    plan = scene_plan()
    archive = ExtrusionArchive(tmp_path)
    archive.create_trial("t1", plan, mode="MEASURE_ONLY")
    out = archive.write_characterization(
        "t1", 1, color=np.zeros((4, 4, 3), np.uint8), depth=np.zeros((4, 4), np.uint16),
        measured_xyz=np.zeros((5, 3)),
        derived_images={"comparison.png": np.zeros((4, 4, 3), np.uint8)},
        report={"radius_mm": 60.0})
    assert out.name == "characterize-01"
    for name in ("color.png", "depth.npy", "measured_path.json", "comparison.png", "report.json"):
        assert (out / name).is_file(), name


# ------------------------------------------------- MEASURE_ONLY job (Task 8)

from test_extrusion_job import (Ctx, FakeCamera, FakeRdk, START_JOINTS,  # noqa: F401
                                SIDE_APPROACH_JOINTS, SIDE_JOINTS,
                                services)
from tasni.modules.extrusion import measure as measure_mod
from tasni.modules.extrusion.measure import MeasureSession, RingMeasureJob
from tasni.modules.extrusion.models import DeviationMetrics, RingGeometry
from tasni.modules.extrusion.processing import ProcessingResult


def fake_measure_processing(**kwargs):
    layer = kwargs["layer"]
    pts = np.array([[p.x_mm, p.y_mm, p.z_mm + 6.0] for p in layer.points])
    metrics = DeviationMetrics(mean_absolute_mm=6.4, rms_mm=7.1, maximum_mm=10.0,
                               measured_center_mm=(10.0, 0.0), measured_radius_mm=40.0,
                               path_completeness=0.99, maximum_angular_gap_deg=5, valid=True,
                               center_offset_mm=(10.0, 0.0), center_offset_norm_mm=10.0,
                               shape_rms_mm=0.3, shape_max_mm=0.8)
    geometry = RingGeometry(top_z_mean_mm=6, top_z_min_mm=5, top_z_max_mm=7, top_z_std_mm=0.5,
                            height_mean_mm=6, height_min_mm=5, height_max_mm=7,
                            height_reference="build_plane", bead_width_mean_mm=8,
                            bead_width_min_mm=7, bead_width_max_mm=9, bead_width_bins=36)
    image = np.zeros((12, 12), np.uint8)
    fake_measure_processing.calls.append(kwargs)
    return ProcessingResult(pts, None, metrics, image, image, np.zeros((12, 12, 3), np.uint8),
                            {"counts": {"raw_depth_pixels": 256}, "timings_ms": {"total_ms": 10.0},
                             "branch_guard_attempts": [{"attempt": 1}]},
                            filtered_xyz=pts.copy(), geometry=geometry)


fake_measure_processing.calls = []


def measure_env(tmp_path, monkeypatch, *, hardware_approved=False, side_photo=False):
    svc, rdk, camera = services(tmp_path)
    svc.config.extrusion.hardware_io_test_approved = hardware_approved
    # The side photo is ON in the cell (it is part of the protocol) but OFF for
    # the tests that count grabs and moves: it would add one of each to every
    # assertion about the MEASUREMENT path, which is not what they are about.
    # The tests that own it turn it back on.
    svc.config.extrusion.side_capture_enabled = side_photo
    monkeypatch.setattr(measure_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(measure_mod, "process_observation", fake_measure_processing)
    monkeypatch.setattr(measure_mod, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(measure_mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(measure_mod.runs, "read_active", lambda module: {"run_id": "cal-1"})
    fake_measure_processing.calls.clear()
    return svc, rdk, camera


def auto_plan(layers=3):
    recipe = CylinderRecipe(radius_mm=40, layer_count=layers, layer_height_mm=6,
                            bead_diameter_mm=8, robot_speed_mm_s=75, extrusion_rate_pct=0,
                            points_per_circle=24)
    setup = CylinderSetup(print_tool="LongCalibTool", work_frame="Tasni Work Frame",
                          inspection_tool="Realsense", inspection_auto=True,
                          center_x_mm=200, center_y_mm=150)
    return generate_cylinder_plan(recipe, setup)


def test_measure_moves_only_the_camera_and_never_touches_the_valve(tmp_path, monkeypatch):
    svc, rdk, camera = measure_env(tmp_path, monkeypatch, hardware_approved=False)
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan, note="rings")
    out = RingMeasureJob(svc, plan, session, 1, annotation={"introduced_offset_mm": None},
                         check_collisions=True)(Ctx())
    kinds = [e[0] for e in rdk.events]
    assert "station-program" not in kinds and "create" not in kinds     # no valve, no layer program
    assert "create-target" in kinds and "create-inspection" in kinds
    assert ("start", "TasniCylinder_MEASURE_%s_L001_Inspect" % plan.fingerprint[:10], True) in rdk.events
    assert rdk.events[-1] == ("move-joints", START_JOINTS)
    assert any(name.endswith("_Inspect") for name in rdk.deleted)
    assert camera.grabs == 2                                             # readiness + one measurement
    assert out["kind"] == "ring_measure" and out["mode"] == "MEASURE_ONLY"
    layer_dir = Path(out["layer_dir"])
    assert layer_dir.name == "layer-001"
    manifest = json.loads((layer_dir / "manifest.json").read_text())
    assert manifest["mode"] == "MEASURE_ONLY" and manifest["take"] == 1
    assert manifest["geometry"]["bead_width_mean_mm"] == 8
    timings = manifest["processing"]["timings_ms"]
    assert timings["capture_ms"] >= 0
    assert timings["acquisition_to_path_ms"] == pytest.approx(timings["capture_ms"] + 10.0)
    assert (layer_dir / "depth.npy").is_file() and (layer_dir / "color.png").is_file()


def test_repeat_takes_and_the_floor_from_the_previous_layer(tmp_path, monkeypatch):
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    plan = auto_plan()
    root = tmp_path / "runs" / "extrusion"
    session = MeasureSession.create(root, plan)
    RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True)(Ctx())
    second = RingMeasureJob(svc, plan, session, 1, annotation={"note": "re-placed"},
                            check_collisions=True)(Ctx())
    assert Path(second["layer_dir"]).name == "layer-001-take02"
    assert fake_measure_processing.calls[-1].get("floor_profile") is None   # layer 1: build plane
    third = RingMeasureJob(svc, plan, session, 2, annotation={"introduced_offset_mm": [10, 0]},
                           check_collisions=True)(Ctx())
    floor = fake_measure_processing.calls[-1]["floor_profile"]
    assert floor is not None and np.asarray(floor).shape[1] == 3          # layer 2: ring 1's top
    assert json.loads((Path(third["layer_dir"]) / "manifest.json").read_text())["annotation"] == {"introduced_offset_mm": [10, 0]}
    # Session survives a restart.
    reloaded = MeasureSession.load(root, session.trial_id)
    assert reloaded.takes == {1: 2, 2: 1}
    assert MeasureSession.latest(root).trial_id == session.trial_id
    assert reloaded.last_pose is not None


def test_measure_archives_the_raw_frame_when_processing_fails(tmp_path, monkeypatch):
    svc, rdk, _ = measure_env(tmp_path, monkeypatch)
    monkeypatch.setattr(measure_mod, "process_observation",
                        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad skeleton")))
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    with pytest.raises(RuntimeError, match="raw RGB-D archived"):
        RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True)(Ctx())
    layer = session.trial_dir / "layer-001"
    assert (layer / "depth.npy").is_file() and "bad skeleton" in (layer / "report.json").read_text()
    assert rdk.events[-1] == ("move-joints", START_JOINTS)                     # still returns home


def test_measure_blocks_before_motion_when_the_camera_is_offline(tmp_path, monkeypatch):
    from tasni.core.camera import CameraError
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    def offline(**kwargs): raise CameraError("camera timeout (100.123.63.127:1024)")
    camera.grab = offline
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    with pytest.raises(RuntimeError, match="camera is not ready"):
        RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True)(Ctx())
    assert rdk.events == []


# ---------------------------------------------- characterize job (Task 9)

from tasni.modules.extrusion.measure import RingCharacterizeJob
from tasni.modules.extrusion.processing import CharacterizationResult


def fake_characterize(**kwargs):
    fake_characterize.calls.append(kwargs)
    image = np.zeros((12, 12), np.uint8)
    return CharacterizationResult(
        radius_mm=61.2, center_mm=(214.0, 141.0), bead_width_mm=8.3, bead_width_min_mm=7.0,
        bead_width_max_mm=9.5, top_z_mean_mm=6.4, top_z_min_mm=5.1, top_z_max_mm=9.8,
        measured_xyz=np.zeros((10, 3)), segmentation=image, skeleton=image,
        comparison=np.zeros((12, 12, 3), np.uint8),
        report={"coarse": {"radius_mm": 60.0}, "timings_ms": {"total_ms": 12.0}})


fake_characterize.calls = []


def test_characterize_job_measures_the_ring_and_stores_it_in_the_session(tmp_path, monkeypatch):
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    monkeypatch.setattr(measure_mod, "characterize_ring", fake_characterize)
    fake_characterize.calls.clear()
    plan = auto_plan()
    root = tmp_path / "runs" / "extrusion"
    session = MeasureSession.create(root, plan)
    out = RingCharacterizeJob(svc, plan, session, check_collisions=True)(Ctx())
    assert out["kind"] == "ring_characterize"
    assert out["characterization"]["radius_mm"] == 61.2
    assert fake_characterize.calls[-1]["search_center_mm"] == (200.0, 150.0)
    assert fake_characterize.calls[-1]["work_frame"] == "Tasni Work Frame"
    assert Path(out["capture_dir"]).name == "characterize-01"
    assert (Path(out["capture_dir"]) / "depth.npy").is_file()
    assert out["characterization"]["inspection_pose"]["standoff_mm"] == 300.0
    kinds = [e[0] for e in rdk.events]
    assert "station-program" not in kinds and "create" not in kinds
    assert rdk.events[-1] == ("move-joints", START_JOINTS)
    assert MeasureSession.load(root, session.trial_id).characterizations[-1]["radius_mm"] == 61.2


def test_measure_only_close_range_requires_explicit_job_option(tmp_path, monkeypatch):
    svc, _, _ = measure_env(tmp_path, monkeypatch)
    monkeypatch.setattr(measure_mod, "characterize_ring", fake_characterize)
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)

    # The shipped default sits at the 1280x720 MinZ-bounded 300 mm floor; use a
    # distinct value here so the test proves the clamp is wired, not a coincidence.
    svc.config.extrusion.measure_close_range_min_mm = 200.0
    out = RingCharacterizeJob(
        svc, plan, session, check_collisions=True,
        close_range_tool_clear=True)(Ctx())

    pose = out["characterization"]["inspection_pose"]
    assert pose["near_mm"] == 200.0
    assert pose["standoff_mm"] == 200.0
    assert pose["d_fit_mm"] < pose["standoff_mm"]
    assert pose["fill_fraction"]["height"] > 0.5


def test_characterize_archives_raw_frame_when_ring_processing_fails(tmp_path, monkeypatch):
    svc, rdk, _ = measure_env(tmp_path, monkeypatch)
    plan = auto_plan()
    root = tmp_path / "runs" / "extrusion"
    session = MeasureSession.create(root, plan)
    # A prior successful characterization occupies index 1; a failed capture
    # must advance to index 2 rather than overwrite it or disappear.
    monkeypatch.setattr(measure_mod, "characterize_ring", fake_characterize)
    RingCharacterizeJob(svc, plan, session, check_collisions=True)(Ctx())
    monkeypatch.setattr(
        measure_mod, "characterize_ring",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("no ring-like cluster")))

    with pytest.raises(RuntimeError, match="raw RGB-D archived"):
        RingCharacterizeJob(svc, plan, session, check_collisions=True)(Ctx())

    failed = session.trial_dir / "characterize-02"
    assert (failed / "color.png").is_file()
    assert (failed / "depth.npy").is_file()
    report = json.loads((failed / "report.json").read_text())
    assert not report["valid"] and "no ring-like cluster" in report["error"]
    assert report["depth_shape"] == [16, 16]
    assert rdk.events[-1] == ("move-joints", START_JOINTS)


# ------------------------------------------------------------- API (Task 10)

from fastapi.testclient import TestClient
from tasni.core.config import AppConfig
from tasni.modules.extrusion import module as extrusion_module
from tasni.webapp.server import create_app


def api_plan(client):
    payload = {"recipe": auto_plan().recipe.model_dump(mode="json"),
               "setup": auto_plan().setup.model_dump(mode="json")}
    return client.post("/api/modules/extrusion/generate", json=payload).json()


def test_measure_layer_is_gated_on_fingerprint_confirm_and_connection_only(tmp_path, monkeypatch):
    monkeypatch.setattr(extrusion_module, "REPO_ROOT", tmp_path)
    cfg = AppConfig()
    cfg.extrusion.hardware_io_test_approved = False          # irrelevant to measuring
    client = TestClient(create_app(cfg))
    plan = api_plan(client)
    body = {"fingerprint": "stale", "layer_index": 1, "annotation": {},
            "confirm_robot_motion": True}
    assert client.post("/api/modules/extrusion/measure/layer", json=body).status_code == 409
    body["fingerprint"] = plan["fingerprint"]
    body["confirm_robot_motion"] = False
    assert client.post("/api/modules/extrusion/measure/layer", json=body).status_code == 400
    body["confirm_robot_motion"] = True
    body["layer_index"] = 99
    assert client.post("/api/modules/extrusion/measure/layer", json=body).status_code == 400
    body["layer_index"] = 1
    refused = client.post("/api/modules/extrusion/measure/layer", json=body)
    assert refused.status_code == 409 and "RoboDK" in refused.json()["detail"]
    assert "hardware" not in refused.json()["detail"].lower()


def test_measure_session_is_created_listed_and_excluded_from_print_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(extrusion_module, "REPO_ROOT", tmp_path)
    client = TestClient(create_app(AppConfig()))
    assert client.get("/api/modules/extrusion/measure/session").json()["session"] is None
    assert client.post("/api/modules/extrusion/measure/session/new",
                       json={"note": "x"}).status_code == 409      # needs a generated plan
    api_plan(client)
    created = client.post("/api/modules/extrusion/measure/session/new", json={"note": "rings"}).json()
    trial_id = created["session"]["trial_id"]
    assert (tmp_path / "runs" / "extrusion" / trial_id / "session.json").is_file()
    assert client.get("/api/modules/extrusion/measure/session").json()["session"]["trial_id"] == trial_id
    assert client.get("/api/modules/extrusion/status").json()["measure_session"] == trial_id
    # A LIVE_PRINT trial beside it: only that one is a printed trial.
    live = ExtrusionArchive(tmp_path / "runs" / "extrusion")
    live.create_trial("20990101-000000-live0000", auto_plan())
    trials = client.get("/api/modules/extrusion/trials").json()
    assert trials["summary"]["total_trials"] == 1
    assert trials["summary"]["measure_only_trials"] == 1
    assert {t["trial_id"]: t["mode"] for t in trials["trials"]} == {
        trial_id: "MEASURE_ONLY", "20990101-000000-live0000": "LIVE_PRINT"}


def test_apply_characterization_rewrites_recipe_and_placement(tmp_path, monkeypatch):
    monkeypatch.setattr(extrusion_module, "REPO_ROOT", tmp_path)
    client = TestClient(create_app(AppConfig()))
    before = api_plan(client)
    assert client.post("/api/modules/extrusion/measure/apply-characterization").status_code == 409
    client.post("/api/modules/extrusion/measure/session/new", json={"note": ""})
    session = MeasureSession.latest(tmp_path / "runs" / "extrusion")
    session.characterizations.append({"index": 1, "radius_mm": 61.24, "center_mm": [214.0, 141.0],
                                      "bead_width_mm": 8.31, "top_z_mean_mm": 6.44,
                                      "top_z_min_mm": 5.1, "top_z_max_mm": 9.8})
    session.save()
    after = client.post("/api/modules/extrusion/measure/apply-characterization").json()
    assert after["fingerprint"] != before["fingerprint"]
    assert after["recipe"]["radius_mm"] == 61.2 and after["recipe"]["bead_diameter_mm"] == 8.3
    assert after["recipe"]["layer_height_mm"] == 6.4
    assert after["setup"]["center_x_mm"] == 214.0 and after["setup"]["center_y_mm"] == 141.0
    assert after["setup"]["build_plane_z_mm"] == 0.0
    assert client.get("/api/modules/extrusion/plan").json()["fingerprint"] == after["fingerprint"]


# ------------------------------- offline reprocessing measures against the TAKE's plan

def test_reprocessing_uses_the_recipe_and_centre_the_take_was_measured_against(tmp_path):
    """A measure-only session is created BEFORE Characterize -> Apply, so trial.json
    carries the pre-Apply plan (cell 2026-08-28: r 40 at (212.1, 149.7) while the
    take was measured at r 42.6 about (214.6, 146.7)). Reprocessing from trial.json
    would score the ring against a plan it was never measured against -- the same
    stale-plan artifact the handoff says must never reach the paper.
    """
    pytest.importorskip("open3d")
    from tasni.modules.extrusion.service import reprocess_saved_layer

    root = tmp_path / "runs" / "extrusion"
    stale = scene_plan(radius=40.0, bead=15.0, layer_height=5.0, center=(212.1, 149.7))
    applied = scene_plan(radius=60.0, bead=8.0, layer_height=6.0, center=CENTER)
    archive = ExtrusionArchive(root)
    # As MeasureSession.create writes it: the trial carries NO processing
    # provenance -- the measure job puts intrinsics and the processing config on
    # each take's manifest, because they are per-take facts.
    archive.create_trial("t-stale", stale, mode="MEASURE_ONLY")
    layer = applied.layers[0]
    T = syn.inspection_camera_T(aim_point_mm(applied.recipe, applied.setup, 1), 300.0)
    depth = syn.render_scene([syn.RingSpec(60.0, 8.0, CENTER, height_fn=syn.flat(6.0))], T,
                             plane_center_xy_mm=CENTER, seed=3)
    color = np.zeros((*depth.shape, 3), np.uint8)
    manifest = LayerManifest(
        trial_id="t-stale", layer_index=1, take=1, mode="MEASURE_ONLY",
        recipe=applied.recipe, toolpath_fingerprint=applied.fingerprint,
        color_file="color.png", depth_file="depth.npy",
        processing={"valid": False, "error": "branch guard exhausted"},
        provenance={"T_work_camera": np.asarray(T, dtype=float).tolist(),
                    "camera_intrinsics": {"K": syn.K_720P.tolist()},
                    "processing_config": ExtrusionConfig().model_dump(mode="json")})
    nominal = points_array(layer)
    archive.write_layer(manifest, nominal_xyz=nominal, commanded_xyz=nominal,
                        color=color, depth=depth, report={"valid": False})

    result = reprocess_saved_layer(root, "t-stale", 1)

    metrics = result["metrics"]
    assert metrics["valid"], metrics["warnings"]
    assert metrics["measured_radius_mm"] == pytest.approx(60.0, abs=1.0)
    assert metrics["center_offset_norm_mm"] < 1.0, (
        "scored against the stale trial.json plan instead of the take's own nominal")
    rewritten = json.loads((root / "t-stale" / "layer-001" / "manifest.json").read_text(
        encoding="utf-8"))
    assert rewritten["metrics"]["valid"] is True
    assert rewritten["processing"]["offline_reprocess"] is True
    assert rewritten["recipe"]["radius_mm"] == 60.0, "the take's recipe must survive"
    # paper_summary reads height/bead from manifest.geometry: a reprocessed take
    # that leaves it empty silently drops out of those statistics.
    assert rewritten["geometry"] is not None
    assert rewritten["geometry"]["height_mean_mm"] == pytest.approx(6.0, abs=1.5)
    assert rewritten["geometry"]["bead_width_mean_mm"] > 0


# --------------------------------------------------- paper summary (Task 11)

from tasni.modules.extrusion.measure import paper_summary


def _write_take(root, trial_id, layer_index, take, *, offset, rms, mean_abs, maximum,
                acq_ms, valid=True, offset_norm=None, measured_offset=None,
                phase=None, offline=False, cycle_ms=None):
    if measured_offset is None:
        measured_offset = (float(offset_norm or 0.0), 0.0)
    if offset_norm is None:
        offset_norm = float(np.hypot(*measured_offset))
    # As the chain reports it: the fitted centre IS the plan centre plus the offset.
    setup = auto_plan().setup
    measured_center = (setup.center_x_mm + float(measured_offset[0]),
                       setup.center_y_mm + float(measured_offset[1]))
    annotation = {"introduced_offset_mm": offset}
    if phase is not None:
        annotation["phase"] = phase
    manifest = LayerManifest(
        trial_id=trial_id, layer_index=layer_index, take=take, mode="MEASURE_ONLY",
        recipe=auto_plan().recipe, toolpath_fingerprint="f" * 64,
        annotation=annotation,
        metrics=DeviationMetrics(mean_absolute_mm=mean_abs, rms_mm=rms, maximum_mm=maximum,
                                 measured_center_mm=measured_center, measured_radius_mm=40,
                                 path_completeness=0.99, maximum_angular_gap_deg=4, valid=valid,
                                 center_offset_mm=tuple(float(v) for v in measured_offset),
                                 center_offset_norm_mm=offset_norm, shape_rms_mm=0.4),
        geometry=RingGeometry(top_z_mean_mm=6, top_z_min_mm=5, top_z_max_mm=9, top_z_std_mm=1,
                              height_mean_mm=6, height_min_mm=5, height_max_mm=9,
                              height_reference="build_plane", bead_width_mean_mm=8,
                              bead_width_min_mm=7, bead_width_max_mm=9, bead_width_bins=36),
        processing={"offline_reprocess": offline,
                    "timings_ms": ({"capture_ms": 40.0, "total_ms": acq_ms - 40.0}
                                   if offline else
                                   {"capture_ms": 40.0, "total_ms": acq_ms - 40.0,
                                    "acquisition_to_path_ms": acq_ms,
                                    **({} if cycle_ms is None
                                       else {"inspection_cycle_ms": float(cycle_ms)})})})
    ExtrusionArchive(root).write_layer(manifest, nominal_xyz=np.zeros((4, 3)),
                                       commanded_xyz=np.zeros((4, 3)))


def test_paper_summary_groups_by_introduced_offset_and_reports_timing(tmp_path):
    root = tmp_path / "runs" / "extrusion"
    session = MeasureSession.create(root, auto_plan(), note="rings")
    t = session.trial_id
    _write_take(root, t, 1, 1, offset=None, offset_norm=0.4, rms=0.5, mean_abs=0.4, maximum=1.1, acq_ms=900)
    _write_take(root, t, 1, 2, offset=[0, 0], offset_norm=0.6, rms=0.6, mean_abs=0.5, maximum=1.3, acq_ms=1100)
    _write_take(root, t, 2, 1, offset=[10, 0], offset_norm=9.8, rms=7.0, mean_abs=6.3, maximum=9.9, acq_ms=1000)
    summary = paper_summary(root, t)
    assert summary["mode"] == "MEASURE_ONLY" and summary["takes"] == 3 and summary["valid"] == 3
    by_name = {c["condition"]: c for c in summary["conditions"]}
    # A condition is the layer AND the ground truth: see
    # test_conditions_separate_the_layer_and_the_phase_the_operator_recorded.
    assert by_name["layer 1 - no introduced offset"]["takes"] == 2
    shifted = by_name["layer 2 - introduced offset (10, 0) mm"]
    assert shifted["takes"] == 1 and shifted["center_offset_norm_mm"]["mean"] == 9.8
    assert summary["timing_ms"]["acquisition_to_path_ms"]["mean"] == pytest.approx(1000.0)
    assert summary["timing_ms"]["acquisition_to_path_ms"]["sd"] == pytest.approx(100.0)
    assert summary["height_mm"]["height_max_mm"]["mean"] == 9.0
    assert "10" in summary["markdown"] and "hand-placed" in summary["markdown"]


def test_paper_summary_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(extrusion_module, "REPO_ROOT", tmp_path)
    client = TestClient(create_app(AppConfig()))
    api_plan(client)
    trial_id = client.post("/api/modules/extrusion/measure/session/new", json={"note": ""}).json()["session"]["trial_id"]
    _write_take(tmp_path / "runs" / "extrusion", trial_id, 1, 1, offset=None, offset_norm=0.5,
                rms=0.5, mean_abs=0.4, maximum=1.0, acq_ms=950)
    got = client.get(f"/api/modules/extrusion/trials/{trial_id}/paper-summary").json()
    assert got["takes"] == 1 and "markdown" in got
    assert client.get("/api/modules/extrusion/trials/nope/paper-summary").status_code == 404


# ---------------------------------- detection error vs the operator's ground truth

def test_paper_summary_scores_the_measurement_against_the_offset_the_operator_typed(tmp_path):
    """The paper's claim is "a 10 mm shift READ AS 10.x mm", not "offset 10.x mm".

    Without the typed offset subtracted, the table only says where the ring was,
    never how well the chain found it -- which is the whole controlled-validation
    claim.
    """
    root = tmp_path / "runs" / "extrusion"
    t = MeasureSession.create(root, auto_plan(), note="rings").trial_id
    _write_take(root, t, 2, 1, offset=[10, 0], measured_offset=[10.4, 0.3],
                rms=7.0, mean_abs=6.3, maximum=9.9, acq_ms=1000)
    _write_take(root, t, 2, 2, offset=[10, 0], measured_offset=[9.6, -0.3],
                rms=7.1, mean_abs=6.4, maximum=10.1, acq_ms=1000)

    shifted = {c["condition"]: c for c in paper_summary(root, t)["conditions"]}[
        "layer 2 - introduced offset (10, 0) mm"]

    assert shifted["introduced_norm_mm"] == pytest.approx(10.0)
    # |(10.4, 0.3) - (10, 0)| and |(9.6, -0.3) - (10, 0)| are both exactly 0.5 mm
    assert shifted["detection_error_mm"]["mean"] == pytest.approx(0.5, abs=1e-9)
    assert shifted["detection_error_mm"]["max"] == pytest.approx(0.5, abs=1e-9)
    assert shifted["detection_error_mm"]["n"] == 2


def test_a_take_with_no_introduced_offset_is_scored_against_zero(tmp_path):
    """The zero-offset group IS the baseline: how far off it reads when nothing moved."""
    root = tmp_path / "runs" / "extrusion"
    t = MeasureSession.create(root, auto_plan(), note="rings").trial_id
    _write_take(root, t, 1, 1, offset=None, measured_offset=[0.3, 0.4],
                rms=0.5, mean_abs=0.4, maximum=1.1, acq_ms=900)

    baseline = paper_summary(root, t)["conditions"][0]

    assert baseline["introduced_norm_mm"] == 0.0
    assert baseline["detection_error_mm"]["mean"] == pytest.approx(0.5, abs=1e-9)
    assert baseline["shift_consistency"] is None, "no shift to check the relation against"


def test_deviations_that_look_like_the_introduced_shift_pass_the_built_in_relation(tmp_path):
    """A pure translation d must read max = d, mean = 2d/pi, RMS = d/sqrt(2)."""
    root = tmp_path / "runs" / "extrusion"
    t = MeasureSession.create(root, auto_plan(), note="rings").trial_id
    _write_take(root, t, 2, 1, offset=[10, 0], measured_offset=[10.0, 0.0],
                rms=7.07, mean_abs=6.37, maximum=10.0, acq_ms=1000)

    check = paper_summary(root, t)["conditions"][0]["shift_consistency"]

    assert check["consistent"] is True
    assert check["expected_mm"]["mean_absolute_mm"] == pytest.approx(6.366, abs=1e-3)
    assert check["expected_mm"]["rms_mm"] == pytest.approx(7.071, abs=1e-3)
    assert check["disagreements"] == []


def test_deviations_that_do_not_look_like_a_pure_shift_are_flagged_not_averaged(tmp_path):
    """The sanity check the handoff calls "stop and investigate" must be machine-checked."""
    root = tmp_path / "runs" / "extrusion"
    t = MeasureSession.create(root, auto_plan(), note="rings").trial_id
    _write_take(root, t, 2, 1, offset=[10, 0], measured_offset=[3.0, 0.0],
                rms=2.4, mean_abs=2.0, maximum=4.0, acq_ms=1000)

    summary = paper_summary(root, t)
    check = summary["conditions"][0]["shift_consistency"]

    assert check["consistent"] is False
    assert set(check["disagreements"]) == {"mean_absolute_mm", "rms_mm", "maximum_mm"}
    assert "does not match a pure" in summary["markdown"]


def test_the_markdown_states_the_recovered_shift_for_every_offset_condition(tmp_path):
    root = tmp_path / "runs" / "extrusion"
    t = MeasureSession.create(root, auto_plan(), note="rings").trial_id
    _write_take(root, t, 2, 1, offset=[10, 0], measured_offset=[10.2, 0.0],
                rms=7.1, mean_abs=6.4, maximum=10.2, acq_ms=1000)
    _write_take(root, t, 2, 2, offset=[10, 0], measured_offset=[9.8, 0.0],
                rms=7.0, mean_abs=6.3, maximum=9.8, acq_ms=1000)

    markdown = paper_summary(root, t)["markdown"]

    assert "10.0 mm introduced offset was recovered as 10.00 +/- 0.28 mm" in markdown
    assert "detection error" in markdown


def test_a_take_archived_before_the_offset_vector_existed_is_left_out_of_the_score(tmp_path):
    """Missing is not zero: an old manifest must not read as a perfect measurement."""
    root = tmp_path / "runs" / "extrusion"
    t = MeasureSession.create(root, auto_plan(), note="rings").trial_id
    _write_take(root, t, 2, 1, offset=[10, 0], measured_offset=[10.4, 0.3],
                rms=7.0, mean_abs=6.3, maximum=9.9, acq_ms=1000)
    stale = root / t / "layer-002" / "manifest.json"
    payload = json.loads(stale.read_text(encoding="utf-8"))
    payload["metrics"].pop("center_offset_mm")
    stale.write_text(json.dumps(payload), encoding="utf-8")

    condition = paper_summary(root, t)["conditions"][0]

    assert condition["detection_error_mm"]["n"] == 0
    assert condition["detection_error_mm"]["mean"] is None


def test_empty_roi_error_reports_which_band_rejected_the_points():
    """A bare "not enough points" cannot be acted on in the cell. The message must
    say how many points each band admitted and what the observed spread was, so an
    operator can tell a Z-offset (wrong build plane) from a radial miss (wrong
    centre) without re-running the print."""
    pytest.importorskip("open3d")
    plan = scene_plan()
    # A real ring, but sitting 250 mm away from the configured centre: the radial
    # band rejects everything while the height band is perfectly happy.
    far = (CENTER[0] + 250.0, CENTER[1])
    with pytest.raises(RuntimeError) as excinfo:
        observe(plan, 1, [syn.RingSpec(60.0, 8.0, far, height_fn=syn.flat(6.0))])
    msg = str(excinfo.value)
    assert "not enough deposited-geometry points" in msg
    assert "height" in msg and "radial" in msg          # both bands named
    assert "in_height_band" in msg and "in_radial_band" in msg


def test_measure_only_requests_default_to_collisions_off():
    """The ring stack is not modelled in the station, so measure-only camera moves
    ship with RoboDK collision validation off; the print paths are unaffected."""
    from tasni.modules.extrusion.module import CharacterizeBody, MeasureLayerBody

    assert CharacterizeBody().collision_check_enabled is False
    assert MeasureLayerBody(fingerprint="f", layer_index=1).collision_check_enabled is False


def test_a_measured_take_leaves_its_figures_next_to_the_frame(tmp_path, monkeypatch):
    """The operator should not have to run a tool to see what was measured."""
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)

    out = RingMeasureJob(svc, plan, session, 1, check_collisions=False)(Ctx())

    figures = Path(out["layer_dir"]) / "figures"
    assert figures.is_dir(), "the take archived no figures"
    assert {p.name for p in figures.glob("*.png")} >= {"plan.png", "profile.png"}


def test_a_figure_that_cannot_be_drawn_never_fails_the_measurement(tmp_path, monkeypatch):
    """A drawing problem must not cost the operator a ring placement."""
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    monkeypatch.setattr(measure_mod, "render_layer_figures",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no matplotlib")))

    out = RingMeasureJob(svc, plan, session, 1, check_collisions=False)(Ctx())

    assert out["valid"] is True
    assert Path(out["layer_dir"], "manifest.json").is_file()


def archived_take(root, monkeypatch):
    """One real MEASURE_ONLY take on disk, with no figures rendered yet."""
    import test_extrusion_figures as figs
    monkeypatch.setattr(extrusion_module, "REPO_ROOT", root.parent.parent)
    return figs.write_take(root)


def test_layer_figures_are_served_and_rendered_on_first_request(tmp_path, monkeypatch):
    """Takes archived before figures existed must still produce them, robot-free."""
    layer_dir = archived_take(tmp_path / "runs" / "extrusion", monkeypatch)
    client = TestClient(create_app(AppConfig()))
    base = "/api/modules/extrusion/trials/t1/layers/layer-001"
    assert not (layer_dir / "figures").exists()

    found = client.get(f"{base}/figures/plan.png")

    assert found.status_code == 200
    assert found.headers["content-type"] == "image/png"
    assert found.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert (layer_dir / "figures" / "plan.png").is_file()
    assert client.get(f"{base}/figures/profile.pdf").status_code == 200


def test_archived_frames_are_served_from_the_allowlist_only(tmp_path, monkeypatch):
    archived_take(tmp_path / "runs" / "extrusion", monkeypatch)
    client = TestClient(create_app(AppConfig()))
    base = "/api/modules/extrusion/trials/t1/layers/layer-001"

    assert client.get(f"{base}/files/depth.npy").status_code == 404
    assert client.get(f"{base}/files/manifest.json").status_code == 404
    assert client.get(f"{base}/figures/plan.svg").status_code == 404


def test_a_layer_directory_outside_the_trial_is_refused(tmp_path, monkeypatch):
    """The directory name is a URL segment, so it must not be able to escape.

    A literal ``..`` is normalised away by the client and never reaches the
    handler; a percent-encoded one does, which is the case worth pinning.
    """
    archived_take(tmp_path / "runs" / "extrusion", monkeypatch)
    client = TestClient(create_app(AppConfig()))
    base = "/api/modules/extrusion/trials/t1/layers"

    for segment in ("%2e%2e", "%2e", "characterize-01", "figures"):
        found = client.get(f"{base}/{segment}/figures/plan.png")
        assert found.status_code == 404, f"{segment} was served"
    for trial in ("%2e%2e", "%2e"):
        found = client.get(f"/api/modules/extrusion/trials/{trial}"
                           f"/layers/layer-001/figures/plan.png")
        assert found.status_code == 404, f"trial {trial} was served"
    # A separator inside a segment is normalised by the router into some other
    # path entirely; whatever that lands on, it is never an archived file.
    for escaping in (f"{base}/../figures/plan.png",
                     f"{base}/layer-001%2f..%2f../figures/plan.png",
                     "/api/modules/extrusion/trials/t1%2f..%2f../layers/layer-001"
                     "/figures/plan.png"):
        found = client.get(escaping)
        assert "image" not in found.headers["content-type"], f"{escaping} served an image"
        assert "pdf" not in found.headers["content-type"], f"{escaping} served a pdf"


def test_trials_reports_which_takes_have_figures(tmp_path, monkeypatch):
    layer_dir = archived_take(tmp_path / "runs" / "extrusion", monkeypatch)
    client = TestClient(create_app(AppConfig()))

    layer = client.get("/api/modules/extrusion/trials").json()["trials"][0]["layers"][0]
    assert layer["layer_dir"] == "layer-001"
    assert layer["has_figures"] is False

    client.get("/api/modules/extrusion/trials/t1/layers/layer-001/figures/plan.png")
    layer = client.get("/api/modules/extrusion/trials").json()["trials"][0]["layers"][0]
    assert layer["has_figures"] is True


def test_the_trial_stack_figure_is_served_for_the_whole_session(tmp_path, monkeypatch):
    """The stack across layers is the picture that shows an introduced offset."""
    import test_extrusion_figures as figs
    root = tmp_path / "runs" / "extrusion"
    monkeypatch.setattr(extrusion_module, "REPO_ROOT", tmp_path)
    figs.write_take(root, layer_index=1)
    figs.write_take(root, layer_index=2, measured=figs._ring_xyz(z=12.0))
    client = TestClient(create_app(AppConfig()))

    found = client.get("/api/modules/extrusion/trials/t1/figures/stack.png")

    assert found.status_code == 200
    assert found.headers["content-type"] == "image/png"
    assert (root / "t1" / "figures" / "stack.png").is_file()
    assert client.get("/api/modules/extrusion/trials/t1/figures/plan.png").status_code == 404
    assert client.get("/api/modules/extrusion/trials/nope/figures/stack.png").status_code == 404


# ============================================================================
# The operator journey: a session that stays truthful across restarts,
# reprocessing, and the conditions the paper protocol deliberately creates.
# ============================================================================

def _archive_failed_take(root, session, plan, *, capture_ms=2874.0, layer_index=1, take=1):
    """A take that failed on the cell: raw RGB-D archived, nothing derived.

    This is exactly what the paper trial's layer-001 looked like the night the
    board-noise defect was found.
    """
    layer = plan.layers[layer_index - 1]
    T = syn.inspection_camera_T(aim_point_mm(plan.recipe, plan.setup, layer_index), 300.0)
    depth = syn.render_scene([syn.RingSpec(60.0, 8.0, CENTER, height_fn=syn.flat(6.0))], T,
                             plane_center_xy_mm=CENTER, seed=3)
    manifest = LayerManifest(
        trial_id=session.trial_id, layer_index=layer_index, take=take, mode="MEASURE_ONLY",
        recipe=plan.recipe, toolpath_fingerprint=plan.fingerprint,
        color_file="color.png", depth_file="depth.npy",
        annotation={"introduced_offset_mm": None, "note": "MAIN-TEST-1"},
        processing={"valid": False, "error": "branch guard exhausted",
                    "timings_ms": {"capture_ms": capture_ms}},
        provenance={"T_work_camera": np.asarray(T, dtype=float).tolist(),
                    "camera_intrinsics": {"K": syn.K_720P.tolist()},
                    "processing_config": ExtrusionConfig().model_dump(mode="json")})
    nominal = points_array(layer)
    ExtrusionArchive(root).write_layer(
        manifest, nominal_xyz=nominal, commanded_xyz=nominal,
        color=np.zeros((*depth.shape, 3), np.uint8), depth=depth, report={"valid": False})
    session.takes[layer_index] = take
    session.save()


def test_reprocessing_an_archived_take_puts_it_back_into_the_session(tmp_path):
    """A take rescued offline must re-enter the session, not only its manifest.

    session.json is what layer N+1's floor and the operator's table read. On the
    cell (2026-08-28) the paper trial's layer-001 was reprocessed to a valid
    measurement while session.json still said records: [] and tops: {} -- so
    layer 2 would have been measured against the build plane instead of ring 1's
    measured top, the floor the spec proves is load-bearing.
    """
    pytest.importorskip("open3d")
    from tasni.modules.extrusion.service import reprocess_saved_layer

    root = tmp_path / "runs" / "extrusion"
    plan = scene_plan(radius=60.0, bead=8.0, layer_height=6.0, center=CENTER)
    session = MeasureSession.create(root, plan, note="rings")
    _archive_failed_take(root, session, plan)

    reprocess_saved_layer(root, session.trial_id, 1)

    reloaded = MeasureSession.load(root, session.trial_id)
    assert reloaded.takes == {1: 1}
    assert [(r["layer_index"], r["take"]) for r in reloaded.records] == [(1, 1)]
    assert reloaded.records[0]["valid"] is True
    assert reloaded.records[0]["reprocessed"] is True
    assert reloaded.records[0]["layer_name"] == "layer-001"     # the UI addresses figures by it
    assert np.asarray(reloaded.tops[1]).shape[1] == 3           # layer 2 now has its floor


def test_a_reprocessed_take_keeps_the_capture_time_it_was_measured_with(tmp_path):
    """Capture time is a fact about the cell; offline processing time is not.

    Overwriting capture_ms with an offline number would corrupt the paper's
    acquisition-to-path statistic with a measurement that never happened.
    """
    pytest.importorskip("open3d")
    from tasni.modules.extrusion.service import reprocess_saved_layer

    root = tmp_path / "runs" / "extrusion"
    plan = scene_plan(radius=60.0, bead=8.0, layer_height=6.0, center=CENTER)
    session = MeasureSession.create(root, plan)
    _archive_failed_take(root, session, plan, capture_ms=2874.0)

    reprocess_saved_layer(root, session.trial_id, 1)

    manifest = json.loads((session.trial_dir / "layer-001" / "manifest.json").read_text())
    timings = manifest["processing"]["timings_ms"]
    assert timings["capture_ms"] == pytest.approx(2874.0)
    assert manifest["processing"]["offline_reprocess"] is True
    # No acquisition-to-path: this take never produced a path on the cell.
    assert "acquisition_to_path_ms" not in timings


def test_an_offline_reprocessed_take_is_kept_out_of_the_live_timing_statistic(tmp_path):
    """Requirement 3 is scan-to-feedback time ON THE CELL, over ~10 runs."""
    root = tmp_path / "runs" / "extrusion"
    session = MeasureSession.create(root, auto_plan(), note="rings")
    t = session.trial_id
    _write_take(root, t, 1, 1, offset=None, offset_norm=0.4, rms=0.5, mean_abs=0.4,
                maximum=1.1, acq_ms=1000)
    _write_take(root, t, 1, 2, offset=None, offset_norm=0.5, rms=0.5, mean_abs=0.4,
                maximum=1.2, acq_ms=99999, offline=True)

    summary = paper_summary(root, t)

    acq = summary["timing_ms"]["acquisition_to_path_ms"]
    assert acq["n"] == 1 and acq["mean"] == pytest.approx(1000.0)
    assert summary["timing_ms"]["offline_reprocessed_takes"] == 1
    assert "reprocessed offline" in summary["markdown"]
    # Both takes still count as measurements of the ring.
    assert summary["takes"] == 2


def test_conditions_separate_the_layer_and_the_phase_the_operator_recorded(tmp_path):
    """Five untouched takes and three re-placed takes are different experiments.

    Pooled into one "no introduced offset" row they hide sensing repeatability
    inside placement repeatability, and the per-layer evidence that the camera
    climbs with the stack disappears entirely.
    """
    root = tmp_path / "runs" / "extrusion"
    session = MeasureSession.create(root, auto_plan(), note="rings")
    t = session.trial_id
    for take in (1, 2):
        _write_take(root, t, 1, take, offset=None, offset_norm=0.4, rms=0.5, mean_abs=0.4,
                    maximum=1.1, acq_ms=1000, phase="noise floor")
    _write_take(root, t, 1, 3, offset=None, offset_norm=1.9, rms=1.6, mean_abs=1.4,
                maximum=2.6, acq_ms=1000, phase="re-placed")
    _write_take(root, t, 2, 1, offset=None, offset_norm=0.8, rms=0.7, mean_abs=0.6,
                maximum=1.4, acq_ms=1000, phase="stacked true")
    _write_take(root, t, 2, 2, offset=[10, 0], offset_norm=9.8, rms=7.0, mean_abs=6.3,
                maximum=9.9, acq_ms=1000, phase="top ring shifted")

    summary = paper_summary(root, t)

    by_name = {c["condition"]: c for c in summary["conditions"]}
    assert by_name["layer 1 - noise floor"]["takes"] == 2
    assert by_name["layer 1 - re-placed"]["takes"] == 1
    assert by_name["layer 2 - stacked true"]["takes"] == 1
    shifted = by_name["layer 2 - top ring shifted - introduced offset (10, 0) mm"]
    assert shifted["takes"] == 1 and shifted["layer_index"] == 2
    assert shifted["introduced_norm_mm"] == pytest.approx(10.0)


def test_takes_of_the_same_layer_stay_separate_when_no_phase_was_recorded(tmp_path):
    """Without a phase the layer still separates the conditions."""
    root = tmp_path / "runs" / "extrusion"
    session = MeasureSession.create(root, auto_plan())
    t = session.trial_id
    _write_take(root, t, 1, 1, offset=None, offset_norm=0.4, rms=0.5, mean_abs=0.4,
                maximum=1.1, acq_ms=1000)
    _write_take(root, t, 2, 1, offset=None, offset_norm=0.6, rms=0.6, mean_abs=0.5,
                maximum=1.3, acq_ms=1000)

    by_name = {c["condition"]: c for c in paper_summary(root, t)["conditions"]}

    assert by_name["layer 1 - no introduced offset"]["takes"] == 1
    assert by_name["layer 2 - no introduced offset"]["takes"] == 1


def test_an_invalid_take_is_reported_but_never_averaged_into_the_deviations(tmp_path):
    """An invalid take has no reconstructed path; its numbers are not measurements."""
    root = tmp_path / "runs" / "extrusion"
    session = MeasureSession.create(root, auto_plan())
    t = session.trial_id
    _write_take(root, t, 1, 1, offset=None, offset_norm=0.5, rms=0.5, mean_abs=0.4,
                maximum=1.0, acq_ms=1000, phase="noise floor")
    _write_take(root, t, 1, 2, offset=None, offset_norm=44.0, rms=51.0, mean_abs=48.0,
                maximum=77.0, acq_ms=1000, phase="noise floor", valid=False)

    condition = paper_summary(root, t)["conditions"][0]

    assert condition["takes"] == 2 and condition["valid"] == 1
    assert condition["rms_mm"]["n"] == 1 and condition["rms_mm"]["mean"] == pytest.approx(0.5)
    assert condition["center_offset_norm_mm"]["mean"] == pytest.approx(0.5)


# --------------------------------------------- the plan a session is bound to

def _characterized(client, root, *, radius_mm=61.24, center=(214.0, 141.0)):
    """Characterize + Apply, as the protocol requires between placing and measuring."""
    client.post("/api/modules/extrusion/measure/session/new", json={"note": "rings"})
    session = MeasureSession.latest(root)
    session.characterizations.append({"index": 1, "radius_mm": radius_mm,
                                      "center_mm": list(center), "bead_width_mm": 8.31,
                                      "top_z_mean_mm": 6.44, "top_z_min_mm": 5.1,
                                      "top_z_max_mm": 9.8})
    session.save()
    return client.post("/api/modules/extrusion/measure/apply-characterization").json()


def test_measuring_against_a_plan_that_is_not_the_applied_one_is_refused(tmp_path, monkeypatch):
    """The stale-plan artifact, blocked at the source.

    On the cell (2026-08-28) *Apply to recipe & placement* was skipped between
    Characterize and Measure, and layer-001 measured the ring against a plan it
    was never placed on: 15.38 mm centre offset, 11.31 mm RMS, both meaningless.
    Pressing *Center on scanned surface -> Generate* after Apply does the same
    damage. Once a characterization is applied, a session measures against THAT
    plan or refuses.
    """
    monkeypatch.setattr(extrusion_module, "REPO_ROOT", tmp_path)
    client = TestClient(create_app(AppConfig()))
    api_plan(client)
    applied = _characterized(client, tmp_path / "runs" / "extrusion")

    regenerated = api_plan(client)                       # the pre-Apply plan, again
    assert regenerated["fingerprint"] != applied["fingerprint"]
    refused = client.post("/api/modules/extrusion/measure/layer",
                          json={"fingerprint": regenerated["fingerprint"], "layer_index": 1,
                                "annotation": {}, "confirm_robot_motion": True})

    assert refused.status_code == 409
    detail = refused.json()["detail"]
    assert "Apply to recipe & placement" in detail and "characteriz" in detail.lower()


def test_the_applied_plan_comes_back_after_a_restart(tmp_path, monkeypatch):
    """The plan lives in memory; the session outlives the process.

    "After a backend restart press Apply FIRST" was folklore the operator had to
    remember mid-experiment. The session already records what was applied.
    """
    monkeypatch.setattr(extrusion_module, "REPO_ROOT", tmp_path)
    client = TestClient(create_app(AppConfig()))
    api_plan(client)
    applied = _characterized(client, tmp_path / "runs" / "extrusion")

    restarted = TestClient(create_app(AppConfig()))              # a fresh process
    plan = restarted.get("/api/modules/extrusion/plan").json()

    assert plan["fingerprint"] == applied["fingerprint"]
    assert plan["recipe"]["radius_mm"] == applied["recipe"]["radius_mm"]
    assert plan["setup"]["center_x_mm"] == pytest.approx(214.0)
    assert plan["restored_from"] == MeasureSession.latest(tmp_path / "runs" / "extrusion").trial_id


def test_measuring_layer_two_before_layer_one_has_a_measured_top_is_refused(tmp_path, monkeypatch):
    """Layer N's ROI floor IS layer N-1's latest measured take.

    Measured without it, a stacked ring blends with the ring beneath and the
    synthetic proof shows the branch guard exhausting outright.
    """
    monkeypatch.setattr(extrusion_module, "REPO_ROOT", tmp_path)
    client = TestClient(create_app(AppConfig()))
    api_plan(client)
    applied = _characterized(client, tmp_path / "runs" / "extrusion")

    refused = client.post("/api/modules/extrusion/measure/layer",
                          json={"fingerprint": applied["fingerprint"], "layer_index": 2,
                                "annotation": {}, "confirm_robot_motion": True})

    assert refused.status_code == 409
    assert "layer 1" in refused.json()["detail"]


def test_a_failed_take_stays_visible_in_the_session(tmp_path, monkeypatch):
    """A failure the operator cannot see is a failure they cannot reprocess.

    The raw frame is archived, so the take is recoverable -- but only if the
    session shows it happened.
    """
    svc, rdk, _ = measure_env(tmp_path, monkeypatch)
    monkeypatch.setattr(measure_mod, "process_observation",
                        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad skeleton")))
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)

    with pytest.raises(RuntimeError, match="raw RGB-D archived"):
        RingMeasureJob(svc, plan, session, 1, annotation={"phase": "noise floor"},
                       check_collisions=True)(Ctx())

    reloaded = MeasureSession.load(tmp_path / "runs" / "extrusion", session.trial_id)
    assert len(reloaded.records) == 1
    record = reloaded.records[0]
    assert record["valid"] is False and record["layer_name"] == "layer-001"
    assert "bad skeleton" in record["error"]
    assert record["annotation"] == {"phase": "noise floor"}
    assert 1 not in reloaded.tops                       # a failure is not a floor


def test_apply_after_a_restart_rebuilds_from_the_sessions_own_trial(tmp_path, monkeypatch):
    """A session that predates the applied-plan record still recovers exactly.

    The paper session (20260828-204846-5b455377) was applied before the session
    recorded what it applied, so a restart can only rebuild it from the trial it
    was created with. Falling back to the module's config defaults instead would
    hand the operator a different work frame, orientation and path resolution --
    a plan the ring was never measured against.
    """
    monkeypatch.setattr(extrusion_module, "REPO_ROOT", tmp_path)
    root = tmp_path / "runs" / "extrusion"
    client = TestClient(create_app(AppConfig()))
    api_plan(client)                                     # points_per_circle 24, not the default 180
    applied = _characterized(client, root)

    session = MeasureSession.latest(root)
    session.applied = None                               # as an archive from before this existed
    session.save()
    restarted = TestClient(create_app(AppConfig()))
    recovered = restarted.post("/api/modules/extrusion/measure/apply-characterization").json()

    assert recovered["fingerprint"] == applied["fingerprint"]
    assert recovered["recipe"]["points_per_circle"] == 24
    assert recovered["setup"]["work_frame"] == "Tasni Work Frame"


# ================================= the Word draft the paper is written from ===

def _docx_content(path):
    """Paragraph texts and tables, as Word will show them."""
    from docx import Document
    document = Document(str(path))
    paragraphs = [p.text for p in document.paragraphs]
    tables = [[[cell.text for cell in row.cells] for row in table.rows]
              for table in document.tables]
    return paragraphs, tables


def _two_condition_trial(root):
    """A noise-floor condition and a 10 mm displaced condition, three takes each."""
    session = MeasureSession.create(root, auto_plan(), note="rings")
    trial_id = session.trial_id
    for take in (1, 2, 3):
        _write_take(root, trial_id, 1, take, offset=None, offset_norm=0.4 + take * 0.1,
                    rms=0.5, mean_abs=0.4, maximum=1.1, acq_ms=1000 + take * 10,
                    phase="noise floor")
    for take in (1, 2, 3):
        _write_take(root, trial_id, 3, take, offset=[10, 0], measured_offset=[10.2, 0.0],
                    rms=7.1, mean_abs=6.4, maximum=10.2, acq_ms=1000,
                    phase="top ring shifted")
    return trial_id


def test_the_word_draft_puts_the_measured_numbers_in_a_real_table(tmp_path):
    """A Markdown block pasted into Word is plain text; the paper needs a table.

    The numbers must also be the SAME object the app reports, never a second
    formatting of the archive that can drift from it.
    """
    pytest.importorskip("docx")
    from tasni.modules.extrusion.paper_docx import build_paper_docx

    root = tmp_path / "runs" / "extrusion"
    trial_id = _two_condition_trial(root)

    out = build_paper_docx(root, trial_id, tmp_path / "draft.docx", embed_figures=False)

    assert out.is_file() and out.suffix == ".docx"
    paragraphs, tables = _docx_content(out)
    # By header, not by position, for the columns too: the draft grows both
    # sections and columns over time, and a positional assertion here just
    # breaks on the next honest addition instead of catching anything.
    conditions = next(t for t in tables if t[0][0] == "Condition")
    column = {name: index for index, name in enumerate(conditions[0])}
    rows = {row[0]: row for row in conditions[1:]}
    assert "layer 1 - noise floor" in rows
    assert rows["layer 1 - noise floor"][column["n"]] == "3"
    shifted = rows["layer 3 - top ring shifted - introduced offset (10, 0) mm"]
    assert shifted[column["n"]] == "3"
    assert "10.20" in shifted[column["Centre offset (mm)"]]
    assert "0.20" in shifted[column["Detection error (mm)"]]
    # How the takes were bought is part of the claim: three frames from one trip
    # and three re-approaches support different sentences about repeatability.
    assert shifted[column["How"]] in {"arm parked", "re-approached", "one take"}
    assert any("10.0 mm introduced offset was recovered" in p for p in paragraphs)


def test_the_draft_says_what_is_still_missing_while_the_run_is_under_way(tmp_path):
    """It is written DURING collection, so it has to show what is not there yet."""
    pytest.importorskip("docx")
    from tasni.modules.extrusion.paper_docx import build_paper_docx

    root = tmp_path / "runs" / "extrusion"
    session = MeasureSession.create(root, auto_plan(), note="rings")
    _write_take(root, session.trial_id, 1, 1, offset=None, offset_norm=0.4, rms=0.5,
                mean_abs=0.4, maximum=1.1, acq_ms=1000, phase="noise floor")

    out = build_paper_docx(root, session.trial_id, tmp_path / "draft.docx",
                           embed_figures=False)

    paragraphs, _ = _docx_content(out)
    text = "\n".join(paragraphs)
    assert "2 more" in text                    # 1 of the 3 takes a condition needs
    assert "11 more" in text                   # 1 of the 12 measurements requirement 3 needs
    assert "not ready to cite" in text.lower()


def test_the_draft_carries_the_wording_the_paper_is_required_to_use(tmp_path):
    """Hand-placed dried beads are not a printed cylinder, and the text says so."""
    pytest.importorskip("docx")
    from tasni.modules.extrusion.paper_docx import build_paper_docx

    root = tmp_path / "runs" / "extrusion"
    trial_id = _two_condition_trial(root)

    out = build_paper_docx(root, trial_id, tmp_path / "draft.docx", embed_figures=False)

    text = "\n".join(_docx_content(out)[0])
    assert "controlled validation" in text
    assert "not the deposition deviation of a printed cylinder" in text
    assert "1.26" in text                      # the error floor, stated where it is claimed


def test_every_take_is_listed_so_the_appendix_needs_no_transcription(tmp_path):
    pytest.importorskip("docx")
    from tasni.modules.extrusion.paper_docx import build_paper_docx

    root = tmp_path / "runs" / "extrusion"
    trial_id = _two_condition_trial(root)

    out = build_paper_docx(root, trial_id, tmp_path / "draft.docx", embed_figures=False)

    takes_table = next(t for t in _docx_content(out)[1] if t[0][0] == "Layer")
    assert takes_table[0][:4] == ["Layer", "Take", "Phase", "Introduced (mm)"]
    assert len(takes_table) == 7                        # header + six takes
    assert [row[0] for row in takes_table[1:]] == ["1", "1", "1", "3", "3", "3"]


def test_the_word_draft_is_served_by_the_api(tmp_path, monkeypatch):
    pytest.importorskip("docx")
    monkeypatch.setattr(extrusion_module, "REPO_ROOT", tmp_path)
    client = TestClient(create_app(AppConfig()))
    api_plan(client)
    trial_id = _two_condition_trial(tmp_path / "runs" / "extrusion")

    got = client.get(f"/api/modules/extrusion/trials/{trial_id}/paper-draft.docx")

    assert got.status_code == 200
    assert got.headers["content-type"] == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert trial_id in got.headers.get("content-disposition", "")
    assert client.get("/api/modules/extrusion/trials/nope/paper-draft.docx").status_code == 404


def test_the_draft_embeds_a_figure_for_every_condition(tmp_path):
    """The figures belong in the draft, not in a folder the writer has to hunt through."""
    pytest.importorskip("docx")
    pytest.importorskip("matplotlib")
    import test_extrusion_figures as figs
    from docx import Document
    from tasni.modules.extrusion.paper_docx import build_paper_docx

    root = tmp_path / "runs" / "extrusion"
    figs.write_take(root, layer_index=1, annotation={"phase": "noise floor"})
    figs.write_take(root, layer_index=2, annotation={"phase": "stacked true"})

    out = build_paper_docx(root, "t1", tmp_path / "draft.docx")

    document = Document(str(out))
    captions = [p.text for p in document.paragraphs if p.text.startswith("Figure ")]
    # The method figure leads, then the stack and the bead as a pipe, then a
    # plan view for each of the two conditions.
    assert len(document.inline_shapes) == len(captions) == 5
    assert "becoming a measured centreline" in captions[0]
    assert any("commanded bead" in c for c in captions)
    assert any("layer 1 - noise floor" in c for c in captions)
    assert any("layer 2 - stacked true" in c for c in captions)


# ------------------------- what inspecting a layer actually costs the print

def test_a_take_records_what_the_inspection_excursion_cost(tmp_path, monkeypatch):
    """The paper has to answer "what does stopping to look cost you?".

    Capture and processing were timed; the robot excursion -- out to the
    viewpoint, settle, and back to the start -- was not, and it is the larger
    half. It is also the only number here that cannot be recovered from the
    archive afterwards, so it has to be taken while the robot is moving.
    """
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)

    out = RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True)(Ctx())

    timings = json.loads(
        (Path(out["layer_dir"]) / "manifest.json").read_text())["processing"]["timings_ms"]
    for key in ("move_to_pose_ms", "settle_ms", "capture_ms", "total_ms",
                "return_ms", "inspection_cycle_ms"):
        assert key in timings, f"{key} was not recorded"
    assert timings["settle_ms"] == pytest.approx(svc.config.extrusion.settle_s * 1000.0)
    assert timings["inspection_cycle_ms"] == pytest.approx(
        timings["move_to_pose_ms"] + timings["settle_ms"] + timings["capture_ms"]
        + timings["total_ms"] + timings["return_ms"], abs=1e-6)
    # The session row carries it too, so the operator sees the cost as they go.
    assert "inspection_cycle_ms" in MeasureSession.load(
        tmp_path / "runs" / "extrusion", session.trial_id).records[0]["timings_ms"]


def test_the_summary_reports_what_inspection_costs_per_layer(tmp_path):
    """A duty-cycle number a reviewer can weigh against a layer time."""
    root = tmp_path / "runs" / "extrusion"
    session = MeasureSession.create(root, auto_plan(), note="rings")
    t = session.trial_id
    _write_take(root, t, 1, 1, offset=None, offset_norm=0.4, rms=0.5, mean_abs=0.4,
                maximum=1.1, acq_ms=1000, cycle_ms=14000)
    _write_take(root, t, 1, 2, offset=None, offset_norm=0.5, rms=0.5, mean_abs=0.4,
                maximum=1.2, acq_ms=1000, cycle_ms=16000)

    summary = paper_summary(root, t)

    cycle = summary["timing_ms"]["inspection_cycle_ms"]
    assert cycle["n"] == 2 and cycle["mean"] == pytest.approx(15000.0)
    markdown = summary["markdown"]
    assert "Inspecting one layer cost 15000" in markdown
    assert "rather than during deposition" in markdown


def test_an_unknown_api_path_fails_as_an_api_not_as_the_web_app(tmp_path, monkeypatch):
    """A missing API route must not answer with the single-page app.

    The catch-all that serves client-side routes was answering /api paths too,
    with index.html and status 200. The Word-draft download link saved that web
    page as a .docx -- a 496-byte HTML file named like a document, with no error
    raised anywhere. A route that does not exist has to say so.
    """
    monkeypatch.setattr(extrusion_module, "REPO_ROOT", tmp_path)
    client = TestClient(create_app(AppConfig()))

    missing = client.get("/api/modules/extrusion/there-is-no-such-endpoint")

    assert missing.status_code == 404
    assert "text/html" not in missing.headers.get("content-type", "")


# ------------------ the processing chain, made visible for the method figure

def test_processing_hands_back_every_stage_it_went_through():
    """The method figure has to show what the code did, not a redrawing of it.

    Each stage is handed back as the array the run actually held, and the last
    one IS what was measured -- so a figure built from these cannot drift away
    from the pipeline it claims to illustrate.
    """
    pytest.importorskip("open3d")
    plan = scene_plan(radius=60.0, bead=8.0, layer_height=6.0)
    stages: dict = {}
    result = observe(plan, 1, [syn.RingSpec(60.0, 8.0, CENTER, height_fn=syn.flat(6.0))],
                     stages=stages)

    for key in ("backprojected", "work_roi", "deposit_cluster",
                "radial_trimmed", "top_surface"):
        assert key in stages and len(stages[key]), f"{key} was not handed back"
        assert stages[key].shape[1] == 3
    # Each stage keeps a subset of the one before it.
    assert len(stages["backprojected"]) > len(stages["work_roi"])
    assert len(stages["work_roi"]) >= len(stages["deposit_cluster"])
    assert len(stages["deposit_cluster"]) >= len(stages["top_surface"])
    # The board is in the raw cloud and gone by the ROI.
    assert stages["backprojected"][:, 2].min() < 1.0
    assert stages["work_roi"][:, 2].min() >= 0.5
    assert np.array_equal(stages["top_surface"], result.filtered_xyz)


def test_an_archived_take_rebuilds_the_plan_it_was_measured_against():
    """Shared by offline reprocessing and by the method figure, so they agree."""
    from tasni.modules.extrusion.processing import plan_for_archived_take
    manifest = {"recipe": scene_plan(radius=42.6, bead=12.8).recipe.model_dump(mode="json"),
                "layer_index": 1}
    trial = {"setup": scene_plan().setup.model_dump(mode="json")}
    nominal = points_array(scene_plan(radius=42.6, center=(214.6, 146.7)).layers[0])

    plan = plan_for_archived_take(manifest, trial, nominal_xyz=nominal)

    assert plan.recipe.radius_mm == 42.6
    assert plan.setup.center_x_mm == pytest.approx(214.6, abs=0.01)
    assert plan.setup.center_y_mm == pytest.approx(146.7, abs=0.01)


def _take_with_provenance(root, trial_id, **extra):
    """A take carrying the provenance a real capture records."""
    _write_take(root, trial_id, 1, 1, offset=None, offset_norm=0.4, rms=0.5, mean_abs=0.4,
                maximum=1.1, acq_ms=1000, phase="noise floor", **extra)
    path = root / trial_id / "layer-001" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance"] = {
        "git_commit": "c48b1ab08a25bb0fa8fb2f6f3b9d3a52f77960e2",
        "camera_resolution": "1280x720",
        "camera_intrinsics": {"K": [[889.87, 0, 648.98], [0, 890.81, 362.0], [0, 0, 1]],
                              "dist_coeffs": [0.1148, -0.2, 0.0, 0.0, 0.0]},
        "calibration": {"run_id": "20260629-130945", "method": "HORAUD",
                        "quality": {"verdict": "borderline", "val_rms_px": 1.115,
                                    "board_consistency_rms_mm": 1.2602}},
        "processing_config": {"deposit_min_height_mm": 0.5, "radial_roi_margin_mm": 30.0,
                              "layer_floor_margin_mm": 2.0, "raster_mm_per_pixel": 1.0,
                              "measured_spline_points": 180, "bead_width_bins": 36,
                              "settle_s": 1.0},
        "inspection_pose": {"standoff_mm": 300.0},
        "work_frame": "Tasni Work Frame", "inspection_tool": "Realsense"}
    payload["processing"]["timings_ms"].update(
        {"move_to_pose_ms": 6100.0, "return_ms": 5400.0, "settle_ms": 1000.0,
         "inspection_cycle_ms": 14460.0})
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_the_draft_states_the_system_it_was_measured_on(tmp_path):
    """A reviewer has to be able to reproduce the setup, not guess at it.

    Every value comes from the take's own provenance -- including the hand-eye
    residual the error-floor claim rests on, which was previously a constant in
    the source and could drift from the calibration actually in use.
    """
    pytest.importorskip("docx")
    from tasni.modules.extrusion.paper_docx import build_paper_docx

    root = tmp_path / "runs" / "extrusion"
    trial_id = MeasureSession.create(root, auto_plan(), note="rings").trial_id
    _take_with_provenance(root, trial_id)

    out = build_paper_docx(root, trial_id, tmp_path / "draft.docx", embed_figures=False)

    paragraphs, tables = _docx_content(out)
    settings = {row[0]: row[1] for table in tables for row in table if len(row) == 2}
    assert "1280x720" in settings["Camera"]
    assert "300" in settings["Inspection standoff"]
    assert "889.9" in settings["Camera intrinsics"]          # fx from the archived K
    assert "1.26" in settings["Hand-eye calibration"] and "borderline" in settings["Hand-eye calibration"]
    assert "20260629-130945" in settings["Hand-eye calibration"]
    assert settings["Work frame"] == "Tasni Work Frame"
    assert "c48b1ab" in settings["Software revision"]
    assert "1.26" in "\n".join(paragraphs)                    # the floor, in the method text


def test_the_draft_breaks_down_what_one_inspection_costs(tmp_path):
    """Requirement 3 is a cycle time; a reviewer will ask what it is made of."""
    pytest.importorskip("docx")
    from tasni.modules.extrusion.paper_docx import build_paper_docx

    root = tmp_path / "runs" / "extrusion"
    trial_id = MeasureSession.create(root, auto_plan(), note="rings").trial_id
    _take_with_provenance(root, trial_id)

    out = build_paper_docx(root, trial_id, tmp_path / "draft.docx", embed_figures=False)

    tables = _docx_content(out)[1]
    timing = next(t for t in tables if t[0][0] == "Stage")
    stages = {row[0]: row for row in timing[1:]}
    assert "Move to the inspection pose" in stages
    assert "6100" in stages["Move to the inspection pose"][1]
    assert "Whole inspection excursion" in stages
    assert "14460" in stages["Whole inspection excursion"][1]


def test_the_draft_states_its_own_limitations(tmp_path):
    """The claims this experiment cannot support, written down before review."""
    pytest.importorskip("docx")
    from tasni.modules.extrusion.paper_docx import build_paper_docx

    root = tmp_path / "runs" / "extrusion"
    trial_id = MeasureSession.create(root, auto_plan(), note="rings").trial_id
    _take_with_provenance(root, trial_id)

    text = "\n".join(_docx_content(
        build_paper_docx(root, trial_id, tmp_path / "d.docx", embed_figures=False))[0])

    assert "between layers" in text and "during deposition" in text
    assert "Limitations" in text
    assert "reprocess" in text.lower()          # the archive claim, with its evidence


# ------------------------------------------------ paired detection error (2026-08-29)
# The steel rule measures the shift FROM WHERE THE RING SAT, so the chain's error
# must be scored the same way: against the ring's own last measured position
# before it was moved, not against the plan centre (which folds the operator's
# "placed true" bias into what the paper calls the chain's error).


def _stacked_then_shifted(root, *, bias=(1.5, -0.8), shift=(10.2, 0.1), typed=(10, 0)):
    """Ring 3 placed 'true' with a placement bias, measured 3x, then displaced."""
    trial_id = MeasureSession.create(root, auto_plan(), note="rings").trial_id
    for take in (1, 2, 3):
        _write_take(root, trial_id, 3, take, offset=None, measured_offset=list(bias),
                    rms=1.2, mean_abs=1.0, maximum=1.8, acq_ms=1000, phase="stacked true")
    after = [bias[0] + shift[0], bias[1] + shift[1]]
    for take in (4, 5, 6):
        _write_take(root, trial_id, 3, take, offset=list(typed), measured_offset=after,
                    rms=7.1, mean_abs=6.4, maximum=10.2, acq_ms=1000, phase="top ring shifted")
    return trial_id


def test_the_introduced_shift_is_scored_against_the_rings_own_position_before_it_moved(tmp_path):
    root = tmp_path / "runs" / "extrusion"
    trial_id = _stacked_then_shifted(root)

    by_name = {c["condition"]: c for c in paper_summary(root, trial_id)["conditions"]}
    shifted = by_name["layer 3 - top ring shifted - introduced offset (10, 0) mm"]

    # Against the plan centre the 1.5/-0.8 placement bias pollutes the score ...
    assert shifted["detection_error_mm"]["mean"] == pytest.approx(np.hypot(1.7, -0.7), abs=1e-9)
    # ... paired against the ring's last pre-shift measurement it is the chain alone.
    assert shifted["paired_detection_error_mm"]["n"] == 3
    assert shifted["paired_detection_error_mm"]["mean"] == pytest.approx(np.hypot(0.2, 0.1), abs=1e-9)
    assert shifted["paired_shift_norm_mm"]["mean"] == pytest.approx(np.hypot(10.2, 0.1), abs=1e-9)
    assert shifted["paired_reference_takes"] == [3]
    assert by_name["layer 3 - stacked true"]["paired_detection_error_mm"]["n"] == 0


def test_the_pre_shift_reference_is_the_last_zero_offset_take_before_it_never_after(tmp_path):
    """A ring put back after the shift is a later zero-offset take; it is not the reference."""
    root = tmp_path / "runs" / "extrusion"
    trial_id = _stacked_then_shifted(root)
    _write_take(root, trial_id, 3, 7, offset=None, measured_offset=[4.0, 4.0],
                rms=1.2, mean_abs=1.0, maximum=1.8, acq_ms=1000, phase="stacked true")

    shifted = {c["condition"]: c for c in paper_summary(root, trial_id)["conditions"]}[
        "layer 3 - top ring shifted - introduced offset (10, 0) mm"]

    assert shifted["paired_reference_takes"] == [3]
    assert shifted["paired_detection_error_mm"]["mean"] == pytest.approx(np.hypot(0.2, 0.1), abs=1e-9)


def test_a_shift_with_no_prior_measurement_of_that_layer_is_scored_against_the_plan_centre_only(tmp_path):
    root = tmp_path / "runs" / "extrusion"
    trial_id = MeasureSession.create(root, auto_plan(), note="rings").trial_id
    _write_take(root, trial_id, 3, 1, offset=[10, 0], measured_offset=[10.4, 0.3],
                rms=7.0, mean_abs=6.3, maximum=9.9, acq_ms=1000, phase="top ring shifted")

    summary = paper_summary(root, trial_id)
    shifted = summary["conditions"][0]

    assert shifted["paired_detection_error_mm"]["n"] == 0
    assert shifted["paired_reference_takes"] == []
    assert shifted["detection_error_mm"]["mean"] == pytest.approx(0.5, abs=1e-9)
    assert any("plan centre only" in p for p in summary["prose"])


def test_the_markdown_and_prose_state_the_paired_recovery(tmp_path):
    root = tmp_path / "runs" / "extrusion"
    trial_id = _stacked_then_shifted(root)

    summary = paper_summary(root, trial_id)

    assert "paired detection error (mm)" in summary["markdown"]
    recovered = next(p for p in summary["prose"] if "10.0 mm introduced offset" in p)
    assert "10.20" in recovered and "0.22" in recovered
    assert "own last measured position" in recovered and "take 3" in recovered


def test_the_word_draft_carries_the_paired_detection_error(tmp_path):
    pytest.importorskip("docx")
    from tasni.modules.extrusion.paper_docx import build_paper_docx

    root = tmp_path / "runs" / "extrusion"
    trial_id = _stacked_then_shifted(root)

    out = build_paper_docx(root, trial_id, tmp_path / "draft.docx", embed_figures=False)

    paragraphs, tables = _docx_content(out)
    conditions = next(t for t in tables if t[0][0] == "Condition")
    column = conditions[0].index("Paired detection error (mm)")
    rows = {row[0]: row for row in conditions[1:]}
    assert "0.22" in rows["layer 3 - top ring shifted - introduced offset (10, 0) mm"][column]
    assert rows["layer 3 - stacked true"][column] == "-"
    takes = next(t for t in tables if t[0][0] == "Layer")
    assert "Paired error (mm)" in takes[0]


def test_the_draft_asks_for_a_pre_shift_measurement_when_a_shift_has_none(tmp_path):
    pytest.importorskip("docx")
    from tasni.modules.extrusion.paper_docx import build_paper_docx

    root = tmp_path / "runs" / "extrusion"
    trial_id = MeasureSession.create(root, auto_plan(), note="rings").trial_id
    for take in (1, 2, 3):
        _write_take(root, trial_id, 3, take, offset=[10, 0], measured_offset=[10.4, 0.3],
                    rms=7.0, mean_abs=6.3, maximum=9.9, acq_ms=1000, phase="top ring shifted")

    text = "\n".join(_docx_content(
        build_paper_docx(root, trial_id, tmp_path / "d.docx", embed_figures=False))[0])

    assert "measure the ring in place before displacing it" in text.lower()


def test_the_timing_requirement_counts_live_measurements_not_takes(tmp_path):
    """An offline reprocess has no live acquisition-to-path time, so it owes one more."""
    pytest.importorskip("docx")
    from tasni.modules.extrusion.paper_docx import build_paper_docx

    root = tmp_path / "runs" / "extrusion"
    trial_id = MeasureSession.create(root, auto_plan(), note="rings").trial_id
    _write_take(root, trial_id, 1, 1, offset=None, offset_norm=0.4, rms=0.5, mean_abs=0.4,
                maximum=1.1, acq_ms=1000, phase="noise floor", offline=True)

    text = "\n".join(_docx_content(
        build_paper_docx(root, trial_id, tmp_path / "d.docx", embed_figures=False))[0])

    assert "0 live measurement(s) recorded" in text and "12 more" in text


def test_the_axis_check_take_is_never_the_pre_shift_reference(tmp_path):
    """The 'which way is +X' take moved the ring an untyped amount: not where it sat."""
    root = tmp_path / "runs" / "extrusion"
    trial_id = MeasureSession.create(root, auto_plan(), note="rings").trial_id
    for take in (1, 2, 3):
        _write_take(root, trial_id, 3, take, offset=None, measured_offset=[1.5, -0.8],
                    rms=1.2, mean_abs=1.0, maximum=1.8, acq_ms=1000, phase="stacked true")
    # The throwaway: ring slid ~9 mm to learn the axis, nothing typed.
    _write_take(root, trial_id, 3, 4, offset=None, measured_offset=[1.5 + 9.0, -0.8],
                rms=6.0, mean_abs=5.5, maximum=9.0, acq_ms=1000, phase="axis check")
    for take in (5, 6, 7):
        _write_take(root, trial_id, 3, take, offset=[10, 0], measured_offset=[11.7, -0.7],
                    rms=7.1, mean_abs=6.4, maximum=10.2, acq_ms=1000, phase="top ring shifted")

    shifted = {c["condition"]: c for c in paper_summary(root, trial_id)["conditions"]}[
        "layer 3 - top ring shifted - introduced offset (10, 0) mm"]

    assert shifted["paired_reference_takes"] == [3]
    assert shifted["paired_detection_error_mm"]["mean"] == pytest.approx(np.hypot(0.2, 0.1), abs=1e-9)


# ------------- the depth must be of the pose it was taken at (cell, 2026-08-29)

def test_depth_plane_check_scales_raw_depth_words_by_the_frames_own_unit():
    """Task 9 review, Critical 1: ``depth`` is raw camera WORDS, not millimetres.

    Protocol 2 native depth is 0.1 mm/word. Left at the unit-blind default this
    reads a real ~312 mm standoff as ~3120 mm, fails loudly, and blames the work
    frame or a frozen camera -- the wrong cause -- on every single measure and
    characterize take.
    """
    from tasni.modules.extrusion.measure import depth_plane_check
    T = np.eye(4)
    T[2, 3] = 312.0                                   # camera 312 mm above the plane
    depth_words = np.full((8, 8), 3120, np.uint16)     # 3120 words * 0.1 mm/word = 312 mm

    scaled = depth_plane_check(depth_words, T, ExtrusionConfig(), unit_mm=0.1)
    assert scaled["observed_depth_mm"] == pytest.approx(312.0)
    assert scaled["agrees"] is True

    # The old, unit-blind call (no unit_mm) reads the same words as 3120 mm --
    # this is the regression the review measured on the checkout.
    unscaled = depth_plane_check(depth_words, T, ExtrusionConfig())
    assert unscaled["observed_depth_mm"] == pytest.approx(3120.0)
    assert unscaled["agrees"] is False


class StaleThenFreshCamera:
    """The Jetson's temporal depth filter blends across frames.

    After the arm moves, the first depth it emits is still weighted toward the
    PREVIOUS distance. On the cell this returned the parked height (447 mm) with
    a correct, freshly captured colour of the board at 312 mm -- geometry that
    could not both be true, and which put every point 135 mm outside the search
    region.
    """

    def __init__(self, stale_mm=640, fresh_mm=500, stale_frames=1):
        self.stale_mm, self.fresh_mm = stale_mm, fresh_mm
        self.stale_frames = stale_frames
        self.grabs = 0

    def grab(self, **kwargs):
        self.grabs += 1
        # The readiness grab happens before the move and is not counted against
        # the stale run: it is the capture after the move that must be fresh.
        value = self.stale_mm if self.grabs <= self.stale_frames + 1 else self.fresh_mm
        from tasni.core.camera import Frame
        return Frame(color=np.zeros((16, 16, 3), np.uint8),
                     depth=np.full((16, 16), value, np.uint16), timestamp=1.0,
                     geometry=gf.aligned(syn.K_720P, (16, 16)))


def test_a_depth_frame_from_the_wrong_pose_is_grabbed_again(tmp_path, monkeypatch):
    """A stale depth is silently wrong, so it must not be measured."""
    svc, rdk, _ = measure_env(tmp_path, monkeypatch)
    svc.camera = StaleThenFreshCamera()
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)

    out = RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True)(Ctx())

    manifest = json.loads((Path(out["layer_dir"]) / "manifest.json").read_text())
    check = manifest["processing"]["depth_plane_check"]
    assert check["retries"] == 1, "the stale frame should have been thrown away"
    assert check["camera_z_mm"] == pytest.approx(506.0)
    assert check["observed_depth_mm"] == pytest.approx(500.0)


def test_a_depth_frame_that_never_matches_the_pose_fails_loudly(tmp_path, monkeypatch):
    """Better a refused measurement than a plausible wrong one."""
    svc, rdk, _ = measure_env(tmp_path, monkeypatch)
    svc.camera = StaleThenFreshCamera(stale_frames=99)
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)

    with pytest.raises(RuntimeError) as failure:
        RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True)(Ctx())

    message = str(failure.value)
    assert "640" in message and "506" in message, message
    # The frames were byte-identical, which is what a stalled stream looks like:
    # the operator is told that, and the one command that clears it.
    assert "frozen" in message.lower(), message
    assert "jetson_deploy.py restart" in message, message
    assert rdk.events[-1] == ("move-joints", START_JOINTS)          # still comes home


# -- one press, several takes ------------------------------------------------
# Two questions the run asks separately, because they cost differently: how
# repeatably the CHAIN sees a ring that has not moved (frames with the arm
# parked, seconds each), and how repeatably the ARM comes back to look at it
# (whole trips out and back, a trip each). Before this, both were three presses
# of the same button and the difference was not recorded anywhere.

def test_parked_repeats_take_several_frames_on_one_trip(tmp_path, monkeypatch):
    """repeats=3 leaves the path ONCE and banks three takes from that pose."""
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)

    out = RingMeasureJob(svc, plan, session, 1, check_collisions=False, repeats=3)(Ctx())

    starts = [e for e in rdk.events if e[0] == "start"]
    homes = [e for e in rdk.events if e == ("move-joints", START_JOINTS)]
    assert len(starts) == 1, "a parked repeat must not re-run the inspection move"
    assert len(homes) == 1, "the arm comes home once, after the last frame"
    assert camera.grabs == 4                       # readiness + three measurements
    assert session.takes == {1: 3}
    assert out["takes_recorded"] == [1, 2, 3]
    names = sorted(p.parent.name for p in (tmp_path / "runs" / "extrusion").glob(
        "*/layer-*/manifest.json"))
    assert names == ["layer-001", "layer-001-take02", "layer-001-take03"]


def test_a_shared_trip_is_never_priced_as_one_take_s_excursion(tmp_path, monkeypatch):
    """The cycle figure is the cost of leaving the path for ONE measurement.

    Three frames from one trip cost one trip between them. Letting each claim an
    inspection_cycle_ms would divide the excursion's real price by three in the
    one statistic the paper quotes as scan-to-feedback turnaround.
    """
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)

    RingMeasureJob(svc, plan, session, 1, check_collisions=False, repeats=3)(Ctx())

    root = tmp_path / "runs" / "extrusion" / session.trial_id
    timings = [json.loads((root / name / "manifest.json").read_text())["processing"]["timings_ms"]
               for name in ("layer-001", "layer-001-take02", "layer-001-take03")]
    assert not any("inspection_cycle_ms" in t for t in timings)
    # The move belongs to the frame that followed it, the return to the last.
    assert "move_to_pose_ms" in timings[0] and "settle_ms" in timings[0]
    assert not any("move_to_pose_ms" in t for t in timings[1:])
    assert "return_ms" in timings[2] and "return_ms" not in timings[0]
    # ...and every frame still carries what it really measured.
    assert all(t["capture_ms"] >= 0 and t["acquisition_to_path_ms"] > 0 for t in timings)


def test_excursions_repeat_the_whole_trip_unattended(tmp_path, monkeypatch):
    """excursions=5 is five presses of the noise-floor button, without pressing.

    The arm must genuinely leave and come back each time -- that re-approach is
    the thing being measured -- and each take is priced as a whole excursion.
    """
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)

    out = RingMeasureJob(svc, plan, session, 1, annotation={"phase": "noise floor"},
                         check_collisions=False, excursions=5)(Ctx())

    assert len([e for e in rdk.events if e[0] == "start"]) == 5
    assert len([e for e in rdk.events if e == ("move-joints", START_JOINTS)]) == 5
    assert out["takes_recorded"] == [1, 2, 3, 4, 5]
    root = tmp_path / "runs" / "extrusion" / session.trial_id
    cycles = [json.loads(p.read_text())["processing"]["timings_ms"].get("inspection_cycle_ms")
              for p in sorted(root.glob("layer-001*/manifest.json"))]
    assert len(cycles) == 5 and all(c is not None for c in cycles)


def test_each_take_records_which_trip_it_came_from(tmp_path, monkeypatch):
    """Three frames of one trip and three re-approaches must be tellable apart.

    The difference between those two spreads IS the finding the noise floor
    supports, so a reader of the archive alone has to be able to separate them.
    """
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)

    RingMeasureJob(svc, plan, session, 1, check_collisions=False,
                   excursions=2, repeats=2)(Ctx())

    root = tmp_path / "runs" / "extrusion" / session.trial_id
    stamps = [json.loads(p.read_text())["provenance"]
              for p in sorted(root.glob("layer-001*/manifest.json"))]
    assert [(s["excursion_index"], s["repeat_index"]) for s in stamps] == [
        (1, 1), (1, 2), (2, 1), (2, 2)]
    assert all(s["repeats_in_excursion"] == 2 for s in stamps)


def test_a_batch_keeps_the_takes_it_already_measured(tmp_path, monkeypatch):
    """A failure on trip three does not throw away trips one and two.

    Unattended batches are the point of excursions; losing a whole run to the
    last frame would make them worse than pressing the button five times.
    """
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    calls = {"n": 0}

    def fail_on_the_third(**kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("branch guard exhausted")
        return fake_measure_processing(**kwargs)

    monkeypatch.setattr(measure_mod, "process_observation", fail_on_the_third)
    with pytest.raises(RuntimeError, match="measurement invalid"):
        RingMeasureJob(svc, plan, session, 1, check_collisions=False, excursions=5)(Ctx())

    assert session.takes == {1: 3}, "the failed take is still numbered and archived"
    valid = [r for r in session.records if r["valid"]]
    assert len(valid) == 2, "the two good takes survive for the next press to continue from"
    # And the arm is home, not parked over the ring.
    assert rdk.events[-1] == ("move-joints", START_JOINTS)


def test_the_api_carries_the_batch_counts_and_caps_them(tmp_path, monkeypatch):
    """A mistyped count must not commit the cell to unattended motion for a quarter hour."""
    from pydantic import ValidationError

    from tasni.modules.extrusion.module import MeasureLayerBody

    body = MeasureLayerBody(fingerprint="f", layer_index=1)
    assert (body.repeats, body.excursions) == (1, 1)
    assert MeasureLayerBody(fingerprint="f", layer_index=1, repeats=3).repeats == 3
    for bad in ({"repeats": 0}, {"excursions": 11}, {"repeats": -1}):
        with pytest.raises(ValidationError):
            MeasureLayerBody(fingerprint="f", layer_index=1, **bad)


def test_the_summary_separates_the_camera_s_repeatability_from_the_robot_s(tmp_path, monkeypatch):
    """One "repeatability" over both would credit the robot's scatter to the camera.

    Parked frames scatter by the sensing chain alone; takes that each cost a
    trip out and back scatter by the chain AND the re-approach. The paper wants
    to say what re-approaching costs, so the two must be pooled separately.
    """
    from tasni.modules.extrusion.measure import capture_style, centre_spread, paper_summary

    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    plan = auto_plan()
    root = tmp_path / "runs" / "extrusion"
    session = MeasureSession.create(root, plan)
    # A wandering fitted centre, so the two pools cannot come out identical by
    # accident: the noise-floor trips scatter ten times as far as the parked ones.
    centres = iter([(200.0, 150.0), (210.0, 150.0), (190.0, 150.0),     # 3 trips
                    (200.0, 150.0), (201.0, 150.0), (199.0, 150.0)])    # 3 parked frames

    def wander(**kwargs):
        out = fake_measure_processing(**kwargs)
        out.metrics.measured_center_mm = next(centres)
        return out

    monkeypatch.setattr(measure_mod, "process_observation", wander)
    RingMeasureJob(svc, plan, session, 1, annotation={"phase": "noise floor"},
                   check_collisions=False, excursions=3)(Ctx())
    RingMeasureJob(svc, plan, session, 1, annotation={"phase": "re-placed"},
                   check_collisions=False, repeats=3)(Ctx())

    summary = paper_summary(root, session.trial_id)
    by_name = {c["condition"]: c for c in summary["conditions"]}
    trips = by_name["layer 1 - noise floor"]
    parked = by_name["layer 1 - re-placed"]
    assert trips["capture"] == "re-approach" and parked["capture"] == "parked"
    assert centre_spread([]) == {"n": 0, "rms_mm": None, "max_mm": None}
    assert capture_style([{}]) == "single"

    repeat = summary["repeatability_mm"]
    assert repeat["sensing"]["takes"] == 3 and repeat["re_approach"]["takes"] == 3
    assert repeat["sensing"]["rms_mm"] == pytest.approx(1.0, abs=0.01)
    assert repeat["re_approach"]["rms_mm"] == pytest.approx(10.0, abs=0.01)
    # And it is stated, not merely computed: the paper is written from the prose.
    said = " ".join(summary["prose"])
    assert "Sensing repeatability" in said and "Re-approach repeatability" in said


def test_old_takes_without_a_trip_stamp_read_as_their_own_excursion(tmp_path, monkeypatch):
    """Every take archived before the batch split WAS one press, so one trip.

    Reading them as parked would retro-credit the chain with a repeatability it
    was never measured to have.
    """
    from tasni.modules.extrusion.measure import capture_style

    legacy = [{"metrics": {"valid": True, "measured_center_mm": [200.0, 150.0]}},
              {"metrics": {"valid": True, "measured_center_mm": [201.0, 150.0]}}]
    assert capture_style(legacy) == "re-approach"


def test_a_ring_placed_displaced_is_paired_against_the_ring_it_sits_on(tmp_path, monkeypatch):
    """Ring 4 arrives already off-centre, so it has no earlier self to pair with.

    The rule measured how far it sits from ring 3, so that is what it must be
    scored against. Scored against the plan centre instead, the whole stack's
    hand-placement error is charged to the sensing chain.
    """
    from tasni.modules.extrusion.measure import paper_summary

    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    plan = auto_plan(layers=4)
    root = tmp_path / "runs" / "extrusion"
    session = MeasureSession.create(root, plan)
    # The stack was hand-placed 3 mm off the plan centre; ring 4 then sits a
    # further 10 mm out. The chain must report 10, not 13.
    centres = iter([(203.0, 150.0)] * 3 + [(213.0, 150.0)] * 3)

    def at_position(**kwargs):
        out = fake_measure_processing(**kwargs)
        centre = next(centres)
        out.metrics.measured_center_mm = centre
        out.metrics.center_offset_mm = (centre[0] - 200.0, centre[1] - 150.0)
        return out

    monkeypatch.setattr(measure_mod, "process_observation", at_position)
    for layer in (1, 2, 3):
        session.tops[layer] = [[0.0, 0.0, 0.0]]          # floors for the layer above
    RingMeasureJob(svc, plan, session, 3, annotation={"phase": "stacked true"},
                   check_collisions=False, repeats=3)(Ctx())
    RingMeasureJob(svc, plan, session, 4,
                   annotation={"phase": "top ring shifted",
                               "introduced_offset_mm": [10, 0]},
                   check_collisions=False, repeats=3)(Ctx())

    summary = paper_summary(root, session.trial_id)
    shifted = next(c for c in summary["conditions"] if c["introduced_norm_mm"] > 0)
    assert shifted["paired_reference_layers"] == [3], "paired against the ring beneath it"
    assert shifted["paired_against_layer_below"] is True
    assert shifted["paired_shift_norm_mm"]["mean"] == pytest.approx(10.0, abs=0.01)
    assert shifted["paired_detection_error_mm"]["mean"] == pytest.approx(0.0, abs=0.01)
    # The unpaired score carries the stack's own 3 mm placement error.
    assert shifted["detection_error_mm"]["mean"] == pytest.approx(3.0, abs=0.01)
    assert any("the ring it was stacked on" in p for p in summary["prose"])


def test_a_ring_that_was_slid_still_pairs_against_its_own_earlier_take(tmp_path, monkeypatch):
    """The same-layer reference stays first choice; the layer below is a fallback.

    A ring measured true and THEN displaced has its own undisplaced position on
    record, which is a tighter reference than the ring beneath it.
    """
    from tasni.modules.extrusion.measure import pre_shift_reference

    same_layer = {"layer_index": 3, "take": 2, "annotation": {},
                  "metrics": {"valid": True, "measured_center_mm": [201.0, 150.0]}}
    below = {"layer_index": 2, "take": 9, "annotation": {},
             "metrics": {"valid": True, "measured_center_mm": [200.0, 150.0]}}
    shifted = {"layer_index": 3, "take": 5,
               "annotation": {"introduced_offset_mm": [10, 0]},
               "metrics": {"valid": True, "measured_center_mm": [211.0, 150.0]}}

    assert pre_shift_reference(shifted, [below, same_layer, shifted]) is same_layer
    # With no take of its own, it falls back to the ring beneath.
    assert pre_shift_reference(shifted, [below, shifted]) is below
    # Layer 1 has nothing beneath it, so it stays unpaired rather than inventing one.
    lonely = {**shifted, "layer_index": 1}
    assert pre_shift_reference(lonely, [below, lonely]) is None

def test_the_side_photo_goes_out_and_back_through_the_approach_target(tmp_path, monkeypatch):
    """Neutral -> approach -> side, and side -> approach -> neutral.

    The approach target is not a nicety: the direct joint move between the
    neutral pose and the side pose is the one that sweeps the arm through the
    things standing around the cell. Skipping it on the way BACK would bump into
    them just as hard as skipping it on the way out.
    """
    svc, rdk, camera = measure_env(tmp_path, monkeypatch, side_photo=True)
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)

    out = RingMeasureJob(svc, plan, session, 1, check_collisions=False, repeats=2)(Ctx())

    route = [e for e in rdk.events if e[0] in {"move-target", "move-joints"}]
    # By STORED JOINTS, not by target item: a cartesian MoveJ resolves against
    # whatever tool/frame the last take left active, which on the cell sent the
    # arm somewhere else entirely (137.8 s excursion, 2.7 s is an inspection move).
    assert route[-4:] == [("move-joints", SIDE_APPROACH_JOINTS),
                          ("move-joints", SIDE_JOINTS),
                          ("move-joints", SIDE_APPROACH_JOINTS),
                          ("move-joints", START_JOINTS)]
    assert out["side_view"]["captured"] is True
    assert camera.witness_grabs == 1, "the side photo is RGB only -- it measures nothing"


def test_a_cartesian_only_side_target_still_moves_but_says_it_is_ambiguous(tmp_path,
                                                                          monkeypatch):
    """No stored joints = no unambiguous move. Do it, but do not do it silently."""
    svc, rdk, camera = measure_env(tmp_path, monkeypatch, side_photo=True)
    rdk.taught_joints["SideCapture"] = None
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    ctx = Ctx()

    out = RingMeasureJob(svc, plan, session, 1, check_collisions=False)(ctx)

    assert out["side_view"]["captured"] is True
    assert ("move-target", "SideCapture") in rdk.events
    assert any("stores no joints" in m and "SideCapture" in m for m in ctx.logs), ctx.logs


def test_the_side_photo_is_taken_once_per_press_and_lands_on_the_last_take(tmp_path, monkeypatch):
    """The ring does not move between the frames of one capture, so one photo.

    It is archived beside the take it belongs to, so a reader of the archive
    finds the picture next to the numbers rather than in a separate pile.
    """
    svc, rdk, camera = measure_env(tmp_path, monkeypatch, side_photo=True)
    plan = auto_plan()
    root = tmp_path / "runs" / "extrusion"
    session = MeasureSession.create(root, plan)

    RingMeasureJob(svc, plan, session, 1, check_collisions=False, repeats=3)(Ctx())

    assert len([e for e in rdk.events if e == ("move-joints", SIDE_JOINTS)]) == 1
    trial = root / session.trial_id
    assert (trial / "layer-001-take03" / "side.png").is_file()
    assert not (trial / "layer-001" / "side.png").exists()
    manifest = json.loads((trial / "layer-001-take03" / "manifest.json").read_text())
    assert manifest["side_view"]["captured"] is True
    assert manifest["side_view"]["image_file"] == "side.png"
    assert manifest["side_view"]["target"] == "SideCapture"
    assert manifest["side_view"]["approach_target"] == "TowardsSideCapture"
    # Not folded into what the paper says an inspection costs.
    assert "side" not in json.dumps(manifest["processing"]["timings_ms"])
    assert session.records[-1]["side_view"]["captured"] is True


def test_a_missing_taught_target_skips_the_photo_and_never_moves(tmp_path, monkeypatch):
    """A station without the targets must still be able to measure.

    And it must not set off toward a target that is not there: the skip has to
    happen BEFORE any motion, not as a failed move.
    """
    svc, rdk, camera = measure_env(tmp_path, monkeypatch, side_photo=True)
    rdk.absent.add("TowardsSideCapture")
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)

    out = RingMeasureJob(svc, plan, session, 1, check_collisions=False)(Ctx())

    assert out["valid"] is True, "the measurement stands"
    assert not any(e[0] == "move-target" for e in rdk.events)
    side = out["side_view"]
    assert side["captured"] is False
    assert "TowardsSideCapture" in side["error"] and "not in the station" in side["error"]
    # Archived anyway: the manifest has to be able to say why there is no photo.
    manifest = json.loads((Path(out["layer_dir"]) / "manifest.json").read_text())
    assert manifest["side_view"]["captured"] is False and manifest["side_view"]["error"]


def test_a_failure_at_the_side_pose_still_retraces_and_keeps_the_measurement(tmp_path, monkeypatch):
    """An arm left parked out at the side pose is the worst outcome available.

    So the return leg runs even when the capture failed -- and the measurement,
    which is already on disk, is not put at risk for a figure.
    """
    svc, rdk, camera = measure_env(tmp_path, monkeypatch, side_photo=True)
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)

    def dead_camera(**kwargs):
        if kwargs.get("color_only"):
            raise RuntimeError("camera stopped answering")
        return FakeCamera.grab(camera, **kwargs)

    monkeypatch.setattr(camera, "grab", dead_camera)
    out = RingMeasureJob(svc, plan, session, 1, check_collisions=False)(Ctx())

    assert out["valid"] is True
    assert out["side_view"]["captured"] is False
    assert "camera stopped answering" in out["side_view"]["error"]
    route = [e for e in rdk.events if e[0] in {"move-target", "move-joints"}]
    assert route[-2:] == [("move-joints", SIDE_APPROACH_JOINTS),
                          ("move-joints", START_JOINTS)]


def test_the_side_photo_is_part_of_the_protocol_by_default(tmp_path, monkeypatch):
    """It is a step of the run, not an opt-in: the paper needs one per layer."""
    from tasni.core.config import AppConfig
    from tasni.modules.extrusion.module import MeasureLayerBody

    config = AppConfig().extrusion
    assert config.side_capture_enabled is True
    assert config.side_capture_target == "SideCapture"
    assert config.side_capture_approach_target == "TowardsSideCapture"
    assert MeasureLayerBody(fingerprint="f", layer_index=1).side_photo is None


def test_the_side_photo_is_served_to_the_browser_and_listed(tmp_path, monkeypatch):
    """A photo the operator cannot see is a photo they cannot check was framed."""
    svc, rdk, camera = measure_env(tmp_path, monkeypatch, side_photo=True)
    plan = auto_plan()
    root = tmp_path / "runs" / "extrusion"
    session = MeasureSession.create(root, plan)
    RingMeasureJob(svc, plan, session, 1, check_collisions=False)(Ctx())
    monkeypatch.setattr(extrusion_module, "REPO_ROOT", tmp_path)
    client = TestClient(create_app(AppConfig()))

    served = client.get(f"/api/modules/extrusion/trials/{session.trial_id}"
                        "/layers/layer-001/files/side.png")
    assert served.status_code == 200 and served.headers["content-type"] == "image/png"

    listed = client.get("/api/modules/extrusion/trials").json()
    take = next(t for i in listed["trials"] if i["trial_id"] == session.trial_id
                for t in i["layers"])
    assert take["has_side_view"] is True
    assert take["side_view"]["captured"] is True

    # A path that is not on the whitelist is still refused.
    assert client.get(f"/api/modules/extrusion/trials/{session.trial_id}"
                      "/layers/layer-001/files/depth.npy").status_code == 404
