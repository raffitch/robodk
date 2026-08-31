"""Measuring layer N of a STACK, where the ring beneath is in the frame too.

Layer 2 has never once been measured validly on the real cell -- 0 of 6 takes
across 2026-08-30 (ChArUco board, completeness 0.56-0.62) and 2026-08-31 (MDF,
0.27-0.36). Replaying the 2026-08-31 archive stage by stage put the loss in one
place: the ring reaches DBSCAN whole (36/36 angular bins in the work ROI) and
leaves it as 5-7 arcs, because the crest of a hand-placed ring 2 swings ~10 mm
around the circumference and the 3D neighbourhood breaks where it steps. Layer 1
fragments the same way (2 arcs) and is rescued by arc assembly; above layer 1
assembly was off, so the LARGEST ARC ALONE became "the ring" -- 110 deg of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extrusion_synthetic as syn  # noqa: E402
from tasni.core.config import ExtrusionConfig  # noqa: E402
from tasni.modules.extrusion.inspection import aim_point_mm  # noqa: E402
from tasni.modules.extrusion.models import CylinderRecipe, CylinderSetup  # noqa: E402
from tasni.modules.extrusion.processing import plan_for_archived_take  # noqa: E402
from tasni.modules.extrusion.toolpath import generate_cylinder_plan  # noqa: E402

CENTER = (200.0, 150.0)
RADIUS = 40.0
BEAD = 10.0
LAYER_HEIGHT = 6.0


def _thin_at(height_mm: float, angles_deg, *, width_deg: float = 12.0):
    """A ring that dips to nothing at ``angles_deg``: DBSCAN sees separate arcs."""
    centres = np.radians(np.asarray(angles_deg, dtype=float))

    def height(theta):
        h = np.full_like(theta, float(height_mm), dtype=float)
        for c in centres:
            d = np.abs(np.mod(theta - c + np.pi, 2 * np.pi) - np.pi)
            h = np.where(d < np.radians(width_deg), 0.0, h)
        return h

    return height


def _stack_plan(center=CENTER):
    recipe = CylinderRecipe(radius_mm=RADIUS, layer_count=2, layer_height_mm=LAYER_HEIGHT,
                            bead_diameter_mm=BEAD, robot_speed_mm_s=75,
                            extrusion_rate_pct=0, points_per_circle=180)
    setup = CylinderSetup(print_tool="LongCalibTool", work_frame="Tasni Work Frame",
                          inspection_tool="Realsense", inspection_auto=True,
                          center_x_mm=center[0], center_y_mm=center[1])
    return generate_cylinder_plan(recipe, setup)


def _observe_stack(plan, layer_index, rings, *, seed=0, stages=None):
    layer = plan.layers[layer_index - 1]
    T = syn.inspection_camera_T(aim_point_mm(plan.recipe, plan.setup, layer_index), 300.0)
    depth = syn.render_scene(rings, T,
                             plane_center_xy_mm=(plan.setup.center_x_mm,
                                                 plan.setup.center_y_mm), seed=seed)
    kwargs = {"stages": stages} if stages is not None else {}
    from tasni.modules.extrusion.processing import measure_take
    return measure_take(depth=depth, geometry=syn.geometry(), T_work_camera=T,
                        plan=plan, layer=layer, config=ExtrusionConfig(), **kwargs)


def _broken_stack(*, shift_mm: float):
    """Ring 1 whole on the plane; ring 2 on top of it, displaced and in arcs."""
    ring1 = syn.RingSpec(RADIUS, BEAD, CENTER, z_base_mm=0.0,
                         height_fn=syn.flat(LAYER_HEIGHT))
    ring2 = syn.RingSpec(RADIUS, BEAD, (CENTER[0] + shift_mm, CENTER[1]),
                         z_base_mm=LAYER_HEIGHT,
                         height_fn=_thin_at(LAYER_HEIGHT, (70.0, 190.0, 300.0)))
    return [ring1, ring2]


# --------------------------------------------------------------- the datum

def test_reprocessing_uses_the_layer_height_the_take_was_measured_at():
    """A stale ``build_plane_z_mm`` must not move the height band on a reprocess.

    ``trial.json`` is written at trial creation, BEFORE Characterize -> Apply, so
    its setup can disagree with the one the take was actually measured with. The
    2026-08-31 archive is exactly this: trial.json says build_plane_z 4.259 while
    the applied setup (and every archived nominal path) says 0.0. The archived
    ring carries the truth -- its Z is the commanded centreline -- so the plan a
    take is scored against has to take Z from there, the same way it already
    takes the CENTRE from there.
    """
    manifest = {"layer_index": 2,
                "recipe": {"radius_mm": 42.2, "layer_count": 4, "layer_height_mm": 4.6,
                           "bead_diameter_mm": 8.4, "robot_speed_mm_s": 75.0,
                           "extrusion_rate_pct": 0.0, "points_per_circle": 180}}
    trial = {"recipe": dict(manifest["recipe"]),
             "setup": {"print_tool": "LongCalibTool", "work_frame": "Tasni Work Frame",
                       "inspection_tool": "Realsense", "inspection_auto": True,
                       "center_x_mm": 196.5, "center_y_mm": 146.6,
                       "build_plane_z_mm": 4.259086258787363}}   # stale: pre-Apply
    theta = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    nominal = np.column_stack((208.48 + 42.2 * np.cos(theta),
                               138.03 + 42.2 * np.sin(theta),
                               np.full(180, 8.8)))              # layer 2, as measured

    plan = plan_for_archived_take(manifest, trial, nominal_xyz=nominal)

    assert plan.layers[1].nominal_z_mm == pytest.approx(8.8, abs=1e-6)
    assert plan.layers[0].nominal_z_mm == pytest.approx(4.2, abs=1e-6)


# ------------------------------------------------- layer 2 of a stack

RING2_STACKED = (Path(__file__).parent / "fixtures" / "extrusion" / "ring2"
                 / "ring2_stacked_layer2_20260831.npz")


def _stacked_layer2_fixture():
    """The cell frame layer 2 has never been measured from. See the ring2 README."""
    import json

    from tasni.core.depth_geometry import CameraGeometry

    f = np.load(RING2_STACKED)
    centre = tuple(float(v) for v in f["nominal_center_mm"])
    bead = float(f["recipe_bead_mm"])
    layer_height = float(f["recipe_layer_height_mm"])
    layer_index = int(f["layer_index"])
    # nominal_z = build_plane_z + bead/2 + (layer_index - 1) * layer_height
    build_plane_z = float(f["nominal_z_mm"]) - bead / 2 - (layer_index - 1) * layer_height
    recipe = CylinderRecipe(radius_mm=float(f["recipe_radius_mm"]), layer_count=4,
                            layer_height_mm=layer_height, bead_diameter_mm=bead,
                            robot_speed_mm_s=75, extrusion_rate_pct=0,
                            points_per_circle=180)
    setup = CylinderSetup(print_tool="LongCalibTool", work_frame="Tasni Work Frame old",
                          inspection_tool="Realsense", inspection_auto=True,
                          center_x_mm=centre[0], center_y_mm=centre[1],
                          build_plane_z_mm=build_plane_z)
    return {
        "depth": f["depth"],
        "geometry": CameraGeometry.from_greeting(json.loads(str(f["camera_geometry"]))),
        "T_work_camera": np.asarray(f["T_work_camera"], dtype=float),
        "plan": generate_cylinder_plan(recipe, setup),
        "layer_index": layer_index,
        "centre": centre,
    }


def test_the_cell_frame_layer_2_is_measured_as_one_ring_not_its_largest_arc():
    """trial 20260831-195459 layer-002 take 1: measured 0.294 complete, 254 deg gap.

    Replayed stage by stage, this frame reaches DBSCAN whole -- 36/36 angular
    bins in the work ROI -- and leaves it as five arcs, of which the largest
    spans 110 deg. That arc alone became the measurement.

    Measured 2026-08-31, this take end to end:

        stage             before -> after
        deposit cluster    11/36 -> 31/36 angular bins
        crest              11/36 -> 29/36
        measured path      11/36 -> 20/36
        completeness       0.294 -> 0.515
        max angular gap    254.1 -> 174.6 deg
        fitted radius       40.6 -> 43.29 mm   (nominal 42.2)

    It does NOT become valid, and must not. The crest still carries a contiguous
    ~50 deg sector with no usable depth return -- a 19 mm stack seen from one
    pose shadows itself -- and the measured path is the largest CONTIGUOUS arc of
    what was found (a partial ring is never closed into a whole one; see
    test_a_ring_measured_only_in_part_is_not_closed_into_a_full_one), so it stops
    at that hole. Hence 0.515 rather than the crest's 0.806.

    That is the point of the change. The gate must reject this frame for what is
    wrong with the CAPTURE, not because segmentation threw away five sixths of a
    ring it had already found.
    """
    pytest.importorskip("open3d")
    from tasni.modules.extrusion.processing import measure_take

    f = _stacked_layer2_fixture()
    out = measure_take(depth=f["depth"], geometry=f["geometry"],
                       T_work_camera=f["T_work_camera"], plan=f["plan"],
                       layer=f["plan"].layers[f["layer_index"] - 1],
                       config=ExtrusionConfig())

    counts = out.report["counts"]
    # The ring is ASSEMBLED from its arcs, not narrowed to the biggest one.
    assert "after_largest_cluster" not in counts, counts
    assert counts.get("assembled_clusters", 1) >= 2, counts
    # Well clear of the 0.294 this take actually scored, and short of the 0.90
    # gate it still has no business passing.
    assert 0.45 <= out.metrics.path_completeness < 0.90, counts
    assert out.metrics.maximum_angular_gap_deg < 200.0
    assert out.metrics.measured_radius_mm == pytest.approx(42.2, abs=2.0)


def test_the_cell_frame_layer_2_still_fails_the_gate_on_its_missing_sector():
    """Recovering the ring must not launder a frame that cannot be measured."""
    pytest.importorskip("open3d")
    from tasni.modules.extrusion.processing import measure_take

    f = _stacked_layer2_fixture()
    out = measure_take(depth=f["depth"], geometry=f["geometry"],
                       T_work_camera=f["T_work_camera"], plan=f["plan"],
                       layer=f["plan"].layers[f["layer_index"] - 1],
                       config=ExtrusionConfig())

    assert not out.metrics.valid
    assert any("completeness" in w or "gap" in w for w in out.metrics.warnings)


def test_a_displaced_layer_2_keeps_its_displacement_and_is_not_pulled_onto_ring_1():
    """The reason assembly was banned above layer 1 -- made structurally impossible.

    Assembly judges candidates on circle-fit shape alone, so nothing in it knows
    that an arc belongs to ring 2 rather than to the ring beneath. Fusing the two
    would return a centre somewhere between them and silently destroy the
    displacement this experiment exists to report. The height floor is what keeps
    that from being possible: the crest of the layer below cannot enter the
    population layer 2 is assembled from.

    Without the floor this frame does not merely measure wrong -- ring 1 and a
    displaced ring 2 raster into one branched blob and the branch guard exhausts,
    so the take yields nothing at all.
    """
    pytest.importorskip("open3d")
    plan = _stack_plan()
    shift = 8.0

    out = _observe_stack(plan, 2, _broken_stack(shift_mm=shift))

    assert out.metrics.center_offset_mm[0] == pytest.approx(shift, abs=1.5)
    assert abs(out.metrics.center_offset_mm[1]) < 1.5
    assert out.metrics.path_completeness >= 0.85, out.report["counts"]


def test_layer_1_of_the_same_stack_is_unchanged():
    """Layer 1 has a passing contract already; this must not move it."""
    pytest.importorskip("open3d")
    plan = _stack_plan()
    ring1 = syn.RingSpec(RADIUS, BEAD, CENTER, z_base_mm=0.0,
                         height_fn=syn.flat(LAYER_HEIGHT))

    out = _observe_stack(plan, 1, [ring1])

    assert out.metrics.valid, out.metrics.warnings
    assert out.metrics.measured_radius_mm == pytest.approx(RADIUS, abs=1.5)
    assert out.metrics.center_offset_norm_mm < 1.5
