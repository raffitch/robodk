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


# ------------------------------------------------- characterize a ring (Task 6)

from tasni.modules.extrusion.processing import characterize_ring


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
    found = characterize_ring(color=color, depth=depth, T_work_camera=T, K=syn.K_720P,
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
        color=color, depth=depth, T_work_camera=T, K=syn.K_720P,
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

    found = characterize_ring(
        color=color, depth=depth, T_work_camera=fixture["T_work_camera"],
        K=fixture["K"], search_center_mm=fixture["search_center_mm"],
        work_frame="Tasni Work Frame", config=ExtrusionConfig())

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


def measure_env(tmp_path, monkeypatch, *, hardware_approved=False):
    svc, rdk, camera = services(tmp_path)
    svc.config.extrusion.hardware_io_test_approved = hardware_approved
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


# --------------------------------------------------- paper summary (Task 11)

from tasni.modules.extrusion.measure import paper_summary


def _write_take(root, trial_id, layer_index, take, *, offset, offset_norm, rms, mean_abs, maximum,
                acq_ms, valid=True):
    manifest = LayerManifest(
        trial_id=trial_id, layer_index=layer_index, take=take, mode="MEASURE_ONLY",
        recipe=auto_plan().recipe, toolpath_fingerprint="f" * 64,
        annotation={"introduced_offset_mm": offset},
        metrics=DeviationMetrics(mean_absolute_mm=mean_abs, rms_mm=rms, maximum_mm=maximum,
                                 measured_center_mm=(0, 0), measured_radius_mm=40,
                                 path_completeness=0.99, maximum_angular_gap_deg=4, valid=valid,
                                 center_offset_norm_mm=offset_norm, shape_rms_mm=0.4),
        geometry=RingGeometry(top_z_mean_mm=6, top_z_min_mm=5, top_z_max_mm=9, top_z_std_mm=1,
                              height_mean_mm=6, height_min_mm=5, height_max_mm=9,
                              height_reference="build_plane", bead_width_mean_mm=8,
                              bead_width_min_mm=7, bead_width_max_mm=9, bead_width_bins=36),
        processing={"timings_ms": {"capture_ms": 40.0, "total_ms": acq_ms - 40.0,
                                   "acquisition_to_path_ms": acq_ms}})
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
    assert by_name["true (no introduced offset)"]["takes"] == 2
    shifted = by_name["introduced offset (10, 0) mm"]
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
