"""Multi-view ring capture: poses, gates, levelling, the joint circle solve, merge."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasni.core.config import ExtrusionConfig  # noqa: E402
from tasni.modules.extrusion.models import (CaptureRecord, LayerManifest,  # noqa: E402
                                            ViewRecord)


def test_multiview_config_defaults_match_the_spec():
    c = ExtrusionConfig()
    assert c.multiview_enabled is False           # opt-in; single view is the validated one
    assert c.multiview_tilt_deg == 15.0           # spec section 3: 20 deg costs 4.97 mm plane RMS
    assert c.multiview_max_tilt_deg == 25.0
    assert c.multiview_azimuths_deg == [0.0, 120.0, 240.0]
    assert c.multiview_min_cos_incidence == 0.5
    assert c.multiview_level_annulus_width_mm == 60.0
    assert c.multiview_level_min_points == 500
    assert c.multiview_max_level_mm == 10.0
    assert c.multiview_min_view_points == 200
    assert c.multiview_min_arc_deg == 90.0
    assert c.multiview_max_offset_mm == 5.0
    assert c.multiview_min_views == 2


def test_view_record_defaults_and_drop_reason():
    v = ViewRecord(name="star-120", tilt_deg=15.0, azimuth_deg=120.0)
    assert v.dropped is False and v.drop_reason is None
    assert v.solved_offset_mm is None and v.chroma_gated is None
    dropped = ViewRecord(name="star-240", tilt_deg=15.0, azimuth_deg=240.0,
                         dropped=True, drop_reason="chroma gate abstained")
    assert dropped.drop_reason == "chroma gate abstained"


def test_capture_record_defaults_to_single_view():
    c = CaptureRecord()
    assert c.style == "single" and c.views == [] and c.merged_points_file is None


def test_old_manifest_without_capture_still_validates(tmp_path):
    """extra='forbid' plus a new required field would break every archived take."""
    from tasni.modules.extrusion.models import CylinderRecipe
    recipe = CylinderRecipe(radius_mm=40.0, layer_count=1, layer_height_mm=6.0,
                            bead_diameter_mm=10.0, robot_speed_mm_s=75,
                            extrusion_rate_pct=0)
    m = LayerManifest(trial_id="t", layer_index=1, recipe=recipe,
                      toolpath_fingerprint="abc")
    assert m.capture is None
    assert LayerManifest.model_validate_json(m.model_dump_json()).capture is None


from tasni.modules.extrusion.inspection import (pose_from_aim,  # noqa: E402
                                                star_view_angles,
                                                star_view_candidates,
                                                multiview_plan)
import extrusion_synthetic as syn  # noqa: E402
import test_extrusion_measure as tem  # noqa: E402
import test_extrusion_job as tej  # noqa: E402


def test_star_angles_are_top_first_then_the_configured_azimuths():
    names = star_view_angles(ExtrusionConfig())
    assert names[0] == ("top", 0.0, 0.0)
    assert [n for n, _, _ in names] == ["top", "star-000", "star-120", "star-240"]
    assert [round(t, 3) for _, t, _ in names[1:]] == [15.0, 15.0, 15.0]
    assert [a for _, _, a in names[1:]] == [0.0, 120.0, 240.0]


def test_star_angles_honour_the_tilt_cap():
    c = ExtrusionConfig(multiview_tilt_deg=40.0, multiview_max_tilt_deg=25.0)
    assert all(t == 25.0 for _, t, _ in star_view_angles(c)[1:])


def test_star_candidates_vary_roll_only_never_tilt_or_azimuth():
    """A fallback may re-roll the wrist. It may NOT quietly become another view."""
    cands = star_view_candidates([0.0, 0.0, 5.0], 300.0, ExtrusionConfig(),
                                 tilt_deg=15.0, azimuth_deg=120.0)
    assert len(cands) == len(ExtrusionConfig().inspection_roll_candidates_deg)
    assert {c["tilt_deg"] for c in cands} == {15.0}
    assert {c["azimuth_deg"] for c in cands} == {120.0}
    assert [c["roll_deg"] for c in cands] == ExtrusionConfig().inspection_roll_candidates_deg


def test_every_star_view_keeps_the_aim_point_on_axis_at_the_standoff():
    aim = np.array([10.0, -20.0, 5.0])
    for _, tilt, azimuth in star_view_angles(ExtrusionConfig()):
        T = pose_from_aim(aim, 300.0, tilt_deg=tilt, azimuth_deg=azimuth)
        camera, axis = T[:3, 3], T[:3, 2]
        assert abs(np.linalg.norm(camera - aim) - 300.0) < 1e-6      # exact standoff
        to_aim = (aim - camera) / np.linalg.norm(aim - camera)
        assert np.dot(axis, to_aim) > 1.0 - 1e-9                     # exactly on axis


def test_star_views_sit_on_a_cone_and_are_120_deg_apart_in_the_work_frame():
    aim = np.array([0.0, 0.0, 5.0])
    cams = [pose_from_aim(aim, 300.0, tilt_deg=t, azimuth_deg=a)[:3, 3]
            for _, t, a in star_view_angles(ExtrusionConfig())[1:]]
    offsets = [c[:2] - aim[:2] for c in cams]
    radii = [float(np.linalg.norm(o)) for o in offsets]
    assert max(radii) - min(radii) < 1e-6                            # one cone
    angles = sorted(round(float(np.degrees(np.arctan2(o[1], o[0]))) % 360.0, 3)
                    for o in offsets)
    assert angles == [0.0, 120.0, 240.0]


def test_multiview_plan_executes_and_returns_correct_shape():
    """multiview_plan must call framing_standoff correctly and return output shape."""
    config = ExtrusionConfig()
    plan = tem.scene_plan()
    result = multiview_plan(plan.recipe, plan.setup, K=syn.K_720P, size_px=syn.SIZE_720P,
                            config=config)
    assert isinstance(result, dict)
    assert "standoff_mm" in result
    assert isinstance(result["standoff_mm"], float)
    assert config.inspection_min_mm <= result["standoff_mm"] <= config.inspection_max_mm
    assert "aim_mm" in result
    assert len(result["aim_mm"]) == 3
    assert all(isinstance(v, float) for v in result["aim_mm"])
    assert "views" in result
    assert isinstance(result["views"], list)
    assert len(result["views"]) == 4
    view_names = [v["name"] for v in result["views"]]
    assert view_names == ["top", "star-000", "star-120", "star-240"]
    view_tilts = [v["tilt_deg"] for v in result["views"]]
    assert view_tilts == [0.0, 15.0, 15.0, 15.0]


from tasni.modules.extrusion.measure import depth_plane_check  # noqa: E402

AIM = np.array([0.0, 0.0, 5.0])          # a layer top 5 mm above the work plane


def _frame(mm: float, shape=(48, 64)):
    """A depth frame whose every pixel reads the same distance, in 1 mm words."""
    return np.full(shape, float(mm), dtype=float)


def test_top_view_gate_is_unchanged_byte_for_byte():
    """Pinned from the implementation BEFORE this task. The single-view path is
    cell-validated and this refactor must not move it by one millimetre."""
    T = pose_from_aim(AIM, 300.0, tilt_deg=0.0)
    out = depth_plane_check(_frame(300.0), T, ExtrusionConfig(), unit_mm=1.0)
    assert out["camera_z_mm"] == pytest.approx(305.0, abs=1e-6)
    assert out["observed_depth_mm"] == pytest.approx(300.0, abs=1e-6)
    assert out["accepted_range_mm"] == [265.0, 320.0]
    assert out["agrees"] is True
    assert out["cos_incidence"] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("standoff, tilt", [(300.0, 25.0), (400.0, 20.0), (500.0, 18.0)])
def test_tilted_views_that_the_old_gate_rejected_now_pass(standoff, tilt):
    """Each row is a case computed against the real constants where the
    height-based gate fails: the median sits standoff*(1-cos t) above camera_z
    and the high side has only depth_plane_slack_mm = 15 mm of budget."""
    T = pose_from_aim(AIM, standoff, tilt_deg=tilt)
    config = ExtrusionConfig()
    camera_z = float(T[2, 3])
    old_high = camera_z + config.depth_plane_slack_mm
    assert standoff > old_high                      # the old gate WOULD have failed
    out = depth_plane_check(_frame(standoff), T, config, unit_mm=1.0)
    assert out["agrees"] is True
    assert out["cos_incidence"] == pytest.approx(np.cos(np.radians(tilt)), abs=1e-9)


def test_the_gate_keeps_its_sensitivity_across_the_whole_envelope():
    """The correction must not merely let tilted views through: the gap between
    expected and the true median has to stay at the aim_z term (+5.0 mm at tilt
    0) instead of growing with tilt, or the gate goes blind exactly where the
    data is worst."""
    for standoff in (300.0, 400.0, 500.0, 800.0):
        for tilt in (0.0, 10.0, 15.0, 20.0, 25.0):
            T = pose_from_aim(AIM, standoff, tilt_deg=tilt)
            out = depth_plane_check(_frame(standoff), T, ExtrusionConfig(), unit_mm=1.0)
            bias = out["expected_depth_mm"] - standoff
            assert 5.0 <= bias <= 5.8, (standoff, tilt, bias)
            assert out["agrees"] is True


def test_a_genuinely_wrong_depth_is_still_refused_at_tilt():
    """The frozen-stream fault the gate exists for (cell 2026-08-29: colour at
    312 mm, depth stuck at 447) must still be caught on a tilted view."""
    T = pose_from_aim(AIM, 300.0, tilt_deg=15.0)
    out = depth_plane_check(_frame(447.0), T, ExtrusionConfig(), unit_mm=1.0)
    assert out["agrees"] is False


def test_a_degenerate_near_horizontal_pose_is_refused_not_divided_by():
    T = pose_from_aim(AIM, 300.0, tilt_deg=80.0)          # cos = 0.17 < 0.5
    out = depth_plane_check(_frame(300.0), T, ExtrusionConfig(), unit_mm=1.0)
    assert out["agrees"] is False
    assert "incidence" in (out.get("refused") or "")


def test_native_depth_units_still_scale():
    T = pose_from_aim(AIM, 300.0, tilt_deg=15.0)
    out = depth_plane_check(_frame(3000.0), T, ExtrusionConfig(), unit_mm=0.1)
    assert out["observed_depth_mm"] == pytest.approx(300.0)
    assert out["agrees"] is True


def test_a_backward_facing_pose_names_the_real_cause_not_just_the_angle():
    """cos_incidence <= 0 is a different mistake from a genuinely oblique view
    (0 < cos_incidence < floor): a real hand-eye pose never points the lens
    away from the plane, so this is almost always an identity or otherwise
    non-camera rotation standing in for a camera pose -- exactly the fixture
    bug this task found in tests/test_extrusion_job.py's FAKE_CAMERA_T. The
    message must say so rather than just reporting a 180 deg incidence."""
    T = np.eye(4)                                     # identity rotation: lens points +Z, i.e. UP
    T[:3, 3] = [0.0, 0.0, 300.0]
    out = depth_plane_check(_frame(300.0), T, ExtrusionConfig(), unit_mm=1.0)
    assert out["agrees"] is False
    assert out["cos_incidence"] == pytest.approx(-1.0, abs=1e-12)
    message = (out.get("refused") or "").lower()
    assert "away" in message
    assert "identity" in message or "non-camera" in message
    # And the too-oblique-but-real branch keeps its old, narrower wording.
    oblique = depth_plane_check(_frame(300.0), pose_from_aim(AIM, 300.0, tilt_deg=80.0),
                                ExtrusionConfig(), unit_mm=1.0)
    assert "away" not in (oblique.get("refused") or "").lower()


def test_the_seam_reproduces_process_observation_exactly():
    """observation_points + process_points must equal process_observation. If it
    does not, every archived number silently changes meaning."""
    pytest.importorskip("open3d")
    import geometry_fixtures as gf
    from tasni.modules.extrusion.inspection import aim_point_mm
    from tasni.modules.extrusion.processing import (observation_points,
                                                    process_observation,
                                                    process_points)

    plan = tem.scene_plan()
    layer = plan.layers[0]
    T = syn.inspection_camera_T(aim_point_mm(plan.recipe, plan.setup, 1), 300.0)
    rings = [syn.RingSpec(60.0, 8.0, (200.0, 150.0), height_fn=syn.flat(6.0))]
    depth = syn.render_scene(rings, T, plane_center_xy_mm=(200.0, 150.0))
    color = np.zeros((syn.SIZE_720P[1], syn.SIZE_720P[0], 3), np.uint8)
    geom, config = gf.aligned(syn.K_720P, syn.SIZE_720P), ExtrusionConfig()

    whole = process_observation(color=color, depth=depth, geometry=geom,
                                T_work_camera=T, K=syn.K_720P, dist=None,
                                plan=plan, layer=layer, config=config)
    points, gated = observation_points(color=color, depth=depth, geometry=geom,
                                       T_work_camera=T, K=syn.K_720P, dist=None,
                                       config=config)
    split = process_points(points, plan=plan, layer=layer, config=config,
                           chroma_gated=gated)
    np.testing.assert_allclose(split.measured_xyz, whole.measured_xyz)
    assert split.metrics.model_dump() == whole.metrics.model_dump()


def test_the_seam_keeps_total_ms_spanning_back_projection():
    """total_ms feeds measure.py's acquisition_to_path_ms, which is the paper's
    scan-to-feedback number. A seam that starts the clock after back-projection
    would silently shorten a published measurement."""
    pytest.importorskip("open3d")
    import geometry_fixtures as gf
    from tasni.modules.extrusion.inspection import aim_point_mm
    from tasni.modules.extrusion.processing import (observation_points,
                                                    process_observation,
                                                    process_points)

    plan = tem.scene_plan()
    layer = plan.layers[0]
    T = syn.inspection_camera_T(aim_point_mm(plan.recipe, plan.setup, 1), 300.0)
    rings = [syn.RingSpec(60.0, 8.0, (200.0, 150.0), height_fn=syn.flat(6.0))]
    depth = syn.render_scene(rings, T, plane_center_xy_mm=(200.0, 150.0))
    color = np.zeros((syn.SIZE_720P[1], syn.SIZE_720P[0], 3), np.uint8)
    geom, config = gf.aligned(syn.K_720P, syn.SIZE_720P), ExtrusionConfig()

    whole = process_observation(color=color, depth=depth, geometry=geom,
                                T_work_camera=T, K=syn.K_720P, dist=None,
                                plan=plan, layer=layer, config=config)
    whole_timings = whole.report["timings_ms"]
    # total_ms has to at least cover the two stages it is timed to span --
    # equality would be fragile (there is centreline/branch-guard work and
    # bookkeeping between the two measured sub-intervals too), >= is not.
    assert whole_timings["total_ms"] >= (whole_timings["backproject_ms"]
                                         + whole_timings["filter_ms"])

    # Pre-compute the same points OUTSIDE any timer process_points owns, then
    # call process_points directly (no `started`): its own total_ms can only
    # span its own work, never the back-projection that already happened above.
    points, gated = observation_points(color=color, depth=depth, geometry=geom,
                                       T_work_camera=T, K=syn.K_720P, dist=None,
                                       config=config)
    direct = process_points(points, plan=plan, layer=layer, config=config,
                            chroma_gated=gated)
    assert whole_timings["total_ms"] > direct.report["timings_ms"]["total_ms"]


# ------------------------------------------------------- multiview: levelling, solve, merge

def test_synthetic_colour_actually_holds_the_chroma_gate():
    """If this fails, every merge test below is silently testing the fallback."""
    import cv2
    rings = [syn.RingSpec(60.0, 8.0, (200.0, 150.0), height_fn=syn.flat(6.0))]
    T = syn.inspection_camera_T(np.array([200.0, 150.0, 6.0]), 300.0)
    color = syn.render_color(rings, T)
    sat = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[:, :, 1]
    fraction = float((sat > 60).mean())
    assert fraction > ExtrusionConfig().deposit_min_chroma_fraction


from tasni.modules.extrusion.multiview import (ViewCloud, fit_circle,  # noqa: E402
                                               level_points, merge_views,
                                               solve_view_offsets)


def _ring_xy(cx, cy, r, n=720, arc_deg=360.0, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.radians(np.linspace(0.0, arc_deg, n, endpoint=False))
    rad = r + rng.normal(0.0, 0.2, n)
    return np.column_stack((cx + rad * np.cos(theta), cy + rad * np.sin(theta)))


def test_fit_circle_recovers_a_known_circle():
    cx, cy, r = fit_circle(_ring_xy(200.0, 150.0, 40.5))
    assert (cx, cy, r) == pytest.approx((200.0, 150.0, 40.5), abs=0.05)


def test_the_joint_solve_recovers_injected_offsets():
    truth = {"top": (0.0, 0.0), "star-000": (1.2, -0.7),
             "star-120": (-0.9, 0.4), "star-240": (0.3, 1.1)}
    mean = np.mean(list(truth.values()), axis=0)
    views = {name: _ring_xy(200.0, 150.0, 40.5, seed=i) + np.array(d)
             for i, (name, d) in enumerate(truth.items())}
    out = solve_view_offsets(views, ExtrusionConfig())
    for name, d in truth.items():
        # Recovered up to the gauge: the solve removes the MEAN displacement,
        # which is unobservable from the ring alone.
        expected = np.array(d) - mean
        assert np.allclose(out["offsets_mm"][name], -expected, atol=0.12), name


def test_the_gauge_is_a_consensus_not_an_anchor():
    """THE test that the old spec's circularity is gone. Displacing the TOP view
    must move the consensus centre by 1/n of the displacement -- not by zero,
    which is what anchoring every view to the top view would give."""
    base = {n: _ring_xy(200.0, 150.0, 40.5, seed=i)
            for i, n in enumerate(["top", "star-000", "star-120", "star-240"])}
    before = solve_view_offsets(base, ExtrusionConfig())["consensus_center_mm"]
    moved = dict(base)
    moved["top"] = base["top"] + np.array([4.0, 0.0])
    after = solve_view_offsets(moved, ExtrusionConfig())["consensus_center_mm"]
    assert after[0] - before[0] == pytest.approx(1.0, abs=0.15)      # 4.0 / 4 views
    assert after[1] - before[1] == pytest.approx(0.0, abs=0.15)


def test_offsets_sum_to_zero():
    views = {n: _ring_xy(200.0, 150.0, 40.5, seed=i) + np.array([i * 0.6, -i * 0.4])
             for i, n in enumerate(["top", "star-000", "star-120", "star-240"])}
    out = solve_view_offsets(views, ExtrusionConfig())
    total = np.sum(list(out["offsets_mm"].values()), axis=0)
    assert np.allclose(total, [0.0, 0.0], atol=1e-6)


def test_a_scaled_view_surfaces_as_residual_not_absorbed():
    """One shared radius is the point: a per-view radius would let a view with
    residual scale error fit itself perfectly and hide the problem."""
    views = {n: _ring_xy(200.0, 150.0, 40.5, seed=i)
             for i, n in enumerate(["top", "star-000", "star-120"])}
    views["star-240"] = _ring_xy(200.0, 150.0, 44.0, seed=9)         # +8.6% scale
    out = solve_view_offsets(views, ExtrusionConfig())
    assert out["residual_rms_mm"]["star-240"] > 5 * max(
        out["residual_rms_mm"][n] for n in ("top", "star-000", "star-120"))


def test_levelling_removes_an_injected_plane_tilt():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-150.0, 150.0, (6000, 2))
    tilt = np.radians(8.0)
    z = xy[:, 0] * np.tan(tilt) + 3.0
    levelled, diag = level_points(np.column_stack((xy, z)), r_inner_mm=90.0,
                                  r_outer_mm=150.0, center_xy=(0.0, 0.0),
                                  config=ExtrusionConfig())
    inside = np.linalg.norm(levelled[:, :2], axis=1) < 150.0
    assert abs(float(np.median(levelled[inside, 2]))) < 0.3
    assert diag["level_mm"] == pytest.approx(3.0, abs=0.5)


def _cloud(name, *, gated=True, cx=200.0, cy=150.0, r=40.5, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2 * np.pi, n)
    rad = r + rng.normal(0.0, 1.0, n)
    ring = np.column_stack((cx + rad * np.cos(theta), cy + rad * np.sin(theta),
                            rng.normal(6.0, 0.4, n)))
    board = np.column_stack((rng.uniform(cx - 160, cx + 160, 8000),
                             rng.uniform(cy - 160, cy + 160, 8000),
                             rng.normal(0.0, 0.3, 8000)))
    return ViewCloud(name=name, points=np.vstack((ring, board)), chroma_gated=gated)


def test_an_abstaining_view_is_dropped_not_contributed():
    plan = tem.scene_plan(radius=40.5, bead=10.0)
    views = [_cloud("top", seed=0), _cloud("star-000", seed=1),
             _cloud("star-120", seed=2), _cloud("star-240", gated=False, seed=3)]
    out = merge_views(views, plan=plan, layer=plan.layers[0], config=ExtrusionConfig())
    assert "star-240" in out.dropped and "colour gate" in out.dropped["star-240"]
    assert out.chroma_gated is True                # the merge keeps the 1.5 mm floor
    assert set(out.used) == {"top", "star-000", "star-120"}


def test_all_abstaining_falls_back_to_the_top_view():
    plan = tem.scene_plan(radius=40.5, bead=10.0)
    views = [_cloud(n, gated=False, seed=i) for i, n in
             enumerate(["top", "star-000", "star-120", "star-240"])]
    out = merge_views(views, plan=plan, layer=plan.layers[0], config=ExtrusionConfig())
    assert out.used == ["top"] and out.chroma_gated is False
    np.testing.assert_array_equal(out.points, views[0].points)


def test_a_wildly_misregistered_view_is_rejected_and_the_rest_survive():
    plan = tem.scene_plan(radius=40.5, bead=10.0)
    views = [_cloud("top", seed=0), _cloud("star-000", seed=1),
             _cloud("star-120", seed=2), _cloud("star-240", cx=230.0, seed=3)]
    out = merge_views(views, plan=plan, layer=plan.layers[0], config=ExtrusionConfig())
    assert "star-240" in out.dropped
    assert set(out.used) == {"top", "star-000", "star-120"}


# ------------------------------------------------------- Task 6: capture the views

def test_archive_writes_the_views_directory_and_the_merged_cloud(tmp_path):
    from tasni.modules.extrusion.archive import ExtrusionArchive
    from tasni.modules.extrusion.models import CaptureRecord, LayerManifest, ViewRecord
    plan = tem.scene_plan()
    archive = ExtrusionArchive(tmp_path)
    archive.create_trial("t1", plan)
    color = np.zeros((8, 8, 3), np.uint8)
    depth = np.ones((8, 8), np.uint16)
    manifest = LayerManifest(
        trial_id="t1", layer_index=1, recipe=plan.recipe,
        toolpath_fingerprint=plan.fingerprint,
        capture=CaptureRecord(style="star", merged_points_file="merged_points.npy",
                              views=[ViewRecord(name="star-120", tilt_deg=15.0,
                                                azimuth_deg=120.0)]))
    layer_dir = archive.write_layer(
        manifest, nominal_xyz=np.zeros((3, 3)), commanded_xyz=np.zeros((3, 3)),
        color=color, depth=depth,
        views=[{"name": "star-120", "color": color, "depth": depth,
                "pose": {"tilt_deg": 15.0}}],
        merged_points_xyz=np.zeros((5, 3)))
    assert (layer_dir / "color.png").is_file()          # top view stays at the root
    assert (layer_dir / "views" / "star-120" / "color.png").is_file()
    assert (layer_dir / "views" / "star-120" / "depth.npy").is_file()
    assert (layer_dir / "views" / "star-120" / "pose.json").is_file()
    assert (layer_dir / "merged_points.npy").is_file()


def test_single_view_take_writes_no_views_directory(tmp_path):
    from tasni.modules.extrusion.archive import ExtrusionArchive
    from tasni.modules.extrusion.models import LayerManifest
    plan = tem.scene_plan()
    archive = ExtrusionArchive(tmp_path)
    archive.create_trial("t1", plan)
    manifest = LayerManifest(trial_id="t1", layer_index=1, recipe=plan.recipe,
                             toolpath_fingerprint=plan.fingerprint)
    layer_dir = archive.write_layer(manifest, nominal_xyz=np.zeros((3, 3)),
                                    commanded_xyz=np.zeros((3, 3)),
                                    color=np.zeros((8, 8, 3), np.uint8),
                                    depth=np.ones((8, 8), np.uint16))
    assert not (layer_dir / "views").exists()


# ---------------------------------------------------- _build_inspection_move (tilt/azimuth)

from tasni.modules.extrusion.service import _build_inspection_move  # noqa: E402


def test_default_tilt_is_byte_identical_to_omitting_it():
    """Off means off: the two cell-validated live callers never pass tilt_deg,
    so the default MUST reach exactly today's fronto-parallel candidate walk."""
    plan = tej.plan(layers=1, auto_inspection=True)
    layer = plan.layers[0]
    svc_a, rdk_a, _ = tej.services(".")
    out_a = _build_inspection_move(
        rdk_a, plan, layer, inspection_name="X_Inspect", config=svc_a.config.extrusion,
        camera=svc_a.config.camera, start_joints=tej.START_JOINTS)
    svc_b, rdk_b, _ = tej.services(".")
    out_b = _build_inspection_move(
        rdk_b, plan, layer, inspection_name="X_Inspect", config=svc_b.config.extrusion,
        camera=svc_b.config.camera, start_joints=tej.START_JOINTS,
        tilt_deg=0.0, azimuth_deg=0.0)
    assert out_a == out_b
    assert rdk_a.events == rdk_b.events
    np.testing.assert_array_equal(rdk_a.targets[0]["T"], rdk_b.targets[0]["T"])
    assert out_a["pose"]["tilt_deg"] == 0.0 and out_a["pose"]["azimuth_deg"] == 0.0


def test_nonzero_tilt_selects_star_view_candidates_not_the_fallback_walk():
    """tilt_deg != 0 must land on the REQUESTED tilt/azimuth (what the view IS),
    never on a different one from the fronto-parallel fallback table."""
    plan = tej.plan(layers=1, auto_inspection=True)
    layer = plan.layers[0]
    svc, rdk, _ = tej.services(".")
    out = _build_inspection_move(
        rdk, plan, layer, inspection_name="X_Inspect", config=svc.config.extrusion,
        camera=svc.config.camera, start_joints=tej.START_JOINTS,
        tilt_deg=15.0, azimuth_deg=120.0)
    assert out["pose"]["tilt_deg"] == 15.0
    assert out["pose"]["azimuth_deg"] == 120.0
    # Only roll varied across the candidates actually tried.
    assert len(rdk.targets) == 1, "the first (roll 0) candidate should have been reachable"


def test_a_star_view_that_cannot_be_reached_raises_and_never_falls_back_to_a_different_tilt():
    plan = tej.plan(layers=1, auto_inspection=True)
    layer = plan.layers[0]
    svc, rdk, _ = tej.services(".")
    rdk.unreachable_targets = 999
    with pytest.raises(RuntimeError, match="no reachable, collision-free inspection pose") as excinfo:
        _build_inspection_move(
            rdk, plan, layer, inspection_name="X_Inspect", config=svc.config.extrusion,
            camera=svc.config.camera, start_joints=tej.START_JOINTS,
            tilt_deg=15.0, azimuth_deg=120.0)
    # Every candidate actually tried (one per configured roll) kept the
    # REQUESTED tilt/azimuth -- a star view never substitutes a different one.
    message = str(excinfo.value)
    assert len(rdk.targets) == len(svc.config.extrusion.inspection_roll_candidates_deg)
    assert "tilt 15/azimuth 120" in message


# ---------------------------------------------------- RingMeasureJob(multiview=True) wiring

from tasni.modules.extrusion import measure as measure_mod  # noqa: E402
from tasni.modules.extrusion.measure import (RingMeasureJob, capture_views,  # noqa: E402
                                             MeasureSession)
from tasni.modules.extrusion.multiview import MergeResult  # noqa: E402


def _fake_observation_points(**kwargs):
    _fake_observation_points.calls.append(kwargs)
    return np.zeros((10, 3)), True


_fake_observation_points.calls = []


def _fake_merge_views(views, *, plan, layer, config):
    names = [v.name for v in views]
    return MergeResult(
        points=np.zeros((10, 3)), chroma_gated=True, used=list(names), dropped={},
        consensus_center_mm=(1.0, 2.0), consensus_radius_mm=40.5,
        offsets_mm={n: (0.1, -0.2) for n in names},
        residual_rms_mm={n: 0.3 for n in names},
        spread_before_mm=0.6, residual_after_mm=0.2)


def _fake_process_points(points, **kwargs):
    return tem.fake_measure_processing(**kwargs)


def _multiview_env(tmp_path, monkeypatch):
    svc, rdk, camera = tem.measure_env(tmp_path, monkeypatch)
    monkeypatch.setattr(measure_mod, "observation_points", _fake_observation_points)
    monkeypatch.setattr(measure_mod, "merge_views", _fake_merge_views)
    monkeypatch.setattr(measure_mod, "process_points", _fake_process_points)
    _fake_observation_points.calls.clear()
    tem.fake_measure_processing.calls.clear()
    return svc, rdk, camera


def test_multiview_off_takes_the_identical_single_view_path(tmp_path, monkeypatch):
    """off means off: with multiview=False, capture_views/merge_views/observation_points
    must never run and the archive/manifest must be exactly what today writes."""
    svc, rdk, camera = _multiview_env(tmp_path, monkeypatch)
    plan = tem.auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    out = RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True,
                         multiview=False)(tem.Ctx())
    assert camera.grabs == 2                       # readiness + exactly one measurement
    assert not any(e[0] == "start" and "_star" in e[1] for e in rdk.events)
    assert _fake_observation_points.calls == []     # the star seam never ran
    manifest = json.loads((Path(out["layer_dir"]) / "manifest.json").read_text())
    assert manifest["capture"] is None
    assert not (Path(out["layer_dir"]) / "views").exists()
    assert not (Path(out["layer_dir"]) / "merged_points.npy").exists()


def test_multiview_on_visits_four_poses_and_archives_the_star(tmp_path, monkeypatch):
    svc, rdk, camera = _multiview_env(tmp_path, monkeypatch)
    plan = tem.auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    out = RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True,
                         multiview=True)(tem.Ctx())
    stem = f"TasniCylinder_MEASURE_{plan.fingerprint[:10]}_L001_Inspect"
    starts = [e[1] for e in rdk.events if e[0] == "start"]
    assert starts == [stem, f"{stem}_star000", f"{stem}_star120", f"{stem}_star240"]
    assert camera.grabs == 1 + 4                    # readiness + one frame per view
    layer_dir = Path(out["layer_dir"])
    assert (layer_dir / "color.png").is_file() and (layer_dir / "depth.npy").is_file()
    assert (layer_dir / "views" / "top" / "pose.json").is_file()
    assert not (layer_dir / "views" / "top" / "color.png").exists()   # top stays at the root
    for name in ("star-000", "star-120", "star-240"):
        assert (layer_dir / "views" / name / "color.png").is_file()
        assert (layer_dir / "views" / name / "depth.npy").is_file()
        assert (layer_dir / "views" / name / "pose.json").is_file()
    assert (layer_dir / "merged_points.npy").is_file()
    manifest = json.loads((layer_dir / "manifest.json").read_text())
    assert manifest["capture"]["style"] == "star"
    assert len(manifest["capture"]["views"]) == 4
    assert {v["name"] for v in manifest["capture"]["views"]} == {
        "top", "star-000", "star-120", "star-240"}
    assert all(not v["dropped"] for v in manifest["capture"]["views"])
    assert manifest["capture"]["consensus_center_mm"] == [1.0, 2.0]
    assert manifest["capture"]["merged_points_file"] == "merged_points.npy"
    assert rdk.events[-1] == ("move-joints", tej.START_JOINTS)         # still returns home


def test_a_dropped_star_view_still_completes_the_take_on_the_rest(tmp_path, monkeypatch):
    """Only the TOP view failing may fail the take (spec section 8)."""
    svc, rdk, camera = _multiview_env(tmp_path, monkeypatch)
    original_move = measure_mod._move_to_inspection

    def flaky_move(services, ctx, plan, layer, *, inspection_name, **kwargs):
        if inspection_name.endswith("_star120"):
            raise RuntimeError("simulated: no reachable pose")
        return original_move(services, ctx, plan, layer,
                             inspection_name=inspection_name, **kwargs)

    monkeypatch.setattr(measure_mod, "_move_to_inspection", flaky_move)
    plan = tem.auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    ctx = tem.Ctx()
    out = RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True,
                         multiview=True)(ctx)
    assert camera.grabs == 1 + 3                    # readiness + top/star-000/star-240 only
    layer_dir = Path(out["layer_dir"])
    assert not (layer_dir / "views" / "star-120").exists()
    assert (layer_dir / "views" / "star-000").exists()
    manifest = json.loads((layer_dir / "manifest.json").read_text())
    views_by_name = {v["name"]: v for v in manifest["capture"]["views"]}
    assert views_by_name["star-120"]["dropped"] is True
    assert "simulated" in views_by_name["star-120"]["drop_reason"]
    assert views_by_name["star-000"]["dropped"] is False
    assert any("star-120" in message and "dropped" in message for message in ctx.logs)


def test_a_failing_top_view_fails_the_whole_take(tmp_path, monkeypatch):
    svc, rdk, camera = _multiview_env(tmp_path, monkeypatch)
    original_move = measure_mod._move_to_inspection

    def flaky_move(services, ctx, plan, layer, *, inspection_name, **kwargs):
        if inspection_name.endswith("_Inspect"):
            raise RuntimeError("simulated: top view unreachable")
        return original_move(services, ctx, plan, layer,
                             inspection_name=inspection_name, **kwargs)

    monkeypatch.setattr(measure_mod, "_move_to_inspection", flaky_move)
    plan = tem.auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    with pytest.raises(RuntimeError, match="simulated: top view unreachable"):
        RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True,
                       multiview=True)(tem.Ctx())
    assert rdk.events[-1] == ("move-joints", tej.START_JOINTS)         # still comes home


def test_a_star_take_that_fails_processing_still_archives_the_top_frame_and_returns_home(
        tmp_path, monkeypatch):
    """Distinct from the top-VIEW-capture failing: here every view is captured
    fine but the merged cloud fails process_points (e.g. not enough deposit
    points) -- the take must still fail loudly, archive the irreplaceable top
    RGB-D exactly as the single-view path does, and the arm must still return."""
    svc, rdk, camera = _multiview_env(tmp_path, monkeypatch)

    def failing_process_points(points, **kwargs):
        raise RuntimeError("simulated: not enough deposited-geometry points")

    monkeypatch.setattr(measure_mod, "process_points", failing_process_points)
    plan = tem.auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    with pytest.raises(RuntimeError, match="raw RGB-D archived"):
        RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True,
                       multiview=True)(tem.Ctx())
    layer_dir = session.trial_dir / "layer-001"
    assert (layer_dir / "depth.npy").is_file() and (layer_dir / "color.png").is_file()
    assert "simulated" in (layer_dir / "report.json").read_text()
    manifest = json.loads((layer_dir / "manifest.json").read_text())
    assert manifest["capture"] is None       # never built once processing failed
    assert not (layer_dir / "merged_points.npy").exists()
    assert rdk.events[-1] == ("move-joints", tej.START_JOINTS)         # still comes home


def test_repeats_visits_each_star_pose_once_not_once_per_repeat(tmp_path, monkeypatch):
    svc, rdk, camera = _multiview_env(tmp_path, monkeypatch)
    plan = tem.auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    out = RingMeasureJob(svc, plan, session, 1, check_collisions=False,
                         multiview=True, repeats=3)(tem.Ctx())
    starts = [e[1] for e in rdk.events if e[0] == "start"]
    assert len(starts) == 4                          # 4 moves total, not 12
    assert out["takes_recorded"] == [1, 2, 3]
    assert camera.grabs == 1 + 4 * 3                  # readiness + 4 views x 3 frames each
    for suffix in ("layer-001", "layer-001-take02", "layer-001-take03"):
        assert (session.trial_dir / suffix / "merged_points.npy").is_file()
        assert (session.trial_dir / suffix / "views" / "star-000" / "color.png").is_file()


def test_multiview_off_default_follows_config(tmp_path, monkeypatch):
    """multiview=None resolves from config.extrusion.multiview_enabled, exactly
    like side_photo already does."""
    svc, rdk, camera = _multiview_env(tmp_path, monkeypatch)
    svc.config.extrusion.multiview_enabled = True
    plan = tem.auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    out = RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True)(tem.Ctx())
    assert Path(out["layer_dir"]).joinpath("merged_points.npy").is_file()


# ---------------------------------------------------- end-to-end: the real colour gate

def test_end_to_end_star_merge_survives_the_real_chroma_gate():
    """Task 5 review finding: merge tests hand-set ``chroma_gated`` on a ViewCloud,
    which never actually exercises chroma_gate_mask. This renders each view with
    BOTH syn.render_scene (depth) and syn.render_color (colour), pushes them
    through the real observation_points (which calls chroma_gate_mask for real),
    and merges the result -- so a synthetic view that fails the real gate would
    be silently dropped and this test would exercise only the top-only fallback.
    """
    pytest.importorskip("open3d")
    import geometry_fixtures as gf
    from tasni.modules.extrusion.processing import observation_points, process_points

    plan = tem.scene_plan(radius=60.0, bead=8.0)
    layer = plan.layers[0]
    config = ExtrusionConfig()
    aim = np.array([plan.setup.center_x_mm, plan.setup.center_y_mm, 6.0])
    rings = [syn.RingSpec(60.0, 8.0, (plan.setup.center_x_mm, plan.setup.center_y_mm),
                          height_fn=syn.flat(6.0))]
    geom = gf.aligned(syn.K_720P, syn.SIZE_720P)

    clouds = []
    for name, tilt, azimuth in star_view_angles(config):
        T = pose_from_aim(aim, 300.0, tilt_deg=tilt, azimuth_deg=azimuth,
                          reference_x=syn.CAMERA_X_AT_PARK)
        depth = syn.render_scene(rings, T, plane_center_xy_mm=(
            plan.setup.center_x_mm, plan.setup.center_y_mm), seed=hash(name) % 1000)
        # The levelling annulus (r 90-150 mm here) sits OUTSIDE the ring, on bare
        # board -- render_color's default board is a flat, zero-saturation grey
        # with no noise, so the real chroma gate would remove every point out
        # there and level_points would starve (0 of the needed 500). Give the
        # board real chroma too (still a different hue from the ring) so the
        # gate has genuine per-pixel saturation to threshold, same as a real
        # frame's board is not perfectly achromatic either.
        color = syn.render_color(rings, T, board_bgr=(120, 160, 210))
        points, gated = observation_points(color=color, depth=depth, geometry=geom,
                                           T_work_camera=T, K=syn.K_720P, dist=None,
                                           config=config)
        assert gated is True, f"{name}: the real chroma gate abstained on synthetic colour"
        clouds.append(ViewCloud(name=name, points=points, chroma_gated=gated,
                                tilt_deg=tilt, azimuth_deg=azimuth, T_work_camera=T))

    merged = merge_views(clouds, plan=plan, layer=layer, config=config)
    # The views must have genuinely survived registration, not fallen back to
    # top-only -- that would silently exercise the degenerate path instead.
    assert len(merged.used) > 1, merged.dropped
    assert "top" in merged.used
    assert merged.chroma_gated is True

    processed = process_points(merged.points, plan=plan, layer=layer, config=config,
                               chroma_gated=merged.chroma_gated)
    assert processed.metrics.valid, processed.metrics.warnings
    assert abs(processed.metrics.measured_radius_mm - 60.0) < 1.5
    assert processed.metrics.center_offset_norm_mm < 1.5


# --------------------------------------- RingCharacterizeJob(multiview=...) attribute

from tasni.modules.extrusion.measure import RingCharacterizeJob  # noqa: E402


def test_characterize_job_resolves_multiview_like_side_photo(tmp_path, monkeypatch):
    """API symmetry with RingMeasureJob (spec 5.8): explicit True/False win,
    None follows config.extrusion.multiview_enabled -- exactly like side_photo."""
    svc, rdk, camera = tem.measure_env(tmp_path, monkeypatch)
    plan = tem.auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    assert RingCharacterizeJob(svc, plan, session).multiview is False
    svc.config.extrusion.multiview_enabled = True
    assert RingCharacterizeJob(svc, plan, session).multiview is True
    assert RingCharacterizeJob(svc, plan, session, multiview=False).multiview is False
