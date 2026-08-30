"""Multi-view ring capture: poses, gates, levelling, the joint circle solve, merge."""
from __future__ import annotations

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
