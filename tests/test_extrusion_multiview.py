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
                                                star_view_candidates)


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
