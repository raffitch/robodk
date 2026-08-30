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
