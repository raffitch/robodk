"""Cross-check the commanded inspection standoff against the measured depth.

Cell failure 2026-08-28: five consecutive live prints failed with "not enough
deposited-geometry points inside the configured work ROI". The scan, work frame,
hand-eye chain and depth were all verified correct; the fault was that at the
moment of capture the arm was ~142 mm from where RoboDK's model said it was, so
every back-projected point was displaced by that amount. The depth frame itself
was perfect, which is exactly why it stayed silent for five runs.

The job already knows both numbers -- the distance it commanded and the distance
the camera reports -- so disagreeing by more than a tolerance is a hard error.

    py -3.10 -m pytest tests/test_extrusion_standoff.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core.config import ExtrusionConfig  # noqa: E402
from tasni.modules.extrusion.inspection import standoff_fault, standoff_report  # noqa: E402


def _T(camera_xyz):
    T = np.eye(4)
    T[:3, 3] = camera_xyz
    return T


def _depth(value_mm, shape=(720, 1280)):
    return np.full(shape, float(value_mm))


def test_report_matches_when_the_arm_is_where_it_was_commanded():
    # camera 330 mm above the aim point, and the camera sees 330 mm
    r = standoff_report(_T([0.0, 0.0, 330.0]), [0.0, 0.0, 0.0], _depth(330.0))
    assert r["expected_mm"] == 330.0
    assert r["measured_mm"] == 330.0
    assert r["delta_mm"] == 0.0


def test_report_measures_the_cell_failure_signature():
    """Commanded 326 mm, camera saw 468 mm -> the arm was 142 mm further away."""
    r = standoff_report(_T([0.0, 0.0, 326.0]), [0.0, 0.0, 0.0], _depth(468.0))
    assert r["expected_mm"] == 326.0
    assert r["measured_mm"] == 468.0
    assert r["delta_mm"] == 142.0


def test_zero_depth_pixels_are_ignored():
    d = _depth(0.0)
    d[300:400, 600:700] = 400.0
    r = standoff_report(_T([0.0, 0.0, 400.0]), [0.0, 0.0, 0.0], d)
    assert r["measured_mm"] == 400.0
    assert r["samples"] > 0


def test_no_fault_within_tolerance():
    cfg = ExtrusionConfig()
    r = standoff_report(_T([0.0, 0.0, 330.0]), [0.0, 0.0, 0.0], _depth(340.0))
    assert standoff_fault(r, cfg.inspection_standoff_tolerance_mm) is None


def test_fault_names_the_discrepancy_and_the_likely_cause():
    cfg = ExtrusionConfig()
    r = standoff_report(_T([0.0, 0.0, 326.0]), [0.0, 0.0, 0.0], _depth(468.0))
    msg = standoff_fault(r, cfg.inspection_standoff_tolerance_mm)
    assert msg is not None
    assert "142" in msg and "326" in msg and "468" in msg


def test_a_depth_frame_with_no_valid_samples_is_a_fault_not_a_pass():
    cfg = ExtrusionConfig()
    r = standoff_report(_T([0.0, 0.0, 326.0]), [0.0, 0.0, 0.0], _depth(0.0))
    assert standoff_fault(r, cfg.inspection_standoff_tolerance_mm) is not None


def test_tolerance_default_is_tight_enough_to_have_caught_the_cell_failure():
    assert 0 < ExtrusionConfig().inspection_standoff_tolerance_mm < 142


def test_report_scales_raw_depth_words_by_the_frames_own_unit():
    """Task 9 review, Critical 1: ``depth`` is raw camera WORDS, not millimetres.

    Protocol 2 native depth is 0.1 mm/word. Left at the unit-blind default this
    reads a real ~312 mm standoff as ~3120 mm and raises a standoff fault on
    every layer, naming a robot-arrival problem that was never there.
    """
    depth_words = _depth(3120, shape=(8, 8))          # 3120 words * 0.1 mm/word = 312 mm

    scaled = standoff_report(_T([0.0, 0.0, 312.0]), [0.0, 0.0, 0.0], depth_words,
                             unit_mm=0.1)
    assert scaled["measured_mm"] == 312.0
    assert scaled["delta_mm"] == 0.0

    # The old, unit-blind call (no unit_mm) reads the same words as 3120 mm --
    # this is the regression the review measured on the checkout.
    unscaled = standoff_report(_T([0.0, 0.0, 312.0]), [0.0, 0.0, 0.0], depth_words)
    assert unscaled["measured_mm"] == 3120.0
    assert unscaled["delta_mm"] == 2808.0
