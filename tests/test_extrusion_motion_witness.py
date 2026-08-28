"""Did the ARM move, or only RoboDK's model?

RoboDK's model is not a witness to its own error: on the cell it advanced to the
target while the controller never executed the motion. The camera is bolted to
the flange, so if the arm moves the view MUST change -- that makes it an
independent witness, and it costs one colour grab either side of the program.

This is what turns "RoboDK predicted 2.2 s and it finished in 0.5 s" from a
suspicion into a statement: view unchanged => the arm did not move; view changed
=> it did, and the duration estimate was simply wrong.

    py -3.10 -m pytest tests/test_extrusion_motion_witness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.modules.extrusion.service import view_changed  # noqa: E402

RNG = np.random.default_rng(0)


def _scene(seed=0):
    r = np.random.default_rng(seed)
    return r.integers(0, 255, (720, 1280, 3), dtype=np.uint8)


def test_identical_frames_mean_the_arm_did_not_move():
    frame = _scene()
    assert view_changed(frame, frame.copy()) is False


def test_sensor_noise_alone_is_not_motion():
    frame = _scene().astype(np.int16)
    noisy = np.clip(frame + RNG.integers(-3, 4, frame.shape), 0, 255).astype(np.uint8)
    assert view_changed(frame.astype(np.uint8), noisy) is False


def test_a_shifted_view_is_motion():
    frame = _scene()
    assert view_changed(frame, np.roll(frame, 120, axis=1)) is True


def test_a_different_scene_is_motion():
    assert view_changed(_scene(1), _scene(2)) is True


def test_missing_frames_are_inconclusive_not_a_verdict():
    """Never claim the arm did not move just because we failed to look."""
    assert view_changed(None, _scene()) is None
    assert view_changed(_scene(), None) is None


def test_mismatched_shapes_are_inconclusive():
    assert view_changed(_scene(), np.zeros((8, 8, 3), np.uint8)) is None
