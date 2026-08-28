"""A program that "succeeds" without moving must not pass silently.

Cell 2026-08-28: start_program() returned success, the controller made a sound,
and the arm did not move. RoboDK's program never reported busy, so _wait_program
returned after its start grace and the job carried on as if the layer had
printed -- the first hint of trouble was an empty ROI two minutes later.

RoboDK already predicts each program's duration (update_program -> time_s). A
layer that should take seconds and "runs" in a fraction of one did not execute.

    py -3.10 -m pytest tests/test_extrusion_runtime.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.modules.extrusion.service import program_runtime_fault  # noqa: E402


def test_the_cell_failure_is_reported():
    """RoboDK predicted ~12 s; the program 'finished' in 1.0 s."""
    msg = program_runtime_fault(expected_s=12.0, actual_s=1.0, observed_busy=False)
    assert msg is not None
    assert "12" in msg and "1.0" in msg


def test_a_program_that_ran_full_length_is_fine():
    assert program_runtime_fault(expected_s=12.0, actual_s=12.4, observed_busy=True) is None


def test_slightly_fast_is_tolerated():
    """Predicted time is an estimate; only a gross shortfall means 'never ran'."""
    assert program_runtime_fault(expected_s=12.0, actual_s=9.0, observed_busy=True) is None


def test_short_programs_are_not_policed():
    """A genuinely brief program can finish before it is ever observed busy."""
    assert program_runtime_fault(expected_s=0.4, actual_s=0.05, observed_busy=False) is None


def test_no_prediction_available_is_not_a_fault():
    assert program_runtime_fault(expected_s=0.0, actual_s=1.0, observed_busy=False) is None


def test_message_points_at_the_controller_not_the_software():
    msg = program_runtime_fault(expected_s=20.0, actual_s=0.8, observed_busy=False)
    assert "did not execute" in msg.lower() or "did not run" in msg.lower()
    assert "drives" in msg.lower() or "mode" in msg.lower()


def test_camera_confirming_motion_clears_a_timing_suspicion():
    """RoboDK's duration is an estimate. If the flange camera saw the view change,
    the arm moved and a short runtime is just a bad prediction -- not a fault."""
    assert program_runtime_fault(expected_s=12.0, actual_s=1.0,
                                 observed_busy=False, arm_moved=True) is None


def test_camera_confirming_no_motion_is_stated_as_fact():
    msg = program_runtime_fault(expected_s=12.0, actual_s=1.0,
                                observed_busy=False, arm_moved=False)
    assert msg is not None
    assert "camera" in msg.lower()
    assert "did not move" in msg.lower()


def test_without_a_witness_the_wording_stays_a_suspicion():
    msg = program_runtime_fault(expected_s=12.0, actual_s=1.0,
                                observed_busy=False, arm_moved=None)
    assert msg is not None and "camera" not in msg.lower()
