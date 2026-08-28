"""Waiting for a program that runs on the REAL robot.

``_wait_program`` was ``while rdk.program_busy(name): sleep(0.05)``. If the first
poll lands before RoboDK marks the program busy, the loop body never runs and it
returns immediately -- the caller then reads a pose from the model while the arm
has not moved. A 99 s print program wins that race; a short inspection move
dispatched with PROGRAM_RUN_ON_ROBOT can lose it. On the cell 2026-08-28 that put
the model 155 mm from the arm and silently displaced every measured point.

    py -3.10 -m pytest tests/test_extrusion_wait.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.modules.extrusion.service import _wait_program  # noqa: E402


class Ctx:
    cancelled = False
    def check_cancel(self): pass


class Rdk:
    """program_busy returns each scripted value in turn, then False forever."""
    def __init__(self, script):
        self.script = list(script)
        self.polls = 0
    def program_busy(self, name):
        self.polls += 1
        return self.script.pop(0) if self.script else False
    def stop_program(self, name): pass


def test_waits_for_a_program_that_has_not_started_yet():
    """The race: not-busy, not-busy, THEN busy, then done."""
    rdk = Rdk([False, False, True, True, False])
    _wait_program(Ctx(), rdk, "P", sleep=lambda s: None)
    # It must have kept polling past the initial not-busy readings and then
    # through the busy ones -- 5 polls minimum, not the 1 the old loop did.
    assert rdk.polls >= 5


def test_still_returns_when_the_program_never_reports_busy():
    """Must not hang forever if the driver never marks the program busy."""
    rdk = Rdk([])
    _wait_program(Ctx(), rdk, "P", start_timeout_s=0.05, sleep=lambda s: None)
    assert rdk.polls >= 1


def test_returns_promptly_once_a_running_program_finishes():
    rdk = Rdk([True, True, True, False])
    _wait_program(Ctx(), rdk, "P", sleep=lambda s: None)
    assert rdk.polls >= 4
