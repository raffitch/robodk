"""API-driven RunCode() follows the STATION run mode, not the program's run type.

RoboDK's own docs draw the line:

  setRunType  -- "specify if a program made using the GUI will be run in
                  simulation mode or on the real robot"
  setRunMode(RUNMODE_RUN_ROBOT) -- "moves the real robot from the PC"

So right-clicking Run in the GUI honours the run TYPE (which is why that worked
on the cell), while our RunCode() honours the run MODE. Anything that simulates
in between -- Program.Update() validation, collision screening -- can leave the
station in RUNMODE_SIMULATE, and then RunCode() moves only the model: the arm
never budges, the program "finishes" in a fraction of a second, and nothing is
deposited. Observed on the cell 2026-08-28.

    py -3.10 -m pytest tests/test_rdk_io_run_mode.py
"""
from __future__ import annotations

from types import SimpleNamespace

import robolink as rl

from tasni.core.config import RoboDKConfig
from tasni.core.rdk_io import RdkIO


class _Program:
    def __init__(self, calls): self.calls = calls
    def Valid(self): return True
    def setRunType(self, t): self.calls.append(("setRunType", t))
    def RunCode(self): self.calls.append(("RunCode",)); return 0


class _Rdk:
    def __init__(self):
        self.calls: list = []
        self.program = _Program(self.calls)
    def Item(self, name, itemtype=None): return self.program
    def setRunMode(self, mode): self.calls.append(("setRunMode", mode))


def _io():
    rdk = _Rdk()
    return RdkIO(SimpleNamespace(rdk=rdk, config=RoboDKConfig())), rdk


def test_real_robot_run_forces_run_robot_mode_before_running():
    io, rdk = _io()
    io.start_program("P", real_robot=True)
    assert ("setRunMode", rl.RUNMODE_RUN_ROBOT) in rdk.calls
    modes = [i for i, c in enumerate(rdk.calls) if c[0] == "setRunMode"]
    runs = [i for i, c in enumerate(rdk.calls) if c[0] == "RunCode"]
    assert modes and runs and modes[-1] < runs[0], "run mode must be set BEFORE RunCode"


def test_simulated_run_forces_simulate_mode():
    io, rdk = _io()
    io.start_program("P", real_robot=False)
    assert ("setRunMode", rl.RUNMODE_SIMULATE) in rdk.calls
    assert ("setRunType", rl.PROGRAM_RUN_ON_SIMULATOR) in rdk.calls


def test_run_type_is_still_set_so_the_gui_agrees_with_the_api():
    io, rdk = _io()
    io.start_program("P", real_robot=True)
    assert ("setRunType", rl.PROGRAM_RUN_ON_ROBOT) in rdk.calls
