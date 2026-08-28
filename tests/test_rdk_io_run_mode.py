"""Program dispatch asserts station mode before either supported start command.

Both the run type and station run mode are asserted immediately before execution.
Cell measurement on 2026-08-28 proved that mode 6 and run type 2 were necessary but
not sufficient for ``RunCode`` on RoboDK 6.0.5: it accepted the program and dispatched
nothing. The alternate item ``Start`` command moved the physical arm under the same
state, so real execution now uses Start while simulation keeps RunCode.

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
    def setParam(self, name, value): self.calls.append(("setParam", name, value)); return "OK"


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
    starts = [i for i, c in enumerate(rdk.calls) if c[0] == "setParam"]
    assert modes and starts and modes[-1] < starts[0], "run mode must be set BEFORE Start"
    assert ("RunCode",) not in rdk.calls


def test_simulated_run_forces_simulate_mode():
    io, rdk = _io()
    io.start_program("P", real_robot=False)
    assert ("setRunMode", rl.RUNMODE_SIMULATE) in rdk.calls
    assert ("setRunType", rl.PROGRAM_RUN_ON_SIMULATOR) in rdk.calls
    assert ("RunCode",) in rdk.calls


def test_run_type_is_still_set_so_the_gui_agrees_with_the_api():
    io, rdk = _io()
    io.start_program("P", real_robot=True)
    assert ("setRunType", rl.PROGRAM_RUN_ON_ROBOT) in rdk.calls
