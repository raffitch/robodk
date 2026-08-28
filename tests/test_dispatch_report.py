"""A dispatch must report which RoboDK start path ran and what it returned.

Cell 2026-08-28: a layer program was "accepted", never observed busy, and the arm
never moved. RoboDK documents ``RunCode()`` as returning "the number of instructions
that can be executed successfully (a quick program check is performed before the
program starts)", and the job rejected only ``< 0`` -- so a **0**, meaning the check
cleared nothing, passed as success and the run continued into measuring an empty
board. Cell bisect then proved that the documented item ``Start`` command moved the
physical arm while ``RunCode`` did not. Real dispatch now uses item Start; simulation
keeps RunCode and its quick-check count.

    py -3.10 -m pytest tests/test_dispatch_report.py
"""
from __future__ import annotations

from types import SimpleNamespace

import robolink as rl

from tasni.core.config import RoboDKConfig
from tasni.core.rdk_io import RdkIO
from tasni.modules.extrusion.service import (describe_dispatch, dispatch_started,
                                             describe_driver_states)


class _Program:
    def __init__(self, calls, run_code=7, instructions=7, start_result="OK"):
        self.calls, self.run_code, self.instructions = calls, run_code, instructions
        self.start_result = start_result
        self.busy = False

    def Valid(self): return True
    def setRunType(self, t): self.calls.append(("setRunType", t))
    def InstructionCount(self): return self.instructions
    def Busy(self): return self.busy

    def RunCode(self):
        self.calls.append(("RunCode",))
        return self.run_code

    def setParam(self, name, value):
        self.calls.append(("setParam", name, value))
        return self.start_result


class _Rdk:
    def __init__(self, run_code=7, instructions=7, mode_sticks=True,
                 start_result="OK"):
        self.calls: list = []
        self.program = _Program(self.calls, run_code, instructions, start_result)
        self.mode = rl.RUNMODE_SIMULATE
        self.mode_sticks = mode_sticks

    def Item(self, name, itemtype=None): return self.program

    def setRunMode(self, mode):
        self.calls.append(("setRunMode", mode))
        if self.mode_sticks:
            self.mode = mode

    def RunMode(self): return self.mode


def _io(**kw):
    rdk = _Rdk(**kw)
    return RdkIO(SimpleNamespace(rdk=rdk, config=RoboDKConfig())), rdk


def test_real_dispatch_uses_item_start_and_reports_mode_readback():
    io, _ = _io(run_code=7, instructions=7)
    report = io.dispatch_program("P", real_robot=True)
    assert report["run_code"] is None
    assert report["started"] is True
    assert report["start_method"] == "item_start"
    assert report["start_result"] == "OK"
    assert report["instruction_count"] == 7
    assert report["run_mode"] == rl.RUNMODE_RUN_ROBOT
    assert report["run_mode_expected"] == rl.RUNMODE_RUN_ROBOT


def test_run_mode_is_read_back_not_assumed():
    """A station that silently refuses RUN_ROBOT must be visible in the report."""
    io, _ = _io(mode_sticks=False)
    report = io.dispatch_program("P", real_robot=True)
    assert report["run_mode"] == rl.RUNMODE_SIMULATE
    assert report["run_mode"] != report["run_mode_expected"]


def test_start_program_returns_zero_for_confirmed_real_item_start():
    io, _ = _io(run_code=4)
    assert io.start_program("P", real_robot=True) == 0


def test_start_program_returns_negative_when_item_start_is_not_confirmed():
    io, _ = _io(start_result="ERROR")
    assert io.start_program("P", real_robot=True) == -1


def test_dispatch_sets_run_mode_before_item_start():
    io, rdk = _io()
    io.dispatch_program("P", real_robot=True)
    modes = [i for i, c in enumerate(rdk.calls) if c[0] == "setRunMode"]
    runs = [i for i, c in enumerate(rdk.calls) if c[0] == "setParam"]
    assert modes and runs and modes[-1] < runs[0]


def test_diagnostics_never_break_a_dispatch_on_a_build_that_lacks_them():
    """RunMode()/InstructionCount() must not be able to stop the robot running."""
    io, rdk = _io()
    rdk.RunMode = lambda: (_ for _ in ()).throw(RuntimeError("no such command"))
    rdk.program.InstructionCount = lambda: (_ for _ in ()).throw(RuntimeError("nope"))
    report = io.dispatch_program("P", real_robot=True)
    assert report["started"] is True and report["start_result"] == "OK"
    assert report["run_mode"] is None and report["instruction_count"] is None


def test_run_station_program_returns_the_report_as_the_simple_case_control():
    io, rdk = _io(run_code=2, instructions=2)
    rdk.program.busy = False
    report = io.run_station_program("AirOff", real_robot=True)
    assert report["started"] is True and report["start_result"] == "OK"


def test_simulation_keeps_runcode_and_its_instruction_count():
    io, _ = _io(run_code=4, instructions=7)
    report = io.dispatch_program("P", real_robot=False)
    assert report["start_method"] == "run_code"
    assert report["run_code"] == 4 and report["started"] is True


# -- the log line -----------------------------------------------------------

def test_zero_instructions_cleared_is_called_out_as_a_refusal():
    line = describe_dispatch({"run_code": 0, "instruction_count": 12,
                              "run_mode": 6, "run_mode_expected": 6})
    assert "ZERO" in line and "refused" in line


def test_a_healthy_dispatch_is_not_called_a_refusal():
    line = describe_dispatch({"run_code": 12, "instruction_count": 12,
                              "run_mode": 6, "run_mode_expected": 6})
    assert "refused" not in line and "12" in line


def test_a_run_mode_that_did_not_stick_is_called_out():
    line = describe_dispatch({"run_code": 12, "instruction_count": 12,
                              "run_mode": 1, "run_mode_expected": 6})
    assert "NOT" in line and "1" in line


def test_an_empty_program_returning_zero_is_not_flagged_as_refused():
    line = describe_dispatch({"run_code": 0, "instruction_count": 0,
                              "run_mode": 6, "run_mode_expected": 6})
    assert "refused" not in line


def test_unreadable_run_mode_is_stated_not_guessed():
    line = describe_dispatch({"run_code": 5, "instruction_count": 5,
                              "run_mode": None, "run_mode_expected": 6})
    assert "unreadable" in line


def test_item_start_log_reports_the_command_roboDK_actually_confirmed():
    report = {"started": True, "start_method": "item_start", "start_result": "OK",
              "run_code": None, "instruction_count": 195,
              "run_mode": 6, "run_mode_expected": 6}
    line = describe_dispatch(report)
    assert "item Start" in line and "'OK'" in line and "195" in line


def test_dispatch_started_accepts_legacy_zero_but_honours_new_started_flag():
    assert dispatch_started({"run_code": 0}) is True
    assert dispatch_started({"run_code": -1}) is False
    assert dispatch_started({"started": False, "run_code": 12}) is False


# -- the driver's own state -------------------------------------------------

def test_driver_state_names_the_robotcom_code():
    io, _ = _io()
    io.robot = lambda: SimpleNamespace(ConnectedState=lambda: (1, ""))
    assert io.driver_state() == {"code": 1, "name": "WORKING", "message": ""}


def test_driver_state_carries_the_controller_message_on_problems():
    io, _ = _io()
    io.robot = lambda: SimpleNamespace(
        ConnectedState=lambda: (-3, "Emergency stop"))
    state = io.driver_state()
    assert state["name"] == "PROBLEMS" and state["message"] == "Emergency stop"


def test_driver_state_never_raises():
    io, _ = _io()
    io.robot = lambda: (_ for _ in ()).throw(RuntimeError("no driver"))
    assert io.driver_state()["name"] == "UNREADABLE"


def test_a_driver_that_never_left_ready_was_never_asked_to_work():
    """The decisive reading: RoboDK accepted the program and dispatched nothing."""
    line = describe_driver_states([{"code": 0, "name": "READY", "message": ""}])
    assert "never asked to work" in line and "dispatched nothing" in line


def test_a_driver_that_worked_is_not_called_out():
    line = describe_driver_states([
        {"code": 0, "name": "READY", "message": ""},
        {"code": 1, "name": "WORKING", "message": ""},
        {"code": 0, "name": "READY", "message": ""}])
    assert "never asked" not in line and "WORKING" in line


def test_a_driver_problem_surfaces_the_controller_message():
    line = describe_driver_states([
        {"code": 0, "name": "READY", "message": ""},
        {"code": -3, "name": "PROBLEMS", "message": "KSS01234 drives off"}])
    assert "KSS01234" in line


# -- the poll cadence -------------------------------------------------------

def test_the_first_polls_are_fast_enough_to_catch_a_program_that_aborts():
    """At a flat 50 ms, a run that dies in 20 ms reads as 'never started'.

    That reading is what the whole 2026-08-28 diagnosis rests on, so it has to be
    trustworthy: poll hard at first, then back off.
    """
    from tasni.modules.extrusion.service import _wait_program

    slept: list[float] = []

    class _Ctx:
        cancelled = False
        def check_cancel(self): pass

    class _Rdk:
        def program_busy(self, name): return False
        def stop_program(self, name): pass

    _wait_program(_Ctx(), _Rdk(), "P", start_timeout_s=0.2,
                  sleep=slept.append)
    assert slept, "must have polled at least once"
    assert max(slept[:10]) <= 0.01, "the first polls must be sub-10 ms"
