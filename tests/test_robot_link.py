"""Real-robot driver link — RdkIO.connect_robot / robot_connected, the shared
link_real_robot helper, and the run-time ensure_real_robot_link gate. No RoboDK:
a tiny fake robot returns scripted ConnectedState/Connect results.

    py -3.10 tests/test_robot_link.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core.rdk_io import RdkIO, link_real_robot  # noqa: E402
from tasni.modules.calibration.service import ensure_real_robot_link  # noqa: E402

READY, NOT_CONN, WAITING = 0, -1, 2          # robolink ROBOTCOM_* subset


class FakeRobot:
    """Returns the scripted ``ConnectedState`` results in order (repeating the last)."""

    def __init__(self, states):
        self._states = list(states)
        self.connect_calls = 0
        self.connect_args: list = []

    def ConnectedState(self):
        return self._states.pop(0) if len(self._states) > 1 else self._states[0]

    def Connect(self, ip="", blocking=True):
        self.connect_calls += 1
        self.connect_args.append((ip, blocking))
        return 1

    def ConnectionParams(self):
        return ("10.1.2.3", 7000, "/prog", "user", "pass")


def _io(robot) -> RdkIO:
    session = SimpleNamespace(rdk=SimpleNamespace(Item=lambda *a, **k: robot),
                              config=SimpleNamespace(robot_name="KUKA"))
    return RdkIO(session)


def _cfg(*, on=True, ip="", timeout=0.0):
    return SimpleNamespace(connect_robot_on_connect=on, robot_ip=ip,
                           robot_connect_timeout_s=timeout)


def test_already_ready_skips_connect():
    robot = FakeRobot([(READY, "ROBOTCOM_READY")])
    ready, msg = _io(robot).connect_robot(timeout_s=5.0)
    assert ready and msg == "ROBOTCOM_READY"
    assert robot.connect_calls == 0          # idempotent: no driver call when already up


def test_connects_then_becomes_ready():
    robot = FakeRobot([(NOT_CONN, "down"), (WAITING, "..."), (READY, "ok")])
    ready, msg = _io(robot).connect_robot(timeout_s=2.0, poll_s=0.01)
    assert ready and msg == "ok"
    assert robot.connect_calls == 1
    assert robot.connect_args[0][1] is False  # initiated non-blocking, we polled


def test_offline_times_out():
    robot = FakeRobot([(NOT_CONN, "controller off")])
    ready, msg = _io(robot).connect_robot(timeout_s=0.0)   # deadline now -> one poll
    assert not ready and "off" in msg
    assert robot.connect_calls == 1


def test_connected_state_never_raises():
    class Boom:
        def ConnectedState(self): raise RuntimeError("no driver")
    ready, msg = _io(Boom()).robot_connected()
    assert not ready and "no driver" in msg


def test_ensure_raises_when_offline():
    robot = FakeRobot([(NOT_CONN, "unreachable")])
    msg = ""
    try:
        ensure_real_robot_link(_io(robot), _cfg(on=True))
        raise AssertionError("expected an offline RuntimeError")
    except RuntimeError as e:
        msg = str(e)
    assert "offline" in msg and "10.1.2.3" in msg     # surfaces the configured IP


def test_ensure_passes_when_ready_and_logs():
    robot = FakeRobot([(READY, "ROBOTCOM_READY")])
    logs: list[str] = []
    ensure_real_robot_link(_io(robot), _cfg(on=True), log=logs.append)
    assert any("linked" in m for m in logs)


def test_ensure_noop_when_disabled():
    robot = FakeRobot([(NOT_CONN, "off")])      # offline, but auto-connect disabled
    ensure_real_robot_link(_io(robot), _cfg(on=False))   # must NOT raise
    assert robot.connect_calls == 0


def test_link_real_robot_summary():
    assert link_real_robot(_io(FakeRobot([(READY, "")])), _cfg(on=False)) is None
    out = link_real_robot(_io(FakeRobot([(READY, "ready")])), _cfg(on=True))
    assert out["connected"] and out["ip"] == "10.1.2.3" and out["configured"] is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("All robot-link checks passed.")
