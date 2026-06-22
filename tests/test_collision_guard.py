"""Unit test for RdkIO.ensure_mounted_tool_collision_pairs — the guard that
re-enables the tool<->arm collision pairs RoboDK excludes by default (a tool can't
collide with its own robot in the default map), which is why a flange spindle
swinging into a forearm link sailed through the generation filter.

Uses a fake ``robolink`` handle (the real ``robolink`` module is still imported for
its ITEM_TYPE_*/COLLISION_ON constants). No RoboDK process, no station.

    py -3.10 tests/test_collision_guard.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import robolink as rl  # noqa: E402  (installed; only constants are used)

from tasni.core.config import RoboDKConfig  # noqa: E402
from tasni.core.rdk_io import RdkIO  # noqa: E402

TOOL, OBJECT, FRAME = rl.ITEM_TYPE_TOOL, rl.ITEM_TYPE_OBJECT, rl.ITEM_TYPE_FRAME


class FakeMat:
    def __init__(self, n): self._n = n
    def Rows(self): return self._n              # robomath returns the row LISTS here
    def list(self):
        if self._n is None:
            raise ValueError("no joints")        # exercises robot_dof's fallback
        return [0.0] * self._n


class FakeItem:
    def __init__(self, itype, name, childs=(), dof=None):
        self._type, self._name, self._childs, self._dof = itype, name, list(childs), dof
    def Type(self): return self._type
    def Name(self): return self._name
    def Childs(self): return list(self._childs)
    def Joints(self): return FakeMat(self._dof)


class FakeRdk:
    """Fake Robolink handle. ItemList(TOOL) returns every tool in the 'station'
    (parentage-independent, like the real API); setCollisionActivePair records each
    call and returns 1 unless the item name is in ``fail``."""
    def __init__(self, robot, tools, fail=()):
        self._robot, self._tools, self._fail = robot, list(tools), set(fail)
        self.calls: list = []
    def Item(self, name, itemtype=0): return self._robot
    def ItemList(self, filter=None, list_names=False):
        if filter == TOOL:
            return list(self._tools)
        return []
    def setCollisionActivePair(self, state, item1, item2, id1=0, id2=0):
        self.calls.append((state, item1.Name(), item2.Name(), id1, id2))
        return 0 if item1.Name() in self._fail else 1


def _io(robot, tools, fail=()):
    rdk = FakeRdk(robot, tools, fail=fail)
    session = SimpleNamespace(rdk=rdk, config=RoboDKConfig())
    return RdkIO(session), rdk


def _cell(dof=6, spindle_child=True):
    """Build a robot subtree. cam is a flange tool with a 3D-model object child;
    a Fixture object hangs off the robot. The spindle is a TOOL that is optionally
    NOT a child of the robot (spindle_child=False) — the real-world case that broke
    discovery — but is always returned by ItemList(TOOL)."""
    cam = FakeItem(TOOL, "Realsense", childs=[FakeItem(OBJECT, "cam_model")])
    spindle = FakeItem(TOOL, "Spindle")
    frame = FakeItem(FRAME, "BaseFrame")            # must be ignored
    fixture = FakeItem(OBJECT, "Fixture")           # object child, included
    kids = [cam, frame, fixture] + ([spindle] if spindle_child else [])
    robot = FakeItem(rl.ITEM_TYPE_ROBOT, RoboDKConfig().robot_name, childs=kids, dof=dof)
    return robot, [cam, spindle]                    # ItemList(TOOL) sees both tools


def test_discovers_all_tools_and_subtree_objects():
    io, _ = _io(*_cell())
    names = set(it.Name() for it in io.mounted_tool_items())
    assert names == {"Realsense", "Spindle", "cam_model", "Fixture"}
    assert "BaseFrame" not in names                 # frames are not bodies
    print("[discover]", sorted(names))


def test_finds_spindle_even_when_not_a_robot_child():
    """The bug: the spindle wasn't a direct child of the robot, so robot.Childs()
    missed it and no pairs were enabled. ItemList(TOOL) finds it regardless."""
    io, _ = _io(*_cell(spindle_child=False))
    names = set(it.Name() for it in io.mounted_tool_items())
    assert "Spindle" in names                       # found via ItemList, not Childs
    print("[orphan spindle]", sorted(names))


def test_link_ids_skip_trailing_wrist_and_flange():
    io, rdk = _io(*_cell(dof=6))
    out = io.ensure_mounted_tool_collision_pairs(skip_trailing=2)
    assert out["dof"] == 6
    assert out["links"] == [0, 1, 2, 3, 4]          # 6-axis, skip A5+A6 -> base..A4
    assert set(out["tools"]) == {"Realsense", "Spindle", "cam_model", "Fixture"}
    assert out["pairs_enabled"] == 4 * 5 and out["pairs_failed"] == 0
    # every call targets the robot's links 0..4, with id1=0 for the tool body
    assert all(c[0] == rl.COLLISION_ON and c[2] == RoboDKConfig().robot_name
               and c[3] == 0 and c[4] in out["links"] for c in rdk.calls)
    print("[links] dof6 skip2 ->", out["links"], "pairs", out["pairs_enabled"])


def test_failed_pairs_excluded_from_guarded_names():
    io, _ = _io(*_cell(dof=6), fail={"Spindle"})
    out = io.ensure_mounted_tool_collision_pairs(skip_trailing=2)
    assert "Spindle" not in out["tools"]            # no pair stuck -> not reported
    assert out["pairs_failed"] == 5                 # 5 links rejected for the spindle
    assert out["pairs_enabled"] == 15               # the other 3 bodies x 5
    print("[partial]", out["tools"], "failed", out["pairs_failed"])


def test_skip_larger_than_dof_degrades_to_base_only():
    io, _ = _io(*_cell(dof=6))
    out = io.ensure_mounted_tool_collision_pairs(skip_trailing=99)
    assert out["links"] == [0]                      # never a negative/empty range
    print("[clamp] skip99 ->", out["links"])


def test_skip_one_keeps_the_wrist_link_guarded():
    """skip_trailing=1 keeps A5 (link 5) in range — the safety-tunable knob."""
    io, _ = _io(*_cell(dof=6))
    out = io.ensure_mounted_tool_collision_pairs(skip_trailing=1)
    assert out["links"] == [0, 1, 2, 3, 4, 5]
    print("[skip1] ->", out["links"])


def test_robot_dof_reads_joint_count():
    robot = FakeItem(rl.ITEM_TYPE_ROBOT, RoboDKConfig().robot_name, childs=[], dof=7)
    io, _ = _io(robot, [])
    assert io.robot_dof() == 7                        # real path: counts joint values
    print("[dof] reads", io.robot_dof())


def test_dof_falls_back_when_unreadable():
    robot = FakeItem(rl.ITEM_TYPE_ROBOT, RoboDKConfig().robot_name, childs=[], dof=None)
    io, _ = _io(robot, [])
    assert io.robot_dof() == 6                        # Joints().list() raises -> fallback 6
    print("[dof fallback] ->", io.robot_dof())


# --- the screen_collisions shape-check regression -------------------------------
# A reachable SolveIK result is a (DOF,1) Mat. Mat.Cols()/Rows() return LISTS, so
# the old `ik.Cols()==1 and ik.Rows()>=6` guard was always False -> MoveJ_Test
# never ran, nothing was ever dropped, and every target was stored cartesian (the
# spindle-into-A4 that slipped through). These lock in that the sweep now runs.
import numpy as np  # noqa: E402
import robodk.robomath as robomath  # noqa: E402


class FakeRobotIK:
    """Robot whose SolveIK returns a real (6,1) joint Mat and whose MoveJ_Test
    reports a configured colliding-pair count per candidate index (call order)."""
    def __init__(self, ncols):           # ncols[i] = colliding pairs for candidate i
        self._ncols = list(ncols)
        self._ik_calls = 0
        self.movej_tests = 0
    def Joints(self): return robomath.Mat([0.0] * 6)
    def setJoints(self, j): pass
    def SolveIK(self, pose, joints_approx=None, tool=None, reference=None):
        return robomath.Mat([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])    # reachable (6,1)
    def MoveJ_Test(self, j1, j2, step):
        n = self._ncols[self.movej_tests]
        self.movej_tests += 1
        return n


class FakeRdkIK:
    SIMULATE = 1
    def __init__(self, robot): self._robot = robot; self.mode = 6
    def Item(self, name, itemtype=0): return self._robot
    def RunMode(self): return self.mode
    def setRunMode(self, m): self.mode = int(m)
    def setCollisionActive(self, flag): return 0
    def Collisions(self): return 0          # resting-config backstop -> clear


def test_screen_collisions_runs_movej_test_and_drops_colliders():
    robot = FakeRobotIK(ncols=[0, 3, 0])   # candidate 1 collides (3 pairs)
    rdk = FakeRdkIK(robot)
    io = RdkIO(SimpleNamespace(rdk=rdk, config=RoboDKConfig()))
    poses = [np.eye(4) for _ in range(3)]

    mask, checked, joints = io.screen_collisions(poses)

    assert robot.movej_tests == 3          # MoveJ_Test ACTUALLY RAN per pose (the fix)
    assert checked is True
    assert mask == [True, False, True]     # the colliding candidate is dropped
    assert all(j is not None for j in joints)   # joint configs recorded -> JOINT targets
    print("[screen] MoveJ_Test ran", robot.movej_tests, "times, mask", mask)


def test_screen_collisions_unreachable_solution_is_none():
    """An unreachable SolveIK (1-element Mat) yields no stored joints and no
    MoveJ_Test for that pose."""
    class _Unreach(FakeRobotIK):
        def SolveIK(self, pose, joints_approx=None, tool=None, reference=None):
            return robomath.Mat([0])       # SolveIK's empty/no-solution form
    robot = _Unreach(ncols=[0])
    io = RdkIO(SimpleNamespace(rdk=FakeRdkIK(robot), config=RoboDKConfig()))

    mask, checked, joints = io.screen_collisions([np.eye(4)])
    assert robot.movej_tests == 0          # no solution -> no sweep
    assert joints == [None] and mask == [True]   # unjudgeable kept (filter, not gate)
    print("[screen] unreachable -> joints None, kept")


# --- the guard-ORDER regression -------------------------------------------------
# setCollisionActive(ON) rebuilds the default collision map (tool<->own-robot
# EXCLUDED), so the tool<->arm pairs MUST be (re)enabled AFTER checking is turned
# on. Enabling them before (as target generation used to) let them be wiped — the
# spindle<->A4 pose (target 12) then swept clean and became a target. This locks in
# that screen_collisions(guard_skip=...) enables the pairs only after COLLISION_ON.
class FakeRobotGuard(FakeRobotIK):
    def Name(self): return RoboDKConfig().robot_name
    def Childs(self): return []            # spindle is found via ItemList(TOOL)


class FakeRdkGuard(FakeRdkIK):
    def __init__(self, robot):
        super().__init__(robot)
        self.events: list = []             # ordered (op, ...) trace
        self._spindle = FakeItem(TOOL, "Spindle")
    def ItemList(self, filter=None, list_names=False):
        return [self._spindle] if filter == TOOL else []
    def setCollisionActive(self, flag):
        self.events.append(("active", int(flag)))
        return 0
    def setCollisionActivePair(self, state, item1, item2, id1=0, id2=0):
        self.events.append(("pair", item1.Name(), id2))
        return 1


def test_screen_collisions_enables_guard_pairs_after_checking_on():
    robot = FakeRobotGuard(ncols=[0, 0])
    rdk = FakeRdkGuard(robot)
    io = RdkIO(SimpleNamespace(rdk=rdk, config=RoboDKConfig()))

    mask, checked, joints = io.screen_collisions([np.eye(4), np.eye(4)], guard_skip=2)

    ops = [e[0] for e in rdk.events]
    first_on = ops.index("active")                 # COLLISION_ON happens first
    assert rdk.events[first_on] == ("active", 1)
    first_pair = ops.index("pair")
    assert first_pair > first_on, "pairs must be enabled AFTER checking is on"
    # Spindle guarded vs arm links 0..4 (skip_trailing=2 on a 6-axis arm).
    pair_links = sorted(e[2] for e in rdk.events if e[0] == "pair" and e[1] == "Spindle")
    assert pair_links == [0, 1, 2, 3, 4]
    assert checked is True and mask == [True, True]
    print("[guard order] COLLISION_ON then", len(pair_links), "Spindle pairs")


def test_screen_collisions_no_guard_when_skip_none():
    """Without guard_skip, screen_collisions touches no collision pairs (back-compat
    — the dry tour / other callers that manage their own pairs are unaffected)."""
    robot = FakeRobotGuard(ncols=[0])
    rdk = FakeRdkGuard(robot)
    io = RdkIO(SimpleNamespace(rdk=rdk, config=RoboDKConfig()))

    io.screen_collisions([np.eye(4)])              # guard_skip defaults to None
    assert not any(e[0] == "pair" for e in rdk.events)
    print("[guard order] no guard_skip -> no pair changes")


if __name__ == "__main__":
    test_discovers_all_tools_and_subtree_objects()
    test_finds_spindle_even_when_not_a_robot_child()
    test_link_ids_skip_trailing_wrist_and_flange()
    test_failed_pairs_excluded_from_guarded_names()
    test_skip_larger_than_dof_degrades_to_base_only()
    test_skip_one_keeps_the_wrist_link_guarded()
    test_robot_dof_reads_joint_count()
    test_dof_falls_back_when_unreadable()
    test_screen_collisions_runs_movej_test_and_drops_colliders()
    test_screen_collisions_unreachable_solution_is_none()
    test_screen_collisions_enables_guard_pairs_after_checking_on()
    test_screen_collisions_no_guard_when_skip_none()
    print("\nCollision-guard tests passed.")
