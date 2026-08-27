"""SolveIK must receive poses in the ROBOT BASE frame, whatever frame is active.

robolink's SolveIK does its pose math client-side against the robot base
("pose must be the robot flange with respect to the robot base unless you
provide the tool and/or reference"); it ignores the station's active reference
frame. use_named_tool_frame() activates an arbitrary work frame, so poses built
in that frame (extrusion inspection, reachability sampling) must be converted
before the solve — otherwise the joint target places the camera translated by
the full base->work offset.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import robodk.robomath as robomath
import robolink as rl

from tasni.core.config import RoboDKConfig
from tasni.core.rdk_io import RdkIO, pose_to_T


class FakeTool:
    def Valid(self): return True
    def PoseTool(self): return robomath.xyzrpw_2_pose([0, 0, 150, 0, 0, 0])


class FakeFrame:
    def __init__(self, pose_abs): self._pose = pose_abs
    def Valid(self): return True
    def Type(self): return rl.ITEM_TYPE_FRAME
    def PoseAbs(self): return self._pose


class FakeTarget:
    def setPose(self, pose): self.pose = pose
    def setAsJointTarget(self): self.joint_target = True
    def setJoints(self, joints): self.joints = joints


class FakeMissing:
    def Valid(self): return False


class FakeRobot:
    def __init__(self, base): self._base = base; self.ik_poses = []
    def Parent(self): return self._base
    def setPoseTool(self, tool): pass
    def setPoseFrame(self, frame): pass
    def Joints(self): return robomath.Mat([0.0] * 6)
    def setJoints(self, joints): pass
    def SolveIK(self, pose, joints_approx=None, tool=None, reference=None):
        self.ik_poses.append(pose)
        return robomath.Mat([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    def SolveIK_All(self, pose, tool=None, reference=None):
        self.ik_poses.append(pose)
        return robomath.Mat([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]])
    def JointsConfig(self, joints): return robomath.Mat([0.0, 0.0, 0.0])


class FakeRdk:
    def __init__(self, robot, items): self._robot = robot; self._items = items
    def Item(self, name, itemtype=0):
        return self._items.get((name, itemtype),
                               self._items.get(name, FakeMissing()))
    def AddTarget(self, name, frame, robot): return FakeTarget()


def _cell(base_xyz, frame_xyz):
    """Pure-translation frames so the expected pose needs no Euler conventions."""
    base = FakeFrame(robomath.xyzrpw_2_pose(list(base_xyz) + [0, 0, 0]))
    robot = FakeRobot(base)
    frame = FakeFrame(robomath.xyzrpw_2_pose(list(frame_xyz) + [0, 0, 0]))
    cfg = RoboDKConfig()
    items = {cfg.robot_name: robot,
             ("Realsense", rl.ITEM_TYPE_TOOL): FakeTool(),
             ("Work", rl.ITEM_TYPE_FRAME): frame}
    return RdkIO(SimpleNamespace(rdk=FakeRdk(robot, items), config=cfg)), robot


def test_named_frame_pose_reaches_solveik_in_base_coords():
    io, robot = _cell(base_xyz=(0, 0, 500), frame_xyz=(1000, 200, 500))
    io.use_named_tool_frame("Realsense", "Work")
    T_work = np.eye(4); T_work[:3, 3] = [10.0, 20.0, 300.0]

    joints = io.solve_joints_for_pose(T_work)

    assert joints is not None
    got = pose_to_T(robot.ik_poses[0])
    expected = np.eye(4); expected[:3, 3] = [1010.0, 220.0, 300.0]
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_base_frame_pose_is_passed_through_unchanged():
    io, robot = _cell(base_xyz=(0, 0, 500), frame_xyz=(1000, 200, 500))
    io.use_camera_tool("Realsense")     # adopts the robot's BASE frame
    T_base = np.eye(4); T_base[:3, 3] = [400.0, 0.0, 800.0]
    io.solve_joints_for_pose(T_base)
    np.testing.assert_allclose(pose_to_T(robot.ik_poses[0]), T_base, atol=1e-9)


def test_create_inspection_target_solves_in_base_but_stores_work_pose():
    io, robot = _cell(base_xyz=(0, 0, 0), frame_xyz=(800, -300, 0))
    T_work = np.eye(4); T_work[:3, 3] = [0.0, 0.0, 300.0]
    made = io.create_inspection_target(
        name="T1", T=T_work, inspection_tool="Realsense", work_frame="Work",
        neutral_joints=robomath.Mat([0.0] * 6))
    assert made["created"] is True
    solved = pose_to_T(robot.ik_poses[0])
    np.testing.assert_allclose(solved[:3, 3], [800.0, -300.0, 300.0], atol=1e-9)


# --- inspection roll reference and the neutral-branch gate --------------------
#
# Measured read-only on the operator's station: the frame-fixed roll-zero pose
# (aim [212.39, 148.69, 15.0], standoff 300, Realsense, Tasni Work Frame) has
# FOUR IK branches and every one is flipped. The seeded SolveIK stored
# [91.9, -50.4, 117.1, -177.9, -82.2, 1.9] -- dA4 -178.1 -- and it passed
# collision, so nothing caught it. These are those four branches.

PARKED = [89.22, -74.25, 147.96, 0.21, -42.52, 0.63]
FLIPPED_BRANCHES = [
    [91.9, -50.4, 117.1, -177.9, -82.2, 1.9],
    [91.9, -50.4, 117.1, 2.1, 82.2, -178.1],
    [95.4, -63.1, 140.2, -177.6, -60.3, 2.4],
    [95.4, -63.1, 140.2, 2.4, 60.3, -177.6],
]
NEUTRAL_SOLUTION = [89.8, -62.5, 147.8, 0.9, -54.1, -0.2]


class _BranchRobot(FakeRobot):
    """A robot whose SolveIK_All returns a chosen branch set."""

    def __init__(self, base, branches, config_of=None):
        super().__init__(base)
        self.branches = [list(b) for b in branches]
        self._config_of = config_of or (lambda joints: [0, 0, 1])
        self.fk_joints = []

    def SolveIK_All(self, pose, tool=None, reference=None):
        self.ik_poses.append(pose)
        if not self.branches:
            return robomath.Mat([[0.0]])
        return robomath.Mat([list(column) for column in zip(*self.branches)])

    def JointsConfig(self, joints):
        values = [float(v) for v in np.asarray(
            joints.list() if hasattr(joints, "list") else joints, float).ravel()]
        return robomath.Mat([float(v) for v in self._config_of(values)])

    def SolveFK(self, joints, tool=None, reference=None):
        self.fk_joints.append([float(v) for v in np.asarray(
            joints.list() if hasattr(joints, "list") else joints, float).ravel()])
        return robomath.xyzrpw_2_pose([500.0, 100.0, 700.0, 0.0, 0.0, 0.0])


def _branch_cell(branches, config_of=None):
    base = FakeFrame(robomath.xyzrpw_2_pose([0, 0, 0, 0, 0, 0]))
    robot = _BranchRobot(base, branches, config_of)
    frame = FakeFrame(robomath.xyzrpw_2_pose([800, -300, 0, 0, 0, 0]))
    cfg = RoboDKConfig()
    items = {cfg.robot_name: robot,
             ("Realsense", rl.ITEM_TYPE_TOOL): FakeTool(),
             ("Work", rl.ITEM_TYPE_FRAME): frame}
    return RdkIO(SimpleNamespace(rdk=FakeRdk(robot, items), config=cfg)), robot


def _flip_config(joints):
    """Parked config is (0, 0, 1); a wrist flip changes the FLIP flag."""
    return [0, 0, 1] if abs(joints[3]) < 90.0 else [0, 0, 0]


def test_an_inspection_pose_reachable_only_through_a_wrist_flip_is_refused():
    io, _robot = _branch_cell(FLIPPED_BRANCHES, _flip_config)
    T = np.eye(4); T[:3, 3] = [0.0, 0.0, 300.0]

    made = io.create_inspection_target(
        name="T1", T=T, inspection_tool="Realsense", work_frame="Work",
        neutral_joints=robomath.Mat(list(PARKED)), maximum_wrist_rotation_deg=90.0)

    assert made["created"] is False
    assert "neutral wrist branch" in made["reason"]
    assert "90" in made["reason"]


def test_an_inspection_pose_on_the_neutral_branch_is_accepted_with_its_deltas():
    io, _robot = _branch_cell([NEUTRAL_SOLUTION] + FLIPPED_BRANCHES, _flip_config)
    T = np.eye(4); T[:3, 3] = [0.0, 0.0, 300.0]

    made = io.create_inspection_target(
        name="T1", T=T, inspection_tool="Realsense", work_frame="Work",
        neutral_joints=robomath.Mat(list(PARKED)), maximum_wrist_rotation_deg=90.0)

    assert made["created"] is True
    assert made["joints"] == pytest.approx(NEUTRAL_SOLUTION, abs=1e-6)
    assert made["axis_4_rotation_deg"] == pytest.approx(0.69, abs=0.01)
    assert made["axis_5_rotation_deg"] == pytest.approx(-11.58, abs=0.01)
    assert made["axis_6_rotation_deg"] == pytest.approx(-0.83, abs=0.01)


def test_camera_axes_in_frame_is_read_only_and_frame_relative():
    """The service needs the camera's own +X at the start pose, without moving."""
    io, robot = _branch_cell([NEUTRAL_SOLUTION])

    T = io.camera_axes_in_frame("Realsense", "Work", robomath.Mat(list(PARKED)))

    # FK gives the flange at [500, 100, 700] in base; the tool adds +150 in Z and
    # the work frame sits at [800, -300, 0], so the camera lands at [-300, 400, 850].
    np.testing.assert_allclose(T[:3, 3], [-300.0, 400.0, 850.0], atol=1e-6)
    assert robot.fk_joints == [PARKED], "asked once, about the given joints only"
