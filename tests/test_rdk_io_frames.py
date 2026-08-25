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
        name="T1", T=T_work, inspection_tool="Realsense", work_frame="Work")
    assert made["created"] is True
    solved = pose_to_T(robot.ik_poses[0])
    np.testing.assert_allclose(solved[:3, 3], [800.0, -300.0, 300.0], atol=1e-9)
