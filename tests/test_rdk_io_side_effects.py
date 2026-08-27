"""Read-only RoboDK queries must not leave the station mutated.

Pose/reachability helpers are documented as running "without robot motion", but
they drive the station to make RoboDK's answers deterministic: SolveIK's seedless
retry returns the branch nearest the robot's CURRENT joints, and IK has to be
asked against the selected tool/frame. Both are legitimate; leaving the robot or
the operator's active tool somewhere else afterwards is not.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import robodk.robomath as robomath
import robolink as rl

from tasni.core.config import RoboDKConfig
from tasni.core.rdk_io import RdkIO

PARKED = [89.22, -74.25, 147.96, 0.21, -42.52, 0.63]   # the real cell's neutral
SEED = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]


class _Missing:
    def Valid(self): return False


class _Item:
    def __init__(self, name="item"): self.name = name
    def Valid(self): return True
    def Name(self): return self.name
    def Type(self): return rl.ITEM_TYPE_FRAME
    def PoseTool(self): return robomath.eye(4)
    def PoseAbs(self): return robomath.eye(4)


class _RetryRobot:
    """Seeded SolveIK finds nothing; the seedless retry does."""

    def __init__(self):
        self._joints = robomath.Mat(list(PARKED))
        self.set_joint_calls: list[list[float]] = []

    def Valid(self): return True
    def Parent(self): return _Missing()
    def Joints(self): return self._joints

    def setJoints(self, joints):
        values = [float(v) for v in joints.list()]
        self.set_joint_calls.append(values)
        self._joints = robomath.Mat(values)

    def SolveIK(self, pose, joints_approx=None, tool=None, reference=None):
        if joints_approx is not None:
            return robomath.Mat([0.0])          # RoboDK's "unreachable" sentinel
        return robomath.Mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


class _ReportRobot:
    """Records tool/frame activation so the restore can be asserted."""

    def __init__(self, prior_tool, prior_frame, joints_raise=False):
        self._prior = {rl.ITEM_TYPE_TOOL: prior_tool,
                       rl.ITEM_TYPE_FRAME: prior_frame}
        self._joints_raise = joints_raise
        self.tool_calls: list[object] = []
        self.frame_calls: list[object] = []

    def Valid(self): return True
    def Parent(self): return _Missing()

    def getLink(self, type_linked=rl.ITEM_TYPE_ROBOT):
        return self._prior.get(type_linked, _Missing())

    def setPoseTool(self, tool): self.tool_calls.append(tool)
    def setPoseFrame(self, frame): self.frame_calls.append(frame)

    def Joints(self):
        if self._joints_raise:
            raise RuntimeError("station query failed")
        return robomath.Mat(list(PARKED))

    def SolveIK_All(self, pose, tool=None, reference=None):
        return robomath.Mat([[0.0]])            # no usable branch
    def SolveIK(self, pose, joints_approx=None, tool=None, reference=None):
        return robomath.Mat([0.0])
    def JointsConfig(self, joints): return robomath.Mat([0.0, 0.0, 0.0, 0.0])


class _Rdk:
    def __init__(self, tool, frame): self._tool = tool; self._frame = frame
    def Item(self, name, item_type=0):
        if item_type == rl.ITEM_TYPE_TOOL:
            return self._tool
        if item_type == rl.ITEM_TYPE_FRAME:
            return self._frame
        return _Missing()


def test_seedless_ik_retry_puts_the_simulated_robot_back():
    """The retry stands the robot at the seed; it must not leave it there."""
    robot = _RetryRobot()
    io = RdkIO(SimpleNamespace(rdk=SimpleNamespace(), config=RoboDKConfig()))
    io.robot = lambda: robot

    solved = io.solve_joints_for_pose(np.eye(4), robomath.Mat(list(SEED)))

    assert solved is not None, "the seedless retry must still find the solution"
    # Determinism is still bought the same way: the robot IS stood at the seed...
    assert SEED in robot.set_joint_calls
    # ...but the station is handed back exactly as it was found.
    assert [float(v) for v in robot.Joints().list()] == PARKED
    assert robot.set_joint_calls[-1] == PARKED


def test_reachability_report_restores_the_operators_tool_and_frame():
    prior_tool, prior_frame = _Item("Realsense"), _Item("World")
    work_tool, work_frame = _Item("LongCalibTool"), _Item("Tasni Work Frame")
    robot = _ReportRobot(prior_tool, prior_frame)
    io = RdkIO(SimpleNamespace(rdk=_Rdk(work_tool, work_frame),
                               config=RoboDKConfig()))
    io.robot = lambda: robot

    io.extrusion_reachability_report(
        points_xyz=np.array([[10.0, 0.0, 3.0], [0.0, 10.0, 3.0],
                             [-10.0, 0.0, 3.0]]),
        orientation_rpy_deg=[0.0, 0.0, 0.0],
        print_tool="LongCalibTool", work_frame="Tasni Work Frame")

    # It must select the print tool/work frame to ask IK the right question...
    assert work_tool in robot.tool_calls and work_frame in robot.frame_calls
    # ...and hand the operator's selection back when it is done.
    assert robot.tool_calls[-1] is prior_tool
    assert robot.frame_calls[-1] is prior_frame


def test_reachability_report_restores_tool_and_frame_even_when_it_fails():
    prior_tool, prior_frame = _Item("Realsense"), _Item("World")
    robot = _ReportRobot(prior_tool, prior_frame, joints_raise=True)
    io = RdkIO(SimpleNamespace(rdk=_Rdk(_Item("LongCalibTool"), _Item("WF")),
                               config=RoboDKConfig()))
    io.robot = lambda: robot

    with pytest.raises(RuntimeError, match="station query failed"):
        io.extrusion_reachability_report(
            points_xyz=np.array([[10.0, 0.0, 3.0], [0.0, 10.0, 3.0],
                                 [-10.0, 0.0, 3.0]]),
            orientation_rpy_deg=[0.0, 0.0, 0.0],
            print_tool="LongCalibTool", work_frame="WF")

    assert robot.tool_calls[-1] is prior_tool
    assert robot.frame_calls[-1] is prior_frame
