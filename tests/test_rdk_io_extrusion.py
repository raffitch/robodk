"""Regression tests for native RoboDK Curve Follow project setup.

The Curve Follow Project owns the path: one curve object, no station targets. It
aligns the tool to the curve normal by its own convention, so the path-to-tool
seed decides whether the generated path lands on the operator's neutral wrist
branch or 180 degrees from it. Generation therefore tries each seed and keeps the
one whose interpolated joint path verifies, falling back to clamping the wrist
axes only as a last resort -- and always handing the robot's travel back.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import robodk.robomath as robomath
import robolink as rl

from tasni.core.config import RoboDKConfig
from tasni.core.rdk_io import RdkIO, pose_to_T

STOCK_LOWER = [-185.0, -140.0, -120.0, -350.0, -125.0, -350.0]
STOCK_UPPER = [185.0, -5.0, 168.0, 350.0, 125.0, 350.0]


class _Missing:
    def Valid(self): return False


class _Item:
    def Valid(self): return True
    def PoseTool(self): return robomath.eye(4)
    def PoseAbs(self): return robomath.eye(4)


class _Robot(_Item):
    def __init__(self):
        self.lower, self.upper = list(STOCK_LOWER), list(STOCK_UPPER)
        self.limit_calls: list[tuple[list[float], list[float]]] = []

    def Joints(self): return robomath.Mat([0.0] * 6)
    def setPoseTool(self, tool): pass
    def setPoseFrame(self, frame): pass
    def Parent(self): return _Missing()

    def JointLimits(self):
        return (robomath.Mat(list(self.lower)), robomath.Mat(list(self.upper)), 0.0)

    def setJointLimits(self, lower, upper):
        low = [float(v) for v in lower.list()]
        high = [float(v) for v in upper.list()]
        self.limit_calls.append((low, high))
        self.lower, self.upper = low, high


class _Curve(_Item):
    def setName(self, name): self.name = name
    def setParent(self, parent): self.parent = parent
    def setValue(self, key, value): pass
    def Delete(self): pass


def _joint_rows(**axis_rotations):
    """A one-sample interpolated joint path with the given axis deviations."""
    rows = [[0.0] for _ in range(10)]
    for axis, value in axis_rotations.items():
        rows[int(axis)][0] = value
    return "", robomath.Mat(rows), 1


class _Program(_Item):
    """A generated program whose interpolated path stays on the neutral branch."""

    def setName(self, name): self.name = name
    def InstructionCount(self): return 0
    def InstructionListJoints(self, **kwargs): return _joint_rows()


class _InterpolatedFlipProgram(_Program):
    """Always interpolates through an axis-4 flip, whatever is tried."""

    def InstructionListJoints(self, **kwargs): return _joint_rows(**{"3": 170.0})


class _Axis5FlipProgram(_Program):
    """Flips through axis 5 only -- the axis the guard used not to look at."""

    def InstructionListJoints(self, **kwargs): return _joint_rows(**{"4": 150.0})


class _FlipOnFirstSeedProgram(_Program):
    """Flipped under the first path-to-tool seed, neutral under the second."""

    def __init__(self): self.generations = 0

    def InstructionListJoints(self, **kwargs):
        self.generations += 1
        return _joint_rows(**({"3": 170.0} if self.generations == 1 else {}))


class _FlipUnlessClampedProgram(_Program):
    """Flips for as long as the robot's axis-4 travel can still reach the flip."""

    def __init__(self, robot): self.robot = robot

    def InstructionListJoints(self, **kwargs):
        reachable = self.robot.lower[3] <= -170.0
        return _joint_rows(**({"3": 170.0} if reachable else {}))


class _Project(_Item):
    def __init__(self, setup_statuses=None):
        self.events = []
        self.program = _Program()
        self.setup_statuses = list(setup_statuses or [0.0])

    def setPoseFrame(self, frame): self.events.append(("frame", frame))
    def setPoseTool(self, tool): self.events.append(("tool", tool))
    def setPose(self, pose): self.events.append(("pose", pose))
    def setJoints(self, joints): self.events.append(("joints", joints))

    def setParam(self, key, value=None):
        self.events.append(("param", key, value))

    def setMachiningParameters(self, *, part):
        self.events.append(("generate", part))
        status = self.setup_statuses.pop(0)
        return (self.program if status >= 0 else _Missing()), status

    def Update(self, collisions):
        self.events.append(("update", collisions))
        return 0, 0, 0, 1.0, ""

    def getLink(self, item_type):
        assert item_type == rl.ITEM_TYPE_PROGRAM
        return self.program


class _Rdk:
    def __init__(self):
        self.frame = _Item()
        self.tool = _Item()
        self.curve = _Curve()
        self.project = _Project()
        self.targets = []

    def Item(self, name, item_type=0):
        if item_type == rl.ITEM_TYPE_FRAME:
            return self.frame
        if item_type == rl.ITEM_TYPE_TOOL:
            return self.tool
        if item_type == rl.ITEM_TYPE_OBJECT and name == "Tasni Work Surface":
            return self.curve
        return _Missing()

    def AddCurve(self, vertices, projection_type):
        self.vertices = vertices
        self.projection_type = projection_type
        return self.curve

    def AddMachiningProject(self, name, robot):
        self.project_name = name
        return self.project

    def AddTarget(self, name, frame, robot):
        target = SimpleNamespace(name=name)
        self.targets.append(target)
        return target


SQUARE_XYZ = np.array([[10.0, 0.0, 7.5], [0.0, 10.0, 7.5],
                       [-10.0, 0.0, 7.5], [10.0, 0.0, 7.5]])


def _io(rdk):
    """An RdkIO wired to the fakes, sharing ONE robot so limits can be asserted."""
    io = RdkIO(SimpleNamespace(rdk=rdk, config=RoboDKConfig()))
    robot = _Robot()
    io.robot = lambda: robot
    io.item_exists_as = lambda name, kind: True
    io.use_named_tool_frame = lambda tool, frame: np.eye(4)
    io.test_robot = robot
    return io


def _build(io, name, **overrides):
    kwargs = dict(
        name=name, points_xyz=SQUARE_XYZ, orientation_rpy_deg=[0.0, 0.0, 0.0],
        print_tool="LongCalibTool", work_frame="Tasni Work Frame",
        speed_mm_s=75.0, travel_speed_mm_s=200.0, rounding_mm=1.0,
        approach_clearance_mm=40.0, retreat_clearance_mm=60.0,
        air_on_program="TasniDryAirOn", air_off_program="TasniDryAirOff")
    kwargs.update(overrides)
    return io.create_extrusion_layer_program(**kwargs)


def test_curve_follow_options_are_applied_before_initial_generation():
    """The first RoboDK solve must not inherit positioner/orientation defaults."""
    rdk = _Rdk()
    result = _build(_io(rdk), "TasniCylinder_DRY_test_L001")

    events = rdk.project.events
    generate_index = next(i for i, event in enumerate(events) if event[0] == "generate")
    machining_index, machining = next(
        (i, event[2]) for i, event in enumerate(events)
        if event[:2] == ("param", "Machining"))
    events_index = next(
        i for i, event in enumerate(events)
        if event[:2] == ("param", "ProgEvents"))
    approach_index = next(
        i for i, event in enumerate(events)
        if event[:2] == ("param", "Approach"))
    retract_index = next(
        i for i, event in enumerate(events)
        if event[:2] == ("param", "Retract"))

    assert max(machining_index, events_index, approach_index, retract_index) < generate_index
    assert machining["TurntableActive"] == 0
    assert machining["FollowAngleOn"] == 0
    assert machining["FollowRealignOn"] == 0
    assert machining["RotZ_Range"] == 0
    assert result["setup_status"] == 0.0


def test_exact_station_item_check_supports_collision_proxy_objects():
    rdk = _Rdk()
    io = RdkIO(SimpleNamespace(rdk=rdk, config=RoboDKConfig()))
    assert io.item_exists_as("Tasni Work Surface", "object") is True
    assert io.item_exists_as("missing", "object") is False


def test_native_generation_emits_one_curve_and_no_station_targets():
    """The project owns the path: a target per point would litter the station."""
    rdk = _Rdk()
    name = "TasniCylinder_QUICK_native_L001"

    result = _build(_io(rdk), name)

    assert result["targets"] == []
    assert rdk.targets == []
    assert result["artifacts"] == [f"{name}_Curve", f"{name}_Settings", name]
    assert result["point_count"] == len(SQUARE_XYZ)
    # The first seed verified, so nothing else had to be tried.
    assert result["tool_path_flip_applied"] is False
    assert result["wrist_limits_clamped"] is False
    assert _io(rdk).test_robot.limit_calls == []


def test_curve_follow_retries_with_internal_flip_when_setup_is_rejected():
    rdk = _Rdk()
    rdk.project = _Project(setup_statuses=[-5.0, 0.0])

    result = _build(_io(rdk), "TasniCylinder_DRY_retry_L001")

    generation_poses = [event[1] for event in rdk.project.events if event[0] == "pose"]
    first = pose_to_T(generation_poses[0])
    fallback = pose_to_T(generation_poses[1])
    np.testing.assert_allclose(first[:3, :3], np.eye(3), atol=1e-9)
    np.testing.assert_allclose(fallback[:3, :3], np.diag([1.0, -1.0, -1.0]),
                               atol=1e-9)
    assert result["setup_attempts"] == [-5.0, 0.0]


def test_a_flipped_first_seed_falls_through_to_the_flipped_path_to_tool_seed():
    """The measured cell behaviour: the identity seed aims the tool 180 deg out."""
    rdk = _Rdk()
    rdk.project.program = _FlipOnFirstSeedProgram()
    io = _io(rdk)

    result = _build(io, "TasniCylinder_QUICK_seed2_L001",
                    maximum_tool_axis_spin_deg=90.0)

    assert result["tool_path_flip_applied"] is True
    assert rdk.project.program.generations == 2
    # Seed selection is enough on its own; the wrist travel is never touched.
    assert result["wrist_limits_clamped"] is False
    assert io.test_robot.limit_calls == []
    # The winning seed is the local-X flip of the requested orientation.
    winning = pose_to_T([event[1] for event in rdk.project.events
                         if event[0] == "pose"][-1])
    np.testing.assert_allclose(winning[:3, :3], np.diag([1.0, -1.0, -1.0]),
                               atol=1e-9)


def test_clamping_wrist_travel_is_the_last_resort_after_both_seeds_flip():
    rdk = _Rdk()
    io = _io(rdk)
    rdk.project.program = _FlipUnlessClampedProgram(io.test_robot)

    result = _build(io, "TasniCylinder_QUICK_clamped_L001",
                    maximum_tool_axis_spin_deg=90.0)

    assert result["wrist_limits_clamped"] is True
    assert result["targets"] == []
    clamped_lower, clamped_upper = io.test_robot.limit_calls[0]
    # The wrist axes are pulled in to the neutral window (neutral is all-zero here)...
    assert (clamped_lower[3], clamped_upper[3]) == (-90.0, 90.0)
    assert (clamped_lower[4], clamped_upper[4]) == (-90.0, 90.0)
    assert (clamped_lower[5], clamped_upper[5]) == (-90.0, 90.0)
    # ...and every other axis keeps the robot's real travel.
    assert (clamped_lower[1], clamped_upper[1]) == (-140.0, -5.0)
    # The clamp is temporary: the robot is handed back its declared travel.
    assert io.test_robot.limit_calls[-1] == (STOCK_LOWER, STOCK_UPPER)


def test_joint_limits_are_restored_when_the_clamped_regeneration_still_flips():
    rdk = _Rdk()
    rdk.project.program = _InterpolatedFlipProgram()
    io = _io(rdk)

    with pytest.raises(RuntimeError, match="turns axis 4.*blocked"):
        _build(io, "TasniCylinder_QUICK_stillflipped_L001",
               maximum_tool_axis_spin_deg=90.0)

    # Clamped once, then restored -- a failed generation must not leave the robot
    # with narrowed wrist travel for every later motion in the station.
    assert len(io.test_robot.limit_calls) == 2
    assert io.test_robot.limit_calls[-1] == (STOCK_LOWER, STOCK_UPPER)


def test_generation_fails_loudly_when_the_station_will_not_accept_a_clamp():
    """No clamp available means the original wrist-flip failure must surface."""
    rdk = _Rdk()
    rdk.project.program = _InterpolatedFlipProgram()
    io = _io(rdk)
    io.robot().JointLimits = lambda: (_ for _ in ()).throw(RuntimeError("no limits"))

    with pytest.raises(RuntimeError, match="turns axis 4.*blocked"):
        _build(io, "TasniCylinder_QUICK_nolimits_L001",
               maximum_tool_axis_spin_deg=90.0)

    assert io.test_robot.limit_calls == []


def test_an_axis_5_flip_is_rejected_too():
    """Axis 5 was unbounded, so a flip realised through it passed unnoticed."""
    rdk = _Rdk()
    rdk.project.program = _Axis5FlipProgram()
    io = _io(rdk)

    with pytest.raises(RuntimeError, match="turns axis 5.*blocked"):
        _build(io, "TasniCylinder_QUICK_axis5_L001",
               maximum_tool_axis_spin_deg=90.0)
