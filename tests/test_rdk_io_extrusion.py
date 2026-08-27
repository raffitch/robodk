"""Regression tests for native RoboDK Curve Follow project setup."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import robodk.robomath as robomath
import robolink as rl

from tasni.core.config import RoboDKConfig
from tasni.core.rdk_io import RdkIO, pose_to_T


class _Missing:
    def Valid(self): return False


class _Item:
    def Valid(self): return True
    def PoseTool(self): return robomath.eye(4)
    def PoseAbs(self): return robomath.eye(4)


class _Robot(_Item):
    def Joints(self): return robomath.Mat([0.0] * 6)
    def setPoseTool(self, tool): pass
    def setPoseFrame(self, frame): pass
    def Parent(self): return _Missing()


class _Curve(_Item):
    def setName(self, name): self.name = name
    def setParent(self, parent): self.parent = parent
    def setValue(self, key, value): pass
    def Delete(self): pass


class _Program(_Item):
    def __init__(self): self.moves = []
    def setName(self, name): self.name = name
    def InstructionCount(self): return 0
    def InstructionDelete(self, index): pass
    def setPoseFrame(self, frame): pass
    def setPoseTool(self, tool): pass
    def setRounding(self, rounding): pass
    def setSpeed(self, speed): pass
    def MoveJ(self, target): self.moves.append(("J", target))
    def MoveL(self, target): self.moves.append(("L", target))
    def RunInstruction(self, name, instruction_type): pass
    def InstructionListJoints(self, **kwargs):
        return "", robomath.Mat([[0.0] for _ in range(10)]), 1


class _GeneratedProgram(_Program):
    """A Curve Follow program that already holds the project's own instructions.

    The real one does; those are what the neutral-branch rebuild has to discard.
    """

    def __init__(self, instruction_count: int = 2):
        super().__init__()
        self._instruction_count = instruction_count
        self.deleted: list[int] = []

    def InstructionCount(self): return self._instruction_count

    def InstructionDelete(self, index):
        self.deleted.append(index)
        self._instruction_count = max(0, self._instruction_count - 1)


class _InterpolatedFlipProgram(_Program):
    """Endpoints stay neutral, but RoboDK interpolates through a wrist flip."""

    def InstructionListJoints(self, **kwargs):
        rows = [[0.0] for _ in range(10)]
        rows[3][0] = 170.0
        return "", robomath.Mat(rows), 1


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
        target = _Target(name)
        self.targets.append(target)
        return target


class _Target(_Item):
    def __init__(self, name):
        self.name = name
        self.joints = None
        self.is_joint_target = False

    def setPose(self, pose): self.pose = pose
    def setAsJointTarget(self): self.is_joint_target = True
    def setJoints(self, joints): self.joints = joints


SQUARE_XYZ = np.array([[10.0, 0.0, 7.5], [0.0, 10.0, 7.5],
                       [-10.0, 0.0, 7.5], [10.0, 0.0, 7.5]])


def _io(rdk, *, joints=None):
    """An RdkIO wired to the fakes, with IK stubbed at the neutral-branch seam.

    ``solve_joints_on_neutral_branch`` enumerates ``SolveIK_All`` and falls back to
    ``solve_joints_for_pose``; the fake robot has no station behind it, so stubbing
    that fallback is what gives these tests a deterministic solution to lock onto.
    """
    io = RdkIO(SimpleNamespace(rdk=rdk, config=RoboDKConfig()))
    io.robot = lambda: _Robot()
    io.item_exists_as = lambda name, kind: True
    io.use_named_tool_frame = lambda tool, frame: np.eye(4)
    io.solve_joints_for_pose = lambda T, seed=None: robomath.Mat(
        list(joints if joints is not None else [0.0] * 6))
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


def test_curve_follow_retries_with_internal_flip_but_keeps_requested_output_pose():
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
    # The internal flip is a generation seed only: the emitted targets still carry
    # the requested orientation, so the printed rotation cannot change silently.
    for target in rdk.targets:
        np.testing.assert_allclose(pose_to_T(target.pose)[:3, :3], np.eye(3),
                                   atol=1e-9)


def test_curve_moves_are_rebuilt_as_neutral_branch_locked_joint_targets():
    """The project's own motion is discarded for branch-locked joint targets.

    A Curve Follow program does not retain joints written through setInstruction,
    so every endpoint is re-emitted as a named joint target instead.
    """
    rdk = _Rdk()
    rdk.project.program = _GeneratedProgram(instruction_count=2)
    name = "TasniCylinder_DRY_locked_L001"

    result = _build(_io(rdk, joints=[0.0, 0.0, 0.0, 0.0, 0.0, 45.0]), name,
                    maximum_tool_axis_spin_deg=90.0)

    program = rdk.project.program
    # Highest index first: deleting low-to-high would renumber the rest.
    assert program.deleted == [1, 0]
    assert result["targets"] == [
        f"{name}_Approach",
        *(f"{name}_P{index:04d}" for index in range(len(SQUARE_XYZ))),
        f"{name}_Retract"]
    assert [target.name for target in rdk.targets] == result["targets"]
    # Joint targets, not cartesian ones: RoboDK cannot re-solve another branch.
    assert all(target.is_joint_target for target in rdk.targets)
    assert all(RdkIO._joint_values(target.joints)[-1] == 45.0
               for target in rdk.targets)
    assert [move for move, _ in program.moves] == ["J"] + ["L"] * 5
    assert [target.name for _, target in program.moves] == result["targets"]
    assert result["maximum_tool_axis_spin_seen_deg"] == 45.0
    assert result["maximum_axis_4_rotation_seen_deg"] == 0.0
    # Every emitted target is disposable, so Reset can remove it.
    assert result["artifacts"][-len(result["targets"]):] == result["targets"]


def test_curve_generation_rejects_axis_6_spin_beyond_limit():
    rdk = _Rdk()
    io = _io(rdk, joints=[0.0, 0.0, 0.0, 0.0, 0.0, 120.0])

    with pytest.raises(RuntimeError, match="neutral.*wrist|axis-4/axis-6"):
        _build(io, "TasniCylinder_DRY_spin_L001", maximum_tool_axis_spin_deg=90.0)

    assert rdk.project.program.moves == []


def test_curve_generation_rejects_axis_4_rotation_beyond_limit():
    rdk = _Rdk()
    io = _io(rdk, joints=[0.0, 0.0, 0.0, 120.0, 0.0, 0.0])

    with pytest.raises(RuntimeError, match="neutral.*wrist|axis 4"):
        _build(io, "TasniCylinder_DRY_axis4", maximum_tool_axis_spin_deg=90.0)

    assert rdk.project.program.moves == []


def test_curve_generation_rejects_hidden_interpolated_axis_4_flip():
    rdk = _Rdk()
    rdk.project.program = _InterpolatedFlipProgram()

    with pytest.raises(RuntimeError, match="turns axis 4.*blocked"):
        _build(_io(rdk), "TasniCylinder_DRY_interpolated_flip",
               maximum_tool_axis_spin_deg=90.0)
