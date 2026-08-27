"""Regression tests for extrusion layer program generation.

A Curve Follow Project solves its own inverse kinematics and cannot be steered
onto a wrist branch, and ``setInstruction`` silently discards written joints --
a move instruction always defers to its target item. So ONE TARGET PER WAYPOINT
is the only mechanism RoboDK's API offers for pinning a wrist configuration.
The curve and project stay in the tree for visibility; the moves are ours.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import robodk.robomath as robomath
import robolink as rl

from tasni.core.config import RoboDKConfig
from tasni.core.rdk_io import RdkIO, curve_follow_seed_T, pose_to_T

NEUTRAL_BRANCH = [93.8, -67.8, 151.8, 5.1, -52.9, 0.1]   # measured on the cell


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


class _Target(_Item):
    def __init__(self, name):
        self.name = name
        self.joints = None
        self.is_joint_target = False

    def setPose(self, pose): self.pose = pose
    def setAsJointTarget(self): self.is_joint_target = True
    def setJoints(self, joints): self.joints = joints


def _joint_rows(**axis_rotations):
    """A one-sample interpolated joint path with the given axis deviations."""
    rows = [[0.0] for _ in range(10)]
    for axis, value in axis_rotations.items():
        rows[int(axis)][0] = value
    return "", robomath.Mat(rows), 1


class _Program(_Item):
    """The project's generated program, gutted and rebuilt by the module."""

    def __init__(self):
        self.moves: list[tuple[str, object]] = []
        self.deleted: list[int] = []
        self._instructions = 2      # the project's own moves, to be discarded

    def setName(self, name): self.name = name
    def InstructionCount(self): return self._instructions

    def InstructionDelete(self, index):
        self.deleted.append(index)
        self._instructions = max(0, self._instructions - 1)

    def setPoseFrame(self, frame): pass
    def setPoseTool(self, tool): pass
    def setRounding(self, rounding): self.rounding = rounding
    def setSpeed(self, speed): self.moves.append(("speed", speed))
    def MoveJ(self, target): self.moves.append(("J", target))
    def MoveL(self, target): self.moves.append(("L", target))
    def RunInstruction(self, name, instruction_type):
        self.moves.append(("call", name))

    def InstructionListJoints(self, **kwargs): return _joint_rows()


class _InterpolatedFlipProgram(_Program):
    """Endpoints are fine, but RoboDK interpolates through an axis-4 flip."""

    def InstructionListJoints(self, **kwargs): return _joint_rows(**{"3": 170.0})


class _Axis5FlipProgram(_Program):
    """Flips through axis 5 -- the axis the guard once did not look at."""

    def InstructionListJoints(self, **kwargs): return _joint_rows(**{"4": 150.0})


class _Project(_Item):
    def __init__(self, setup_statuses=None, program=None):
        self.events = []
        self.program = program or _Program()
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


class _Rdk:
    def __init__(self):
        self.frame = _Item()
        self.tool = _Item()
        self.curve = _Curve()
        self.project = _Project()
        self.targets: list[_Target] = []

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
        return self.curve

    def AddMachiningProject(self, name, robot):
        self.project_name = name
        return self.project

    def AddTarget(self, name, frame, robot):
        target = _Target(name)
        self.targets.append(target)
        return target


SQUARE_XYZ = np.array([[10.0, 0.0, 7.5], [0.0, 10.0, 7.5],
                       [-10.0, 0.0, 7.5], [10.0, 0.0, 7.5]])


def _io(rdk, branch=NEUTRAL_BRANCH):
    """An RdkIO wired to the fakes. ``branch=None`` = no neutral solution."""
    io = RdkIO(SimpleNamespace(rdk=rdk, config=RoboDKConfig()))
    io.robot = lambda: _Robot()
    io.item_exists_as = lambda name, kind: True
    io.use_named_tool_frame = lambda tool, frame: np.eye(4)
    io.solve_joints_on_neutral_branch = (
        lambda T, neutral, previous=None, limit=90.0:
        None if branch is None else robomath.Mat(list(branch)))
    return io


def _build(io, name, **overrides):
    kwargs = dict(
        name=name, points_xyz=SQUARE_XYZ, orientation_rpy_deg=[0.0, 0.0, 0.0],
        print_tool="LongCalibTool", work_frame="Tasni Work Frame",
        speed_mm_s=75.0, travel_speed_mm_s=200.0, rounding_mm=1.0,
        approach_clearance_mm=40.0, retreat_clearance_mm=60.0,
        air_on_program="TasniDryAirOn", air_off_program="TasniDryAirOff",
        maximum_path_targets=0)
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
    for key in ("ProgEvents", "Approach", "Retract"):
        index = next(i for i, event in enumerate(events)
                     if event[:2] == ("param", key))
        assert index < generate_index, f"{key} must be set before generation"

    assert machining_index < generate_index
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


def test_waypoints_become_branch_locked_joint_targets():
    """One target per waypoint is the ONLY way RoboDK's API pins a wrist branch.

    setInstruction discards written joints (a move defers to its target item) and
    inline poses cannot be appended, so these targets are load-bearing.
    """
    rdk = _Rdk()
    name = "TasniCylinder_QUICK_locked_L001"

    result = _build(_io(rdk), name)

    assert result["waypoint_count"] == len(SQUARE_XYZ)
    assert result["targets"] == [
        f"{name}_Approach",
        *(f"{name}_P{index:04d}" for index in range(len(SQUARE_XYZ))),
        f"{name}_Retract"]
    assert [target.name for target in rdk.targets] == result["targets"]
    assert all(target.is_joint_target for target in rdk.targets)
    # approx: the joints are unwrapped to the nearest equivalent revolution.
    assert all(RdkIO._joint_values(target.joints) == pytest.approx(NEUTRAL_BRANCH)
               for target in rdk.targets)
    # The project's own generated moves are discarded, highest index first:
    # deleting low-to-high would renumber the instructions still to be removed.
    assert rdk.project.program.deleted == [1, 0]
    # MoveJ into the approach, then linear along the path and out.
    motion = [(kind, item) for kind, item in rdk.project.program.moves
              if kind in ("J", "L")]
    assert [kind for kind, _ in motion] == ["J"] + ["L"] * (len(SQUARE_XYZ) + 1)
    assert [item.name for _, item in motion] == result["targets"]
    # Targets are listed separately so the job can bin them even on failure.
    assert result["artifacts"][-len(result["targets"]):] == result["targets"]


def test_dense_paths_are_thinned_to_the_configured_waypoint_budget():
    """180 points on a 37.5 mm circle sit ~1.3 mm apart -- far finer than needed."""
    dense = np.array([[10.0, float(index), 7.5] for index in range(180)])
    rdk = _Rdk()

    result = _build(_io(rdk), "TasniCylinder_QUICK_thin_L001",
                    points_xyz=dense, maximum_path_targets=60)

    assert result["point_count"] == 180, "the plan's own path is untouched"
    assert result["waypoint_count"] == 60
    assert len(rdk.targets) == 62, "60 waypoints plus approach and retract"


def test_thinning_always_keeps_both_ends_of_the_path():
    thinned = RdkIO._waypoint_indices(180, 60)
    assert thinned[0] == 0 and thinned[-1] == 179
    assert len(thinned) == 60
    assert RdkIO._waypoint_indices(5, 60) == [0, 1, 2, 3, 4], "never pads"
    assert RdkIO._waypoint_indices(5, 0) == [0, 1, 2, 3, 4], "0 disables thinning"


def test_generation_is_cancellable_between_waypoints():
    """The old target loop held ~1800 RPCs with no cancel check inside it."""
    rdk = _Rdk()
    calls = []

    def check_cancel():
        calls.append(1)
        if len(calls) > 2:
            raise KeyboardInterrupt("cancelled")

    with pytest.raises(KeyboardInterrupt):
        _build(_io(rdk), "TasniCylinder_QUICK_cancel_L001",
               check_cancel=check_cancel)

    assert len(rdk.targets) < len(SQUARE_XYZ) + 2, "it stopped part-way"


def test_generation_fails_when_a_waypoint_has_no_neutral_solution():
    rdk = _Rdk()

    with pytest.raises(RuntimeError, match="no IK solution on the neutral"):
        _build(_io(rdk, branch=None), "TasniCylinder_QUICK_unsolvable_L001")


def test_curve_follow_retries_with_internal_flip_when_setup_is_rejected():
    rdk = _Rdk()
    rdk.project = _Project(setup_statuses=[-5.0, 0.0])

    result = _build(_io(rdk), "TasniCylinder_DRY_retry_L001")

    poses = [event[1] for event in rdk.project.events if event[0] == "pose"]
    np.testing.assert_allclose(pose_to_T(poses[0])[:3, :3], np.eye(3), atol=1e-9)
    np.testing.assert_allclose(pose_to_T(poses[1])[:3, :3],
                               np.diag([1.0, -1.0, -1.0]), atol=1e-9)
    assert result["setup_attempts"] == [-5.0, 0.0]


def test_a_hidden_interpolated_axis_4_flip_is_rejected():
    rdk = _Rdk()
    rdk.project = _Project(program=_InterpolatedFlipProgram())

    with pytest.raises(RuntimeError, match="turns axis 4.*blocked"):
        _build(_io(rdk), "TasniCylinder_QUICK_flip_L001")


def test_a_hidden_interpolated_axis_5_flip_is_rejected():
    """Axis 5 was unbounded, so a flip realised through it passed unnoticed."""
    rdk = _Rdk()
    rdk.project = _Project(program=_Axis5FlipProgram())

    with pytest.raises(RuntimeError, match="turns axis 5.*blocked"):
        _build(_io(rdk), "TasniCylinder_QUICK_axis5_L001")


# --- the path-to-tool seed ----------------------------------------------------
#
# RoboDK's Curve Follow Project does not reproduce the roll it is seeded with: it
# MIRRORS it. Measured on the cell with tools/probe_extrusion_branch.py (probe R),
# a project seeded with ``X . rotx(pi) . rotz(pi)`` generates the rotation
# ``Rz(180) . S . X . S`` where ``S = diag(1, -1, 1)``. The model below is that
# measurement, so these tests pin our inverse against RoboDK's real behaviour
# rather than against itself.

_MIRROR_S = np.diag([1.0, -1.0, 1.0, 1.0])
_SEED_SUFFIX = pose_to_T(robomath.rotx(np.pi) * robomath.rotz(np.pi))


def _robodk_generated_rotation(seed_T):
    """What RoboDK emits when a Curve Follow Project is seeded with ``seed_T``."""
    source = np.asarray(seed_T, float) @ np.linalg.inv(_SEED_SUFFIX)
    return pose_to_T(robomath.rotz(np.pi)) @ _MIRROR_S @ source @ _MIRROR_S


def _rotation_gap_deg(A, B):
    R = np.asarray(A, float)[:3, :3].T @ np.asarray(B, float)[:3, :3]
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))))


# Yaw-only cases plus deliberately tilted ones: the tilted cases are what prove a
# coordinate-free inverse is required. The naive seed scored 60.4 deg at the last.
SEED_ORIENTATIONS = [
    [0.0, 0.0, 0.0],
    [-0.01, 0.05, 90.69],      # this cell's parked commanded orientation
    [0.0, 0.0, 110.69],
    [0.0, 0.0, 135.69],
    [0.0, 10.0, 0.0],          # tilted off the surface normal
    [0.0, -10.0, 0.0],
    [5.0, 10.0, 0.0],
    [5.0, 10.0, 30.0],
]


@pytest.mark.parametrize("rpy", SEED_ORIENTATIONS)
def test_curve_follow_seed_makes_robodk_generate_the_commanded_rotation(rpy):
    commanded = RdkIO.xyzrpy_pose_T([0.0, 0.0, 0.0], rpy)

    generated = _robodk_generated_rotation(curve_follow_seed_T(commanded))

    assert _rotation_gap_deg(commanded, generated) < 1e-6, (
        f"seeding the commanded orientation {rpy} must generate it back")


def test_curve_follow_seed_is_a_rotation_and_leaves_its_input_alone():
    commanded = RdkIO.xyzrpy_pose_T([0.0, 0.0, 0.0], [5.0, 10.0, 30.0])
    original = commanded.copy()

    seed = curve_follow_seed_T(commanded)

    assert np.allclose(commanded, original), "the caller's pose must not be mutated"
    assert np.allclose(seed[:3, :3] @ seed[:3, :3].T, np.eye(3), atol=1e-9)
    assert float(np.linalg.det(seed[:3, :3])) == pytest.approx(1.0)
    assert np.allclose(seed[:3, 3], 0.0), "a path-to-tool seed carries no offset"


def test_the_naive_seed_is_only_correct_near_ninety_degrees_of_yaw():
    """Why the inverse exists at all: the old seed was a coincidence.

    ``orientation @ rotx(pi) @ rotz(pi)`` scored 1.4 deg at this cell's ~90 deg
    yaw and 91.4 deg at 135 deg yaw, measured on the cell.
    """
    near_ninety = RdkIO.xyzrpy_pose_T([0.0, 0.0, 0.0], [-0.01, 0.05, 90.69])
    far_off = RdkIO.xyzrpy_pose_T([0.0, 0.0, 0.0], [0.0, 0.0, 135.69])

    def naive_error(commanded):
        return _rotation_gap_deg(
            commanded, _robodk_generated_rotation(commanded @ _SEED_SUFFIX))

    assert naive_error(near_ninety) == pytest.approx(1.37, abs=0.05)
    assert naive_error(far_off) == pytest.approx(91.37, abs=0.05)
