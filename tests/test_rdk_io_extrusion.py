"""Regression tests for extrusion layer program generation.

The layer is ONE native RoboDK Curve Follow program over ONE curve, with zero
station targets. What used to make that impossible was the path-to-tool seed:
RoboDK mirrors the roll it is seeded with, so the module's old seed produced a
path rotated ~180 degrees about the tool axis and RoboDK realised that rotation
by flipping axis 4. ``curve_follow_seed_T`` inverts the mirror, and RoboDK's own
generated program is then kept as-is and verified rather than replaced.
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
AIR_ON, AIR_OFF = "TasniDryAirOn", "TasniDryAirOff"
SQUARE_XYZ = np.array([[10.0, 0.0, 7.5], [0.0, 10.0, 7.5],
                       [-10.0, 0.0, 7.5], [10.0, 0.0, 7.5]])
APPROACH_MM, RETRACT_MM = 40.0, 60.0


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


def _move(name, position, rotation):
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(rotation, float)[:3, :3]
    pose[:3, 3] = np.asarray(position, float)
    return (name, rl.INS_TYPE_MOVE, rl.MOVE_TYPE_LINEAR, 0,
            robomath.Mat(pose.tolist()), robomath.Mat([0.0] * 6))


def _native_instructions(points=SQUARE_XYZ, rotation=None, *,
                         normal=(0.0, 0.0, 1.0), air_on_at_standoff=False,
                         valve_lifts_before_off=False):
    """RoboDK's generated Curve Follow program, in its MEASURED instruction order.

    Measured on the cell (``tools/probe_extrusion_branch.py``, probe V): five
    preamble instructions, a rapid and an approach move at the standoff, the
    descent to the first path point, then CallPathStart, the path, CallPathFinish
    before the lift, and the retract.
    """
    rotation = np.eye(4) if rotation is None else np.asarray(rotation, float)
    points = np.asarray(points, float)
    normal = np.asarray(normal, float)
    standoff = points[0] + normal * APPROACH_MM
    lifted = points[-1] + normal * RETRACT_MM
    air_on = (AIR_ON, rl.INS_TYPE_CODE, None, None, None, None)
    air_off = (AIR_OFF, rl.INS_TYPE_CODE, None, None, None, None)
    out = [
        ("Smooth(1)", rl.INS_TYPE_ROUNDING, None, None, None, None),
        ("Set speed (200.0 mm/s)", rl.INS_TYPE_CHANGESPEED, None, None, None, None),
        ("Set Ref.: Tasni Work Frame", rl.INS_TYPE_CHANGEFRAME, None, None, None, None),
        ("Set Tool: LongCalibTool", rl.INS_TYPE_CHANGETOOL, None, None, None, None),
        ("Show LongCalibTool", rl.INS_TYPE_EVENT, None, None, None, None),
        _move("MoveJ 1", standoff, rotation),
        _move("MoveL 2", standoff, rotation),
    ]
    if air_on_at_standoff:
        out.append(air_on)                     # the unsafe layout: valve open high
    out.append(_move("MoveL 3", points[0], rotation))
    if not air_on_at_standoff:
        out.append(air_on)
    for index, point in enumerate(points[1:], start=4):
        out.append(_move(f"MoveL {index}", point, rotation))
    if valve_lifts_before_off:
        out.append(_move("MoveL lift", lifted, rotation))
    out.append(("Set speed (200.0 mm/s)", rl.INS_TYPE_CHANGESPEED, None, None, None, None))
    out.append(air_off)
    out.append(_move("MoveL last", points[-1], rotation))
    out.append(_move("MoveL retract", lifted, rotation))
    return out


class _Program(_Item):
    """RoboDK's own generated program. The module keeps it and verifies it."""

    def __init__(self, instructions=None):
        self.instructions = (_native_instructions() if instructions is None
                             else list(instructions))
        self.deleted: list[int] = []
        self.moves: list[tuple[str, object]] = []

    def setName(self, name): self.name = name
    def InstructionCount(self): return len(self.instructions)
    def Instruction(self, index): return self.instructions[index]

    def InstructionDelete(self, index):
        self.deleted.append(index)
        self.instructions.pop(index)

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

    def ItemList(self, item_type):
        if item_type == rl.ITEM_TYPE_TARGET:
            return list(self.targets)
        return []

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
        approach_clearance_mm=APPROACH_MM, retreat_clearance_mm=RETRACT_MM,
        air_on_program=AIR_ON, air_off_program=AIR_OFF)
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


# --- the native program is kept, not rebuilt ----------------------------------

def test_the_layer_creates_no_station_targets_at_all():
    """The whole point: the operator rejected per-point targets outright."""
    rdk = _Rdk()

    result = _build(_io(rdk), "TasniCylinder_QUICK_native_L001")

    assert rdk.targets == [], "layer generation must not create targets"
    assert result["targets"] == []
    assert result["artifacts"] == ["TasniCylinder_QUICK_native_L001_Curve",
                                   "TasniCylinder_QUICK_native_L001_Settings",
                                   "TasniCylinder_QUICK_native_L001"]


def test_robodks_own_generated_instructions_are_kept():
    rdk = _Rdk()
    generated = len(rdk.project.program.instructions)

    result = _build(_io(rdk), "TasniCylinder_QUICK_keep_L001")

    assert rdk.project.program.deleted == [], "the native solve must survive"
    assert rdk.project.program.moves == [], "no moves are appended by us"
    assert result["instruction_count"] == generated


def test_the_curve_normal_is_the_surface_normal_not_the_tool_axis():
    """The old code used the commanded tool Z, which points INTO the table for
    any tool whose Z does not happen to point up, as LongCalibTool's does."""
    rdk = _Rdk()
    # A commanded orientation whose tool Z points DOWN (roll 180 degrees). The
    # fake program must carry it too, or the pose gate rejects it first.
    upside_down = RdkIO.xyzrpy_pose_T([0.0, 0.0, 0.0], [180.0, 0.0, 0.0])
    rdk.project = _Project(program=_Program(_native_instructions(rotation=upside_down)))

    _build(_io(rdk), "TasniCylinder_QUICK_normal_L001",
           orientation_rpy_deg=[180.0, 0.0, 0.0])

    normals = np.array([vertex[3:6] for vertex in rdk.vertices])
    np.testing.assert_allclose(normals, np.tile([0.0, 0.0, 1.0], (len(SQUARE_XYZ), 1)),
                               atol=1e-9)


def test_the_path_to_tool_seed_is_the_computed_inverse_and_is_tried_once():
    rdk = _Rdk()
    orientation = RdkIO.xyzrpy_pose_T([0.0, 0.0, 0.0], [0.0, 0.0, 110.69])
    rdk.project = _Project(program=_Program(_native_instructions(rotation=orientation)))

    _build(_io(rdk), "TasniCylinder_QUICK_seed_L001",
           orientation_rpy_deg=[0.0, 0.0, 110.69])

    poses = [event[1] for event in rdk.project.events if event[0] == "pose"]
    assert len(poses) == 1, "one computed seed, not a sweep of guesses"
    np.testing.assert_allclose(pose_to_T(poses[0]), curve_follow_seed_T(orientation),
                               atol=1e-9)


# --- verification gates -------------------------------------------------------

def test_a_flipped_generated_pose_is_rejected_and_names_the_angle():
    """RoboDK emitting a 178 degree roll must never be silently accepted."""
    rdk = _Rdk()
    flipped = pose_to_T(robomath.rotz(np.radians(178.0)))
    rdk.project = _Project(program=_Program(_native_instructions(rotation=flipped)))

    with pytest.raises(RuntimeError, match=r"178\.0 deg"):
        _build(_io(rdk), "TasniCylinder_QUICK_rolled_L001")


def test_a_small_generated_pose_error_is_accepted():
    rdk = _Rdk()
    nudged = pose_to_T(robomath.rotz(np.radians(0.3)))
    rdk.project = _Project(program=_Program(_native_instructions(rotation=nudged)))

    result = _build(_io(rdk), "TasniCylinder_QUICK_nudged_L001")

    assert result["maximum_pose_error_deg"] == pytest.approx(0.3, abs=0.01)


def test_the_pose_tolerance_is_configurable():
    rdk = _Rdk()
    nudged = pose_to_T(robomath.rotz(np.radians(0.3)))
    rdk.project = _Project(program=_Program(_native_instructions(rotation=nudged)))

    with pytest.raises(RuntimeError, match=r"0\.3 deg"):
        _build(_io(rdk), "TasniCylinder_QUICK_tight_L001",
               maximum_pose_error_deg=0.1)


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


def test_the_valve_may_not_open_while_the_nozzle_is_still_at_the_standoff():
    """Safety: extruding 40 mm above the plane would dump material in mid-air."""
    rdk = _Rdk()
    rdk.project = _Project(
        program=_Program(_native_instructions(air_on_at_standoff=True)))

    with pytest.raises(RuntimeError, match="valve"):
        _build(_io(rdk), "TasniCylinder_QUICK_highvalve_L001")


def test_the_valve_may_not_stay_open_while_the_nozzle_lifts():
    rdk = _Rdk()
    rdk.project = _Project(
        program=_Program(_native_instructions(valve_lifts_before_off=True)))

    with pytest.raises(RuntimeError, match="valve"):
        _build(_io(rdk), "TasniCylinder_QUICK_liftvalve_L001")


def test_a_missing_valve_event_is_rejected():
    rdk = _Rdk()
    without = [row for row in _native_instructions() if row[0] != AIR_OFF]
    rdk.project = _Project(program=_Program(without))

    with pytest.raises(RuntimeError, match="valve"):
        _build(_io(rdk), "TasniCylinder_QUICK_novalve_L001")


def test_the_measured_valve_placement_is_accepted_and_reported():
    rdk = _Rdk()

    result = _build(_io(rdk), "TasniCylinder_QUICK_valveok_L001")

    assert result["air_on_instruction"] == 8
    assert result["air_off_instruction"] == len(rdk.project.program.instructions) - 3


def test_generation_fails_loudly_when_robodk_refuses_the_path():
    rdk = _Rdk()
    rdk.project = _Project(setup_statuses=[-5.0])

    with pytest.raises(RuntimeError, match="no feasible start/path"):
        _build(_io(rdk), "TasniCylinder_QUICK_refused_L001")


def test_generation_is_cancellable():
    rdk = _Rdk()
    calls = []

    def check_cancel():
        calls.append(1)
        raise KeyboardInterrupt("cancelled")

    with pytest.raises(KeyboardInterrupt):
        _build(_io(rdk), "TasniCylinder_QUICK_cancel_L001",
               check_cancel=check_cancel)

    assert calls, "cancellation must be checked at least once"


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
