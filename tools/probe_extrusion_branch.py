"""Cell diagnostic: which RoboDK route yields a NATIVE INLINE extrusion program
that stays on the parked (neutral) wrist branch?

Background: ``RdkIO.create_extrusion_layer_program`` currently discards RoboDK's
own Curve Follow solve and rebuilds the layer from one station target per
waypoint, because the native solve turns axis 4 by ~180 deg away from the parked
pose.  The operator rejects per-point targets outright (see
``docs/extrusion-wrist-flip-handoff.md``), so this probe measures the candidate
routes side by side, in ONE process, against the real station:

* **Probe A** - native Curve Follow Project, swept over
  ``tool x path-to-tool seed``.  The decisive readout is whether the generated
  first path instruction carries the SAME pose as the commanded one (RoboDK only
  picked the wrong IK branch) or a pose rotated ~180 deg (the generated pose
  itself is wrong).
* **Probe R** - the seed that wins probe A only reproduces the commanded roll
  because this cell's commanded yaw is ~90 deg.  RoboDK MIRRORS the seeded roll
  (measured: generated RPW = ``[-r, p, 180 - w]``), so probe R re-runs the seed at
  several deliberately rolled orientations and compares it against a seed built
  from the pre-un-mirrored orientation.
* **Probe B** - KUKA ``.src`` import via ``Robolink.AddFile``, both hand-authored
  KRL (B1) and a RoboDK post-processor round trip (B2).  This is the shape of the
  operator's known-good ``myprog``: inline cartesian moves, no targets.  The
  import half is opt-in (``--try-src-import``): RoboDK opens a MODAL import dialog
  for a ``.src``, so headless it blocks forever and the dialog lands on the
  operator's screen.  Writing the ``.src`` (``Item.MakeProgram``) works fine.
* **Probe C** - one-target neutral prefix on Probe A's best program (fallback).

Measured verdicts on this cell (2026-08-27), full numbers in the repo handoff:
probe A **works** with the un-mirrored seed (zero targets, native one-curve
program, path start lands exactly on the parked joints); probe B is **blocked**
headless by that modal dialog; probe C **fails** - a neutral joint target in front
of the flipped program does not pull the following inline moves onto the branch,
because those inline poses are themselves rotated ~180 deg about the tool axis.

Safety: this NEVER attaches to the operator's RoboDK GUI and NEVER saves the
station.  It launches a private, headless, *licensed* instance (no ``-SKIPINI``,
which would drop the license and silently degrade Curve Follow) and loads
``Tasni.rdk`` read-only.  The robot is only moved in that private simulation.

Usage::

    py -3.10 tools/probe_extrusion_branch.py
    py -3.10 tools/probe_extrusion_branch.py --station "C:/path/Tasni.rdk" --probes A,B
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
import traceback

import numpy as np

import robolink
import robodk.robomath as rm

# --- the job the handoff measured -------------------------------------------
DEFAULT_STATION = r"D:\DesktopStuff\RAFFI NO TOUCH\backuprobodk\RoboDkClaude\Tasni.rdk"
LICENSED_ISOLATED_ARGS = ["-NEWINSTANCE", "-NOUI", "-EXIT_LAST_COM"]
PARKED_JOINTS = [89.22, -74.25, 147.96, 0.21, -42.52, 0.63]
WORK_FRAME_NAME = "Tasni Work Frame"
PRINT_TOOL_NAME = "LongCalibTool"
MIRROR_TOOL_NAME = "TasniPrintTCP"
AIR_ON, AIR_OFF = "TasniDryAirOn", "TasniDryAirOff"

RADIUS_MM = 37.5
POINTS_PER_CIRCLE = 180
APPROACH_MM = 40.0
RETRACT_MM = 60.0
SPEED_MM_S = 20.0
TRAVEL_MM_S = 200.0
ROUNDING_MM = 1.0
WRIST_LIMIT_DEG = 90.0

LOG_LINES: list[str] = []


def say(message: str = "") -> None:
    print(message, flush=True)
    LOG_LINES.append(message)


# --- small numeric helpers (mirrors of the ones in tasni/core/rdk_io.py) -----

def T_of(pose) -> np.ndarray:
    return np.array(pose.Rows(), dtype=float)


def pose_of(T) -> rm.Mat:
    return rm.Mat(np.asarray(T, dtype=float).tolist())


def jvals(joints) -> list[float]:
    try:
        return [float(v) for v in np.asarray(joints.list(), dtype=float).ravel()]
    except Exception:
        return [float(v) for v in np.asarray(joints, dtype=float).ravel()]


def wrap180(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def wrist_deltas(joints, reference) -> list[float]:
    a, b = jvals(joints), jvals(reference)
    return [wrap180(a[i] - b[i]) for i in (3, 4, 5)]


def fmt(values, digits: int = 2) -> str:
    return "[" + ", ".join(f"{float(v):.{digits}f}" for v in values) + "]"


def rot_delta(R_reference: np.ndarray, R_actual: np.ndarray):
    """Angle (deg) and axis of the rotation taking ``R_reference`` to ``R_actual``."""
    R = np.asarray(R_reference, float).T @ np.asarray(R_actual, float)
    cos = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    angle = math.degrees(math.acos(cos))
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:                      # 0 or 180 deg: recover the axis from R + I
        if angle < 1.0:
            return angle, np.array([0.0, 0.0, 1.0])
        values, vectors = np.linalg.eigh((R + np.eye(3)) / 2.0)
        axis = vectors[:, int(np.argmax(values))]
    else:
        axis = axis / norm
    return angle, axis


def name_the_rotation(angle: float, axis: np.ndarray) -> str:
    """Human verdict for a relative rotation, in the reference frame's own axes."""
    if angle < 1.0:
        return "SAME POSE (identical orientation)"
    labels = {0: "local X", 1: "local Y", 2: "local Z"}
    dominant = int(np.argmax(np.abs(axis)))
    purity = abs(float(axis[dominant])) / max(1e-9, float(np.linalg.norm(axis)))
    where = labels[dominant] if purity > 0.95 else f"axis {fmt(axis, 3)}"
    return f"ROTATED {angle:.1f} deg about {where}"


def instruction_rows(program, count: int) -> list[str]:
    rows = []
    types = {value: key for key, value in vars(robolink).items()
             if key.startswith("INS_TYPE_")}
    moves = {getattr(robolink, "MOVE_TYPE_JOINT", 1): "MoveJ",
             getattr(robolink, "MOVE_TYPE_LINEAR", 2): "MoveL",
             getattr(robolink, "MOVE_TYPE_CIRCULAR", 3): "MoveC"}
    for index in range(min(count, program.InstructionCount())):
        name, instype, movetype, isjoint, target, joints = program.Instruction(index)
        row = (f"    [{index}] {name!r} type={types.get(instype, instype)}"
               f" move={moves.get(movetype, movetype)} is_joint_target={isjoint}")
        if target is not None:
            row += "\n         pose XYZRPW=" + fmt(rm.pose_2_xyzrpw(target))
        if joints is not None:
            values = jvals(joints)
            if values:
                row += "\n         joints=" + fmt(values)
        rows.append(row)
    return rows


def first_moves(program, limit: int = 3):
    out = []
    for index in range(program.InstructionCount()):
        record = program.Instruction(index)
        if record[1] == robolink.INS_TYPE_MOVE:
            out.append((index, record))
            if len(out) >= limit:
                break
    return out


def path_joint_report(program, parked) -> dict:
    """Sample the interpolated joint path; return worst wrist deviation vs parked."""
    message, joint_list, status = program.InstructionListJoints(
        mm_step=10.0, deg_step=3.0, collision_check=0)
    if int(status) < 0:
        return {"ok": False, "note": f"status {status}: {message}"}
    samples = [jvals(sample) for sample in joint_list]
    samples = [s for s in samples if len(s) >= 6]
    if not samples:
        return {"ok": False, "note": "no joint samples returned"}
    worst = [max(abs(wrap180(s[axis] - parked[axis])) for s in samples)
             for axis in (3, 4, 5)]
    return {"ok": True, "note": str(message or "").strip(), "samples": len(samples),
            "dA4": round(worst[0], 2), "dA5": round(worst[1], 2),
            "dA6": round(worst[2], 2)}


# --- station / geometry setup ------------------------------------------------

def build_circle(centre_xy, plane_z: float) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, POINTS_PER_CIRCLE + 1)
    x = centre_xy[0] + RADIUS_MM * np.cos(theta)
    y = centre_xy[1] + RADIUS_MM * np.sin(theta)
    x[-1], y[-1] = x[0], y[0]
    return np.column_stack((x, y, np.full_like(x, float(plane_z))))


def neutral_branch(robot, T_frame, toolpose, framepose, parked, parked_config):
    """Every IK branch for a TCP pose in the work frame, filtered to the parked
    wrist branch exactly the way ``solve_joints_on_neutral_branch`` filters."""
    solutions = robot.SolveIK_All(pose_of(T_frame), toolpose, framepose)
    accepted, total = [], 0
    for candidate in solutions:
        values = jvals(candidate)
        if len(values) < 6:
            continue
        total += 1
        values = [parked[i] + wrap180(values[i] - parked[i]) for i in range(6)]
        if abs(values[3] - parked[3]) > WRIST_LIMIT_DEG:
            continue
        if abs(values[5] - parked[5]) > WRIST_LIMIT_DEG:
            continue
        config = tuple(int(round(v)) for v in robot.JointsConfig(values).list()[:3])
        if parked_config is not None and config != parked_config:
            continue
        accepted.append(values)
    return accepted, total


def ensure_valve_programs(rdk, robot) -> None:
    for name, state in ((AIR_ON, "ON"), (AIR_OFF, "OFF")):
        item = rdk.Item(name, robolink.ITEM_TYPE_PROGRAM)
        if item.Valid():
            continue
        program = rdk.AddProgram(name, robot)
        program.RunInstruction(f"PROBE mock valve {state}; physical outputs blocked",
                               robolink.INSTRUCTION_COMMENT)


def drop(rdk, name: str) -> None:
    for item_type in (robolink.ITEM_TYPE_PROGRAM, robolink.ITEM_TYPE_MACHINING,
                      robolink.ITEM_TYPE_OBJECT, robolink.ITEM_TYPE_TARGET,
                      robolink.ITEM_TYPE_TOOL):
        item = rdk.Item(name, item_type)
        if item.Valid():
            item.Delete()


def target_count(rdk) -> int:
    return len(rdk.ItemList(robolink.ITEM_TYPE_TARGET))


def guarded(call, seconds: float):
    """Run ``call`` on a worker thread; give up after ``seconds``.

    ``Robolink.AddFile`` blocks on RoboDK's own 60 s socket deadline, and a
    headless instance that pops a modal import wizard never answers at all.  A
    hang there must not cost the results already gathered, and the RoboDK socket
    is desynchronised afterwards, so the caller stops issuing RPCs.
    """
    import threading

    box = {}

    def run():
        try:
            box["value"] = call()
        except Exception as error:                      # noqa: BLE001 - reported
            box["error"] = f"{type(error).__name__}: {error}"

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        return "TIMEOUT", None
    if "error" in box:
        return "ERROR", box["error"]
    return "OK", box["value"]


def robodk_windows(pid: int) -> list:
    """Every top-level window title owned by ``pid`` (Windows only).

    Evidence for the difference between "the importer is slow" and "the importer
    opened a dialog a ``-NOUI`` instance can never show anyone".
    """
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return ["<ctypes unavailable>"]
    titles = []
    user32 = ctypes.windll.user32
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def each(hwnd, _lparam):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value != pid:
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        titles.append("%r visible=%s class=%s"
                      % (buffer.value, bool(user32.IsWindowVisible(hwnd)),
                         _window_class(hwnd)))
        return True

    try:
        user32.EnumWindows(callback_type(each), 0)
    except Exception as error:
        return ["<enumeration failed: %s>" % error]
    return titles


def _window_class(hwnd) -> str:
    import ctypes

    buffer = ctypes.create_unicode_buffer(256)
    ctypes.windll.user32.GetClassNameW(hwnd, buffer, 256)
    return buffer.value


def robodk_pid(rdk) -> int:
    try:
        return int(rdk.NEW_INSTANCE.pid)
    except Exception:
        return 0


# --- PROBE A ------------------------------------------------------------------

def probe_a(context) -> list:
    rdk, robot, frame = context["rdk"], context["robot"], context["frame"]
    parked = context["parked"]
    points = context["points"]
    surface_normal = np.array([0.0, 0.0, 1.0])   # +Z of the work frame

    seeds = [
        ("commanded", np.eye(4)),
        ("commanded*rotx(pi)", T_of(rm.rotx(math.pi))),
        ("commanded*rotz(pi)", T_of(rm.rotz(math.pi))),
        ("commanded*rotx(pi)*rotz(pi)", T_of(rm.rotx(math.pi) * rm.rotz(math.pi))),
        ("identity", None),
    ]
    results = []
    combo = 0
    for tool_name in (PRINT_TOOL_NAME, MIRROR_TOOL_NAME):
        tool = context["tools"][tool_name]
        R_commanded = context["orientation"][tool_name][:3, :3]
        for seed_label, seed_T in seeds:
            combo += 1
            label = f"A{combo:02d} tool={tool_name} seed={seed_label}"
            base = "SpikeA%02d" % combo
            row = {"label": label, "base": base, "tool": tool_name, "seed": seed_label}
            targets_before = target_count(rdk)
            started = time.time()
            try:
                for suffix in ("", "_Curve", "_Settings"):
                    drop(rdk, base + suffix)
                vertices = np.column_stack(
                    (points, np.repeat(surface_normal.reshape(1, 3), len(points), axis=0)))
                curve = rdk.AddCurve(vertices.tolist(),
                                     projection_type=robolink.PROJECTION_NONE)
                if not curve.Valid():
                    raise RuntimeError("AddCurve failed")
                curve.setName(base + "_Curve")
                curve.setParent(frame)

                project = rdk.AddMachiningProject(base + "_Settings", robot)
                if not project.Valid():
                    raise RuntimeError("AddMachiningProject failed")
                project.setPoseFrame(frame)
                project.setPoseTool(tool)
                project.setJoints(parked)
                project.setParam("Machining", {
                    "Algorithm": 0, "ApproachRetractAll": 1, "AutoUpdate": 0,
                    "AvoidCollisions": 0, "FollowAngleOn": 0, "FollowRealignOn": 0,
                    "FollowStepOn": 0, "JoinCurvesTol": 0.1,
                    "PointApproach": float(APPROACH_MM), "RapidApproachRetract": 1,
                    "RotZ_Range": 0, "SpeedOperation": float(SPEED_MM_S),
                    "SpeedRapid": float(TRAVEL_MM_S), "TurntableActive": 0,
                    "VisibleNormals": 1,
                })
                project.setParam("ProgEvents", {
                    "CallPathStart": AIR_ON, "CallPathStartOn": 1,
                    "CallPathFinish": AIR_OFF, "CallPathFinishOn": 1,
                    "RapidSpeed": float(TRAVEL_MM_S), "Rounding": float(ROUNDING_MM),
                    "RoundingOn": 1,
                })
                project.setParam("Approach", "NTS %.6f 0 0" % APPROACH_MM)
                project.setParam("Retract", "NTS %.6f 0 0" % RETRACT_MM)

                if seed_T is None:
                    seed_pose = rm.eye(4)
                else:
                    seed_pose = pose_of(context["orientation"][tool_name] @ seed_T)
                project.setPose(seed_pose)

                program, status = project.setMachiningParameters(part=curve)
                row["setup_status"] = float(status)
                row["generated"] = bool(program.Valid()) and float(status) >= 0
                if not row["generated"]:
                    row["verdict"] = "no program generated"
                    results.append(row)
                    say(f"  {label}: setup_status={status} -> NO PROGRAM "
                        f"({time.time() - started:.1f}s)")
                    continue
                program.setName(base)
                row["instructions"] = program.InstructionCount()
                row["first_moves"] = []
                for index, record in first_moves(program, 3):
                    mname, _t, movetype, isjoint, target, joints = record
                    row["first_moves"].append({
                        "index": index, "name": mname, "movetype": movetype,
                        "is_joint_target": isjoint,
                        "xyzrpw": rm.pose_2_xyzrpw(target) if target is not None else None,
                        "joints": jvals(joints) if joints is not None else None})
                # Decide POSE-vs-BRANCH on the first move that lands on the curve.
                verdict = "no move instruction lands on the curve"
                approach_note = "n/a"
                for index in range(program.InstructionCount()):
                    record = program.Instruction(index)
                    if record[1] != robolink.INS_TYPE_MOVE or record[4] is None:
                        continue
                    T_ins = T_of(record[4])
                    distance = float(np.min(np.linalg.norm(points - T_ins[:3, 3], axis=1)))
                    height = float(T_ins[2, 3] - points[0][2])
                    if approach_note == "n/a":
                        approach_note = ("first move Z is %+.1f mm vs the plane (%s)"
                                         % (height, "ABOVE" if height > 0 else "BELOW"))
                    if distance <= 1.0:
                        angle, axis = rot_delta(R_commanded, T_ins[:3, :3])
                        verdict = name_the_rotation(angle, axis)
                        row["pose_delta_deg"] = round(angle, 2)
                        row["generated_toolZ_in_frame"] = [round(float(v), 3)
                                                           for v in T_ins[:3, 2]]
                        row["pose_match_index"] = index
                        break
                row["verdict"] = verdict
                row["approach"] = approach_note
                row.update({("path_" + k): v for k, v in
                            path_joint_report(program, parked).items()})
                row["targets_delta"] = target_count(rdk) - targets_before
                say(f"  {label}: status={status} instr={row['instructions']} | {verdict}"
                    f" | dA4={row.get('path_dA4')} dA5={row.get('path_dA5')}"
                    f" dA6={row.get('path_dA6')} | targets{row['targets_delta']:+d}"
                    f" | {approach_note} | {time.time() - started:.1f}s")
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                row["generated"] = False
                row["verdict"] = "exception"
                say(f"  {label}: EXCEPTION {row['error']}")
            results.append(row)
    return results


# --- PROBE R ------------------------------------------------------------------

MIRROR_S = np.diag([1.0, -1.0, 1.0, 1.0])


def seed_source_matrix(commanded: np.ndarray) -> np.ndarray:
    """Invert, in matrix form, the mirror RoboDK imposes on the seeded roll.

    Measured: seeding ``R_in . rotx(pi) . rotz(pi)`` makes RoboDK generate
    ``R_out = Rz(180) . S . R_in . S`` with ``S = diag(1, -1, 1)``.  That map is
    its own inverse, so feeding it the commanded orientation yields the source
    whose generated result IS the commanded orientation.  Unlike the RPW form it
    never decomposes into Euler angles, so it cannot pick up a gimbal artefact
    when the commanded orientation is tilted off the surface normal.
    """
    return T_of(rm.rotz(math.pi)) @ MIRROR_S @ commanded @ MIRROR_S


def seed_source_rpw(commanded: np.ndarray) -> np.ndarray:
    """The same inverse expressed on RPW components: [r, p, w] -> [-r, p, 180-w]."""
    r, p, w = rm.pose_2_xyzrpw(pose_of(commanded))[3:]
    return T_of(rm.xyzrpw_2_pose([0.0, 0.0, 0.0, -r, p, 180.0 - w]))


def seed_from_source(seed_source: np.ndarray) -> np.ndarray:
    return seed_source @ T_of(rm.rotx(math.pi)) @ T_of(rm.rotz(math.pi))


def generate_native_program(context, base: str, tool, commanded: np.ndarray,
                            seed: np.ndarray):
    """Build curve + Curve Follow Project exactly as production does and return
    ``(program, status)`` keeping RoboDK's own generated instructions."""
    rdk, frame, parked = context["rdk"], context["frame"], context["parked"]
    points = context["points"]
    for suffix in ("", "_Curve", "_Settings"):
        drop(rdk, base + suffix)
    surface_normal = np.array([0.0, 0.0, 1.0])      # +Z of the work frame
    vertices = np.column_stack(
        (points, np.repeat(surface_normal.reshape(1, 3), len(points), axis=0)))
    curve = rdk.AddCurve(vertices.tolist(), projection_type=robolink.PROJECTION_NONE)
    curve.setName(base + "_Curve")
    curve.setParent(frame)
    project = rdk.AddMachiningProject(base + "_Settings", context["robot"])
    project.setPoseFrame(frame)
    project.setPoseTool(tool)
    project.setJoints(parked)
    project.setParam("Machining", {
        "Algorithm": 0, "ApproachRetractAll": 1, "AutoUpdate": 0,
        "AvoidCollisions": 0, "FollowAngleOn": 0, "FollowRealignOn": 0,
        "FollowStepOn": 0, "JoinCurvesTol": 0.1,
        "PointApproach": float(APPROACH_MM), "RapidApproachRetract": 1,
        "RotZ_Range": 0, "SpeedOperation": float(SPEED_MM_S),
        "SpeedRapid": float(TRAVEL_MM_S), "TurntableActive": 0, "VisibleNormals": 1,
    })
    project.setParam("ProgEvents", {
        "CallPathStart": AIR_ON, "CallPathStartOn": 1,
        "CallPathFinish": AIR_OFF, "CallPathFinishOn": 1,
        "RapidSpeed": float(TRAVEL_MM_S), "Rounding": float(ROUNDING_MM),
        "RoundingOn": 1,
    })
    project.setParam("Approach", "NTS %.6f 0 0" % APPROACH_MM)
    project.setParam("Retract", "NTS %.6f 0 0" % RETRACT_MM)
    project.setPose(pose_of(seed))
    program, status = project.setMachiningParameters(part=curve)
    if program.Valid() and float(status) >= 0:
        program.setName(base)
    return program, float(status)


def measure_program(program, commanded: np.ndarray, points, parked) -> dict:
    """Worst pose error over EVERY move instruction, plus the wrist report."""
    row = {}
    worst_angle, worst_axis, on_curve = -1.0, None, 0
    for index in range(program.InstructionCount()):
        record = program.Instruction(index)
        if record[1] != robolink.INS_TYPE_MOVE or record[4] is None:
            continue
        T_ins = T_of(record[4])
        angle, axis = rot_delta(commanded[:3, :3], T_ins[:3, :3])
        if angle > worst_angle:
            worst_angle, worst_axis = angle, axis
        if float(np.min(np.linalg.norm(points - T_ins[:3, 3], axis=1))) <= 1.0:
            if on_curve == 0:
                row["commanded_rpw"] = [round(v, 2) for v in
                                        rm.pose_2_xyzrpw(pose_of(commanded))[3:]]
                row["generated_rpw"] = [round(v, 2) for v in
                                        rm.pose_2_xyzrpw(record[4])[3:]]
            on_curve += 1
    row["moves_on_curve"] = on_curve
    row["pose_delta_deg"] = round(worst_angle, 3)
    row["verdict"] = (name_the_rotation(worst_angle, worst_axis)
                      if worst_axis is not None else "no move instruction")
    row["first_moves"] = []
    for index, record in first_moves(program, 3):
        mname, _t, movetype, isjoint, target, joints = record
        row["first_moves"].append({
            "index": index, "name": mname, "movetype": movetype,
            "is_joint_target": isjoint,
            "xyzrpw": rm.pose_2_xyzrpw(target) if target is not None else None,
            "joints": jvals(joints) if joints is not None else None})
    row.update({("path_" + k): v for k, v in
                path_joint_report(program, parked).items()})
    return row


def probe_roll(context) -> list:
    """Does a seed TRACK the commanded orientation, or only mirror it?

    At this cell's parked pose the commanded RPW yaw is ~90.69 deg and the
    generated one comes back as ~89.31 = 180 - 90.69.  Those two are 1.4 deg
    apart *by coincidence of being near 90 deg*: RoboDK reflects the seeded roll
    rather than reproducing it, so the naive seed is catastrophically wrong at any
    other orientation.  Sweep three seed forms over yawed AND tilted commanded
    orientations.  The tilted cases are the ones that decide whether the
    coordinate-free matrix inverse may be used, or whether the RPW inverse (which
    decomposes into Euler angles and can gimbal) has to be fenced off by a
    documented tilt limit.
    """
    tool = context["tools"][PRINT_TOOL_NAME]
    base_orientation = context["orientation"][PRINT_TOOL_NAME]
    points, parked = context["points"], context["parked"]
    # (yaw about the tool axis, pitch, roll) applied to the commanded orientation
    orientations = [(0.0, 0.0, 0.0), (20.0, 0.0, 0.0), (-20.0, 0.0, 0.0),
                    (45.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, -10.0, 0.0),
                    (0.0, 10.0, 5.0), (30.0, 10.0, 5.0)]
    results = []
    index = 0
    for yaw, pitch, roll in orientations:
        commanded = (base_orientation
                     @ T_of(rm.rotz(math.radians(yaw)))
                     @ T_of(rm.roty(math.radians(pitch)))
                     @ T_of(rm.rotx(math.radians(roll))))
        for mode in ("plain", "rpw", "matrix"):
            base = "SpikeR%02d" % index
            index += 1
            row = {"yaw": yaw, "pitch": pitch, "roll": roll,
                   "seed_mode": mode, "base": base}
            label = "yaw%+5.1f pitch%+5.1f roll%+5.1f %-6s" % (yaw, pitch, roll, mode)
            try:
                probe_T = commanded.copy()
                probe_T[:3, 3] = points[0]
                accepted, _ = neutral_branch(
                    context["robot"], probe_T, tool.PoseTool(),
                    context["frame_pose"], parked, context["parked_config"])
                row["neutral_solutions_at_start"] = len(accepted)
                seed_source = {"plain": commanded,
                               "rpw": seed_source_rpw(commanded),
                               "matrix": seed_source_matrix(commanded)}[mode]
                program, status = generate_native_program(
                    context, base, tool, commanded, seed_from_source(seed_source))
                row["setup_status"] = status
                if not (program.Valid() and status >= 0):
                    row["verdict"] = "no program generated"
                    say("  %s: status=%s -> NO PROGRAM (neutral IK at start=%s)"
                        % (label, status, row["neutral_solutions_at_start"]))
                    results.append(row)
                    continue
                row.update(measure_program(program, commanded, points, parked))
                say("  %s: worst pose err=%7.3f deg | %s | dA4=%s dA5=%s dA6=%s"
                    " | neutral IK at start=%s"
                    % (label, row["pose_delta_deg"], row["verdict"],
                       row.get("path_dA4"), row.get("path_dA5"),
                       row.get("path_dA6"), row["neutral_solutions_at_start"]))
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                say("  %s: EXCEPTION %s" % (label, row["error"]))
            results.append(row)
    return results


# --- PROBE B ------------------------------------------------------------------

def krl_pose(T) -> str:
    x, y, z, a, b, c = rm.Pose_2_KUKA(pose_of(T))
    return ("{X %.3f,Y %.3f,Z %.3f,A %.3f,B %.3f,C %.3f}" % (x, y, z, a, b, c))


def write_krl(path: str, program_name: str, tool_T, base_T, start_joints,
              sequence, air_calls: bool) -> str:
    axes = ",".join("A%d %.3f" % (i + 1, v)
                    for i, v in enumerate(jvals(start_joints)[:6]))
    lines = [
        "&ACCESS RVP",
        "&REL 1",
        "DEF %s ( )" % program_name,
        ";FOLD INI",
        "  BAS (#INITMOV,0 )",
        ";ENDFOLD (INI)",
        # RoboDK's own KUKA post writes the FRAME: prefix, so match that dialect.
        "$TOOL = {FRAME: %s}" % krl_pose(tool_T)[1:-1],
        "$BASE = {FRAME: %s}" % krl_pose(base_T)[1:-1],
        "$VEL.CP = 0.020",
        "$APO.CDIS = %.3f" % ROUNDING_MM,
        "$ADVANCE = 3",
        "PTP {%s}" % axes,
    ]
    for kind, T in sequence:
        if kind == "air_on":
            if air_calls:
                lines.append("%s()" % AIR_ON)
            continue
        if kind == "air_off":
            if air_calls:
                lines.append("%s()" % AIR_OFF)
            continue
        lines.append("LIN %s C_DIS" % krl_pose(T))
    lines += ["END", ""]
    text = "\r\n".join(lines)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return text


def describe_imported(rdk, program, parked, targets_before) -> dict:
    row = {}
    if program is None or not program.Valid():
        row["imported"] = False
        return row
    row["imported"] = True
    row["name"] = program.Name()
    row["item_type"] = program.Type()
    row["instructions"] = program.InstructionCount()
    row["targets_delta"] = target_count(rdk) - targets_before
    inline = joint_moves = code = 0
    for index in range(program.InstructionCount()):
        _name, instype, _movetype, isjoint, _target, _joints = program.Instruction(index)
        if instype == robolink.INS_TYPE_MOVE:
            if isjoint:
                joint_moves += 1
            else:
                inline += 1
        elif instype == getattr(robolink, "INS_TYPE_CODE", 6):
            code += 1
    row["cartesian_moves"] = inline
    row["joint_moves"] = joint_moves
    row["code_instructions"] = code
    for kind, key in ((robolink.ITEM_TYPE_TOOL, "tool"),
                      (robolink.ITEM_TYPE_FRAME, "frame")):
        try:
            item = program.getLink(kind)
            row[key] = item.Name() if item.Valid() else None
        except Exception:
            row[key] = None
    row["rows"] = instruction_rows(program, 5)
    row.update({("path_" + k): v for k, v in path_joint_report(program, parked).items()})
    try:
        row["update"] = [str(v) for v in program.Update()]
    except Exception as error:
        row["update"] = f"{type(error).__name__}: {error}"
    return row


def probe_b(context, folder: str) -> dict:
    rdk, robot, frame = context["rdk"], context["robot"], context["frame"]
    parked, points = context["parked"], context["points"]
    tool = context["tools"][PRINT_TOOL_NAME]
    tool_T = T_of(tool.PoseTool())
    base_T = context["frame_wrt_base"]
    orientation = context["orientation"][PRINT_TOOL_NAME]
    normal = np.array([0.0, 0.0, 1.0])
    out = {}

    def pose_at(xyz):
        T = orientation.copy()
        T[:3, 3] = np.asarray(xyz, float)
        return T

    picked = [int(round(i)) for i in np.linspace(0, len(points) - 1, 8)]
    sequence = [("move", pose_at(points[0] + normal * APPROACH_MM)),
                ("move", pose_at(points[0])), ("air_on", None)]
    sequence += [("move", pose_at(points[i])) for i in picked[1:]]
    sequence += [("air_off", None),
                 ("move", pose_at(points[picked[-1]] + normal * RETRACT_MM))]

    approach_T = pose_at(points[0] + normal * APPROACH_MM)
    accepted, _total = neutral_branch(robot, approach_T, tool.PoseTool(),
                                      context["frame_pose"], parked,
                                      context["parked_config"])
    if not accepted:
        out["blocked"] = "no neutral-branch IK at the approach point"
        return out
    start_joints = accepted[0]
    out["start_joints"] = fmt(start_joints)

    # --- B1: hand-authored KRL (written now, imported at the very end) --------
    krl_files = []
    for variant, air_calls in (("B1", True), ("B1noair", False)):
        name = "TasniSpike%s" % variant
        drop(rdk, name)
        path = os.path.join(folder, name + ".src")
        text = write_krl(path, name, tool_T, base_T, start_joints, sequence, air_calls)
        out[variant + "_src"] = text.replace("\r\n", "\n").split("\n")
        krl_files.append((variant, path))
    out["_imports"] = krl_files
    say("  B1 hand-authored KRL written (%d files); import is attempted last"
        % len(krl_files))

    # --- B2: RoboDK post-processor round trip ---------------------------------
    scratch = "TasniSpikeB2Src"
    drop(rdk, scratch)
    made_targets = []
    try:
        program = rdk.AddProgram(scratch, robot)
        program.setPoseFrame(frame)
        program.setPoseTool(tool)
        program.setSpeed(float(TRAVEL_MM_S))
        previous = start_joints
        for index, (kind, T) in enumerate(sequence):
            if kind != "move":
                program.RunInstruction(AIR_ON if kind == "air_on" else AIR_OFF,
                                       robolink.INSTRUCTION_CALL_PROGRAM)
                continue
            accepted, _ = neutral_branch(robot, T, tool.PoseTool(), context["frame_pose"],
                                         parked, context["parked_config"])
            if not accepted:
                raise RuntimeError("no neutral-branch IK for scratch waypoint %d" % index)
            solution = min(accepted, key=lambda j: sum(
                wrap180(j[k] - previous[k]) ** 2 for k in range(6)))
            previous = solution
            target_name = "%s_T%02d" % (scratch, index)
            made_targets.append(target_name)
            target = rdk.AddTarget(target_name, frame, robot)
            target.setPose(pose_of(T))
            if index == 0:
                # Only the leading PTP is a joint target -- that is what locks the
                # wrist branch. The post refuses to emit a LIN to a joint target
                # ("Linear movement using joint targets is not supported", which
                # turned every path line into a comment on the first run), so every
                # following waypoint stays cartesian. That is myprog's shape.
                target.setAsJointTarget()
                target.setJoints(solution)
                program.MoveJ(target)
            else:
                program.MoveL(target)
        try:
            robot.setParam("PostProcessor", "KUKA_KRC4")
            out["B2_post_set"] = "KUKA_KRC4"
        except Exception as error:
            out["B2_post_set"] = "could not set post: %s" % error
        made = program.MakeProgram(folder)
        out["B2_make"] = [str(v) for v in
                          (made if isinstance(made, (list, tuple)) else [made])]
        produced = None
        for candidate in sorted(os.listdir(folder)):
            if (candidate.lower().startswith(scratch.lower())
                    and candidate.lower().endswith(".src")):
                produced = os.path.join(folder, candidate)
        out["B2_file"] = produced
        if produced:
            with open(produced, "r", encoding="utf-8", errors="replace") as handle:
                body = handle.read()
            out["B2_src"] = body.replace("\r\n", "\n").split("\n")[:60]
    finally:
        drop(rdk, scratch)
        for target_name in made_targets:
            drop(rdk, target_name)

    if out.get("B2_file"):
        krl_files.append(("B2roundtrip", out["B2_file"]))
    return out


def probe_b_imports(context, attempts) -> dict:
    """AddFile every candidate .src.  Runs LAST: a hang here desynchronises the
    RoboDK socket, so nothing may be measured after the first timeout."""
    rdk, robot, parked = context["rdk"], context["robot"], context["parked"]
    out = {}
    for variant, path in attempts:
        for parent_label, parent in (("robot", robot), ("station", 0)):
            key = "%s_parent_%s" % (variant, parent_label)
            targets_before = target_count(rdk)
            state, value = guarded(lambda p=path, q=parent: rdk.AddFile(p, q), 25.0)
            if state == "TIMEOUT":
                pid = robodk_pid(rdk)
                windows = robodk_windows(pid) if pid else ["<pid unknown>"]
                out[key] = {"imported": False, "blocked": "AddFile did not answer in 25 s",
                            "robodk_pid": pid, "windows": windows}
                say("  %s AddFile(parent=%s): BLOCKED - no answer in 25 s. "
                    "RoboDK windows owned by pid %s: %s"
                    % (variant, parent_label, pid, windows))
                out["_socket_desynchronised"] = True
                return out
            if state == "ERROR":
                out[key] = {"imported": False, "error": value}
                say("  %s AddFile(parent=%s): EXCEPTION %s" % (variant, parent_label, value))
                out["_socket_desynchronised"] = True
                return out
            record = describe_imported(rdk, value, parked, targets_before)
            out[key] = record
            if record.get("imported"):
                say("  %s AddFile(parent=%s): imported %r, %s instructions, "
                    "%s cartesian / %s joint moves, %s code, targets%+d, "
                    "dA4=%s dA5=%s dA6=%s"
                    % (variant, parent_label, record.get("name"),
                       record.get("instructions"), record.get("cartesian_moves"),
                       record.get("joint_moves"), record.get("code_instructions"),
                       record.get("targets_delta", 0), record.get("path_dA4"),
                       record.get("path_dA5"), record.get("path_dA6")))
                created = record.get("name")
                if created:
                    drop(rdk, created)
                break
            say("  %s AddFile(parent=%s): NOT IMPORTED" % (variant, parent_label))
    return out


# --- PROBE C ------------------------------------------------------------------

def probe_c(context, base: str, tool_name: str) -> dict:
    rdk, robot, frame = context["rdk"], context["robot"], context["frame"]
    parked, points = context["parked"], context["points"]
    program = rdk.Item(base, robolink.ITEM_TYPE_PROGRAM)
    out = {"program": base}
    if not program.Valid():
        out["error"] = "program not found"
        return out
    tool = context["tools"][tool_name]
    move_index = None
    for index in range(program.InstructionCount()):
        if program.Instruction(index)[1] == robolink.INS_TYPE_MOVE:
            move_index = index
            break
    if move_index is None:
        out["error"] = "no move instruction"
        return out
    out["original_first_move_index"] = move_index
    approach_T = context["orientation"][tool_name].copy()
    approach_T[:3, 3] = points[0] + np.array([0.0, 0.0, APPROACH_MM])
    accepted, _ = neutral_branch(robot, approach_T, tool.PoseTool(),
                                 context["frame_pose"], parked, context["parked_config"])
    if not accepted:
        out["error"] = "no neutral-branch IK at the approach point"
        return out
    target_name = "SpikeC_NeutralStart"
    drop(rdk, target_name)
    targets_before = target_count(rdk)
    target = rdk.AddTarget(target_name, frame, robot)
    target.setPose(pose_of(approach_T))
    target.setAsJointTarget()
    target.setJoints(accepted[0])
    program.InstructionSelect(move_index - 1)
    program.MoveJ(target)                    # inserted right after the selection
    program.InstructionDelete(move_index + 1)  # the original MoveJ, now shifted by 1
    out["targets_delta"] = target_count(rdk) - targets_before
    out["rows"] = instruction_rows(program, 5)
    out.update({("path_" + k): v for k, v in path_joint_report(program, parked).items()})
    return out


# --- main ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="extrusion inline-program probe")
    parser.add_argument("--station", default=DEFAULT_STATION)
    parser.add_argument("--probes", default="A,R,B,C")
    parser.add_argument("--try-src-import", action="store_true",
                        help="attempt Robolink.AddFile on the generated .src files. "
                             "OFF by default: the importer opens a modal dialog, so "
                             "headless it hangs and the dialog lands on the operator's "
                             "screen for them to cancel")
    parser.add_argument("--force-c", action="store_true",
                        help="run probe C even when probe A already produced a "
                             "neutral-branch program (grafts onto the WORST one)")
    parser.add_argument("--log", default="")
    options = parser.parse_args()
    wanted = {p.strip().upper() for p in options.probes.split(",") if p.strip()}
    folder = tempfile.mkdtemp(prefix="tasni_spike_")

    say("=" * 78)
    say("EXTRUSION INLINE-PROGRAM PROBE (private headless licensed instance)")
    say("station : %s" % options.station)
    say("scratch : %s" % folder)
    say("=" * 78)

    started = time.time()
    rdk = robolink.Robolink(args=LICENSED_ISOLATED_ARGS, quit_on_close=True)
    say("connected to a private RoboDK instance in %.1fs" % (time.time() - started))
    started = time.time()
    station = rdk.AddFile(options.station)
    if not station.Valid():
        say("FATAL: the station did not load")
        return 2
    say("station loaded in %.1fs" % (time.time() - started))

    robot = rdk.Item("", robolink.ITEM_TYPE_ROBOT)
    say("robot   : %r" % robot.Name())
    frame = rdk.Item(WORK_FRAME_NAME, robolink.ITEM_TYPE_FRAME)
    if frame.Valid():
        say("frame   : %r (found in the on-disk station)" % WORK_FRAME_NAME)
    else:
        frame = robot.Parent()
        say("frame   : %r NOT in the on-disk station -> using the robot base frame %r"
            % (WORK_FRAME_NAME, frame.Name()))
    long_tool = rdk.Item(PRINT_TOOL_NAME, robolink.ITEM_TYPE_TOOL)
    if not long_tool.Valid():
        say("FATAL: tool %r not found" % PRINT_TOOL_NAME)
        return 2

    ensure_valve_programs(rdk, robot)
    robot.setJoints(PARKED_JOINTS)
    parked = list(PARKED_JOINTS)
    parked_config = tuple(int(round(v)) for v in robot.JointsConfig(parked).list()[:3])
    say("parked  : %s config(REAR,LOWER,FLIP)=%s" % (fmt(parked), parked_config))
    limits = robot.JointLimits()
    say("limits  : lower=%s upper=%s" % (fmt(jvals(limits[0])), fmt(jvals(limits[1]))))

    # mirrored TCP: same origin, frame rotated 180 deg about its local X
    drop(rdk, MIRROR_TOOL_NAME)
    mirror_tool = robot.AddTool(long_tool.PoseTool() * rm.rotx(math.pi), MIRROR_TOOL_NAME)
    if not mirror_tool.Valid():
        say("FATAL: could not create %r" % MIRROR_TOOL_NAME)
        return 2
    tools = {PRINT_TOOL_NAME: long_tool, MIRROR_TOOL_NAME: mirror_tool}

    robot.setPoseFrame(frame)
    frame_pose = robot.PoseFrame()
    frame_wrt_base = T_of(frame_pose)

    orientation, tcp_xyzrpw = {}, {}
    for tool_name, tool in tools.items():
        robot.setPoseTool(tool)
        robot.setPoseFrame(frame)
        robot.setJoints(parked)
        tcp = robot.Pose()
        xyzrpw = rm.pose_2_xyzrpw(tcp)
        tcp_xyzrpw[tool_name] = xyzrpw
        orientation[tool_name] = T_of(rm.xyzrpw_2_pose([0.0, 0.0, 0.0] + list(xyzrpw[3:6])))
        toolZ = T_of(tcp)[:3, 2]
        say("tool %-15s TCP in frame XYZ=%s RPW=%s toolZ=%s (%s)"
            % (tool_name, fmt(xyzrpw[:3]), fmt(xyzrpw[3:]), fmt(toolZ, 3),
               "UP, away from the work" if toolZ[2] > 0 else "DOWN, into the work"))

    seed_xyz = tcp_xyzrpw[PRINT_TOOL_NAME][:3]
    centre = [seed_xyz[0] - RADIUS_MM, seed_xyz[1]]
    points = build_circle(centre, seed_xyz[2])
    say("path    : circle r=%.1f mm, %d points, centre=%s, plane Z=%.2f (work-frame mm)"
        % (RADIUS_MM, len(points), fmt(centre), seed_xyz[2]))

    # confirm the handoff's claim: exactly one branch passes, for every pose
    robot.setPoseTool(long_tool)
    robot.setJoints(parked)
    sample_indices = [int(round(i)) for i in np.linspace(0, len(points) - 1, 8)]
    probe_poses = [("path%03d" % i, points[i]) for i in sample_indices]
    probe_poses.append(("APPROACH", points[0] + np.array([0.0, 0.0, APPROACH_MM])))
    probe_poses.append(("RETRACT", points[-1] + np.array([0.0, 0.0, RETRACT_MM])))
    checks = []
    for label, xyz in probe_poses:
        T = orientation[PRINT_TOOL_NAME].copy()
        T[:3, 3] = xyz
        accepted, total = neutral_branch(robot, T, long_tool.PoseTool(), frame_pose,
                                         parked, parked_config)
        checks.append((label, total, len(accepted), accepted[0] if accepted else None))
    say("neutral-branch reference (SolveIK_All filtered to the parked branch):")
    for label, total, count, solution in checks:
        line = "    %-10s branches=%-3d accepted=%d" % (label, total, count)
        if solution:
            line += "  joints=%s  dA4/5/6=%s" % (fmt(solution),
                                                 fmt(wrist_deltas(solution, parked)))
        say(line)
    unique = sorted({count for _l, _t, count, _s in checks})
    say("    -> accepted counts across the job: %s%s"
        % (unique, "  (matches the handoff: exactly one branch)" if unique == [1]
           else "  (DIFFERS from the handoff)"))

    context = {"rdk": rdk, "robot": robot, "frame": frame, "frame_pose": frame_pose,
               "frame_wrt_base": frame_wrt_base, "tools": tools,
               "orientation": orientation, "parked": parked,
               "parked_config": parked_config, "points": points}

    results = {}
    if "A" in wanted:
        say("")
        say("-" * 78)
        say("PROBE A - native Curve Follow sweep (tool x path-to-tool seed)")
        say("-" * 78)
        results["A"] = probe_a(context)

    if "R" in wanted:
        say("")
        say("-" * 78)
        say("PROBE R - does the winning seed TRACK the commanded roll or mirror it?")
        say("-" * 78)
        results["R"] = probe_roll(context)

    if "B" in wanted:
        say("")
        say("-" * 78)
        say("PROBE B - KUKA .src import via Robolink.AddFile")
        say("-" * 78)
        try:
            results["B"] = probe_b(context, folder)
        except Exception as error:
            results["B"] = {"error": f"{type(error).__name__}: {error}",
                            "traceback": traceback.format_exc()}
            say("  PROBE B EXCEPTION: %s" % results["B"]["error"])

    neutral_from_a = [row for row in results.get("A", [])
                      if row.get("generated") and row.get("path_ok")
                      and float(row.get("path_dA4", 999)) <= WRIST_LIMIT_DEG]
    if "C" in wanted:
        say("")
        say("-" * 78)
        say("PROBE C - one-target neutral prefix (fallback)")
        say("-" * 78)
        if neutral_from_a and not options.force_c:
            say("  skipped: probe A already produced a neutral-branch program.")
        else:
            if neutral_from_a:
                say("  FORCED (--force-c): probe A already solved this; run only to "
                    "measure what the fallback would do to a wrong-branch program.")
            generated = [row for row in results.get("A", []) if row.get("generated")]
            if not generated:
                say("  skipped: probe A produced no program to graft onto.")
            else:
                # Normally the least-flipped program is the one worth rescuing.
                # When forced past a probe A that already works, graft onto the
                # WORST one instead -- that is the case the fallback exists for.
                pick = max if neutral_from_a else min
                best = pick(generated, key=lambda r: float(r.get("path_dA4", 999)))
                say("  grafting onto %s (%s)" % (best["base"], best["label"]))
                try:
                    results["C"] = probe_c(context, best["base"], best["tool"])
                except Exception as error:
                    results["C"] = {"error": f"{type(error).__name__}: {error}",
                                    "traceback": traceback.format_exc()}
                for key, value in results["C"].items():
                    if key == "rows":
                        for line in value:
                            say(line)
                    else:
                        say("    %s: %s" % (key, value))

    # MEASURED, then confirmed by the operator watching the screen: AddFile on a
    # .src opens a modal import dialog.  A -NOUI instance can never dismiss it, so
    # the call blocks past RoboDK's own 60 s socket deadline and leaves the link
    # desynchronised -- and the dialog surfaces on the operator's display, where
    # they have to cancel it by hand.  This stage is therefore opt-in, runs LAST,
    # and nothing may be measured after it.
    attempts = results.get("B", {}).pop("_imports", None) if "B" in results else None
    if attempts and not options.try_src_import:
        say("")
        say("PROBE B (imports) SKIPPED: Robolink.AddFile on a .src opens a modal "
            "import dialog. Headless it never returns, and the dialog lands on the "
            "operator's screen. Pass --try-src-import to attempt it anyway.")
        attempts = None
    if attempts:
        say("")
        say("-" * 78)
        say("PROBE B (imports) - AddFile on each candidate .src")
        say("-" * 78)
        try:
            results["B"].update(probe_b_imports(context, attempts))
        except Exception as error:
            results["B"]["import_error"] = f"{type(error).__name__}: {error}"
            say("  import stage EXCEPTION: %s" % results["B"]["import_error"])

    # ---- detail dump ---------------------------------------------------------
    say("")
    say("=" * 78)
    say("DETAIL")
    say("=" * 78)
    for row in results.get("A", []):
        say("[%s]  (%s)" % (row["label"], row["base"]))
        for key in ("setup_status", "generated", "instructions", "verdict",
                    "pose_delta_deg", "generated_toolZ_in_frame", "approach",
                    "targets_delta", "path_ok", "path_note", "path_samples",
                    "path_dA4", "path_dA5", "path_dA6", "error"):
            if key in row:
                say("    %s: %s" % (key, row[key]))
        for move in row.get("first_moves", []):
            say("    move[%s] %r movetype=%s is_joint_target=%s"
                % (move["index"], move["name"], move["movetype"],
                   move["is_joint_target"]))
            if move["xyzrpw"]:
                say("        XYZRPW=" + fmt(move["xyzrpw"]))
            if move["joints"]:
                say("        joints=" + fmt(move["joints"]))
    for row in results.get("R", []):
        say("[PROBE R yaw%+.1f pitch%+.1f roll%+.1f %s]  (%s)"
            % (row["yaw"], row["pitch"], row["roll"], row["seed_mode"], row["base"]))
        for key in ("setup_status", "verdict", "pose_delta_deg", "commanded_rpw",
                    "generated_rpw", "moves_on_curve", "neutral_solutions_at_start",
                    "path_dA4", "path_dA5", "path_dA6", "path_samples", "error"):
            if key in row:
                say("    %s: %s" % (key, row[key]))
        for move in row.get("first_moves", []):
            say("    move[%s] %r movetype=%s is_joint_target=%s"
                % (move["index"], move["name"], move["movetype"],
                   move["is_joint_target"]))
            if move["xyzrpw"]:
                say("        XYZRPW=" + fmt(move["xyzrpw"]))
            if move["joints"]:
                say("        joints=" + fmt(move["joints"]))
    if "B" in results:
        say("[PROBE B]")
        for key, value in results["B"].items():
            if isinstance(value, dict):
                say("    %s:" % key)
                for sub_key, sub_value in value.items():
                    if sub_key == "rows":
                        for line in sub_value:
                            say("    " + line)
                    else:
                        say("        %s: %s" % (sub_key, sub_value))
            elif isinstance(value, list):
                say("    %s:" % key)
                for line in value:
                    say("        %s" % line)
            else:
                say("    %s: %s" % (key, value))

    say("")
    say("NOTE: the station was never saved; this ran in a private headless instance.")
    log_path = options.log or os.path.join(folder, "probe_report.txt")
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(LOG_LINES) + "\n")
    say("report written to %s" % log_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
