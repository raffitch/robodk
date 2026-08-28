"""Localise the live-print dispatch blocker: does API RunCode() reach the arm?

Cell 2026-08-28: the app dispatches a layer program, RoboDK accepts it, neither the
program nor the robot is ever observed busy, the cell clicks once, the arm does not
move — and right-clicking the SAME program in RoboDK afterwards moves it. See
``docs/live-print-dispatch-handoff-2026-08-28.md``.

This script attaches to the RoboDK window you already have open (the same instance the
app uses — ``attach`` mode) and walks four rungs. Run them IN ORDER and stop at the
first that fails: that rung names the layer the fault lives in.

    py -3.10 tools/dispatch_bisect.py link                 # no motion
    py -3.10 tools/dispatch_bisect.py jog                  # MOVES the arm ~2 deg on A6
    py -3.10 tools/dispatch_bisect.py trivial              # MOVES: 2-instruction program
    py -3.10 tools/dispatch_bisect.py program <NAME>       # MOVES: the kept layer program

How to read the result:

    jog moves, trivial does not      -> the driver + KRL loop + pendant are FINE.
                                        API *program* execution is the fault.
    jog does not move                -> the fault is below RoboDK: driver, KUKAVARPROXY,
                                        or RoboDKsync570 not running on the pendant.
                                        (Check the pendant NOW, before anything else.)
    trivial moves, program does not   -> the generated Curve Follow program is the fault,
                                        not the dispatch path.
    all move                          -> the fault is in the app's state/timing, not the
                                        API. Compare this script's numbers with the
                                        app's log for the same program.

Nothing here prints material: no valve program is ever called. ``jog`` and ``trivial``
DO move the arm — stand clear, keep the e-stop in hand, and make sure the path is free.
"""
from __future__ import annotations

import sys
import time

DEFAULT_ROBOT = "KUKA KR150 R2700"   # tasni.core.config.RoboDKConfig.robot_name
JOG_AXIS = 6                         # A6: wrist roll, the safest 2 degrees on this cell
JOG_DEG = 2.0
WATCH_S = 10.0
BISECT_PREFIX = "TasniBisect_"


def _rdk():
    """Attach to the RUNNING RoboDK window (never -NEWINSTANCE: we want the app's)."""
    from robolink import Robolink

    return Robolink()


def _robot(rdk, name: str):
    import robolink as rl

    robot = rdk.Item(name, rl.ITEM_TYPE_ROBOT)
    if not robot.Valid():
        names = [i.Name() for i in rdk.ItemList(rl.ITEM_TYPE_ROBOT)]
        raise SystemExit(f"robot {name!r} not found. Robots in the station: {names}")
    return robot


def _report_link(rdk, robot) -> None:
    import robolink as rl

    status, message = robot.ConnectedState()
    ip, port, *_ = robot.ConnectionParams()
    print(f"  robot            : {robot.Name()}")
    print(f"  driver link      : status={status} "
          f"({'READY' if status == rl.ROBOTCOM_READY else 'NOT READY'}) {message!r}")
    print(f"  controller       : {ip}:{port}")
    print(f"  station run mode : {rdk.RunMode()} "
          f"(1=SIMULATE, 6=RUN_ROBOT — 6 is what moves the arm)")
    print(f"  joints           : {[round(v, 2) for v in robot.Joints().list()]}")


def _watch(robot, program=None, seconds: float = WATCH_S) -> bool:
    """Poll Busy() and joints. Returns True if the joints ever changed.

    The joints are the point: RoboDK's model can advance while the controller does
    nothing, but if the joints never change even in the MODEL, RoboDK never ran it.
    """
    start = robot.Joints().list()
    moved = False
    seen_busy = False
    t0 = time.time()
    while time.time() - t0 < seconds:
        now = robot.Joints().list()
        delta = max(abs(a - b) for a, b in zip(now, start))
        busy_p = bool(program.Busy()) if program is not None else False
        busy_r = bool(robot.Busy())
        seen_busy = seen_busy or busy_p or busy_r
        moved = moved or delta > 0.01
        print(f"    t={time.time() - t0:5.2f}s  prog.Busy={int(busy_p)}  "
              f"robot.Busy={int(busy_r)}  max|dJ|={delta:7.3f} deg")
        if moved and not (busy_p or busy_r) and time.time() - t0 > 1.0:
            break
        time.sleep(0.25)
    print(f"    -> ever busy: {seen_busy}   model joints changed: {moved}")
    print("    -> ASK THE OPERATOR: did the PHYSICAL arm move? "
          "(the model moving is not the arm moving)")
    return moved


def _dispatch(rdk, program, real_robot: bool = True) -> int:
    import robolink as rl

    program.setRunType(rl.PROGRAM_RUN_ON_ROBOT if real_robot
                       else rl.PROGRAM_RUN_ON_SIMULATOR)
    rdk.setRunMode(rl.RUNMODE_RUN_ROBOT if real_robot else rl.RUNMODE_SIMULATE)
    read_back = rdk.RunMode()
    try:
        count = program.InstructionCount()
    except Exception:
        count = None
    code = program.RunCode()
    print(f"    setRunMode -> read back {read_back} "
          f"{'(OK)' if read_back == (6 if real_robot else 1) else '<-- NOT WHAT WE SET'}")
    print(f"    RunCode()  -> {code}   (program has {count} instructions; RoboDK "
          "documents this as the number that passed its pre-run check)")
    if code == 0 and count:
        print("    !! RunCode cleared ZERO instructions — RoboDK accepted the call "
              "and refused the program. This is the fault, and it is RoboDK-side.")
    return code


def cmd_link(rdk, robot, _args) -> None:
    print("[link] state only — no motion.")
    _report_link(rdk, robot)
    print("\n  If status is not READY, stop here: connect the driver first.")
    print("  If it IS ready, that only proves the socket to KUKAVARPROXY is up —")
    print("  it does NOT prove RoboDKsync570 is selected and cycling on the pendant.")


def cmd_jog(rdk, robot, _args) -> None:
    """Direct driver motion: no program, no machining project, no valve."""
    import robolink as rl

    print(f"[jog] MOVES the arm {JOG_DEG} deg on A{JOG_AXIS} via the driver "
          "(CASE 2 / PTP COM_E6AXIS). No program is involved.")
    _report_link(rdk, robot)
    input("  Path clear, e-stop in hand? ENTER to move, Ctrl-C to abort: ")
    rdk.setRunMode(rl.RUNMODE_RUN_ROBOT)
    print(f"    setRunMode -> read back {rdk.RunMode()}")
    joints = robot.Joints().list()
    joints[JOG_AXIS - 1] += JOG_DEG
    robot.MoveJ(joints)          # blocking on the driver
    print(f"    MoveJ returned; joints now "
          f"{[round(v, 2) for v in robot.Joints().list()]}")
    print("  If the ARM moved: driver, KUKAVARPROXY and the pendant are all fine.")
    print("  If it did NOT: the fault is below RoboDK. Check the pendant now —")
    print("  is RoboDKsync570 selected and running, and what is $OV_PRO?")


def cmd_trivial(rdk, robot, _args) -> None:
    """The simplest possible PROGRAM: one MoveJ. Isolates program dispatch."""
    import robolink as rl

    print("[trivial] MOVES the arm via a 2-instruction program built here.")
    _report_link(rdk, robot)
    input("  Path clear, e-stop in hand? ENTER to build and run, Ctrl-C to abort: ")
    joints = robot.Joints().list()
    joints[JOG_AXIS - 1] += JOG_DEG
    target_name, program_name = BISECT_PREFIX + "Target", BISECT_PREFIX + "Program"
    for name in (program_name, target_name):
        old = rdk.Item(name)
        if old.Valid():
            old.Delete()
    target = rdk.AddTarget(target_name, 0, robot)
    target.setAsJointTarget()
    target.setJoints(joints)
    program = rdk.AddProgram(program_name, robot)
    program.MoveJ(target)
    try:
        code = _dispatch(rdk, program)
        if code >= 0:
            _watch(robot, program)
    finally:
        for name in (program_name, target_name):
            item = rdk.Item(name)
            if item.Valid():
                item.Delete()
        print(f"    cleaned up {program_name} / {target_name}")
    print("  Arm moved here but not for a layer program -> the generated Curve Follow")
    print("  program is the fault. Arm did not move here but 'jog' worked -> API")
    print("  program execution is the fault, for every program.")


def cmd_program(rdk, robot, args) -> None:
    """Run an EXISTING program (the layer the app left in the station)."""
    import robolink as rl

    if not args:
        programs = [i.Name() for i in rdk.ItemList(rl.ITEM_TYPE_PROGRAM)]
        raise SystemExit("give a program name. Programs in the station:\n  "
                         + "\n  ".join(programs))
    name = args[0]
    program = rdk.Item(name, rl.ITEM_TYPE_PROGRAM)
    if not program.Valid():
        raise SystemExit(f"program {name!r} not found")
    print(f"[program] MOVES the arm: dispatching {name!r} exactly as the app does.")
    _report_link(rdk, robot)
    print("  NOTE: a layer program opens its valve at path start. Make sure that is "
          "acceptable, or use a program you know is motion-only.")
    input("  Path clear, e-stop in hand? ENTER to dispatch, Ctrl-C to abort: ")
    code = _dispatch(rdk, program)
    if code >= 0:
        _watch(robot, program)
    print("  Now, WITHOUT touching the pendant, right-click the same program in")
    print("  RoboDK and Run it. If that moves the arm and this did not, the")
    print("  difference is inside RoboDK's API path — capture its driver log")
    print("  (Connect > Connect robot > show log) across both attempts.")


COMMANDS = {"link": cmd_link, "jog": cmd_jog,
            "trivial": cmd_trivial, "program": cmd_program}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        print(f"commands: {', '.join(COMMANDS)}")
        return 2
    command, rest = argv[0], argv[1:]
    robot_name = DEFAULT_ROBOT
    if "--robot" in rest:
        i = rest.index("--robot")
        robot_name = rest[i + 1]
        rest = rest[:i] + rest[i + 2:]
    rdk = _rdk()
    robot = _robot(rdk, robot_name)
    print(f"attached to RoboDK, station {rdk.ActiveStation().Name()!r}\n")
    COMMANDS[command](rdk, robot, rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
