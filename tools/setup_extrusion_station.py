"""Create/verify fail-safe AirOn and AirOff programs in a RoboDK station.

Uses a private headless RoboDK instance and never runs either program. The I/O
mapping is versioned in ``ExtrusionConfig`` and was recovered from the legacy
paper station; physical use still requires the separate hardware-test interlock.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robodk import robolink

from tasni.core.config import load_config

LICENSED_ISOLATED_ARGS = ["-NEWINSTANCE", "-NOUI", "-EXIT_LAST_COM"]


def connect_for_station_save():
    """Private RoboDK instance that retains the user's license settings.

    The normal extraction helper adds ``-SKIPINI`` for maximal reproducibility,
    but that also made this writer start without the active RoboDK license.  We
    still force a separate headless instance, so this never attaches to the GUI.
    """
    return robolink.Robolink(args=LICENSED_ISOLATED_ARGS, quit_on_close=True)


def expected_instructions(outputs: list[str], value: int) -> list[str]:
    return [f"Set {output}={value}" for output in outputs]


def program_instructions(program) -> list[str]:
    return [str(program.Instruction(i)[0]) for i in range(program.InstructionCount())]


def ensure_program(rdk, robot, name: str, outputs: list[str], value: int, *, replace: bool) -> str:
    expected = expected_instructions(outputs, value)
    existing = rdk.Item(name, robolink.ITEM_TYPE_PROGRAM)
    if existing.Valid():
        actual = program_instructions(existing)
        if actual == expected:
            return "verified"
        if not replace:
            raise RuntimeError(f"{name} exists with different instructions: {actual!r}")
        existing.Delete()
    program = rdk.AddProgram(name, robot)
    for output in outputs:
        program.setDO(output, value)
    actual = program_instructions(program)
    if actual != expected:
        raise RuntimeError(f"failed to create {name}: {actual!r}")
    return "created"


def verify_saved_station(path: Path, cfg) -> None:
    """Reopen the on-disk artifact; RoboDK can reject Save due to licensing."""
    rdk = connect_for_station_save()
    try:
        station = rdk.AddFile(str(path))
        if not station.Valid():
            raise RuntimeError(f"RoboDK could not reopen saved station {path}")
        for name, value in ((cfg.air_on_program, cfg.valve_active_value),
                            (cfg.air_off_program, cfg.valve_inactive_value)):
            program = rdk.Item(name, robolink.ITEM_TYPE_PROGRAM)
            actual = program_instructions(program) if program.Valid() else []
            expected = expected_instructions(cfg.valve_outputs, value)
            if actual != expected:
                raise RuntimeError(
                    f"saved station verification failed for {name}: {actual!r}; "
                    "RoboDK did not persist the requested program instructions")
    finally:
        try:
            rdk.CloseRoboDK()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("station", nargs="?", default="Tasni.rdk")
    parser.add_argument("--output", help="output station (default: <station>.extrusion.rdk)")
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--replace", action="store_true",
                        help="replace conflicting AirOn/AirOff programs")
    args = parser.parse_args()
    source = Path(args.station).resolve()
    if not source.is_file():
        parser.error(f"station not found: {source}")
    if args.inplace and args.output:
        parser.error("choose either --inplace or --output")
    output = source if args.inplace else Path(args.output).resolve() if args.output else source.with_name(source.stem + ".extrusion.rdk")

    app_cfg = load_config()
    cfg = app_cfg.extrusion
    rdk = connect_for_station_save()
    try:
        station = rdk.AddFile(str(source))
        if not station.Valid():
            raise RuntimeError(f"RoboDK could not load {source}")
        robot = rdk.Item(app_cfg.robodk.robot_name, robolink.ITEM_TYPE_ROBOT)
        if not robot.Valid():
            raise RuntimeError("configured robot is missing from the station")
        on = ensure_program(rdk, robot, cfg.air_on_program, cfg.valve_outputs,
                            cfg.valve_active_value, replace=args.replace)
        off = ensure_program(rdk, robot, cfg.air_off_program, cfg.valve_outputs,
                             cfg.valve_inactive_value, replace=args.replace)
        station.Save(str(output))
    finally:
        try:
            rdk.CloseRoboDK()
        except Exception:
            pass
    verify_saved_station(output, cfg)
    print(f"AirOn {on}; AirOff {off}; saved and reopened successfully: {output}")
    print("Programs were not executed. Hardware I/O approval remains required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
