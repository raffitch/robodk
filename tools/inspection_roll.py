"""Arm (or disarm) a FORCED camera roll for the next inspection capture.

    py -3.10 tools/inspection_roll.py            # what is armed right now
    py -3.10 tools/inspection_roll.py 90         # force every pose to roll 90
    py -3.10 tools/inspection_roll.py --disarm   # back to the normal ladder

The roll probe is the only instrument that separates a CAMERA-locked dropout
from a SCENE-locked one: rotate the sensor about its own optical axis and see
whether the dead sectors follow it. See docs/ring2-scan-handoff.md.

Three traps this exists to prevent, all of which silently produce a run that
looks fine and answers nothing:

1. **Adding 90 to the candidate list does nothing.** The default is the FALLBACK
   LADDER ``[0, 180, 90, 270]`` and the generator accepts the FIRST candidate
   that has IK and passes validation -- roll 0 always does, so 90 is never
   reached. A forced roll must be the ONLY candidate, which is what this writes.
2. **A leftover ladder means a silent fallback.** With ``[90, 0]`` a refused 90
   quietly captures at roll 0 and archives a take that looks ordinary. Armed
   here the list is a single value, so a refusal fails the run LOUDLY instead.
3. **The config is read at startup.** Editing it changes nothing until the
   backend restarts, so this always says so.

WRIST COST: a 90 deg roll about a nadir optical axis is essentially 90 deg of
A6, which lands exactly ON ``max_tool_axis_spin_deg``'s default of 90. The gate
is ``abs(delta) > limit``, so exact-90 passes only if floating point cooperates
-- it was refused on this cell before. Give it headroom in the TRIAL's setup
(``maximum_tool_axis_spin_deg``), not in the global default: the looser limit
then applies to one measure-only trial, is recorded in its fingerprint and
trial.json, and never reaches a print path. This tool warns when the armed roll
is close enough to the default limit for that to matter.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "tasni.config.json"
KEY = "inspection_roll_candidates_deg"
# The default ladder, restored on --disarm. Roll first (free -- the surface stays
# square to the sensor), which is why 0 always wins when the ladder is present.
LADDER = [0.0, 180.0, 90.0, 270.0]
DEFAULT_SPIN_LIMIT = 90.0


def load() -> dict:
    if not CONFIG.is_file():
        raise SystemExit(f"no config at {CONFIG}")
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def save(data: dict) -> None:
    # utf-8 explicitly: this repo's config has been mojibaked by round-tripping
    # through PowerShell's default encoding before.
    CONFIG.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def show(data: dict) -> None:
    extrusion = data.get("extrusion") or {}
    armed = extrusion.get(KEY)
    limit = extrusion.get("max_tool_axis_spin_deg", DEFAULT_SPIN_LIMIT)
    if armed is None:
        print(f"roll: NOT armed - using the default ladder {LADDER}\n"
              f"      (the ladder means roll 0 always wins; no capture is rolled)")
    elif len(armed) == 1:
        print(f"roll: ARMED at {armed[0]:g} deg - every inspection pose is forced "
              f"to it,\n      and a refusal fails the run loudly.")
    else:
        print(f"roll: {armed} - MORE THAN ONE candidate, so a refused first choice "
              f"falls\n      back silently. Arm a single value instead.")
    print(f"wrist limit (global default): {limit:g} deg")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roll", nargs="?", type=float, help="degrees to force")
    ap.add_argument("--disarm", action="store_true",
                    help="restore the normal fallback ladder")
    args = ap.parse_args()

    data = load()
    if args.roll is None and not args.disarm:
        show(data)
        return 0

    extrusion = data.setdefault("extrusion", {})
    if args.disarm:
        extrusion.pop(KEY, None)
        save(data)
        print("roll DISARMED - back to the default ladder.")
    else:
        extrusion[KEY] = [float(args.roll)]
        save(data)
        print(f"roll ARMED at {args.roll:g} deg (single candidate: a refusal now "
              f"fails loudly).")
        limit = float(extrusion.get("max_tool_axis_spin_deg", DEFAULT_SPIN_LIMIT))
        if abs(args.roll) >= limit - 5.0:
            print(
                f"\n  WARNING: {args.roll:g} deg is at or near the wrist limit "
                f"{limit:g} deg.\n"
                f"  A roll about a nadir axis costs ~that much A6, and the gate is\n"
                f"  abs(delta) > limit -- so this is a coin flip at best.\n"
                f"  Give it headroom in the TRIAL's setup (maximum_tool_axis_spin_deg,\n"
                f"  e.g. {abs(args.roll) + 20:g}), NOT in the global default: keep the\n"
                f"  looser limit out of every print path.")

    print("\nRESTART the backend (.\\start.ps1) - config is read at startup.")
    print("Then VERIFY the achieved roll from the first take's provenance:\n"
          "  T_work_camera's X axis angle in the work frame must have MOVED.\n"
          "  Never trust that a roll happened because you asked for it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
