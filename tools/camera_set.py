"""Read or change the Jetson's depth-filter chain at runtime, from the workstation.

    py -3.10 tools/camera_set.py                      # READ-ONLY: the achieved chain
    py -3.10 tools/camera_set.py spatial=0            # one A/B arm, no deploy
    py -3.10 tools/camera_set.py --restore            # back to the stock chain
    py -3.10 tools/camera_set.py --restore --dry-run  # print the line, send nothing

The server has spoken ``SET`` since 2026-09-01 (``server_unicast_syncronous.py``,
``_handle_set``/``apply_filter_settings``); this is the host half, so an A/B arm
costs one command instead of a hand-rolled socket script.

Three things the server guarantees, and this tool leans on:

* A **bare** ``SET`` is read-only. Use it to confirm the arm BEFORE and AFTER a
  run -- what you sent is not evidence, only the read-back is.
* A successful **write** retires the camera generation: every session greeted
  before it is closed and reconnects. So never send one while a capture is in
  flight, or you kill the take.
* Overrides **die on restart**. The unit file stays the boot truth, which is why
  a sweep must send an explicit restore between arms rather than trusting the
  previous arm's leftover state (runtime-parameters spec 4.1) -- ``--restore``.

Provenance: every take archives the achieved chain at
``provenance.camera_geometry.filter_options``. Read a take's arm from THAT, never
from what this tool sent -- a Jetson service restart mid-sweep silently reverts
to the unit file's defaults, and the per-take read-back is how you catch it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core.camera import CameraClient, CameraError  # noqa: E402
from tasni.core.config import load_config  # noqa: E402

# Re-exported from the client so the CLI, its tests and the read-only web
# endpoint all speak of ONE limit. The server refuses a SET line of this many
# bytes or more AND ends the session, because a truncated line cannot be
# resynced (see the server's _recv_line).
SET_LINE_MAXLEN = CameraClient.SET_LINE_MAXLEN

# The stock chain, as read back off the device 2026-09-01. `hole_filling` reads
# back null -- no such filter in the chain -- so it is OMITTED rather than sent.
STOCK = (
    "spatial=1 spatial_smooth_delta=20 spatial_magnitude=2 spatial_smooth_alpha=0.5 "
    "spatial_holes_fill=0 temporal_smooth_alpha=0.4 temporal_smooth_delta=20 "
    "temporal_persistency=3 depth_min_m=0.15 depth_max_m=1.5 decimation=0"
)


def send(assignments: list[str], *, timeout: float = 30.0) -> dict:
    """Send one SET line and return the server's parsed reply. No args = read-only.

    The protocol itself lives on ``CameraClient.filter_chain`` so the CLI and the
    read-only web endpoint cannot drift apart; this only turns the client's
    errors into clean CLI exits.
    """
    try:
        return CameraClient(load_config().camera).filter_chain(
            assignments, timeout=timeout)
    except CameraError as exc:
        raise SystemExit(str(exc)) from exc


def describe(reply: dict) -> str:
    options = reply.get("filter_options") or {}
    delta = options.get("spatial_smooth_delta")
    arm = ("spatial OFF" if delta is None else
           "STOCK" if delta == 20.0 else f"spatial_smooth_delta={delta}")
    return (f"chain : {reply.get('filters')}\n"
            f"arm   : {arm}\n"
            f"{json.dumps(options, indent=2)}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Read or set the Jetson depth-filter chain at runtime.")
    ap.add_argument("assignments", nargs="*", metavar="key=value",
                    help="filter settings to write; omit for a read-only query")
    ap.add_argument("--restore", action="store_true",
                    help="write the full stock chain (spec 4.1 explicit restore)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the line that would be sent, and send nothing")
    args = ap.parse_args()

    if args.restore and args.assignments:
        raise SystemExit("--restore writes the whole chain; do not also pass assignments")
    assignments = STOCK.split() if args.restore else args.assignments

    if args.dry_run:
        line = ("SET " + " ".join(assignments)).strip() if assignments else "SET"
        print(f"{line}\n({len(line) + 1} bytes, limit {SET_LINE_MAXLEN})")
        return 0

    if assignments:
        print("WRITING -- this retires the camera generation; never do it mid-capture.")
    reply = send(assignments)
    if not reply.get("ok"):
        print(f"SET REFUSED: {reply.get('error')}", file=sys.stderr)
        return 1
    print(describe(reply))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
