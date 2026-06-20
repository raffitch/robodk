"""Headless calibration runner — same job as the web module, no browser.

Useful on the bench / for smoke tests. Streams progress to stdout via a console
event sink instead of the WebSocket.

    py -3.10 -m tasni.cli                 # run, print metrics, do NOT apply
    py -3.10 -m tasni.cli --apply TOOL    # ...and write the result into TOOL
"""
from __future__ import annotations

import argparse

from .core.events import JobEvent
from .core.jobrunner import JobContext
import threading

from .modules.base import ServiceContainer
from .modules.calibration.service import CalibrationJob, CalibrationParams


class _ConsoleBus:
    """Minimal EventBus stand-in that prints events synchronously."""

    def bind_loop(self, loop):  # noqa: D401 - interface shim
        pass

    def publish(self, event: JobEvent) -> None:
        if event.type == "progress":
            p = event.payload
            print(f"[{p['step']}/{p['total']}] {p['message']}")
        elif event.type == "log":
            print("   ", event.payload["message"])
        elif event.type == "error":
            print("ERROR:", event.payload["message"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="tasni headless calibration")
    ap.add_argument("--apply", action="store_true",
                    help="apply the solved pose to the camera tool after solving")
    ap.add_argument("--no-refine", action="store_true", help="skip refinement")
    ap.add_argument("--holdout", type=int, default=None, help="validation poses")
    args = ap.parse_args(argv)

    services = ServiceContainer.build()
    services.bus = _ConsoleBus()  # type: ignore[assignment]

    # The camera tool is forced (RealSense-only); --apply writes the solve into it.
    params = CalibrationParams(
        holdout_count=args.holdout,
        refine=False if args.no_refine else None,
    )
    job = CalibrationJob(services, params)
    ctx = JobContext(services.bus, threading.Event())
    result = job(ctx)

    print("\n" + result["summary"])
    print("artifacts:", result["run_dir"])
    if args.apply:
        tool = job.apply_to_tool()
        print(f"applied to tool: {tool}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
