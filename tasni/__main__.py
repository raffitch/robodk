"""Launch the tasni web app:  py -3.10 -m tasni  (then open the printed URL)."""
from __future__ import annotations

import argparse
import datetime as _dt
import faulthandler
import os
import tempfile
from pathlib import Path

import uvicorn

from .core.config import load_config

#: Kept alive for the process lifetime on purpose: ``faulthandler`` writes to the
#: file DESCRIPTOR, so letting this object be collected would close the fd out
#: from under the handler and the one dump that matters would go nowhere.
_CRASH_LOG = None


def _arm_crash_dump() -> Path:
    """Make a NATIVE crash name the Python code that was running.

    The backend hard-crashed seven times across 2026-08-30/31 (five with the same
    ntdll heap access violation, once as BEX64) and left nothing to work from:
    ``start.ps1`` captures stdout/stderr, but an access violation unwinds no
    Python frames, so the log simply stops mid-request and the operator sees only
    the UI's "Backend not responding". Every one of those crashes cost a cell
    session's evidence.

    ``faulthandler`` installs a handler for exactly these faults (on Windows it
    hooks the access violation too) and dumps the Python stack of EVERY thread at
    the moment of the fault -- which is the one fact the WER report cannot give:
    the process is full of native extensions (Open3D, OpenCV, onnxruntime,
    PySide2/Qt via robolink, numpy/scipy) and the faulting module alone does not
    say which of them was on the stack, nor whether the job thread, the camera
    thread or a request handler was running.

    Appended, never truncated: crashes come in runs, and the second one is only
    interpretable next to the first.
    """
    path = Path(os.environ.get("TASNI_CRASH_LOG")
                or Path(tempfile.gettempdir()) / "tasni-backend.crash.log")
    global _CRASH_LOG
    _CRASH_LOG = path.open("a", buffering=1, encoding="utf-8", errors="replace")
    _CRASH_LOG.write(f"\n===== tasni backend started {_dt.datetime.now().isoformat()} "
                     f"pid {os.getpid()} =====\n")
    faulthandler.enable(file=_CRASH_LOG, all_threads=True)
    return path


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="tasni control panel (web)")
    ap.add_argument("--host", default=cfg.web.host)
    ap.add_argument("--port", type=int, default=cfg.web.port)
    ap.add_argument("--reload", action="store_true", help="dev autoreload")
    args = ap.parse_args(argv)

    crash_log = _arm_crash_dump()
    print(f"tasni -> http://{args.host}:{args.port}")
    print(f"tasni: native-crash tracebacks -> {crash_log}")
    if args.reload:
        uvicorn.run("tasni.webapp.server:create_app", host=args.host,
                    port=args.port, reload=True, factory=True)
    else:
        from .webapp.server import create_app
        uvicorn.run(create_app(cfg), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
