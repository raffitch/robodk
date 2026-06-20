"""Run-artifact registry — locate, load and stamp per-run output on disk.

Each module run drops its artifacts in ``runs/<module>/<stamp>/`` (the folder
:func:`tasni.core.logging.new_run_dir` creates). This module is the *reader/index*
over that tree, plus the "which run is currently applied" pointer:

* :func:`run_dir` / :func:`load_report` / :func:`load_meta` — resolve and read one
  run's files, so apply-by-run-id survives a server restart (the in-memory last
  job is gone, but ``report.json`` on disk still holds the solved transform).
* :func:`list_runs` — the newest-first index the Dashboard lists (factored out of
  the web shell so it is testable and reusable).
* :func:`write_active` / :func:`read_active` — a per-module ``active.json`` pointer
  recording which run is live in the cell right now (run-id, date, key metrics), so
  the Dashboard can show "cell calibrated: <date> · <quality>".

``module`` and ``stamp`` arrive from HTTP, so every path that joins them guards
against traversal (no separators / ``..`` / absolute). ``root=`` is a test seam,
mirroring :func:`tasni.core.logging.new_run_dir`. This stays in core and imports no
``modules.*`` so every workflow (scan/print next) reuses the same shape.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .logging import REPO_ROOT

ACTIVE_FILE = "active.json"
REPORT_FILE = "report.json"
META_FILE = "meta.json"


class RunNotFound(FileNotFoundError):
    """Raised when a requested run / artifact is not on disk."""


def runs_root(root: Path | None = None) -> Path:
    return (root or REPO_ROOT) / "runs"


def _safe_segment(name: str, kind: str) -> str:
    """Validate a single untrusted path segment (a module id or a run stamp).

    These come straight off the HTTP surface (``?run_id=...``), so reject anything
    that could climb out of the runs tree: empty, separators, ``..``, or absolute.
    """
    if not name or name in (".", ".."):
        raise ValueError(f"invalid {kind}: {name!r}")
    if "/" in name or "\\" in name or os.sep in name or (os.altsep and os.altsep in name):
        raise ValueError(f"invalid {kind} (path separator): {name!r}")
    if Path(name).name != name:
        raise ValueError(f"invalid {kind}: {name!r}")
    return name


def module_dir(module_id: str, root: Path | None = None) -> Path:
    return runs_root(root) / _safe_segment(module_id, "module")


def run_dir(module_id: str, stamp: str, root: Path | None = None) -> Path:
    """Path to ``runs/<module>/<stamp>/`` (guarded; not created — use
    :func:`tasni.core.logging.new_run_dir` to create a fresh run)."""
    return module_dir(module_id, root) / _safe_segment(stamp, "stamp")


def load_report(module_id: str, stamp: str, root: Path | None = None) -> dict:
    """Load a run's ``report.json`` (the solved transform + metrics). Raises
    :class:`RunNotFound` if the run or report is missing."""
    path = run_dir(module_id, stamp, root) / REPORT_FILE
    if not path.is_file():
        raise RunNotFound(f"no {REPORT_FILE} for {module_id}/{stamp}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_meta(module_id: str, stamp: str, root: Path | None = None) -> dict | None:
    """Load a run's ``meta.json`` (stamp, tool, ...) if present, else ``None``."""
    path = run_dir(module_id, stamp, root) / META_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_meta(module_id: str, stamp: str, meta: dict, root: Path | None = None) -> Path:
    """Write ``meta.json`` beside a run's artifacts (the run dir must already exist)."""
    path = run_dir(module_id, stamp, root) / META_FILE
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def list_runs(limit: int = 20, root: Path | None = None) -> list[dict]:
    """Recent run folders across all modules, newest first (by stamp). The
    per-module ``active.json`` pointer is a file, not a run dir, so it is skipped."""
    base = runs_root(root)
    items: list[dict] = []
    if base.exists():
        for mdir in base.iterdir():
            if not mdir.is_dir():
                continue
            for run in mdir.iterdir():
                if run.is_dir():
                    items.append({"module": mdir.name, "stamp": run.name,
                                  "path": str(run)})
    items.sort(key=lambda r: r["stamp"], reverse=True)
    return items[:limit]


def write_active(module_id: str, payload: dict, root: Path | None = None) -> Path:
    """Atomically record which run is currently applied for ``module_id``
    (``runs/<module>/active.json``). The caller supplies any timestamp in
    ``payload`` (core stays clock-free). Written tmp-then-replace so a reader never
    sees a half-written file."""
    mdir = module_dir(module_id, root)
    mdir.mkdir(parents=True, exist_ok=True)
    final = mdir / ACTIVE_FILE
    tmp = mdir / (ACTIVE_FILE + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, final)        # atomic on the same filesystem
    return final


def read_active(module_id: str, root: Path | None = None) -> dict | None:
    """The currently-applied run for ``module_id`` (``active.json``), or ``None``."""
    path = module_dir(module_id, root) / ACTIVE_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
