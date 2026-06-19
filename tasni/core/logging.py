"""Logging + per-run artifact folders.

Each module run gets a timestamped directory under ``runs/`` to drop artifacts
(metrics JSON, annotated frames, the solved transform). The timestamp is passed
in by the caller so this module stays free of wall-clock calls.
"""
from __future__ import annotations

import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = REPO_ROOT  # backward-compatible alias


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def new_run_dir(module_id: str, stamp: str, root: Path | None = None) -> Path:
    """Create and return ``runs/<module_id>/<stamp>/`` for a run's artifacts."""
    base = (root or _REPO_ROOT) / "runs" / module_id / stamp
    base.mkdir(parents=True, exist_ok=True)
    return base
