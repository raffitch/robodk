#!/usr/bin/env python
"""Keep the assistant's persistent memory inside the repo instead of on one laptop.

Claude Code stores per-project memory OUTSIDE the checkout, under
``~/.claude/projects/<mangled-path>/memory/``. That directory is the accumulated
train of thought for this cell -- 44-odd notes recording what was measured, what
was tried and refused, and which fixes are load-bearing. It lives on exactly one
machine and is in no backup, so a reinstall or a new laptop loses all of it.

This script mirrors that directory to ``.claude/memory/`` in the repo (and back),
so the notes travel with the code.

    py -3.10 tools/sync_claude_memory.py            # live -> repo (before committing)
    py -3.10 tools/sync_claude_memory.py --to-live  # repo -> live (on a fresh machine)
    py -3.10 tools/sync_claude_memory.py --check    # report drift; exit 1 if any

Copies are byte-exact (shutil.copy2). Nothing is deleted unless --prune is given,
so an out-of-date mirror can never silently destroy the live notes.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_MEMORY = REPO_ROOT / ".claude" / "memory"


def live_memory_dir() -> Path:
    """Where Claude Code keeps this project's memory.

    The project folder name is the absolute repo path with every character that
    is not a letter or digit replaced by a dash -- so
    ``D:\DesktopStuff\RAFFI NO TOUCH\backuprobodk\RoboDkClaude`` becomes
    ``D--DesktopStuff-RAFFI-NO-TOUCH-backuprobodk-RoboDkClaude``. Override with
    CLAUDE_MEMORY_DIR if your install differs.
    """
    override = os.environ.get("CLAUDE_MEMORY_DIR")
    if override:
        return Path(override)
    mangled = re.sub(r"[^A-Za-z0-9]", "-", str(REPO_ROOT))
    return Path.home() / ".claude" / "projects" / mangled / "memory"


def sync(src: Path, dst: Path, *, prune: bool) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(src.glob("*.md")):
        target = dst / path.name
        if target.exists() and filecmp.cmp(path, target, shallow=False):
            continue
        shutil.copy2(path, target)
        print(f"  + {path.name}")
        copied += 1
    if prune:
        for path in sorted(dst.glob("*.md")):
            if not (src / path.name).exists():
                path.unlink()
                print(f"  - {path.name} (pruned)")
    return copied


def check(live: Path, repo: Path) -> int:
    live_names = {p.name for p in live.glob("*.md")} if live.exists() else set()
    repo_names = {p.name for p in repo.glob("*.md")} if repo.exists() else set()
    drift = False
    for name in sorted(live_names - repo_names):
        print(f"  only in live: {name}")
        drift = True
    for name in sorted(repo_names - live_names):
        print(f"  only in repo: {name}")
        drift = True
    for name in sorted(live_names & repo_names):
        if not filecmp.cmp(live / name, repo / name, shallow=False):
            print(f"  differs:      {name}")
            drift = True
    return 1 if drift else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    direction = parser.add_mutually_exclusive_group()
    direction.add_argument("--to-live", action="store_true", help="copy repo -> live (restore on a new machine)")
    direction.add_argument("--check", action="store_true", help="report drift only; exit 1 if the two differ")
    parser.add_argument("--prune", action="store_true", help="also delete files missing from the source")
    args = parser.parse_args()

    live = live_memory_dir()
    print(f"live: {live}")
    print(f"repo: {REPO_MEMORY}")

    if args.check:
        rc = check(live, REPO_MEMORY)
        print("in sync" if rc == 0 else "DRIFT (run without --check to mirror)")
        return rc

    if args.to_live:
        if not REPO_MEMORY.exists():
            print("nothing to restore: .claude/memory/ is missing", file=sys.stderr)
            return 1
        n = sync(REPO_MEMORY, live, prune=args.prune)
    else:
        if not live.exists():
            print(f"no live memory at {live}", file=sys.stderr)
            print("(is this the right machine? set CLAUDE_MEMORY_DIR to override)", file=sys.stderr)
            return 1
        n = sync(live, REPO_MEMORY, prune=args.prune)

    print(f"{n} file(s) updated" if n else "already in sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
