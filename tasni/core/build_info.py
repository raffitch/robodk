"""What code is actually RUNNING, as opposed to what is checked out.

The app is a long-lived process and Python caches imported modules, so editing
anything under ``tasni/`` has no effect until it restarts. Two live-cell test
cycles were spent on stale code before this module existed, and the run report
made it worse: it recorded ``git rev-parse HEAD`` *at report time*, so a report
could claim a commit the process had never loaded.

The fingerprint here is captured at import — i.e. at process start — and is
therefore the running build. Comparing it against the working tree on demand is
what turns "why didn't my fix do anything?" into a visible warning.
"""
from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent      # .../tasni
REPO_ROOT = PACKAGE_ROOT.parent


# Scanning ~60 files costs ~65 ms on this workstation and /api/health is polled,
# so results are reused briefly. The TTLs are short enough that an edit still
# shows up well before anyone can restart and launch a run against it.
_SCAN_TTL_S = 2.0
_GIT_TTL_S = 30.0
_scan_cache: tuple[float, str, dict[str, float]] | None = None
_git_cache: tuple[float, str] | None = None


def _fingerprint(*, use_cache: bool = True) -> tuple[str, dict[str, float]]:
    """A digest of every packaged source file, plus their modification times."""
    global _scan_cache

    now = time.monotonic()
    if use_cache and _scan_cache is not None and now - _scan_cache[0] < _SCAN_TTL_S:
        return _scan_cache[1], _scan_cache[2]
    digest = hashlib.sha256()
    mtimes: dict[str, float] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        try:
            modified = path.stat().st_mtime
        except OSError:      # deleted mid-scan; it will show up as a change
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        mtimes[relative] = modified
        digest.update(relative.encode())
        digest.update(f"{modified:.6f}".encode())
    sha = digest.hexdigest()[:12]
    _scan_cache = (now, sha, mtimes)
    return sha, mtimes


#: Captured at import time. This is the build the process is actually executing.
LOADED_SHA, _LOADED_MTIMES = _fingerprint(use_cache=False)
LOADED_AT = time.time()


def git_commit() -> str:
    """The commit currently CHECKED OUT — not necessarily the one running."""
    global _git_cache

    now = time.monotonic()
    if _git_cache is not None and now - _git_cache[0] < _GIT_TTL_S:
        return _git_cache[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unknown"
    _git_cache = (now, commit)
    return commit


def changed_since_start() -> list[str]:
    """Packaged source files added, removed or edited since the process started."""
    _, current = _fingerprint()
    changed = {path for path, modified in current.items()
               if _LOADED_MTIMES.get(path) != modified}
    changed |= set(_LOADED_MTIMES) - set(current)
    return sorted(changed)


def build_info() -> dict:
    """Running build, checked-out commit, and whether they have diverged.

    ``stale`` True means the working tree moved after this process started, so
    the code being exercised is NOT the code on disk. Restart the app before
    trusting any result produced in that state.
    """
    changed = changed_since_start()
    info = {
        "loaded_sha": LOADED_SHA,
        "loaded_at": LOADED_AT,
        # Deliberately named: reading git tells you what is checked out, which
        # is exactly the misleading value this module exists to replace.
        "git_commit_checked_out": git_commit(),
        "stale": bool(changed),
        "changed_since_start": changed[:20],
        "changed_count": len(changed),
    }
    if changed:
        info["warning"] = (
            f"{len(changed)} packaged source file(s) changed since this process "
            "started, so it is running STALE code. Restart the app "
            "(.\\start.ps1) before trusting this result: "
            + ", ".join(changed[:5]) + ("..." if len(changed) > 5 else ""))
    return info


def staleness_warning() -> str | None:
    """The human-readable warning, or ``None`` when the running build is current."""
    return build_info().get("warning")
