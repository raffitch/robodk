"""Camera-ownership lease (core/camera_lease.py) — pure threading, no camera.

The lease enforces the single-client rule of the unicast Jetson server. Covers:
  * single owner: a second acquire is refused while held
  * **non-blocking acquire** (the job-worker safety property) returns False, never hangs
  * the owner label is reported (for "camera held by …" errors + health)
  * hold() releases on exit, even on exception, and raises CameraBusy when busy
  * release is safe: wrong-owner / double release is a no-op, never raises
  * a blocking acquire waits and then succeeds once the holder releases

    py -3.10 tests/test_camera_lease.py
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core.camera_lease import CameraBusy, CameraLease  # noqa: E402


def test_single_owner_and_label():
    lease = CameraLease()
    assert lease.acquire("live-preview") is True
    assert lease.held is True and lease.owner == "live-preview"
    # a second taker is refused (non-blocking) and the label is unchanged
    assert lease.acquire("calibration-run") is False
    assert lease.owner == "live-preview"
    assert lease.release("live-preview") is True
    assert lease.held is False and lease.owner is None
    print("[single owner] second acquire refused; label reported; released")


def test_nonblocking_acquire_never_hangs():
    """The job worker thread acquires non-blocking — if held, it must return False
    immediately rather than block the only worker forever."""
    lease = CameraLease()
    lease.acquire("live-preview")
    done = threading.Event()
    result = {}

    def worker():
        result["got"] = lease.acquire("calibration-run")     # default: non-blocking
        done.set()

    t = threading.Thread(target=worker)
    t.start()
    assert done.wait(2.0), "non-blocking acquire hung — would deadlock the worker"
    t.join(2.0)
    assert result["got"] is False
    lease.release("live-preview")
    print("[non-blocking] held lease -> acquire returns False without hanging")


def test_hold_releases_and_raises_when_busy():
    lease = CameraLease()
    with lease.hold("target-creation"):
        assert lease.owner == "target-creation"
        try:
            with lease.hold("calibration-run"):
                raise AssertionError("should not have acquired a busy lease")
        except CameraBusy as e:
            assert e.owner == "target-creation"
    assert lease.held is False                       # released on exit
    # released even if the body raises
    try:
        with lease.hold("calibration-run"):
            raise ValueError("boom")
    except ValueError:
        pass
    assert lease.held is False
    print("[hold] scoped acquire/release; CameraBusy carries the holder; released on error")


def test_release_is_safe():
    lease = CameraLease()
    assert lease.release("nobody") is False          # not held -> no-op
    lease.acquire("live-preview")
    assert lease.release("someone-else") is False     # wrong owner -> no-op, still held
    assert lease.held is True and lease.owner == "live-preview"
    assert lease.release("live-preview") is True
    assert lease.release("live-preview") is False     # double release -> no-op
    print("[release] wrong-owner / double / unheld release are safe no-ops")


def test_blocking_acquire_waits_then_succeeds():
    lease = CameraLease()
    lease.acquire("live-preview")
    got = {}

    def waiter():
        got["ok"] = lease.acquire("calibration-run", blocking=True, timeout=2.0)

    t = threading.Thread(target=waiter)
    t.start()
    lease.release("live-preview")                    # hand it over
    t.join(2.0)
    assert got["ok"] is True and lease.owner == "calibration-run"
    lease.release("calibration-run")
    print("[blocking] waiter blocks, then acquires once the holder releases")


if __name__ == "__main__":
    test_single_owner_and_label()
    test_nonblocking_acquire_never_hangs()
    test_hold_releases_and_raises_when_busy()
    test_release_is_safe()
    test_blocking_acquire_waits_then_succeeds()
    print("\nCamera-lease tests passed.")
