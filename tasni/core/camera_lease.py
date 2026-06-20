"""Single-owner lease over the (unicast) camera.

The Jetson server serves **one** TCP client at a time, so two readers racing the
socket — the live aiming preview and a calibration capture grab — corrupt each
other's frames. Until now that was avoided *by convention* ("stop live before you
grab"). This makes the rule explicit and enforced: whoever wants the camera takes a
labelled lease first; a second taker is refused with a clear "camera held by …"
error instead of silently stealing the stream.

Threading contract (important):
* **Non-blocking acquire for jobs.** The JobRunner has a single worker thread; if a
  job blocked waiting on a lease that only the request thread releases, that worker
  would deadlock and no future job could run. So the job-side path acquires
  *non-blocking* and fails fast (:class:`CameraBusy`) rather than waiting.
* The live preview holds the lease for its whole run and **releases after its thread
  has joined** (so the socket is genuinely free before the next owner grabs).

Pure ``threading`` — no camera, no RoboDK — so it is unit-testable anywhere.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager


class CameraBusy(RuntimeError):
    """Raised when the camera is already leased by someone else."""

    def __init__(self, owner: str | None):
        self.owner = owner
        super().__init__(
            f"camera is in use by {owner!r}" if owner
            else "camera is in use by another task")


class CameraLease:
    """A labelled single-owner mutex. ``owner`` is a human string for errors/health."""

    def __init__(self):
        self._lock = threading.Lock()
        self._guard = threading.Lock()      # protects the owner label
        self._owner: str | None = None

    @property
    def owner(self) -> str | None:
        with self._guard:
            return self._owner

    @property
    def held(self) -> bool:
        return self._lock.locked()

    def acquire(self, owner: str, *, blocking: bool = False,
                timeout: float = -1.0) -> bool:
        """Take the lease for ``owner``. Default **non-blocking** (returns ``False``
        immediately if held) — the safe choice on the single job worker thread."""
        got = (self._lock.acquire(True, timeout) if blocking
               else self._lock.acquire(False))
        if got:
            with self._guard:
                self._owner = owner
        return got

    def release(self, owner: str | None = None) -> bool:
        """Release the lease. A no-op (returns ``False``) if it is not held, or is
        held by a *different* owner — so a double release / release-without-acquire
        can never explode or steal someone else's lease."""
        with self._guard:
            if self._owner is None:
                return False
            if owner is not None and self._owner != owner:
                return False
            self._owner = None
        self._lock.release()
        return True

    @contextmanager
    def hold(self, owner: str, *, blocking: bool = False, timeout: float = -1.0):
        """Scoped lease: acquire (default non-blocking) or raise :class:`CameraBusy`,
        and always release on the way out."""
        if not self.acquire(owner, blocking=blocking, timeout=timeout):
            raise CameraBusy(self.owner)
        try:
            yield
        finally:
            self.release(owner)
