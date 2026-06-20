"""Live camera preview — a shared core service that streams analysed frames.

A workflow that needs a live view (calibration's aiming gate today; scan/aruco
later) hands :meth:`LivePreview.start` an *analyzer*: ``analyze(frame) -> (jpeg
bytes, metrics dict)``. The service owns the background thread and the camera
cadence; it grabs at ``fps``, runs the analyzer, and publishes two events per
frame on the shared :class:`~tasni.core.events.EventBus`:

    "frame"  {"jpeg_b64": ...}     the (annotated) image
    "gate"   {... analyzer metrics, "live": True}   the HUD readiness state

This keeps the *module* free of threads and sockets (the module just supplies the
analyzer) and keeps live streaming off the single-job :class:`JobRunner`, so the
operator can preview while jogging and the camera is released the moment a robot
job starts. Camera grabs are unicast/one-at-a-time, so a robot job must stop the
preview first (the module does this).
"""
from __future__ import annotations

import base64
import threading
from typing import Callable

from .camera import CameraClient, CameraError
from .camera_lease import CameraLease
from .events import EventBus, JobEvent

Analyzer = Callable[[object], "tuple[bytes, dict]"]

LEASE_OWNER = "live-preview"


class LivePreview:
    """Owns the preview thread for one camera + event bus."""

    def __init__(self, camera: CameraClient, bus: EventBus,
                 lease: CameraLease | None = None):
        self.camera = camera
        self.bus = bus
        self.lease = lease
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.last: dict | None = None      # most recent metrics (for HTTP polls)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, analyze: Analyzer, *, fps: float = 6.0,
              timeout_s: float = 2.0, color_only: bool = False) -> None:
        """Start streaming. Takes the camera lease first (raising
        :class:`~tasni.core.camera_lease.CameraBusy` if a job holds the camera), and
        holds it for the whole run — released in :meth:`stop` after the thread joins."""
        if self.running:
            return
        if self.lease is not None and not self.lease.acquire(LEASE_OWNER):
            from .camera_lease import CameraBusy
            raise CameraBusy(self.lease.owner)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(analyze, fps, timeout_s, color_only),
            name="live-preview", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the loop and wait for the in-flight grab to finish, then release
        the camera lease — so the socket is genuinely free before the caller (e.g.
        a robot job) grabs."""
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=5.0)
        self._thread = None
        if self.lease is not None:
            self.lease.release(LEASE_OWNER)

    def _loop(self, analyze: Analyzer, fps: float, timeout_s: float,
              color_only: bool) -> None:
        # fps caps the publish rate; reads are paced by frame arrival (the link),
        # so we stay near-realtime rather than draining a backlog.
        min_period = 1.0 / fps if fps > 0 else 0.0
        while not self._stop.is_set():
            try:
                with self.camera.stream(timeout=timeout_s, color_only=color_only) as stream:
                    while not self._stop.is_set():
                        # drain to the newest buffered frame so the preview stays
                        # at the live edge even if detection can't keep up
                        frame = stream.read(drain=True)
                        jpeg, metrics = analyze(frame)
                        self.last = metrics
                        self.bus.publish(JobEvent("frame",
                            {"jpeg_b64": base64.b64encode(jpeg).decode("ascii")}))
                        self.bus.publish(JobEvent("gate", {**metrics, "live": True}))
                        if min_period:
                            self._stop.wait(min_period)
            except CameraError as e:
                # Uniform gate shape so consumers never see a partial reading
                # (the HUD reads gates.* directly). Back off, then reconnect.
                self.last = {"detected": False, "ok": False, "error": str(e),
                             "gates": {"detected": False, "distance": False,
                                       "angle": False}}
                self.bus.publish(JobEvent("gate", {**self.last, "live": True}))
                self._stop.wait(1.0)
            except Exception as e:  # noqa: BLE001 - never let the loop die silently
                self.bus.publish(JobEvent("log",
                    {"message": f"live preview error: {type(e).__name__}: {e}"}))
                self._stop.wait(1.0)
