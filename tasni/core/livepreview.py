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
import time
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
              timeout_s: float = 2.0, color_only: bool = False,
              quality: int | None = None, codec: str = "jpeg",
              bitrate: int | None = None, with_depth: bool = False,
              depth_probe: "Callable[[object], dict] | None" = None,
              depth_period_s: float = 1.5) -> None:
        """Start streaming. Takes the camera lease first (raising
        :class:`~tasni.core.camera_lease.CameraBusy` if a job holds the camera), and
        holds it for the whole run — released in :meth:`stop` after the thread joins.
        ``quality`` (if set) asks the server to encode the preview JPEG smaller;
        ``codec="h264"`` switches to the Nano's hardware NVENC stream (``bitrate``
        in kbps), decoded client-side. ``with_depth`` decodes the depth payload into
        each ``frame.depth`` so a depth-based analyzer (the scan standoff gate) can
        read it — keep ``color_only=False`` then, since color-only/h264 carry no
        depth.

        ``depth_probe`` (the scan standoff gate): when set, the video streams
        **color-only** (fast, like calibration) and ``analyze`` only renders frames,
        while a depth frame is grabbed on a ~``depth_period_s`` interleave and passed
        to ``depth_probe(frame) -> gate dict`` to update the gate. The camera is
        unicast, so a fast color stream and depth can't run at once — we alternate:
        stream color for ``depth_period_s``, briefly grab one depth frame, repeat.
        This keeps the preview at color framerate instead of the slow depth path."""
        if self.running:
            return
        if self.lease is not None and not self.lease.acquire(LEASE_OWNER):
            from .camera_lease import CameraBusy
            raise CameraBusy(self.lease.owner)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            args=(analyze, fps, timeout_s, color_only, quality, codec, bitrate,
                  with_depth, depth_probe, depth_period_s),
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
              color_only: bool, quality: int | None = None,
              codec: str = "jpeg", bitrate: int | None = None,
              with_depth: bool = False,
              depth_probe: "Callable[[object], dict] | None" = None,
              depth_period_s: float = 1.5) -> None:
        # fps caps the publish rate; reads are paced by frame arrival (the link),
        # so we stay near-realtime rather than draining a backlog.
        min_period = 1.0 / fps if fps > 0 else 0.0
        interleave = depth_probe is not None
        # Interleave: the video is color-only (fast); depth comes from periodic grabs.
        vid_color_only = True if interleave else color_only
        vid_with_depth = False if interleave else with_depth
        last_depth = 0.0

        def _publish_frame(jpeg: bytes) -> None:
            self.bus.publish(JobEvent("frame",
                {"jpeg_b64": base64.b64encode(jpeg).decode("ascii")}))

        def _publish_gate(metrics: dict) -> None:
            self.last = metrics
            self.bus.publish(JobEvent("gate", {**metrics, "live": True}))

        def _gate_error(e) -> None:
            _publish_gate({"detected": False, "ok": False, "error": str(e),
                           "gates": {"detected": False, "distance": False, "angle": False}})

        while not self._stop.is_set():
            try:
                with self.camera.stream(timeout=timeout_s, color_only=vid_color_only,
                                        quality=quality, codec=codec,
                                        bitrate=bitrate) as stream:
                    while not self._stop.is_set():
                        # drain to the newest buffered frame so the preview stays
                        # at the live edge even if detection can't keep up
                        frame = stream.read(drain=True, with_depth=vid_with_depth)
                        jpeg, metrics = analyze(frame)
                        _publish_frame(jpeg)
                        if not interleave:
                            _publish_gate(metrics)
                        elif time.monotonic() - last_depth >= depth_period_s:
                            break    # leave the color stream to refresh the depth gate
                        if min_period:
                            self._stop.wait(min_period)
                # Interleave depth sample: the color stream is now closed, so the
                # unicast camera is free for one full (depth) grab.
                if interleave and not self._stop.is_set():
                    try:
                        df = self.camera.grab(with_depth=True, timeout=timeout_s)
                        _publish_gate(depth_probe(df))
                    except CameraError as e:
                        _gate_error(e)
                    last_depth = time.monotonic()
            except CameraError as e:
                # Uniform gate shape so consumers never see a partial reading
                # (the HUD reads gates.* directly). Back off, then reconnect.
                _gate_error(e)
                self._stop.wait(1.0)
            except Exception as e:  # noqa: BLE001 - never let the loop die silently
                self.bus.publish(JobEvent("log",
                    {"message": f"live preview error: {type(e).__name__}: {e}"}))
                self._stop.wait(1.0)
