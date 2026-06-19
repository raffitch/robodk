"""Run a module's long job off the request thread, with progress + cancel.

A *job* is any callable ``job(ctx: JobContext) -> result``. The runner executes
it in a single background thread (one job at a time — the robot is a shared,
exclusive resource), and the job reports through ``ctx``:

    ctx.progress(step, total, message)   -> a "progress" event
    ctx.log(message)                     -> a "log" event
    ctx.frame(jpeg_bytes)                -> a "frame" event (live preview)
    ctx.check_cancel()                   -> raises JobCancelled if asked to stop

Results and errors are published as "result"/"error" events and also stored on
the runner so a late HTTP poll can still read the outcome.
"""
from __future__ import annotations

import base64
import threading
import traceback
from typing import Any, Callable

from .events import EventBus, JobEvent


class JobCancelled(Exception):
    pass


class JobBusy(RuntimeError):
    pass


class JobContext:
    """Handed to a job so it can report progress and check for cancellation."""

    def __init__(self, bus: EventBus, cancel_event: threading.Event):
        self._bus = bus
        self._cancel = cancel_event

    def progress(self, step: int, total: int, message: str = "") -> None:
        self._bus.publish(JobEvent("progress",
                                   {"step": step, "total": total, "message": message}))

    def log(self, message: str) -> None:
        self._bus.publish(JobEvent("log", {"message": message}))

    def frame(self, jpeg_bytes: bytes) -> None:
        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        self._bus.publish(JobEvent("frame", {"jpeg_b64": b64}))

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled()


class JobRunner:
    """Owns the single worker thread and the current job's status/result."""

    def __init__(self, bus: EventBus):
        self.bus = bus
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self.status: str = "idle"     # idle | running | done | error | cancelled
        self.result: Any = None
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, job: Callable[[JobContext], Any], *, name: str = "job") -> None:
        with self._lock:
            if self.running:
                raise JobBusy("a job is already running")
            self._cancel.clear()
            self.status = "running"
            self.result = None
            self.error = None
            ctx = JobContext(self.bus, self._cancel)
            self._thread = threading.Thread(
                target=self._run, args=(job, ctx, name), name=name, daemon=True)
            self._thread.start()
        self.bus.publish(JobEvent("status", {"status": "running", "name": name}))

    def cancel(self) -> None:
        self._cancel.set()

    def _run(self, job: Callable[[JobContext], Any], ctx: JobContext, name: str) -> None:
        try:
            self.result = job(ctx)
            self.status = "done"
            self.bus.publish(JobEvent("result", {"name": name, "result": self.result}))
        except JobCancelled:
            self.status = "cancelled"
            self.bus.publish(JobEvent("status", {"status": "cancelled", "name": name}))
        except Exception as e:  # noqa: BLE001 - surface any job failure to the UI
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"
            self.bus.publish(JobEvent("error", {
                "name": name,
                "message": self.error,
                "traceback": traceback.format_exc(),
            }))
