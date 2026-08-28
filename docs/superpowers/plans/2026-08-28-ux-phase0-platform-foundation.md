# UX overhaul — Phase 0: platform foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the platform's shared state truthful before any page is re-laid-out: module-scoped job events with unique execution ids, per-(module, kind) job history, a station-only cell-level Connect, an explicit real-robot link with a server-side live-pose gate, a recorded-vs-present readiness endpoint, and one frontend `PlatformProvider` + topbar Connect that every page reads.

**Architecture:** Backend changes are confined to `tasni/core/{events,jobrunner,livepreview,rdk_io,config}.py`, `tasni/webapp/server.py`, and each module's `module.py` (call-site stamping, `/status` shape, one new gate). No job logic in `service.py` changes except the link-helper consolidation. Frontend: `api/events.tsx` + `api/useHealth.ts` become `platform/PlatformProvider.tsx`; `Layout.tsx` gains the Connect button; the three pages drop their banners and filter events by `module` + `job_id`/`stream_id`. Layout of the pages is otherwise unchanged (phase 1 does the stepper).

**Tech Stack:** Python 3.10, FastAPI, pytest (`py -3.10 -m pytest`), React 18 + Vite + TypeScript, new: vitest + @testing-library/react + jsdom.

**Spec:** `docs/superpowers/specs/2026-08-28-ux-workflow-shell-design.md` — §4.5 (connect, link, events/status), §4.7 (readiness), §7, §8, §9 phase 0, Appendix B phase 0.

## Global Constraints

- Python is `py -3.10` (no `python` on PATH). Never run the full pytest suite — always `py -3.10 -m pytest tests/<file>.py -q` or `-k`.
- Never round-trip source through PowerShell `Get-Content`/`Set-Content` (mojibakes UTF-8). Use the Edit tool or Git Bash.
- The Tasni backend caches imports: after editing `tasni/**.py`, restart `.\start.ps1` before any cell test.
- **No job/motion logic changes** in `tasni/modules/*/service.py` (spec non-goal) — the only service edit is delegating `ensure_real_robot_link` to the consolidated core helper.
- **No page layout changes** in phase 0 (spec §9): pages keep their cards; only connection/event plumbing and the new gate reasons change.
- Every event carries `module`; every job event carries `job_id` + `kind`; every live-preview event carries `stream_id`; `jobs.start()` returns the `job_id` and every start endpoint echoes `{"job_id": ...}` (spec §4.5).
- Platform Connect **never** calls `connect_robot` (spec decision 9). Linking happens only via `POST /api/rdk/link` and inside runs.
- Every cell state transition — connect, link, job start, live-preview start — goes through `services.arbiter` (`CellArbiter.hold`, non-blocking, fails fast). No page auto-connects (spec decision 13).
- "running" is published before a job's worker thread starts; pages reconcile from their module `/status` right after every start response and whenever the socket (re)connects.
- Copy rule: `lockReason` strings are one sentence, imperative, no spec vocabulary.
- Commit after every task; push `ux-overhaul` at the end of every task group (CLAUDE.md working agreement). Phase 0 ships from the branch `ux-phase0` off `ux-overhaul`.
- Frontend must pass `npm run typecheck && npm run build` (run from `tasni/webui`) before each commit that touches `webui/`.

## File structure

| File | Responsibility |
|---|---|
| `tasni/core/events.py` | `JobEvent` (+ `module`, `job_id`, `kind`, `stream_id`), `EventBus.scoped(module)` |
| `tasni/core/jobrunner.py` | `JobRecord`, per-(module, kind) history, `start() -> job_id`, `module_status()` |
| `tasni/core/livepreview.py` | `start(owner=) -> stream_id`, stamps live events |
| `tasni/core/rdk_io.py` | `ensure_robot_link(rdk, cfg, *, strict)` (consolidated) |
| `tasni/core/config.py` | `RoboDKConfig.require_live_pose`, rewritten `connect_robot_on_connect` doc |
| `tasni/webapp/server.py` | `POST /api/rdk/connect`, `POST /api/rdk/link`, `GET /api/readiness` |
| `tasni/webapp/readiness.py` | pure readiness composition (recorded vs present) |
| `tasni/modules/{calibration,scan,extrusion}/module.py` | scoped publishes, `kind=/module=` starts, `job_id` echo, new `/status`, live-pose gate |
| `tasni/modules/calibration/service.py` | `ensure_real_robot_link` delegates to core |
| `tests/test_jobrunner_scope.py`, `tests/test_livepreview.py`, `tests/test_platform_connect.py`, `tests/test_robot_link.py`, `tests/test_readiness.py`, `tests/test_module_status.py` | backend tests |
| `tasni/webui/src/platform/PlatformProvider.tsx` | health poll, rdk status, `connect()`, `link()`, `subscribe(module, h)` |
| `tasni/webui/src/components/Layout.tsx` | topbar pills + Connect button |
| `tasni/webui/src/pages/{Calibration,Scan,Extrusion}.tsx` | use provider; drop banners; filter by id |
| `tasni/webui/src/test/*.test.tsx`, `vitest.config.ts` | frontend integration tests |

---

### Task 1: `JobEvent` scoping fields + `JobRunner` history and `job_id`

**Files:**
- Modify: `tasni/core/events.py:15-24`
- Modify: `tasni/core/jobrunner.py` (whole file)
- Test: `tests/test_jobrunner_scope.py` (new)

**Interfaces:**
- Produces: `JobEvent(type, payload, module=None, job_id=None, kind=None, stream_id=None)`; `EventBus.scoped(module) -> ScopedBus` with `.publish(JobEvent)`; `JobRecord` dataclass with `to_dict()`; `JobRunner.start(job, *, kind="job", module="platform", name=None) -> str` (returns `job_id`; `name=` is a legacy alias for `kind`); `JobRunner.current: JobRecord | None`; `JobRunner.record(module, kind) -> JobRecord | None`; `JobRunner.module_status(module) -> dict` = `{"running": {module, kind, job_id} | None, "status": str, "jobs": {kind: record.to_dict()}}`. Existing attributes `status/result/error/running` are kept (used by `/api/health` and by pages until Task 8).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_jobrunner_scope.py
"""JobRunner: module/kind/job_id stamping + per-(module, kind) history (spec §4.5)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core.events import EventBus, JobEvent, ScopedBus  # noqa: E402
from tasni.core.jobrunner import JobRunner  # noqa: E402


class _FakeBus:
    def __init__(self):
        self.events: list[JobEvent] = []

    def publish(self, ev):
        self.events.append(ev)


def _wait(runner, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    while runner.running and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.02)   # let the terminal event land


def test_start_returns_job_id_and_stamps_every_event():
    bus = _FakeBus()
    r = JobRunner(bus)

    def job(ctx):
        ctx.progress(1, 2, "half")
        ctx.log("hello")
        ctx.frame(b"\xff\xd8")
        return {"ok": True}

    job_id = r.start(job, kind="sim_tour", module="calibration")
    assert isinstance(job_id, str) and len(job_id) == 32
    _wait(r)
    types = [e.type for e in bus.events]
    assert types == ["status", "progress", "log", "frame", "result"]
    for e in bus.events:
        assert e.module == "calibration"
        assert e.kind == "sim_tour"
        assert e.job_id == job_id
        assert e.stream_id is None
    assert bus.events[-1].payload["result"] == {"ok": True}
    assert bus.events[-1].payload["name"] == "sim_tour"   # legacy field kept


def test_running_event_precedes_a_fast_jobs_result_and_record_is_final():
    """A job that finishes before start() even returns must still leave a clean
    trail: status(running) first, result last, record already 'done' — the page
    reconciles from /status (Task 8b) precisely because of this window."""
    bus = _FakeBus()
    r = JobRunner(bus)
    job_id = r.start(lambda ctx: "fast", kind="sim_tour", module="calibration")
    _wait(r)
    types = [e.type for e in bus.events]
    assert types[0] == "status" and bus.events[0].payload["status"] == "running"
    assert types[-1] == "result"
    rec = r.record("calibration", "sim_tour")
    assert rec.job_id == job_id and rec.status == "done" and rec.result == "fast"


def test_two_runs_of_one_kind_get_distinct_ids():
    r = JobRunner(_FakeBus())
    a = r.start(lambda ctx: 1, kind="scan", module="scan")
    _wait(r)
    b = r.start(lambda ctx: 2, kind="scan", module="scan")
    _wait(r)
    assert a != b
    assert r.record("scan", "scan").job_id == b
    assert r.record("scan", "scan").result == 2


def test_history_is_per_module_and_kind():
    r = JobRunner(_FakeBus())
    solve = r.start(lambda ctx: {"can_apply": True}, kind="calibration", module="calibration")
    _wait(r)
    r.start(lambda ctx: {"all_ok": True}, kind="sim_tour", module="calibration")
    _wait(r)
    r.start(lambda ctx: {"mesh": 1}, kind="scan", module="scan")
    _wait(r)
    rec = r.record("calibration", "calibration")
    assert rec is not None and rec.job_id == solve and rec.result == {"can_apply": True}
    assert rec.status == "done" and rec.finished_at is not None
    assert r.record("calibration", "sim_tour").result == {"all_ok": True}
    assert r.record("scan", "scan").result == {"mesh": 1}
    assert r.record("extrusion", "extrusion-print") is None


def test_module_status_reports_foreign_running_job():
    r = JobRunner(_FakeBus())
    gate = {"go": False}

    def slow(ctx):
        while not gate["go"]:
            time.sleep(0.005)
        return "done"

    job_id = r.start(slow, kind="calibration", module="calibration")
    try:
        s_cal = r.module_status("calibration")
        s_scan = r.module_status("scan")
        assert s_cal["running"] == {"module": "calibration", "kind": "calibration",
                                    "job_id": job_id}
        assert s_cal["status"] == "running"
        assert s_scan["running"] == s_cal["running"]
        assert s_scan["status"] == "busy"          # someone else's job
        assert s_scan["jobs"] == {}
    finally:
        gate["go"] = True
        _wait(r)
    assert r.module_status("calibration")["running"] is None
    assert r.module_status("calibration")["jobs"]["calibration"]["status"] == "done"
    assert r.module_status("scan")["status"] == "idle"


def test_error_and_cancel_records():
    r = JobRunner(_FakeBus())

    def boom(ctx):
        raise ValueError("nope")

    r.start(boom, kind="scan", module="scan")
    _wait(r)
    rec = r.record("scan", "scan")
    assert rec.status == "error" and rec.error == "ValueError: nope"
    assert r.module_status("scan")["status"] == "error"


def test_legacy_name_keyword_still_works():
    r = JobRunner(_FakeBus())
    r.start(lambda ctx: 1, name="legacy")
    _wait(r)
    assert r.record("platform", "legacy").result == 1


def test_scoped_bus_fills_module():
    bus = _FakeBus()
    scoped = ScopedBus(bus, "scan")          # wraps any object with .publish()
    scoped.publish(JobEvent("log", {"message": "x"}))
    assert bus.events[0].module == "scan"
    scoped.publish(JobEvent("log", {}, module="calibration"))
    assert bus.events[1].module == "calibration"        # explicit wins
    assert isinstance(EventBus().scoped("scan"), ScopedBus)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3.10 -m pytest tests/test_jobrunner_scope.py -q`
Expected: FAIL — `TypeError: start() got an unexpected keyword argument 'kind'` / `AttributeError: 'EventBus' has no attribute 'scoped'`.

- [ ] **Step 3: Extend `JobEvent` and add `EventBus.scoped`**

Replace the `JobEvent` dataclass in `tasni/core/events.py` (lines 15-24) with:

```python
@dataclass
class JobEvent:
    """One thing that happened during a job or a live preview, fan-out to all
    subscribers. ``module`` names the owning workflow module; job events carry
    ``job_id`` (unique per execution) + ``kind`` (the job's kind, e.g. "sim_tour");
    live-preview events carry ``stream_id`` instead (spec §4.5)."""

    type: str                       # progress | log | frame | gate | status | result | error | ...
    payload: dict[str, Any] = field(default_factory=dict)
    module: str | None = None
    job_id: str | None = None
    kind: str | None = None
    stream_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ScopedBus:
    """``publish()`` with ``module`` pre-filled, for request-path publishes made
    directly by a module (not through a JobContext/LivePreview)."""

    def __init__(self, bus, module: str):
        self._bus = bus
        self._module = module

    def publish(self, event: JobEvent) -> None:
        if event.module is None:
            event.module = self._module
        self._bus.publish(event)
```

and add to `EventBus` (after `unsubscribe`):

```python
    def scoped(self, module: str) -> "ScopedBus":
        return ScopedBus(self, module)
```

- [ ] **Step 4: Rewrite `tasni/core/jobrunner.py`**

Replace the whole file with:

```python
"""Run a module's long job off the request thread, with progress + cancel.

A *job* is any callable ``job(ctx: JobContext) -> result``. The runner executes
it in a single background thread (one job at a time — the robot is a shared,
exclusive resource), and the job reports through ``ctx``:

    ctx.progress(step, total, message)   -> a "progress" event
    ctx.log(message)                     -> a "log" event
    ctx.frame(jpeg_bytes)                -> a "frame" event (live preview)
    ctx.check_cancel()                   -> raises JobCancelled if asked to stop

Every event is stamped with the owning ``module``, the job's ``kind`` and a
``job_id`` unique to this execution, so a page can ignore other modules' jobs and
delayed events from a previous run (spec §4.5). Outcomes are kept per
(module, kind) in ``last_jobs`` so a later dry run cannot erase a solve result.
"""
from __future__ import annotations

import base64
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable

from .events import EventBus, JobEvent


class JobCancelled(Exception):
    pass


class JobBusy(RuntimeError):
    pass


@dataclass
class JobRecord:
    """Outcome of one job execution (spec §4.5)."""

    job_id: str
    module: str
    kind: str
    status: str                     # running | done | error | cancelled
    result: Any = None
    error: str | None = None
    started_at: float = 0.0
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def ref(self) -> dict[str, str]:
        return {"module": self.module, "kind": self.kind, "job_id": self.job_id}


class JobContext:
    """Handed to a job so it can report progress and check for cancellation."""

    def __init__(self, bus: EventBus, cancel_event: threading.Event, *,
                 module: str = "platform", job_id: str = "", kind: str = "job"):
        self._bus = bus
        self._cancel = cancel_event
        self.module = module
        self.job_id = job_id
        self.kind = kind

    def _emit(self, type_: str, payload: dict) -> None:
        self._bus.publish(JobEvent(type_, payload, module=self.module,
                                   job_id=self.job_id, kind=self.kind))

    def progress(self, step: int, total: int, message: str = "") -> None:
        self._emit("progress", {"step": step, "total": total, "message": message})

    def log(self, message: str) -> None:
        self._emit("log", {"message": message})

    def frame(self, jpeg_bytes: bytes) -> None:
        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        self._emit("frame", {"jpeg_b64": b64})

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled()


class JobRunner:
    """Owns the single worker thread, the current job and the per-(module, kind)
    history of outcomes."""

    def __init__(self, bus: EventBus):
        self.bus = bus
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        # Legacy global view (used by /api/health and pre-phase-0 callers).
        self.status: str = "idle"     # idle | running | done | error | cancelled
        self.result: Any = None
        self.error: str | None = None
        self.current: JobRecord | None = None
        self.last_jobs: dict[str, dict[str, JobRecord]] = {}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, job: Callable[[JobContext], Any], *, kind: str = "job",
              module: str = "platform", name: str | None = None) -> str:
        """Start ``job`` and return its unique ``job_id``. ``name`` is the legacy
        spelling of ``kind``."""
        kind = name or kind
        with self._lock:
            if self.running:
                raise JobBusy("a job is already running")
            self._cancel.clear()
            job_id = uuid.uuid4().hex
            rec = JobRecord(job_id=job_id, module=module, kind=kind,
                            status="running", started_at=time.time())
            self.current = rec
            self.last_jobs.setdefault(module, {})[kind] = rec
            self.status, self.result, self.error = "running", None, None
            ctx = JobContext(self.bus, self._cancel, module=module,
                             job_id=job_id, kind=kind)
            self._thread = threading.Thread(
                target=self._run, args=(job, ctx, rec), name=f"{module}:{kind}",
                daemon=True)
            # "running" goes out BEFORE the worker starts, so no job event can ever
            # precede it — a fast job could otherwise publish its result first.
            self._publish(rec, "status", {"status": "running", "name": kind})
            self._thread.start()
        return job_id

    def cancel(self) -> None:
        self._cancel.set()

    def record(self, module: str, kind: str) -> JobRecord | None:
        return self.last_jobs.get(module, {}).get(kind)

    def module_status(self, module: str) -> dict:
        """The shape every module ``/status`` embeds (spec §4.5): who is running
        (anyone's job), this module's own status word, and its own records."""
        cur = self.current if self.running else None
        if cur is None:
            own = self.last_jobs.get(module, {})
            latest = max(own.values(), key=lambda r: r.started_at, default=None)
            status = latest.status if latest is not None else "idle"
        else:
            status = "running" if cur.module == module else "busy"
        return {
            "running": cur.ref() if cur is not None else None,
            "status": status,
            "jobs": {k: r.to_dict() for k, r in self.last_jobs.get(module, {}).items()},
        }

    def _publish(self, rec: JobRecord, type_: str, payload: dict) -> None:
        self.bus.publish(JobEvent(type_, payload, module=rec.module,
                                  job_id=rec.job_id, kind=rec.kind))

    def _run(self, job: Callable[[JobContext], Any], ctx: JobContext,
             rec: JobRecord) -> None:
        try:
            rec.result = job(ctx)
            rec.status = "done"
            self.result, self.status = rec.result, "done"
            rec.finished_at = time.time()
            self._publish(rec, "result", {"name": rec.kind, "result": rec.result})
        except JobCancelled:
            rec.status = self.status = "cancelled"
            rec.finished_at = time.time()
            self._publish(rec, "status", {"status": "cancelled", "name": rec.kind})
        except Exception as e:  # noqa: BLE001 - surface any job failure to the UI
            rec.status = self.status = "error"
            rec.error = self.error = f"{type(e).__name__}: {e}"
            rec.finished_at = time.time()
            self._publish(rec, "error", {
                "name": rec.kind,
                "message": rec.error,
                "traceback": traceback.format_exc(),
            })
```

- [ ] **Step 5: Run the tests**

Run: `py -3.10 -m pytest tests/test_jobrunner_scope.py tests/test_sim_tour.py tests/test_calibration_job.py -q`
Expected: all PASS (the two existing files use `JobContext` subclasses with no-arg `__init__` overrides and `jobs.start(..., name=)`, both still supported).

- [ ] **Step 6: Commit**

```bash
git checkout -b ux-phase0 ux-overhaul
git add tasni/core/events.py tasni/core/jobrunner.py tests/test_jobrunner_scope.py
git commit -m "feat(core): job events carry module/job_id/kind; runner keeps per-(module, kind) history"
```

### Task 2: `LivePreview.start(owner=)` mints a `stream_id` and stamps live events

**Files:**
- Modify: `tasni/core/livepreview.py:34-97` (`__init__`, `start`), `:99-125` (`_loop` signature + publishers), `:172-176` (error log)
- Test: `tests/test_livepreview.py` (append one test)

**Interfaces:**
- Produces: `LivePreview.start(analyze, *, owner: str | None = None, stream_id: str | None = None, ...) -> str` — returns the `stream_id`; mints one unless `stream_id` is given (an **internal resume** keeps the id the browser already holds); when already running returns the existing id for the same owner and raises `CameraBusy("live preview is owned by <owner>")` for a different one. `LivePreview.stream_id: str | None`; `LivePreview.owner: str | None`. Every `frame`/`gate`/`log` event the loop publishes carries `module=owner, stream_id=<id>`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_livepreview.py`)

```python
def test_start_mints_stream_id_and_stamps_live_events():
    frames = [SimpleNamespace(color=0, depth=None, telemetry=None)]

    class FakeStream:
        def read(self, *, with_depth=False, drain=False):
            return frames[0]

    class FakeCamera:
        @contextmanager
        def stream(self, **kw):
            yield FakeStream()

        def grab(self, **kw):
            return frames[0]

    bus = _FakeBus()
    lp = LivePreview(FakeCamera(), bus)
    sid = lp.start(lambda f: (b"jpg", {"ok": True, "gates": {"detected": True}}),
                   fps=50.0, owner="scan")
    assert isinstance(sid, str) and len(sid) == 32
    assert lp.stream_id == sid and lp.owner == "scan"
    assert lp.start(lambda f: (b"", {}), owner="scan") == sid     # already running
    _run_until(lambda: len(bus.events) >= 2, lp)
    lp.stop()
    kinds = {e.type for e in bus.events}
    assert {"frame", "gate"} <= kinds
    for e in bus.events:
        assert e.module == "scan" and e.stream_id == sid
        assert e.job_id is None and e.kind is None
    sid2 = lp.start(lambda f: (b"jpg", {"ok": True}), fps=50.0, owner="calibration")
    lp.stop()
    assert sid2 != sid and lp.owner == "calibration"
    # Internal resume (scan restarts its preview after a survey capture): keep the id.
    sid3 = lp.start(lambda f: (b"jpg", {"ok": True}), fps=50.0, owner="scan", stream_id=sid)
    lp.stop()
    assert sid3 == sid


def test_other_module_cannot_take_a_running_preview():
    from tasni.core.camera_lease import CameraBusy

    class FakeStream:
        def read(self, *, with_depth=False, drain=False):
            return SimpleNamespace(color=0, depth=None, telemetry=None)

    class FakeCamera:
        @contextmanager
        def stream(self, **kw):
            yield FakeStream()

    lp = LivePreview(FakeCamera(), _FakeBus())
    lp.start(lambda f: (b"", {}), fps=50.0, owner="scan")
    try:
        with pytest.raises(CameraBusy) as e:
            lp.start(lambda f: (b"", {}), owner="calibration")
        assert "owned by scan" in str(e.value)
    finally:
        lp.stop()
```
(add `import pytest` at the top of `tests/test_livepreview.py`)

- [ ] **Step 2: Run it to verify it fails**

Run: `py -3.10 -m pytest tests/test_livepreview.py -q -k stream_id`
Expected: FAIL — `TypeError: start() got an unexpected keyword argument 'owner'`.

- [ ] **Step 3: Implement**

In `tasni/core/livepreview.py`:

1. Add `import uuid` after `import time`.
2. In `__init__`, after `self.last: dict | None = None` add:
```python
        self.stream_id: str | None = None   # minted per start(); stamped on every event
        self.owner: str | None = None       # module id that started this preview
```
3. Change the `start` signature to add `owner: str | None = None, stream_id: str | None = None` (last keywords) and return `str`; replace its body's first lines and the thread creation:
```python
        from .camera_lease import CameraBusy
        if self.running:
            if owner is not None and owner != self.owner:
                raise CameraBusy(f"live preview is owned by {self.owner}")
            return self.stream_id or ""
        if self.lease is not None and not self.lease.acquire(LEASE_OWNER):
            raise CameraBusy(self.lease.owner)
        self.stream_id = stream_id or uuid.uuid4().hex   # given => internal resume
        self.owner = owner
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            args=(analyze, fps, timeout_s, color_only, quality, codec, bitrate,
                  with_depth, depth_probe, depth_period_s, scan_telemetry,
                  owner, self.stream_id),
            name="live-preview", daemon=True)
        self._thread.start()
        return self.stream_id
```
4. Extend `_loop`'s signature with `owner: str | None = None, stream_id: str | None = None` (after `scan_telemetry: bool = False`) and change the three publishers inside it:
```python
        def _publish_frame(jpeg: bytes) -> None:
            self.bus.publish(JobEvent("frame",
                {"jpeg_b64": base64.b64encode(jpeg).decode("ascii")},
                module=owner, stream_id=stream_id))

        def _publish_gate(metrics: dict) -> None:
            self.last = metrics
            self.bus.publish(JobEvent("gate", {**metrics, "live": True},
                                      module=owner, stream_id=stream_id))
```
and the loop's `except Exception` log publish:
```python
                self.bus.publish(JobEvent("log",
                    {"message": f"live preview error: {type(e).__name__}: {e}"},
                    module=owner, stream_id=stream_id))
```

- [ ] **Step 4: Run the tests**

Run: `py -3.10 -m pytest tests/test_livepreview.py tests/test_live_diag.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tasni/core/livepreview.py tests/test_livepreview.py
git commit -m "feat(core): live preview mints a stream_id and stamps its events with module + stream_id"
```

### Task 3: Stamp every module publish and job start; echo `job_id` / `stream_id`

**Files:**
- Modify: `tasni/modules/calibration/service.py` (27 publish sites), `tasni/modules/calibration/module.py:258,344,361,465` (+ responses), `tasni/modules/scan/service.py` (publish sites), `tasni/modules/scan/module.py:355,423,634,795,890` (+ responses), `tasni/modules/extrusion/module.py:381,440,467,557,597` (+ responses)
- Test: `tests/test_event_scoping_lint.py` (new)

**Interfaces:**
- Consumes: `EventBus.scoped`, `JobRunner.start(kind=, module=) -> str`, `LivePreview.start(owner=) -> str` (Tasks 1–2).
- Produces: every start endpoint's JSON includes `"job_id": <str>`; `/live/start` responses include `"stream_id": <str>`; no module ever calls `services.bus.publish(` unscoped.

- [ ] **Step 1: Write the failing lint test**

```python
# tests/test_event_scoping_lint.py
"""Every event a module publishes is module-scoped (spec §4.5): no raw
``services.bus.publish(`` remains, every ``jobs.start(`` names its module, every
``live.start(`` names its owner."""
from __future__ import annotations

import re
from pathlib import Path

MODULES = Path(__file__).resolve().parents[1] / "tasni" / "modules"


def _sources():
    return [p for p in MODULES.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_unscoped_module_publish():
    bad = [(p, i) for p in _sources()
           for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
           if re.search(r"services\.bus\.publish\(", line)]
    assert bad == [], f"unscoped publishes: {bad}"


def test_every_job_start_names_its_module():
    bad = [(p, i) for p in _sources()
           for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
           if "jobs.start(" in line and "module=" not in line
           and not line.strip().startswith("#")]
    # multi-line calls: the module= keyword may sit on the next line — accept if the
    # following two lines carry it.
    real_bad = []
    for p, i in bad:
        lines = p.read_text(encoding="utf-8").splitlines()
        window = " ".join(lines[i - 1:i + 2])
        if "module=" not in window:
            real_bad.append((p, i))
    assert real_bad == [], f"jobs.start without module=: {real_bad}"


def test_every_live_start_names_its_owner():
    bad = []
    for p in _sources():
        lines = p.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, 1):
            if "live.start(" in line and "def " not in line:
                window = " ".join(lines[i - 1:i + 6])
                if "owner=" not in window:
                    bad.append((p, i))
    assert bad == [], f"live.start without owner=: {bad}"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `py -3.10 -m pytest tests/test_event_scoping_lint.py -q`
Expected: 3 FAIL with the offending file/line lists.

- [ ] **Step 3: Scope the direct publishes (mechanical, Git Bash)**

```bash
cd tasni/modules
sed -i 's/services\.bus\.publish(JobEvent(/services.bus.scoped("calibration").publish(JobEvent(/g' calibration/service.py calibration/module.py
sed -i 's/services\.bus\.publish(JobEvent(/services.bus.scoped("scan").publish(JobEvent(/g' scan/service.py scan/module.py
grep -rn "bus.publish(" . | grep -v scoped   # must print nothing
```

- [ ] **Step 4: Stamp the job starts and echo `job_id`**

Calibration `module.py`:
```python
            services.live.stop()    # free the camera thread; the dry tour owns the robot
            job_id = services.jobs.start(SimTourJob(services), kind="sim_tour",
                                         module="calibration")
            return {"status": "started", "job_id": job_id}
```
```python
            self._active_job = CalibrationJob(services, params)
            job_id = services.jobs.start(self._active_job, kind="calibration",
                                         module="calibration")
            return {"status": "started", "job_id": job_id}
```
Scan `module.py` (`run()` method, line 355, and `/poses/simulate`, line 890):
```python
        job_id = services.jobs.start(self._active_job, kind="scan", module="scan")
        return {"status": "started", "job_id": job_id}
```
```python
            job_id = services.jobs.start(SimTourJob(
                services, target_prefix=sc.target_prefix,
                collision_self_pairs=sc.collision_self_pairs,
                collision_skip_wrist_links=sc.collision_skip_wrist_links),
                kind="sim_tour", module="scan")
            return {"status": "started", "job_id": job_id}
```
Extrusion `module.py` — for each of the five starts replace `name="extrusion-…"` with `kind="extrusion-…", module="extrusion"`, capture `job_id = services.jobs.start(...)` and add `"job_id": job_id` to the returned dict. Example (quick-sim):
```python
            try:
                job_id = services.jobs.start(
                    self._active_quick_job, kind="extrusion-quick-sim", module="extrusion")
            except JobBusy as exc:
                raise HTTPException(409, str(exc))
            return {"status": "started", "job_id": job_id, "mode": "QUICK_SIMULATION",
                    "collision_check_enabled": False,
                    "layer_indices": requested,
                    "approve_full_plan_on_pass": body.approve_full_plan,
                    "fingerprint": self._plan.fingerprint}
```
Do the same for `extrusion-dry-run` (line 440), `extrusion-print` (467), `extrusion-characterize` (557), `extrusion-measure` (597).

- [ ] **Step 5: Stamp the live starts and echo `stream_id`**

Calibration `module.py` `/live/start` (line 209 + 258-266):
```python
            if services.live.running:
                return {"status": "already running", "stream_id": services.live.stream_id}
```
```python
            try:
                stream_id = services.live.start(
                    analyze, fps=cc.preview_fps,
                    timeout_s=cc.preview_timeout_s, color_only=True,
                    quality=cc.preview_jpeg_quality,
                    codec=cc.preview_codec,
                    bitrate=cc.preview_h264_bitrate_kbps,
                    owner="calibration")
            except CameraBusy as e:
                raise HTTPException(409, str(e))
            return {"status": "started", "stream_id": stream_id}
```
Calibration `/intrinsics/live/start` (line 465): add `owner="calibration"` to the `services.live.start(` call and return `{"status": "started", "stream_id": stream_id}` the same way.
Scan `module.py` `/live/start` (line 604 + 612 + 795-801) — the same closure is
also called internally to **resume** the preview after a five-position capture
(`:416`, via `self._live_start`); a resume must keep the `stream_id` the browser
already holds, or every frame after the capture would be discarded:
```python
        @router.post("/live/start")
        def live_start(resume: bool = False) -> dict:
```
```python
            if services.live.running:
                return {"status": "already running", "stream_id": services.live.stream_id}
```
```python
            try:
                stream_id = services.live.start(
                    analyze, owner="scan",
                    stream_id=(services.live.stream_id if resume else None), **kwargs)
            except CameraBusy as e:
                ...
            return {"status": "started", "stream_id": stream_id}
```
and the internal restart in `survey_capture`'s `finally` (`:416`) becomes `live_start(resume=True)`.
The boundary publisher inside that closure (`:634`) must carry the stream id too:
```python
            def _publish_boundary(cb):
                services.bus.scoped("scan").publish(JobEvent("boundary", {
                    "outline_uv": cb["outline_uv"],
                    "polygon_uv": cb["polygon_uv"],
                    "overruns": cb["overruns"],
                    "contrast": cb["contrast"],
                }, stream_id=services.live.stream_id))
```
(`survey` events, `scan/service.py:1441`, and the locked `frame`/`gate` pair,
`:724-726`, are request-path publishes — module-scoped, no stream id; pages accept
id-less events of their own module. The spec's §4.5 wording is corrected to say so.)

Add to `tests/test_event_scoping_lint.py`:
```python
def test_boundary_publish_carries_stream_id():
    src = (MODULES / "scan" / "module.py").read_text(encoding="utf-8")
    i = src.index('JobEvent("boundary"')
    assert "stream_id=services.live.stream_id" in src[i:i + 400]
```

- [ ] **Step 6: Run the lint + the job tests**

Run: `py -3.10 -m pytest tests/test_event_scoping_lint.py tests/test_calibration_job.py tests/test_scan_job.py tests/test_extrusion_job.py tests/test_extrusion_measure.py tests/test_sim_tour.py -q`
Expected: PASS. Then `py -3.10 -c "import tasni.modules.calibration.module, tasni.modules.scan.module, tasni.modules.extrusion.module"` prints nothing.

- [ ] **Step 7: Commit**

```bash
git add tasni/modules tests/test_event_scoping_lint.py
git commit -m "feat(modules): scope every publish to its module; starts return job_id, live starts return stream_id"
```

### Task 4: Module `/status` returns `running` + per-kind `jobs` + workflow fields

**Files:**
- Modify: `tasni/modules/calibration/module.py:408-415`, `tasni/modules/scan/module.py:914-917`, `tasni/modules/extrusion/module.py:603-610`
- Modify (readers, minimal): `tasni/webui/src/pages/Calibration.tsx:153-158`, `tasni/webui/src/pages/Scan.tsx:296-303`, `tasni/webui/src/pages/Extrusion.tsx` (`Status` interface + `status?.running` uses)
- Test: `tests/test_module_status.py` (new)

**Interfaces:**
- Consumes: `JobRunner.module_status(module)` (Task 1).
- Produces: every module `GET /status` = `{running: {module, kind, job_id} | null, status: "idle"|"running"|"busy"|"done"|"error"|"cancelled", jobs: {kind: JobRecord}, result, error, ...workflow fields}` where `result`/`error` mirror the module's *primary* record (calibration: kind `calibration`; scan: kind `scan`; extrusion: the most recently finished record) for readers that predate per-kind records. Calibration adds `applied` (= `runs.read_active("calibration")`); scan adds `locked`, `prepared`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_module_status.py
"""Module /status: per-module, per-kind history + workflow fields (spec §4.5)."""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from tasni.core import runs
from tasni.core.config import AppConfig
from tasni.webapp.server import create_app


def _client(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "REPO_ROOT", tmp_path)
    app = create_app(AppConfig())
    return TestClient(app), app.state.services


def _run(services, fn, *, kind, module):
    services.jobs.start(fn, kind=kind, module=module)
    deadline = time.monotonic() + 3.0
    while services.jobs.running and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.02)


def test_status_shape_when_idle(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    for m in ("calibration", "scan", "extrusion"):
        s = client.get(f"/api/modules/{m}/status").json()
        assert s["running"] is None and s["status"] == "idle" and s["jobs"] == {}
        assert s["result"] is None and s["error"] is None
    assert client.get("/api/modules/calibration/status").json()["applied"] is None
    scan = client.get("/api/modules/scan/status").json()
    assert scan["locked"] is False and scan["prepared"] is False


def test_solve_record_survives_a_later_tour_and_another_module(tmp_path, monkeypatch):
    client, services = _client(tmp_path, monkeypatch)
    _run(services, lambda ctx: {"can_apply": True}, kind="calibration", module="calibration")
    _run(services, lambda ctx: {"all_ok": True}, kind="sim_tour", module="calibration")
    _run(services, lambda ctx: {"can_insert": True}, kind="scan", module="scan")
    cal = client.get("/api/modules/calibration/status").json()
    assert cal["result"] == {"can_apply": True}            # NOT the tour
    assert cal["jobs"]["sim_tour"]["result"] == {"all_ok": True}
    assert cal["jobs"]["calibration"]["status"] == "done"
    scan = client.get("/api/modules/scan/status").json()
    assert scan["result"] == {"can_insert": True}
    ext = client.get("/api/modules/extrusion/status").json()
    assert ext["jobs"] == {} and ext["result"] is None and ext["running"] is None


def test_extrusion_result_is_the_latest_finished_kind(tmp_path, monkeypatch):
    client, services = _client(tmp_path, monkeypatch)
    _run(services, lambda ctx: {"kind": "cylinder_quick_simulation"},
         kind="extrusion-quick-sim", module="extrusion")
    _run(services, lambda ctx: {"kind": "cylinder_print"},
         kind="extrusion-print", module="extrusion")
    ext = client.get("/api/modules/extrusion/status").json()
    assert ext["result"] == {"kind": "cylinder_print"}
    assert set(ext["jobs"]) == {"extrusion-quick-sim", "extrusion-print"}
    assert ext["live_print_enabled"] is False              # workflow fields still there
```

- [ ] **Step 2: Run to verify failure**

Run: `py -3.10 -m pytest tests/test_module_status.py -q`
Expected: FAIL — `KeyError: 'jobs'`.

- [ ] **Step 3: Rewrite the three `/status` routes**

Calibration (`module.py:408-415`):
```python
        @router.get("/status")
        def status() -> dict:
            """This module's job history + workflow state (spec §4.5). ``result`` /
            ``error`` mirror the SOLVE record (kind "calibration") so a later dry
            run can no longer hide an applyable result."""
            from ...core import runs

            s = services.jobs.module_status("calibration")
            solve = s["jobs"].get("calibration")
            return {**s,
                    "result": solve["result"] if solve else None,
                    "error": solve["error"] if solve else None,
                    "applied": runs.read_active("calibration")}
```
Scan (`module.py:914-917`):
```python
        @router.get("/status")
        def status() -> dict:
            s = services.jobs.module_status("scan")
            scan = s["jobs"].get("scan")
            return {**s,
                    "result": scan["result"] if scan else None,
                    "error": scan["error"] if scan else None,
                    "locked": self._locked_surface is not None,
                    "prepared": self._prepared_result is not None}
```
Extrusion (`module.py:603-610`) — replace the first four keys of the returned dict:
```python
        @router.get("/status")
        def status() -> dict:
            fingerprint = self._plan.fingerprint if self._plan else None
            s = services.jobs.module_status("extrusion")
            finished = [r for r in s["jobs"].values() if r["finished_at"] is not None]
            latest = max(finished, key=lambda r: r["finished_at"], default=None)
            return {
                **s,
                "result": latest["result"] if latest else None,
                "error": latest["error"] if latest else None,
                "fingerprint": fingerprint,
                # ... every remaining key exactly as today ...
```

- [ ] **Step 4: Update the three readers so the pages keep working**

`Calibration.tsx:153-158`:
```ts
  const refreshJob = useCallback(async () => {
    try {
      const s = await api.get<{ jobs?: { calibration?: { result: RunResult | null } } }>("/status");
      const res = s.jobs?.calibration?.result;
      if (res?.can_apply) { setResult(res); setCanApply(true); }
    } catch { /* no prior run — fine */ }
  }, []);
```
`Scan.tsx:296-303`:
```ts
  const refreshJob = useCallback(async () => {
    try {
      const s = await api.get<{ jobs?: { scan?: { result: ScanResult | null } } }>("/status");
      const res = s.jobs?.scan?.result;
      if (res?.can_insert) { setResult(res); setViewerNonce((n) => n + 1); }
    } catch { /* no prior scan */ }
  }, []);
```
`Extrusion.tsx`: in `interface Status` change `running: boolean` to
```ts
  running: { module: string; kind: string; job_id: string } | null;
  jobs: Record<string, { job_id: string; kind: string; status: string; result: any;
                         error: string | null; finished_at: number | null }>;
```
then directly after `const [status, setStatus] = useState<Status | null>(null);` add
```ts
  const jobRunning = !!status?.running;
```
and replace every other `status?.running` in the file with `jobRunning` (Git Bash: `grep -c "status?.running" tasni/webui/src/pages/Extrusion.tsx` first, then `sed -i 's/status?\.running/jobRunning/g' tasni/webui/src/pages/Extrusion.tsx`, then restore the one declaration line to `const jobRunning = !!status?.running;`). `if (status && !status.running)` (no `?.`) stays as is — a null object is falsy.

- [ ] **Step 5: Run tests + typecheck**

Run: `py -3.10 -m pytest tests/test_module_status.py tests/test_extrusion.py -q` → PASS.
Run: `cd tasni/webui && npm run typecheck && npm run build` → clean.

- [ ] **Step 6: Commit**

```bash
git add tasni/modules tasni/webui/src/pages tests/test_module_status.py
git commit -m "feat(modules): /status returns running + per-kind job records + workflow fields"
```

### Task 5a: `CellArbiter` — one atomic gate for connect, link, job start and live start

**Files:**
- Create: `tasni/core/arbiter.py`
- Modify: `tasni/modules/base.py:30-75` (`ServiceContainer.arbiter` + wiring), `tasni/core/jobrunner.py` (`__init__`, `start`), `tasni/core/livepreview.py` (`__init__`, `start`), `tasni/modules/calibration/module.py:344,361` and `tasni/modules/scan/module.py:355,890` (catch `JobBusy` → 409)
- Test: `tests/test_arbiter.py` (new)

**Interfaces:**
- Produces: `CellArbiter.hold(owner)` context manager — non-blocking, raises `CellBusy("cell is busy: <owner>")` when held; `CellArbiter.owner: str | None`. `JobRunner(bus, arbiter=None)` — `start()` raises `JobBusy` while the arbiter is held; `LivePreview(camera, bus, lease=None, arbiter=None)` — `start()` raises `CameraBusy` likewise. `ServiceContainer.arbiter`. Why: Connect's "is anything running?" check and a job/preview start could interleave, letting Connect reset the session under live work (plan review #3); every transition now takes the same gate.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_arbiter.py
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tasni.core.arbiter import CellArbiter, CellBusy
from tasni.core.camera_lease import CameraBusy
from tasni.core.jobrunner import JobBusy, JobRunner
from tasni.core.livepreview import LivePreview


class _Bus:
    def __init__(self): self.events = []
    def publish(self, ev): self.events.append(ev)


class _Camera:
    @contextmanager
    def stream(self, **kw):
        yield SimpleNamespace(read=lambda **k: SimpleNamespace(color=0, depth=None, telemetry=None))


def test_hold_is_exclusive_and_names_the_owner():
    a = CellArbiter()
    with a.hold("connect"):
        assert a.owner == "connect"
        with pytest.raises(CellBusy) as e:
            with a.hold("job:scan:scan"):
                pass
        assert str(e.value) == "cell is busy: connect"
    assert a.owner is None
    with a.hold("link"):
        pass


def test_job_start_refused_while_connect_holds_the_cell_and_releases_after():
    a = CellArbiter()
    r = JobRunner(_Bus(), arbiter=a)
    with a.hold("connect"):
        with pytest.raises(JobBusy) as e:
            r.start(lambda ctx: 1, kind="scan", module="scan")
        assert "cell is busy: connect" in str(e.value)
    gate = threading.Event()
    r.start(lambda ctx: gate.wait(2), kind="scan", module="scan")
    assert a.owner is None          # a start holds the gate only for the transition
    gate.set()
    time.sleep(0.05)


def test_live_start_refused_while_connect_holds_the_cell():
    a = CellArbiter()
    lp = LivePreview(_Camera(), _Bus(), arbiter=a)
    with a.hold("connect"):
        with pytest.raises(CameraBusy):
            lp.start(lambda f: (b"", {}), owner="scan")
    sid = lp.start(lambda f: (b"", {}), fps=50.0, owner="scan")
    lp.stop()
    assert sid and a.owner is None
```

- [ ] **Step 2: Run to verify failure**

Run: `py -3.10 -m pytest tests/test_arbiter.py -q` → `ModuleNotFoundError: tasni.core.arbiter`.

- [ ] **Step 3: Implement**

`tasni/core/arbiter.py`:
```python
"""One non-blocking gate for every transition of the shared cell's state.

Connect (which may reset the RoboDK session), link, job start and live-preview
start all take it, so "is anything running?" and "start" can no longer
interleave. Holders are either momentary (a start) or exclusive by nature (a
connect). Never blocks: a contender fails fast with CellBusy naming the holder.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager


class CellBusy(RuntimeError):
    pass


class CellArbiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.owner: str | None = None

    @contextmanager
    def hold(self, owner: str):
        if not self._lock.acquire(blocking=False):
            raise CellBusy(f"cell is busy: {self.owner}")
        self.owner = owner
        try:
            yield self
        finally:
            self.owner = None
            self._lock.release()
```
`tasni/core/jobrunner.py`: `def __init__(self, bus: EventBus, arbiter=None)` storing `self._arbiter = arbiter`; add `from contextlib import nullcontext` and `from .arbiter import CellBusy`; in `start` wrap the critical section:
```python
        kind = name or kind
        gate = self._arbiter.hold(f"job:{module}:{kind}") if self._arbiter else nullcontext()
        try:
            with gate, self._lock:
                # ... existing body up to and including self._thread.start() ...
        except CellBusy as e:
            raise JobBusy(str(e)) from e
        return job_id
```
`tasni/core/livepreview.py`: `def __init__(self, camera, bus, lease=None, arbiter=None)` storing `self.arbiter = arbiter`; in `start` wrap everything from the `if self.running:` check through `self._thread.start()` in
```python
        gate = self.arbiter.hold(f"live:{owner}") if self.arbiter else nullcontext()
        try:
            with gate:
                ...
        except CellBusy as e:
            raise CameraBusy(str(e)) from e
        return self.stream_id
```
`tasni/modules/base.py`: `from ..core.arbiter import CellArbiter`; add the field `arbiter: CellArbiter` directly before `calib_dry_tour_required`; in `build()`: `arbiter = CellArbiter()` and `jobs=JobRunner(bus, arbiter=arbiter)`, `live=LivePreview(camera, bus, lease=lease, arbiter=arbiter)`, `arbiter=arbiter`.
Calibration `module.py` (`/poses/simulate`, `/run`) and scan `module.py` (`run()`, `/poses/simulate`): import `JobBusy` from `...core.jobrunner` and wrap each `services.jobs.start(...)`:
```python
            try:
                job_id = services.jobs.start(..., kind=..., module=...)
            except JobBusy as exc:
                raise HTTPException(409, str(exc))
```
(extrusion already does this at all five sites).

- [ ] **Step 4: Run tests, commit**

Run: `py -3.10 -m pytest tests/test_arbiter.py tests/test_jobrunner_scope.py tests/test_livepreview.py tests/test_module_status.py -q` → PASS.
```bash
git add tasni/core/arbiter.py tasni/core/jobrunner.py tasni/core/livepreview.py tasni/modules/base.py tasni/modules/calibration/module.py tasni/modules/scan/module.py tests/test_arbiter.py
git commit -m "feat(core): CellArbiter makes connect/link/job-start/live-start transitions atomic"
```

### Task 5: Station-only `POST /api/rdk/connect` (409 + arbiter) and `POST /api/rdk/link`; consolidate the link helper

**Files:**
- Modify: `tasni/core/rdk_io.py:65-79` (replace `link_real_robot` with `ensure_robot_link` + back-compat wrapper)
- Modify: `tasni/modules/calibration/service.py:198-220` (`ensure_real_robot_link` delegates)
- Modify: `tasni/core/config.py:130-138` (docstring), `tasni/webapp/server.py` (two routes)
- Test: `tests/test_platform_connect.py` (new)

**Interfaces:**
- Produces: `ensure_robot_link(rdk, cfg, *, strict=False, log=None) -> dict` = `{enabled, connected, message, ip, configured}`; raises `RuntimeError` when `strict` and not connected. `link_real_robot(rdk, cfg) -> dict | None` kept as a wrapper. `POST /api/rdk/connect` → the `/api/rdk/status` shape (409 while a job / live preview / camera lease is active, 409 `"connect in progress"` on a concurrent call, 503 on timeout). `POST /api/rdk/link` → `/api/rdk/status` shape plus `link_attempt` (409 when the session is closed or a job runs). Neither route calls `connect_robot` except `/link`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_platform_connect.py
"""Platform connect/link (spec §4.5): station-only connect with 409 + lock; explicit
link; the consolidated core helper."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tasni.core.config import AppConfig
from tasni.core.rdk_io import ensure_robot_link, link_real_robot
from tasni.webapp.server import create_app


class FakeRobot:
    def __init__(self, valid=True):
        self._valid = valid

    def Valid(self):
        return self._valid


class FakeRdk:
    def __init__(self, *, valid=True, tool=True, linked=False, fail_first=0, block=None):
        self._valid, self._tool, self.linked = valid, tool, linked
        self.fail_first, self.block = fail_first, block
        self.robot_calls, self.connect_calls = 0, 0

    def robot(self):
        self.robot_calls += 1
        if self.block is not None:
            self.block.wait()
        if self.fail_first > 0:
            self.fail_first -= 1
            raise OSError("socket closed while the station loads")
        return FakeRobot(self._valid)

    def item_exists(self, name):
        return self._tool

    def robot_connected(self):
        return (self.linked, "ROBOTCOM_READY" if self.linked else "not connected")

    def robot_connection_params(self):
        return {"ip": "10.0.0.5", "port": 7000}

    def connect_robot(self, ip="", *, timeout_s=10.0, poll_s=0.4):
        self.connect_calls += 1
        self.linked = True
        return True, "ROBOTCOM_READY"


class FakeSession:
    def __init__(self, is_open=True):
        self.is_open = is_open
        self.resets = 0

    def reset(self):
        self.resets += 1


def _app(rdk: FakeRdk, *, session_open=True, timeout_s=0.0):
    app = create_app(AppConfig())
    s = app.state.services
    s.rdk, s.session = rdk, FakeSession(session_open)
    s.config.robodk.connect_timeout_s = timeout_s
    return TestClient(app), s


def test_connect_returns_status_shape_and_never_links():
    client, _ = _app(FakeRdk())
    r = client.post("/api/rdk/connect")
    assert r.status_code == 200
    body = r.json()
    assert body["connected"] and body["ready"] and body["tool_present"]
    assert body["robot_link"]["connected"] is False
    assert client.app.state.services.rdk.connect_calls == 0


def test_connect_reports_missing_tool():
    client, _ = _app(FakeRdk(tool=False))
    body = client.post("/api/rdk/connect").json()
    assert body["ready"] is False and "tool 'Realsense'" in body["missing"][0]


def test_connect_409_while_a_job_runs():
    client, s = _app(FakeRdk())
    gate = threading.Event()
    s.jobs.start(lambda ctx: gate.wait(3), kind="calibration", module="calibration")
    try:
        r = client.post("/api/rdk/connect")
        assert r.status_code == 409 and "calibration job" in r.json()["detail"]
    finally:
        gate.set()
        time.sleep(0.05)


def test_connect_409_while_live_preview_runs():
    client, s = _app(FakeRdk())
    s.live = SimpleNamespace(running=True, owner="scan", stream_id="x")
    r = client.post("/api/rdk/connect")
    assert r.status_code == 409 and "scan camera preview" in r.json()["detail"]


def test_connect_resets_session_on_socket_error_then_succeeds():
    rdk = FakeRdk(fail_first=1)
    client, s = _app(rdk, timeout_s=3.0)
    r = client.post("/api/rdk/connect")
    assert r.status_code == 200 and r.json()["ready"]
    assert s.session.resets == 1 and rdk.robot_calls == 2


def test_connect_503_when_robodk_never_answers():
    client, _ = _app(FakeRdk(fail_first=99), timeout_s=0.0)
    r = client.post("/api/rdk/connect")
    assert r.status_code == 503 and "still be loading" in r.json()["detail"]


def test_concurrent_connect_gets_409():
    block = threading.Event()
    client, _ = _app(FakeRdk(block=block))
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("first", client.post("/api/rdk/connect")))
    t.start()
    time.sleep(0.1)
    second = client.post("/api/rdk/connect")
    assert second.status_code == 409 and second.json()["detail"] == "connect in progress"
    block.set()
    t.join(3)
    assert out["first"].status_code == 200


def test_link_calls_connect_robot_and_reports():
    rdk = FakeRdk()
    client, _ = _app(rdk)
    r = client.post("/api/rdk/link")
    assert r.status_code == 200
    body = r.json()
    assert rdk.connect_calls == 1
    assert body["link_attempt"] == {"enabled": True, "connected": True,
                                    "message": "ROBOTCOM_READY", "ip": "10.0.0.5",
                                    "configured": True}
    assert body["robot_link"]["connected"] is True


def test_link_409_when_not_connected_or_busy():
    client, s = _app(FakeRdk(), session_open=False)
    assert client.post("/api/rdk/link").status_code == 409
    client, s = _app(FakeRdk())
    gate = threading.Event()
    s.jobs.start(lambda ctx: gate.wait(3), kind="scan", module="scan")
    try:
        assert client.post("/api/rdk/link").status_code == 409
    finally:
        gate.set()
        time.sleep(0.05)


def test_link_respects_auto_link_config():
    rdk = FakeRdk()
    client, s = _app(rdk)
    s.config.robodk.connect_robot_on_connect = False
    body = client.post("/api/rdk/link").json()
    assert body["link_attempt"]["enabled"] is False and rdk.connect_calls == 0


def test_ensure_robot_link_strict_raises_with_actionable_message():
    class Offline(FakeRdk):
        def connect_robot(self, ip="", *, timeout_s=10.0, poll_s=0.4):
            self.connect_calls += 1
            return False, "driver not running"

    cfg = SimpleNamespace(connect_robot_on_connect=True, robot_ip="",
                          robot_connect_timeout_s=0.1)
    state = ensure_robot_link(Offline(), cfg)
    assert state["connected"] is False and state["enabled"] is True
    with pytest.raises(RuntimeError) as e:
        ensure_robot_link(Offline(), cfg, strict=True)
    assert "real robot offline" in str(e.value) and "10.0.0.5" in str(e.value)
    assert link_real_robot(Offline(), SimpleNamespace(connect_robot_on_connect=False)) is None
    assert link_real_robot(FakeRdk(), cfg) == {"connected": True, "message": "ROBOTCOM_READY",
                                               "ip": "10.0.0.5", "configured": True}


def test_strict_verifies_a_manual_link_when_auto_link_is_off():
    manual = SimpleNamespace(connect_robot_on_connect=False, robot_ip="",
                             robot_connect_timeout_s=0.1)
    linked = FakeRdk(linked=True)
    state = ensure_robot_link(linked, manual, strict=True)
    assert state["connected"] is True and linked.connect_calls == 0   # verified, not attempted
    with pytest.raises(RuntimeError):
        ensure_robot_link(FakeRdk(linked=False), manual, strict=True)
    assert ensure_robot_link(FakeRdk(linked=False), manual)["enabled"] is False  # non-strict: no-op
```
Before running: `tests/test_extrusion_job.py:143` sets `connect_robot_on_connect = False`, so its fake rdk must answer `robot_connected()` — if that fake lacks it, add `def robot_connected(self): return True, "ROBOTCOM_READY"` to it (the print job now verifies the manual link strictly).

- [ ] **Step 2: Run to verify failure**

Run: `py -3.10 -m pytest tests/test_platform_connect.py -q`
Expected: FAIL — `ImportError: cannot import name 'ensure_robot_link'`.

- [ ] **Step 3: Consolidate the helper in `tasni/core/rdk_io.py`**

Replace `link_real_robot` (lines 65-79) with:

```python
def ensure_robot_link(rdk: "RdkIO", cfg, *, strict: bool = False, log=None) -> dict:
    """Link RoboDK's driver to the PHYSICAL controller and start position
    monitoring — that is what makes the model track the arm, so the seed pose
    target creation reads is the arm's ACTUAL pose (spec §4.5). Never moves the
    robot. Best-effort unless ``strict``: then a failed link raises a
    ``RuntimeError`` with the actionable message every real run relies on.
    ``{"enabled": False}`` (no attempt) when ``connect_robot_on_connect`` is off,
    so the operator can keep linking by hand."""
    auto = bool(getattr(cfg, "connect_robot_on_connect", False))
    if auto:
        ready, msg = rdk.connect_robot(cfg.robot_ip, timeout_s=cfg.robot_connect_timeout_s)
    elif strict:
        # Auto-link off = the operator links by hand. Strict callers (a real run)
        # still VERIFY that link exists; they never attempt one.
        ready, msg = rdk.robot_connected()
    else:
        return {"enabled": False, "connected": False, "message": "auto-link disabled",
                "ip": "", "configured": False}
    params = rdk.robot_connection_params()
    state = {"enabled": auto, "connected": bool(ready), "message": str(msg or ""),
             "ip": params.get("ip", ""), "configured": bool(params.get("ip"))}
    if log is not None:
        log(f"real robot {'ONLINE' if ready else 'OFFLINE'}: {msg or ''}")
    if strict and not ready:
        where = (f" at {params['ip']}" if params.get("ip")
                 else " (no controller IP set on the robot in RoboDK)")
        raise RuntimeError(
            f"real robot offline — RoboDK is not linked to the KUKA "
            f"controller{where}: {msg or 'not ready'}. Check the controller is on, "
            f"the RoboDK robot driver is running, and the network, then run again. "
            f"(The link gives RoboDK monitoring and control of the arm; only a run "
            f"command moves it, and a run needs this link.)")
    return state


def link_real_robot(rdk: "RdkIO", cfg) -> dict | None:
    """Back-compat summary of a best-effort link (``None`` when auto-link is off).
    Prefer :func:`ensure_robot_link`."""
    state = ensure_robot_link(rdk, cfg)
    if not state["enabled"]:
        return None
    return {k: state[k] for k in ("connected", "message", "ip", "configured")}
```

In `tasni/modules/calibration/service.py:198-220` replace the body of `ensure_real_robot_link` with:
```python
def ensure_real_robot_link(rdk: RdkIO, robodk_cfg, *, log=None) -> None:
    """Strict link before real motion — delegates to the shared core helper
    (spec §4.5). Kept because scan and extrusion import it from here."""
    from ...core.rdk_io import ensure_robot_link
    ensure_robot_link(rdk, robodk_cfg, strict=True, log=log)
```

In `tasni/core/config.py:130-138` replace the comment block above `connect_robot_on_connect` with:
```python
    # Real-robot driver link. With the driver linked (and monitoring) the RoboDK
    # model tracks the physical arm, so the seed pose Create-targets / Lock surface
    # read is the arm's ACTUAL pose. With this on, the app links the controller
    # (best-effort; it may be off) when the Aim/Survey step starts (POST
    # /api/rdk/link) and every real run re-links strictly right before moving.
    # Platform Connect itself never links. Off = the operator links by hand in
    # RoboDK's "Connect robot" panel (and require_live_pose must be off too).
    connect_robot_on_connect: bool = True
```

- [ ] **Step 4: Add the two routes to `tasni/webapp/server.py`**

Add `import time` to the imports, `from ..core.arbiter import CellBusy`, `from ..core.rdk_io import ensure_robot_link`, and inside `create_app` after the `rdk_status` route:

```python
    def _busy_reason() -> str | None:
        if services.jobs.running:
            cur = services.jobs.current
            return (f"a {cur.module} job ({cur.kind}) is running" if cur
                    else "a job is running")
        if services.live.running:
            return f"the {services.live.owner or 'live'} camera preview is running"
        if services.camera_lease.held:
            return f"the camera is in use by {services.camera_lease.owner}"
        return None

    def _connect_locked() -> dict:
        c = services.config.robodk
        deadline = time.monotonic() + float(c.connect_timeout_s)
        last_err: Exception | None = None
        while True:
            try:
                if services.rdk.robot().Valid():
                    return rdk_status()
                last_err = None            # connected; robot not loaded yet
            except Exception as e:         # socket/timeout while RoboDK loads
                last_err = e
                try:
                    services.session.reset()
                except Exception:
                    pass
            if time.monotonic() >= deadline:
                break
            time.sleep(1.0)
        if last_err is not None:
            raise HTTPException(503,
                f"RoboDK didn't become ready within {float(c.connect_timeout_s):.0f}s "
                f"— it may still be loading the station. Give it a moment and click "
                f"Connect again. ({last_err})")
        return rdk_status()

    @app.post("/api/rdk/connect")
    def rdk_connect() -> dict:
        """Station-only connect (spec §4.5): open/attach the cell's station, poll
        through the slow first load, report robot + camera tool. Never links the
        real robot — see /api/rdk/link. Holds the cell arbiter for the WHOLE
        connect, so the busy check below is atomic: no job or preview can start
        after it (they fail fast with 409 "cell is busy: connect") — and its error
        path resets the session, which must never happen under a running job."""
        try:
            with services.arbiter.hold("connect"):
                reason = _busy_reason()
                if reason:
                    raise HTTPException(409, f"cannot (re)connect while {reason}")
                return _connect_locked()
        except CellBusy as e:
            holder = str(e).rsplit(": ", 1)[-1]
            raise HTTPException(409, "connect in progress" if holder == "connect"
                                else f"cannot (re)connect while {holder} is starting")

    @app.post("/api/rdk/link")
    def rdk_link() -> dict:
        """Best-effort link of RoboDK's driver to the physical controller (starts
        position monitoring; never moves the robot). Called when Aim/Survey starts;
        runs re-check strictly before motion. Allowed while the live preview runs
        (that is exactly when it is needed); refused while a job runs; serialized
        against connect through the arbiter (it holds it for up to
        robot_connect_timeout_s, so start the camera BEFORE linking — Task 8b)."""
        if not services.session.is_open:
            raise HTTPException(409, "connect the cell first")
        if services.jobs.running:
            raise HTTPException(409, "a job is running")
        try:
            with services.arbiter.hold("link"):
                attempt = ensure_robot_link(services.rdk, services.config.robodk)
        except CellBusy as e:
            raise HTTPException(409, f"cannot link now: {e}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(503, f"RoboDK unavailable: {e}")
        return {**rdk_status(), "link_attempt": attempt}
```

Add to `tests/test_platform_connect.py` (imports: `from tasni.core.camera_lease import CameraBusy`, `from tasni.core.jobrunner import JobBusy`):
```python
def test_job_and_live_start_refused_while_connecting():
    block = threading.Event()
    client, s = _app(FakeRdk(block=block))
    t = threading.Thread(target=lambda: client.post("/api/rdk/connect"))
    t.start()
    time.sleep(0.1)
    with pytest.raises(JobBusy) as e:
        s.jobs.start(lambda ctx: 1, kind="scan", module="scan")
    assert "cell is busy: connect" in str(e.value)
    with pytest.raises(CameraBusy):
        s.live.start(lambda f: (b"", {}), owner="scan")   # refused before any camera I/O
    block.set()
    t.join(3)
    assert s.arbiter.owner is None
```

- [ ] **Step 5: Run the tests**

Run: `py -3.10 -m pytest tests/test_platform_connect.py tests/test_robot_link.py tests/test_calibration_job.py tests/test_extrusion_job.py -q`
Expected: PASS (the connect tests take ~1–2 s because of the 1 s poll).

- [ ] **Step 6: Commit**

```bash
git add tasni/core/rdk_io.py tasni/core/config.py tasni/modules/calibration/service.py tasni/webapp/server.py tests/test_platform_connect.py
git commit -m "feat(platform): station-only POST /api/rdk/connect (409+lock) and POST /api/rdk/link; consolidate ensure_robot_link"
```

### Task 6: Server-side live-pose gate on target creation and surface lock; `pose_live` in the calibration gate

**Files:**
- Modify: `tasni/core/config.py` (add `require_live_pose` after `robot_connect_timeout_s`)
- Modify: `tasni/modules/calibration/module.py:313-331` (`/poses/generate`), `:200-256` (`analyze` closure in `/live/start`)
- Modify: `tasni/modules/scan/module.py` (`surface_lock` method, after the `jobs.running` check)
- Test: `tests/test_live_pose_gate.py` (new)

**Interfaces:**
- Produces: `RoboDKConfig.require_live_pose: bool = True`. `POST /api/modules/calibration/poses/generate` and `POST /api/modules/scan/surface/lock` return **409** with `detail = {"error": "pose_not_live", "message": "...", "driver": "<msg>"}` unless `rdk.robot_connected()[0]` (skipped when `require_live_pose` is False). Calibration `gate` events gain `pose_live: bool` and `pose_live_required: bool`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_live_pose_gate.py
"""Target creation and surface locking refuse without a live pose (spec §4.5)."""
from __future__ import annotations

import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from tasni.core.config import AppConfig
from tasni.webapp.server import create_app


class FakeRdk:
    def __init__(self, linked):
        self.linked = linked

    def robot_connected(self):
        return (self.linked, "ROBOTCOM_READY" if self.linked else "driver not running")


def _app(monkeypatch, *, linked, require=True):
    app = create_app(AppConfig())
    s = app.state.services
    s.rdk = FakeRdk(linked)
    s.session = SimpleNamespace(is_open=True, reset=lambda: None)
    s.config.robodk.require_live_pose = require
    monkeypatch.setattr("tasni.modules.calibration.module.generate_calibration_targets",
                        lambda services: {"created": 3, "gate": {}})
    monkeypatch.setattr("tasni.modules.scan.module.lock_scan_surface",
                        lambda services, **kw: SimpleNamespace(
                            lock_token="t1", gate_payload={}, survey_record=None))
    cal = next(m for m in app.state.registry.all() if m.id == "calibration")
    cal._seed_ready_until = time.monotonic() + 60
    return TestClient(app)


def test_create_targets_refuses_without_live_pose(monkeypatch):
    client = _app(monkeypatch, linked=False)
    r = client.post("/api/modules/calibration/poses/generate")
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["error"] == "pose_not_live" and d["driver"] == "driver not running"
    assert "Link the real robot" in d["message"]


def test_create_targets_passes_with_live_pose(monkeypatch):
    client = _app(monkeypatch, linked=True)
    r = client.post("/api/modules/calibration/poses/generate")
    assert r.status_code == 200 and r.json()["created"] == 3


def test_surface_lock_refuses_without_live_pose(monkeypatch):
    client = _app(monkeypatch, linked=False)
    r = client.post("/api/modules/scan/surface/lock", json={})
    assert r.status_code == 409 and r.json()["detail"]["error"] == "pose_not_live"


def test_surface_lock_passes_with_live_pose(monkeypatch):
    client = _app(monkeypatch, linked=True)
    r = client.post("/api/modules/scan/surface/lock", json={})
    assert r.status_code == 200 and r.json()["status"] == "locked"


def test_require_live_pose_off_bypasses_both_gates(monkeypatch):
    client = _app(monkeypatch, linked=False, require=False)
    assert client.post("/api/modules/calibration/poses/generate").status_code == 200
    assert client.post("/api/modules/scan/surface/lock", json={}).status_code == 200
```

- [ ] **Step 2: Run to verify failure**

Run: `py -3.10 -m pytest tests/test_live_pose_gate.py -q`
Expected: the two `refuses` tests FAIL (200 instead of 409); `require_live_pose` attribute error on the bypass test.

- [ ] **Step 3: Config + gate helper + the two checks**

`tasni/core/config.py`, after `robot_connect_timeout_s: float = 10.0`:
```python
    # Hard gate (spec §4.5): target creation and surface locking refuse unless the
    # driver is linked, because the model pose they seed from is otherwise stale.
    # Bench escape hatch: set False together with connect_robot_on_connect=False.
    require_live_pose: bool = True
```

Add to `tasni/core/rdk_io.py` right after `link_real_robot`:
```python
def pose_not_live_detail(rdk: "RdkIO", cfg) -> dict | None:
    """The 409 payload for a stale-pose refusal, or ``None`` when the pose is live
    (or the gate is disabled). One place, so calibration and scan refuse alike."""
    if not getattr(cfg, "require_live_pose", True):
        return None
    ready, msg = rdk.robot_connected()
    if ready:
        return None
    return {"error": "pose_not_live",
            "message": "Real robot not linked — the model pose may be stale. "
                       "Link the real robot (Aim step) and try again.",
            "driver": str(msg or "")}
```

Calibration `module.py` `/poses/generate` — after the `_seed_ready_until` check:
```python
            from ...core.rdk_io import pose_not_live_detail
            stale = pose_not_live_detail(services.rdk, services.config.robodk)
            if stale is not None:
                raise HTTPException(409, stale)
```
Scan `module.py` `surface_lock` method — after `raise HTTPException(409, "a job is already running")`:
```python
        from ...core.rdk_io import pose_not_live_detail
        stale = pose_not_live_detail(services.rdk, services.config.robodk)
        if stale is not None:
            raise HTTPException(409, stale)
```

Calibration `/live/start` `analyze` closure — add `last_driver_check = 0.0` and `driver_ok = False` next to `stable_since = None`, make them `nonlocal` inside `analyze`, and before `payload = reading.to_dict()`:
```python
                # Driver-link status is a cheap RoboDK round trip but not free at
                # video rate — recheck at most every 2 s (same pattern as scan).
                if now - last_driver_check >= 2.0:
                    last_driver_check = now
                    try:
                        driver_ok = bool(services.rdk.robot_connected()[0])
                    except Exception:
                        driver_ok = False
```
and after `payload["stable_required_s"] = ...`:
```python
                payload["pose_live"] = driver_ok
                payload["pose_live_required"] = bool(c.robodk.require_live_pose)
```

- [ ] **Step 4: Run tests**

Run: `py -3.10 -m pytest tests/test_live_pose_gate.py tests/test_platform_connect.py tests/test_scan_config.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add tasni/core/config.py tasni/core/rdk_io.py tasni/modules/calibration/module.py tasni/modules/scan/module.py tests/test_live_pose_gate.py
git commit -m "feat(gate): refuse target creation / surface lock without a live robot pose; calibration gate reports pose_live"
```

### Task 7: `GET /api/readiness` — recorded vs present

**Files:**
- Create: `tasni/webapp/readiness.py`
- Modify: `tasni/webapp/server.py` (one route)
- Test: `tests/test_readiness.py` (new)

**Interfaces:**
- Produces: `readiness(services) -> dict` = `{"calibration": Card, "surface": Card}` with `Card = {"recorded": dict | None, "present": bool | None, "reason": str}`; `present` is `None` (reason `"not connected"` / `"job running"`) whenever the station may not be queried. `GET /api/readiness` returns it. The print-plan card is *not* here — the Dashboard reads `/api/modules/extrusion/status` (spec §4.7).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_readiness.py
"""GET /api/readiness distinguishes recorded (active.json) from present (station)."""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from tasni.core import runs
from tasni.core.config import AppConfig
from tasni.webapp.server import create_app

X = np.eye(4); X[:3, 3] = [40.0, -15.0, 55.0]


class FakeRdk:
    def __init__(self, *, items=(), tool_T=None):
        self.items = set(items)
        self.tool_T = X if tool_T is None else tool_T

    def item_exists_as(self, name, kind):
        return (name, kind) in self.items

    def item_exists(self, name):
        return any(n == name for n, _ in self.items)

    def get_tool_pose_T(self, name):
        return self.tool_T


def _app(tmp_path, monkeypatch, rdk, *, session_open=True):
    monkeypatch.setattr(runs, "REPO_ROOT", tmp_path)
    app = create_app(AppConfig())
    s = app.state.services
    s.rdk = rdk
    s.session = SimpleNamespace(is_open=session_open, reset=lambda: None)
    return TestClient(app), s


def _record_calibration(tmp_path):
    d = runs.run_dir("calibration", "20260814-1022", tmp_path)
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps({"X_cam2gripper": X.tolist()}), encoding="utf-8")
    runs.write_active("calibration", {"module": "calibration", "run_id": "20260814-1022",
                                      "applied_at": "2026-08-14T10:22:00", "tool": "Realsense",
                                      "quality": {"verdict": "pass"}}, tmp_path)


def _record_surface(tmp_path):
    runs.write_active("scan", {"module": "scan", "run_id": "20260827-1500",
                               "applied_at": "2026-08-27T15:00:00",
                               "frame": "Frame_Scan", "rectangle": "Rect_Scan",
                               "size_mm": [1200, 800],
                               "boundary_provenance": "camera measured - complete boundary"}, tmp_path)


def test_nothing_recorded(tmp_path, monkeypatch):
    client, _ = _app(tmp_path, monkeypatch, FakeRdk())
    r = client.get("/api/readiness").json()
    assert r["calibration"] == {"recorded": None, "present": None, "reason": "nothing applied"}
    assert r["surface"] == {"recorded": None, "present": None, "reason": "nothing inserted"}


def test_recorded_but_not_connected(tmp_path, monkeypatch):
    _record_calibration(tmp_path); _record_surface(tmp_path)
    client, _ = _app(tmp_path, monkeypatch, FakeRdk(), session_open=False)
    r = client.get("/api/readiness").json()
    assert r["calibration"]["recorded"]["run_id"] == "20260814-1022"
    assert r["calibration"]["present"] is None and r["calibration"]["reason"] == "not connected"
    assert r["surface"]["recorded"]["frame"] == "Frame_Scan"
    assert r["surface"]["present"] is None


def test_present_when_station_matches(tmp_path, monkeypatch):
    _record_calibration(tmp_path); _record_surface(tmp_path)
    rdk = FakeRdk(items={("Realsense", "tool"), ("Frame_Scan", "frame"), ("Rect_Scan", "object")})
    client, _ = _app(tmp_path, monkeypatch, rdk)
    r = client.get("/api/readiness").json()
    assert r["calibration"]["present"] is True and r["calibration"]["reason"] == "tool pose matches the applied run"
    assert r["surface"]["present"] is True


def test_absent_when_tool_moved_or_frame_deleted(tmp_path, monkeypatch):
    _record_calibration(tmp_path); _record_surface(tmp_path)
    moved = np.eye(4); moved[:3, 3] = [40.0, -15.0, 56.0]
    rdk = FakeRdk(items={("Realsense", "tool"), ("Rect_Scan", "object")}, tool_T=moved)
    client, _ = _app(tmp_path, monkeypatch, rdk)
    r = client.get("/api/readiness").json()
    assert r["calibration"]["present"] is False
    assert r["calibration"]["reason"] == "tool pose differs from the applied run — re-apply"
    assert r["surface"]["present"] is False
    assert r["surface"]["reason"] == "frame 'Frame_Scan' is not in the station — re-insert"


def test_null_while_a_job_runs(tmp_path, monkeypatch):
    _record_calibration(tmp_path)
    client, s = _app(tmp_path, monkeypatch, FakeRdk(items={("Realsense", "tool")}))
    gate = threading.Event()
    s.jobs.start(lambda ctx: gate.wait(3), kind="scan", module="scan")
    try:
        r = client.get("/api/readiness").json()
        assert r["calibration"]["present"] is None and r["calibration"]["reason"] == "job running"
    finally:
        gate.set(); time.sleep(0.05)
```

- [ ] **Step 2: Run to verify failure**

Run: `py -3.10 -m pytest tests/test_readiness.py -q` → FAIL 404.

- [ ] **Step 3: Implement `tasni/webapp/readiness.py`**

```python
"""Cell readiness for the Dashboard (spec §4.7): what is RECORDED (``active.json``)
versus what is PRESENT in the station that is open right now. A recorded
calibration or surface can be gone from the station (unsaved reload, deleted
item), so the two are reported separately and never conflated."""
from __future__ import annotations

import numpy as np

from ..core import runs

POSE_TOL = 1e-3     # mm / dimensionless rotation terms — an exact re-read of the applied pose


def _card(recorded, present, reason) -> dict:
    return {"recorded": recorded, "present": present, "reason": reason}


def _station_block_reason(services) -> str | None:
    if not services.session.is_open:
        return "not connected"
    if services.jobs.running:
        return "job running"
    return None


def calibration_readiness(services) -> dict:
    recorded = runs.read_active("calibration")
    if not recorded:
        return _card(None, None, "nothing applied")
    blocked = _station_block_reason(services)
    if blocked:
        return _card(recorded, None, blocked)
    tool = str(recorded.get("tool") or services.config.robodk.camera_tool)
    try:
        if not services.rdk.item_exists_as(tool, "tool"):
            return _card(recorded, False, f"tool '{tool}' is not in the station — re-apply")
        report = runs.load_report("calibration", str(recorded.get("run_id")))
        X = np.asarray(report["X_cam2gripper"], dtype=float)
        T = np.asarray(services.rdk.get_tool_pose_T(tool), dtype=float)
    except Exception as e:  # noqa: BLE001 - readiness must never take the Dashboard down
        return _card(recorded, None, f"could not verify: {e}")
    if X.shape == (4, 4) and T.shape == (4, 4) and np.allclose(T, X, atol=POSE_TOL):
        return _card(recorded, True, "tool pose matches the applied run")
    return _card(recorded, False, "tool pose differs from the applied run — re-apply")


def surface_readiness(services) -> dict:
    recorded = runs.read_active("scan")
    if not recorded or not recorded.get("frame"):
        return _card(None, None, "nothing inserted")
    blocked = _station_block_reason(services)
    if blocked:
        return _card(recorded, None, blocked)
    frame, rect = str(recorded["frame"]), recorded.get("rectangle")
    try:
        if not services.rdk.item_exists_as(frame, "frame"):
            return _card(recorded, False, f"frame '{frame}' is not in the station — re-insert")
        if rect and not services.rdk.item_exists(str(rect)):
            return _card(recorded, False, f"rectangle '{rect}' is not in the station — re-insert")
    except Exception as e:  # noqa: BLE001
        return _card(recorded, None, f"could not verify: {e}")
    return _card(recorded, True, "frame and rectangle are in the station")


def readiness(services) -> dict:
    return {"calibration": calibration_readiness(services),
            "surface": surface_readiness(services)}
```

`tasni/webapp/server.py` — add `from .readiness import readiness as cell_readiness` and, after `/api/runs/active`:
```python
    @app.get("/api/readiness")
    def readiness_route() -> dict:
        """Dashboard journey strip: recorded vs present for calibration + surface
        (spec §4.7). Never queries the station while a job runs."""
        return cell_readiness(services)
```

- [ ] **Step 4: Run tests**

Run: `py -3.10 -m pytest tests/test_readiness.py tests/test_runs.py -q` → PASS.

- [ ] **Step 5: Commit + push**

```bash
git add tasni/webapp/readiness.py tasni/webapp/server.py tests/test_readiness.py
git commit -m "feat(platform): GET /api/readiness distinguishes recorded from present-in-station"
git push -u origin ux-phase0
```

### Task 8a: `PlatformProvider` + topbar Connect (frontend foundation)

**Files:**
- Create: `tasni/webui/src/platform/PlatformProvider.tsx`
- Modify: `tasni/webui/src/api/events.tsx` (becomes a thin shim until 8d), `tasni/webui/src/App.tsx`, `tasni/webui/src/components/Layout.tsx`, `tasni/webui/src/pages/Home.tsx:1-20`, `tasni/webui/src/index.css` (topbar additions)
- Delete: `tasni/webui/src/api/useHealth.ts`

**Interfaces:**
- Produces: `usePlatform(): { health, rdk, ready, connecting, connectError, connect(), link(), refreshRdk(), eventsConnected, subscribe(module, handler) }` with the types below. `subscribe("*", h)` receives every event; `subscribe("scan", h)` only events whose `module === "scan"`. `rdk` refresh rules: mount, after `connect()`/`link()`, on every terminal job event (`result`, `error`, `status` with `cancelled`), a 10 s poll that **skips while `health.job.running`**, and `ready` drops immediately when `health.robodk.ok === false`.

- [ ] **Step 1: Write `tasni/webui/src/platform/PlatformProvider.tsx`**

```tsx
// One provider for everything cell-wide (spec §4.5): health, the shared RoboDK
// session state, Connect/Link, and the job-event WebSocket with per-module
// filtering. Pages never own a connection banner or a raw subscription again.
import {
  createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode,
} from "react";
import { apiGet, apiPost } from "../api/client";

export interface Health {
  robodk: { ok: boolean | null; detail: string };
  camera: { ok: boolean | null; state: "connected" | "offline" | "in_use";
            detail: string; route: string; endpoint: string };
  job: { status: string; running: boolean };
}
export interface RobotLink { connected: boolean; message: string; ip: string; configured: boolean; }
export interface RdkStatus {
  connected: boolean; ready: boolean; robot: string; tool: string; missing: string[];
  robot_valid?: boolean; tool_present?: boolean; robot_link?: RobotLink | null;
  error?: string;
  link_attempt?: { enabled: boolean; connected: boolean; message: string } | null;
}
export interface JobEvent {
  type: "progress" | "log" | "frame" | "gate" | "result" | "error" | "status"
      | "survey" | "boundary" | string;
  payload: any;
  module: string | null;
  job_id: string | null;
  kind: string | null;
  stream_id: string | null;
}
export type Handler = (e: JobEvent) => void;

export interface PlatformValue {
  health: Health | null;
  rdk: RdkStatus | null;
  ready: boolean;
  connecting: boolean;
  connectError: string | null;
  connect: () => Promise<void>;
  link: () => Promise<RdkStatus | null>;
  refreshRdk: () => Promise<void>;
  eventsConnected: boolean;
  subscribe: (module: string, handler: Handler) => () => void;
}

const HEALTH_MS = 4000;
const RDK_POLL_MS = 10000;
const TERMINAL = new Set(["result", "error"]);

const PlatformContext = createContext<PlatformValue>({
  health: null, rdk: null, ready: false, connecting: false, connectError: null,
  connect: async () => {}, link: async () => null, refreshRdk: async () => {},
  eventsConnected: false, subscribe: () => () => {},
});

export function PlatformProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<Health | null>(null);
  const [rdk, setRdk] = useState<RdkStatus | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [connectError, setConnectError] = useState<string | null>(null);
  const [eventsConnected, setEventsConnected] = useState(false);
  const handlers = useRef<Map<Handler, string>>(new Map());
  const healthRef = useRef<Health | null>(null);

  const refreshRdk = useCallback(async () => {
    try { setRdk(await apiGet<RdkStatus>("/api/rdk/status")); }
    catch { /* opportunistic — the health poll will flag a dead backend */ }
  }, []);

  // Health: 4 s. A failed RoboDK probe drops readiness immediately (spec §4.5).
  useEffect(() => {
    let alive = true;
    const tick = () => apiGet<Health>("/api/health").then((h) => {
      if (!alive) return;
      healthRef.current = h;
      setHealth(h);
      if (h.robodk.ok === false) {
        setRdk((prev) => prev && prev.ready ? { ...prev, connected: false, ready: false } : prev);
      }
    }).catch(() => {});
    tick();
    const t = setInterval(tick, HEALTH_MS);
    return () => { alive = false; clearInterval(t); };
  }, []);

  // RoboDK session state: on mount + a slow poll that pauses while a job runs.
  useEffect(() => {
    refreshRdk();
    const t = setInterval(() => {
      if (healthRef.current?.job.running) return;
      refreshRdk();
    }, RDK_POLL_MS);
    return () => clearInterval(t);
  }, [refreshRdk]);

  // One WebSocket; fan out by module; refresh rdk on every terminal job event.
  useEffect(() => {
    let stopped = false;
    let ws: WebSocket | undefined;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const connectWs = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onopen = () => setEventsConnected(true);
      ws.onclose = () => {
        setEventsConnected(false);
        if (!stopped) timer = setTimeout(connectWs, 1500);
      };
      ws.onmessage = (m) => {
        const ev: JobEvent = JSON.parse(m.data);
        handlers.current.forEach((module, h) => {
          if (module !== "*" && ev.module !== module) return;
          try { h(ev); } catch (e) { console.error(e); }
        });
        if (TERMINAL.has(ev.type) || (ev.type === "status" && ev.payload?.status === "cancelled")) {
          refreshRdk();
        }
      };
    };
    connectWs();
    return () => { stopped = true; if (timer) clearTimeout(timer); ws?.close(); };
  }, [refreshRdk]);

  const subscribe = useCallback((module: string, h: Handler) => {
    handlers.current.set(h, module);
    return () => { handlers.current.delete(h); };
  }, []);

  const connect = useCallback(async () => {
    setConnecting(true); setConnectError(null);
    try {
      setRdk(await apiPost<RdkStatus>("/api/rdk/connect"));
    } catch (e: any) {
      setConnectError(e.message);
      await refreshRdk();
    } finally { setConnecting(false); }
  }, [refreshRdk]);

  const link = useCallback(async () => {
    try {
      const r = await apiPost<RdkStatus>("/api/rdk/link");
      setRdk(r);
      return r;
    } catch { await refreshRdk(); return null; }
  }, [refreshRdk]);

  const value: PlatformValue = {
    health, rdk,
    // Truthful readiness: a stale-green rdk poll must not outrank a failed
    // health probe (plan review #5).
    ready: rdk?.ready === true && health?.robodk.ok !== false,
    connecting, connectError,
    connect, link, refreshRdk, eventsConnected, subscribe,
  };
  return <PlatformContext.Provider value={value}>{children}</PlatformContext.Provider>;
}

export const usePlatform = () => useContext(PlatformContext);

// One line about the real-robot driver link, shared by the topbar and the pages.
export function robotLinkNote(link?: RobotLink | null): string {
  if (!link) return "";
  if (link.connected) return `Real robot ONLINE${link.ip ? ` (${link.ip})` : ""}.`;
  if (!link.configured) return "Real robot not linked — no controller IP set on the robot in RoboDK.";
  return `Real robot OFFLINE — ${link.message || "controller not reachable"}. `
    + "Power the controller + driver to run on the real arm.";
}
```

- [ ] **Step 2: Turn `api/events.tsx` into a shim** (pages still call `useEvents().subscribe(h)` until 8b–8d)

Replace the whole file with:
```tsx
// Transitional shim (removed in Task 8d): old pages call subscribe(handler) with
// no module filter. New code must use usePlatform().subscribe(module, handler).
import { usePlatform, type JobEvent } from "../platform/PlatformProvider";
export type { JobEvent };
export function useEvents() {
  const { eventsConnected, subscribe } = usePlatform();
  return { connected: eventsConnected,
           subscribe: (h: (e: JobEvent) => void) => subscribe("*", h) };
}
```
Delete `src/api/useHealth.ts`.

- [ ] **Step 3: Wire the provider and the topbar**

`App.tsx`: replace `import { EventsProvider } from "./api/events";` with `import { PlatformProvider } from "./platform/PlatformProvider";` and `<EventsProvider>…</EventsProvider>` with `<PlatformProvider>…</PlatformProvider>`.

`components/Layout.tsx` — replace the imports of `useEvents`/`useHealth` with `import { usePlatform, robotLinkNote } from "../platform/PlatformProvider";`, replace `const { connected } = useEvents(); const health = useHealth();` with
```tsx
  const { health, rdk, ready, connecting, connectError, connect, eventsConnected } = usePlatform();
  // Job OR camera preview/lease (health reports camera.state "in_use" for both).
  // Advisory only — the server's arbiter is the real guard (409).
  const busy = !!health?.job.running || health?.camera.state === "in_use";
  const rdkSummary = ready
    ? `${rdk?.robot} · ${rdk?.tool}${rdk?.robot_link ? " · " + robotLinkNote(rdk.robot_link).replace(/\.$/, "") : ""}`
    : rdk?.connected ? `missing ${rdk.missing.join(", ")}` : "not connected";
```
and replace the `<div className="pills">…</div>` block with
```tsx
        <div className="pills">
          <StatusPill label="robodk" ok={ready ? true : rdk?.connected ? null : health?.robodk.ok}
                      detail={health?.robodk.detail} summary={rdkSummary} />
          <StatusPill label="camera" ok={health?.camera.ok} detail={health?.camera.detail}
            summary={health?.camera
              ? `${health.camera.route} · ${health.camera.endpoint}`
              : "checking…"} />
          <StatusPill label="link" ok={eventsConnected} detail="job event stream" />
          <button className={ready ? "secondary" : ""} onClick={connect}
                  disabled={connecting || busy}
                  title={busy ? "A job or camera preview is running — Connect is locked until it ends"
                       : connecting ? "Opening the Tasni station…" : "Open / attach the Tasni station"}>
            {connecting ? "Connecting…" : ready ? "Reconnect" : "Connect"}
          </button>
          {connectError && <span className="topbar-toast" role="alert">{connectError}</span>}
        </div>
```
`pages/Home.tsx`: replace `import { useHealth } from "../api/useHealth";` with `import { usePlatform } from "../platform/PlatformProvider";` and `const health = useHealth();` with `const { health } = usePlatform();`.

`index.css` — after the `.pills` rule add:
```css
.pills button { padding: 6px 12px; font-size: 12px; }
.topbar-toast { max-width: 420px; font-size: 12px; color: var(--err); background: rgba(248,81,73,.10);
  border: 1px solid var(--err); border-radius: 8px; padding: 4px 10px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
```

- [ ] **Step 4: Typecheck + build**

Run: `cd tasni/webui && npm run typecheck && npm run build` → clean (pages still compile through the shim).

- [ ] **Step 5: Commit**

```bash
git add tasni/webui/src
git commit -m "feat(webui): PlatformProvider (health + rdk + connect/link + module-filtered events) and topbar Connect"
```

### Task 8b: Calibration page on the provider — no banner, id-filtered events, link + pose gate

**Files:**
- Modify: `tasni/webui/src/pages/Calibration.tsx` (imports `:1-30`, state `:92-135`, `refreshJob` `:153-158`, `hydrateConnection`/`connect`/mount effect `:171-238`, subscription `:239-272`, `beginLive`/`startLive` `:274-290`, `generateTargets` catch `:361-366`, `dryRun`/`doRun` `:370-393`, JSX banner `:442-458`, Aim card buttons `:522-540`, Run buttons `:610-620`)
- Modify: `tasni/webui/src/pages/AimHud.tsx` (`GateReading` gains `pose_live?: boolean; pose_live_required?: boolean;`)

**Interfaces:**
- Consumes: `usePlatform()`, `robotLinkNote` (8a); `/status` shape (Task 4); `{job_id}` / `{stream_id}` echoes (Task 3); 409 `pose_not_live` (Task 6).
- Produces: `export { robotLinkNote } from "../platform/PlatformProvider";` re-export (Scan imports it from `./Calibration` until 8c). No `useEvents`, no `/connect` call, no `conn` state left in the file.

- [ ] **Step 1: Imports and state**

Replace lines 1-30 (imports + the local `robotLinkNote` + `RdkStatus`) so the file starts:
```tsx
import { useCallback, useEffect, useRef, useState } from "react";
import { moduleApi, type ApiError } from "../api/client";
import { usePlatform, robotLinkNote, type JobEvent } from "../platform/PlatformProvider";
import AimHud, { type GateReading } from "./AimHud";
import CalibrationGuide from "./CalibrationGuide";
import ConeDiagram from "./ConeDiagram";
import StreamStats, { useStreamStats } from "./StreamStats";
import CollisionPanel, { type CollisionStatus } from "../components/CollisionPanel";

// Scan still imports this from here until it moves onto the provider (Task 8c).
export { robotLinkNote } from "../platform/PlatformProvider";

const api = moduleApi("calibration");
const TARGET_PREFIX = "TasniCalib_";   // must match service.py TARGET_PREFIX
```
(keep `interface CalibConfig` … `interface RunResult` … exactly as they are; delete the `RdkStatus` interface).

In the component, replace
```tsx
  const { subscribe } = useEvents();
```
with
```tsx
  const { ready, rdk, link, subscribe } = usePlatform();
```
and replace the four lines `const [conn, setConn] = …`, `const [connInfo, setConnInfo] = …`, `const ready = conn === "ready";` (keep `running`/`status`) with
```tsx
  // Cell connection is platform-owned (spec §4.5). Ownership refs: the job we
  // started/rehydrated and the live stream we started — events with other ids are
  // ignored (delayed events from a previous run, another module's job).
  const jobIdRef = useRef<string | null>(null);
  const streamIdRef = useRef<string | null>(null);
  const [foreignJob, setForeignJob] = useState<{ module: string; kind: string } | null>(null);
  const [linkNote, setLinkNote] = useState("");
```

- [ ] **Step 2: Rehydrate from the module's own `/status` (running job + solve result)**

Add above the component:
```tsx
interface JobRec { job_id: string; kind: string; status: string; result: any; error: string | null; }
interface ModuleStatus {
  running: { module: string; kind: string; job_id: string } | null;
  status: string; jobs?: Record<string, JobRec>;
}
```
Replace `refreshJob` (`:153-158`) with — it is now the single **reconciler**: it
hydrates a solve result, follows an own running job, locks on a foreign one, and
settles a job whose terminal event was missed (fast job, socket reconnect —
plan review #1/#6), then clears stale running state:
```tsx
  const refreshJob = useCallback(async () => {
    try {
      const s = await api.get<ModuleStatus>("/status");
      const res = s.jobs?.calibration?.result as RunResult | null | undefined;
      if (res?.can_apply) { setResult(res); setCanApply(true); }
      if (s.running?.module === "calibration") {
        jobIdRef.current = s.running.job_id;
        setRunning(true);
        setRunKind(s.running.kind === "sim_tour" ? "tour" : "run");
        setStatus(`${s.running.kind} running…`);
        setForeignJob(null);
        return;
      }
      setForeignJob(s.running ? { module: s.running.module, kind: s.running.kind } : null);
      const mine = jobIdRef.current
        ? Object.values(s.jobs ?? {}).find((j) => j.job_id === jobIdRef.current) : undefined;
      if (mine && mine.status !== "running") {
        if (mine.status === "done") {
          if (mine.kind === "sim_tour") { setTour(mine.result as TourResult); setStatus("dry run complete"); }
          else { setResult(mine.result as RunResult); setCanApply(!!mine.result?.can_apply); setStatus("done"); }
          setPct(100);
        } else if (mine.status === "error") {
          addLog(mine.error ?? "job failed", true); setRunError(mine.error ?? "job failed"); setStatus("error");
        } else { addLog("cancelled."); setStatus("cancelled"); }
        jobIdRef.current = null;
      }
      setRunning(false); setRunKind(null);
    } catch { /* no prior run — fine */ }
  }, []);
```
Also destructure `eventsConnected` from `usePlatform()` and add, after the mount effect:
```tsx
  // A (re)connected socket may have missed terminal events — reconcile.
  useEffect(() => { if (eventsConnected) refreshJob(); }, [eventsConnected, refreshJob]);
```
Delete `hydrateConnection` (`:171-182`) and `connect` (`:210-233`). Replace the mount effect (`:199-200`) with:
```tsx
  useEffect(() => { loadConfig(); refreshJob(); }, [loadConfig, refreshJob]);
  // Once the cell is connected (topbar), pick up targets / solved run already there.
  useEffect(() => {
    if (!ready) return;
    refreshTargets(); refreshJob();
    setLinkNote(robotLinkNote(rdk?.robot_link));
  }, [ready]);   // eslint-disable-line react-hooks/exhaustive-deps
```

- [ ] **Step 3: Module-filtered, id-checked subscription**

Replace `return subscribe((ev: JobEvent) => {` (`:240`) with
```tsx
    return subscribe("calibration", (ev: JobEvent) => {
      if (ev.job_id && ev.job_id !== jobIdRef.current) return;      // not our job
      if (ev.stream_id && ev.stream_id !== streamIdRef.current) return; // not our stream
```
and inside it `if (ev.payload.name === "sim_tour") {` → `if (ev.kind === "sim_tour") {`. Add a second effect right after it, so a foreign job's end unlocks the page:
```tsx
  useEffect(() => subscribe("*", (ev: JobEvent) => {
    if (ev.module !== "calibration"
        && (ev.type === "result" || ev.type === "error" || ev.type === "status")) refreshJob();
  }), [subscribe, refreshJob]);
```

- [ ] **Step 4: Capture ids; link when the camera starts; surface the pose gate**

`beginLive` (`:274-279`): replace `await api.post("/live/start");` with
```tsx
    const r = await api.post<{ stream_id?: string }>("/live/start");
    streamIdRef.current = r.stream_id ?? null;
```
`startLive` (`:281-290`): directly **after** `await beginLive(true);` add
```tsx
      // Aim needs the model to track the arm (spec §4.5): link the driver, best-
      // effort, not awaited, and only AFTER the camera is up — /api/rdk/link holds
      // the cell arbiter for up to robot_connect_timeout_s, and a live start during
      // that window would be refused.
      if (ready) link().then((l) => setLinkNote(robotLinkNote(l?.robot_link)));
```
`generateTargets` catch (`:361-366`): replace the two `addLog`/`setRunError` lines with
```tsx
      const detail = (e as ApiError).detail as { error?: string; message?: string } | undefined;
      const msg = detail?.error === "pose_not_live" ? detail.message! : e.message;
      addLog("create targets: " + msg, true);
      setRunError("Create targets: " + msg);
```
`dryRun` (`:373`): `await api.post("/poses/simulate");` →
```tsx
      const r = await api.post<{ job_id?: string }>("/poses/simulate");
      jobIdRef.current = r.job_id ?? null;
      refreshJob();   // events published before this response was known are reconciled from /status
```
`doRun` (`:388`): `await api.post("/run", { holdout_count: holdout, refine });` →
```tsx
      const r = await api.post<{ job_id?: string }>("/run", { holdout_count: holdout, refine });
      jobIdRef.current = r.job_id ?? null;
      refreshJob();
```
Add after `const holdoutInvalid = …` (`:427`):
```tsx
  const poseStale = gate?.pose_live_required !== false && gate?.pose_live === false;
  const locked = running || !!foreignJob;
```

- [ ] **Step 5: JSX — drop the banner, add the lock/link notes, gate the buttons**

Delete the whole `<div className={"card conn-banner " + conn}>…</div>` block (`:442-458`) and put in its place:
```tsx
      {!ready && (
        <div className="hint" style={{ marginBottom: 12 }}>
          Connect the cell (top right) to create targets and run. The board can be printed and
          the camera previewed without it.
        </div>
      )}
      {foreignJob && (
        <div className="run-error" style={{ marginBottom: 12 }}>
          <span className="run-error-tag">BUSY</span>
          <span>{foreignJob.module} job ({foreignJob.kind}) is running — calibration controls are
            locked until it ends.</span>
        </div>
      )}
```
In the Aim card, directly after the `<div className="lamps">…</div>` block add:
```tsx
        <div className="hint">
          {linkNote || "Real robot link not checked yet — Start camera links it automatically."}
          {" "}
          <button className="secondary mini" disabled={!ready || locked}
                  onClick={() => link().then((l) => setLinkNote(robotLinkNote(l?.robot_link)))}>
            Link real robot
          </button>
        </div>
        {poseStale && (
          <div className="warn-text" style={{ marginTop: 6, fontSize: 12 }}>
            Real robot not linked — the model pose may be stale. Link the real robot to create targets.
          </div>
        )}
```
Buttons: Create targets `disabled={!ready || locked || generating || !gate?.ok || poseStale}`; Dry run `disabled={locked || !ready || targets == null}`; Run `disabled={locked || !ready || targets == null || !scaleOk || holdoutInvalid}`; Clear targets `disabled={locked}`; Cancel stays `disabled={!running}`.

`AimHud.tsx`: in `export interface GateReading` add `pose_live?: boolean; pose_live_required?: boolean;`.

- [ ] **Step 6: Typecheck + build, then commit**

Run: `cd tasni/webui && grep -n "useEvents\|/connect\"\|setConn(" src/pages/Calibration.tsx` → nothing; `npm run typecheck && npm run build` → clean.
```bash
git add tasni/webui/src/pages/Calibration.tsx tasni/webui/src/pages/AimHud.tsx
git commit -m "feat(webui/calibration): platform connection, id-filtered events, auto-link on Aim, pose gate"
```

### Task 8c: Scan page on the provider

**Files:**
- Modify: `tasni/webui/src/pages/Scan.tsx` (imports `:1-10`, state `:204-212`, `refreshJob` `:296-303`, `hydrateConnection` `:304-313`, mount/auto-connect effects `:363-410`, subscription `:412-490`, `beginLive`/`startLive` `:558-578`, `generateTargets` catch `:709-724`, `lockSurface` catch `:791-810`, `dryRun`/`doRun` `:914-933`, JSX banner `:1009-1025`, Survey-card buttons `:1174-1204`, Run buttons `:1272-1278`)

**Interfaces:** same as 8b. Scan keeps its today's behaviour of auto-connecting on entry — it now calls the platform `connect()` once when the backend reports no session.

- [ ] **Step 1: Imports and state**

Replace `import { apiGet, moduleApi, type ApiError } from "../api/client";` (keep — `apiGet` is still used by `hydrateCalibration`), replace `import { useEvents, type JobEvent } from "../api/events";` with `import { usePlatform, robotLinkNote, type JobEvent } from "../platform/PlatformProvider";`, delete `import { robotLinkNote } from "./Calibration";`, delete the local `RdkStatus` interface.
Replace `const { subscribe } = useEvents();` with `const { ready, rdk, connect, link, subscribe } = usePlatform();`; delete `const [conn, setConn] = …`, `const [connInfo, setConnInfo] = …`, `const ready = conn === "ready";`; add
```tsx
  const jobIdRef = useRef<string | null>(null);
  const streamIdRef = useRef<string | null>(null);
  const [foreignJob, setForeignJob] = useState<{ module: string; kind: string } | null>(null);
  const [linkNote, setLinkNote] = useState("");
```

- [ ] **Step 2: Rehydrate + connection effects**

Add the same `JobRec` / `ModuleStatus` interfaces as 8b above the component. Replace `refreshJob` (`:296-303`) with the reconciler:
```tsx
  const refreshJob = useCallback(async () => {
    try {
      const s = await api.get<ModuleStatus>("/status");
      const res = s.jobs?.scan?.result as ScanResult | null | undefined;
      if (res?.can_insert) { setResult(res); setViewerNonce((n) => n + 1); }
      if (s.running?.module === "scan") {
        jobIdRef.current = s.running.job_id;
        setRunning(true); setRunKind(s.running.kind === "sim_tour" ? "tour" : "run");
        setStatus(`${s.running.kind} running…`); setForeignJob(null);
        return;
      }
      setForeignJob(s.running ? { module: s.running.module, kind: s.running.kind } : null);
      const mine = jobIdRef.current
        ? Object.values(s.jobs ?? {}).find((j) => j.job_id === jobIdRef.current) : undefined;
      if (mine && mine.status !== "running") {
        if (mine.status === "done") {
          if (mine.kind === "sim_tour") { setTour(mine.result as TourResult); setStatus("dry run complete"); }
          else { setResult(mine.result as ScanResult); setViewerNonce((n) => n + 1); setInserted(false); setStatus("done"); }
          setPct(100);
        } else if (mine.status === "error") {
          addLog(mine.error ?? "job failed", true); setRunError(mine.error ?? "job failed"); setStatus("error");
        } else { addLog("cancelled."); setStatus("cancelled"); }
        jobIdRef.current = null;
      }
      setRunning(false); setRunKind(null);
    } catch { /* no prior scan */ }
  }, []);
```
Destructure `eventsConnected` from `usePlatform()` too, and add after the mount effect:
```tsx
  useEffect(() => { if (eventsConnected) refreshJob(); }, [eventsConnected, refreshJob]);
```
Delete `hydrateConnection` (`:304-313`) and `connect` (`:389-404`). Replace the mount effect (`:363-365`) with
```tsx
  useEffect(() => { loadConfig(); refreshJob(); hydrateSurvey(); hydrateCalibration(); },
            [loadConfig, refreshJob, hydrateSurvey, hydrateCalibration]);
  useEffect(() => {
    if (!ready) return;
    refreshTargets(); refreshJob();
    setLinkNote(robotLinkNote(rdk?.robot_link));
  }, [ready]);   // eslint-disable-line react-hooks/exhaustive-deps
```
and **delete** the auto-connect effect (`:406-410`) together with `autoConnectRef`
(`:220`) and drop `connect` from the `usePlatform()` destructuring: connecting the
cell is an explicit topbar action (spec decision 13 — a page silently opening the
117 MB station on navigation undermines the one-Connect model). The camera
auto-preview (`autoPreviewRef`, `:579-584`) stays — it is camera-only and needs no
station.

- [ ] **Step 3: Subscription, ids, link, gate errors**

`:413`: `return subscribe((ev: JobEvent) => {` →
```tsx
    return subscribe("scan", (ev: JobEvent) => {
      if (ev.job_id && ev.job_id !== jobIdRef.current) return;
      if (ev.stream_id && ev.stream_id !== streamIdRef.current) return;
```
`:475`: `if (ev.payload.name === "sim_tour") {` → `if (ev.kind === "sim_tour") {`. Add after the effect:
```tsx
  useEffect(() => subscribe("*", (ev: JobEvent) => {
    if (ev.module !== "scan"
        && (ev.type === "result" || ev.type === "error" || ev.type === "status")) refreshJob();
  }), [subscribe, refreshJob]);
```
`beginLive` (`:570`): `await api.post("/live/start");` →
```tsx
    const r = await api.post<{ stream_id?: string }>("/live/start");
    streamIdRef.current = r.stream_id ?? null;
```
`startLive` (`:573-578`): directly **after** `await beginLive(true);` (camera first — the link holds the arbiter for up to 10 s):
```tsx
      if (ready) link().then((l) => setLinkNote(robotLinkNote(l?.robot_link)));
```
`lockSurface` catch (`:791-810`) — add a branch before the generic `else`:
```tsx
      } else if (detail?.error === "pose_not_live") {
        addLog("lock surface: " + (detail as any).message, true);
        setRunError("Lock surface: " + (detail as any).message);
      } else {
```
`generateTargets` catch (`:713-722`): same three-line branch keyed on `"create targets: "`.
`dryRun` (`:917`): `try { await api.post("/poses/simulate"); }` →
```tsx
    try {
      const r = await api.post<{ job_id?: string }>("/poses/simulate");
      jobIdRef.current = r.job_id ?? null;
      refreshJob();
    }
```
`doRun` (`:928`): `try { await api.post("/run"); }` → same with `"/run"`.
After `const canLockSurface = …` (`:967`) add:
```tsx
  const poseStale = gate?.pose_live_required !== false && gate?.pose_live === false;
  const locked = running || !!foreignJob;
```

- [ ] **Step 4: JSX**

Delete the `conn-banner` card (`:1009-1025`); in its place:
```tsx
      {!ready && (
        <div className="hint" style={{ marginBottom: 12 }}>
          Connect the cell (top right) to lock the surface and create targets; the surface
          feed works without it.
        </div>
      )}
      {foreignJob && (
        <div className="run-error" style={{ marginBottom: 12 }}>
          <span className="run-error-tag">BUSY</span>
          <span>{foreignJob.module} job ({foreignJob.kind}) is running — scan controls are locked until it ends.</span>
        </div>
      )}
      <div className="hint">The scan uses the stored camera calibration; it never runs one.
        If none is on file it warns and proceeds (the mesh/frame may be less accurate).</div>
```
In the Survey card after the `<div className="lamps">…</div>` block add the same link note/button + `poseStale` warning as 8b (text: "…Link the real robot to lock the surface."). Buttons: Lock `disabled={!ready || locked || locking || generating || !live || !canLockSurface || poseStale}`; Accept region & create targets `disabled={!ready || locked || generating}`; Reposition `disabled={locked || generating || preparing}`; Clear targets `disabled={locked}`; Dry run / Run scan `disabled={locked || !ready || targets == null}`; `beginSurvey` buttons add `|| locked`.

- [ ] **Step 5: Typecheck + build, commit**

Run: `cd tasni/webui && grep -n "useEvents\|from \"./Calibration\"\|setConn(" src/pages/Scan.tsx` → nothing; `npm run typecheck && npm run build` → clean. Then remove the re-export line added in 8b from `Calibration.tsx`.
```bash
git add tasni/webui/src/pages/Scan.tsx tasni/webui/src/pages/Calibration.tsx
git commit -m "feat(webui/scan): platform connection, id-filtered events, auto-link on Survey, pose gate"
```

### Task 8d: Extrusion page on the provider; delete the events shim

**Files:**
- Modify: `tasni/webui/src/pages/Extrusion.tsx` (imports `:1-4`, state `:209`, `refreshStatus` `:235-240`, subscription `:255-284`, `connect` `:363-370`, `startJob` `:400-419`, `characterize` `:440-447`, `measure` `:458-471`, JSX banner `:570-574`)
- Delete: `tasni/webui/src/api/events.tsx`

- [ ] **Step 1: Imports, state, hydration**

`import { useEvents, type JobEvent } from "../api/events";` → `import { usePlatform, type JobEvent } from "../platform/PlatformProvider";`. `const { subscribe } = useEvents();` → `const { ready, connect: connectCell, subscribe } = usePlatform();` and add `const jobIdRef = useRef<string | null>(null);`. Keep `const [connected, setConnected] = useState(false);` (= station items discovered).

`refreshStatus` (`:235-240`) — also the reconciler for a missed terminal event:
```tsx
  const refreshStatus = useCallback(() => {
    api.get<Status>("/status").then((value) => {
      setStatus(value);
      if (value.result?.kind?.startsWith("cylinder_")) setResult(value.result);
      const mine = jobIdRef.current
        ? Object.values(value.jobs ?? {}).find((j) => j.job_id === jobIdRef.current) : undefined;
      if (value.running?.module === "extrusion") {
        jobIdRef.current = value.running.job_id; setBusy(true);
      } else if (mine && mine.status !== "running") {
        // The job we started has finished but we never saw its terminal event.
        if (mine.status === "done" && mine.result?.kind?.startsWith("cylinder_")) setResult(mine.result);
        if (mine.status === "error") setMessage(mine.error ?? "job failed");
        setBusy(false); setCancelling(false); jobIdRef.current = null;
      } else if (value.running) {
        setMessage(`${value.running.module} job (${value.running.kind}) is running — extrusion is locked until it ends.`);
      }
    }).catch(() => {});
  }, []);
```
Destructure `eventsConnected` from `usePlatform()` and add:
```tsx
  useEffect(() => { if (eventsConnected) refreshStatus(); }, [eventsConnected, refreshStatus]);
```
(`startJob`, `characterize` and `measure` already call `refreshStatus()` right after
the start response — keep that; it is the post-start reconcile.)
Add after the mount effect (`:244-252`):
```tsx
  // Station items follow the platform connection (topbar Connect).
  useEffect(() => {
    if (!ready || options) return;
    api.get<StationOptions>("/station-options").then((d) => {
      setOptions(d); setConnected(true); refreshSurface();
    }).catch(() => {});
  }, [ready, options, refreshSurface]);
```

- [ ] **Step 2: Subscription by id**

Replace the subscription effect (`:255-284`) head with
```tsx
  useEffect(() => subscribe("extrusion", (event: JobEvent) => {
    if (event.job_id !== jobIdRef.current) return;     // ours only (spec §4.5)
    const name = event.kind ?? undefined;
    if (event.type === "progress") {
      setProgress(event.payload); setMessage(event.payload.message || "Working…");
    } else if (event.type === "log") {
      setLogs((old) => [...old, event.payload.message]);
    } else if (event.type === "result") {
```
and change the remaining `else if (event.type === "error" && name?.startsWith("extrusion-"))` / `"status" && name?.startsWith(…)` to `else if (event.type === "error")` / `else if (event.type === "status")`. Keep the inner bodies (they use `name` for the message text).

- [ ] **Step 3: Capture `job_id`; platform connect**

`startJob` (`:405`): `await api.post(`/${kind}`, …);` → `const r = await api.post<{ job_id?: string }>(`/${kind}`, …); jobIdRef.current = r.job_id ?? null;`.
`characterize` (`:443`) and `measure` (`:462`): same pattern (`const r = await api.post<{ job_id?: string }>(…); jobIdRef.current = r.job_id ?? null;`).
`connect` (`:363-370`):
```tsx
  const connect = async () => {
    setBusy(true); setMessage("Loading RoboDK station…");
    try {
      await connectCell();
      const discovered = await api.get<StationOptions>("/station-options");
      setOptions(discovered); setConnected(true); refreshSurface();
      setMessage("Station loaded. Select print/inspection tools, work frame, and inspection target.");
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
```
JSX (`:570-574`): replace the `conn-banner` card with
```tsx
    {!connected && (
      <div className="hint" style={{ marginBottom: 12 }}>
        {ready ? "Discovering station items…" : "Connect the cell (top right) to pick station items."}
        {" "}<button className="secondary mini" disabled={busy || jobRunning} onClick={connect}>
          {ready ? "Refresh items" : "Connect"}</button>
      </div>
    )}
```

- [ ] **Step 4: Delete the shim, verify, commit**

```bash
git rm tasni/webui/src/api/events.tsx
cd tasni/webui && grep -rn "useEvents\|api/events\|useHealth" src || echo "clean"
npm run typecheck && npm run build
git add -A tasni/webui/src
git commit -m "feat(webui/extrusion): platform connection + id-filtered events; remove events shim"
git push origin ux-phase0
```

### Task 9: Frontend test runner + integration tests (fake WebSocket / fetch)

**Files:**
- Modify: `tasni/webui/package.json` (dev deps + `test` script), `tasni/webui/tsconfig.json` (`"types": ["vitest/globals"]`)
- Create: `tasni/webui/vitest.config.ts`, `tasni/webui/src/test/setup.ts`, `tasni/webui/src/test/fakes.ts`, `tasni/webui/src/test/platform.test.tsx`, `tasni/webui/src/test/calibration-page.test.tsx`

**Interfaces:**
- Produces: `npm test` (vitest, jsdom). `installFakes({ routes })` from `fakes.ts` replaces `fetch` + `WebSocket`; `FakeWebSocket.last().emit(event)` pushes a `JobEvent` into the app; `fetchCalls()` lists requests.

- [ ] **Step 1: Tooling**

`package.json` — add to `scripts`: `"test": "vitest run"`, `"test:watch": "vitest"`; add to `devDependencies`:
```json
    "@testing-library/dom": "^10.4.0",
    "@testing-library/jest-dom": "^6.6.3",
    "@testing-library/react": "^16.1.0",
    "jsdom": "^25.0.1",
    "vitest": "^2.1.9"
```
then `cd tasni/webui && npm install`. `tsconfig.json`: add `"types": ["vitest/globals", "@testing-library/jest-dom"]` inside `compilerOptions`.

`vitest.config.ts`:
```ts
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: { environment: "jsdom", globals: true, setupFiles: ["src/test/setup.ts"] },
});
```
`src/test/setup.ts` (jest-dom supplies `toBeEnabled` / `toBeDisabled`):
```ts
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";
afterEach(() => cleanup());
```

- [ ] **Step 2: Fakes**

`src/test/fakes.ts`:
```ts
// Test doubles for the two things the platform talks to: fetch() and the /ws socket.
import { vi } from "vitest";
import type { JobEvent } from "../platform/PlatformProvider";

type Route = (init?: RequestInit) => unknown;
const calls: Array<{ url: string; method: string }> = [];

export class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  static last() { return FakeWebSocket.instances[FakeWebSocket.instances.length - 1]; }
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((m: { data: string }) => void) | null = null;
  constructor(public url: string) { FakeWebSocket.instances.push(this); }
  open() { this.onopen?.(); }
  emit(ev: Partial<JobEvent> & { type: string }) {
    const full: JobEvent = { payload: {}, module: null, job_id: null, kind: null,
                             stream_id: null, ...ev };
    this.onmessage?.({ data: JSON.stringify(full) });
  }
  close() { this.onclose?.(); }
}

export function installFakes(routes: Record<string, Route>) {
  calls.length = 0;
  FakeWebSocket.instances = [];
  (globalThis as any).WebSocket = FakeWebSocket;
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : (input as Request).url ?? String(input);
    const path = url.replace(/^https?:\/\/[^/]+/, "");
    calls.push({ url: path, method: init?.method ?? "GET" });
    const key = `${init?.method ?? "GET"} ${path}`;
    const route = routes[key] ?? routes[path];
    if (!route) return new Response(JSON.stringify({ detail: `no fake route for ${key}` }), { status: 404 });
    const out = await route(init);          // a route may return a Promise (delayed responses)
    if (out instanceof Response) return out;
    return new Response(JSON.stringify(out), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  (globalThis as any).fetch = fetchMock;
  return { fetchMock, routes };
}

export const fetchCalls = () => calls;
export const countCalls = (path: string) => calls.filter((c) => c.url.startsWith(path)).length;

export const okHealth = () => ({
  robodk: { ok: true, detail: "API :20500" },
  camera: { ok: true, state: "connected", detail: "", route: "Direct/LAN", endpoint: "10.12.171.70:1024" },
  job: { status: "idle", running: false },
});
export const readyRdk = () => ({
  connected: true, ready: true, robot: "KUKA KR150 R2700", tool: "Realsense", missing: [],
  robot_valid: true, tool_present: true,
  robot_link: { connected: false, message: "not connected", ip: "10.0.0.5", configured: true },
});
```

- [ ] **Step 3: Provider tests**

`src/test/platform.test.tsx`:
```tsx
import { useEffect, useState } from "react";
import { act, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PlatformProvider, usePlatform } from "../platform/PlatformProvider";
import { FakeWebSocket, countCalls, installFakes, okHealth, readyRdk } from "./fakes";

function Probe({ module }: { module: string }) {
  const { ready, subscribe } = usePlatform();
  const [seen, setSeen] = useState<string[]>([]);
  useEffect(() => subscribe(module, (e) => setSeen((s) => [...s, `${e.module}:${e.type}`])),
            [subscribe, module]);
  return <div><span data-testid="ready">{String(ready)}</span><span data-testid="seen">{seen.join(",")}</span></div>;
}

describe("PlatformProvider", () => {
  it("reports ready from /api/rdk/status and only delivers the module's own events", async () => {
    installFakes({ "/api/health": okHealth, "/api/rdk/status": readyRdk });
    render(<PlatformProvider><Probe module="scan" /></PlatformProvider>);
    await waitFor(() => expect(screen.getByTestId("ready").textContent).toBe("true"));
    act(() => { FakeWebSocket.last().open(); });
    act(() => {
      FakeWebSocket.last().emit({ type: "progress", module: "calibration", job_id: "a", kind: "calibration" });
      FakeWebSocket.last().emit({ type: "progress", module: "scan", job_id: "b", kind: "scan" });
    });
    expect(screen.getByTestId("seen").textContent).toBe("scan:progress");
  });

  it("drops readiness the moment health says RoboDK is down", async () => {
    let robodkOk = true;
    installFakes({ "/api/health": () => ({ ...okHealth(), robodk: { ok: robodkOk, detail: "" } }),
                   "/api/rdk/status": readyRdk });
    vi.useFakeTimers({ shouldAdvanceTime: true });
    render(<PlatformProvider><Probe module="scan" /></PlatformProvider>);
    await waitFor(() => expect(screen.getByTestId("ready").textContent).toBe("true"));
    robodkOk = false;
    await act(async () => { await vi.advanceTimersByTimeAsync(4100); });   // one health tick, no rdk poll
    expect(screen.getByTestId("ready").textContent).toBe("false");
    vi.useRealTimers();
  });

  it("refreshes rdk status on a terminal job event and not on progress", async () => {
    installFakes({ "/api/health": okHealth, "/api/rdk/status": readyRdk });
    render(<PlatformProvider><Probe module="scan" /></PlatformProvider>);
    await waitFor(() => expect(countCalls("/api/rdk/status")).toBe(1));
    act(() => { FakeWebSocket.last().open(); });
    act(() => { FakeWebSocket.last().emit({ type: "progress", module: "scan", job_id: "b", kind: "scan" }); });
    expect(countCalls("/api/rdk/status")).toBe(1);
    act(() => { FakeWebSocket.last().emit({ type: "result", module: "scan", job_id: "b", kind: "scan", payload: { name: "scan" } }); });
    await waitFor(() => expect(countCalls("/api/rdk/status")).toBe(2));
  });

  it("stays not-ready while health says RoboDK is down even if /rdk/status is stale-green", async () => {
    installFakes({ "/api/health": () => ({ ...okHealth(), robodk: { ok: false, detail: "" } }),
                   "/api/rdk/status": readyRdk });
    render(<PlatformProvider><Probe module="scan" /></PlatformProvider>);
    await waitFor(() => expect(countCalls("/api/rdk/status")).toBe(1));
    await waitFor(() => expect(countCalls("/api/health")).toBe(1));
    expect(screen.getByTestId("ready").textContent).toBe("false");
  });

  it("connect() posts /api/rdk/connect and never /api/rdk/link", async () => {
    installFakes({ "/api/health": okHealth, "/api/rdk/status": () => ({ ...readyRdk(), connected: false, ready: false }),
                   "POST /api/rdk/connect": readyRdk });
    function Btn() { const { connect, ready } = usePlatform(); return <button onClick={connect}>{String(ready)}</button>; }
    render(<PlatformProvider><Btn /></PlatformProvider>);
    await waitFor(() => expect(countCalls("/api/rdk/status")).toBe(1));
    act(() => { screen.getByRole("button").click(); });
    await waitFor(() => expect(screen.getByRole("button").textContent).toBe("true"));
    expect(countCalls("/api/rdk/connect")).toBe(1);
    expect(countCalls("/api/rdk/link")).toBe(0);
  });
});
```

- [ ] **Step 4: Calibration page integration test**

`src/test/calibration-page.test.tsx`:
```tsx
import { act, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { PlatformProvider } from "../platform/PlatformProvider";
import Calibration from "../pages/Calibration";
import { FakeWebSocket, installFakes, okHealth, readyRdk } from "./fakes";

const config = () => ({
  robot: "KUKA KR150 R2700", camera_tool: "Realsense",
  board: { squares_x: 8, squares_y: 6, square_size_mm: 30, marker_size_mm: 22, dictionary: "DICT_4X4_50", paper_size: "A4" },
  camera: { ip: "10.12.171.70", port: 1024, resolution: "1280x720" },
  calibration: { holdout_count: 3, refine: true, pose_count: 15, cone_half_angle_deg: 25, roll_max_deg: 30,
                 distance_jitter: 0.1, jog_invert_x: false, jog_invert_y: false, jog_invert_z: false, seed_stable_s: 1 },
  gate: { ideal_distance_mm: 450, distance_tol_mm: 80, max_tilt_deg: 10, center_tol_mm: 40,
          min_board_area_frac: 0.1, max_board_area_frac: 0.4, stable_s: 1 },
});
const base = (status: unknown) => ({
  "/api/health": okHealth, "/api/rdk/status": readyRdk,
  "/api/modules/calibration/config": config,
  "/api/modules/calibration/status": () => status,
  "/api/modules/calibration/targets": () => ({ targets: [] }),
  "/api/modules/calibration/board/spec?page=A4": () => ({ dictionary: "DICT_4X4_50", squares_x: 8, squares_y: 6,
    square_size_mm: 30, marker_size_mm: 22, board_w_mm: 240, board_h_mm: 180, page: "A4", landscape: true, fits: true, pages: ["A4", "A3", "Letter"] }),
  "/api/modules/calibration/collision/status": () => ({ available: false }),
  "POST /api/modules/calibration/live/stop": () => ({ status: "stopped" }),
});
const mount = () => render(<MemoryRouter><PlatformProvider><Calibration /></PlatformProvider></MemoryRouter>);

describe("Calibration page (phase 0)", () => {
  it("rehydrates its own running job and ignores events with another job_id", async () => {
    installFakes(base({ running: { module: "calibration", kind: "calibration", job_id: "j9" },
                        status: "running", jobs: {} }));
    mount();
    await waitFor(() => expect(screen.getByRole("button", { name: "Cancel" })).toBeEnabled());
    act(() => { FakeWebSocket.last().open(); });
    act(() => {
      FakeWebSocket.last().emit({ type: "progress", module: "calibration", job_id: "old", kind: "calibration",
                                  payload: { step: 1, total: 15, message: "STALE" } });
      FakeWebSocket.last().emit({ type: "progress", module: "calibration", job_id: "j9", kind: "calibration",
                                  payload: { step: 7, total: 15, message: "pose 7" } });
    });
    expect(screen.queryByText(/STALE/)).toBeNull();
    expect(screen.getByText(/7\/15\s+pose 7/)).toBeTruthy();
  });

  it("locks its controls while another module's job runs", async () => {
    installFakes(base({ running: { module: "scan", kind: "scan", job_id: "s1" }, status: "busy", jobs: {} }));
    mount();
    await waitFor(() => expect(screen.getByText(/scan job \(scan\) is running/)).toBeTruthy());
    expect(screen.getByRole("button", { name: /Create targets/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Run calibration/ })).toBeDisabled();
  });

  it("shows no per-page Connect button", async () => {
    installFakes(base({ running: null, status: "idle", jobs: {} }));
    mount();
    await waitFor(() => expect(screen.getByRole("button", { name: /Start camera/ })).toBeTruthy());
    expect(screen.queryByRole("button", { name: /Connect & open Tasni station/ })).toBeNull();
  });

  it("settles a job whose result arrived before the start response (fast job)", async () => {
    // The tour finishes before POST /poses/simulate returns: its result event is
    // emitted while jobIdRef is still null (dropped), and /status already holds the
    // finished record. The post-start reconcile must show "Dry run passed".
    let release!: () => void;
    const started = new Promise<void>((r) => { release = r; });
    let statusBody: unknown = { running: null, status: "idle", jobs: {} };
    const routes = base(undefined as unknown);
    routes["/api/modules/calibration/status"] = () => statusBody;
    routes["/api/modules/calibration/targets"] = () => ({ targets: ["TasniCalib_1", "TasniCalib_2", "TasniCalib_3", "TasniCalib_4"] });
    routes["POST /api/modules/calibration/poses/simulate"] = () => started.then(() => ({ status: "started", job_id: "t1" }));
    installFakes(routes);
    mount();
    await waitFor(() => expect(screen.getByRole("button", { name: /Dry run/ })).toBeEnabled());
    act(() => { FakeWebSocket.last().open(); });
    act(() => { screen.getByRole("button", { name: /Dry run/ }).click(); });
    const tour = { all_ok: true, passed: 4, total: 4, returned_to_start: true, poses: [] };
    act(() => { FakeWebSocket.last().emit({ type: "result", module: "calibration", job_id: "t1", kind: "sim_tour",
                                            payload: { name: "sim_tour", result: tour } }); });
    statusBody = { running: null, status: "done",
                   jobs: { sim_tour: { job_id: "t1", kind: "sim_tour", status: "done", result: tour, error: null } } };
    act(() => { release(); });
    await waitFor(() => expect(screen.getByText(/Dry run passed/)).toBeTruthy());
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
  });
});
```
(If `CalibrationGuide` fetches `board.png`, jsdom simply leaves the `<img>` unloaded — no route needed.)

- [ ] **Step 4b: Scan and Extrusion page filtering tests**

`src/test/scan-page.test.tsx`:
```tsx
import { act, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { PlatformProvider } from "../platform/PlatformProvider";
import Scan from "../pages/Scan";
import { FakeWebSocket, countCalls, installFakes, okHealth, readyRdk } from "./fakes";

const routes = (status: unknown) => ({
  "/api/health": okHealth, "/api/rdk/status": readyRdk, "POST /api/rdk/link": readyRdk,
  "/api/modules/scan/config": () => ({
    robot: "KUKA KR150 R2700", camera: { ip: "10.12.171.70", port: 1024, resolution: "1280x720" },
    scan: { voxel_size_m: 0.003, pose_count: 14, cone_half_angle_deg: 20 },
    gate: { ideal_distance_mm: 500, distance_tol_mm: 50, max_tilt_deg: 10 } }),
  "/api/modules/scan/status": () => status,
  "/api/modules/scan/targets": () => ({ targets: [] }),
  "/api/modules/scan/survey/state": () => ({ step: null }),
  "/api/runs/active?module=calibration": () => ({ active: null }),
  "/api/modules/scan/collision/status": () => ({ available: false }),
  "POST /api/modules/scan/live/start": () => ({ status: "started", stream_id: "s1" }),
  "POST /api/modules/scan/live/stop": () => ({ status: "stopped" }),
});
const mount = () => render(<MemoryRouter><PlatformProvider><Scan /></PlatformProvider></MemoryRouter>);

describe("Scan page (phase 0)", () => {
  it("never auto-connects; starts its camera when ready; keeps only its own stream", async () => {
    installFakes(routes({ running: null, status: "idle", jobs: {}, locked: false, prepared: false }));
    mount();
    await waitFor(() => expect(countCalls("/api/modules/scan/live/start")).toBe(1));
    expect(countCalls("/api/rdk/connect")).toBe(0);
    act(() => { FakeWebSocket.last().open(); });
    act(() => {
      FakeWebSocket.last().emit({ type: "log", module: "scan", stream_id: "other", payload: { message: "STALE-STREAM" } });
      FakeWebSocket.last().emit({ type: "log", module: "calibration", payload: { message: "OTHER-MODULE" } });
      FakeWebSocket.last().emit({ type: "log", module: "scan", payload: { message: "MINE" } });
    });
    expect(screen.queryByText(/STALE-STREAM/)).toBeNull();
    expect(screen.queryByText(/OTHER-MODULE/)).toBeNull();
    expect(screen.getByText(/MINE/)).toBeTruthy();
  });

  it("locks its controls while another module's job runs", async () => {
    installFakes(routes({ running: { module: "extrusion", kind: "extrusion-print", job_id: "p1" },
                          status: "busy", jobs: {}, locked: false, prepared: false }));
    mount();
    await waitFor(() => expect(screen.getByText(/extrusion job \(extrusion-print\) is running/)).toBeTruthy());
    expect(screen.getByRole("button", { name: /Lock & create targets/ })).toBeDisabled();
  });
});
```

`src/test/extrusion-page.test.tsx`:
```tsx
import { act, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { PlatformProvider } from "../platform/PlatformProvider";
import Extrusion from "../pages/Extrusion";
import { FakeWebSocket, installFakes, okHealth, readyRdk } from "./fakes";

const recipe = { radius_mm: 40, layer_count: 3, layer_height_mm: 5, bead_diameter_mm: 6,
  robot_speed_mm_s: 75, travel_speed_mm_s: 150, path_rounding_mm: 2, extrusion_rate_pct: 30,
  points_per_circle: 72, correction_enabled: false, material: [] };
const setup = { print_tool: "", work_frame: "", inspection_tool: "", inspection_target: "",
  inspection_auto: true, center_x_mm: 0, center_y_mm: 0, build_plane_z_mm: 0, scan_run_id: null,
  orientation_rpy_deg: [180, 0, 90], maximum_tool_axis_spin_deg: 90,
  approach_clearance_mm: 30, retreat_clearance_mm: 50 };
const routes = (status: unknown) => ({
  "/api/health": okHealth, "/api/rdk/status": readyRdk,
  "/api/modules/extrusion/config": () => ({ defaults: recipe, setup_defaults: setup,
    integration: { valve_outputs: ["IO_508", "IO_601"], air_on_program: "AirOn", air_off_program: "AirOff" } }),
  "/api/modules/extrusion/plan": () => new Response(JSON.stringify({ detail: "no plan" }), { status: 404 }),
  "/api/modules/extrusion/status": () => status,
  "/api/modules/extrusion/scan-surface": () => ({ applied: false, available: false, note: "" }),
  "/api/modules/extrusion/measure/session": () => ({ session: null }),
  "/api/modules/extrusion/station-options": () => ({ tools: ["Nozzle"], frames: ["World"], targets: [], programs: [] }),
});
const mount = () => render(<MemoryRouter><PlatformProvider><Extrusion /></PlatformProvider></MemoryRouter>);

describe("Extrusion page (phase 0)", () => {
  it("rehydrates its running job and ignores other job ids", async () => {
    const base = { fingerprint: "abc", geometry_preflight_passed: false, quick_sim_passed: false,
      quick_sim_layers: [], quick_sim_live_approved: false, dry_run_passed: false,
      hardware_io_test_approved: false, live_print_enabled: false, jobs: {} };
    installFakes(routes({ ...base, running: { module: "extrusion", kind: "extrusion-print", job_id: "e1" }, status: "running" }));
    mount();
    await waitFor(() => expect(screen.getByRole("button", { name: /Cancel safely/ })).toBeTruthy());
    act(() => { FakeWebSocket.last().open(); });
    act(() => {
      FakeWebSocket.last().emit({ type: "progress", module: "extrusion", job_id: "e0", kind: "extrusion-print",
                                  payload: { step: 1, total: 3, message: "STALE" } });
      FakeWebSocket.last().emit({ type: "progress", module: "extrusion", job_id: "e1", kind: "extrusion-print",
                                  payload: { step: 2, total: 3, message: "layer 2" } });
    });
    expect(screen.queryByText(/STALE/)).toBeNull();
    expect(screen.getByText(/layer 2/)).toBeTruthy();
  });
});
```

- [ ] **Step 5: Run, then commit**

Run: `cd tasni/webui && npm test` → all green; `npm run typecheck && npm run build` → clean.
```bash
git add tasni/webui/package.json tasni/webui/package-lock.json tasni/webui/tsconfig.json tasni/webui/vitest.config.ts tasni/webui/src/test
git commit -m "test(webui): vitest + integration tests for the provider and the calibration, scan, extrusion pages"
```

### Task 10: Docs, cell validation checklist, merge

**Files:**
- Modify: `tasni/README.md` (Run section + Architecture list), `CLAUDE.md` (roadmap bullet), `docs/superpowers/specs/2026-08-28-ux-workflow-shell-design.md` (status line)

- [ ] **Step 1: README + CLAUDE.md**

In `tasni/README.md` "Run" paragraph replace the sentence beginning "The guide's **Open Tasni station** button loads `Tasni.rdk`…" with:
> The topbar **Connect** button opens `Tasni.rdk` (`POST /api/rdk/connect`, station-only, refused while a job runs); every module reads that one connection. Entering Aim/Survey links the real-robot driver (`POST /api/rdk/link`) so the RoboDK model tracks the arm — **Create targets** and **Lock surface** refuse (409 `pose_not_live`) until it does, because they seed from the model pose. `GET /api/readiness` tells the Dashboard what is recorded vs actually present in the open station.

Under "Architecture" add to the `core/` list: `jobrunner, events  … every event carries module + job_id + kind (live previews: stream_id); outcomes kept per (module, kind)`.

In `CLAUDE.md` roadmap add:
> - ✅ **UX overhaul phase 0 — platform foundation** (branch `ux-overhaul`): module-scoped job events + per-(module, kind) history, station-only topbar Connect, explicit real-robot link with a server-side live-pose gate, `/api/readiness`, `PlatformProvider`; vitest added. Spec: `docs/superpowers/specs/2026-08-28-ux-workflow-shell-design.md`. Next: phase 1 (workflow shell + Calibration + Dashboard strip).

Spec status line → `Status: **approved 2026-08-28 · phase 0 implemented on ux-phase0 — awaiting cell validation (Appendix B phase 0)**`. Only after the operator's checklist passes does a follow-up commit change it to `phase 0 cell-validated <date>`; never before.

- [ ] **Step 2: Backend regression sweep (targeted, never the full suite)**

Run: `py -3.10 -m pytest tests/test_jobrunner_scope.py tests/test_livepreview.py tests/test_event_scoping_lint.py tests/test_module_status.py tests/test_platform_connect.py tests/test_live_pose_gate.py tests/test_readiness.py tests/test_robot_link.py tests/test_calibration_job.py tests/test_scan_job.py tests/test_extrusion.py tests/test_extrusion_job.py tests/test_extrusion_measure.py tests/test_sim_tour.py tests/test_runs.py -q` → all PASS.

- [ ] **Step 3: Commit, push, cell validation (operator), merge**

```bash
git add tasni/README.md CLAUDE.md docs/superpowers/specs/2026-08-28-ux-workflow-shell-design.md
git commit -m "docs: phase 0 platform foundation — connect/link/readiness + event scoping"
git push origin ux-phase0
```
Restart `.\start.ps1` (the backend caches imports) and walk **Appendix B, Phase 0** of the spec on the cell: topbar Connect with RoboDK closed; module switching without re-connect; Aim with the controller OFF → link chip OFFLINE + Create targets locked with "Real robot not linked"; controller on → Link → unlocks; same for Scan Lock; Calibration dry run then switch to Scan → "calibration job (sim_tour) is running" + Connect disabled; back to Calibration → solve result + Apply still shown after a further dry run; kill RoboDK → pill red within 4 s, Reconnect works; delete the inserted frame → `GET /api/readiness` reports `present: false` (Dashboard card lands in phase 1). Record the outcome in the commit message of the merge.

Then use `superpowers:finishing-a-development-branch` to merge `ux-phase0` → `ux-overhaul` and push.

---

## Self-review (done while writing)

- **Spec coverage (§4.5, §4.7, §7, §8, §9 phase 0):** events + ids → T1–T3; per-(module, kind) history + `/status` → T1, T4; station-only connect with 409/lock → T5; link endpoint + consolidation + config doc → T5; live-pose gate + `pose_live` in the calibration gate + `require_live_pose` → T6; readiness recorded-vs-present → T7; provider refresh rules, filtered `subscribe`, topbar Connect, banners dropped, auto-link on Aim/Survey, gate reasons → T8a–d; integration tests (foreign events, rehydrate, health failure, terminal refresh) + backend scope/connect/readiness tests → T1–T7, T9. **Deferred to phase 1 by design:** `?step=` seeding, the Dashboard readiness *cards* (the endpoint ships here), per-module last-job UI beyond `result`.
- **Deviations to note:** module `/connect` routes stay until phase 4 (spec); `robotLinkNote` moved into the provider (was in `Calibration.tsx`); `survey` events are request-path (module-scoped, no `stream_id`) — the spec's §4.5 list is corrected accordingly.
- **Plan review (2026-08-28, second agent) — all seven findings accepted and folded in:** (1) "running" now precedes the worker + post-start/reconnect reconcile via `/status` + fast-job tests (T1, T8b–d, T9); (2) `LivePreview.start(stream_id=)` resume + scan `live_start(resume=True)` + boundary stamped (T2, T3); (3) `CellArbiter` makes connect/link/job/live transitions atomic (T5a, T5); (4) strict link verifies a manual link when auto-link is off; error text corrected (T5); (5) `ready = rdk.ready && health.robodk.ok !== false`, Connect disabled on camera `in_use` too (T8a); (6) reconciler clears stale `running` (T8b–d); (7) tests for fast job, reconnect racing starts, cross-module preview ownership, Scan/Extrusion filtering, stale-green readiness (T2, T5, T9). Scan no longer auto-connects; the spec status changes only after cell validation (T8c, T10).
- **Type consistency:** `JobRunner.start(kind=, module=, name=) -> str`, `module_status()` keys `running/status/jobs`, `JobEvent.{module,job_id,kind,stream_id}`, `LivePreview.start(owner=) -> str` / `.stream_id` / `.owner`, `ensure_robot_link(rdk, cfg, *, strict, log) -> {enabled, connected, message, ip, configured}`, `pose_not_live_detail(rdk, cfg)`, `usePlatform()` value keys — used identically across tasks.
