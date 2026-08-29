"""A wedged RealSense pipeline must recover itself, or fail loudly and fast.

The cell failure of 2026-08-29: the camera streamed happily for two hours, then
stopped delivering frames at 14:44:04. From then on EVERY acquisition raised
``RuntimeError: Frame didn't arrive within 5000`` -- 57 in a row -- while the
server stayed ``active``, kept LISTENING on 1024 and kept accepting clients. The
operator saw an app that connected normally and then showed nothing, with no
error anywhere near the UI, and eight sockets piled up in CLOSE_WAIT.

Nothing in the server ever rebuilt the pipeline: ``pipeline``/``align`` are built
once under ``if __name__ == '__main__'`` and every client thread reads those
globals forever. So a stall that librealsense could often recover from could only
ever be cleared by a human restarting the service.

That specific camera turned out to be dead at the USB layer (it stopped
answering ``setup address`` with -71 and never re-enumerated), which no software
can fix. The point of this module is the OTHER half of the failure: the server
must not sit there pretending to serve. It retries briefly, rebuilds the pipeline
once if the stall persists, and if that cannot be done it stops claiming to be a
camera -- ``Restart=always``/``RestartSec=3`` then makes the failure visible in
the journal instead of invisible in the UI.

    py -3.10 -m pytest tests/test_camera_recovery.py
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
sys.modules.setdefault("pyrealsense2", SimpleNamespace())
sys.modules.setdefault("turbojpeg", SimpleNamespace())

from server import server_unicast_syncronous as srv  # noqa: E402


TIMEOUT = "Frame didn't arrive within 5000"


class FakePipeline:
    """Stands in for rs.pipeline: hands out frames, or wedges like the real one."""

    def __init__(self, name="pipe", timeouts=0, always_wedged=False):
        self.name = name
        self.remaining_timeouts = timeouts
        self.always_wedged = always_wedged
        self.calls = 0
        self.stopped = False
        self._lock = threading.Lock()

    def wait_for_frames(self):
        with self._lock:
            self.calls += 1
            if self.always_wedged or self.remaining_timeouts > 0:
                if not self.always_wedged:
                    self.remaining_timeouts -= 1
                raise RuntimeError(TIMEOUT)
        return f"frames-from-{self.name}"

    def stop(self):
        self.stopped = True


@pytest.fixture(autouse=True)
def _clean_camera_state(monkeypatch):
    """Each test starts from a healthy, un-wedged supervisor and never sleeps."""
    monkeypatch.setattr(srv, "_recovery_sleep", lambda _s: None)
    srv._reset_camera_state()
    yield
    srv._reset_camera_state()


def _install(pipeline):
    srv.pipeline = pipeline
    srv.align = SimpleNamespace(name="align")


def test_healthy_pipeline_is_left_alone(monkeypatch):
    """The common path must not pay anything for the recovery machinery."""
    opened = []
    monkeypatch.setattr(srv, "openPipeline", lambda: opened.append(1))
    pipe = FakePipeline()
    _install(pipe)

    assert srv.read_frames() == "frames-from-pipe"
    assert pipe.calls == 1
    assert opened == [], "a healthy pipeline must never be rebuilt"


def test_a_brief_stall_is_retried_rather_than_rebuilt(monkeypatch):
    """Rebuilding costs seconds and re-opens the USB device -- too heavy a
    response to the odd dropped frameset, which librealsense recovers from on
    its own. Only a PERSISTENT stall justifies it."""
    opened = []
    monkeypatch.setattr(srv, "openPipeline", lambda: opened.append(1))
    pipe = FakePipeline(timeouts=srv.FRAME_WEDGE_THRESHOLD - 1)
    _install(pipe)

    assert srv.read_frames() == "frames-from-pipe"
    assert opened == [], "a transient stall must not trigger a rebuild"


def test_a_persistent_stall_rebuilds_the_pipeline(monkeypatch):
    """The 2026-08-29 signature: every frame times out until something acts."""
    fresh = FakePipeline(name="fresh")
    opened = []

    def fake_open():
        opened.append(1)
        return fresh, SimpleNamespace(name="fresh-align")

    monkeypatch.setattr(srv, "openPipeline", fake_open)
    wedged = FakePipeline(name="wedged", always_wedged=True)
    _install(wedged)

    assert srv.read_frames() == "frames-from-fresh"
    assert len(opened) == 1, "expected exactly one rebuild"
    assert wedged.stopped, "the wedged pipeline must be released before reopening"
    assert srv.pipeline is fresh, "the new pipeline must become the shared one"
    assert srv.align.name == "fresh-align", "align must be rebound too"


def test_frames_flow_again_for_every_later_caller(monkeypatch):
    """A rebuild is only useful if the OTHER client threads pick it up."""
    fresh = FakePipeline(name="fresh")
    monkeypatch.setattr(srv, "openPipeline",
                        lambda: (fresh, SimpleNamespace(name="fresh-align")))
    _install(FakePipeline(name="wedged", always_wedged=True))

    srv.read_frames()
    assert srv.read_frames() == "frames-from-fresh"


def test_a_real_error_is_not_mistaken_for_a_stall(monkeypatch):
    """Only the frame-timeout is recoverable. Swallowing anything else would
    turn a genuine bug into an endless rebuild loop."""
    opened = []
    monkeypatch.setattr(srv, "openPipeline", lambda: opened.append(1))

    class Broken(FakePipeline):
        def wait_for_frames(self):
            raise RuntimeError("No device connected")

    _install(Broken())

    with pytest.raises(RuntimeError, match="No device connected"):
        srv.read_frames()
    assert opened == [], "a non-timeout error must not trigger a rebuild"


def test_recovered_frames_clear_the_stall_count(monkeypatch):
    """Without a reset, timeouts accumulated over hours of healthy streaming
    would eventually trip a rebuild for no reason."""
    monkeypatch.setattr(srv, "openPipeline", lambda: pytest.fail("no rebuild"))
    _install(FakePipeline(timeouts=srv.FRAME_WEDGE_THRESHOLD - 1))
    srv.read_frames()

    assert srv._consecutive_timeouts == 0
    _install(FakePipeline(timeouts=srv.FRAME_WEDGE_THRESHOLD - 1))
    srv.read_frames()  # must not escalate: the earlier stall was cleared


def test_concurrent_clients_rebuild_the_camera_only_once(monkeypatch):
    """There is ONE camera behind many client threads. Live runs hold several at
    once (preview + telemetry + capture), and they all wedge together -- so a
    naive per-thread rebuild would re-open the USB device several times over,
    which is exactly the hammering that precedes a dead host controller."""
    fresh = FakePipeline(name="fresh")
    opened = []
    open_lock = threading.Lock()

    def fake_open():
        with open_lock:
            opened.append(1)
        return fresh, SimpleNamespace(name="fresh-align")

    monkeypatch.setattr(srv, "openPipeline", fake_open)
    _install(FakePipeline(name="wedged", always_wedged=True))

    results, errors = [], []

    def client():
        try:
            results.append(srv.read_frames())
        except Exception as exc:            # pragma: no cover - failure detail
            errors.append(exc)

    threads = [threading.Thread(target=client) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"clients failed: {errors}"
    assert len(opened) == 1, f"camera re-opened {len(opened)} times, expected 1"
    assert results == ["frames-from-fresh"] * 8


def test_an_unrecoverable_camera_stops_pretending(monkeypatch):
    """The real 2026-08-29 case: the device was gone at the USB layer, so every
    reopen fails too. The server must stop serving rather than keep accepting
    clients it can never feed."""
    def fake_open():
        raise RuntimeError("No device connected")

    monkeypatch.setattr(srv, "openPipeline", fake_open)
    gave_up = []
    monkeypatch.setattr(srv, "_give_up", lambda reason: gave_up.append(reason))
    _install(FakePipeline(always_wedged=True))

    with pytest.raises(srv.CameraUnavailable):
        srv.read_frames()
    assert len(gave_up) == 1, "the operator must be told the camera is gone"
    assert "No device connected" in gave_up[0], "the real cause must be reported"


def test_recovery_is_bounded(monkeypatch):
    """A dead camera must not be retried forever -- that is what turned a stall
    into a flood that buried the kernel log in EPROTO spam."""
    attempts = []

    def fake_open():
        attempts.append(1)
        raise RuntimeError("No device connected")

    monkeypatch.setattr(srv, "openPipeline", fake_open)
    monkeypatch.setattr(srv, "_give_up", lambda reason: None)
    _install(FakePipeline(always_wedged=True))

    with pytest.raises(srv.CameraUnavailable):
        srv.read_frames()
    assert len(attempts) == len(srv.RECOVERY_BACKOFF_S)


def test_backoff_between_reopen_attempts(monkeypatch):
    """Re-opening a USB device in a tight loop is how you wedge a controller."""
    slept = []
    monkeypatch.setattr(srv, "_recovery_sleep", slept.append)
    monkeypatch.setattr(srv, "openPipeline",
                        lambda: (_ for _ in ()).throw(RuntimeError("No device connected")))
    monkeypatch.setattr(srv, "_give_up", lambda reason: None)
    _install(FakePipeline(always_wedged=True))

    with pytest.raises(srv.CameraUnavailable):
        srv.read_frames()
    assert slept == list(srv.RECOVERY_BACKOFF_S)
    assert slept == sorted(slept), "backoff must not shrink between attempts"


def test_a_dead_camera_is_reported_once_not_re_probed(monkeypatch):
    """Once the camera is known gone, later clients must fail immediately rather
    than each paying a full round of 5-second timeouts and reopen attempts."""
    monkeypatch.setattr(srv, "openPipeline",
                        lambda: (_ for _ in ()).throw(RuntimeError("No device connected")))
    monkeypatch.setattr(srv, "_give_up", lambda reason: None)
    _install(FakePipeline(always_wedged=True))

    with pytest.raises(srv.CameraUnavailable):
        srv.read_frames()
    calls_after_first = srv.pipeline.calls

    with pytest.raises(srv.CameraUnavailable):
        srv.read_frames()
    assert srv.pipeline.calls == calls_after_first, "second client re-probed a dead camera"


def test_a_camera_that_reopens_but_never_streams_is_not_rebuilt_forever(monkeypatch):
    """A device can enumerate cleanly and still deliver nothing. Counting the
    reopen as a success would then loop: three timeouts, rebuild, three timeouts,
    rebuild -- re-opening the USB device every ~20 seconds for as long as the
    service runs, which is precisely the hammering the backoff exists to avoid.
    A rebuild only counts once frames actually flow again."""
    opened = []

    def fake_open():
        opened.append(1)
        return (FakePipeline(name=f"dud{len(opened)}", always_wedged=True),
                SimpleNamespace(name="dud-align"))

    monkeypatch.setattr(srv, "openPipeline", fake_open)
    gave_up = []
    monkeypatch.setattr(srv, "_give_up", lambda reason: gave_up.append(reason))
    _install(FakePipeline(always_wedged=True))

    with pytest.raises(srv.CameraUnavailable):
        srv.read_frames()

    assert len(opened) == srv.MAX_REBUILDS_WITHOUT_PROGRESS + 1, (
        f"camera re-opened {len(opened)} times; expected to stop after "
        f"{srv.MAX_REBUILDS_WITHOUT_PROGRESS + 1}")
    assert gave_up, "an endlessly flapping camera must be reported"


def test_a_camera_that_really_recovers_is_forgiven(monkeypatch):
    """Sustained streaming after a rebuild clears the count, so a camera that
    hiccups once a day is never eventually declared dead for it."""
    fresh = FakePipeline(name="fresh")
    monkeypatch.setattr(srv, "openPipeline",
                        lambda: (fresh, SimpleNamespace(name="fresh-align")))
    _install(FakePipeline(name="wedged", always_wedged=True))

    srv.read_frames()                       # triggers the rebuild
    assert srv._rebuilds_without_progress == 1
    for _ in range(srv.HEALTHY_FRAMES_AFTER_REBUILD):
        srv.read_frames()
    assert srv._rebuilds_without_progress == 0, "a healthy camera stayed on probation"


def test_every_acquisition_site_goes_through_the_supervisor():
    """Guard against the supervisor existing while a call site still reaches
    around it -- that would pass every test above and wedge in production. The
    colour-only fast path and the H.264 feeder are separate acquisition loops
    from getFrames, and all three ran the raw call before this change."""
    import inspect
    source = inspect.getsource(srv)
    raw = [line.strip() for line in source.splitlines()
           if "wait_for_frames()" in line and not line.strip().startswith("#")]
    assert len(raw) == 1, (
        f"expected exactly one raw wait_for_frames (the supervisor's), found: {raw}")
    assert "wait_for_frames()" in inspect.getsource(srv.read_frames), (
        "the one raw acquisition call must be the supervisor's own")

    # ...and the three former call sites must now go through it.
    for func in (srv.getFrames, srv.handle_client, srv.stream_h264):
        assert "read_frames()" in inspect.getsource(func), (
            f"{func.__name__} does not acquire through the supervisor")


def test_client_threads_surface_the_failure(monkeypatch):
    """CameraUnavailable must reach _serve_client so the socket is closed and
    the journal names the cause, instead of the thread dying silently."""
    def unavailable(conn, addr):
        raise srv.CameraUnavailable("camera is not delivering frames")

    monkeypatch.setattr(srv, "handle_client", unavailable)
    import socket
    conn, peer = socket.socketpair()
    try:
        srv._serve_client(conn, ("10.12.172.19", 44800))
        assert conn.fileno() == -1, "socket leaked when the camera was unavailable"
    finally:
        conn.close()
        peer.close()
