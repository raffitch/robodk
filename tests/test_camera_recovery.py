"""A wedged RealSense pipeline must recover itself, or fail loudly and fast.

The cell failure of 2026-08-29: the camera streamed happily for two hours, then
stopped delivering frames at 14:44:04. From then on EVERY acquisition raised
``RuntimeError: Frame didn't arrive within 5000`` -- 57 in a row -- while the
server stayed ``active``, kept LISTENING on 1024 and kept accepting clients. The
operator saw an app that connected normally and then showed nothing, with no
error anywhere near the UI, and eight sockets piled up in CLOSE_WAIT.

Nothing in the server ever rebuilt the pipeline: the camera state was built once
under ``if __name__ == '__main__'`` and every client thread read those globals
forever. So a stall that librealsense could often recover from could only ever be
cleared by a human restarting the service.

That specific camera turned out to be dead at the USB layer (it stopped
answering ``setup address`` with -71 and never re-enumerated), which no software
can fix. The point of this module is the OTHER half of the failure: the server
must not sit there pretending to serve. It retries briefly, rebuilds the pipeline
once if the stall persists, and if that cannot be done it stops claiming to be a
camera -- ``Restart=always``/``RestartSec=3`` then makes the failure visible in
the journal instead of invisible in the UI.

"Camera state" is more than the pipeline. ``openPipeline()`` returns the pipeline
(protocol 2 dropped the ``rs2::align`` it used to return alongside it) and, as a
side effect, rebinds the module globals that describe what those frames MEAN:
``STATIC_GEOMETRY`` (depth->colour extrinsics + both intrinsics), ``ACHIEVED_OPTIONS``
and ``DEVICE_INFO`` -- all three served by ``make_greeting()`` -- while
``_rebuild_pipeline`` re-reads ``depth_unit_mm`` off the new device. A rebuild that
swapped only the pipeline would stream frames from the new open and describe them
with the dead device's numbers: no exception, no log line, just silently wrong
millimetres downstream. These tests hold that whole set together.

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

# Everything a client is served from: the frame source, and the numbers that turn
# its frames into millimetres. A rebuild has to refresh all of them together.
CAMERA_GLOBALS = ("pipeline", "depth_unit_mm", "STATIC_GEOMETRY",
                  "ACHIEVED_OPTIONS", "DEVICE_INFO")


class FakePipeline:
    """Stands in for rs.pipeline: hands out frames, or wedges like the real one.

    ``depth_scale`` is the device's metres-per-count, as ``_rebuild_pipeline`` reads
    it back off the reopened device. Left None, ``get_active_profile()`` raises the
    way a binding that cannot report it would -- the rebuild then has to fall back to
    what ``openPipeline`` read off that same reopened device, and say so.
    """

    def __init__(self, name="pipe", timeouts=0, always_wedged=False, depth_scale=None):
        self.name = name
        self.remaining_timeouts = timeouts
        self.always_wedged = always_wedged
        self.depth_scale = depth_scale
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

    def get_active_profile(self):
        if self.depth_scale is None:
            raise RuntimeError("no active profile")
        sensor = SimpleNamespace(get_depth_scale=lambda: self.depth_scale)
        device = SimpleNamespace(first_depth_sensor=lambda: sensor)
        return SimpleNamespace(get_device=lambda: device)

    def stop(self):
        self.stopped = True


@pytest.fixture(autouse=True)
def _clean_camera_state(monkeypatch):
    """Each test starts from a healthy, un-wedged supervisor and never sleeps."""
    monkeypatch.setattr(srv, "_recovery_sleep", lambda _s: None)
    # Re-set each camera global to its own current value purely to register the
    # undo, so a test that installs a fake camera cannot leak it into the next one.
    for name in CAMERA_GLOBALS:
        monkeypatch.setattr(srv, name, getattr(srv, name))
    srv._reset_camera_state()
    yield
    srv._reset_camera_state()


def _install(pipeline, tag="stale"):
    """Put the server in the state a previous open left behind.

    Tagging every camera global lets a test tell "produced by THIS open" from
    "whatever happened to be there before the stall".
    """
    srv.pipeline = pipeline
    srv.depth_unit_mm = 1.0
    srv.STATIC_GEOMETRY = SimpleNamespace(tag=tag)
    srv.ACHIEVED_OPTIONS = {"tag": tag}
    srv.DEVICE_INFO = {"tag": tag, "serial": f"serial-{tag}",
                       "color_auto_exposure_priority": 0}


def _opener(pipeline, tag, opened=None, lock=None, depth_unit_mm=0.1):
    """A stand-in for openPipeline with the same side effects as the real one.

    The real ``openPipeline`` declares ``global STATIC_GEOMETRY, ACHIEVED_OPTIONS,
    DEVICE_INFO`` and rebinds all three while starting the streams, then returns the
    pipeline alone. Anything that models it has to do both halves.

    ``ACHIEVED_OPTIONS`` always carries a ``depth_unit_mm`` key because the real one
    does: ``rs_config.configure_depth_sensor`` sets ``depth_units`` on the reopened
    sensor and reads the achieved scale straight back (storing None when the device
    will not report it, which ``depth_unit_mm=None`` models). That key is the
    rebuild's fallback when the direct re-read fails, so it is part of what an opener
    has to reproduce -- 0.1 is the pinned 0.1 mm/count of protocol 2.
    """
    def fake_open():
        if opened is not None:
            if lock is not None:
                with lock:
                    opened.append(1)
            else:
                opened.append(1)
        srv.STATIC_GEOMETRY = SimpleNamespace(tag=tag)
        srv.ACHIEVED_OPTIONS = {"tag": tag, "depth_unit_mm": depth_unit_mm}
        srv.DEVICE_INFO = {"tag": tag, "serial": f"serial-{tag}",
                           "color_auto_exposure_priority": 1}
        return pipeline
    return fake_open


def _assert_serves_one_camera(tag):
    """Frames and the numbers that describe them must come from the SAME open.

    ``make_greeting()`` serves STATIC_GEOMETRY, depth_unit_mm, ACHIEVED_OPTIONS and
    DEVICE_INFO; ``stream_h264``'s telemetry loop back-projects live frames through
    ``STATIC_GEOMETRY.R_dc``/``t_dc_mm`` and scales raw counts by ``depth_unit_mm``.
    All of those are module globals, so a rebuild that rebinds ``pipeline`` and
    nothing else leaves the server streaming the new device through the old device's
    geometry -- a silent metric error, not a crash. This is the check that catches it.

    Asserted through ``_camera_snapshot()`` -- the one coherent read every serving
    path now goes through -- rather than off the globals directly, so this checks
    what a client is actually SERVED and not merely what the module happens to hold.
    """
    snap = srv._camera_snapshot()
    assert snap.pipeline.name == tag, "frames are not coming from the newest open"
    assert snap.geometry.tag == tag, (
        "STATIC_GEOMETRY is stale: the greeting and the telemetry back-projection "
        "would describe new frames with the pre-stall extrinsics")
    assert snap.achieved.get("tag") == tag, (
        "ACHIEVED_OPTIONS is stale: the greeting would report options the reopened "
        "sensor was never actually configured with")
    assert snap.device.get("tag") == tag, (
        "DEVICE_INFO is stale: the greeting would report the pre-stall serial/"
        "firmware and colour auto_exposure_priority")


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
    fresh = FakePipeline(name="fresh", depth_scale=0.0001)
    opened = []
    monkeypatch.setattr(srv, "openPipeline", _opener(fresh, "fresh", opened))
    wedged = FakePipeline(name="wedged", always_wedged=True)
    _install(wedged)

    assert srv.read_frames() == "frames-from-fresh"
    assert len(opened) == 1, "expected exactly one rebuild"
    assert wedged.stopped, "the wedged pipeline must be released before reopening"
    assert srv.pipeline is fresh, "the new pipeline must become the shared one"
    # ...and the pipeline is only half of it: everything the greeting and the
    # telemetry loop read must have been rebound by the same open.
    _assert_serves_one_camera("fresh")
    assert srv.depth_unit_mm == pytest.approx(0.1), (
        "depth_unit_mm was not re-read off the reopened device; protocol 2 sends "
        "raw counts, so a stale scale silently mis-sizes every depth the host reads")


def test_an_unreadable_depth_scale_falls_back_to_the_reopened_device(monkeypatch):
    """The direct re-read failing must not leave the PREVIOUS open's scale in place.

    ``depth_unit_mm`` is the one number every host measurement is multiplied by --
    protocol 2 ships raw counts, so the host multiplies by whatever the greeting
    says. The re-read used to sit in a bare ``try/except: pass`` justified by "same
    device; the startup value still holds", but the value is a per-OPEN fact:
    ``configure_depth_sensor`` sets ``depth_units`` and reads it back on every open,
    and a device that comes back at its 1 mm/count default while the module still
    says 0.1 mm mis-sizes every depth by 10x, silently.

    ``openPipeline`` has already read the scale off the REOPENED device into
    ``ACHIEVED_OPTIONS``, so that -- not the stale global -- is what a failed direct
    read must fall back to.
    """
    fresh = FakePipeline(name="fresh", depth_scale=None)      # direct re-read raises
    monkeypatch.setattr(srv, "openPipeline",
                        _opener(fresh, "fresh", depth_unit_mm=0.1))
    _install(FakePipeline(name="wedged", always_wedged=True))  # leaves depth_unit_mm 1.0

    assert srv.read_frames() == "frames-from-fresh"
    assert srv.depth_unit_mm == pytest.approx(0.1), (
        "the rebuild kept the previous open's mm/count instead of the value "
        "openPipeline read off the reopened device")
    assert srv._camera_snapshot().depth_unit_mm == pytest.approx(0.1), (
        "the greeting would still quote the pre-stall depth scale")


def test_a_depth_scale_that_cannot_be_read_at_all_is_reported(monkeypatch, capsys):
    """When NEITHER source works the value really is possibly stale -- so say so.

    Silence was the actual defect: a bare ``except: pass`` on the scale factor that
    every downstream millimetre depends on. Carrying the old value forward is the
    right behaviour (there is nothing better to use), but it has to be visible in
    the journal, because from that point on no measurement is trustworthy without
    someone checking.
    """
    fresh = FakePipeline(name="fresh", depth_scale=None)      # direct re-read raises
    monkeypatch.setattr(srv, "openPipeline",
                        _opener(fresh, "fresh", depth_unit_mm=None))  # and so did openPipeline
    _install(FakePipeline(name="wedged", always_wedged=True))

    assert srv.read_frames() == "frames-from-fresh"
    assert srv.depth_unit_mm == 1.0, "with no reading available, keep the last known one"

    # ...and the operator has to be able to SEE that, in the rebuild's own output.
    logged = capsys.readouterr().out
    assert "could not re-read the depth scale" in logged, (
        f"the failed depth-scale read was swallowed silently; log was {logged!r}")
    assert "KEEPING 1.0 mm/count" in logged, (
        f"the journal does not say which scale the camera is now serving: {logged!r}")


def test_the_depth_scale_re_read_never_raises(monkeypatch):
    """It runs inside the recovery path, and the unit is Restart=always with NO
    start limit -- an exception here is an infinite crash-loop with the camera dark
    for every module, which is strictly worse than a stale number plus a log line."""
    broken = FakePipeline(name="broken", depth_scale=None)
    for junk in (None, {}, {"depth_unit_mm": None}, {"depth_unit_mm": "0.1 mm"},
                 SimpleNamespace()):
        monkeypatch.setattr(srv, "ACHIEVED_OPTIONS", junk)
        assert srv._reread_depth_unit_mm(broken, 0.1) == 0.1


def test_a_rebuild_that_forgets_to_rebind_is_caught(monkeypatch):
    """Proof that the rebind assertions above can actually FAIL.

    The defect modelled here is one line: drop ``global STATIC_GEOMETRY,
    ACHIEVED_OPTIONS, DEVICE_INFO`` from openPipeline and its three assignments
    become function locals -- the streams restart, the pipeline is returned and
    rebound, and the module keeps the geometry of the device that just died. Nothing
    raises; frames flow again; the greeting and the telemetry back-projection go on
    using the pre-stall numbers forever.

    So: run the supervisor against exactly that opener, confirm the failure really is
    invisible from the frames alone, and confirm the invariant check catches it.
    """
    fresh = FakePipeline(name="fresh")
    monkeypatch.setattr(srv, "openPipeline", lambda: fresh)   # rebinds nothing else
    _install(FakePipeline(name="wedged", always_wedged=True), tag="dead-device")

    assert srv.read_frames() == "frames-from-fresh"     # looks perfectly healthy
    assert srv.pipeline is fresh                        # ...and the pipeline IS fresh
    assert srv.STATIC_GEOMETRY.tag == "dead-device"     # but the geometry is not

    with pytest.raises(AssertionError, match="STATIC_GEOMETRY is stale"):
        _assert_serves_one_camera("fresh")


def test_frames_flow_again_for_every_later_caller(monkeypatch):
    """A rebuild is only useful if the OTHER client threads pick it up."""
    fresh = FakePipeline(name="fresh")
    monkeypatch.setattr(srv, "openPipeline", _opener(fresh, "fresh"))
    _install(FakePipeline(name="wedged", always_wedged=True))

    srv.read_frames()
    assert srv.read_frames() == "frames-from-fresh"
    _assert_serves_one_camera("fresh")


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
    monkeypatch.setattr(srv, "openPipeline",
                        _opener(fresh, "fresh", opened, threading.Lock()))
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
    # One rebuild, one set of globals: the eight threads must not have left a
    # half-updated camera behind (new pipeline, some earlier open's geometry).
    _assert_serves_one_camera("fresh")


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
        dud = FakePipeline(name=f"dud{len(opened)}", always_wedged=True)
        return _opener(dud, f"dud{len(opened)}")()

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
    monkeypatch.setattr(srv, "openPipeline", _opener(fresh, "fresh"))
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


def test_openpipeline_returns_the_pipeline_alone():
    """Protocol 2 removed rs2::align, so openPipeline returns ONE object and the
    rebuild installs it directly. If it ever grows a second return value again,
    ``pipeline = openPipeline()`` silently becomes a tuple and EVERY acquisition
    dies with 'tuple object has no attribute wait_for_frames' -- which is exactly
    how this test module rotted. Pin the contract at both ends."""
    import inspect
    src = inspect.getsource(srv.openPipeline)
    returns = [line.split("#")[0].strip() for line in src.splitlines()
               if line.strip().startswith("return ")]
    assert returns == ["return pipeline"], (
        f"openPipeline's return contract changed: {returns}")

    calls = [line.strip() for line in inspect.getsource(srv._rebuild_pipeline).splitlines()
             if "openPipeline()" in line and not line.strip().startswith("#")]
    assert calls, "_rebuild_pipeline no longer re-opens through openPipeline"
    for line in calls:
        assert "," not in line.split("=")[0], (
            f"_rebuild_pipeline unpacks openPipeline's result as a tuple: {line}")


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
