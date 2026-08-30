"""The protocol-2 greeting must describe ONE camera open -- the one still streaming.

Two live defects, both invisible: no exception, no log line, just wrong millimetres.

DEFECT 1 -- a torn greeting. ``make_greeting()`` read ``pipeline``,
``STATIC_GEOMETRY``, ``ACHIEVED_OPTIONS``, ``DEVICE_INFO`` and ``depth_unit_mm``
WITHOUT ``_camera_lock``, while ``_rebuild_pipeline`` rebinds those same globals one
at a time UNDER it (``openPipeline`` sets three as a side effect of restarting the
streams; the rebuild then sets the other two). A client connecting during the
~1-21 s recovery window could be handed a mixture of two opens -- the reopened
device's serial with the dead device's extrinsics, or the new geometry with the old
depth scale. The host turns the greeting into ONE ``CameraGeometry`` and
back-projects every depth frame of that connection through it, so a mix that happens
not to raise is a silent metric error for the life of the connection.

DEFECT 2 -- a greeting that was true once. It is sent ONCE per connection and the
frame stream carries no generation marker, so a client already connected when the
camera is rebuilt keeps using the pre-rebuild numbers. Same physical device, so they
are usually identical -- but ``configure_depth_sensor`` re-applies options on every
open, so ``depth_unit_mm`` is not guaranteed to be. Observed on the cell 2026-08-30:
client 40437 connected 01:03:42, the camera rebuilt at 01:03:57, and that connection
lived until 01:08:35 -- 4.5 minutes describing the new open with the old open's scale.

Re-sending the greeting would be a wire-format change (the host reads exactly one
greeting line per connection), so the fix is to END the connection whose greeting has
gone stale and let the client's existing reconnect path fetch a fresh one. That close
is scoped to connections that were actually greeted: colour-only and H.264 clients
carry no geometry and are left alone.

    py -3.10 -m pytest tests/test_camera_greeting_coherence.py
"""
from __future__ import annotations

import inspect
import io
import json
import socket
import struct
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import lz4.frame as lz4f
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
sys.modules.setdefault("pyrealsense2", SimpleNamespace())
sys.modules.setdefault("turbojpeg", SimpleNamespace())

from server import server_unicast_syncronous as srv  # noqa: E402


TIMEOUT = "Frame didn't arrive within 5000"

CAMERA_GLOBALS = ("pipeline", "depth_unit_mm", "STATIC_GEOMETRY",
                  "ACHIEVED_OPTIONS", "DEVICE_INFO", "depth_filters")

# Every open gets its own value for every number the greeting carries, so a torn
# greeting is READABLE: which field came from which open is written on its face.
# depth_unit_mm is the one the rebuild re-reads off the device, so the fake
# pipeline's depth_scale (metres/count) has to agree with it.
OPENS = {
    "stale": {"unit_mm": 1.0, "fx": 640.0, "tx_mm": 15.0},
    "fresh": {"unit_mm": 0.2, "fx": 700.0, "tx_mm": 16.0},
}
_UNIT_TO_OPEN = {v["unit_mm"]: k for k, v in OPENS.items()}
_TX_TO_OPEN = {v["tx_mm"]: k for k, v in OPENS.items()}


class _KeptStreaming(RuntimeError):
    """Raised by a fake frame source to stop a serving loop that will not stop itself."""


class FakePipeline:
    """Stands in for rs.pipeline. ``depth_scale`` is metres-per-count, as
    ``_rebuild_pipeline`` reads it back off the reopened device."""

    def __init__(self, name="pipe", depth_scale=None, always_wedged=False):
        self.name = name
        self.depth_scale = depth_scale
        self.always_wedged = always_wedged
        self.stopped = False

    def wait_for_frames(self):
        if self.always_wedged:
            raise RuntimeError(TIMEOUT)
        return f"frames-from-{self.name}"

    def get_active_profile(self):
        sensor = SimpleNamespace(get_depth_scale=lambda: self.depth_scale)
        device = SimpleNamespace(first_depth_sensor=lambda: sensor)
        return SimpleNamespace(get_device=lambda: device)

    def stop(self):
        self.stopped = True


def _geometry(tag):
    """A stand-in for rs_geometry.StaticGeometry, marked with the open that made it."""
    fx = OPENS[tag]["fx"]
    return SimpleNamespace(
        tag=tag,
        depth={"open": tag, "width": 1280, "height": 720, "fx": fx, "fy": fx,
               "ppx": 640.0, "ppy": 360.0, "model": "none", "coeffs": [0.0] * 5},
        color={"open": tag, "width": 1920, "height": 1080, "fx": fx * 1.5,
               "fy": fx * 1.5, "ppx": 960.0, "ppy": 540.0, "model": "none",
               "coeffs": [0.0] * 5},
        R_dc=np.eye(3),
        t_dc_mm=np.array([OPENS[tag]["tx_mm"], 0.0, 0.0]),
        depth_size=(1280, 720), color_size=(1920, 1080))


def _publish(tag, pipeline=None):
    """Bind every camera global to the same open, the way a completed open leaves them."""
    if pipeline is not None:
        srv.pipeline = pipeline
    srv.depth_unit_mm = OPENS[tag]["unit_mm"]
    srv.STATIC_GEOMETRY = _geometry(tag)
    srv.ACHIEVED_OPTIONS = {"visual_preset": tag, "laser_power": tag}
    srv.DEVICE_INFO = {"open": tag, "serial": f"serial-{tag}", "fw": "5.16.0.1",
                       "librealsense": "2.55.1", "color_auto_exposure_priority": 0}


def _opens_named_in(greeting):
    """Which open each part of a greeting came from. One distinct value = coherent."""
    return {
        "depth intrinsics": greeting["depth"]["open"],
        "colour intrinsics": greeting["color"]["open"],
        "depth->colour extrinsic":
            _TX_TO_OPEN.get(greeting["depth_to_color"]["translation_mm"][0]),
        "depth_unit_mm": _UNIT_TO_OPEN.get(greeting["depth_unit_mm"]),
        "achieved options": greeting["device"]["visual_preset"],
        "device info": greeting["device"]["open"],
    }


def _assert_greeting_describes_one_open(greeting, expected):
    named = _opens_named_in(greeting)
    assert set(named.values()) == {expected}, (
        f"the greeting mixes two camera opens (expected all '{expected}'): {named}. "
        f"The host builds ONE CameraGeometry from this line and back-projects every "
        f"depth frame of the connection through it, so a mix is wrong millimetres "
        f"with no error and no log line.")


@pytest.fixture(autouse=True)
def _clean_camera_state(monkeypatch):
    monkeypatch.setattr(srv, "_recovery_sleep", lambda _s: None)
    for name in CAMERA_GLOBALS:
        monkeypatch.setattr(srv, name, getattr(srv, name))
    srv._reset_camera_state()
    yield
    srv._reset_camera_state()


# --- Defect 1: a greeting built while the camera is being rebuilt ------------

def test_a_greeting_built_mid_rebuild_never_mixes_the_geometry_of_two_opens(monkeypatch):
    """The first half of the tear: openPipeline rebinds the geometry BEFORE the
    options/device block, so a client that reads in between gets the reopened
    device's extrinsics with the dead device's serial (or vice versa).

    Interleaved on purpose rather than raced: the fake open publishes the new
    geometry, hands the connecting client the floor, and only then publishes the
    rest. Without a coherent read the client sees exactly the half-updated module.
    """
    half_rebuilt = threading.Event()
    greeting_done = threading.Event()
    fresh = FakePipeline("fresh", depth_scale=OPENS["fresh"]["unit_mm"] / 1000.0)

    def fake_open():
        srv.STATIC_GEOMETRY = _geometry("fresh")        # openPipeline's side effect...
        half_rebuilt.set()
        greeting_done.wait(0.5)                          # ...the client's chance to tear
        srv.ACHIEVED_OPTIONS = {"visual_preset": "fresh", "laser_power": "fresh"}
        srv.DEVICE_INFO = {"open": "fresh", "serial": "serial-fresh", "fw": "5.16.0.1",
                           "librealsense": "2.55.1", "color_auto_exposure_priority": 1}
        return fresh

    monkeypatch.setattr(srv, "openPipeline", fake_open)
    _publish("stale", FakePipeline("wedged", always_wedged=True))

    result = {}

    def connecting_client():
        half_rebuilt.wait(5.0)
        try:
            result["greeting"] = srv.make_greeting()
        except Exception as exc:                        # a raise is benign, a mix is not
            result["error"] = exc
        finally:
            greeting_done.set()

    client = threading.Thread(target=connecting_client, name="connecting-client")
    client.start()
    assert srv.read_frames() == "frames-from-fresh"      # the wedge triggers the rebuild
    client.join(timeout=10)
    assert not client.is_alive(), "the connecting client never finished"

    if "error" in result:
        pytest.fail(f"greeting raised instead of being served: {result['error']!r}")
    _assert_greeting_describes_one_open(result["greeting"], "fresh")


def test_a_greeting_built_mid_rebuild_never_pairs_new_geometry_with_the_old_scale(monkeypatch):
    """The second half of the tear, and the expensive one.

    ``_rebuild_pipeline`` installs the new pipeline, then re-reads ``depth_unit_mm``
    off it. Between those two statements the module holds the new open's geometry and
    the old open's millimetres-per-count -- a greeting built there tells the host to
    read 0.1 mm words as 1 mm ones, which reads a real 300 mm standoff as 3000 mm.
    The gate is the depth-scale read itself, so the interleaving is exact.
    """
    reading_scale = threading.Event()
    greeting_done = threading.Event()

    class GatedPipeline(FakePipeline):
        """Blocks the first time the rebuild reads the depth scale off it."""

        def __init__(self):
            super().__init__("fresh", depth_scale=OPENS["fresh"]["unit_mm"] / 1000.0)
            self._gated = False

        def get_active_profile(self):
            if not self._gated:
                self._gated = True
                reading_scale.set()
                greeting_done.wait(0.5)
            return super().get_active_profile()

    fresh = GatedPipeline()

    def fake_open():
        srv.STATIC_GEOMETRY = _geometry("fresh")
        srv.ACHIEVED_OPTIONS = {"visual_preset": "fresh", "laser_power": "fresh"}
        srv.DEVICE_INFO = {"open": "fresh", "serial": "serial-fresh", "fw": "5.16.0.1",
                           "librealsense": "2.55.1", "color_auto_exposure_priority": 1}
        return fresh

    monkeypatch.setattr(srv, "openPipeline", fake_open)
    _publish("stale", FakePipeline("wedged", always_wedged=True))

    result = {}

    def connecting_client():
        reading_scale.wait(5.0)
        try:
            result["greeting"] = srv.make_greeting()
        except Exception as exc:
            result["error"] = exc
        finally:
            greeting_done.set()

    client = threading.Thread(target=connecting_client, name="connecting-client")
    client.start()
    assert srv.read_frames() == "frames-from-fresh"
    client.join(timeout=10)
    assert not client.is_alive(), "the connecting client never finished"

    if "error" in result:
        pytest.fail(f"greeting raised instead of being served: {result['error']!r}")
    _assert_greeting_describes_one_open(result["greeting"], "fresh")
    assert result["greeting"]["depth_unit_mm"] == pytest.approx(OPENS["fresh"]["unit_mm"])


def test_building_a_greeting_cannot_hold_the_camera_lock_across_device_io():
    """Coherence must not be bought by letting a client stall the supervisor.

    ``make_greeting`` does live device I/O (temperatures, global time) on a camera
    that may be half-dead. If that ran while holding ``_camera_lock``, one hung
    control transfer would keep ``_rebuild_pipeline`` from ever starting -- turning a
    recoverable stall into the permanent one the supervisor exists to prevent. So the
    lock covers the six reads and nothing else.
    """
    in_device_io = threading.Event()
    release = threading.Event()

    class SlowPipeline(FakePipeline):
        def get_active_profile(self):
            in_device_io.set()
            release.wait(5.0)
            return super().get_active_profile()

    _publish("stale", SlowPipeline("slow", depth_scale=0.001))
    greeter = threading.Thread(target=srv.make_greeting, name="slow-greeter")
    greeter.start()
    try:
        assert in_device_io.wait(5.0), "the greeter never reached its device I/O"
        acquired = srv._camera_lock.acquire(timeout=2.0)
        if acquired:
            srv._camera_lock.release()
        assert acquired, (
            "a client stuck in make_greeting's device I/O is holding _camera_lock, so "
            "the supervisor could never rebuild the camera it is stuck on")
    finally:
        release.set()
        greeter.join(timeout=5)


# --- Defect 2: a greeting that a rebuild has made a lie ---------------------

def _fake_jpeg(monkeypatch):
    monkeypatch.setattr(srv, "turbojpeg", SimpleNamespace(
        TurboJPEG=lambda _path: SimpleNamespace(
            encode=lambda img, quality=None: b"JPEGBYTES")))


def _drain(sock):
    sock.settimeout(5.0)
    out = bytearray()
    while True:
        try:
            chunk = sock.recv(65536)
        except socket.timeout:                          # pragma: no cover - failure detail
            pytest.fail("the server never closed the connection")
        if not chunk:
            return bytes(out)
        out.extend(chunk)


def _split_greeting(wire, preamble=b""):
    """The greeting this connection was sent, and every byte that followed it.

    Everything after that one line is payload the client will scale with the
    greeting -- so "what did this connection actually deliver?" is answered by
    looking at the tail, not by counting acquisitions inside the server.
    """
    assert wire.startswith(preamble), f"expected {preamble!r} first, got {wire[:32]!r}"
    line, sep, tail = wire[len(preamble):].partition(b"\n")
    assert sep, "the server sent no greeting line at all"
    return json.loads(line.decode("utf-8")), tail


def _depth_frames_in(tail):
    """Decode the protocol-2 frames in ``tail`` exactly as the host's reader does:
    ``<I depth_len><I color_len><d ts>`` then lz4(np.save(depth)) then colour JPEG."""
    frames = []
    while len(tail) >= 16:
        depth_len, color_len = struct.unpack("<II", tail[:8])
        body = tail[16:16 + depth_len + color_len]
        frames.append(np.load(io.BytesIO(lz4f.decompress(body[:depth_len])),
                              allow_pickle=False))
        tail = tail[16 + depth_len + color_len:]
    return frames


def test_a_client_greeted_before_a_rebuild_is_disconnected(monkeypatch, capsys):
    """The 2026-08-30 case: greeted at generation N, still streaming at N+1.

    Nothing on the wire can tell that client its numbers changed, so the only honest
    move is to end the connection. It reconnects through the unchanged handshake and
    is greeted with the camera that is actually streaming.
    """
    _fake_jpeg(monkeypatch)
    _publish("stale", FakePipeline("stale", depth_scale=0.001))
    depth = np.full((4, 4), 3000, dtype=np.uint16)
    color = np.zeros((8, 8, 3), dtype=np.uint8)
    calls = []
    served_after_rebuild = 25

    def fake_get_frames(_filters):
        calls.append(1)
        if len(calls) > served_after_rebuild:
            raise _KeptStreaming(
                f"the camera was rebuilt after frame 1 and this connection went on to "
                f"serve {served_after_rebuild} more frames through the greeting it was "
                f"given before the rebuild")
        if len(calls) == 1:
            srv._camera_generation += 1                 # a rebuild completes
        return depth, color, 1.0

    monkeypatch.setattr(srv, "getFrames", fake_get_frames)
    server_sock, client_sock = socket.socketpair()
    try:
        client_sock.sendall(b"MODE FULL V2\n")
        srv.handle_client(server_sock, ("10.12.172.19", 40437))

        assert len(calls) <= 1, (
            f"the connection went on acquiring after the rebuild ({len(calls)} "
            f"acquisitions); it must end at the first frame after the rebuild")
        assert server_sock.fileno() == -1, "the stale connection was not closed"
        wire = _drain(client_sock)
        greeting, tail = _split_greeting(wire)
        # It was greeted honestly -- with the open that was live at the time. That is
        # exactly why it cannot be allowed to keep streaming afterwards.
        _assert_greeting_describes_one_open(greeting, "stale")
        # The measurable contract is BYTES, not acquisitions: one frame is a whole
        # measurement (CameraClient.grab() reads exactly one per connection), so
        # "the loop only ran once" is not a pass -- zero frames delivered is.
        assert tail == b"", (
            f"{len(_depth_frames_in(tail))} frame(s) ({len(tail)} bytes) were "
            f"delivered after the rebuild, described by the pre-rebuild greeting")
        assert "camera rebuilt" in capsys.readouterr().out, (
            "a server-initiated close must name its reason in the journal")
    finally:
        server_sock.close()
        client_sock.close()


def test_a_rebuild_landing_inside_getframes_leaks_no_frame(monkeypatch, capsys):
    """The hole left by checking staleness only at the TOP of the serve loop.

    ``getFrames`` runs the Nano's whole filter chain -- roughly a second -- so a
    rebuild that lands *inside* it has already got past the top-of-loop check. The
    frame is then compressed and written to the socket under the PRE-rebuild
    greeting, and only the next iteration notices. One frame is the entire defect:
    ``CameraClient.grab()`` reads exactly one frame per connection, and that is the
    per-pose scan capture, the gate grab and the extrusion measurement. Here the
    reopened device publishes 0.2 mm/count while the greeting still says 1.0, so the
    leaked frame reads a real 600 mm standoff as 3000 mm -- silently, 5x out.
    """
    _fake_jpeg(monkeypatch)
    _publish("stale", FakePipeline("stale", depth_scale=0.001))
    depth = np.full((4, 4), 3000, dtype=np.uint16)
    color = np.zeros((8, 8, 3), dtype=np.uint8)
    calls = []

    def fake_get_frames(_filters):
        calls.append(1)
        if len(calls) == 1:
            # The rebuild completes DURING the acquisition, and leaves every camera
            # global bound to the new open the way _rebuild_pipeline does.
            _publish("fresh")
            srv._camera_generation += 1
        if len(calls) > 5:                              # pragma: no cover - failure detail
            raise _KeptStreaming("the connection never noticed the rebuild at all")
        return depth, color, 1.0

    monkeypatch.setattr(srv, "getFrames", fake_get_frames)
    server_sock, client_sock = socket.socketpair()
    try:
        client_sock.sendall(b"MODE FULL V2\n")
        srv.handle_client(server_sock, ("10.12.172.19", 40437))
        wire = _drain(client_sock)
        greeting, tail = _split_greeting(wire)
        _assert_greeting_describes_one_open(greeting, "stale")

        leaked = _depth_frames_in(tail)
        detail = ""
        if leaked:                                      # pragma: no cover - failure detail
            word = float(leaked[0].flat[0])
            detail = (f" The host reads {word * greeting['depth_unit_mm']:.1f} mm "
                      f"where the truth is {word * OPENS['fresh']['unit_mm']:.1f} mm.")
        assert tail == b"", (
            f"{len(leaked)} frame(s) ({len(tail)} bytes) acquired across the rebuild "
            f"were sent under the pre-rebuild greeting "
            f"(depth_unit_mm={greeting['depth_unit_mm']}, the camera now reports "
            f"{OPENS['fresh']['unit_mm']}).{detail}")
        assert "camera rebuilt" in capsys.readouterr().out
    finally:
        server_sock.close()
        client_sock.close()


def test_a_burst_session_ends_when_its_greeting_goes_stale(monkeypatch, capsys):
    """A burst tour buffers frames on the Jetson under one greeting and pulls them at
    the end. A rebuild mid-tour means the buffer straddles two opens, and the host
    would scale every buffered frame with the first open's numbers. End the session:
    ``_capture`` catches the error and falls back to the per-pose grab path."""
    _fake_jpeg(monkeypatch)
    _publish("stale", FakePipeline("stale", depth_scale=0.001))
    depth = np.full((4, 4), 3000, dtype=np.uint16)
    color = np.zeros((8, 8, 3), dtype=np.uint8)
    calls = []

    def fake_get_frames(_filters):
        calls.append(1)
        if len(calls) == 1:
            srv._camera_generation += 1                 # a rebuild completes mid-tour
        return depth, color, 1.0

    monkeypatch.setattr(srv, "getFrames", fake_get_frames)
    server_sock, client_sock = socket.socketpair()
    try:
        client_sock.sendall(b"CAP\nCAP\n")
        client_sock.shutdown(socket.SHUT_WR)            # so a server that ignores the
        srv.stream_burst(server_sock, ("10.12.172.19", 40437))   # rebuild still ends

        assert len(calls) <= 1, (
            f"the burst session captured {len(calls)} frames; the second CAP came "
            f"after the rebuild and would have been described by the first open's "
            f"greeting")
        assert "camera rebuilt" in capsys.readouterr().out
    finally:
        server_sock.close()
        client_sock.close()


def test_a_burst_cap_spanning_a_rebuild_is_neither_buffered_nor_answered(monkeypatch,
                                                                        capsys):
    """The burst path has the same hole, in the same place.

    ``CAP`` acquires through ``getFrames`` -- the same ~1 s filter chain -- so a
    rebuild landing inside it slips past the top-of-loop check. The frame is then
    appended to the RAM buffer (which a later ``GET`` ships wholesale) and its
    thumbnail is written to the socket, all under the pre-rebuild greeting. Nothing
    captured across the rebuild may reach the client or the buffer.
    """
    _fake_jpeg(monkeypatch)
    _publish("stale", FakePipeline("stale", depth_scale=0.001))
    depth = np.full((4, 4), 3000, dtype=np.uint16)
    color = np.zeros((8, 8, 3), dtype=np.uint8)
    calls = []

    def fake_get_frames(_filters):
        calls.append(1)
        if len(calls) == 1:
            _publish("fresh")
            srv._camera_generation += 1                 # a rebuild completes mid-CAP
        return depth, color, 1.0

    monkeypatch.setattr(srv, "getFrames", fake_get_frames)
    server_sock, client_sock = socket.socketpair()
    try:
        client_sock.sendall(b"CAP\n")
        client_sock.shutdown(socket.SHUT_WR)            # so a server that ignores the
        srv.stream_burst(server_sock, ("10.12.172.19", 40437))   # rebuild still ends
        server_sock.shutdown(socket.SHUT_WR)   # EOF for _drain (stream_burst does not
                                               # close; handle_client's caller does)

        greeting, tail = _split_greeting(_drain(client_sock), preamble=b"BURST READY\n")
        _assert_greeting_describes_one_open(greeting, "stale")
        assert tail == b"", (
            f"the burst session answered {len(tail)} byte(s) for a CAP whose frame "
            f"was acquired across the rebuild; the greeting says depth_unit_mm="
            f"{greeting['depth_unit_mm']} and the camera now reports "
            f"{OPENS['fresh']['unit_mm']}")
        assert "camera rebuilt" in capsys.readouterr().out
    finally:
        server_sock.close()
        client_sock.close()


def test_a_colour_only_client_is_not_dropped_by_a_depth_rebuild(monkeypatch):
    """Scoping. A colour-only connection reads no greeting and carries no geometry,
    so a rebuild tells it nothing -- dropping it would turn a fixed silent error into
    a gratuitous live-preview flap. It must stream straight through."""
    _fake_jpeg(monkeypatch)
    _publish("stale", FakePipeline("stale", depth_scale=0.001))
    color = np.zeros((8, 8, 3), dtype=np.uint8)
    frames = SimpleNamespace(
        get_color_frame=lambda: SimpleNamespace(get_data=lambda: color),
        get_timestamp=lambda: 1.0)
    calls = []

    def fake_read_frames():
        calls.append(1)
        if len(calls) == 1:
            srv._camera_generation += 1                 # a rebuild completes
        if len(calls) > 10:
            raise _KeptStreaming("streamed on, as a colour-only client should")
        return frames

    monkeypatch.setattr(srv, "read_frames", fake_read_frames)
    server_sock, client_sock = socket.socketpair()
    try:
        client_sock.sendall(b"MODE COLOR\n")
        with pytest.raises(_KeptStreaming):
            srv.handle_client(server_sock, ("10.12.172.19", 40438))
        assert len(calls) == 11, (
            "a colour-only client was cut off by a depth rebuild it does not depend on")
    finally:
        server_sock.close()
        client_sock.close()


# --- structural guards ------------------------------------------------------

def test_the_greeting_has_exactly_one_exit_and_it_stamps_the_generation():
    """Both defects come back the moment a new client path sends a greeting of its
    own. ``greet()`` is the single door: it takes ONE coherent snapshot and hands the
    caller back the generation to watch. Pin that there is no other way out."""
    source = inspect.getsource(srv)
    sends = [line.strip() for line in source.splitlines()
             if "greeting_line(" in line and not line.strip().startswith("#")]
    assert len(sends) == 1, (
        f"the greeting is sent from more than one place, so a path can send one "
        f"without recording the camera generation it describes: {sends}")
    assert "greeting_line(" in inspect.getsource(srv.greet), (
        "the one greeting send is not greet()'s")
    assert "return snap.generation" in inspect.getsource(srv.greet), (
        "greet() no longer returns the generation its greeting describes")


def test_the_greeting_is_built_from_one_snapshot_not_from_the_globals():
    """The defect was five separate global reads. Reading them individually again --
    in make_greeting or in the telemetry loop -- reopens the same window."""
    for func in (srv.make_greeting, srv.stream_h264):
        src = inspect.getsource(func)
        assert "_camera_snapshot()" in src, (
            f"{func.__name__} does not take a coherent camera snapshot")
        for name in ("STATIC_GEOMETRY", "ACHIEVED_OPTIONS", "DEVICE_INFO"):
            assert name not in src, (
                f"{func.__name__} still reads the global {name} directly, so it can "
                f"pair it with another open's numbers")
    assert "with _camera_lock:" in inspect.getsource(srv._camera_snapshot), (
        "the snapshot is not taken under the lock every writer holds, so it is not "
        "atomic against a rebuild")
