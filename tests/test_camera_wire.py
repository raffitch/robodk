"""Wire-format round-trip between the Jetson camera server and CameraClient.

The server (`server/server_unicast_syncronous.py`) and the client
(`tasni/core/camera.py`) live in one repo and share a hand-rolled framing:

    <I depth_len><I color_len><d timestamp>  then  depth(lz4 .npy) + color(JPEG)

and a `MODE COLOR` handshake where color-only frames carry depth_len=0. That's
exactly the kind of pair that silently desyncs when one side's struct or order
changes. This test encodes a frame *the server's way* (mirroring
server_unicast_syncronous.py:107-112), feeds the bytes through an in-memory fake
socket (so it exercises the real `_recv_exact`/`_read_raw` reassembly with no OS
sockets), decodes via `CameraClient`, and asserts the pixels + depth + timestamp
survive — and that color-only really means depth_len=0.

    py -3.10 tests/test_camera_wire.py
"""
from __future__ import annotations

import io
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core.camera import CameraClient, CameraError, _HEADER  # noqa: E402
from tasni.core.config import CameraConfig  # noqa: E402


def _server_encode(color: np.ndarray, depth: np.ndarray | None,
                   timestamp: float) -> "tuple[bytes, int]":
    """Encode one frame exactly the way the Jetson server does
    (server_unicast_syncronous.py:107-112). Returns (frame_bytes, depth_len).

    color_only is modelled by passing ``depth=None`` -> depth_len=0 and no depth
    payload (the server's fast path)."""
    import cv2  # the server uses turbojpeg.encode; JPEG-is-JPEG, both decode it.
    import lz4.frame as lz4f

    if depth is None:
        depth_compressed = b""
    else:
        buf = io.BytesIO()
        np.save(buf, depth)
        buf.seek(0)
        depth_compressed = lz4f.compress(buf.read())

    ok, jpeg = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    data_color = jpeg.tobytes()

    length_depth = struct.pack("<I", len(depth_compressed))
    length_color = struct.pack("<I", len(data_color))
    ts = struct.pack("<d", timestamp)
    frame = length_depth + length_color + ts + depth_compressed + data_color
    return frame, len(depth_compressed)


class FakeSocket:
    """Serves ``buf`` through ``recv(n)`` in small chunks (so `_recv_exact` really
    has to reassemble a header/payload split across reads) and records anything
    sent (to confirm the MODE COLOR handshake)."""

    def __init__(self, buf: bytes, chunk: int = 7):
        self._buf = memoryview(buf)
        self._pos = 0
        self._chunk = chunk
        self.sent = bytearray()

    def recv(self, n: int) -> bytes:
        if self._pos >= len(self._buf):
            return b""                       # peer closed
        end = min(self._pos + min(n, self._chunk), len(self._buf))
        out = bytes(self._buf[self._pos:end])
        self._pos = end
        return out

    def sendall(self, data: bytes) -> None:
        self.sent.extend(data)

    # used by grab()/stream() lifecycle — all no-ops for the in-memory socket
    def settimeout(self, *_a): pass
    def setsockopt(self, *_a): pass
    def connect(self, *_a): pass
    def shutdown(self, *_a): pass
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _make_color(w=64, h=48) -> np.ndarray:
    """A few flat colour blocks. JPEG rings hard on the sharp block edges, so the
    round-trip is asserted on block *interiors* + the overall mean, not per-pixel."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:, : w // 2] = (200, 30, 30)      # BGR
    img[: h // 2, w // 2:] = (30, 200, 30)
    img[h // 2:, w // 2:] = (30, 30, 200)
    return img


def _assert_color_roundtrip(decoded: np.ndarray, original: np.ndarray) -> None:
    """JPEG is lossy and rings at the sharp block edges, so check the *interior* of
    each block (near-exact) and the overall mean (no gross corruption / channel
    swap), rather than every pixel."""
    assert decoded.shape == original.shape
    h, w = original.shape[:2]
    for cy, cx in ((h // 4, w // 4), (h // 4, 3 * w // 4), (3 * h // 4, 3 * w // 4)):
        got, want = decoded[cy, cx].astype(int), original[cy, cx].astype(int)
        assert np.abs(got - want).max() <= 12, f"interior pixel {(cy, cx)}: {got} vs {want}"
    assert np.abs(decoded.astype(int) - original.astype(int)).mean() <= 3.0


def test_full_frame_roundtrip():
    client = CameraClient(CameraConfig())
    color = _make_color()
    depth = (np.arange(64 * 48, dtype=np.uint16).reshape(48, 64) * 3)
    frame_bytes, depth_len = _server_encode(color, depth, 12345.678)
    assert depth_len > 0

    sock = FakeSocket(frame_bytes)
    frame = client._read_frame(sock, with_depth=True)

    _assert_color_roundtrip(frame.color, color)
    assert frame.depth is not None
    np.testing.assert_array_equal(frame.depth, depth)       # lz4+npy is lossless
    assert abs(frame.timestamp - 12345.678) < 1e-6
    print("[full] color+depth+timestamp round-trip ok")


def test_color_only_means_depth_len_zero():
    client = CameraClient(CameraConfig())
    color = _make_color()
    frame_bytes, depth_len = _server_encode(color, None, 0.5)
    # The contract: color-only frames carry depth_len=0 and no depth payload.
    assert depth_len == 0
    header = frame_bytes[: _HEADER.size]
    hdr_depth_len, hdr_color_len, ts = _HEADER.unpack(header)
    assert hdr_depth_len == 0
    assert hdr_color_len == len(frame_bytes) - _HEADER.size

    sock = FakeSocket(frame_bytes)
    frame = client._read_frame(sock, with_depth=False)
    assert frame.depth is None
    _assert_color_roundtrip(frame.color, color)
    print("[color-only] depth_len=0, no depth payload, color round-trip ok")


def test_grab_sends_mode_color_handshake():
    """grab(color_only=True) must send `MODE COLOR\\n` so the server takes the
    light path. Patch socket.socket to hand back our in-memory frame."""
    import tasni.core.camera as camera_mod

    client = CameraClient(CameraConfig())
    color = _make_color()
    frame_bytes, _ = _server_encode(color, None, 1.0)
    sock = FakeSocket(frame_bytes)

    orig = camera_mod.socket.socket
    camera_mod.socket.socket = lambda *a, **k: sock
    try:
        frame = client.grab(color_only=True)
    finally:
        camera_mod.socket.socket = orig

    assert bytes(sock.sent) == b"MODE COLOR\n"
    assert frame.depth is None
    assert frame.color.shape == color.shape
    print("[handshake] grab(color_only=True) sent MODE COLOR, depth skipped")


def _server_parse_handshake(req: bytes) -> "tuple[bool, int | None]":
    """Mirror the server's handshake parse (server_unicast_syncronous.py:69-77):
    color-only flag + optional clamped JPEG quality. Kept in lockstep with the
    server the same way _server_encode mirrors its framing."""
    req = req.strip().upper()
    color_only = req.startswith(b"MODE COLOR") or req == b"C"
    quality = None
    for tok in req.split():
        if tok.startswith(b"Q") and tok[1:].isdigit():
            quality = max(10, min(100, int(tok[1:])))
    return color_only, quality


def test_color_only_quality_handshake_string():
    """grab(color_only=True, quality=n) must append ` Q<n>`; with no quality it
    stays the bare `MODE COLOR\\n` (back-compat with servers that ignore Q)."""
    import tasni.core.camera as camera_mod

    client = CameraClient(CameraConfig())
    frame_bytes, _ = _server_encode(_make_color(), None, 1.0)

    for quality, expect in ((55, b"MODE COLOR Q55\n"), (None, b"MODE COLOR\n")):
        sock = FakeSocket(frame_bytes)
        orig = camera_mod.socket.socket
        camera_mod.socket.socket = lambda *a, **k: sock
        try:
            client.grab(color_only=True, quality=quality)
        finally:
            camera_mod.socket.socket = orig
        assert bytes(sock.sent) == expect, f"q={quality}: {bytes(sock.sent)!r}"
    print("[handshake] quality token appended only when requested")


def test_server_parses_quality_handshake():
    """The server contract: MODE COLOR [Q<n>] -> (color_only, clamped quality);
    anything else -> FULL. Guards both sides of the wire."""
    assert _server_parse_handshake(b"MODE COLOR\n") == (True, None)
    assert _server_parse_handshake(b"MODE COLOR Q60\n") == (True, 60)
    assert _server_parse_handshake(b"C") == (True, None)
    assert _server_parse_handshake(b"") == (False, None)          # -> FULL
    assert _server_parse_handshake(b"garbage") == (False, None)   # -> FULL
    assert _server_parse_handshake(b"MODE COLOR Q5\n")[1] == 10    # clamped low
    assert _server_parse_handshake(b"MODE COLOR Q999\n")[1] == 100 # clamped high
    print("[handshake] server parse: color-only flag + clamped quality")


def test_scan_h264_handshake_requests_telemetry():
    client = CameraClient(CameraConfig())
    sock = FakeSocket(b"")
    client._request_color_only(
        sock, codec="h264", bitrate=4000, scan_telemetry=True)
    assert bytes(sock.sent) == b"MODE COLOR H264 B4000 SCAN\n"
    print("[handshake] scan H264 requests compact depth telemetry")


def test_recv_exact_reassembles_across_chunks():
    """The header alone (16 bytes) must be reassembled from 7-byte recv chunks —
    a guard against a future reader that assumes one recv == one logical read."""
    client = CameraClient(CameraConfig())
    frame_bytes, _ = _server_encode(_make_color(), None, 9.0)
    sock = FakeSocket(frame_bytes, chunk=7)         # < header size, < payload
    depth_raw, color_raw, ts = client._read_raw(sock)
    assert depth_raw == b"" and len(color_raw) > 0 and ts == 9.0
    print("[recv_exact] 16-byte header reassembled from 7-byte chunks")


def test_decode_color_cv2_fallback_matches_turbojpeg_path():
    """The turbojpeg decode and the OpenCV fallback must agree (the fallback is
    what runs on a box without libjpeg-turbo)."""
    import cv2

    color = _make_color()
    ok, jpeg = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    data = jpeg.tobytes()

    client = CameraClient(CameraConfig())
    primary = client._decode_color(data)            # turbojpeg if present, else cv2

    forced = CameraClient(CameraConfig())
    forced._jpeg = object()                          # no .decode -> AttributeError -> cv2 fallback
    fallback = forced._decode_color(data)

    assert primary.shape == color.shape == fallback.shape
    assert np.abs(primary.astype(int) - fallback.astype(int)).max() <= 2
    print("[decode] turbojpeg and cv2-fallback decode agree")


# -- burst capture ----------------------------------------------------------
def _thumb_jpeg(color: np.ndarray) -> bytes:
    """A small color thumbnail, the server's way (~4x downscale -> JPEG)."""
    import cv2

    ok, jpeg = cv2.imencode(".jpg", np.ascontiguousarray(color[::4, ::4]),
                            [cv2.IMWRITE_JPEG_QUALITY, 60])
    assert ok
    return jpeg.tobytes()


class FakeBurstSocket:
    """In-memory burst *server*: interprets MODE BURST / CAP / GET / CLEAR and queues
    exactly the bytes the real ``stream_burst`` would send, so the client's
    ``_BurstSession`` exercises the real framing with no OS sockets. Kept in lockstep
    with server_unicast_syncronous.py:stream_burst the same way ``_server_encode``
    mirrors the per-frame framing."""

    def __init__(self, frames: "list[tuple[np.ndarray, np.ndarray, float]]"):
        self._frames = frames            # (color, depth, ts) to serve, in CAP order
        self._buffer: list[bytes] = []   # full per-frame bytes captured so far
        self._out = bytearray()          # bytes queued for the client to recv
        self._pos = 0
        self.cleared = False

    def sendall(self, data: bytes) -> None:
        for line in bytes(data).split(b"\n"):
            cmd = line.strip().upper()
            if not cmd:
                continue
            if cmd.startswith(b"MODE BURST"):
                self._out.extend(b"BURST READY\n")
            elif cmd == b"CAP":
                color, depth, ts = self._frames[len(self._buffer)]
                frame, _ = _server_encode(color, depth, ts)
                idx = len(self._buffer)
                self._buffer.append(frame)
                thumb = _thumb_jpeg(color)
                self._out += struct.pack("<I", idx) + struct.pack("<I", len(thumb)) + thumb
            elif cmd == b"GET":
                self._out += struct.pack("<I", len(self._buffer))
                for frame in self._buffer:
                    self._out += frame
            elif cmd == b"CLEAR":
                self._buffer = []
                self.cleared = True
                self._out += struct.pack("<I", 0)

    def recv(self, n: int) -> bytes:
        if self._pos >= len(self._out):
            return b""
        end = min(self._pos + min(n, 7), len(self._out))   # 7-byte chunks
        out = bytes(self._out[self._pos:end])
        self._pos = end
        return out

    def settimeout(self, *_a): pass
    def setsockopt(self, *_a): pass
    def connect(self, *_a): pass
    def shutdown(self, *_a): pass
    def close(self): pass


def test_burst_session_roundtrip():
    """CAP buffers a frame + returns a thumbnail; GET bursts every frame back in
    order (depth+color+ts lossless); CLEAR drops the Jetson buffer."""
    import tasni.core.camera as camera_mod

    client = CameraClient(CameraConfig())
    colors = [_make_color() for _ in range(3)]
    depths = [np.arange(64 * 48, dtype=np.uint16).reshape(48, 64) * (k + 1) for k in range(3)]
    frames_in = list(zip(colors, depths, [1.0, 2.0, 3.0]))
    sock = FakeBurstSocket(frames_in)

    orig = camera_mod.socket.socket
    camera_mod.socket.socket = lambda *a, **k: sock
    try:
        with client.burst() as bs:
            thumbs = [bs.capture() for _ in range(3)]
            frames = bs.fetch_all()
            bs.clear()
    finally:
        camera_mod.socket.socket = orig

    assert all(t for t in thumbs), "each CAP should return a thumbnail"
    assert len(frames) == 3
    for fr, (color, depth, ts) in zip(frames, frames_in):
        _assert_color_roundtrip(fr.color, color)
        np.testing.assert_array_equal(fr.depth, depth)     # lz4+npy is lossless
        assert abs(fr.timestamp - ts) < 1e-6
    assert sock.cleared, "CLEAR must drop the server buffer (no Jetson garbage)"
    print("[burst] CAP buffers+thumb, GET bursts all frames in order, CLEAR drops buffer")


def test_burst_handshake_rejected_on_old_server():
    """A pre-burst server doesn't answer with BURST READY (it would start a full
    stream), so the client must raise CameraError — the caller falls back to grab()."""
    import tasni.core.camera as camera_mod

    client = CameraClient(CameraConfig())
    frame_bytes, _ = _server_encode(_make_color(), None, 1.0)   # old server just streams
    sock = FakeSocket(frame_bytes)

    orig = camera_mod.socket.socket
    camera_mod.socket.socket = lambda *a, **k: sock
    raised = False
    try:
        with client.burst():
            pass
    except CameraError:
        raised = True
    finally:
        camera_mod.socket.socket = orig
    assert raised, "burst() must reject a server that doesn't advertise BURST READY"
    print("[burst] old server (no BURST READY) -> CameraError; client falls back")


def test_server_burst_handshake_detection():
    """Mirror the server's routing: a `MODE BURST` line selects burst mode (and is
    distinct from MODE COLOR / FULL)."""
    def detects_burst(req: bytes) -> bool:
        return req.strip().upper().startswith(b"MODE BURST")
    assert detects_burst(b"MODE BURST\n")
    assert not detects_burst(b"MODE COLOR\n")
    assert not detects_burst(b"")
    assert not detects_burst(b"garbage")
    print("[handshake] MODE BURST routes to burst mode")


if __name__ == "__main__":
    test_full_frame_roundtrip()
    test_color_only_means_depth_len_zero()
    test_grab_sends_mode_color_handshake()
    test_color_only_quality_handshake_string()
    test_server_parses_quality_handshake()
    test_scan_h264_handshake_requests_telemetry()
    test_recv_exact_reassembles_across_chunks()
    test_decode_color_cv2_fallback_matches_turbojpeg_path()
    test_burst_session_roundtrip()
    test_burst_handshake_rejected_on_old_server()
    test_server_burst_handshake_detection()
    print("\nCamera wire-format round-trip tests passed.")
