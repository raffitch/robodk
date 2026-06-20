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

from tasni.core.camera import CameraClient, _HEADER  # noqa: E402
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


if __name__ == "__main__":
    test_full_frame_roundtrip()
    test_color_only_means_depth_len_zero()
    test_grab_sends_mode_color_handshake()
    test_recv_exact_reassembles_across_chunks()
    test_decode_color_cv2_fallback_matches_turbojpeg_path()
    print("\nCamera wire-format round-trip tests passed.")
