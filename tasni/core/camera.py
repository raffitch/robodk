"""RealSense-over-TCP client (the Jetson camera server on port 1024).

Reuses the wire format and decode logic from the original macros:

    16-byte header  ``<I depth_len><I color_len><d timestamp>``
    then ``depth`` (lz4-compressed ``.npy``) + ``color`` (JPEG)

The server is unicast/synchronous, so — like the macros — we open one socket per
grab. ``turbojpeg``/``lz4`` are imported lazily with an OpenCV fallback for JPEG,
so importing this module never hard-requires the native libjpeg-turbo build.
"""
from __future__ import annotations

import socket
import struct
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from .config import CameraConfig

_HEADER = struct.Struct("<IId")  # depth_len, color_len, timestamp


@dataclass
class Frame:
    color: np.ndarray              # HxWx3 BGR
    depth: np.ndarray | None       # HxW depth (uint16/float) or None if not decoded
    timestamp: float


class CameraError(RuntimeError):
    pass


class CameraClient:
    """Thin client for one D435i streamed by the Jetson server."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self._jpeg = None

    # -- decode helpers -----------------------------------------------------
    def _decode_color(self, data: bytes) -> np.ndarray:
        try:
            if self._jpeg is None:
                from turbojpeg import TurboJPEG

                self._jpeg = TurboJPEG()
            return self._jpeg.decode(data)
        except Exception:
            import cv2

            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise CameraError("failed to decode JPEG color frame")
            return img

    @staticmethod
    def _decode_depth(data: bytes) -> np.ndarray:
        import io

        import lz4.frame as lz4f

        return np.load(io.BytesIO(lz4f.decompress(data)))

    # -- socket I/O ---------------------------------------------------------
    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            packet = sock.recv(n - len(buf))
            if not packet:
                raise CameraError("connection closed by camera server mid-frame")
            buf.extend(packet)
        return bytes(buf)

    def _read_frame(self, sock: socket.socket, with_depth: bool) -> Frame:
        """Read + decode exactly one frame from an already-connected socket.

        The server always sends depth then color (depth precedes color in the
        wire order), so we must receive the depth bytes even when we only want
        color — we just skip *decoding* them."""
        cfg = self.config
        try:
            header = self._recv_exact(sock, _HEADER.size)
            depth_len, color_len, timestamp = _HEADER.unpack(header)
            depth_raw = self._recv_exact(sock, depth_len)
            color_raw = self._recv_exact(sock, color_len)
        except socket.timeout as e:
            raise CameraError(f"camera timeout ({cfg.ip}:{cfg.port})") from e
        except OSError as e:
            raise CameraError(f"camera socket error: {e}") from e
        color = self._decode_color(color_raw)
        depth = self._decode_depth(depth_raw) if with_depth else None
        return Frame(color=color, depth=depth, timestamp=timestamp)

    @staticmethod
    def _request_color_only(sock: socket.socket) -> None:
        """Ask the server for a color-only stream (it replies depth_len=0). A
        server that predates this just ignores the byte and sends full frames —
        the decoder handles both, so this is always safe to send."""
        try:
            sock.sendall(b"C")
        except OSError:
            pass

    def grab(self, *, with_depth: bool = False, timeout: float | None = None,
             color_only: bool = False) -> Frame:
        """Connect, read one frame, close. Returns a decoded :class:`Frame`.

        One-shot: used for the authoritative gate grab and per-pose capture.
        ``timeout`` overrides the configured socket timeout. ``color_only`` asks
        the server to skip the (unused-for-calibration) depth payload. For
        continuous live preview use :meth:`stream` — re-connecting per frame is slow."""
        cfg = self.config
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(cfg.timeout_s if timeout is None else timeout)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            try:
                s.connect((cfg.ip, cfg.port))
            except socket.timeout as e:
                raise CameraError(f"camera timeout ({cfg.ip}:{cfg.port})") from e
            except OSError as e:
                raise CameraError(f"camera socket error: {e}") from e
            if color_only:
                self._request_color_only(s)
            try:
                return self._read_frame(s, with_depth)
            finally:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    @contextmanager
    def stream(self, *, timeout: float | None = None, color_only: bool = False):
        """Hold one connection open and read frames back-to-back.

        The Jetson server streams continuously over a single connection, so for
        live preview this avoids a TCP handshake + slow-start *per frame* (the
        dominant cost over the cell's Wi-Fi). ``color_only`` further asks the
        server to drop the depth payload (the bulk of the bytes), which is what
        makes the preview realtime. Yields an object with
        ``read(with_depth=False) -> Frame``. Unicast: stop any other camera user
        first (the platform stops the live preview before one-shot grabs)."""
        cfg = self.config
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(cfg.timeout_s if timeout is None else timeout)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        try:
            s.connect((cfg.ip, cfg.port))
        except socket.timeout as e:
            s.close()
            raise CameraError(f"camera timeout ({cfg.ip}:{cfg.port})") from e
        except OSError as e:
            s.close()
            raise CameraError(f"camera socket error: {e}") from e
        if color_only:
            self._request_color_only(s)
        try:
            yield _CameraStream(self, s)
        finally:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()

    def grab_color(self, *, with_depth: bool = False) -> np.ndarray:
        return self.grab(with_depth=with_depth).color


class _CameraStream:
    """A held-open camera connection; ``read()`` returns the next frame."""

    def __init__(self, client: "CameraClient", sock: socket.socket):
        self._client = client
        self._sock = sock

    def read(self, *, with_depth: bool = False) -> Frame:
        return self._client._read_frame(self._sock, with_depth)
