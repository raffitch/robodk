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

    def grab(self, *, with_depth: bool = False) -> Frame:
        """Connect, read one frame, close. Returns a decoded :class:`Frame`."""
        cfg = self.config
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(cfg.timeout_s)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            try:
                s.connect((cfg.ip, cfg.port))
                header = self._recv_exact(s, _HEADER.size)
                depth_len, color_len, timestamp = _HEADER.unpack(header)
                depth_raw = self._recv_exact(s, depth_len)
                color_raw = self._recv_exact(s, color_len)
            except socket.timeout as e:
                raise CameraError(f"camera timeout ({cfg.ip}:{cfg.port})") from e
            except OSError as e:
                raise CameraError(f"camera socket error: {e}") from e
            finally:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        color = self._decode_color(color_raw)
        depth = self._decode_depth(depth_raw) if with_depth else None
        return Frame(color=color, depth=depth, timestamp=timestamp)

    def grab_color(self, *, with_depth: bool = False) -> np.ndarray:
        return self.grab(with_depth=with_depth).color
