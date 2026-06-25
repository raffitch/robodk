"""RealSense-over-TCP client (the Jetson camera server on port 1024).

Reuses the wire format and decode logic from the original macros:

    16-byte header  ``<I depth_len><I color_len><d timestamp>``
    then ``depth`` (lz4-compressed ``.npy``) + ``color`` (JPEG)

The server is unicast/synchronous, so — like the macros — we open one socket per
grab. ``turbojpeg``/``lz4`` are imported lazily with an OpenCV fallback for JPEG,
so importing this module never hard-requires the native libjpeg-turbo build.
"""
from __future__ import annotations

import select
import socket
import struct
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from .config import CameraConfig

_HEADER = struct.Struct("<IId")  # depth_len, color_len, timestamp


def _set_nodelay(sock: socket.socket) -> None:
    """Disable Nagle so frame bytes flush immediately (lower latency); harmless if
    the platform lacks the option."""
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass


@dataclass
class Frame:
    color: np.ndarray              # HxWx3 BGR
    depth: np.ndarray | None       # HxW depth (uint16/float) or None if not decoded
    timestamp: float
    telemetry: dict | None = None


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

    def _read_raw(self, sock: socket.socket) -> "tuple[bytes, bytes, float]":
        """Read one frame's raw bytes (depth_raw, color_raw, timestamp) from an
        already-connected socket, without decoding. The server always sends depth
        then color, so we must receive the depth bytes even when discarding them
        (for color-only the server sends depth_len=0)."""
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
        return depth_raw, color_raw, timestamp

    def _read_frame(self, sock: socket.socket, with_depth: bool) -> Frame:
        """Read + decode exactly one frame from an already-connected socket."""
        depth_raw, color_raw, timestamp = self._read_raw(sock)
        color = self._decode_color(color_raw)
        depth = self._decode_depth(depth_raw) if with_depth else None
        return Frame(color=color, depth=depth, timestamp=timestamp)

    @staticmethod
    def _request_color_only(sock: socket.socket, quality: int | None = None,
                            codec: str = "jpeg", bitrate: int | None = None,
                            scan_telemetry: bool = False) -> None:
        """Send the color-only handshake (``MODE COLOR``).

        For the default JPEG codec, ``quality`` optionally asks the server to encode
        smaller (``MODE COLOR Q<n>``) — fewer bytes over Wi-Fi, used by the live
        preview where a little softness is fine. For ``codec="h264"`` it instead
        requests the hardware-NVENC H.264 byte-stream (``MODE COLOR H264 [B<kbps>]``),
        which cuts preview bandwidth ~10-20x; ``bitrate`` (kbps) tunes the encoder.
        Full clients send nothing — the server defaults to the full depth+color
        stream — so this is the only explicit request needed and existing depth
        clients are untouched. A server that predates the handshake falls back to
        full frames, which the JPEG decoder still handles, so sending this is always
        safe (an old server ignores the trailing tokens too)."""
        msg = b"MODE COLOR"
        if codec == "h264":
            msg += b" H264" + (f" B{int(bitrate)}".encode() if bitrate else b"")
        elif quality:
            msg += f" Q{int(quality)}".encode()
        if scan_telemetry:
            msg += b" SCAN"
        try:
            sock.sendall(msg + b"\n")
        except OSError:
            pass

    def grab(self, *, with_depth: bool = False, timeout: float | None = None,
             color_only: bool = False, quality: int | None = None) -> Frame:
        """Connect, read one frame, close. Returns a decoded :class:`Frame`.

        One-shot: used for the authoritative gate grab and per-pose capture (which
        leave ``quality`` at the server default — high — for crisp ChArUco corners).
        ``timeout`` overrides the configured socket timeout. ``color_only`` asks
        the server to skip the (unused-for-calibration) depth payload. For
        continuous live preview use :meth:`stream` — re-connecting per frame is slow."""
        cfg = self.config
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(cfg.timeout_s if timeout is None else timeout)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            _set_nodelay(s)
            try:
                s.connect((cfg.ip, cfg.port))
            except socket.timeout as e:
                raise CameraError(f"camera timeout ({cfg.ip}:{cfg.port})") from e
            except OSError as e:
                raise CameraError(f"camera socket error: {e}") from e
            if color_only:
                self._request_color_only(s, quality)
            try:
                return self._read_frame(s, with_depth)
            finally:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    @contextmanager
    def stream(self, *, timeout: float | None = None, color_only: bool = False,
               quality: int | None = None, codec: str = "jpeg",
               bitrate: int | None = None, scan_telemetry: bool = False):
        """Hold one connection open and read frames back-to-back.

        The Jetson server streams continuously over a single connection, so for
        live preview this avoids a TCP handshake + slow-start *per frame* (the
        dominant cost over the cell's Wi-Fi). ``color_only`` further asks the
        server to drop the depth payload (the bulk of the bytes), and ``quality``
        asks it to encode the JPEG smaller — together that is what makes the preview
        realtime. ``codec="h264"`` instead pulls the Nano's hardware-NVENC H.264
        stream (color-only, decoded here via PyAV; ``bitrate`` in kbps) for an even
        lighter, lower-latency preview. Yields an object with
        ``read(with_depth=False) -> Frame``. Unicast: stop any other camera user
        first (the platform stops the live preview before one-shot grabs)."""
        cfg = self.config
        h264 = codec == "h264"
        telemetry_reader = None
        if scan_telemetry:
            telemetry_reader = _TelemetryReader(cfg.ip, cfg.port, timeout_s=timeout)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(cfg.timeout_s if timeout is None else timeout)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        _set_nodelay(s)
        try:
            s.connect((cfg.ip, cfg.port))
        except socket.timeout as e:
            s.close()
            raise CameraError(f"camera timeout ({cfg.ip}:{cfg.port})") from e
        except OSError as e:
            s.close()
            raise CameraError(f"camera socket error: {e}") from e
        if color_only or h264:                       # h264 is inherently color-only
            self._request_color_only(
                s, quality, codec=codec, bitrate=bitrate,
                scan_telemetry=scan_telemetry)
        try:
            yield (_H264Stream(s, telemetry_reader=telemetry_reader) if h264
                   else _CameraStream(self, s, telemetry_reader=telemetry_reader))
        finally:
            if telemetry_reader is not None:
                telemetry_reader.close()
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()

    def grab_color(self, *, with_depth: bool = False) -> np.ndarray:
        return self.grab(with_depth=with_depth).color

    @contextmanager
    def burst(self, *, timeout: float | None = None):
        """Open a burst-capture session (see :class:`_BurstSession`).

        At each pose the client tells the server to buffer one depth+color frame in
        RAM (a fast, near-thumbnail round-trip); after the tour all frames are pulled
        in ONE transfer and the server buffer is dropped. This keeps the robot tour
        from stalling on a per-pose depth transfer over Wi-Fi while preserving the
        exact same per-frame data the per-pose path uses (so fusion is identical).

        Negotiates support first: sends ``MODE BURST`` and expects ``BURST READY``.
        A server that predates burst would instead start a full stream, so the ack
        check fails and we raise :class:`CameraError` — letting the caller fall back
        to per-pose :meth:`grab`. Unicast: stop any other camera user first."""
        cfg = self.config
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(cfg.timeout_s if timeout is None else timeout)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        _set_nodelay(s)
        try:
            s.connect((cfg.ip, cfg.port))
        except (socket.timeout, OSError) as e:
            s.close()
            raise CameraError(f"camera socket error: {e}") from e
        ready = b""
        try:
            s.sendall(b"MODE BURST\n")
            ready = self._recv_exact(s, len(b"BURST READY\n"))
        except (CameraError, OSError) as e:
            s.close()
            raise CameraError(f"burst handshake failed (old server?): {e}") from e
        if not ready.startswith(b"BURST READY"):
            s.close()
            raise CameraError("camera server does not support burst capture")
        try:
            yield _BurstSession(self, s)
        finally:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()


class _BurstSession:
    """An open burst connection. The server buffers frames in RAM and ships them on
    :meth:`fetch_all`; :meth:`clear` drops that buffer (so nothing is left on the
    Jetson). Commands are newline-terminated; replies are length-prefixed."""

    def __init__(self, client: "CameraClient", sock: socket.socket):
        self._client = client
        self._sock = sock

    def capture(self) -> bytes | None:
        """Tell the server to grab + buffer one depth+color frame. Returns a small
        color thumbnail (JPEG bytes) for the live per-pose strip, or ``None`` if the
        server skipped it (no valid frame, or the buffer is full)."""
        self._sock.sendall(b"CAP\n")
        _idx = struct.unpack("<I", self._client._recv_exact(self._sock, 4))[0]
        thumb_len = struct.unpack("<I", self._client._recv_exact(self._sock, 4))[0]
        if thumb_len == 0:
            return None
        return self._client._recv_exact(self._sock, thumb_len)

    def fetch_all(self) -> "list[Frame]":
        """Pull every buffered frame in one burst, in capture order (the per-frame
        framing is identical to the normal stream, so decode is shared)."""
        self._sock.sendall(b"GET\n")
        count = struct.unpack("<I", self._client._recv_exact(self._sock, 4))[0]
        return [self._client._read_frame(self._sock, with_depth=True)
                for _ in range(count)]

    def clear(self) -> None:
        """Drop the server's RAM buffer — delete the captured data on the Jetson."""
        self._sock.sendall(b"CLEAR\n")
        self._client._recv_exact(self._sock, 4)        # ack


class _TelemetryReader:
    """Background reader for compact depth-plane JSON on a second TCP channel."""

    def __init__(self, host: str, port: int, timeout_s: float | None = None):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(10.0 if timeout_s is None else timeout_s)
        try:
            self._sock.connect((host, port))
            self._sock.sendall(b"MODE TELEMETRY\n")
        except (socket.timeout, OSError) as e:
            self._sock.close()
            raise CameraError(f"scan telemetry connection failed: {e}") from e
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="scan-telemetry",
                                        daemon=True)
        self._thread.start()

    def _loop(self):
        try:
            while not self._stop.is_set():
                raw_len = CameraClient._recv_exact(self._sock, 4)
                n = struct.unpack("<I", raw_len)[0]
                if n <= 0 or n > 65536:
                    raise CameraError(f"invalid telemetry length {n}")
                payload = json.loads(
                    CameraClient._recv_exact(self._sock, n).decode("utf-8"))
                with self._lock:
                    self._latest = payload
        except (CameraError, OSError, socket.timeout, ValueError, json.JSONDecodeError):
            pass

    def latest(self):
        with self._lock:
            return dict(self._latest) if self._latest is not None else None

    def close(self):
        self._stop.set()
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()
        self._thread.join(timeout=1.0)


class _CameraStream:
    """A held-open camera connection; ``read()`` returns the next frame."""

    def __init__(self, client: "CameraClient", sock: socket.socket,
                 telemetry_reader: _TelemetryReader | None = None):
        self._client = client
        self._sock = sock
        self._telemetry_reader = telemetry_reader

    def read(self, *, with_depth: bool = False, drain: bool = False) -> Frame:
        """Return the next frame. With ``drain=True``, first skip any frames
        already buffered in the socket (reading their bytes but not decoding) and
        return only the newest — this keeps the live preview at the live edge
        instead of falling behind when the producer outruns the consumer."""
        raw = self._client._read_raw(self._sock)
        if drain:
            for _ in range(64):     # safety cap; normally drains a few frames
                ready, _, _ = select.select([self._sock], [], [], 0)
                if not ready:
                    break
                raw = self._client._read_raw(self._sock)
        depth_raw, color_raw, ts = raw
        color = self._client._decode_color(color_raw)
        depth = self._client._decode_depth(depth_raw) if with_depth else None
        telemetry = (self._telemetry_reader.latest()
                     if self._telemetry_reader is not None else None)
        return Frame(color=color, depth=depth, timestamp=ts, telemetry=telemetry)


class _H264Stream:
    """A held-open connection to the server's hardware-NVENC H.264 byte-stream.

    Unlike :class:`_CameraStream` there is no per-frame framing: the server relays a
    continuous Annex-B byte-stream, so we feed raw socket bytes to a PyAV decoder and
    pull decoded frames out. PyAV's parser finds the access-unit boundaries, buffers
    partial NAL units, and emits frames as they complete. H.264 here is color-only
    (the preview/aiming path); depth is never available, so ``read`` returns
    ``Frame.depth = None``. PyAV is an optional dependency — if it is missing we raise
    a clear :class:`CameraError` pointing at the JPEG fallback (``preview_codec``)."""

    def __init__(self, sock: socket.socket,
                 telemetry_reader: _TelemetryReader | None = None):
        try:
            import av  # noqa: F401  (optional dependency)
        except Exception as e:  # noqa: BLE001
            raise CameraError(
                "H.264 preview needs PyAV — `pip install av`, or set "
                "calibration.preview_codec='jpeg' to use the JPEG path") from e
        self._av = av
        self._sock = sock
        self._telemetry_reader = telemetry_reader
        self._codec = av.codec.CodecContext.create("h264", "r")
        self._pending: list = []   # decoded frames not yet returned to the caller

    def _recv(self) -> bytes:
        try:
            data = self._sock.recv(65536)
        except socket.timeout as e:
            raise CameraError("camera timeout (h264 stream)") from e
        except OSError as e:
            raise CameraError(f"camera socket error: {e}") from e
        if not data:
            raise CameraError("connection closed by camera server")
        return data

    def _feed(self, data: bytes) -> None:
        # parse() chunks the byte-stream into packets; decode() yields 0+ frames.
        # Transient errors before the first IDR (or on a dropped packet) are
        # expected — swallow them and keep reading rather than killing the preview.
        try:
            for packet in self._codec.parse(data):
                self._pending.extend(self._codec.decode(packet))
        except self._av.error.AVError:
            pass

    def read(self, *, with_depth: bool = False, drain: bool = False) -> Frame:
        """Return the next decoded frame. With ``drain=True``, consume every byte
        already available on the socket and return only the newest decoded frame —
        keeping the live preview at the live edge (mirrors
        :meth:`_CameraStream.read`)."""
        while not self._pending:
            self._feed(self._recv())
        if drain:
            while True:
                ready, _, _ = select.select([self._sock], [], [], 0)
                if not ready:
                    break
                self._feed(self._recv())
            frame = self._pending[-1]
            self._pending.clear()
        else:
            frame = self._pending.pop(0)
        color = frame.to_ndarray(format="bgr24")
        telemetry = (self._telemetry_reader.latest()
                     if self._telemetry_reader is not None else None)
        return Frame(color=color, depth=None, timestamp=0.0, telemetry=telemetry)
