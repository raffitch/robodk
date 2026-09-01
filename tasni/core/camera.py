"""RealSense-over-TCP client (the Jetson camera server on port 1024).

Protocol 2 (Tasks 4-7): the per-frame wire header is unchanged --

    16-byte header  ``<I depth_len><I color_len><d timestamp>``
    then ``depth`` (lz4-compressed ``.npy``) + ``color`` (JPEG)

-- but a depth-carrying connection now opens with ``MODE FULL V2`` (or
``MODE BURST V2``), answered by ONE newline-terminated JSON *greeting* line
before any frame bytes: the depth intrinsics, the depth->colour extrinsic, and
the depth unit (0.1 mm; see :mod:`.depth_geometry`). Depth itself is NATIVE
(1280x720) and UNALIGNED to colour, so ``Frame.geometry`` carries that greeting
and every depth consumer must backproject through it rather than assume
aligned 1 mm depth. A server that refuses (still on the old protocol, or the
host never restarted) answers with an ``ERR`` line instead of the greeting,
which we turn into a loud :class:`CameraError` — the whole point being that
misreading that JSON line as a raw frame length must never just hang.
``MODE COLOR`` paths are unchanged: no V2 token is sent, no greeting is read,
``Frame.geometry`` stays ``None``.

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
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from .config import CameraConfig
from .depth_geometry import CameraGeometry

_HEADER = struct.Struct("<IId")  # depth_len, color_len, timestamp
# Protocol-2 handshake tokens. A depth-carrying connection sends one of these
# instead of nothing (the pre-V2 default); MODE COLOR is unchanged (see
# _request_color_only) since color-only frames never need the depth greeting.
_HELLO_FULL = b"MODE FULL V2\n"
_HELLO_BURST = b"MODE BURST V2\n"


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
    depth: np.ndarray | None       # HxW NATIVE depth (uint16, geometry.depth_unit_mm per step) or None
    timestamp: float
    telemetry: dict | None = None
    geometry: "CameraGeometry | None" = None   # the connection's greeting; None on colour-only paths


class CameraError(RuntimeError):
    pass


class CameraClient:
    """Thin client for one D435i streamed by the Jetson server."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self._jpeg = None
        self._host: str | None = None      # last host that actually connected
        self.geometry: CameraGeometry | None = None   # last-parsed protocol-2 greeting

    # -- host selection -----------------------------------------------------
    @property
    def active_host(self) -> str:
        """The host we are actually talking to (``config.ip`` before we know).

        The dashboard reports this rather than ``config.ip`` so a silently
        degraded direct path — a moved DHCP lease — is visible instead of the
        UI claiming a route it is not using."""
        return self._host or self.config.ip

    def _candidates(self) -> "list[str]":
        """Hosts to try, in order. A resolved host short-circuits the ladder so a
        per-pose ``grab()`` loop does not re-probe the LAN on every frame."""
        if self._host:
            return [self._host]
        ordered = [self.config.lan_ip, self.config.ip]
        return list(dict.fromkeys(h for h in ordered if h))   # dedup, keep order

    def resolve_via(self, probe) -> "tuple[str, bool]":
        """Walk the ladder with an injected ``probe(host, port) -> bool`` and cache
        the winner. Returns ``(host, reachable)``.

        This is how the *dashboard* learns the route: a capture resolves the host
        as a side effect, but the health poll must not wait for one — otherwise the
        UI reports the configured fallback until the operator happens to start a
        preview. The probe is injected so the caller owns the timeout and the
        "don't touch the camera mid-capture" rule."""
        for host in self._candidates():
            if probe(host, self.config.port):
                self._host = host
                return host, True
        if self._host:                         # cached host is gone — re-ladder
            self._host = None
            for host in self._candidates():
                if probe(host, self.config.port):
                    self._host = host
                    return host, True
        return self.active_host, False

    def _connect(self, timeout: float | None = None) -> "tuple[socket.socket, str]":
        """Connect to the first reachable candidate; cache and return it.

        The connection attempt *is* the probe — the camera server is unicast, so
        a separate reachability check would open a second connection and risk
        disturbing a capture. Non-final candidates get the short probe timeout;
        the last one gets the full budget so a slow-but-working relay is not cut
        off. A cached host that has since died is dropped and the full ladder
        re-runs once, so moving between networks recovers without a restart."""
        cfg = self.config
        full = cfg.timeout_s if timeout is None else timeout
        hosts = self._candidates()
        errors: "list[str]" = []
        for attempt, host in enumerate(hosts):
            last = attempt == len(hosts) - 1
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(full if last else min(cfg.connect_probe_timeout_s, full))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            _set_nodelay(s)
            try:
                s.connect((host, cfg.port))
            except (socket.timeout, OSError) as e:
                s.close()
                errors.append(f"{host}:{cfg.port} ({e})")
                continue
            s.settimeout(full)                 # probe budget was only for connect
            self._host = host
            return s, host
        if self._host:                         # cached host is stale — re-ladder
            self._host = None
            return self._connect(timeout)
        raise CameraError("camera unreachable: " + "; ".join(errors))

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

    @staticmethod
    def _read_line(sock: socket.socket, maxlen: int = 65536) -> bytes:
        """Read one newline-terminated line, byte-by-byte, from an
        already-connected socket. Used only for the protocol-2 greeting, which is
        a single short JSON line -- there is no framing to tell us its length in
        advance, so we cannot use ``_recv_exact``. Every caller of a depth path
        expects a :class:`CameraError` (mirrors ``_read_raw``'s wrapping below),
        so a dropped connection during the greeting (``ConnectionResetError`` /
        ``BrokenPipeError`` -- both ``OSError``, neither ``socket.timeout``) must
        not surface as a raw ``OSError``."""
        buf = bytearray()
        while len(buf) < maxlen:
            try:
                ch = sock.recv(1)
            except socket.timeout as e:
                raise CameraError("camera timeout waiting for the protocol-2 greeting") from e
            except OSError as e:
                raise CameraError(f"camera socket error while reading the greeting: {e}") from e
            if not ch:
                raise CameraError("connection closed by camera server before the greeting")
            buf.extend(ch)
            if ch == b"\n":
                return bytes(buf)
        raise CameraError("camera greeting exceeded 64 KB")

    def _read_greeting(self, sock: socket.socket) -> CameraGeometry:
        """Protocol 2: one JSON line before any frame. A refusal line means the
        server changed protocol under a host that never restarted."""
        line = self._read_line(sock)
        if line.startswith(b"ERR"):
            raise CameraError(
                f"camera server refused the depth stream: {line.decode(errors='replace').strip()} "
                "- restart the Tasni backend (it is speaking an older protocol)")
        try:
            geom = CameraGeometry.from_greeting(json.loads(line.decode("utf-8")))
        except UnicodeDecodeError as e:
            # Binary frame bytes where greeting belongs = server is still pre-protocol-2.
            # Measured live on 2026-08-29 during deploy window (host v2, server old).
            raise CameraError(
                f"camera server sent binary frame bytes where the protocol-2 greeting should be "
                f"(byte 0x{line[0]:02x} at position 0) — the server is still on the old protocol. "
                f"Deploy the new server and restart it: `py -3.10 tools/jetson_deploy.py deploy`"
            ) from e
        except (ValueError, json.JSONDecodeError) as e:
            raise CameraError(f"invalid camera greeting: {e}") from e
        self.geometry = geom
        return geom

    def check_color_size(self, geom: CameraGeometry | None) -> None:
        """Fail loudly when the live colour stream is not the size ``config.K`` is for.

        The host picks its CALIBRATED colour intrinsics by a config *string*:
        ``CameraConfig.K`` indexes ``intrinsics[resolution]`` and ``CameraConfig.size``
        parses the same string. The size the camera is ACTUALLY sending arrives
        independently, in the protocol-2 greeting (``CameraGeometry.color_size``).
        Nothing else compares them, and a registration holds both halves at once:
        :class:`~.depth_geometry.ColorRegistered` takes its canvas from the greeting
        and its pixel coordinates from the config K. A stale ``camera.resolution``
        (say 1280x720 left over while the server streams 1920x1080) therefore does
        not fail — it silently squashes every registered point into the top-left
        ~44% of the colour canvas, so polygon/centre-patch tests select the wrong
        pixels and every colour read off a registered point is the wrong one.
        Measured on a synthetic
        full-frame plane at 450 mm: u spans -186..2148 with the correct K and
        -124..1432 with the stale one, and the centre-patch population changes 2.3x.

        A wrong measurement published quietly is worse than a loud stop, so this
        raises. It runs where a protocol-2 connection is first PUT TO WORK (a frame
        produced, a burst pose buffered, a live stream opened) rather than the
        instant the greeting is parsed: merely opening and closing a connection
        never uses the intrinsics, so it is not made to care about them.
        Colour-only paths pass ``geom=None`` (they read no greeting) and are
        skipped; ``CameraGeometry.legacy_aligned`` is built by the ARCHIVE readers
        from a take's own K/size and never reaches a :class:`CameraClient`."""
        if geom is None:
            return
        got = (int(geom.color_size[0]), int(geom.color_size[1]))
        try:
            want = tuple(self.config.size)
        except (ValueError, AttributeError) as e:
            # config.size just splits the string on "x"; a malformed one would
            # otherwise surface as a bare ValueError from inside the frame loop.
            raise CameraError(
                f"camera.resolution is not a WIDTHxHEIGHT string "
                f"({self.config.resolution!r}); the camera is streaming "
                f"{got[0]}x{got[1]} — set that in tasni.config.json.") from e
        if got != want:
            raise CameraError(
                f"camera colour size mismatch: the camera is streaming "
                f"{got[0]}x{got[1]} but camera.resolution is "
                f"'{self.config.resolution}' ({want[0]}x{want[1]}), so the host "
                f"would project depth through colour intrinsics for the wrong image "
                f"size and every registered point would land at the wrong pixel. "
                f"Set \"camera\": {{\"resolution\": \"{got[0]}x{got[1]}\"}} in "
                f"tasni.config.json (and make sure camera.intrinsics has an entry "
                f"for that size), then restart the Tasni backend.")

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
            raise CameraError(f"camera timeout ({self.active_host}:{cfg.port})") from e
        except OSError as e:
            raise CameraError(f"camera socket error: {e}") from e
        return depth_raw, color_raw, timestamp

    def _read_frame(self, sock: socket.socket, with_depth: bool,
                    geometry: CameraGeometry | None = None) -> Frame:
        """Read + decode exactly one frame from an already-connected socket.
        ``geometry`` is the greeting already read for this connection (None on a
        colour-only path, which never reads one)."""
        self.check_color_size(geometry)
        depth_raw, color_raw, timestamp = self._read_raw(sock)
        color = self._decode_color(color_raw)
        depth = self._decode_depth(depth_raw) if with_depth else None
        return Frame(color=color, depth=depth, timestamp=timestamp, geometry=geometry)

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
        the server to skip the (unused-for-calibration) depth payload — and skips
        the V2 handshake entirely, since there is no depth greeting to read. For
        continuous live preview use :meth:`stream` — re-connecting per frame is slow."""
        with self._connect(timeout)[0] as s:
            geometry = None
            if color_only:
                self._request_color_only(s, quality)
            else:
                s.sendall(_HELLO_FULL)
                geometry = self._read_greeting(s)
            try:
                return self._read_frame(s, with_depth, geometry=geometry)
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
        # Resolve the host FIRST so the telemetry side-channel dials the same
        # box the frame stream is on (they must not straddle two routes).
        s, host = self._connect(timeout)
        telemetry_reader = None
        if scan_telemetry:
            telemetry_reader = _TelemetryReader(host, cfg.port, timeout_s=timeout)
        geometry = None
        if color_only or h264:                       # h264 is inherently color-only
            self._request_color_only(
                s, quality, codec=codec, bitrate=bitrate,
                scan_telemetry=scan_telemetry)
        else:
            s.sendall(_HELLO_FULL)
            # A refusal/close here happens BEFORE the try/finally below, so it
            # would otherwise leak this socket (and the telemetry side-channel)
            # on every retry of a live preview against a stale backend --
            # exactly the failure protocol 2's loud-refusal contract exists to
            # avoid. Close what we opened, then let the same CameraError through
            # unchanged.
            try:
                geometry = self._read_greeting(s)
            except Exception:
                if telemetry_reader is not None:
                    telemetry_reader.close()
                s.close()
                raise
        try:
            yield (_H264Stream(s, telemetry_reader=telemetry_reader) if h264
                   else _CameraStream(self, s, telemetry_reader=telemetry_reader, geometry=geometry))
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

    # The server refuses a SET line of this many bytes or more AND ends the
    # session (a truncated line cannot be resynced -- see the server's
    # _recv_line), so an over-long line is caught here, before a connection is
    # opened: a readable error beats a dropped session.
    SET_LINE_MAXLEN = 512

    def filter_chain(self, assignments=(), *, timeout: float | None = None) -> dict:
        """Read -- or with ``assignments``, change -- the depth filter chain.

        ``assignments`` is a sequence of ``"key=value"`` strings. **Empty means a
        bare ``SET``, which is READ-ONLY**: it returns the achieved chain without
        touching it, and is the only trustworthy way to confirm which arm of an
        A/B the device is actually on.

        Two things a caller must respect, both enforced by the server rather than
        here. A successful WRITE retires the camera generation, closing every
        session greeted before it -- so it must never be sent while a capture is
        in flight, or the take dies. And a write DIES ON RESTART: the unit file
        stays the boot truth, which is why a sweep sends an explicit restore
        between arms instead of trusting leftover state.

        Returns the server's parsed reply: ``{"ok": bool, "filters": [...],
        "filter_options": {...}}`` with ACHIEVED values, or ``ok`` False plus an
        ``error``. Never read a take's arm from what you sent -- read it from the
        archived ``filter_options``.
        """
        line = ("SET " + " ".join(assignments)).strip() if assignments else "SET"
        payload = (line + "\n").encode()
        if len(payload) >= self.SET_LINE_MAXLEN:
            raise CameraError(
                f"SET line is {len(payload)} bytes; the server refuses "
                f">= {self.SET_LINE_MAXLEN} and ENDS the session. Send fewer keys.")
        with self.burst(timeout=timeout) as session:
            return session.filter_chain(payload)

    @contextmanager
    def burst(self, *, timeout: float | None = None):
        """Open a burst-capture session (see :class:`_BurstSession`).

        At each pose the client tells the server to buffer one depth+color frame in
        RAM (a fast, near-thumbnail round-trip); after the tour all frames are pulled
        in ONE transfer and the server buffer is dropped. This keeps the robot tour
        from stalling on a per-pose depth transfer over Wi-Fi while preserving the
        exact same per-frame data the per-pose path uses (so fusion is identical).

        Negotiates support first: sends ``MODE BURST V2`` and expects
        ``BURST READY``, immediately followed by the protocol-2 greeting (same as
        :meth:`grab`/:meth:`stream` — burst frames are still depth+color and need
        the same depth intrinsics/extrinsic to interpret). A server that predates
        burst would instead start a full stream, so the ack check fails and we
        raise :class:`CameraError` — letting the caller fall back to per-pose
        :meth:`grab`. A server that predates V2 but still speaks burst would answer
        with the protocol refusal line instead of ``BURST READY``, which the same
        ack check turns into the same :class:`CameraError`. Unicast: stop any other
        camera user first."""
        s = self._connect(timeout)[0]
        ready = b""
        try:
            s.sendall(_HELLO_BURST)
            ready = self._recv_exact(s, len(b"BURST READY\n"))
        except (CameraError, OSError) as e:
            s.close()
            raise CameraError(f"burst handshake failed (old server?): {e}") from e
        if not ready.startswith(b"BURST READY"):
            s.close()
            if ready.startswith(b"ERR"):
                # A protocol-2 server refusing MODE BURST V2 (host never
                # restarted) looks identical to "no burst support" unless we
                # check for the ERR prefix here -- give the operator the same
                # actionable hint grab()/stream() give instead of the generic
                # fallback message.
                raise CameraError(
                    "camera server refused the burst stream (protocol mismatch) "
                    "- restart the Tasni backend (it is speaking an older protocol)")
            raise CameraError("camera server does not support burst capture")
        # Same leak concern as stream(): a refusal/close here happens BEFORE the
        # try/finally below, so close what we opened before letting the error
        # through unchanged.
        try:
            geometry = self._read_greeting(s)
        except Exception:
            s.close()
            raise
        try:
            yield _BurstSession(self, s, geometry)
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

    def __init__(self, client: "CameraClient", sock: socket.socket,
                 geometry: CameraGeometry | None = None):
        self._client = client
        self._sock = sock
        self._geometry = geometry

    def capture(self) -> bytes | None:
        """Tell the server to grab + buffer one depth+color frame. Returns a small
        color thumbnail (JPEG bytes) for the live per-pose strip, or ``None`` if the
        server skipped it (no valid frame, or the buffer is full)."""
        # Checked here as well as in _read_frame() so a stale camera.resolution stops
        # the tour at its FIRST pose, not after the robot has visited every one.
        self._client.check_color_size(self._geometry)
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
        return [self._client._read_frame(self._sock, with_depth=True, geometry=self._geometry)
                for _ in range(count)]

    def clear(self) -> None:
        """Drop the server's RAM buffer — delete the captured data on the Jetson."""
        self._sock.sendall(b"CLEAR\n")
        self._client._recv_exact(self._sock, 4)        # ack

    def filter_chain(self, payload: bytes) -> dict:
        """Send one already-framed ``SET`` line; return the single JSON reply.

        Unlike CAP/GET/CLEAR the reply here is a newline-terminated JSON LINE,
        not a length prefix, so it is read a byte at a time to avoid over-reading
        into whatever follows on the connection.
        """
        self._sock.sendall(payload)
        buf = bytearray()
        while not buf.endswith(b"\n"):
            chunk = self._sock.recv(1)
            if not chunk:
                raise CameraError(
                    f"camera server closed mid-SET-reply; got {bytes(buf)!r}")
            buf.extend(chunk)
        return json.loads(bytes(buf).decode())


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
                # HOST arrival time, for staleness checks. The payload's own
                # "timestamp" is time.time() ON THE JETSON, and a Nano has no RTC
                # battery: measured on the cell 2026-08-13, its clock sat 3777 s
                # behind the host, so any host-clock-minus-jetson-stamp age gate
                # discarded 100% of telemetry and froze the HUD. Staleness must be
                # judged by when the payload REACHED US, never by cross-machine
                # wall-clock arithmetic.
                payload["_received_at"] = time.time()
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
                 telemetry_reader: _TelemetryReader | None = None,
                 geometry: CameraGeometry | None = None):
        self._client = client
        self._sock = sock
        self._telemetry_reader = telemetry_reader
        self._geometry = geometry
        # read() builds its Frame directly (it owns the drain loop) and so does not
        # go through CameraClient._read_frame -- check once here, at construction,
        # so a live preview against a stale camera.resolution fails before the first
        # frame instead of once per frame. stream() constructs this INSIDE its
        # try/finally, so the socket is still closed if this raises.
        client.check_color_size(geometry)

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
        return Frame(color=color, depth=depth, timestamp=ts, telemetry=telemetry,
                     geometry=self._geometry)


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
