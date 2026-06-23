import socket
import struct
import subprocess
import threading
import pyrealsense2 as rs
import numpy as np
import lz4.frame as lz4f
import io
import turbojpeg

port = 1024

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


def getFrames(pipeline, align, depth_filters):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth = aligned_frames.get_depth_frame()
    color = aligned_frames.get_color_frame()

    if not depth or not color:
        return None, None, None

    for filter in depth_filters:
        depth = filter.process(depth)

    depth_data = np.asanyarray(depth.get_data())
    color_data = np.asanyarray(color.get_data())

    ts = frames.get_timestamp()

    return depth_data, color_data, ts

width = 1280;
height = 720;
def openPipeline():
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.infrared, 1)
    pipeline = rs.pipeline()
    pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    return pipeline, align

def handle_client(conn, addr):
    jpeg = turbojpeg.TurboJPEG('/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0')

    # Latency: disable Nagle so each frame's bytes go out immediately instead of
    # being coalesced/delayed (Nagle + delayed-ACK adds tens of ms per frame on a
    # request/stream protocol like this). Costs nothing for our large sends.
    try:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass

    # Optional, backward-compatible handshake. Right after connecting a client may
    # send ONE line declaring the stream it wants:
    #     "MODE COLOR"        -> lightweight COLOR-ONLY (depth_len=0, no align/filter/lz4)
    #     "MODE COLOR Q60"    -> color-only AND encode JPEG at quality 60 (smaller =
    #                            fewer bytes over Wi-Fi = higher preview fps)
    #     "MODE COLOR H264"   -> color-only, hardware-NVENC H.264 byte-stream (see
    #                            stream_h264) instead of per-frame JPEG. "B<kbps>"
    #                            sets the encoder bitrate (e.g. "MODE COLOR H264 B4000").
    #     "MODE FULL"         -> the full depth+color stream
    # No line (timeout) / anything unrecognized => FULL at default quality, so
    # existing depth/scan clients are unaffected (they just connect and read). A
    # bare 'C' is also accepted. Color-only is used by the live aiming preview +
    # calibration (which never use depth); it cuts ~75% of the per-frame bytes AND
    # the Nano's align+filter CPU — the difference between a ~0.5 fps preview and a
    # realtime one. The optional Q<n> lets the *preview* trade a little image
    # quality for speed while full-res captures keep the default (high) quality.
    # H264 goes further — the Nano's dedicated hardware encoder cuts preview
    # bandwidth ~10-20x and offloads the CPU. It is lossy + inter-frame (can soften
    # ChArUco corners) so it's the live-preview path only; one-shot captures keep
    # the JPEG/lossless path (the client never requests H264 for those).
    #     "MODE BURST"        -> burst capture: buffer aligned depth+color frames in
    #                            RAM and ship them all in one transfer at the end (see
    #                            stream_burst). Keeps the robot tour from stalling on a
    #                            per-pose depth transfer over Wi-Fi.
    color_only = False
    quality = None
    codec = 'jpeg'
    burst = False
    h264_bitrate = 4000  # kbps; overridden by a "B<kbps>" handshake token
    try:
        conn.settimeout(0.5)
        req = conn.recv(64).strip().upper()
        burst = req.startswith(b'MODE BURST')
        color_only = req.startswith(b'MODE COLOR') or req == b'C'
        for tok in req.split():
            if tok == b'H264':
                codec = 'h264'
            elif tok.startswith(b'Q') and tok[1:].isdigit():
                quality = max(10, min(100, int(tok[1:])))
            elif tok.startswith(b'B') and tok[1:].isdigit():
                h264_bitrate = max(500, min(20000, int(tok[1:])))
    except (socket.timeout, OSError):
        pass
    finally:
        # Send timeout: this server is single-threaded with listen(1), so a client
        # that dies without a clean RST would otherwise leave sendall blocked
        # forever and the server would stop accepting anyone (the "NO SIGNAL"
        # wedge). With a timeout, a stuck/dead client just gets dropped and we go
        # back to accept(). Generous enough that a slow-but-alive link (a full
        # depth+color frame over slow Wi-Fi can take a few seconds) is not killed.
        conn.settimeout(10.0)
    print(f"Connection from {addr} (color_only={color_only}, codec={codec}, "
          f"quality={quality}, bitrate={h264_bitrate}, burst={burst})")

    if burst:
        # Burst capture: interactive CAP/GET/CLEAR loop on this connection; returns
        # to accept() when the client disconnects (or the buffer is done).
        stream_burst(conn, addr)
        conn.close()
        return

    if codec == 'h264':
        # Hardware H.264 path: relay the NVENC byte-stream over this connection and
        # return to accept() when the client disconnects (or the encoder dies).
        stream_h264(conn, addr, width, height, h264_bitrate)
        conn.close()
        return

    while True:
        if color_only:
            # Fast path: skip align (depth->color) AND the spatial depth filter
            # entirely — they cost the Nano ~a second per frame and we're throwing
            # depth away anyway. Just grab the raw color frame. (align leaves color
            # unchanged, so intrinsics/detection are identical to the full path.)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            timestamp = frames.get_timestamp()
            depth_compressed = b''
        else:
            depth, color, timestamp = getFrames(pipeline, align, depth_filters)
            if depth is None or color is None:
                continue
            depth_buffer = io.BytesIO()
            np.save(depth_buffer, depth)
            depth_buffer.seek(0)
            depth_compressed = lz4f.compress(depth_buffer.read())

        length_depth = struct.pack('<I', len(depth_compressed))
        data_color = (jpeg.encode(color, quality=quality) if quality is not None
                      else jpeg.encode(color))
        length_color = struct.pack('<I', len(data_color))
        ts = struct.pack('<d', timestamp)

        frame_data = length_depth + length_color + ts + depth_compressed + data_color
        try:
            conn.sendall(frame_data)
        except (ConnectionResetError, BrokenPipeError, socket.timeout) as e:
            print(f"Lost connection to {addr}: {e}")
            break

    conn.close()


def _write_all(stream, data):
    """Write every byte of ``data`` to an unbuffered (bufsize=0) pipe, which can
    accept a partial write and return a short count."""
    mv = memoryview(data)
    while mv:
        n = stream.write(mv)
        if n is None:        # only on a non-blocking fd; ours is blocking
            continue
        mv = mv[n:]


def stream_h264(conn, addr, width, height, bitrate_kbps):
    """Encode the live color stream with the Nano's hardware H.264 encoder (NVENC)
    and relay the resulting Annex-B byte-stream over ``conn``.

    Rather than depend on GStreamer Python bindings (``gi`` is only available under
    the system Python 3.6, not this server's 3.10 venv), we drive ``gst-launch-1.0``
    as a subprocess: raw BGR frames go in on its stdin (``fdsrc``), encoded H.264
    comes out on its stdout (``fdsink``). A feeder thread keeps the encoder fed with
    the newest camera frames while this thread shovels encoder output to the socket;
    decoupling the two means a slow link can't stall the capture/encode.

    Wire format here is just the raw H.264 byte-stream (no per-frame header) — the
    client feeds it straight to a streaming decoder, which finds the access-unit
    boundaries itself. Baseline profile (no B-frames) keeps latency low and the SPS/
    PPS are inlined (``insert-sps-pps``/``config-interval=-1``) so a client that
    connects mid-stream can start decoding at the next IDR.
    """
    cmd = [
        'gst-launch-1.0', '-q',
        'fdsrc', 'fd=0', '!',
        'rawvideoparse', 'use-sink-caps=false',
        f'width={width}', f'height={height}', 'format=bgr', 'framerate=30/1', '!',
        'videoconvert', '!', 'video/x-raw,format=BGRx', '!',
        'nvvidconv', '!', 'video/x-raw(memory:NVMM),format=NV12', '!',
        'nvv4l2h264enc', 'control-rate=1', f'bitrate={int(bitrate_kbps) * 1000}',
        'iframeinterval=30', 'idrinterval=30', 'insert-sps-pps=1',
        'maxperf-enable=1', 'preset-level=1', 'profile=0', '!',
        'h264parse', 'config-interval=-1', '!',
        'video/x-h264,stream-format=byte-stream,alignment=au', '!',
        'fdsink', 'fd=1',
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, bufsize=0)
    except (OSError, ValueError) as e:
        print(f"H264: cannot start gst-launch ({e}); dropping {addr}")
        return

    frame_bytes = width * height * 3  # BGR8
    stop = threading.Event()

    def feeder():
        try:
            while not stop.is_set():
                frames = pipeline.wait_for_frames()
                color = frames.get_color_frame()
                if not color:
                    continue
                buf = np.asanyarray(color.get_data())
                if buf.nbytes != frame_bytes:        # unexpected size -> skip
                    continue
                _write_all(proc.stdin, buf.tobytes())
        except (BrokenPipeError, OSError, ValueError):
            pass  # encoder gone / pipe closed -> sender loop will end too
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    feeder_thread = threading.Thread(target=feeder, name='h264-feeder', daemon=True)
    feeder_thread.start()
    print(f"H264 stream to {addr} @ {bitrate_kbps} kbps")
    try:
        while True:
            # bufsize=0 -> stdout is raw, so read() returns as soon as any bytes are
            # available (one syscall) rather than blocking to fill the buffer.
            chunk = proc.stdout.read(65536)
            if not chunk:
                break                                # encoder exited / EOF
            conn.sendall(chunk)
    except (ConnectionResetError, BrokenPipeError, socket.timeout, OSError) as e:
        print(f"Lost H264 connection to {addr}: {e}")
    finally:
        stop.set()
        try:
            proc.terminate()
        except OSError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
        feeder_thread.join(timeout=2)


def _recv_line(conn, maxlen=64):
    """Read one newline-terminated command line from ``conn`` (CAP/GET/CLEAR).

    Commands are tiny and the client sends one at a time (it waits for each reply
    before sending the next), so a byte-at-a-time read can't over-read into the next
    command or into frame data. Returns b'' if the peer closed."""
    buf = bytearray()
    while len(buf) < maxlen:
        ch = conn.recv(1)
        if not ch:
            break                 # peer closed
        if ch == b'\n':
            break
        buf.extend(ch)
    return bytes(buf)


def stream_burst(conn, addr, max_frames=64):
    """Burst capture: buffer aligned depth+color frames on the Jetson, then transfer
    them all in one burst — so the robot tour isn't stalled by a per-pose depth
    transfer over the cell's Wi-Fi (a full depth+color frame can take 6-11 s).

    Protocol on this connection (newline-terminated commands; length-prefixed replies):

        (server first sends ``BURST READY\\n`` so the client can confirm support)
        CAP    -> grab + align + filter + compress ONE frame into a RAM buffer; reply
                  ``<I idx><I thumb_len><thumb JPEG>`` (a small color thumbnail for the
                  client's live per-pose strip). thumb_len=0 signals a skip.
        GET    -> reply ``<I count>`` then each buffered frame as
                  ``<I depth_len><I color_len><d ts> + depth(lz4 .npy) + color(JPEG)``
                  (identical per-frame framing to the normal stream, so the client
                  reuses its decoder).
        CLEAR  -> drop the buffer; reply ``<I 0>``.

    The buffer is RAM-only and is ALSO dropped when the connection ends (finally), so
    an abandoned/dropped burst leaves NO data on the Jetson — no disk garbage. Between
    CAP commands the robot is moving, so the command read uses a generous timeout."""
    jpeg = turbojpeg.TurboJPEG('/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0')
    buffer = []   # list of (depth_compressed: bytes, color_jpeg: bytes, ts: float)
    try:
        conn.sendall(b'BURST READY\n')
    except OSError:
        return
    print(f"Burst session opened with {addr}")
    try:
        while True:
            # The next command may be many seconds away (the robot is moving between
            # poses); wait generously rather than dropping the connection.
            conn.settimeout(180.0)
            cmd = _recv_line(conn).strip().upper()
            if not cmd:
                break                      # peer closed

            if cmd == b'CAP':
                depth = color = None
                tries = 0
                while (depth is None or color is None) and tries < 10:
                    depth, color, _ts = getFrames(pipeline, align, depth_filters)
                    tries += 1
                if depth is None or color is None or len(buffer) >= max_frames:
                    # signal a skip (no frame / buffer full) — index still advances
                    conn.sendall(struct.pack('<I', len(buffer)) + struct.pack('<I', 0))
                    continue
                depth_buffer = io.BytesIO()
                np.save(depth_buffer, depth)
                depth_buffer.seek(0)
                depth_compressed = lz4f.compress(depth_buffer.read())
                color_jpeg = jpeg.encode(color)
                idx = len(buffer)
                buffer.append((depth_compressed, color_jpeg, _ts))
                # Small thumbnail (~4x downscale) for the live per-pose strip.
                thumb_src = np.ascontiguousarray(color[::4, ::4])
                thumb = jpeg.encode(thumb_src, quality=60)
                conn.sendall(struct.pack('<I', idx) + struct.pack('<I', len(thumb)) + thumb)

            elif cmd == b'GET':
                # One bulk transfer of all buffered frames. Send frame-by-frame (each
                # bounded like the normal path) under a generous timeout.
                conn.settimeout(120.0)
                conn.sendall(struct.pack('<I', len(buffer)))
                for depth_compressed, color_jpeg, ts in buffer:
                    frame_data = (struct.pack('<I', len(depth_compressed))
                                  + struct.pack('<I', len(color_jpeg))
                                  + struct.pack('<d', ts)
                                  + depth_compressed + color_jpeg)
                    conn.sendall(frame_data)

            elif cmd == b'CLEAR':
                buffer = []
                conn.sendall(struct.pack('<I', 0))

            # unknown commands are ignored (keeps the protocol forgiving)
    except (ConnectionResetError, BrokenPipeError, socket.timeout, OSError) as e:
        print(f"Burst connection to {addr} ended: {e}")
    finally:
        n = len(buffer)
        buffer = []                        # RAM buffer dropped — nothing left on the Jetson
        print(f"Burst session with {addr} closed; cleared {n} buffered frame(s)")


def setup_depth_filters():
    # Decimation Filter
    # decimation = rs.decimation_filter()
    # decimation.set_option(rs.option.filter_magnitude, 2)

    # Spatial Filter
    spatial = rs.spatial_filter()


    # Temporal Filter with the desired settings
    # smooth_alpha = 0.4
    # smooth_delta = 20
    # persistence_control = 7  # Valid in 1 / last 8
    #
    # temporal = rs.temporal_filter(smooth_alpha, smooth_delta, persistence_control)


    # Hole filling
    # hole_filling = rs.hole_filling_filter()

    return [spatial]

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen(1)  # max number of queued clients
    print(f'Server ip-address: {get_ip_address()}:{port}')

    while True:
        conn, addr = server_socket.accept()
        handle_client(conn, addr)

if __name__ == '__main__':
    print(f"Initiating Jetson-Realsense Wi-Fi Server with resolution {width}x{height}")
    try:
        pipeline, align = openPipeline()
        depth_filters = setup_depth_filters()

        main()
    except Exception as e:
        print(f"Unexpected error: {e}")

