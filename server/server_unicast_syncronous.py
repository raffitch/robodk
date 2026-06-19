import socket
import struct
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

    # Optional, backward-compatible handshake. Right after connecting a client may
    # send ONE line declaring the stream it wants:
    #     "MODE COLOR"  -> lightweight COLOR-ONLY (depth_len=0, no align/filter/lz4)
    #     "MODE FULL"   -> the full depth+color stream
    # No line (timeout) / anything unrecognized => FULL, so existing depth/scan
    # clients are unaffected (they just connect and read). A bare 'C' is also
    # accepted, for the first color-only client. Color-only is used by the live
    # aiming preview + calibration (which never use depth); it cuts ~75% of the
    # per-frame bytes AND the Nano's align+filter CPU — the difference between a
    # ~0.5 fps preview and a realtime one.
    color_only = False
    try:
        conn.settimeout(0.5)
        req = conn.recv(64).strip().upper()
        color_only = req.startswith(b'MODE COLOR') or req == b'C'
    except (socket.timeout, OSError):
        pass
    finally:
        conn.settimeout(None)
    print(f"Connection from {addr} (color_only={color_only})")

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
        data_color = jpeg.encode(color)
        length_color = struct.pack('<I', len(data_color))
        ts = struct.pack('<d', timestamp)

        frame_data = length_depth + length_color + ts + depth_compressed + data_color
        try:
            conn.sendall(frame_data)
        except (ConnectionResetError, BrokenPipeError):
            print(f"Lost connection to {addr}")
            break

    conn.close()


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

