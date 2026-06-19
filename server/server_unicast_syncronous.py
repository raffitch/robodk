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
    print(f"Connection from {addr}")
    jpeg = turbojpeg.TurboJPEG('/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0')

    while True:
        depth, color, timestamp = getFrames(pipeline, align, depth_filters)
        if depth is not None and color is not None:
            depth_buffer = io.BytesIO()
            np.save(depth_buffer, depth)
            depth_buffer.seek(0)
            data_depth = depth_buffer.read()

            depth_compressed = lz4f.compress(data_depth)
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

