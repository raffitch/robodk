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


def getFrames(pipeline, align, depth_filter):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth = aligned_frames.get_depth_frame()
    color = aligned_frames.get_color_frame()


    if not depth or not color:
        return None, None, None

    depth = depth_filter.process(depth)
    depth_data = np.asanyarray(depth.get_data())
    color_data = np.asanyarray(color.get_data())

    ts = frames.get_timestamp()

    return depth_data, color_data, ts

def openPipeline(depth_width=640, depth_height=480, color_width=640, color_height=480):
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.infrared, 1)
    pipeline = rs.pipeline()
    pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    return pipeline, align

def getCameraIntrinsics(pipeline):
    # Obtain the intrinsics from the color stream profile
    profile = pipeline.get_active_profile()
    color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    # Convert intrinsics to arrays (modify this if you need more details from the intrinsics)
    color_mtx = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                          [0, color_intrinsics.fy, color_intrinsics.ppy],
                          [0, 0, 1]])
    dist_coeff = np.array(color_intrinsics.coeffs)
    dist_coeff = dist_coeff.astype(np.float32).reshape(-1, 1)  # Convert to float32 and reshape to column vector

    return color_mtx, dist_coeff

def handle_client(conn, addr):
    print(f"Connection from {addr}")
    jpeg = turbojpeg.TurboJPEG('/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0')

    resolution_command = conn.recv(64).decode('utf-8') # Assume command is within 64 bytes
    if "RESOLUTION" in resolution_command:
        width, height = map(int, resolution_command.split(':')[1].split('x'))
        print(f"Client {addr} requested resolution {width}x{height}")
        pipeline, align = openPipeline(width, height)
    else:
        pipeline, align = openPipeline()  # Use default if no valid command is received

    while True:
        depth, color, timestamp = getFrames(pipeline, align, depth_filter)
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

            # Get the intrinsics for the current pipeline
            color_mtx, dist_coeff = getCameraIntrinsics(pipeline)

            # Convert intrinsics to byte data (here using numpy's save mechanism)
            mtx_buffer = io.BytesIO()
            np.save(mtx_buffer, color_mtx)
            dist_buffer = io.BytesIO()
            np.save(dist_buffer, dist_coeff)

            intrinsics_data = mtx_buffer.getvalue() + dist_buffer.getvalue()
            intrinsics_length = struct.pack('<I', len(intrinsics_data))
            print(f"Length of depth data: {len(data_depth)}")
            print(f"Length of color data: {len(data_color)}")
            print(f"Length of intrinsics data: {len(intrinsics_data)}")
            print(f"Total Length of message (without lengths and timestamps): {len(data_depth) + len(data_color) + len(intrinsics_data)}")
            # Prepend intrinsics data length and the intrinsics data to your frame data
            frame_data = intrinsics_length + intrinsics_data + length_depth + length_color + ts + depth_compressed + data_color

            try:
                conn.sendall(frame_data)
            except (ConnectionResetError, BrokenPipeError):
                print(f"Lost connection to {addr}")
                break

    conn.close()


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen(5)  # max number of queued clients
    print(f'Serving on {get_ip_address()}:{port}')

    while True:
        conn, addr = server_socket.accept()
        handle_client(conn, addr)

if __name__ == '__main__':
    print("Launching Realsense Camera Server")
    try:
        depth_filter = rs.decimation_filter()
        depth_filter.set_option(rs.option.filter_magnitude, 2)
        main()
    except Exception as e:
        print(f"Unexpected error: {e}")

