import asyncio
import struct
import pyrealsense2 as rs
import numpy as np
import socket
import lz4.frame as lz4f
import io
import turbojpeg


port = 1024

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't have to be reachable
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

def openPipeline():
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.infrared, 1)
    pipeline = rs.pipeline()
    pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    return pipeline, align

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    jpeg = turbojpeg.TurboJPEG('/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0')
    addr = writer.get_extra_info('peername')
    print(f"Connection from {addr!r}")
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

            frame_data = length_depth + length_color + ts + depth_compressed + data_color

            try:
                writer.write(frame_data)
                await writer.drain()
            except ConnectionResetError:
                print(f"Lost connection to {addr!r}")
                break
    writer.close()

async def main():
    server = await asyncio.start_server(handle_client, '0.0.0.0', port)
    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')
    print(f"Actual IP address is: {get_ip_address()}")
    async with server:
        await server.serve_forever()

if __name__ == '__main__':
    print("Launching Realsense Camera Server")
    try:
        pipeline, align = openPipeline()
        depth_filter = rs.decimation_filter()
        depth_filter.set_option(rs.option.filter_magnitude, 2)
        asyncio.run(main())
    except Exception as e:
        print(f"Unexpected error: {e}")