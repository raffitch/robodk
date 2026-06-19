import asyncio
import struct
import cv2
import io
import numpy as np
from datetime import datetime
from threading import Thread
import lz4.frame as lz4f
from turbojpeg import TurboJPEG

server_ip_address = '10.5.5.19'  # IP address of the server
port = 1024  # Port number for communication

class CountsPerSec:
    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        self._start_time = datetime.now()
        return self

    def increment(self):
        self._num_occurrences += 1

    def countsPerSec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time

class VideoGet:
    def __init__(self, protocol):
        self.protocol = protocol
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            self.protocol.handle_frame()

    def stop(self):
        self.stopped = True

class RS2ClientProtocol(asyncio.Protocol):
    def __init__(self):
        self.transport = None
        self.buffer = bytearray()
        self.length_depth = None
        self.length_color = None
        self.timestamp = None
        self.video_getter = VideoGet(self).start()
        self.jpeg = TurboJPEG()
        # cv2.namedWindow("DepthMap", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("ColorMap", cv2.WINDOW_NORMAL)

    def data_received(self, data):
        self.buffer += data

        while True:
            if self.length_depth is None:
                if len(self.buffer) < 4:
                    break
                self.length_depth = struct.unpack('<I', self.buffer[:4])[0]
                self.buffer = self.buffer[4:]
            elif self.length_color is None:
                if len(self.buffer) < 4:
                    break
                self.length_color = struct.unpack('<I', self.buffer[:4])[0]
                self.buffer = self.buffer[4:]
            elif self.timestamp is None:
                if len(self.buffer) < 8:
                    break
                self.timestamp = struct.unpack('<d', self.buffer[:8])[0]
                self.buffer = self.buffer[8:]
            else:
                if len(self.buffer) < self.length_depth + self.length_color:
                    break
                self.handle_frame()
                self.length_depth = None
                self.length_color = None
                self.timestamp = None

    def handle_frame(self):
        # depth_compressed = io.BytesIO(self.buffer[:self.length_depth])
        # decompressed_depth_data = lz4f.decompress(depth_compressed.read())
        # depth_buffer = io.BytesIO(decompressed_depth_data)
        # depth_data = np.load(depth_buffer)

        color_compressed = self.buffer[self.length_depth:self.length_depth + self.length_color]
        color_data = self.jpeg.decode(color_compressed)

        # fixedSize = (1920, 480)
        # bigDepth = cv2.resize(depth_data, fixedSize, interpolation=cv2.INTER_NEAREST)
        # bigColor = cv2.resize(color_data, fixedSize, interpolation=cv2.INTER_LINEAR)

        # cv2.putText(bigDepth, str(self.timestamp), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 65536, 2, cv2.LINE_AA)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(bigDepth, alpha=0.03), cv2.COLORMAP_JET)
        # images = np.hstack((depth_colormap, bigColor))
        cv2.imshow("client", color_data)


        cv2.waitKey(1)
        self.buffer = self.buffer[self.length_depth + self.length_color:]

    def connection_made(self, transport):
        self.transport = transport

    def connection_lost(self, exc):
        print('The server closed the connection')
        self.video_getter.stop()
        cv2.destroyAllWindows()
        asyncio.get_event_loop().stop()

    def error_received(self, exc):
        print('Error received:', exc)

    def eof_received(self):
        print('EOF received. The server is done sending data')


async def main():
    loop = asyncio.get_running_loop()

    on_con_lost = loop.create_future()

    transport, protocol = await loop.create_connection(
        lambda: RS2ClientProtocol(),
        server_ip_address, port)

    try:
        await on_con_lost
    finally:
        transport.close()


asyncio.run(main())