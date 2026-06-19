import socket
import struct
import cv2
import io
import numpy as np
import lz4.frame as lz4f
from turbojpeg import TurboJPEG
import pyrealsense2 as rs
import robolink
import sys
import robodk.robomath as rm
import time
import pyperclip
import csv
import os
SERVER_IP_ADDRESS = '10.5.5.19'  # IP address of the server
PORT = 1024  # Port number for communication

#ROBOT CONNECTION
RDK = robolink.Robolink()
robot = RDK.Item('KUKA KR150 R2700')



buffer = bytearray()

jpeg = TurboJPEG()
resolution_mtx = {
    "640x480": np.array([[605.400024414062, 0, 326.824310302734], [0, 605.427429199219, 244.43229675293], [0, 0, 1]]),
    "1280x720": np.array([[908.100036621094, 0, 650.236450195312], [0, 908.14111328125, 366.6484375], [0, 0, 1]]),
    "1920x1080": np.array([[1362.15002441406, 0, 975.354675292969], [0, 1362.21166992188, 549.97265625], [0, 0, 1]])
}

depth_mtx_data = {
    "640x480": np.array([[382.19091796875, 0, 318.838348388672], [0, 382.19091796875, 242.532287597656], [0, 0, 1]]),
    "1280x720": np.array([[636.98486328125, 0, 638.063903808594], [0, 636.98486328125, 364.200469970703], [0, 0, 1]]),
}

# Intrinsics for different resolutions
color_intrinsics_data = {
    "640x480": {
        'width': 640,
        'height': 480,
        'ppx': 326.8243103027344,
        'ppy': 244.4322967529297,
        'fx': 605.4000244140625,
        'fy': 605.4274291992188,
        'model': rs.distortion.inverse_brown_conrady,
        'coeffs': [0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "1280x720": {
        'width': 1280,
        'height': 720,
        'ppx': 650.236450195312,
        'ppy': 366.6484375,
        'fx': 908.100036621094,
        'fy': 908.14111328125,
        'model': rs.distortion.inverse_brown_conrady,
        'coeffs': [0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "1920x1080": {
        'width': 1920,
        'height': 1080,
        'ppx': 975.354675292969,
        'ppy': 549.97265625,
        'fx': 1362.15002441406,
        'fy': 1362.21166992188,
        'model': rs.distortion.inverse_brown_conrady,
        'coeffs': [0.0, 0.0, 0.0, 0.0, 0.0]
    }
}

# Intrinsics for depth resolutions
depth_intrinsics_data = {
    "640x480": {
        'width': 640,
        'height': 480,
        'ppx': 318.838348388672,
        'ppy': 242.532287597656,
        'fx': 382.19091796875,
        'fy': 382.19091796875,
        'model': rs.distortion.brown_conrady,  # Changed to Brown Conrady as per your data
        'coeffs': [0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "1280x720": {
        'width': 1280,
        'height': 720,
        'ppx': 638.063903808594,
        'ppy': 364.200469970703,
        'fx': 636.98486328125,
        'fy': 636.98486328125,
        'model': rs.distortion.brown_conrady,
        'coeffs': [0.0, 0.0, 0.0, 0.0, 0.0]
    }
    # You can add more resolutions as needed
}


resolution = "1280x720"  # This can be changed as needed

# Get the intrinsic matrix based on the chosen resolution
color_mtx = resolution_mtx[resolution]
depth_mtx = depth_mtx_data[resolution]
# Set the color_intrinsics based on the selected resolution
color_intrinsics = rs.intrinsics()
selected_intrinsics = color_intrinsics_data[resolution]
color_intrinsics.width = selected_intrinsics['width']
color_intrinsics.height = selected_intrinsics['height']
color_intrinsics.ppx = selected_intrinsics['ppx']
color_intrinsics.ppy = selected_intrinsics['ppy']
color_intrinsics.fx = selected_intrinsics['fx']
color_intrinsics.fy = selected_intrinsics['fy']
color_intrinsics.model = selected_intrinsics['model']
color_intrinsics.coeffs = selected_intrinsics['coeffs']

# Set the depth_intrinsics based on the selected resolution
depth_intrinsics = rs.intrinsics()
selected_depth_intrinsics = depth_intrinsics_data[resolution]
depth_intrinsics.width = selected_depth_intrinsics['width']
depth_intrinsics.height = selected_depth_intrinsics['height']
depth_intrinsics.ppx = selected_depth_intrinsics['ppx']
depth_intrinsics.ppy = selected_depth_intrinsics['ppy']
depth_intrinsics.fx = selected_depth_intrinsics['fx']
depth_intrinsics.fy = selected_depth_intrinsics['fy']
depth_intrinsics.model = selected_depth_intrinsics['model']
depth_intrinsics.coeffs = selected_depth_intrinsics['coeffs']


dist_coeff = np.array([0, 0, 0, 0, 0])
dist_coeff = dist_coeff.astype(np.float32).reshape(-1, 1)  # Convert to float32 and reshape to column vector

# Extract width and height from the resolution variable
width, height = map(int, resolution.split("x"))

# Create the OpenCV window
cv2.namedWindow(f"EyeInHand Calibration {resolution}", cv2.WINDOW_NORMAL)
cv2.resizeWindow(f"EyeInHand Calibration {resolution}", 1300, 731)

def receive_data(sock):
    global buffer
    global length_depth
    global length_color
    header = receive_exact(sock, 16)  # Receive exact 16 bytes for header
    if len(header) == 0:
        return None, None

    length_depth = struct.unpack('<I', header[:4])[0]
    length_color = struct.unpack('<I', header[4:8])[0]
    timestamp = struct.unpack('<d', header[8:])[0]

    depth_data = receive_exact(sock, length_depth)
    color_data = receive_exact(sock, length_color)

    buffer = depth_data + color_data

    return buffer, length_depth, length_color

def receive_exact(sock, n):
    data = bytearray()
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
            if not packet:
                print("Connection closed by the server.")
                return None
        except socket.error as e:
            print(f"Socket error while receiving data: {e}")
            return None
        data.extend(packet)
    return data

def decompress_and_resize(buffer, length_depth, length_color):
    try:
        depth_compressed = buffer[:length_depth]
        decompressed_depth_data = lz4f.decompress(depth_compressed)
        depth_data = np.load(io.BytesIO(decompressed_depth_data))
    except RuntimeError as e:
        print(f"Error decompressing data: {e}")
        depth_data = None

    color_compressed = buffer[length_depth:length_depth + length_color]
    color_data = jpeg.decode(color_compressed)

    # fixedSize = (width, height)
    # bigDepth = cv2.resize(depth_data, fixedSize, interpolation=cv2.INTER_NEAREST) if depth_data is not None else None
    # bigColor = cv2.resize(color_data, fixedSize, interpolation=cv2.INTER_LINEAR)

    return depth_data, color_data

# def handle_frame(bigDepth, bigColor, marker_3d_positions, marker_rotations, marker_translations, corners_marker):
#     camera = RDK.ItemUserPick('Select the "Eye" tool to use', RDK.ItemList(robolink.ITEM_TYPE_TOOL))
#     T_TC = camera.PoseTool()
#     T_RB = robot.Pose()
#     for position, rotation, translation, marker_corners in zip(marker_3d_positions, marker_rotations, marker_translations, corners_marker):
#         rotation_n = tuple(rotation[0])
#         rotation_n = (0, 0, 0)
#         T_CM = rm.TxyzRxyz_2_Pose(position + rotation_n)
#         T_RM = T_RB * T_TC * T_CM
#         point_item = RDK.AddPoints([T_RM.Pos()])
#         point_item.setColor([1, 1, 0, 1])


def handle_frame(bigDepth, bigColor, marker_3d_positions, marker_rotations, marker_translations, corners_marker):
    camera = RDK.ItemUserPick('Select the "Eye" tool to use', RDK.ItemList(robolink.ITEM_TYPE_TOOL))
    T_TC = camera.PoseTool()
    T_RB = robot.Pose()
    
    # Prepare a list to store all the points
    points_list = []

    for position, rotation, translation, marker_corners in zip(marker_3d_positions, marker_rotations, marker_translations, corners_marker):
        rotation_n = tuple(rotation[0])
        rotation_n = (0, 0, 0)
        T_CM = rm.TxyzRxyz_2_Pose(position + rotation_n)
        T_RM = T_RB * T_TC * T_CM
        point_item = RDK.AddPoints([T_RM.Pos()])
        point_item.setColor([1, 1, 0, 1])
        
        # Add the point to the list
        points_list.append(T_RM.Pos())

    # Save the points to a CSV file
    csv_filename = "exported_points.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['X', 'Y', 'Z'])  # Header
        csv_writer.writerows(points_list)  # Points data

    # Optional: Open the containing folder
    os.system(f'explorer /select,"{os.path.abspath(csv_filename)}"')

def process_markers(depth_data, color_data):
    marker_3d_positions = []
    marker_rotations = []
    marker_translations = []

    depth_data_undistorted = cv2.undistort(depth_data, depth_mtx, None)
    gray = cv2.cvtColor(color_data, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    corners_marker, _, _ = cv2.aruco.detectMarkers(gray, dictionary)

    if len(corners_marker) > 0:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners_marker, 0.022, depth_mtx, dist_coeff)
        for marker_corners, rvec, tvec in zip(corners_marker, rvecs, tvecs):
            point = np.average(marker_corners[0], axis=0)
            depth = depth_data_undistorted[int(point[1]), int(point[0])]

            if depth != 0:
                x, y = point
                z = depth
                xrs, yrs, zrs = rs.rs2_deproject_pixel_to_point(color_intrinsics, [x, y], z)

                marker_3d_positions.append((xrs, yrs, zrs))
                marker_rotations.append(rvec)
                marker_translations.append(tvec)

    return marker_3d_positions, marker_rotations, marker_translations, corners_marker

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(10.0)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        try:
            s.connect((SERVER_IP_ADDRESS, PORT))
            buffer, length_depth, length_color = receive_data(s)
            bigDepth, bigColor = decompress_and_resize(buffer, length_depth, length_color)

            if bigDepth is not None and bigColor is not None:
                marker_3d_positions, marker_rotations, marker_translations, corners_marker = process_markers(bigDepth, bigColor)
                handle_frame(bigDepth, bigColor, marker_3d_positions, marker_rotations, marker_translations, corners_marker)
        except socket.timeout:
            print("Socket operation timed out.")
        except socket.error as e:
            print(f"Socket error: {e}")
        finally:
            s.shutdown(socket.SHUT_RDWR)
            s.close()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()

