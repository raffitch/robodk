import socket
import struct
import cv2
import numpy as np
from turbojpeg import TurboJPEG
import robolink
import robodk.robomath as rm
import time
import io
import lz4.frame as lz4f
import pyrealsense2 as rs
import open3d as o3d
import os
import tkinter as tk
from tkinter import ttk
import subprocess
import csv
import math
import itertools
from tempfile import TemporaryDirectory
from robodk.robolink import *

SERVER_IP_ADDRESS = '10.12.171.70'
PORT = 1024

# ROBOT CONNECTION
RDK = robolink.Robolink()
robot = RDK.Item('KUKA KR150 R2700')
framedraw = RDK.Item('KUKA KR150 R2700 Base')
flange = RDK.Item('Realsense')
robot.setPoseFrame(framedraw)
robot.setPoseTool(flange)
RDK.setRunMode(6)

buffer = bytearray()
jpeg = TurboJPEG()

O3D_NORMALS_K_SIZE = 100
O3D_MESH_POISSON_DEPTH = 9
O3D_MESH_DENSITIES_QUANTILE = 0.05
O3D_DISPLAY_POINTS = False
O3D_DISPLAY_WIREFRAME = True

# Intrinsics for different resolutions
color_intrinsics_data = {
    "1280x720": {
        'width': 1280,
        'height': 720,
        'ppx': 650.236450195312,
        'ppy': 366.6484375,
        'fx': 908.100036621094,
        'fy': 908.14111328125,
        'model': rs.distortion.inverse_brown_conrady,
        'coeffs': [0.0, 0.0, 0.0, 0.0, 0.0]
    }
}

resolution = "1280x720"

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

color_mtx = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                      [0, color_intrinsics.fy, color_intrinsics.ppy],
                      [0, 0, 1]])

def is_gui_available():
    return os.environ.get('DISPLAY') or os.name == 'nt' or os.name == 'mac'

def select_scanning_mode():
    root = tk.Tk()
    root.title("Select Scanning Mode")

    selected_mode = tk.StringVar(value="object")
    label = ttk.Label(root, text="Choose the scanning mode:")
    label.pack(pady=10)

    modes = ["object", "platform"]
    mode_menu = ttk.OptionMenu(root, selected_mode, *modes)
    mode_menu.pack(pady=10)

    def confirm():
        root.destroy()

    confirm_button = ttk.Button(root, text="Confirm", command=confirm)
    confirm_button.pack(pady=20)

    root.mainloop()
    return selected_mode.get()

def receive_data(sock):
    global buffer
    header = receive_exact(sock, 16)
    if len(header) == 0:
        return None, None

    length_depth = struct.unpack('<I', header[:4])[0]
    length_color = struct.unpack('<I', header[4:8])[0]
    depth_data = receive_exact(sock, length_depth)
    color_data = receive_exact(sock, length_color)

    buffer = depth_data + color_data
    return buffer, length_depth, length_color

def receive_exact(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def decompress_and_resize(buffer, length_depth, length_color, depth_mtx):
    try:
        depth_compressed = buffer[:length_depth]
        decompressed_depth_data = lz4f.decompress(depth_compressed)
        depth_data = np.load(io.BytesIO(decompressed_depth_data))
        depth_data_undistorted = cv2.undistort(depth_data, depth_mtx, None)
    except RuntimeError:
        depth_data_undistorted = None

    color_compressed = buffer[length_depth:length_depth + length_color]
    color_data = jpeg.decode(color_compressed)
    return depth_data, color_data

def segment_based_on_mode(pcd, mode, distance_threshold=0.007):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=3,
                                             num_iterations=1000)
    if mode == 'platform':
        return pcd.select_by_index(inliers, invert=False)
    elif mode == 'object':
        return pcd.select_by_index(inliers, invert=True)

def filter_pcd_by_mode(pcd, mode):
    if mode == 'object':
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    elif mode == 'platform':
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
    return pcd

def process_point_cloud(depth_data, color_data, T, mode):
    depth_image = o3d.geometry.Image(depth_data)
    color_image = o3d.geometry.Image(color_data)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(width=color_intrinsics.width, 
                                                                                                      height=color_intrinsics.height, 
                                                                                                      fx=color_intrinsics.fx, 
                                                                                                      fy=color_intrinsics.fy, 
                                                                                                      cx=color_intrinsics.ppx, 
                                                                                                      cy=color_intrinsics.ppy))
    T_list = [[T[i, j] for j in range(4)] for i in range(4)]
    T_list[0][3] /= 1000
    T_list[1][3] /= 1000
    T_list[2][3] /= 1000
    pcd.transform(T_list)
    pcd = segment_based_on_mode(pcd, mode)
    pcd = filter_pcd_by_mode(pcd, mode)
    return pcd

def visit_targets_and_collect_data(mode):
    all_pcds = []
    for target_name in RDK.ItemList(robolink.ITEM_TYPE_TARGET):
        robot.MoveJ(RDK.Item(target_name))
        T = robot.Pose()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((SERVER_IP_ADDRESS, PORT))
            buffer, length_depth, length_color = receive_data(s)
            depth_data, color_data = decompress_and_resize(buffer, length_depth, length_color, color_mtx)
            if depth_data is not None and color_data is not None:
                pcd = process_point_cloud(depth_data, color_data, T, mode)
                all_pcds.append(pcd)
    return all_pcds

def voxel_grid_fusion(point_clouds, voxel_size=0.006):
    merged_pcd = o3d.geometry.PointCloud()
    for pcd in point_clouds:
        downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        merged_pcd += downsampled
    return merged_pcd

def main():
    if not is_gui_available():
        print("GUI not available. Exiting.")
        return

    SCANNING_MODE = select_scanning_mode()
    print(f"Returned scanning mode: {SCANNING_MODE}")
    all_pcds = visit_targets_and_collect_data(SCANNING_MODE)
    accumulated_pcd = voxel_grid_fusion(all_pcds)
    o3d.visualization.draw_geometries([accumulated_pcd])

if __name__ == '__main__':
    main()
