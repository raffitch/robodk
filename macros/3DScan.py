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
import pyperclip
import pyrealsense2 as rs
import open3d as o3d
import os
import subprocess
import csv
import math
import itertools
from tempfile import TemporaryDirectory
from robodk.robolink import *

SERVER_IP_ADDRESS = '10.12.171.70'
PORT = 1024

#ROBOT CONNECTION
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
}

resolution = "1280x720"  # This can be changed as needed

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

# Get the intrinsic matrix based on the chosen resolution
color_mtx = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                      [0, color_intrinsics.fy, color_intrinsics.ppy],
                      [0, 0, 1]])

depth_mtx = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                      [0, color_intrinsics.fy, color_intrinsics.ppy],
                      [0, 0, 1]])


dist_coeff = np.array(depth_intrinsics.coeffs)
dist_coeff = dist_coeff.astype(np.float32).reshape(-1, 1)

# Extract width and height from the resolution variable
width = depth_intrinsics.width
height = depth_intrinsics.height

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

def decompress_and_resize(buffer, length_depth, length_color, depth_mtx):
    try:
        depth_compressed = buffer[:length_depth]
        decompressed_depth_data = lz4f.decompress(depth_compressed)
        depth_data = np.load(io.BytesIO(decompressed_depth_data))
        depth_data_undistorted = cv2.undistort(depth_data, depth_mtx, None)
    except RuntimeError as e:
        print(f"Error decompressing data: {e}")
        depth_data_undistorted = None

    color_compressed = buffer[length_depth:length_depth + length_color]
    color_data = jpeg.decode(color_compressed)

    return depth_data, color_data

def get_all_target_names():
    all_items = RDK.ItemList(robolink.ITEM_TYPE_TARGET)
    return sorted([item.Name() for item in all_items if item.Name().startswith('Capt')])

target_names = get_all_target_names()

def handle_frame(bigColor, target_name):

    
    if bigColor is None:
        cv2.imshow(f"EyeInHand Calibration {resolution}", bigColor)
        cv2.waitKey(1)
        return
    cv2.putText(bigColor, f'{target_name}', (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow(f"3D Scanning {resolution}", bigColor)
    cv2.resizeWindow(f"3D Scanning {resolution}", 1300, 731)
    cv2.waitKey(1)

def visit_targets_and_collect_data():
    all_depth_data = []  # List to store all the depth data
    all_color_data = []
    all_transformations = []
    # For each target
    for target_name in target_names:
        target = RDK.Item(target_name)
        if not target.Valid():
            RDK.ShowMessage(f"Target {target_name} not found!", True)
            continue

        robot.MoveJ(target)
        T = get_transformation() 
        all_transformations.append(T)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(10.0)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            try:
                s.connect((SERVER_IP_ADDRESS, PORT))
                buffer, length_depth, length_color = receive_data(s)
                bigDepth, bigColor = decompress_and_resize(buffer, length_depth, length_color, depth_mtx)
                handle_frame(bigColor, target_name)
                del buffer

                if bigDepth is not None:
                    all_depth_data.append(bigDepth)
                if bigColor is not None:
                    all_color_data.append(bigColor)
            except socket.timeout:
                RDK.ShowMessage("Socket operation timed out.", True)
            except socket.error as e:
                RDK.ShowMessage(f"Socket error: {e}", True)
            finally:
                # time.sleep(4)
                s.shutdown(socket.SHUT_RDWR)
                s.close()

    return all_depth_data, all_color_data, all_transformations

# def get_transformation():
#     camera = RDK.Item('Realsense')
#     T_TC = camera.PoseTool()
#     T_RB = robot.Pose()
#     return T_RB * T_TC

def get_transformation():
    T_RB = robot.Pose()
    return T_RB

def segment_and_subtract_plane(pcd, distance_threshold=0.007):
    # Using RANSAC plane segmentation
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=3,
                                             num_iterations=1000)
  
    # Extracting the object (non-plane)
    object_pcd = pcd.select_by_index(inliers, invert=True)
    
    # Statistical Outlier Removal
    object_pcd, _ = object_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    return object_pcd

def filter_depth_data(pcd):
    # Create bounding box:
    bounds = [[-math.inf, math.inf], [-math.inf, math.inf], [0.8, 0]]  # set the bounds
    bounding_box_points = list(itertools.product(*bounds))  # create limit points
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bounding_box_points))  # create bounding box object

    # Crop the point cloud using the bounding box:
    pcd_filtered = pcd.crop(bounding_box)
    pcd_filtered = pcd_filtered.voxel_down_sample(voxel_size=0.01)
    
    # Statistical outlier removal:
    pcd_filtered, ind_stat = pcd_filtered.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_filtered, _ = pcd_filtered.remove_radius_outlier(nb_points=16, radius=0.1)

    return pcd_filtered

def remove_small_clusters(pcd, eps=0.02, min_points=10):
    # Clustering using DBSCAN
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    # Finding the largest cluster
    max_label = labels.max()
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    labels_unique, counts = np.unique(labels, return_counts=True)
    largest_cluster_label = labels_unique[np.argmax(counts)]
    
    # Retaining only the largest cluster
    pcd.points = o3d.utility.Vector3dVector(points[labels == largest_cluster_label])
    pcd.colors = o3d.utility.Vector3dVector(colors[labels == largest_cluster_label])
    
    return pcd

def voxel_grid_fusion(point_clouds, voxel_size=0.006):
    
    merged_pcd = o3d.geometry.PointCloud()
    for pcd in point_clouds:
        downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        merged_pcd += downsampled
    return merged_pcd

def save_point_cloud(all_depth_data, all_color_data, all_transformations, cam_mtx):
    

    point_clouds = []
    point_cloud_index = 0
    accumulated_pcd = o3d.geometry.PointCloud()
    
    for depth_data, color_data, T in zip(all_depth_data, all_color_data, all_transformations): 
        depth_image = o3d.geometry.Image(depth_data)
        color_image = o3d.geometry.Image(color_data)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cam_mtx)
        pcd_filtered = filter_depth_data(pcd)
        
        T_list = [[T[i,j] for j in range(4)] for i in range(4)]
        T_list[0][3] /= 1000
        T_list[1][3] /= 1000
        T_list[2][3] /= 1000
        pcd_filtered.transform(T_list)

        if len(pcd_filtered.points) != 0:
            point_clouds.append(pcd_filtered)
        
     
    accumulated_pcd = voxel_grid_fusion(point_clouds)
    # o3d.visualization.draw_geometries([accumulated_pcd])
    # Outlier Removal
    # object_only_pcd = remove_small_clusters(segment_and_subtract_plane(accumulated_pcd))
    object_only_pcd = remove_small_clusters(accumulated_pcd)
    
    o3d.visualization.draw_geometries([object_only_pcd], width=1280, height=731, left=0, top=720)
    cv2.destroyAllWindows()
    # o3d.visualization.draw_geometries([object_only_pcd])
    # Recompute and Reorient normals for the object_only_pcd
    object_only_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
    object_only_pcd.orient_normals_consistent_tangent_plane(O3D_NORMALS_K_SIZE)

    # o3d.visualization.draw_geometries([object_only_pcd])
    # Mesh creation and importing to RoboDK
    mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(object_only_pcd, depth=O3D_MESH_POISSON_DEPTH)
    vertices_to_remove = densities < np.quantile(densities, O3D_MESH_DENSITIES_QUANTILE)
    mesh_poisson.remove_vertices_by_mask(vertices_to_remove)
    # o3d.visualization.draw_geometries([object_only_pcd, mesh_poisson] if O3D_DISPLAY_POINTS else [mesh_poisson], mesh_show_back_face=True, mesh_show_wireframe=O3D_DISPLAY_WIREFRAME, width=1300, height=731, left=0, top=670)
    o3d.visualization.draw_geometries([object_only_pcd, mesh_poisson] if O3D_DISPLAY_POINTS else [mesh_poisson], mesh_show_wireframe=O3D_DISPLAY_WIREFRAME, mesh_show_back_face=True, width=1280, height=731, left=0, top=720)
    
    with TemporaryDirectory(prefix='robodk_') as td:
        tf = td + '/mesh.obj'
        o3d.io.write_triangle_mesh(tf, mesh_poisson)
        mesh_item = RDK.AddFile(tf)
        mesh_item.setColor([1, 0.333, 0, 1])
        # mesh_item.setPose(get_transformation())
        mesh_item.setName("Mesh")
        point_cloud_index += 1
    
        # Convert Open3D point cloud to NumPy arrays
    points_np = np.asarray(object_only_pcd.points)
    normals_np = np.asarray(object_only_pcd.normals)
    subprocess.run(["wsl", "touch", "/root/points.csv"])
    subprocess.run(["wsl", "touch", "/root/normals.csv"])

    # Saving points
    with open('points.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'z'])  # Header row for points

        for point in points_np:
            writer.writerow(point.tolist())
    subprocess.run(["wsl", "mv", "points.csv", "/root/points.csv"])


    # Saving normals
    with open('normals.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['normal_x', 'normal_y', 'normal_z'])  # Header row for normals

        for normal in normals_np:
            writer.writerow(normal.tolist())
    subprocess.run(["wsl", "mv", "normals.csv", "/root/normals.csv"])

    # Call WSL script for meshing using Torch and nksr
    subprocess.run(["wsl", "python3", "/root/nksr_reconstruct.py"])  # Replace with the correct path to your WSL script

    # Deserialize the mesh result (assuming mesh is saved in OBJ format)
    mesh_poisson = o3d.io.read_triangle_mesh(r"\\wsl.localhost\Ubuntu\root\mesh_output.obj")
    o3d.visualization.draw_geometries([mesh_poisson], mesh_show_wireframe=O3D_DISPLAY_WIREFRAME, mesh_show_back_face=True, width=1280, height=731, left=0, top=720)
    
    with TemporaryDirectory(prefix='robodk_') as td:
        tf = td + '/mesh.obj'
        o3d.io.write_triangle_mesh(tf, mesh_poisson)
        mesh_item = RDK.AddFile(tf)
        mesh_item.setColor([1, 0.333, 0, 1])
        # mesh_item.setPose(get_transformation())
        mesh_item.setName("Mesh with NKSR")
        point_cloud_index += 1
    
    program = RDK.Item("myprog", ITEM_TYPE_PROGRAM)
    program.setRunType(PROGRAM_RUN_ON_ROBOT)
    program.RunCode()

def main():
    cam_mtx = o3d.camera.PinholeCameraIntrinsic(width=depth_intrinsics.width, 
                                                height=depth_intrinsics.height, 
                                                fx=depth_intrinsics.fx, 
                                                fy=depth_intrinsics.fy, 
                                                cx=depth_intrinsics.ppx, 
                                                cy=depth_intrinsics.ppy)

    color_cam_mtx = o3d.camera.PinholeCameraIntrinsic(width=color_intrinsics.width, 
                                            height=color_intrinsics.height, 
                                            fx=color_intrinsics.fx, 
                                            fy=color_intrinsics.fy, 
                                            cx=color_intrinsics.ppx, 
                                            cy=color_intrinsics.ppy)
    # Visit the targets and collect the depth data
    all_depth_data, all_color_data, all_transformations = visit_targets_and_collect_data()

    if not all_depth_data:
        RDK.ShowMessage("No depth data collected. Exiting.", True)
        return

    # Reconstruct the mesh from the collected depth data
    save_point_cloud(all_depth_data, all_color_data, all_transformations, color_cam_mtx)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
