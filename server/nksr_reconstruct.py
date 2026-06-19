#this needs to run inside a linux vm, i ran it using wsl on windows, no docker needed but can be used as well.

import numpy as np
import torch
import nksr  # Or whatever the package actually is
import open3d as o3d

# Load points and normals from CSV
points_path = "/root/points.csv"
normals_path = "/root/normals.csv"

points = np.loadtxt(points_path, delimiter=',', skiprows=1)
normals = np.loadtxt(normals_path, delimiter=',', skiprows=1)

# Convert to torch tensor
device = torch.device("cuda:0")
input_xyz = torch.from_numpy(points).float().to(device)
input_normal = torch.from_numpy(normals).float().to(device)

# Reconstruct mesh using nksr
reconstructor = nksr.Reconstructor(device)
# field = reconstructor.reconstruct(input_xyz, input_normal, detail_level=1.0)
field = reconstructor.reconstruct(input_xyz, input_normal, voxel_size=0.0085)
# field = reconstructor.reconstruct(input_xyz, input_normal, detail_level=None, chunk_size=1)
mesh = field.extract_dual_mesh(mise_iter=1)

# Convert to open3d mesh for saving in OBJ format
try:
    vertices = np.asarray(mesh.v.cpu())  # assuming mesh.v is the vertices
    triangles = np.asarray(mesh.f.cpu())  # assuming mesh.f is the face indices

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Save mesh as OBJ
    o3d.io.write_triangle_mesh("/root/mesh_output.obj", o3d_mesh)
except AttributeError:
    print("Failed to convert to Open3D mesh.")
