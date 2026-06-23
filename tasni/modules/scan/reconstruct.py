"""TSDF fusion of posed RGBD views into one mesh — the scan's quality core.

Replaces the old ``macros/3DScan.py`` path (``voxel_grid_fusion`` = plain
concatenation, then a WSL/CSV round-trip to NKSR). A TSDF volume is a 3D weighted
average of the depth views: it **denoises while it fuses** and yields a watertight
marching-cubes mesh. Per the project's best-practices review this is "the single
biggest scan-quality improvement available" for posed RGBD — and it needs no GPU /
NKSR for a clean, dense, multi-view capture of a table or object.

Inputs are exactly what the capture job already has: per view a color + depth frame
and the **camera pose in the robot base frame** (from the *stored* hand-eye result —
the scan never re-runs calibration). Open3D works in **metres**; RoboDK poses are in
**mm**, so the camera-pose translation is converted to metres here, and the fused
geometry comes out in **base-frame metres** (the caller scales to mm for RoboDK).

open3d is imported lazily so importing this module (and the package) never requires
it — only an actual fuse does. Install with ``pip install -e .[scan]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScanView:
    """One captured viewpoint for fusion."""

    color: np.ndarray         # HxWx3 BGR uint8 (as the camera client decodes it)
    depth: np.ndarray         # HxW depth, raw units (uint16 mm for the D435i)
    pose_T: np.ndarray        # 4x4 base->camera, translation in mm (RoboDK units)


@dataclass
class FusionResult:
    mesh: object              # open3d.geometry.TriangleMesh (base-frame metres)
    cloud: object             # open3d.geometry.PointCloud (TSDF cloud, base metres)
    n_views: int


def _intrinsic(K: np.ndarray, width: int, height: int):
    import open3d as o3d

    K = np.asarray(K, dtype=float)
    return o3d.camera.PinholeCameraIntrinsic(
        width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])


def _pose_to_extrinsic_m(pose_T: np.ndarray) -> np.ndarray:
    """Open3D's ``integrate`` extrinsic is **world->camera in metres**. The view pose
    is base->camera with the translation in mm, so convert mm->m then invert."""
    from ...core.geometry import invert_T

    T = np.asarray(pose_T, dtype=float).reshape(4, 4).copy()
    T[:3, 3] /= 1000.0
    return invert_T(T)


def fuse_views(views: list[ScanView], K: np.ndarray, width: int, height: int, *,
               voxel_size_m: float = 0.004, sdf_trunc_m: float = 0.02,
               depth_scale: float = 1000.0, depth_min_m: float = 0.2,
               depth_max_m: float = 1.5) -> FusionResult:
    """Integrate every posed RGBD view into a TSDF volume and extract the mesh + cloud.

    Geometry is returned in the **robot base frame, in metres**. ``depth_scale``
    converts the raw depth to metres (1000 for uint16 mm); depth outside
    ``[depth_min_m, depth_max_m]`` is dropped (near-sensor noise / far background).
    """
    import open3d as o3d

    if not views:
        raise ValueError("no views to fuse")

    intrinsic = _intrinsic(K, width, height)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_size_m), sdf_trunc=float(sdf_trunc_m),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    min_raw = float(depth_min_m) * float(depth_scale)
    n_used = 0
    for v in views:
        color = np.asarray(v.color)
        depth = np.asarray(v.depth)
        if depth.shape[:2] != color.shape[:2]:
            raise ValueError(
                f"depth {depth.shape[:2]} and color {color.shape[:2]} differ — the "
                f"server must align depth to color (full depth+color stream)")
        # BGR (camera client) -> RGB (Open3D); drop too-near depth (keep dtype).
        rgb = np.ascontiguousarray(color[:, :, ::-1])
        d = depth.copy()
        d[d < min_raw] = 0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), o3d.geometry.Image(np.ascontiguousarray(d)),
            depth_scale=float(depth_scale), depth_trunc=float(depth_max_m),
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, _pose_to_extrinsic_m(v.pose_T))
        n_used += 1

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    cloud = volume.extract_point_cloud()
    return FusionResult(mesh=mesh, cloud=cloud, n_views=n_used)


def look_point_from_views(views: list[ScanView], *, patch_frac: float = 0.25
                          ) -> np.ndarray | None:
    """The point the camera was aimed at, in the base frame (mm) — for ROI cropping.

    For each view, the surface point on the optical axis is ``cam_pos + d * forward``
    where ``d`` is the central-patch median depth (mm) and ``forward`` is the camera
    +Z in base. Averaging across views gives the work-surface centre robustly, even
    for near-parallel top-down views (where a ray-intersection would be ill-posed).
    Returns ``None`` if no view had usable central depth.
    """
    pts: list[np.ndarray] = []
    for v in views:
        d = np.asarray(v.depth)
        if d.ndim < 2:
            continue
        h, w = d.shape[:2]
        cw, ch = max(2, int(w * patch_frac)), max(2, int(h * patch_frac))
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        valid = d[y0:y0 + ch, x0:x0 + cw]
        valid = valid[valid > 0]
        if valid.size < 10:
            continue
        T = np.asarray(v.pose_T, dtype=float)
        pts.append(T[:3, 3] + float(np.median(valid)) * T[:3, 2])
    if not pts:
        return None
    return np.mean(np.asarray(pts), axis=0)


def crop_box(geometry, center_m, *, radius_m: float, below_m: float, above_m: float):
    """Crop an Open3D mesh/cloud to an axis-aligned box around ``center_m`` (metres):
    ``±radius_m`` in X/Y, ``[-below_m, +above_m]`` in base Z (Z up). Works for both
    ``TriangleMesh`` and ``PointCloud``."""
    import open3d as o3d

    c = np.asarray(center_m, dtype=float)
    aabb = o3d.geometry.AxisAlignedBoundingBox(
        (c[0] - radius_m, c[1] - radius_m, c[2] - below_m),
        (c[0] + radius_m, c[1] + radius_m, c[2] + above_m))
    return geometry.crop(aabb)


def cloud_points_m(cloud) -> np.ndarray:
    """The fused cloud's points as an ``(N,3)`` numpy array (base-frame metres) —
    fed straight to :func:`tasni.modules.scan.plane.work_plane_from_points`."""
    return np.asarray(cloud.points, dtype=float)


def decimate_for_preview(cloud, max_points: int = 60000
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Down-sample the fused cloud to <= ``max_points`` for the browser viewer.

    Returns ``(points (N,3) float32 metres, colors (N,3) float32 in 0..1)``. Uses a
    deterministic uniform stride (no RNG) so re-renders are stable; colors default to
    grey when the cloud has none.
    """
    pts = np.asarray(cloud.points, dtype=np.float32)
    n = len(pts)
    if n == 0:
        return pts.reshape(0, 3), np.zeros((0, 3), np.float32)
    has_color = bool(np.asarray(cloud.colors).size)
    cols = (np.asarray(cloud.colors, dtype=np.float32) if has_color
            else np.full((n, 3), 0.6, np.float32))
    if n > max_points:
        step = int(np.ceil(n / max_points))
        pts, cols = pts[::step], cols[::step]
    return pts, cols


def save_mesh(mesh, path: str) -> None:
    """Write the fused mesh to ``path`` (format by extension: ``.obj`` for RoboDK
    import, ``.ply``/``.glb`` for the browser viewer)."""
    import open3d as o3d

    o3d.io.write_triangle_mesh(str(path), mesh)
