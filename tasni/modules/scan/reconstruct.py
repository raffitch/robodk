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


def planar_surface_points(cloud, normal: np.ndarray, centroid: np.ndarray, *,
                          distance_m: float, max_points: int = 60000
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Project dominant-plane inliers exactly onto the fitted work plane."""
    pts = np.asarray(cloud.points, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    n = np.asarray(normal, dtype=float)
    n /= np.linalg.norm(n)
    c = np.asarray(centroid, dtype=float)
    signed = (pts - c) @ n
    keep = np.abs(signed) < float(distance_m)
    pts = pts[keep] - signed[keep, None] * n

    all_colors = np.asarray(cloud.colors, dtype=np.float32)
    cols = all_colors[keep] if all_colors.size else np.full((len(pts), 3), 0.62, np.float32)
    if len(pts) > max_points:
        step = int(np.ceil(len(pts) / max_points))
        pts, cols = pts[::step], cols[::step]
    return pts.astype(np.float32), cols.astype(np.float32)


def planar_rectangle_mesh(corners: np.ndarray, *, spacing_m: float = 0.005):
    """Build a dense, perfectly flat triangle mesh over four cyclic corners."""
    import open3d as o3d

    c = np.asarray(corners, dtype=float).reshape(4, 3)
    edge_u = c[1] - c[0]
    edge_v = c[3] - c[0]
    nu = max(1, int(np.ceil(np.linalg.norm(edge_u) / max(spacing_m, 1e-4))))
    nv = max(1, int(np.ceil(np.linalg.norm(edge_v) / max(spacing_m, 1e-4))))
    us = np.linspace(0.0, 1.0, nu + 1)
    vs = np.linspace(0.0, 1.0, nv + 1)
    vertices = np.array([c[0] + u * edge_u + v * edge_v for v in vs for u in us])
    triangles = []
    row = nu + 1
    for j in range(nv):
        for i in range(nu):
            a = j * row + i
            b = a + 1
            d = a + row
            e = d + 1
            triangles.extend(((a, b, e), (a, e, d)))
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)))
    mesh.compute_vertex_normals()
    return mesh


def _mesh_from_vertex_mask(mesh, vertex_mask: np.ndarray):
    """Return a copy of ``mesh`` containing only triangles whose vertices pass."""
    import open3d as o3d

    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    if len(vertices) == 0 or len(triangles) == 0:
        return o3d.geometry.TriangleMesh()
    keep_tri = np.asarray(vertex_mask, bool)[triangles].all(axis=1)
    if not np.any(keep_tri):
        return o3d.geometry.TriangleMesh()
    old_used = np.unique(triangles[keep_tri].reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[old_used] = np.arange(len(old_used), dtype=np.int32)
    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices[old_used]),
        o3d.utility.Vector3iVector(remap[triangles[keep_tri]]))
    colors = np.asarray(mesh.vertex_colors, dtype=float)
    if len(colors) == len(vertices):
        out.vertex_colors = o3d.utility.Vector3dVector(colors[old_used])
    out.remove_duplicated_vertices()
    out.remove_duplicated_triangles()
    out.remove_degenerate_triangles()
    out.remove_unreferenced_vertices()
    if len(out.triangles):
        out.compute_vertex_normals()
    return out


def _keep_largest_component(mesh):
    """Drop disconnected islands, keeping the dominant attached surface patch."""
    if len(mesh.triangles) == 0:
        return mesh, 0, 0.0
    labels, counts, areas = mesh.cluster_connected_triangles()
    labels = np.asarray(labels, dtype=np.int32)
    counts = np.asarray(counts, dtype=np.int64)
    areas = np.asarray(areas, dtype=float)
    if len(counts) <= 1:
        return mesh, int(len(counts)), float(areas[0]) if len(areas) else 0.0
    keep_label = int(np.argmax(areas))
    mesh.remove_triangles_by_mask(labels != keep_label)
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles):
        mesh.compute_vertex_normals()
    return mesh, int(len(counts)), float(areas[keep_label])


def _view_support_counts(vertices_m: np.ndarray, views: list[ScanView], K: np.ndarray,
                         width: int, height: int, *, depth_scale: float,
                         tolerance_m: float, depth_min_m: float,
                         depth_max_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Count how many camera views actually support each mesh vertex.

    A vertex is considered supported by a view when it projects inside the image and
    the measured depth at that pixel is close to the vertex's projected camera-Z.
    This is a pragmatic confidence/probability proxy for TSDF meshes: random flying
    surfaces usually have weak multi-view support, while real surface patches are
    confirmed by repeated observations.
    """
    n = len(vertices_m)
    supported = np.zeros(n, dtype=np.uint16)
    observable = np.zeros(n, dtype=np.uint16)
    if n == 0 or not views:
        return supported, observable

    fx, fy, cx, cy = map(float, (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    verts_mm = np.asarray(vertices_m, dtype=float) * 1000.0
    tol_m = float(tolerance_m)
    for v in views:
        depth = np.asarray(v.depth)
        if depth.ndim != 2 or depth.size == 0:
            continue
        h, w = depth.shape[:2]
        T = np.asarray(v.pose_T, dtype=float).reshape(4, 4)
        R, t = T[:3, :3], T[:3, 3]
        pc_mm = (verts_mm - t) @ R
        z_m = pc_mm[:, 2] / 1000.0
        in_front = (z_m >= float(depth_min_m)) & (z_m <= float(depth_max_m))
        if not np.any(in_front):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.rint(fx * (pc_mm[:, 0] / pc_mm[:, 2]) + cx).astype(np.int32)
            vv = np.rint(fy * (pc_mm[:, 1] / pc_mm[:, 2]) + cy).astype(np.int32)
        inside = in_front & (u >= 0) & (u < min(width, w)) & (vv >= 0) & (vv < min(height, h))
        idx = np.flatnonzero(inside)
        if len(idx) == 0:
            continue
        d_m = depth[vv[idx], u[idx]].astype(float) / float(depth_scale)
        has_depth = d_m > 0.0
        if not np.any(has_depth):
            continue
        obs_idx = idx[has_depth]
        observable[obs_idx] += 1
        supported[obs_idx[np.abs(d_m[has_depth] - z_m[obs_idx]) <= tol_m]] += 1
    return supported, observable


def clean_measured_surface_mesh(mesh, views: list[ScanView], wp, K: np.ndarray,
                                width: int, height: int, *, plane_band_m: float,
                                rect_margin_m: float, support_tolerance_m: float,
                                min_support_views: int, min_support_ratio: float,
                                depth_scale: float, depth_min_m: float,
                                depth_max_m: float,
                                keep_largest_component: bool = True):
    """Extract the measured work-surface mesh from the raw TSDF mesh.

    Filtering has three stages:
      1. keep only vertices near the fitted work plane and inside the locked/fitted
         rectangle plus a small margin,
      2. keep only vertices with enough camera-depth support, using support/visible
         counts as a confidence proxy,
      3. drop disconnected islands so loose fragments not attached to the rectangle
         are not imported into RoboDK.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    stats = {
        "input_vertices": int(len(vertices)),
        "input_triangles": int(len(triangles)),
        "plane_band_mm": float(plane_band_m * 1000.0),
        "rect_margin_mm": float(rect_margin_m * 1000.0),
        "support_tolerance_mm": float(support_tolerance_m * 1000.0),
        "min_support_views": int(min_support_views),
        "min_support_ratio": float(min_support_ratio),
    }
    if len(vertices) == 0 or len(triangles) == 0:
        stats.update({"kept_vertices": 0, "kept_triangles": 0, "components": 0})
        return _mesh_from_vertex_mask(mesh, np.zeros(len(vertices), bool)), stats

    R = np.asarray(wp.frame_T[:3, :3], dtype=float)
    origin = np.asarray(wp.frame_T[:3, 3], dtype=float)
    local = (vertices - origin) @ R
    corners_local = (np.asarray(wp.corners, dtype=float) - origin) @ R
    xmin, xmax = float(corners_local[:, 0].min()), float(corners_local[:, 0].max())
    ymin, ymax = float(corners_local[:, 1].min()), float(corners_local[:, 1].max())
    geometry_mask = (
        (np.abs(local[:, 2]) <= float(plane_band_m))
        & (local[:, 0] >= xmin - float(rect_margin_m))
        & (local[:, 0] <= xmax + float(rect_margin_m))
        & (local[:, 1] >= ymin - float(rect_margin_m))
        & (local[:, 1] <= ymax + float(rect_margin_m))
    )

    supported, observable = _view_support_counts(
        vertices, views, K, width, height, depth_scale=depth_scale,
        tolerance_m=support_tolerance_m, depth_min_m=depth_min_m,
        depth_max_m=depth_max_m)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(supported, observable, out=np.zeros_like(supported, dtype=float),
                          where=observable > 0)
    confidence_mask = (
        (supported >= max(1, int(min_support_views)))
        & (ratio >= float(min_support_ratio))
    )
    combined = geometry_mask & confidence_mask
    # If the depth-support gate is too strict for a sparse edge case, fall back to
    # geometry-only instead of silently exporting no measured mesh.
    stats.update({
        "geometry_vertices": int(geometry_mask.sum()),
        "support_vertices": int((geometry_mask & confidence_mask).sum()),
        "support_mean_views": float(supported[geometry_mask].mean()) if np.any(geometry_mask) else 0.0,
        "support_mean_ratio": float(ratio[geometry_mask & (observable > 0)].mean())
        if np.any(geometry_mask & (observable > 0)) else 0.0,
        "support_fallback": False,
    })
    if int(combined.sum()) < 100:
        combined = geometry_mask
        stats["support_fallback"] = True

    out = _mesh_from_vertex_mask(mesh, combined)
    components, component_area = 0, 0.0
    if keep_largest_component and len(out.triangles):
        out, components, component_area = _keep_largest_component(out)
    stats.update({
        "components": int(components),
        "largest_component_area_mm2": float(component_area * 1_000_000.0),
        "kept_vertices": int(len(out.vertices)),
        "kept_triangles": int(len(out.triangles)),
    })
    return out, stats


def mesh_preview_points(mesh, *, max_points: int = 300000
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Use mesh vertices as the browser preview point set."""
    pts = np.asarray(mesh.vertices, dtype=np.float32)
    if len(pts) == 0:
        return pts.reshape(0, 3), np.zeros((0, 3), np.float32)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
    if len(colors) != len(pts):
        colors = np.full((len(pts), 3), (0.45, 0.72, 0.62), dtype=np.float32)
    if len(pts) > max_points:
        step = int(np.ceil(len(pts) / max_points))
        pts, colors = pts[::step], colors[::step]
    return pts, colors


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
