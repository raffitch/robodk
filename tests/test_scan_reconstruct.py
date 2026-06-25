"""reconstruct.py — TSDF fusion of posed RGBD, end-to-end with plane.py.

Renders synthetic depth images of a flat square (a "table") from a few camera
poses, fuses them through the real Open3D TSDF path, then fits the work plane on
the fused cloud. Validates the whole capture->fuse->plane chain with no hardware:
the fused surface normal is +Z and its extent matches the square.

Requires open3d (`pip install -e .[scan]`); skips cleanly if it's absent.

    py -3.10 tests/test_scan_reconstruct.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core.geometry import Rt_to_T  # noqa: E402
from tasni.modules.scan import reconstruct as rc  # noqa: E402
from tasni.modules.scan.plane import work_plane_from_points  # noqa: E402

# Small synthetic camera (keeps the test fast); units are mm (RoboDK base units).
W, H = 320, 240
K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
SQUARE_HALF_MM = 150.0           # 300 x 300 mm "table" centred on the base origin


def _look_at(cam_pos, target):
    """OpenCV optical pose (X right, Y down, Z forward) looking from cam at target."""
    cam_pos = np.asarray(cam_pos, float)
    z = np.asarray(target, float) - cam_pos
    z /= np.linalg.norm(z)
    a = np.array([1.0, 0, 0]) if abs(z[2]) > 0.9 else np.array([0, 0, 1.0])
    x = np.cross(a, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return Rt_to_T(np.column_stack([x, y, z]), cam_pos)


def _render(T_base_cam):
    """Depth (uint16 mm) + flat-grey color of the z=0 square seen from this pose."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    dirs_cam = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, float)], -1)
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    dirs_base = dirs_cam @ R.T                       # rotate rays into the base frame
    dz = dirs_base[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (0.0 - t[2]) / dz                        # ray param hitting z=0; = cam-frame depth
    P = t + s[..., None] * dirs_base
    valid = ((np.abs(P[..., 0]) <= SQUARE_HALF_MM) & (np.abs(P[..., 1]) <= SQUARE_HALF_MM)
             & (s > 0) & np.isfinite(s))
    depth = np.where(valid, s, 0).astype(np.uint16)
    color = np.full((H, W, 3), 128, np.uint8)
    return color, depth


def test_fuse_and_plane_end_to_end():
    try:
        import open3d  # noqa: F401
    except Exception:
        print("[skip] open3d not installed — `pip install -e .[scan]`")
        return

    poses = [_look_at((0, 0, 500), (0, 0, 0)),
             _look_at((120, 0, 520), (0, 0, 0)),
             _look_at((0, 120, 520), (0, 0, 0))]
    views = [rc.ScanView(*_render(T), pose_T=T) for T in poses]

    res = rc.fuse_views(views, K, W, H, voxel_size_m=0.005, sdf_trunc_m=0.02,
                        depth_scale=1000.0, depth_min_m=0.2, depth_max_m=1.5)
    assert res.n_views == 3
    pts = rc.cloud_points_m(res.cloud)
    assert len(pts) > 500, f"fused cloud nearly empty ({len(pts)} pts)"
    assert len(res.mesh.vertices) > 0, "no mesh extracted"
    # The fused surface sits at base z=0 (metres).
    assert abs(float(np.median(pts[:, 2]))) < 0.01, np.median(pts[:, 2])

    wp = work_plane_from_points(pts, distance=0.006, min_inlier_frac=0.5)
    assert float(wp.normal @ [0, 0, 1]) > 0.99, wp.normal       # surface normal up
    # extent ~ 0.30 m (a little erosion at silhouette edges / voxel size is expected)
    assert 0.24 < wp.size[0] < 0.34 and 0.24 < wp.size[1] < 0.34, wp.size
    assert wp.inlier_frac > 0.8, wp.inlier_frac

    pp, cc = rc.decimate_for_preview(res.cloud, max_points=1000)
    assert len(pp) <= 1000 and pp.shape[1] == 3 and cc.shape == pp.shape
    flat_pp, flat_cc = rc.planar_surface_points(
        res.cloud, wp.normal, wp.centroid, distance_m=0.006, max_points=1000)
    assert len(flat_pp) > 0 and flat_cc.shape == flat_pp.shape
    plane_error = np.abs((flat_pp - wp.centroid) @ wp.normal)
    assert float(plane_error.max()) < 1e-6, plane_error.max()
    clean_mesh = rc.planar_rectangle_mesh(wp.corners, spacing_m=0.01)
    assert len(clean_mesh.vertices) > 4 and len(clean_mesh.triangles) > 2
    print("[fuse] views 3 ->", len(pts), "pts; size",
          tuple(round(s, 3) for s in wp.size), "m; inliers",
          f"{wp.inlier_frac:.0%}; planar preview", len(flat_pp))


if __name__ == "__main__":
    test_fuse_and_plane_end_to_end()
    print("\nreconstruct.py fusion chain test passed.")
