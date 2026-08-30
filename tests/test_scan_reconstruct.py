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
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasni.core.geometry import Rt_to_T  # noqa: E402
from tasni.modules.scan import reconstruct as rc  # noqa: E402
from tasni.modules.scan.plane import work_plane_from_points  # noqa: E402

import geometry_fixtures as gf  # noqa: E402

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
    pytest.importorskip("open3d", reason="open3d not installed — `pip install -e .[scan]`")

    poses = [_look_at((0, 0, 500), (0, 0, 0)),
             _look_at((120, 0, 520), (0, 0, 0)),
             _look_at((0, 120, 520), (0, 0, 0))]
    views = [rc.ScanView(*_render(T), pose_T=T, geometry=gf.aligned(K, (W, H)))
             for T in poses]

    res = rc.fuse_views(views, voxel_size_m=0.005, sdf_trunc_m=0.02,
                        depth_min_m=0.2, depth_max_m=1.5)
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


def test_fuse_handles_0_1mm_native_units():
    """R24: fuse_views must scale by the VIEW'S OWN geometry.depth_unit_mm, not a
    hard-coded 1000 (uint16-mm) constant -- protocol 2 streams native 0.1 mm raw
    depth. Re-quantizing the SAME synthetic scene at 0.1 mm units (10x the raw
    integer value, same real-world mm) must fuse to the same plane."""
    pytest.importorskip("open3d", reason="open3d not installed — `pip install -e .[scan]`")

    poses = [_look_at((0, 0, 500), (0, 0, 0)),
             _look_at((120, 0, 520), (0, 0, 0)),
             _look_at((0, 120, 520), (0, 0, 0))]
    geom = gf.aligned(K, (W, H), depth_unit_mm=0.1)
    views = []
    for T in poses:
        color, depth = _render(T)
        depth10 = (depth.astype(np.uint32) * 10).astype(np.uint16)
        views.append(rc.ScanView(color, depth10, pose_T=T, geometry=geom))

    res = rc.fuse_views(views, voxel_size_m=0.005, sdf_trunc_m=0.02,
                        depth_min_m=0.2, depth_max_m=1.5)
    assert res.n_views == 3
    pts = rc.cloud_points_m(res.cloud)
    assert len(pts) > 500, f"fused cloud nearly empty ({len(pts)} pts)"

    wp = work_plane_from_points(pts, distance=0.006, min_inlier_frac=0.5)
    normal_err_deg = float(np.degrees(np.arccos(np.clip(wp.normal @ [0, 0, 1], -1.0, 1.0))))
    assert normal_err_deg < 1.0, normal_err_deg          # plane normal +Z within 1 deg
    expect_m = 2.0 * SQUARE_HALF_MM / 1000.0
    assert abs(wp.size[0] - expect_m) / expect_m < 0.05, wp.size   # extent within 5%
    assert abs(wp.size[1] - expect_m) / expect_m < 0.05, wp.size
    print("[fuse 0.1mm] native 0.1mm-unit view fuses to the same plane, normal err",
          round(normal_err_deg, 3), "deg; size", tuple(round(s, 3) for s in wp.size))


def test_measured_mesh_cleaner_drops_disconnected_island():
    o3d = pytest.importorskip("open3d",
                              reason="open3d not installed — `pip install -e .[scan]`")

    surface = np.array([
        [-0.15, -0.15, 0.0], [0.15, -0.15, 0.0],
        [0.15, 0.15, 0.0], [-0.15, 0.15, 0.0],
        [0.05, 0.05, 0.0], [0.08, 0.05, 0.0], [0.05, 0.08, 0.0],
    ], dtype=float)
    triangles = np.array([
        [0, 1, 2], [0, 2, 3],     # main work surface
        [4, 5, 6],                # disconnected island inside the rectangle
    ], dtype=np.int32)
    raw = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(surface),
        o3d.utility.Vector3iVector(triangles))
    raw.compute_vertex_normals()

    wp = work_plane_from_points(surface[:4], distance=0.002, min_inlier_frac=0.9)
    cleaned, stats = rc.clean_measured_surface_mesh(
        raw, [], wp,
        plane_band_m=0.01,
        rect_margin_m=0.0,
        support_tolerance_m=0.005,
        min_support_views=2,
        min_support_ratio=0.35,
        min_normal_dot=0.35,
        depth_min_m=0.2,
        depth_max_m=1.5,
        keep_largest_component=True,
        project_to_plane=True,
        neutral_color=True)
    assert stats["support_fallback"] is True
    assert stats["components"] == 2, stats
    assert len(cleaned.triangles) == 2, len(cleaned.triangles)
    assert len(cleaned.vertices) == 4, len(cleaned.vertices)
    print("[clean mesh] disconnected island dropped;",
          len(cleaned.vertices), "verts", len(cleaned.triangles), "tris")


if __name__ == "__main__":
    test_fuse_and_plane_end_to_end()
    test_fuse_handles_0_1mm_native_units()
    test_measured_mesh_cleaner_drops_disconnected_island()
    print("\nreconstruct.py fusion chain test passed.")


# -- 2026-08-30 cell failure: MemoryError: bad allocation after a 16-pose tour ----
# fuse_views integrated the WHOLE ROOM at the work-surface voxel and the ROI crop
# then discarded 22,613,100 of 22,821,703 surface voxels (99.1%). Open3D charges
# 192 KB per touched 16^3 volume unit -- 48 bytes per voxel, because a
# ScalableTSDFVolume keeps an Eigen::Vector3d colour per voxel even under NoColor --
# so surface area / voxel^2 sets the bill and a room at 1.2 mm runs to tens of GB.

FLOOR_Z_MM = -750.0          # a far floor plane: the "room" the ROI exists to drop


def _render_table_and_floor(T_base_cam):
    """Depth of the z=0 table square where it is hit, else the far floor plane."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    dirs_cam = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, float)], -1)
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    dirs_base = dirs_cam @ R.T
    dz = dirs_base[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s_tab = (0.0 - t[2]) / dz
        s_flr = (FLOOR_Z_MM - t[2]) / dz
    P = t + s_tab[..., None] * dirs_base
    on_table = ((np.abs(P[..., 0]) <= SQUARE_HALF_MM) & (np.abs(P[..., 1]) <= SQUARE_HALF_MM)
                & (s_tab > 0) & np.isfinite(s_tab))
    s = np.where(on_table, s_tab, s_flr)
    ok = on_table | ((s_flr > 0) & np.isfinite(s_flr))
    depth = np.where(ok, np.clip(s, 0, 65000), 0).astype(np.uint16)
    return np.full((H, W, 3), 128, np.uint8), depth


def test_fusion_roi_prefilter_drops_the_room_but_not_the_crop():
    """Masking the ROI out of the depth BEFORE integration must collapse what the
    TSDF holds while leaving the cropped result untouched -- the crop discarded that
    geometry anyway, so paying to integrate it was pure waste (and the OOM)."""
    pytest.importorskip("open3d", reason="open3d not installed — `pip install -e .[scan]`")
    from tasni.core.config import AppConfig
    from tasni.modules.scan import service as scan_service

    scfg = AppConfig().scan
    poses = [_look_at((0, 0, 500), (0, 0, 0)),
             _look_at((120, 0, 520), (0, 0, 0)),
             _look_at((0, 120, 520), (0, 0, 0))]
    views = [rc.ScanView(*_render_table_and_floor(T), pose_T=T,
                         geometry=gf.aligned(K, (W, H))) for T in poses]

    center_mm = rc.look_point_from_views(views)
    assert center_mm is not None and abs(center_mm[2]) < 20.0, center_mm
    box = scan_service._fusion_roi_box(scfg, center_mm)
    assert box is not None

    # The prefilter box must be STRICTLY larger than the crop box on every face --
    # that is what makes the optimisation lossless rather than a silent crop change.
    c = np.asarray(center_mm, float) / 1000.0
    lo, hi = box
    assert lo[0] < c[0] - scfg.roi_radius_m and hi[0] > c[0] + scfg.roi_radius_m
    assert lo[1] < c[1] - scfg.roi_radius_m and hi[1] > c[1] + scfg.roi_radius_m
    assert lo[2] < c[2] - scfg.roi_below_m and hi[2] > c[2] + scfg.roi_above_m

    kw = dict(voxel_size_m=0.005, sdf_trunc_m=0.02, depth_min_m=0.2, depth_max_m=1.5)
    whole_room = rc.fuse_views(views, **kw)
    prefiltered = rc.fuse_views(views, roi_box_m=box, **kw)

    n_room = len(whole_room.cloud.points)
    n_pre = len(prefiltered.cloud.points)
    # The floor is ~15x the table's area here; the real cell run was ~100x worse.
    assert n_pre < 0.25 * n_room, (n_pre, n_room)

    # ...and nothing the run actually measures gets worse. Compare the fitted plane
    # and the cleaned measured mesh's COVERAGE -- what the edge-support gate reads --
    # rather than raw TSDF point counts: dropping a far surface also drops the
    # sub-surface silhouette skirt the TSDF grows where that surface met a kept one
    # (here an extreme case, a table floating 750 mm above the floor; on the cell the
    # platform sits on a continuous table well inside the box).
    roi = dict(radius_m=scfg.roi_radius_m, below_m=scfg.roi_below_m,
               above_m=scfg.roi_above_m)

    def _measured(res):
        cloud = rc.crop_box(res.cloud, c, **roi)
        pts = rc.cloud_points_m(cloud)
        wp = work_plane_from_points(pts, distance=0.006, min_inlier_frac=0.5)
        cleaned, stats = rc.clean_measured_surface_mesh(
            rc.crop_box(res.mesh, c, **roi), views, wp,
            plane_band_m=scfg.measured_mesh_plane_band_m,
            rect_margin_m=scfg.measured_mesh_rect_margin_m,
            support_tolerance_m=scfg.measured_mesh_support_tolerance_m,
            min_support_views=scfg.measured_mesh_min_support_views,
            min_support_ratio=scfg.measured_mesh_min_support_ratio,
            min_normal_dot=scfg.measured_mesh_min_normal_dot,
            depth_min_m=scfg.depth_min_m, depth_max_m=scfg.depth_max_m,
            keep_largest_component=scfg.measured_mesh_keep_largest_component,
            project_to_plane=scfg.measured_mesh_project_to_plane,
            neutral_color=scfg.measured_mesh_neutral_color)
        cov = scan_service._surface_coverage(
            np.asarray(cleaned.vertices, float), wp,
            bin_m=scfg.actual_coverage_bin_m,
            edge_band_m=scfg.actual_coverage_edge_band_m)
        return len(pts), wp, cov, stats

    n_kept_room, wp_room, cov_room, stats_room = _measured(whole_room)
    n_kept_pre, wp_pre, cov_pre, stats_pre = _measured(prefiltered)
    assert n_kept_room > 500, n_kept_room
    # Same PLANE (a fit's centroid slides freely within the plane, so compare the
    # normal and the offset along it), and no shrunken extent.
    assert abs(float(wp_pre.normal @ wp_room.normal)) > 0.999, (wp_pre.normal, wp_room.normal)
    assert abs(float((wp_pre.centroid - wp_room.centroid) @ wp_room.normal)) < 0.001
    assert wp_pre.size[0] >= wp_room.size[0] - 0.010, (wp_pre.size, wp_room.size)
    assert wp_pre.size[1] >= wp_room.size[1] - 0.010, (wp_pre.size, wp_room.size)
    assert wp_pre.inlier_frac >= wp_room.inlier_frac - 0.01,         (wp_pre.inlier_frac, wp_room.inlier_frac)
    # The gate must not read WORSE because we stopped integrating the room.
    assert cov_pre["weakest_edge"] >= cov_room["weakest_edge"] - 0.02,         (cov_pre["weakest_edge"], cov_room["weakest_edge"])
    assert cov_pre["fill"] >= cov_room["fill"] - 0.02, (cov_pre["fill"], cov_room["fill"])
    assert cov_pre["interior"] >= cov_room["interior"] - 0.02,         (cov_pre["interior"], cov_room["interior"])
    # point_count is a raw vertex tally, not a coverage measure: it drops slightly
    # because the skirt vertices are gone, while fill/interior/weakest_edge above
    # show those vertices occupied no bin of their own. What must hold is the gate's
    # VERDICT -- the prefilter may never invent a new rejection reason.
    assert cov_pre["point_count"] >= 0.9 * cov_room["point_count"],         (cov_pre["point_count"], cov_room["point_count"])
    reasons_room = scan_service._surface_quality_reasons(cov_room, stats_room, scfg)
    reasons_pre = scan_service._surface_quality_reasons(cov_pre, stats_pre, scfg)
    assert len(reasons_pre) <= len(reasons_room), (reasons_pre, reasons_room)

    # roi_enabled=False keeps the old fuse-everything behaviour (the escape hatch).
    scfg.roi_enabled = False
    assert scan_service._fusion_roi_box(scfg, center_mm) is None
    scfg.roi_enabled = True
    assert scan_service._fusion_roi_box(scfg, None) is None

    print(f"[roi prefilter] integrated {n_room} -> {n_pre} pts ({n_pre / n_room:.1%}); "
          f"cropped {n_kept_room} vs {n_kept_pre}; weakest edge "
          f"{cov_room['weakest_edge']:.3f} -> {cov_pre['weakest_edge']:.3f}, fill "
          f"{cov_room['fill']:.3f} -> {cov_pre['fill']:.3f}")
