"""plane.py — work-plane / rectangle / frame-convention math (pure numpy).

Synthetic rectangular surfaces (flat + tilted) -> assert the recovered normal,
rectangle size, and the frame CONVENTION (origin = corner nearest the base origin,
X = long edge, Z = up, right-handed). No RoboDK / open3d / cv2.

    py -3.10 tests/test_scan_plane.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.modules.scan.plane import bounded_work_plane, work_plane_from_points  # noqa: E402


def _rect_grid(xr, yr, nx=50, ny=40, z_noise=0.0, seed=0):
    """A filled rectangle of points in the z=0 plane, optional gaussian z-noise."""
    xs = np.linspace(xr[0], xr[1], nx)
    ys = np.linspace(yr[0], yr[1], ny)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    if z_noise:
        pts[:, 2] += np.random.default_rng(seed).normal(0, z_noise, len(pts))
    return pts


def _rot_x(deg):
    a = np.deg2rad(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def test_flat_rectangle_frame_convention():
    # 400 x 300 rectangle, offset from the origin -> nearest corner is (200,100,0).
    pts = _rect_grid((200.0, 600.0), (100.0, 400.0), z_noise=0.4)
    wp = work_plane_from_points(pts, distance=3.0, min_inlier_frac=0.5)

    # Z = up (plane normal)
    assert abs(float(wp.normal @ [0, 0, 1])) > 0.999, wp.normal
    # extent ~ (400, 300) mm
    assert abs(wp.size[0] - 400) < 12 and abs(wp.size[1] - 300) < 12, wp.size

    R = wp.frame_T[:3, :3]
    origin = wp.frame_T[:3, 3]
    # origin = corner nearest the base origin
    assert np.allclose(origin, [200, 100, 0], atol=8), origin
    # X along the LONGER edge -> +X here; Z up; right-handed
    assert float(R[:, 0] @ [1, 0, 0]) > 0.99, R[:, 0]      # X
    assert float(R[:, 2] @ [0, 0, 1]) > 0.999, R[:, 2]     # Z
    assert abs(np.linalg.det(R) - 1.0) < 1e-6              # proper rotation
    assert np.allclose(R[:, 1], np.cross(R[:, 2], R[:, 0]), atol=1e-6)  # Y = Z x X
    print("[flat] size", tuple(round(s, 1) for s in wp.size),
          "origin", origin.round(1), "inliers", f"{wp.inlier_frac:.0%}")


def test_tilted_plane_normal_recovered():
    flat = _rect_grid((-200.0, 200.0), (-150.0, 150.0), z_noise=0.3)
    R = _rot_x(25.0)
    tilted = flat @ R.T                                    # rotate the whole surface
    expected_normal = R @ np.array([0, 0, 1.0])
    wp = work_plane_from_points(tilted, distance=3.0, min_inlier_frac=0.5)
    # recovered normal aligns with the tilted surface normal (up to orientation)
    assert abs(float(wp.normal @ expected_normal)) > 0.999, (wp.normal, expected_normal)
    # extent preserved through the rotation (~400 x 300)
    assert abs(wp.size[0] - 400) < 12 and abs(wp.size[1] - 300) < 12, wp.size
    print("[tilted] normal", wp.normal.round(3), "size",
          tuple(round(s, 1) for s in wp.size))


def test_no_dominant_plane_raises():
    # 60% on a plane, 40% scattered -> below a 0.9 inlier-fraction floor -> refuse.
    plane = _rect_grid((-100.0, 100.0), (-100.0, 100.0), nx=30, ny=30, z_noise=0.5)
    rng = np.random.default_rng(1)
    junk = rng.uniform(-150, 150, size=(int(len(plane) * 2 / 3), 3))
    cloud = np.vstack([plane, junk])
    try:
        work_plane_from_points(cloud, distance=3.0, min_inlier_frac=0.9)
        raise AssertionError("expected a refusal — no plane covers 90% of the cloud")
    except ValueError as e:
        assert "plane covers" in str(e)
    print("[guard] thin/cluttered cloud refused at the inlier-fraction floor")


def test_sparse_edge_noise_does_not_make_diagonal_frame():
    surface = _rect_grid((200.0, 600.0), (100.0, 400.0), nx=80, ny=60, z_noise=0.3)
    # Sparse in-plane silhouette fragments outside two corners previously had
    # unlimited convex-hull leverage and could rotate the minimum-area box.
    fringe = np.array([
        [185.0, 130.0, 0.1], [615.0, 370.0, -0.2],
        [230.0, 85.0, 0.0], [570.0, 415.0, 0.2],
    ])
    wp = work_plane_from_points(
        np.vstack([surface, fringe]), distance=3.0, min_inlier_frac=0.5)
    x = wp.frame_T[:3, 0]
    assert abs(float(x @ [1, 0, 0])) > 0.98, x
    assert 380 < wp.size[0] < 420 and 280 < wp.size[1] < 320, wp.size
    print("[robust footprint] sparse fringe retained an axis-aligned frame")


def test_large_surface_can_be_bounded_around_camera_aim():
    surface = _rect_grid((-1000.0, 1000.0), (-800.0, 800.0), nx=80, ny=60)
    wp = work_plane_from_points(surface, distance=2.0, min_inlier_frac=0.8)
    bounded = bounded_work_plane(wp, np.array([120.0, -50.0, 30.0]), (600.0, 350.0))
    assert np.allclose(bounded.centroid, [120.0, -50.0, 0.0], atol=1e-6)
    assert np.allclose(bounded.size, [600.0, 350.0], atol=1e-6)
    assert np.allclose(bounded.corners[:, 2], 0.0, atol=1e-6)
    print("[large crop] bounded 2 m plane to a 600×350 mm aimed work region")


if __name__ == "__main__":
    test_flat_rectangle_frame_convention()
    test_tilted_plane_normal_recovered()
    test_no_dominant_plane_raises()
    test_sparse_edge_noise_does_not_make_diagonal_frame()
    test_large_surface_can_be_bounded_around_camera_aim()
    print("\nplane.py convention + fit tests passed.")
