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

from tasni.modules.scan.plane import (  # noqa: E402
    bounded_work_plane, reticle_plane_square, work_plane_from_points)


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


def test_density_trim_hugs_dense_board_not_halo():
    """A sparse coplanar halo just past the real edge must NOT inflate the rectangle.
    The board is 400 mm on its long axis; a sparse band of points extends ~30 mm
    past the +x edge — too many for the 0.5% quantile to remove, but far less dense
    than the board. The per-edge density trim should pull the long edge back to ~400,
    not the ~430 over-run the quantile-only box produced."""
    board = _rect_grid((200.0, 600.0), (100.0, 400.0), nx=80, ny=60, z_noise=0.2)
    # Sparse halo: 6 columns x 8 rows over (600..630) x (100..400) => ~1.0% of points,
    # ~8 per x-bin vs ~60 on the board, so it is a clear sub-threshold fringe.
    ghx, ghy = np.meshgrid(np.linspace(600.0, 630.0, 6), np.linspace(100.0, 400.0, 8))
    halo = np.column_stack([ghx.ravel(), ghy.ravel(), np.zeros(ghx.size)])
    wp = work_plane_from_points(np.vstack([board, halo]),
                                distance=3.0, min_inlier_frac=0.5)
    assert abs(wp.size[0] - 400) < 18, wp.size  # hugs the board, not 400+30 over-run
    print("[density trim] long edge", round(wp.size[0], 1),
          "mm (board 400 + 30 mm halo)")


def test_density_trim_keeps_uniform_board():
    """A uniformly dense board (no halo) is preserved: the trim must not eat a real,
    fully-sampled edge — only a genuine density cliff is trimmed."""
    pts = _rect_grid((0.0, 500.0), (0.0, 350.0), nx=90, ny=70, z_noise=0.2)
    wp = work_plane_from_points(pts, distance=3.0, min_inlier_frac=0.5)
    assert abs(wp.size[0] - 500) < 12 and abs(wp.size[1] - 350) < 12, wp.size
    print("[density trim] uniform board preserved", tuple(round(s, 1) for s in wp.size))


def test_reticle_square_centered_on_optical_axis():
    """A generic work square on a frontal plane is centred where the +Z optical axis
    (the reticle) pierces it — independent of the in-plane centroid offset — with the
    requested size and screen-aligned (first axis ~ camera +X)."""
    normal = np.array([0.0, 0.0, -1.0])                  # frontal, faces the camera
    centroid = np.array([80.0, -40.0, 500.0])            # offset in-plane
    corners, u, v, reticle = reticle_plane_square(normal, centroid, (1000.0, 1000.0))
    assert np.allclose(corners.mean(axis=0), [0.0, 0.0, 500.0], atol=1e-6), corners.mean(0)
    assert np.allclose(reticle, [0.0, 0.0, 500.0], atol=1e-6), reticle
    edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    assert np.allclose(edges, 1000.0, atol=1e-6), edges
    assert abs(float(u @ [1, 0, 0])) > 0.999, u          # screen-stable axis
    assert abs(float(u @ normal)) < 1e-9 and abs(float(v @ normal)) < 1e-9  # in-plane
    print("[reticle square] centred on the optical axis, 1000 mm, screen-aligned")


def test_reticle_square_lies_on_tilted_plane():
    """On a tilted plane every corner is exactly on the plane, the centre is on the
    optical axis, and the side lengths match the requested (sx, sy)."""
    a = np.deg2rad(20.0)
    normal = np.array([np.sin(a), 0.0, -np.cos(a)])      # tilted about Y, faces camera
    centroid = np.array([0.0, 25.0, 600.0])
    corners, u, v, reticle = reticle_plane_square(normal, centroid, (800.0, 600.0))
    assert np.allclose((corners - centroid) @ normal, 0.0, atol=1e-6)  # all on the plane
    assert abs(reticle[0]) < 1e-9 and abs(reticle[1]) < 1e-9 and reticle[2] > 0
    e = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    assert np.allclose(np.sort(e), [600, 600, 800, 800], atol=1e-6), e
    print("[reticle square] lies on a 20° plane, 800×600, centred on the axis")


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
    test_density_trim_hugs_dense_board_not_halo()
    test_density_trim_keeps_uniform_board()
    test_reticle_square_centered_on_optical_axis()
    test_reticle_square_lies_on_tilted_plane()
    test_large_surface_can_be_bounded_around_camera_aim()
    print("\nplane.py convention + fit tests passed.")
