"""Jetson live plane expansion: rough top surface + center-connected selection."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The server module defines pure telemetry helpers but also imports Jetson-only
# camera packages at module import time. Stub only those imports for this unit test.
sys.modules.setdefault("pyrealsense2", SimpleNamespace())
sys.modules.setdefault("turbojpeg", SimpleNamespace())

from server.server_unicast_syncronous import (  # noqa: E402
    center_connected_mask, convex_hull_2d, fit_nearest_plane,
    footprint_edge_angle_deg,
    min_area_rectangle_2d,
    scan_plane_telemetry,
)


def test_center_connected_component_excludes_remote_plane():
    mask = np.zeros((30, 40), bool)
    mask[8:24, 10:31] = True       # intended center surface
    mask[2:7, 33:39] = True        # unrelated coplanar patch
    out = center_connected_mask(mask)
    assert out[15, 20]
    assert not out[4, 35]
    print("[telemetry component] center surface kept; remote plane excluded")


def test_rough_top_expands_beyond_clean_center():
    h, w = 240, 320
    depth = np.zeros((h, w), np.uint16)
    # Entire intended top: 250×170 px. Outer area is rough by ±10 mm while the
    # center patch is cleaner. The old fixed ±8 mm band clipped this footprint.
    rng = np.random.default_rng(4)
    rough = 500.0 + rng.integers(-3, 4, size=(170, 250))
    depth[35:205, 35:285] = rough.astype(np.uint16)
    depth[90:150, 120:200] = 500
    # Disconnected coplanar distractor must not enlarge the selected rectangle.
    depth[8:28, 290:315] = 500
    intr = SimpleNamespace(fx=300.0, fy=300.0, ppx=160.0, ppy=120.0)
    p = scan_plane_telemetry(depth, intr)
    assert p["detected"]
    assert 3.0 <= p["plane_tolerance_mm"] <= 7.0
    assert p["fully_framed"] is True
    assert p["surface_mode"] == "full"
    assert p["color_fit_standoff_per_margin_mm"] > 0
    assert len(p["outline_uv"]) == 4
    assert len(p["visible_outline_uv"]) >= 4
    assert p["extent_mm"][0] > 380.0, p["extent_mm"]
    assert p["extent_mm"][1] > 260.0, p["extent_mm"]
    print("[telemetry rough top] extent", tuple(round(x) for x in p["extent_mm"]),
          "mm at tolerance", round(p["plane_tolerance_mm"], 1), "mm")


def test_visible_hull_follows_points_and_stays_inside_image():
    pts = np.array([
        [-0.05, 0.20], [0.75, 0.15], [0.80, 0.85], [-0.02, 0.90],
        [0.35, 0.50], [0.40, 0.55],
    ])
    hull = convex_hull_2d(pts)
    assert len(hull) == 4
    clipped = np.clip(hull, 0.0, 1.0)
    assert clipped[:, 0].min() == 0.0
    assert clipped[:, 0].max() <= 1.0
    print("[telemetry hull] visible boundary follows segmented pixels and clips at image edge")


def test_nearest_coherent_plane_wins_over_lower_base():
    h, w = 240, 320
    depth = np.zeros((h, w), np.uint16)
    depth[25:215, 25:295] = 510
    depth[70:170, 100:220] = 480
    intr = SimpleNamespace(fx=300.0, fy=300.0, ppx=160.0, ppy=120.0)
    p = scan_plane_telemetry(depth, intr)
    assert p["detected"]
    assert abs(p["distance_mm"] - 480.0) < 2.0, p["distance_mm"]
    assert p["plane_tolerance_mm"] <= 7.0
    assert 150.0 < p["extent_mm"][0] < 230.0, p["extent_mm"]
    assert 120.0 < p["extent_mm"][1] < 200.0, p["extent_mm"]
    print("[telemetry stacked] raised top selected; lower base excluded")


def test_nearest_plane_helper_selects_smallest_depth_layer():
    rng = np.random.default_rng(12)
    xy = rng.uniform(-50, 50, size=(200, 2))
    upper = np.column_stack([xy, 480.0 + rng.normal(0, 0.6, 200)])
    lower = np.column_stack([xy, 510.0 + rng.normal(0, 0.6, 200)])
    normal, centroid, mask = fit_nearest_plane(np.vstack([lower, upper]))
    assert abs(centroid[2] - 480.0) < 2.0
    assert int(mask.sum()) >= 180
    assert normal[2] < 0


def test_edge_angle_returns_smallest_a_correction():
    a = np.deg2rad(12.0)
    edge = np.array([np.cos(a), np.sin(a)])
    normal = np.array([-edge[1], edge[0]])
    hull = np.array([[0, 0], edge, edge + 0.5 * normal, 0.5 * normal])
    assert abs(footprint_edge_angle_deg(hull) - 12.0) < 1e-6
    print("[telemetry edge] dominant platform edge requests A correction")


def test_rectangle_ignores_a_notch_in_visible_depth_silhouette():
    border = []
    for x in np.linspace(-2.0, 2.0, 80):
        border.extend(([x, -1.0], [x, 1.0]))
    for y in np.linspace(-1.0, 1.0, 50):
        if not (-0.35 < y < 0.35):
            border.append([2.0, y])
        border.append([-2.0, y])
    pts = np.asarray(border)
    ux, uy, width, height, _, _ = min_area_rectangle_2d(
        pts, preferred_axis=np.array([1.0, 0.0]))
    assert abs(width - 4.0) < 1e-6
    assert abs(height - 2.0) < 1e-6
    assert abs(float(ux @ [1.0, 0.0])) > 0.999
    assert abs(float(uy @ [0.0, 1.0])) > 0.999
    print("[telemetry rectangle] missing right-edge depth did not bend the work box")


def test_solid_rectangle_uses_distortion_aware_corner_projector():
    h, w = 240, 320
    depth = np.zeros((h, w), np.uint16)
    depth[35:205, 35:285] = 500
    intr = SimpleNamespace(fx=300.0, fy=300.0, ppx=160.0, ppy=120.0)
    scalar_calls = []

    def scalar_project(point):
        scalar_calls.append(point)
        return [160.0 + point[0] * 0.61, 120.0 + point[1] * 0.59]

    def batch_project(points):
        points = np.asarray(points)
        return np.column_stack([
            160.0 + points[:, 0] * 0.60,
            120.0 + points[:, 1] * 0.60,
        ])

    p = scan_plane_telemetry(
        depth, intr,
        overlay_project=scalar_project,
        overlay_project_points=batch_project,
        overlay_size=(w, h),
    )
    assert len(scalar_calls) == 4, "solid rectangle corners must use RealSense projection"
    assert len(p["rectangle_size_mm"]) == 2
    assert p["rectangle_size_mm"][0] > 0 and p["rectangle_size_mm"][1] > 0
    print("[telemetry projection] solid corners use distortion-aware projector")


if __name__ == "__main__":
    test_center_connected_component_excludes_remote_plane()
    test_rough_top_expands_beyond_clean_center()
    test_nearest_coherent_plane_wins_over_lower_base()
    test_nearest_plane_helper_selects_smallest_depth_layer()
    test_visible_hull_follows_points_and_stays_inside_image()
    test_edge_angle_returns_smallest_a_correction()
    test_rectangle_ignores_a_notch_in_visible_depth_silhouette()
    test_solid_rectangle_uses_distortion_aware_corner_projector()
    print("\nJetson scan telemetry tests passed.")
