# tests/geometry_fixtures.py
"""Synthetic CameraGeometry objects shared by the depth-geometry, scan and
extrusion tests. ``aligned(K, size)`` is the legacy identity registration (depth
image == colour image, 1 mm units) so existing synthetic renders keep their maths;
``offset(...)`` is a real registration: a different depth K/size, 0.1 mm units and a
non-identity depth->colour extrinsic, so a test can prove the mapping is applied."""
from __future__ import annotations

import numpy as np

from tasni.core.depth_geometry import CameraGeometry
from tasni.core.geometry import Rt_to_T


def aligned(K, size, *, depth_unit_mm: float = 1.0) -> CameraGeometry:
    return CameraGeometry.legacy_aligned(np.asarray(K, float), tuple(size),
                                         depth_unit_mm=depth_unit_mm)


def _rot(axis, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    if axis == "x": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def offset(*, color_K, color_size, depth_K=None, depth_size=(160, 120),
           depth_unit_mm: float = 0.1, rot_deg=(0.4, -0.3, 0.2),
           t_mm=(14.7, -0.2, 0.3)) -> CameraGeometry:
    depth_K = np.array([[130.0, 0, 80.0], [0, 130.0, 60.0], [0, 0, 1.0]]) if depth_K is None \
        else np.asarray(depth_K, float)
    R = _rot("x", rot_deg[0]) @ _rot("y", rot_deg[1]) @ _rot("z", rot_deg[2])
    greeting = {
        "protocol": 2, "aligned": False, "depth_unit_mm": depth_unit_mm,
        "depth": {"width": depth_size[0], "height": depth_size[1], "fx": depth_K[0, 0],
                  "fy": depth_K[1, 1], "ppx": depth_K[0, 2], "ppy": depth_K[1, 2],
                  "model": "brown_conrady", "coeffs": [0, 0, 0, 0, 0]},
        "color": {"width": color_size[0], "height": color_size[1], "fx": float(color_K[0][0]),
                  "fy": float(color_K[1][1]), "ppx": float(color_K[0][2]),
                  "ppy": float(color_K[1][2]), "model": "brown_conrady",
                  "coeffs": [0, 0, 0, 0, 0]},
        "depth_to_color": {"rotation_row_major": R.tolist(), "translation_mm": list(t_mm)},
        "filters": ["threshold", "disparity", "spatial", "temporal", "disparity_inv"],
        "device": {"serial": "synthetic", "fw": "0", "librealsense": "0", "visual_preset": 0,
                   "laser_power": 150},
        "temps": {"asic_c": 40.0, "projector_c": 37.0}, "global_time_enabled": True}
    return CameraGeometry.from_greeting(greeting)


def render_depth_in_depth_camera(points_color_mm, geom: CameraGeometry) -> np.ndarray:
    """Splat colour-frame points into the DEPTH camera's image (uint16 in geom units,
    nearest-wins). The inverse of backproject(), for round-trip tests."""
    from tasni.core.geometry import invert_T, transform_points
    p = transform_points(invert_T(geom.T_color_depth), np.asarray(points_color_mm, float))
    w, h = geom.depth_size
    K = geom.depth_K
    z = p[:, 2]
    ok = z > 1e-6
    u = np.rint(K[0, 0] * p[ok, 0] / z[ok] + K[0, 2]).astype(int)
    v = np.rint(K[1, 1] * p[ok, 1] / z[ok] + K[1, 2]).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    raw = np.rint(z[ok][inside] / geom.depth_unit_mm).astype(np.uint16)
    depth = np.zeros((h, w), np.uint16)
    order = np.argsort(-raw)                     # nearest wins: write far first
    depth[v[inside][order], u[inside][order]] = raw[order]
    return depth
