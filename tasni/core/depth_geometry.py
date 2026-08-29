"""Depth-frame geometry: the ONE place raw RealSense depth becomes 3D points.

The Jetson streams NATIVE depth (1280x720, 0.1 mm units, not aligned to colour)
and, once per connection, a greeting with the depth intrinsics and the
depth->colour extrinsic. Everything on the host that reads ``Frame.depth`` comes
through here. Points are returned in the **colour camera frame**: the hand-eye
(``RoboDKIO.camera_pose_T()``) is the colour camera's pose, so it applies to them
directly and every existing "camera frame" convention downstream is unchanged --
only the pixel->point step moved. ``depth_pose`` exists for the two consumers that
work on the depth *image* itself (Open3D TSDF integration, per-view support).

Colour-space selections (the aiming reticle, a survey rectangle, a ChArUco corner)
are answered by :class:`ColorRegistered`: back-project every valid depth pixel,
project into the calibrated colour model, keep what lands inside. There is no
inverse mapping and no resampling -- nothing here invents a depth value.

``CameraGeometry.legacy_aligned`` reproduces the pre-protocol-2 convention
(depth == colour image, 1 mm) for ARCHIVED takes only; live code never builds one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import transform_points

_ZERO_DIST = np.zeros((5, 1), np.float32)


@dataclass(frozen=True)
class CameraGeometry:
    protocol: int
    depth_unit_mm: float
    depth_K: np.ndarray
    depth_size: tuple[int, int]
    depth_dist: np.ndarray
    color_size: tuple[int, int]
    color_K_factory: np.ndarray
    T_color_depth: np.ndarray
    temps: dict
    device: dict
    raw: dict
    legacy: bool = False

    @staticmethod
    def _K(block: dict) -> np.ndarray:
        return np.array([[float(block["fx"]), 0.0, float(block["ppx"])],
                         [0.0, float(block["fy"]), float(block["ppy"])],
                         [0.0, 0.0, 1.0]])

    @classmethod
    def from_greeting(cls, d: dict) -> "CameraGeometry":
        if int(d.get("protocol", 0)) != 2:
            raise ValueError(f"camera greeting protocol {d.get('protocol')!r} is not 2")
        for key in ("depth_unit_mm", "depth", "color", "depth_to_color"):
            if key not in d:
                raise ValueError(f"camera greeting is missing {key!r}")
        dc = d["depth_to_color"]
        R = np.asarray(dc["rotation_row_major"], float).reshape(3, 3)
        t = np.asarray(dc["translation_mm"], float).reshape(3)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        depth, color = d["depth"], d["color"]
        coeffs = np.asarray(depth.get("coeffs") or [0, 0, 0, 0, 0], float)[:5]
        return cls(
            protocol=2, depth_unit_mm=float(d["depth_unit_mm"]),
            depth_K=cls._K(depth), depth_size=(int(depth["width"]), int(depth["height"])),
            depth_dist=coeffs, color_size=(int(color["width"]), int(color["height"])),
            color_K_factory=cls._K(color), T_color_depth=T,
            temps=dict(d.get("temps") or {}), device=dict(d.get("device") or {}),
            raw=dict(d), legacy=False)

    @classmethod
    def legacy_aligned(cls, K_color, size, *, depth_unit_mm: float = 1.0) -> "CameraGeometry":
        """The pre-protocol-2 convention, for ARCHIVED takes without camera_geometry:
        depth was aligned to colour on the Jetson, so depth K == colour K, depth
        size == colour size, and the extrinsic is the identity."""
        K = np.asarray(K_color, float).reshape(3, 3)
        size = (int(size[0]), int(size[1]))
        raw = {"legacy_aligned": True, "protocol": 1, "depth_unit_mm": float(depth_unit_mm),
               "K": K.tolist(), "size": list(size)}
        return cls(protocol=1, depth_unit_mm=float(depth_unit_mm), depth_K=K, depth_size=size,
                   depth_dist=np.zeros(5), color_size=size, color_K_factory=K,
                   T_color_depth=np.eye(4), temps={}, device={}, raw=raw, legacy=True)

    def to_dict(self) -> dict:
        return dict(self.raw)


def backproject(depth, geom: CameraGeometry, *, stride: int = 1, mask=None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Valid depth pixels -> (N,3) mm in the COLOUR camera frame, plus their (N,2)
    [u, v] depth-image pixels. ``stride`` subsamples rows/cols; ``mask`` (depth-image
    shape) restricts the pixels. The depth image is rectified, so its distortion
    coefficients are not applied."""
    d = np.asarray(depth)
    if d.ndim != 2 or d.size == 0:
        return np.zeros((0, 3), float), np.zeros((0, 2), int)
    stride = max(1, int(stride))
    sub = d[::stride, ::stride]
    valid = sub > 0
    if mask is not None:
        valid &= np.asarray(mask, bool)[::stride, ::stride]
    vs, us = np.nonzero(valid)
    if len(vs) == 0:
        return np.zeros((0, 3), float), np.zeros((0, 2), int)
    u = us * stride
    v = vs * stride
    z = sub[vs, us].astype(np.float64) * float(geom.depth_unit_mm)
    K = geom.depth_K
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pts_depth = np.column_stack([x, y, z])
    if geom.legacy:
        pts_color = pts_depth
    else:
        pts_color = transform_points(geom.T_color_depth, pts_depth)
    return pts_color, np.column_stack([u, v]).astype(int)


def depth_pose(T_x_color, geom: CameraGeometry) -> np.ndarray:
    """The depth camera's pose from the colour camera's: T_x_depth = T_x_color @ T_color_depth."""
    return np.asarray(T_x_color, float).reshape(4, 4) @ geom.T_color_depth


def project_to_color(pts_color_mm, K_color, dist_color) -> np.ndarray:
    """Colour-frame points (mm) -> (N,2) float colour pixels through the CALIBRATED model."""
    import cv2
    p = np.asarray(pts_color_mm, np.float64).reshape(-1, 3)
    if len(p) == 0:
        return np.zeros((0, 2), float)
    dist = _ZERO_DIST if dist_color is None else np.asarray(dist_color, np.float64).reshape(-1, 1)
    uv, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), np.asarray(K_color, np.float64), dist)
    return uv.reshape(-1, 2)


def ray_point(u, v, z_mm, K_color, dist_color) -> np.ndarray:
    """The colour-frame point at depth ``z_mm`` on the UNDISTORTED ray through (u, v)."""
    import cv2
    dist = _ZERO_DIST if dist_color is None else np.asarray(dist_color, np.float64).reshape(-1, 1)
    xy = cv2.undistortPoints(np.array([[[float(u), float(v)]]], np.float64),
                             np.asarray(K_color, np.float64), dist).reshape(2)
    return np.array([xy[0] * z_mm, xy[1] * z_mm, float(z_mm)])


class ColorRegistered:
    """One frame's valid depth points with their positions in the colour image."""

    def __init__(self, pts_mm, uv, uv_depth, color_size, depth_size, stride):
        self.pts_mm = np.asarray(pts_mm, float)
        self.uv = np.asarray(uv, float)
        self.uv_depth = np.asarray(uv_depth, int)
        self.color_size = (int(color_size[0]), int(color_size[1]))
        self.depth_size = (int(depth_size[0]), int(depth_size[1]))
        self.stride = int(stride)
        self._tree = None

    @classmethod
    def build(cls, depth, geom: CameraGeometry, K_color, dist_color, *, stride: int = 1
              ) -> "ColorRegistered":
        pts, uv_depth = backproject(depth, geom, stride=stride)
        uv = project_to_color(pts, K_color, dist_color)
        d = np.asarray(depth)
        return cls(pts, uv, uv_depth, geom.color_size, (d.shape[1], d.shape[0]), stride)

    def __len__(self) -> int:
        return len(self.pts_mm)

    def in_polygon(self, polygon_uv_norm) -> np.ndarray:
        import cv2
        w, h = self.color_size
        poly = np.asarray(polygon_uv_norm, float).reshape(-1, 2) * [w, h]
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.rint(poly).astype(np.int32)], 1)
        u = np.clip(np.rint(self.uv[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.rint(self.uv[:, 1]).astype(int), 0, h - 1)
        inside = (self.uv[:, 0] >= 0) & (self.uv[:, 0] < w) & (self.uv[:, 1] >= 0) & (self.uv[:, 1] < h)
        return inside & (mask[v, u] > 0)

    def _center_patch_bounds(self, frac):
        w, h = self.color_size
        pf = float(np.clip(frac, 0.05, 1.0))
        cw, ch = max(2, int(w * pf)), max(2, int(h * pf))
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        return x0, y0, x0 + cw, y0 + ch

    def in_center_patch(self, frac) -> np.ndarray:
        x0, y0, x1, y1 = self._center_patch_bounds(frac)
        return ((self.uv[:, 0] >= x0) & (self.uv[:, 0] < x1)
                & (self.uv[:, 1] >= y0) & (self.uv[:, 1] < y1))

    def valid_frac_in_center_patch(self, frac) -> float:
        """Points found in the colour centre patch / the points a FULLY valid depth
        image would put there. The denominator is the patch's area in colour pixels
        scaled by the depth image's angular pixel density relative to colour --
        exact for a fronto-parallel surface, and a ratio, so units cancel."""
        x0, y0, x1, y1 = self._center_patch_bounds(frac)
        area_color = float((x1 - x0) * (y1 - y0))
        if area_color <= 0:
            return 0.0
        return float(self.in_center_patch(frac).sum()) / (area_color * self._density_ratio())

    def _density_ratio(self) -> float:
        """ESTIMATE of depth pixels per colour pixel, from the registered points'
        own footprint (total depth pixels after stride, over the colour-image area
        they actually cover) -- not a measurement, only good enough to feed
        valid_frac_in_center_patch's threshold. depth pixels per colour pixel from
        the two images' sizes and FOVs alone is not known here (the colour FOV/crop
        relative to depth is not carried in the greeting), so a partially valid
        frame does not inflate the ratio: it uses the points' own spread, not the
        full sensor geometry."""
        w_c, h_c = self.color_size
        w_d, h_d = self.depth_size
        n_depth = (w_d // self.stride) * (h_d // self.stride)
        if len(self.uv) == 0:
            return n_depth / float(w_c * h_c)
        span_u = max(1.0, float(np.ptp(self.uv[:, 0])))
        span_v = max(1.0, float(np.ptp(self.uv[:, 1])))
        cover = min(float(w_c * h_c), span_u * span_v * n_depth / max(len(self.uv), 1))
        return n_depth / max(cover, 1.0)

    def near(self, u_px, v_px, radius_px) -> np.ndarray:
        if len(self.uv) == 0:
            return np.zeros(0, int)
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.uv)
        return np.asarray(self._tree.query_ball_point([float(u_px), float(v_px)], float(radius_px)), int)

    def median_z_near(self, u_px, v_px, radius_px) -> float:
        idx = self.near(u_px, v_px, radius_px)
        if len(idx) == 0:
            return float("nan")
        return float(np.median(self.pts_mm[idx, 2]))
