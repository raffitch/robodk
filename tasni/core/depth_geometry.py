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
    # The smooth_delta the Jetson's spatial filter ACTUALLY ran at, or None when the
    # chain had no spatial filter (`"spatial" not in raw["filters"]`) -- and also None
    # for every greeting archived before the server started reporting it, which is
    # every take under runs/ up to 2026-09-01. Provenance only: nothing here
    # back-projects with it. It exists so the two arms of a smooth_delta A/B are
    # distinguishable on disk (docs/inspection-roll-probe-handoff.md 3.1).
    spatial_smooth_delta: float | None = None

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
        # `filter_options` arrived 2026-09-01; every greeting archived under runs/
        # before that has no such key, and those archives are re-read on every
        # reprocess and every figure render. Absent, malformed or null all mean the
        # same readable thing -- unknown -- and none of them may raise.
        options = d.get("filter_options")
        delta = options.get("spatial_smooth_delta") if isinstance(options, dict) else None
        try:
            delta = None if delta is None else float(delta)
        except (TypeError, ValueError):
            delta = None
        return cls(
            protocol=2, depth_unit_mm=float(d["depth_unit_mm"]),
            depth_K=cls._K(depth), depth_size=(int(depth["width"]), int(depth["height"])),
            depth_dist=coeffs, color_size=(int(color["width"]), int(color["height"])),
            color_K_factory=cls._K(color), T_color_depth=T,
            temps=dict(d.get("temps") or {}), device=dict(d.get("device") or {}),
            raw=dict(d), legacy=False, spatial_smooth_delta=delta)

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
    """One frame's valid depth points with their positions in the colour image.

    Note the two independent sources here: ``color_size`` (the canvas every
    ``in_polygon`` / ``in_center_patch`` test is measured against) comes from the
    GREETING, while ``uv`` comes from the caller's CALIBRATED ``K_color``, which
    the host selects by the ``camera.resolution`` config string. If those two
    describe different image sizes, nothing in this class notices -- the points
    simply land in the wrong part of the canvas. ``CameraClient.check_color_size``
    is what keeps them consistent for live frames; archive readers build both from
    one take's own K/size via :meth:`CameraGeometry.legacy_aligned`."""

    def __init__(self, pts_mm, uv, uv_depth, color_size, depth_size, stride,
                 depth_K, color_K):
        self.pts_mm = np.asarray(pts_mm, float)
        self.uv = np.asarray(uv, float)
        self.uv_depth = np.asarray(uv_depth, int)
        self.color_size = (int(color_size[0]), int(color_size[1]))
        self.depth_size = (int(depth_size[0]), int(depth_size[1]))
        self.stride = int(stride)
        # R25: the geometry's own depth K and the colour K that PRODUCED ``uv``, so
        # _density_ratio can compute the depth-px/colour-px ratio ANALYTICALLY
        # instead of estimating it from the registered points' footprint (which was
        # biased ~27% low -- see _density_ratio's docstring). ``color_K`` must be
        # the same matrix ``build`` projected with (the CALIBRATED model), not the
        # greeting's factory K: the two disagree by ~2% per axis on this D435i, and
        # it is the projection that sets where the points land.
        # It is REQUIRED, and named for what it must be. A ``color_K_factory=``
        # keyword alias briefly survived here for one test's benefit; a parameter
        # named for the factory K is an invitation to pass ``geom.color_K_factory``,
        # which is exactly the ~4% density bias this argument exists to prevent.
        self._depth_K = np.asarray(depth_K, float)
        self._color_K = np.asarray(color_K, float)
        self._tree = None

    @classmethod
    def build(cls, depth, geom: CameraGeometry, K_color, dist_color, *, stride: int = 1
              ) -> "ColorRegistered":
        pts, uv_depth = backproject(depth, geom, stride=stride)
        uv = project_to_color(pts, K_color, dist_color)
        d = np.asarray(depth)
        # K_color (calibrated), NOT geom.color_K_factory: uv above came out of it.
        return cls(pts, uv, uv_depth, geom.color_size, (d.shape[1], d.shape[0]), stride,
                   geom.depth_K, K_color)

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
        """The depth-pixels-per-colour-pixel ratio, ANALYTIC (R25) not estimated.

        For a fronto-parallel surface at distance z, one depth pixel subtends
        ``(z/fx_d)*(z/fy_d)`` of world area and one colour pixel subtends
        ``(z/fx_c)*(z/fy_c)``; z cancels, so the ratio is exactly
        ``(fx_d*fy_d)/(fx_c*fy_c)`` -- a constant of the two cameras' intrinsics,
        independent of distance or how much of the frame is actually covered.

        ``fx_c/fy_c`` are the CALIBRATED colour focal lengths -- the ones
        ``build`` projected the points with -- NOT the greeting's factory K.
        ``uv`` is where the points landed, so the density of registered points
        per colour pixel is governed by the matrix that put them there; dividing
        by anything else is a constant bias. This one was measurable: on this
        D435i the factory K reads fx,fy = 1362.15, 1362.21 against the
        calibration's 1334.81, 1336.21, so the factory denominator inflated
        ``valid_frac`` by a flat x1.0403 -- a fully covered synthetic patch read
        1.0376 instead of 1.0, and ``evaluate_depth_gate``'s
        ``min_valid_depth_frac = 0.5`` actually tripped at a true coverage of
        0.4806. It erred permissive (an early DETECT, never a false refusal) and
        the gate clamps with ``min(1.0, ...)``, but the biased number is also
        persisted into scan records, where a later reader cannot see the bias.
        Fixing it makes the DETECT gate ~4% stricter; a real live scene measures
        ~0.92 against the 0.5 threshold, so the headroom is ample.

        For ``legacy_aligned`` geometry the archive callers hand the SAME config
        K to ``legacy_aligned`` (which makes it ``depth_K``) and to ``build``
        (which makes it ``color_K``) -- ``extrusion/service.py``'s reprocess and
        ``extrusion/figures.py``'s ``geometry_for_take``/``_compute_stages`` both
        read one ``intrinsics["K"]``/``take.K`` -- so ``(a*b)/(a*b)`` is exactly
        1.0 at stride 1, matching the pre-protocol-2 convention (depth pixel ==
        colour pixel) by construction.

        This REPLACES a footprint-based estimate (total depth pixels after stride,
        over the colour-image area the registered points' own bounding box
        covered) that R19 accepted as "inside the test's band" without checking
        whether that band was the right answer: measured on a fully-covered centre
        patch it returned 0.25 against the true 0.1878 for the depth_gate test
        fixture, biasing valid_frac to ~0.73 for a 100%-covered patch that should
        read 1.0 (R25). Approximate only insofar as the surface is tilted (a
        tilted patch's true footprint differs from the fronto-parallel constant
        by a small, tilt-dependent factor) -- the same approximation the old
        estimate made, but centred correctly at zero tilt instead of biased low
        there too.

        Divided by ``stride**2``: a caller that builds this registration with
        ``stride > 1`` (``evaluate_depth_gate``'s stride=2, cheap for the live HUD)
        only samples 1-in-``stride**2`` native depth pixels, so the number of
        REGISTERED points expected to fill a fully-covered patch drops by that
        same factor -- exactly what the old footprint estimate's own
        ``n_depth = (w_d // stride) * (h_d // stride)`` accounted for and a
        stride-naive analytic formula would silently drop (caught by re-running
        the full Step-9 suite after this fix: stride=2 callers went from
        ``detected`` to not-detected until this factor was restored)."""
        fx_d, fy_d = float(self._depth_K[0, 0]), float(self._depth_K[1, 1])
        fx_c, fy_c = float(self._color_K[0, 0]), float(self._color_K[1, 1])
        if fx_c <= 0.0 or fy_c <= 0.0:
            return 1.0
        return (fx_d * fy_d) / (fx_c * fy_c) / float(self.stride ** 2)

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
