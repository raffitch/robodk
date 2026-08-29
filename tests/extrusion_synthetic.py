"""Synthetic RGB-D rendering of dried rings on a flat work surface.

Point-splat + z-buffer, numpy only: dense surface samples in the WORK frame are
moved into the camera frame with inv(T_work_camera), projected with K, and the
nearest depth per pixel is kept. Depth is uint16 millimetres, 0 = no return --
exactly what ``processing.depth_to_work_points`` expects when handed an ALIGNED
legacy geometry (``geometry_fixtures.aligned(K, size)``): depth image == colour
image, 1 mm units, so this renderer's single K/size pair describes both.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from tasni.core.geometry import transform_points
from tasni.modules.extrusion.inspection import pose_from_aim

# The cell's calibrated 1280x720 colour intrinsics (tasni.config.json, 2026-08).
K_720P = np.array([[889.8742117827221, 0.0, 648.9804252459749],
                   [0.0, 890.8099396048351, 362.00464151468503],
                   [0.0, 0.0, 1.0]])
SIZE_720P = (1280, 720)
# The camera's +X at the parked joints reads [-1, 0, 0] in Tasni Work Frame
# (measured on the cell 2026-08-27; see inspection.py).
CAMERA_X_AT_PARK = [-1.0, 0.0, 0.0]

HeightFn = Callable[[np.ndarray], np.ndarray]


def flat(height_mm: float) -> HeightFn:
    return lambda theta: np.full_like(theta, float(height_mm), dtype=float)


def wavy(mean_mm: float, amplitude_mm: float, lobes: int = 2) -> HeightFn:
    """A 'snake' ring: height oscillates ``lobes`` times around the circumference."""
    return lambda theta: mean_mm + amplitude_mm * np.sin(lobes * theta)


@dataclass
class RingSpec:
    """One dried ring: circular centreline, a flattened rounded cross-section.

    Cross-section at angle theta: half-width ``bead_mm / 2``, height
    ``height_fn(theta)``, profile ``z = h * sin(phi) ** 0.5`` for phi in [0, pi]
    (flatter crest than a semi-ellipse -- a slumped bead, which is what the
    upward-normal filter sees on the real material).
    """
    radius_mm: float
    bead_mm: float
    center_xy_mm: tuple[float, float] = (0.0, 0.0)
    z_base_mm: float = 0.0
    height_fn: HeightFn = field(default_factory=lambda: flat(6.0))
    crest_exponent: float = 0.5

    def surface_points(self, step_mm: float = 0.25) -> np.ndarray:
        n_theta = max(64, int(np.ceil(2 * np.pi * (self.radius_mm + self.bead_mm) / step_mm)))
        n_phi = max(8, int(np.ceil(np.pi * self.bead_mm / 2 / step_mm)))
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        phi = np.linspace(0.0, np.pi, n_phi)
        T, P = np.meshgrid(theta, phi, indexing="ij")
        h = self.height_fn(T)
        r = self.radius_mm + (self.bead_mm / 2.0) * np.cos(P)
        x = self.center_xy_mm[0] + r * np.cos(T)
        y = self.center_xy_mm[1] + r * np.sin(T)
        z = self.z_base_mm + h * np.sin(P) ** self.crest_exponent
        return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def plane_points(*, extent_mm: float = 220.0, step_mm: float = 1.0, z_mm: float = 0.0,
                 center_xy_mm: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    axis = np.arange(-extent_mm, extent_mm + step_mm, step_mm)
    X, Y = np.meshgrid(axis + center_xy_mm[0], axis + center_xy_mm[1], indexing="ij")
    return np.column_stack((X.ravel(), Y.ravel(), np.full(X.size, float(z_mm))))


def inspection_camera_T(aim_xyz_mm, standoff_mm: float = 300.0) -> np.ndarray:
    """Camera pose in the work frame, straight down at ``aim``, as the job derives it."""
    return pose_from_aim(np.asarray(aim_xyz_mm, dtype=float), standoff_mm,
                         reference_x=CAMERA_X_AT_PARK)


def render_depth(points_work: np.ndarray, T_work_camera: np.ndarray, *,
                 K: np.ndarray = K_720P, size_px: tuple[int, int] = SIZE_720P,
                 noise_mm: float = 0.5, seed: int = 0) -> np.ndarray:
    """uint16 depth in mm, z-buffered (nearest surface wins); 0 where nothing was hit."""
    width, height = int(size_px[0]), int(size_px[1])
    cam = transform_points(np.linalg.inv(T_work_camera), points_work)
    cam = cam[cam[:, 2] > 1.0]
    u = np.rint(K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]).astype(int)
    v = np.rint(K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]).astype(int)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside], v[inside], cam[inside, 2]
    depth = np.full((height, width), np.inf)
    np.minimum.at(depth, (v, u), z)
    hit = np.isfinite(depth)
    rng = np.random.default_rng(seed)
    depth[hit] += rng.normal(0.0, noise_mm, size=int(hit.sum()))
    depth[~hit] = 0.0
    return np.clip(np.rint(depth), 0, 65535).astype(np.uint16)


def render_scene(rings: list[RingSpec], T_work_camera: np.ndarray, *,
                 plane_z_mm: float = 0.0,
                 plane_center_xy_mm: tuple[float, float] = (0.0, 0.0),
                 noise_mm: float = 0.5, seed: int = 0,
                 K: np.ndarray = K_720P, size_px: tuple[int, int] = SIZE_720P) -> np.ndarray:
    parts = [plane_points(z_mm=plane_z_mm, center_xy_mm=plane_center_xy_mm)]
    parts += [ring.surface_points() for ring in rings]
    return render_depth(np.vstack(parts), T_work_camera, K=K, size_px=size_px,
                        noise_mm=noise_mm, seed=seed)
