"""Auto-generate calibration viewpoints around a seed camera pose.

The seed is the robot's current (live-gated) pose — the operator jogs until the
aiming HUD locks green, and these poses orbit that view. Why not the old dome
macro: it aligned every pose's Z to a look-at point but added NO roll and swept a
full 360deg azimuth — which (a) under-constrains the hand-eye rotation solve
(rotation axes end up near-coplanar) and (b) orbits the camera to the board's
unprintable back. Here we instead sample a CONE around the seed viewing direction
(board stays visible), with deliberate ROLL about the optical axis and DISTANCE
variation — the diversity hand-eye actually needs.

Pure numpy + reproducible (deterministic spiral, no RNG). Poses are TCP/camera
poses in the same frame as ``seed_T``; the caller filters them by reachability.
"""
from __future__ import annotations

import numpy as np

from ...core.geometry import Rt_to_T, T_to_Rt

_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))  # ~2.39996 rad, even azimuthal spread


def _basis_from_z(z: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = z / np.linalg.norm(z)
    if abs(float(np.dot(z, up_hint))) > 0.95:           # up_hint ~parallel to z
        up_hint = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(up_hint, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y


def _apply_roll(x: np.ndarray, y: np.ndarray, deg: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.deg2rad(deg)
    return np.cos(a) * x + np.sin(a) * y, -np.sin(a) * x + np.cos(a) * y


def generate_calibration_poses(
    seed_T: np.ndarray, *, count: int = 15, look_distance_mm: float = 500.0,
    cone_half_angle_deg: float = 32.0, roll_max_deg: float = 75.0,
    distance_jitter: float = 0.12, oversample: int = 3,
) -> list[np.ndarray]:
    """Candidate camera poses orbiting the point ``seed`` looks at.

    Returns ``count * oversample`` candidates (deterministic), ordered so the
    first is closest to the seed view; the caller keeps the first ``count`` that
    are reachable. ``seed_T`` is the camera (TCP) pose; its +Z points at the board.
    """
    seed_R, seed_pos = T_to_Rt(seed_T)
    fwd, up = seed_R[:, 2], seed_R[:, 1]                 # camera forward, up
    right = seed_R[:, 0]
    center = seed_pos + look_distance_mm * fwd           # ~board location
    half = np.deg2rad(cone_half_angle_deg)

    n = max(count * oversample, count)
    poses: list[np.ndarray] = []
    for i in range(n):
        frac = (i + 0.5) / n
        theta = half * np.sqrt(frac)                     # polar from seed dir (denser center)
        phi = i * _GOLDEN_ANGLE                          # azimuth
        roll = roll_max_deg * (((i * 0.6180339887) % 1.0) * 2 - 1)
        dist = look_distance_mm * (1 + distance_jitter * (((i * 0.3819660113) % 1.0) * 2 - 1))

        # viewpoint direction from the board center, within the cone around -fwd
        d = (-np.cos(theta) * fwd
             + np.sin(theta) * (np.cos(phi) * right + np.sin(phi) * up))
        cam_pos = center + dist * d
        z = center - cam_pos                             # look back at the board
        x, y = _basis_from_z(z, up)
        x, y = _apply_roll(x, y, roll)
        poses.append(Rt_to_T(np.column_stack([x, y, z / np.linalg.norm(z)]), cam_pos))
    return poses
