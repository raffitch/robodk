"""Pure-numpy rigid-transform helpers.

No RoboDK / OpenCV dependency lives here so the math is unit-testable on any
machine. RoboDK ``robomath.Mat`` <-> numpy bridges live in :mod:`tasni.core.rdk_io`
(that module imports ``robodk``); OpenCV Rodrigues conversions stay in the
calibration module where ``cv2`` is already required.

Conventions
-----------
A transform ``T`` is a 4x4 homogeneous matrix mapping a point in frame *A* to
frame *B*: ``p_B = T_B_A @ p_A``. We name transforms ``T_<to>_<from>`` so chains
read left-to-right, e.g. ``T_base_cam = T_base_gripper @ T_gripper_cam``.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "Rt_to_T",
    "T_to_Rt",
    "invert_T",
    "compose",
    "transform_points",
]


def Rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3x3 rotation and 3-vector."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def T_to_Rt(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 4x4 transform into ``(R 3x3, t 3-vector)``."""
    T = np.asarray(T, dtype=float).reshape(4, 4)
    return T[:3, :3].copy(), T[:3, 3].copy()


def invert_T(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 transform (transpose-rotation form, no solve)."""
    R, t = T_to_Rt(T)
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def compose(*transforms: np.ndarray) -> np.ndarray:
    """Left-to-right product of 4x4 transforms: ``compose(A, B, C) == A @ B @ C``."""
    out = np.eye(4)
    for T in transforms:
        out = out @ np.asarray(T, dtype=float).reshape(4, 4)
    return out


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an ``(N, 3)`` array of points."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    R, t = T_to_Rt(T)
    return pts @ R.T + t
