"""Eye-in-hand hand-eye solve (OpenCV ``calibrateHandEye`` TSAI) + refinement.

Frame conventions (T_A_B maps a point in B to A: ``p_A = T_A_B @ p_B``):

* ``T_base_gripper``  robot flange pose in the base frame (the RoboDK target pose)
* ``X = T_gripper_cam``  camera pose in the gripper frame  -- the unknown we solve
* ``T_cam_target``   board pose in the camera frame (from ChArUco detection)
* ``T_base_target``  board pose in the base frame -- fixed; recovered for metrics

The board is stationary, so for every view::

    T_base_target == T_base_gripper @ X @ T_cam_target

TSAI gives ``X``. The optional refinement then jointly nudges ``X`` and the
consensus ``T_base_target`` to minimize reprojection error (the research-backed
post-solve step). We deliberately rebuild the gripper pose cleanly here rather
than reuse the old macro's mixed-convention ``pose_2_Rt`` -- the reprojection
metric in :mod:`quality` is what validates that this chain is correct.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ...core.geometry import Rt_to_T, T_to_Rt, compose, invert_T


@dataclass
class CalibrationView:
    """One captured pose: robot flange + detected board + corners for reprojection."""

    name: str
    T_base_gripper: np.ndarray   # 4x4 flange pose in base (gripper2base)
    R_target2cam: np.ndarray     # 3x3
    t_target2cam: np.ndarray     # (3,)
    corners: np.ndarray          # (N,1,2) detected charuco corner pixels
    obj_points: np.ndarray       # (N,3) board-frame coords (mm)

    @property
    def T_cam_target(self) -> np.ndarray:
        return Rt_to_T(self.R_target2cam, self.t_target2cam)


def solve_tsai(views: list[CalibrationView]) -> np.ndarray:
    """Run OpenCV ``calibrateHandEye`` (TSAI) and return ``X = T_gripper_cam`` (4x4)."""
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for v in views:
        R, t = T_to_Rt(v.T_base_gripper)
        R_g2b.append(R)
        t_g2b.append(t)
        R_t2c.append(v.R_target2cam)
        t_t2c.append(v.t_target2cam)
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI)
    return Rt_to_T(R_cam2gripper, t_cam2gripper.reshape(3))


def average_transform(transforms: list[np.ndarray]) -> np.ndarray:
    """Consensus of rigid transforms: mean translation + Markley quaternion mean."""
    Ts = [np.asarray(T, dtype=float) for T in transforms]
    t_mean = np.mean([T[:3, 3] for T in Ts], axis=0)
    # Markley et al. average rotation: largest eigenvector of sum(q q^T).
    M = np.zeros((4, 4))
    for T in Ts:
        q = _quat_from_R(T[:3, :3])
        M += np.outer(q, q)
    _, vecs = np.linalg.eigh(M)
    q_avg = vecs[:, -1]
    return Rt_to_T(_R_from_quat(q_avg), t_mean)


def estimate_board_in_base(views: list[CalibrationView], X: np.ndarray) -> np.ndarray:
    """Consensus ``T_base_target`` implied by ``X`` across all views."""
    per_view = [compose(v.T_base_gripper, X, v.T_cam_target) for v in views]
    return average_transform(per_view)


def refine(views: list[CalibrationView], X0: np.ndarray, T_bt0: np.ndarray,
           K: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Refine ``(X, T_base_target)`` by minimizing reprojection error (scipy)."""
    from scipy.optimize import least_squares

    p0 = np.concatenate([_se3_from_T(X0), _se3_from_T(T_bt0)])

    def residuals(p: np.ndarray) -> np.ndarray:
        X = _T_from_se3(p[:6])
        T_bt = _T_from_se3(p[6:])
        res = []
        for v in views:
            T_cam_target = compose(invert_T(X), invert_T(v.T_base_gripper), T_bt)
            res.append(_reproj_residual(v, T_cam_target, K, dist))
        return np.concatenate(res) if res else np.zeros(0)

    sol = least_squares(residuals, p0, method="lm")
    return _T_from_se3(sol.x[:6]), _T_from_se3(sol.x[6:])


# -- reprojection helpers (shared with quality.py) -------------------------
def reproject(obj_points: np.ndarray, T_cam_target: np.ndarray,
              K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Project board points (board frame) into the image via ``T_cam_target``."""
    R, t = T_to_Rt(T_cam_target)
    rvec, _ = cv2.Rodrigues(R)
    pts, _ = cv2.projectPoints(obj_points.astype(np.float64), rvec, t, K, dist)
    return pts.reshape(-1, 2)


def _reproj_residual(view: CalibrationView, T_cam_target: np.ndarray,
                     K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    pred = reproject(view.obj_points, T_cam_target, K, dist)
    obs = view.corners.reshape(-1, 2)
    return (pred - obs).ravel()


# -- se3 / quaternion plumbing ---------------------------------------------
def _se3_from_T(T: np.ndarray) -> np.ndarray:
    R, t = T_to_Rt(T)
    rvec, _ = cv2.Rodrigues(R)
    return np.concatenate([rvec.reshape(3), t])


def _T_from_se3(p: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(p[:3].reshape(3, 1))
    return Rt_to_T(R, p[3:6])


def _quat_from_R(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion [w, x, y, z]."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _R_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
