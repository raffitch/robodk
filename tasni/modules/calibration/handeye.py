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

from dataclasses import dataclass, field

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
    capture: dict = field(default_factory=dict)

    @property
    def T_cam_target(self) -> np.ndarray:
        return Rt_to_T(self.R_target2cam, self.t_target2cam)


# OpenCV's linear hand-eye solvers. TSAI degenerates as the camera->flange mount
# rotation approaches 180deg (its rotation parameterization is singular there);
# PARK/HORAUD/ANDREFF/DANIILIDIS stay exact on identical data. So rather than
# trust one, ``solve_best`` runs them all and keeps whichever reprojects best.
_HANDEYE_METHODS: dict[str, int] = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def solve_handeye(views: list[CalibrationView], method: str = "TSAI") -> np.ndarray:
    """Run OpenCV ``calibrateHandEye`` with one method, return ``X = T_gripper_cam``."""
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for v in views:
        R, t = T_to_Rt(v.T_base_gripper)
        R_g2b.append(R)
        t_g2b.append(t)
        R_t2c.append(v.R_target2cam)
        t_t2c.append(v.t_target2cam)
    R_c2g, t_c2g = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=_HANDEYE_METHODS[method])
    return Rt_to_T(R_c2g, t_c2g.reshape(3))


def solve_tsai(views: list[CalibrationView]) -> np.ndarray:
    """TSAI hand-eye solve (kept for callers/tests that pin the method)."""
    return solve_handeye(views, "TSAI")


def reproj_rms(views: list[CalibrationView], X: np.ndarray, T_base_target: np.ndarray,
               K: np.ndarray, dist: np.ndarray) -> float:
    """Aggregate reprojection RMS (px) of ``X`` over ``views`` — corner-weighted,
    same convention as :mod:`quality`. Used to rank solvers."""
    sq = 0.0
    n = 0
    for v in views:
        T_cam_target = compose(invert_T(X), invert_T(v.T_base_gripper), T_base_target)
        pred = reproject(v.obj_points, T_cam_target, K, dist)
        d = np.linalg.norm(pred - v.corners.reshape(-1, 2), axis=1)
        sq += float(np.sum(d ** 2))
        n += int(d.shape[0])
    return float(np.sqrt(sq / n)) if n else 0.0


def solve_best(views: list[CalibrationView], K: np.ndarray, dist: np.ndarray,
               methods: list[str] | None = None
               ) -> tuple[np.ndarray, str, list[tuple[str, float]]]:
    """Solve with every linear method and keep the lowest-reprojection one.

    Returns ``(X, method_name, ranking)`` with ``ranking = [(method, rms_px), ...]``
    sorted best-first. Cheap and robust: the linear solves cost microseconds
    against data already captured, and letting PARK/HORAUD/ANDREFF/DANIILIDIS win
    when TSAI degenerates removes the near-180deg-mount singularity entirely.
    """
    methods = methods or list(_HANDEYE_METHODS)
    scored: list[tuple[float, str, np.ndarray]] = []
    for m in methods:
        try:
            X = solve_handeye(views, m)
            rms = reproj_rms(views, X, estimate_board_in_base(views, X), K, dist)
        except cv2.error:
            continue
        if np.isfinite(rms):
            scored.append((rms, m, X))
    if not scored:
        raise RuntimeError("all hand-eye solvers failed on these views")
    scored.sort(key=lambda s: s[0])
    rms, method, X = scored[0]
    return X, method, [(m, r) for r, m, _ in scored]


def _solve_for_cv(train: list[CalibrationView], method: str,
                  K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Solve one cross-val fold. ``method == "best"`` re-runs the whole
    multi-method selection *inside* the fold, so the reported cross-val number
    accounts for the method-selection step (otherwise it would be optimistically
    biased toward the method already chosen on the full set)."""
    if method == "best":
        X, _, _ = solve_best(train, K, dist)
        return X
    return solve_handeye(train, method)


def cross_validate(views: list[CalibrationView], method: str, K: np.ndarray,
                   dist: np.ndarray, folds: int = 5) -> float | None:
    """K-fold held-out reprojection RMS (px) for ``method`` — an honest
    generalization number independent of the single train/val split, and cheap
    (linear solves only, no refinement). ``None`` if there are too few views.
    ``method == "best"`` selects the best solver per fold (see :func:`_solve_for_cv`)."""
    n = len(views)
    folds = min(folds, n)
    if folds < 2 or n < 4:
        return None
    sq = 0.0
    count = 0
    for f in range(folds):
        test = views[f::folds]
        train = [v for i, v in enumerate(views) if i % folds != f]
        if len(train) < 3 or not test:
            continue
        try:
            X = _solve_for_cv(train, method, K, dist)
            T_bt = estimate_board_in_base(train, X)
        except (cv2.error, RuntimeError):
            continue
        for v in test:
            T_cam_target = compose(invert_T(X), invert_T(v.T_base_gripper), T_bt)
            pred = reproject(v.obj_points, T_cam_target, K, dist)
            d = np.linalg.norm(pred - v.corners.reshape(-1, 2), axis=1)
            sq += float(np.sum(d ** 2))
            count += int(d.shape[0])
    return float(np.sqrt(sq / count)) if count else None


def per_view_reproj_px(views: list[CalibrationView], X: np.ndarray,
                       T_base_target: np.ndarray, K: np.ndarray,
                       dist: np.ndarray) -> np.ndarray:
    """Per-view reprojection RMS (px) of ``X`` — one number per view, in order."""
    out = []
    for v in views:
        T_cam_target = compose(invert_T(X), invert_T(v.T_base_gripper), T_base_target)
        pred = reproject(v.obj_points, T_cam_target, K, dist)
        d = np.linalg.norm(pred - v.corners.reshape(-1, 2), axis=1)
        out.append(float(np.sqrt(np.mean(d ** 2))))
    return np.asarray(out)


def reject_outliers(views: list[CalibrationView], X: np.ndarray,
                    T_base_target: np.ndarray, K: np.ndarray, dist: np.ndarray,
                    *, abs_px: float = 3.0, factor: float = 3.0, min_keep: int = 6
                    ) -> tuple[list[CalibrationView], list[str], float]:
    """Drop views whose reprojection RMS is an outlier, returning
    ``(kept_views, dropped_names, threshold_px)``.

    A view is an outlier only if it exceeds ``max(abs_px, factor * median)`` — i.e.
    it must be both large in absolute terms *and* large relative to the rest. This
    is deliberately conservative (a clean capture drops nothing) and robust to the
    overall noise floor (a noisy-but-consistent set isn't decimated). Refuses to cut
    below ``min_keep`` survivors, so rejection can never starve the solve.
    """
    if not views:
        return list(views), [], float("inf")
    errs = per_view_reproj_px(views, X, T_base_target, K, dist)
    thresh = max(abs_px, factor * float(np.median(errs)))
    under = errs <= thresh
    if under.sum() >= min_keep:
        keep_mask = under
    else:
        # Too few survivors under the threshold — keep the min_keep lowest-error
        # views instead (if there are fewer than min_keep total, keep them all).
        keep_mask = np.zeros(len(errs), dtype=bool)
        keep_mask[np.argsort(errs)[:min_keep]] = True
    kept = [v for v, k in zip(views, keep_mask) if k]
    dropped = [v.name for v, k in zip(views, keep_mask) if not k]
    return kept, dropped, thresh


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
