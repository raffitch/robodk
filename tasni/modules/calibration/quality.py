"""Calibration quality metrics — the gap the original macro never reported.

Three numbers, all computed from the *solved* ``X`` (not the per-view measured
board pose), so they actually measure calibration consistency:

* **reprojection error (px)** on the poses used to solve  -- training fit
* **reprojection error (px)** on held-out poses           -- honest generalization
* **board-consistency (mm)** -- spread of the board's recovered base-frame
  position across views. On a real arm the true hand-eye transform is unknowable,
  so this (plus reprojection px) is the standard proxy. A large mm spread with a
  small px error points at depth/robot noise rather than the calibration method
  -- exactly the open question in docs/best-practices-review.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ...core.geometry import compose, invert_T
from .handeye import CalibrationView, reproject


@dataclass
class ViewError:
    name: str
    n_corners: int
    rms_px: float
    max_px: float


@dataclass
class SplitMetrics:
    n_views: int
    rms_px: float
    max_px: float
    per_view: list[ViewError] = field(default_factory=list)


@dataclass
class CalibrationReport:
    refined: bool
    X_cam2gripper: list[list[float]]
    T_base_target: list[list[float]]
    train: SplitMetrics
    validation: SplitMetrics | None
    board_consistency_mm: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"solver: TSAI{' + reprojection refinement' if self.refined else ''}",
            f"train  ({self.train.n_views} poses): "
            f"reproj RMS {self.train.rms_px:.3f} px, max {self.train.max_px:.3f} px",
        ]
        if self.validation:
            lines.append(
                f"val    ({self.validation.n_views} poses): "
                f"reproj RMS {self.validation.rms_px:.3f} px, "
                f"max {self.validation.max_px:.3f} px")
        bc = self.board_consistency_mm
        lines.append(
            f"board consistency: RMS {bc['rms']:.3f} mm, max {bc['max']:.3f} mm")
        return "\n".join(lines)


def _view_error(view: CalibrationView, X: np.ndarray, T_base_target: np.ndarray,
                K: np.ndarray, dist: np.ndarray) -> ViewError:
    # Predicted board-in-camera from the calibration, not the measured detection.
    T_cam_target = compose(invert_T(X), invert_T(view.T_base_gripper), T_base_target)
    pred = reproject(view.obj_points, T_cam_target, K, dist)
    obs = view.corners.reshape(-1, 2)
    d = np.linalg.norm(pred - obs, axis=1)
    return ViewError(view.name, int(d.shape[0]),
                     float(np.sqrt(np.mean(d ** 2))), float(np.max(d)))


def _split_metrics(views: list[CalibrationView], X: np.ndarray,
                   T_base_target: np.ndarray, K: np.ndarray,
                   dist: np.ndarray) -> SplitMetrics:
    per_view = [_view_error(v, X, T_base_target, K, dist) for v in views]
    # Aggregate over all corners (weight by corner count), not mean-of-means.
    sq = [e.rms_px ** 2 * e.n_corners for e in per_view]
    n = sum(e.n_corners for e in per_view)
    rms = float(np.sqrt(sum(sq) / n)) if n else 0.0
    max_px = max((e.max_px for e in per_view), default=0.0)
    return SplitMetrics(len(views), rms, max_px, per_view)


def board_consistency_mm(views: list[CalibrationView], X: np.ndarray) -> dict[str, float]:
    """Spread (mm) of the board's recovered base-frame origin across views."""
    origins = np.array([compose(v.T_base_gripper, X, v.T_cam_target)[:3, 3]
                        for v in views])
    centroid = origins.mean(axis=0)
    d = np.linalg.norm(origins - centroid, axis=1)
    return {"rms": float(np.sqrt(np.mean(d ** 2))), "max": float(np.max(d))}


def evaluate(train: list[CalibrationView], validation: list[CalibrationView],
             X: np.ndarray, T_base_target: np.ndarray, K: np.ndarray,
             dist: np.ndarray, *, refined: bool) -> CalibrationReport:
    return CalibrationReport(
        refined=refined,
        X_cam2gripper=np.asarray(X).tolist(),
        T_base_target=np.asarray(T_base_target).tolist(),
        train=_split_metrics(train, X, T_base_target, K, dist),
        validation=_split_metrics(validation, X, T_base_target, K, dist) if validation else None,
        board_consistency_mm=board_consistency_mm(train + validation, X),
    )
