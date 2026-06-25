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

import cv2
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
    method: str
    X_cam2gripper: list[list[float]]
    T_base_target: list[list[float]]
    train: SplitMetrics
    validation: SplitMetrics | None
    board_consistency_mm: dict[str, float]
    motion_diversity: dict
    method_ranking: list | None = None
    intrinsics_check: dict | None = None
    cross_val_rms_px: float | None = None
    rejected_views: list = field(default_factory=list)
    diagnosis: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"solver: {self.method}{' + reprojection refinement' if self.refined else ''}",
            f"train  ({self.train.n_views} poses): "
            f"reproj RMS {self.train.rms_px:.3f} px, max {self.train.max_px:.3f} px",
        ]
        if self.validation:
            lines.append(
                f"val    ({self.validation.n_views} poses): "
                f"reproj RMS {self.validation.rms_px:.3f} px, "
                f"max {self.validation.max_px:.3f} px")
        if self.cross_val_rms_px is not None:
            lines.append(
                f"cross-val (k-fold): reproj RMS {self.cross_val_rms_px:.3f} px")
        bc = self.board_consistency_mm
        lines.append(
            f"board consistency: RMS {bc['rms']:.3f} mm, max {bc['max']:.3f} mm")
        md = self.motion_diversity
        lines.append(
            f"motion diversity: axis-spread {md['axis_spread']:.2f}, "
            f"rot {md['min_pair_deg']:.0f}-{md['max_pair_deg']:.0f} deg"
            f"{'' if md['well_conditioned'] else '  [WEAK - re-seed]'}")
        if self.rejected_views:
            lines.append(f"outliers rejected ({len(self.rejected_views)}): "
                         f"{', '.join(self.rejected_views)}")
        if self.intrinsics_check and self.intrinsics_check.get("warn"):
            lines.append(f"intrinsics: WARNING - {self.intrinsics_check['note']}")
        if self.diagnosis:
            lines.append(f"verdict: {self.diagnosis['verdict'].upper()} - "
                         f"{self.diagnosis['headline']}")
            for cause in self.diagnosis.get("causes", []):
                lines.append(f"  - {cause}")
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


def motion_diversity(views: list[CalibrationView]) -> dict:
    """Conditioning of the captured motion set. Hand-eye is well-posed only when
    the relative rotations between views span multiple axes; if their axes are
    near-coplanar (or the rotations are tiny) the solve is under-constrained and a
    low reprojection error can still hide a bad ``X``. Pure diagnostic — surfaces a
    poor seed / pose set before the numbers are trusted.

    ``axis_spread`` is the smallest/largest eigenvalue ratio of the rotation-axis
    scatter matrix (0 = coplanar/degenerate, ->1 = isotropic).
    """
    Rs = [np.asarray(v.T_base_gripper, dtype=float)[:3, :3] for v in views]
    axes, angles = [], []
    for i in range(len(Rs)):
        for j in range(i + 1, len(Rs)):
            rvec, _ = cv2.Rodrigues(Rs[i].T @ Rs[j])
            ang = float(np.linalg.norm(rvec))
            if ang > 1e-6:
                axes.append(rvec.reshape(3) / ang)
                angles.append(float(np.rad2deg(ang)))
    if len(axes) < 2:
        return {"axis_spread": 0.0, "min_pair_deg": 0.0, "max_pair_deg": 0.0,
                "n_pairs": len(axes), "well_conditioned": False,
                "note": "not enough distinct rotations between poses"}
    A = np.asarray(axes)
    eig = np.linalg.eigvalsh(A.T @ A / len(A))     # ascending; isotropic -> all ~1/3
    axis_spread = float(eig[0] / eig[-1]) if eig[-1] > 0 else 0.0
    well = axis_spread >= 0.05 and max(angles) >= 30.0
    return {
        "axis_spread": axis_spread,
        "min_pair_deg": float(min(angles)), "max_pair_deg": float(max(angles)),
        "n_pairs": len(axes), "well_conditioned": bool(well),
        "note": ("good rotational diversity" if well else
                 "weak rotational diversity - poses may be near-coplanar; re-seed "
                 "at a more varied view"),
    }


# Diagnosis thresholds (px / mm). Tuned to the live-gate working distance
# (~450 mm) and a 30 mm-square ChArUco board; deliberately conservative — a
# marginal solve reads "borderline" rather than silently passing. The reprojection
# bands match the UI colour bands (good <1, warn <3) in Calibration.tsx.
REPROJ_PASS_PX = 1.0
REPROJ_FAIL_PX = 3.0
BOARDCONS_PASS_MM = 1.5
BOARDCONS_FAIL_MM = 5.0
OVERFIT_RATIO = 2.5      # validation/train reprojection above this => suspect overfit


def transform_repeatability(current: np.ndarray, previous: np.ndarray,
                            reference_distance_mm: float = 500.0) -> dict:
    """Difference between two independently solved camera mounts."""
    a = np.asarray(current, dtype=float)
    b = np.asarray(previous, dtype=float)
    translation_mm = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    relative_R = a[:3, :3] @ b[:3, :3].T
    rotation_deg = float(np.degrees(np.arccos(np.clip(
        (np.trace(relative_R) - 1.0) / 2.0, -1.0, 1.0))))
    lever_mm = 2.0 * float(reference_distance_mm) * np.sin(
        np.deg2rad(rotation_deg) / 2.0)
    reference_delta_mm = float(translation_mm + lever_mm)
    high_confidence = translation_mm <= 1.0 and rotation_deg <= 0.2
    return {
        "translation_mm": translation_mm,
        "rotation_deg": rotation_deg,
        "reference_distance_mm": float(reference_distance_mm),
        "reference_delta_mm": reference_delta_mm,
        "high_confidence": bool(high_confidence),
        "note": (
            "repeat calibration agrees within the high-confidence band"
            if high_confidence else
            "repeat calibration differs beyond 1.0 mm or 0.2 deg; validate "
            "against a known point before claiming sub-millimetre accuracy"),
    }


def diagnose(report: "CalibrationReport") -> dict:
    """Turn the metric *pattern* into an operator verdict — a pass/borderline/fail
    headline plus the most-likely cause(s), each a next action. Pure function of the
    report numbers (no new measurement). Uses the honest reprojection figure (held-out
    validation if present, else training).

    The key discriminator: a high reprojection error with a *tight* board-consistency
    spread blames the camera model (intrinsics/distortion); a large spread blames the
    geometry (robot pose / depth noise) — the open question in the best-practices review.
    """
    train = report.train.rms_px
    val = report.validation.rms_px if report.validation else None
    reproj = val if val is not None else train
    bc = report.board_consistency_mm["rms"]
    well = bool(report.motion_diversity.get("well_conditioned", True))
    intr_warn = bool(report.intrinsics_check and report.intrinsics_check.get("warn"))

    mid_reproj = reproj >= REPROJ_PASS_PX
    high_reproj = reproj >= REPROJ_FAIL_PX
    mid_spread = bc >= BOARDCONS_PASS_MM
    high_spread = bc >= BOARDCONS_FAIL_MM

    causes: list[str] = []
    if mid_reproj and not mid_spread:
        causes.append("High reprojection with a tight board-consistency spread points "
                      "at the camera model — likely intrinsics/distortion (the factory "
                      "K or zero distortion may be wrong). Verify the camera intrinsics.")
    if mid_spread:
        causes.append("Large board-consistency spread points at geometry, not the lens "
                      "— likely robot-pose accuracy or depth/marker noise. Check the "
                      "robot's absolute accuracy and the board's rigidity.")
    if intr_warn:
        causes.append("The intrinsics self-check disagrees with the configured camera "
                      "matrix — recalibrate the camera intrinsics or update the config.")
    if not well:
        causes.append("Weak motion diversity — poses may be near-coplanar, so the errors "
                      "can read optimistically. Re-seed at a more varied view and recapture.")
    if val is not None and train > 0 and val >= OVERFIT_RATIO * train and mid_reproj:
        causes.append("Validation error is much larger than the training fit — possible "
                      "overfit; capture more (and more varied) poses.")

    if high_reproj or high_spread:
        verdict = "fail"
        headline = "Fail — do not apply; this calibration is not trustworthy yet."
    elif mid_reproj or mid_spread or intr_warn or not well:
        verdict = "borderline"
        headline = "Borderline — usable, but address the cause below before relying on it."
    else:
        verdict = "pass"
        headline = "Pass — reprojection, board-consistency and conditioning are all in band."
        causes.append("Reprojection and board-consistency are tight and the motion set "
                      "is well-conditioned.")
    return {"verdict": verdict, "headline": headline, "causes": causes}


def evaluate(train: list[CalibrationView], validation: list[CalibrationView],
             X: np.ndarray, T_base_target: np.ndarray, K: np.ndarray,
             dist: np.ndarray, *, refined: bool, method: str = "TSAI",
             method_ranking: list | None = None, intrinsics_check: dict | None = None,
             cross_val_rms_px: float | None = None,
             rejected_views: list | None = None) -> CalibrationReport:
    report = CalibrationReport(
        refined=refined,
        method=method,
        X_cam2gripper=np.asarray(X).tolist(),
        T_base_target=np.asarray(T_base_target).tolist(),
        train=_split_metrics(train, X, T_base_target, K, dist),
        validation=_split_metrics(validation, X, T_base_target, K, dist) if validation else None,
        board_consistency_mm=board_consistency_mm(train + validation, X),
        motion_diversity=motion_diversity(train + validation),
        method_ranking=method_ranking,
        intrinsics_check=intrinsics_check,
        cross_val_rms_px=cross_val_rms_px,
        rejected_views=list(rejected_views or []),
    )
    report.diagnosis = diagnose(report)
    return report
