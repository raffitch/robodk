"""Synthetic ground-truth checks for the calibration math.

Builds a known hand-eye transform ``X_true`` and a fixed board, generates views
by projecting board corners from a realistic look-at dome, then verifies the
pipeline recovers ``X_true`` and reports the right reprojection error. Validates
the frame conventions in handeye.py / quality.py end-to-end with no robot or
camera. Run directly:

    py -3.10 tests/test_calibration_synthetic.py

Note on the solver: OpenCV's TSAI (Tsai-Lenz) is numerically fragile as the
camera->gripper mount rotation grows toward 180deg (its rotation parameterization
degenerates there) -- PARK/HORAUD/ANDREFF stay exact on the same data. We keep
TSAI per the project decision and rely on the reprojection metric + refinement;
``test_metrics_flag_a_bad_solve`` pins that the metric makes such a failure
visible instead of silently shipping a wrong calibration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core.geometry import Rt_to_T, compose, invert_T  # noqa: E402
from tasni.modules.calibration import handeye, quality  # noqa: E402
from tasni.modules.calibration.handeye import CalibrationView  # noqa: E402

K = np.array([[1362.15, 0, 975.35], [0, 1362.21, 549.97], [0, 0, 1]])
DIST = np.zeros((5, 1))


def _rot(axis, deg):
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    R, _ = cv2.Rodrigues(ax * np.deg2rad(deg))
    return R


# A realistic eye-in-hand mount: camera tilted ~25deg off the flange, offset a
# few cm. Well inside TSAI's well-behaved range.
FRIENDLY_X = Rt_to_T(_rot([0.3, 0.2, 1.0], 25), [40.0, -15.0, 55.0])
# A pathological ~180deg flip mount that breaks OpenCV's TSAI specifically.
SINGULAR_X = Rt_to_T(_rot([1, 0, 0], 180) @ _rot([0, 0, 1], 12), [35.0, -10.0, 60.0])


def _make_board_points():
    # 7x5 inner-corner grid at 47 mm pitch, centered, board frame z=0.
    xs = (np.arange(7) - 3) * 47.0
    ys = (np.arange(5) - 2) * 47.0
    return np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float64)


def _look_at(cam_pos, target_pos, roll_deg, up=(0, 0, 1)):
    """Camera pose in base whose +z optical axis points at the board center,
    with a roll about that axis (roll variety -> non-parallel rotation axes)."""
    cam_pos = np.asarray(cam_pos, float)
    fwd = np.asarray(target_pos, float) - cam_pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(np.asarray(up, float), fwd)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.column_stack([right, down, fwd]) @ _rot([0, 0, 1], roll_deg)
    return Rt_to_T(R, cam_pos)


def _build_views(X_true=FRIENDLY_X, *, n=18, noise_px=0.0, seed=1):
    rng = np.random.default_rng(seed)
    T_base_target = Rt_to_T(_rot([1, 0, 0], 5), [800.0, 0.0, 200.0])
    board_center = T_base_target[:3, 3]
    obj = _make_board_points()

    views = []
    for i in range(n):
        az = np.deg2rad(rng.uniform(-70, 70))
        el = np.deg2rad(rng.uniform(15, 75))
        r = rng.uniform(450, 650)
        cam_pos = board_center + r * np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        T_base_cam = _look_at(cam_pos, board_center, rng.uniform(-120, 120))
        T_base_gripper = compose(T_base_cam, invert_T(X_true))   # cam = gripper @ X
        T_cam_target = compose(invert_T(T_base_cam), T_base_target)
        pred = handeye.reproject(obj, T_cam_target, K, DIST)
        if noise_px:
            pred = pred + rng.normal(0, noise_px, pred.shape)
        views.append(CalibrationView(
            f"Target {i}", T_base_gripper,
            T_cam_target[:3, :3], T_cam_target[:3, 3],
            pred.reshape(-1, 1, 2).astype(np.float64), obj))
    return X_true, T_base_target, views


def _rot_err_deg(A, B):
    return float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(A[:3, :3].T @ B[:3, :3])[0])))


def test_recovers_ground_truth():
    X_true, _, views = _build_views()
    train, val = views[:12], views[12:]

    X = handeye.solve_tsai(views)
    assert _rot_err_deg(X, X_true) < 2.0
    assert np.linalg.norm(X[:3, 3] - X_true[:3, 3]) < 3.0

    # Refinement should pull a clean solve all the way to ground truth.
    T_bt = handeye.estimate_board_in_base(train, X)
    Xr, T_btr = handeye.refine(train, X, T_bt, K, DIST)
    report = quality.evaluate(train, val, Xr, T_btr, K, DIST, refined=True)
    assert _rot_err_deg(Xr, X_true) < 1e-2
    assert np.linalg.norm(Xr[:3, 3] - X_true[:3, 3]) < 1e-2
    assert report.train.rms_px < 1e-3
    assert report.validation.rms_px < 1e-3
    print("[perfect+refined]\n" + report.summary())


def test_noise_is_bounded_and_refine_does_not_overfit():
    X_true, _, views = _build_views(noise_px=0.5, seed=4)
    train, val = views[:12], views[12:]
    X = handeye.solve_tsai(views)
    base = quality.evaluate(train, val, X, handeye.estimate_board_in_base(views, X),
                            K, DIST, refined=False)
    Xr, T_btr = handeye.refine(train, X, handeye.estimate_board_in_base(train, X), K, DIST)
    ref = quality.evaluate(train, val, Xr, T_btr, K, DIST, refined=True)
    assert 0.0 < base.train.rms_px < 3.0
    assert ref.train.rms_px <= base.train.rms_px + 1e-6   # never worse on train
    assert ref.validation.rms_px < 3.0                    # generalizes, no blow-up
    print("[noisy base]\n" + base.summary())
    print("[noisy refined]\n" + ref.summary())


def test_metrics_flag_a_bad_solve():
    """A near-180deg mount makes OpenCV TSAI return garbage; the reprojection
    metric must expose it (the old macro reported nothing and would apply it)."""
    X_true, _, views = _build_views(X_true=SINGULAR_X)
    train, val = views[:12], views[12:]
    X = handeye.solve_tsai(views)
    report = quality.evaluate(train, val, X, handeye.estimate_board_in_base(views, X),
                              K, DIST, refined=False)
    assert _rot_err_deg(X, X_true) > 30.0          # TSAI really did fail here
    assert report.train.rms_px > 50.0              # ...and the metric shows it
    print(f"[bad solve caught] TSAI rot_err {_rot_err_deg(X, X_true):.1f} deg, "
          f"reproj RMS {report.train.rms_px:.1f} px")


if __name__ == "__main__":
    test_recovers_ground_truth()
    test_noise_is_bounded_and_refine_does_not_overfit()
    test_metrics_flag_a_bad_solve()
    print("\nAll synthetic calibration checks passed.")
