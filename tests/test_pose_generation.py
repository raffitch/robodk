"""Check that auto-generated calibration poses are well-conditioned for hand-eye.

Builds a board + a known X_true, generates poses around a seed view, synthesizes
the board observations from those poses, and confirms TSAI(+refine) recovers
X_true with low error — i.e. the cone+roll generator produces a usable pose set.

    py -3.10 tests/test_pose_generation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasni.core.geometry import Rt_to_T, compose, invert_T  # noqa: E402
from tasni.modules.calibration import handeye, quality  # noqa: E402
from tasni.modules.calibration.handeye import CalibrationView  # noqa: E402
from tasni.modules.calibration.poses import generate_calibration_poses  # noqa: E402
import test_calibration_synthetic as syn  # noqa: E402

K, DIST = syn.K, syn.DIST


def _build():
    board_center = np.array([800.0, 0.0, 200.0])
    seed_pos = np.array([300.0, 0.0, 560.0])
    look = float(np.linalg.norm(board_center - seed_pos))
    seed_T = syn._look_at(seed_pos, board_center, 0.0)          # camera +Z at board
    T_base_target = syn._look_at(board_center, seed_pos, 0.0)   # board faces the cameras
    X_true = Rt_to_T(syn._rot([0.3, 0.2, 1.0], 25), [40.0, -15.0, 55.0])  # cam2flange

    cam_poses = generate_calibration_poses(
        seed_T, count=15, look_distance_mm=look, cone_half_angle_deg=32,
        roll_max_deg=75, distance_jitter=0.12)[:15]
    obj = syn._make_board_points()

    views = []
    for i, cam in enumerate(cam_poses):
        flange = compose(cam, invert_T(X_true))                # cam = flange @ X
        T_cam_target = compose(invert_T(cam), T_base_target)
        pred = handeye.reproject(obj, T_cam_target, K, DIST)
        views.append(CalibrationView(f"TasniCalib {i}", flange,
                                     T_cam_target[:3, :3], T_cam_target[:3, 3],
                                     pred.reshape(-1, 1, 2), obj))
    return X_true, views


def test_generated_poses_solve_well():
    X_true, views = _build()
    assert len(views) == 15
    # first generated pose should sit near the seed (small view-angle change)
    train, val = views[:12], views[12:]

    X = handeye.solve_tsai(train)
    rot = float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(X[:3, :3].T @ X_true[:3, :3])[0])))
    assert rot < 2.0, f"TSAI rotation off by {rot:.2f} deg on generated poses"

    T_bt = handeye.estimate_board_in_base(train, X)
    Xr, T_btr = handeye.refine(train, X, T_bt, K, DIST)
    report = quality.evaluate(train, val, Xr, T_btr, K, DIST, refined=True)
    assert float(np.rad2deg(np.linalg.norm(
        cv2.Rodrigues(Xr[:3, :3].T @ X_true[:3, :3])[0]))) < 1e-2
    assert report.train.rms_px < 1e-2
    assert report.validation.rms_px < 1e-2
    print("[generated poses]\n" + report.summary())


if __name__ == "__main__":
    test_generated_poses_solve_well()
    print("\nPose-generation conditioning check passed.")
