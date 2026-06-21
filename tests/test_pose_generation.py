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

from tasni.core.config import CalibrationConfig  # noqa: E402
from tasni.core.geometry import Rt_to_T, compose, invert_T  # noqa: E402
from tasni.modules.calibration import handeye, quality  # noqa: E402
from tasni.modules.calibration.handeye import CalibrationView  # noqa: E402
from tasni.modules.calibration.poses import (  # noqa: E402
    generate_calibration_poses, select_diverse, viewing_angle_span)
import test_calibration_synthetic as syn  # noqa: E402

K, DIST = syn.K, syn.DIST


def _build():
    board_center = np.array([800.0, 0.0, 200.0])
    seed_pos = np.array([300.0, 0.0, 560.0])
    look = float(np.linalg.norm(board_center - seed_pos))
    seed_T = syn._look_at(seed_pos, board_center, 0.0)          # camera +Z at board
    T_base_target = syn._look_at(board_center, seed_pos, 0.0)   # board faces the cameras
    X_true = Rt_to_T(syn._rot([0.3, 0.2, 1.0], 25), [40.0, -15.0, 55.0])  # cam2flange

    cc = CalibrationConfig()                          # use the production defaults
    cam_poses = generate_calibration_poses(
        seed_T, count=cc.pose_count, look_distance_mm=look,
        cone_half_angle_deg=cc.cone_half_angle_deg, roll_max_deg=cc.roll_max_deg,
        distance_jitter=cc.distance_jitter)[:cc.pose_count]
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

    # The widened cone (45deg default) must keep the pose set well-conditioned;
    # lock in the diversity gain so a narrower cone can't silently regress it.
    md = report.motion_diversity
    assert md["well_conditioned"], f"generated set not well-conditioned: {md}"
    assert md["axis_spread"] > 0.15, \
        f"widened cone should lift axis-spread, got {md['axis_spread']:.3f}"
    print("[generated poses]\n" + report.summary())


def _seed_and_candidates():
    cc = CalibrationConfig()
    board_center = np.array([800.0, 0.0, 200.0])
    seed_pos = np.array([300.0, 0.0, 560.0])
    look = float(np.linalg.norm(board_center - seed_pos))
    seed_T = syn._look_at(seed_pos, board_center, 0.0)
    cands = generate_calibration_poses(
        seed_T, count=cc.pose_count, look_distance_mm=look,
        cone_half_angle_deg=cc.cone_half_angle_deg, roll_max_deg=cc.roll_max_deg,
        distance_jitter=cc.distance_jitter)
    return cc, seed_T, cands


def test_select_diverse_beats_first_n():
    """Choosing a spread (FPS) instead of the first N reachable must widen the
    effective cone — the fix for the workspace-edge clustering finding."""
    cc, seed_T, cands = _seed_and_candidates()
    seed_fwd = seed_T[:3, 2]

    first_n = cands[: cc.pose_count]
    sel = select_diverse(cands, cc.pose_count, seed_fwd=seed_fwd)
    diverse = [cands[i] for i in sel]

    _, max_first, mean_first = viewing_angle_span(first_n, seed_fwd)
    _, max_div, mean_div = viewing_angle_span(diverse, seed_fwd)
    # FPS should reach noticeably wider than the innermost-N spiral prefix.
    assert max_div > max_first + 5.0, (max_div, max_first)
    assert mean_div > mean_first, (mean_div, mean_first)
    # Anchored at the most fronto-parallel pose -> at least one easy-detect view.
    assert min(viewing_angle_span(diverse, seed_fwd)[:1]) < 15.0


def test_select_diverse_respects_count_and_membership():
    cc, seed_T, cands = _seed_and_candidates()
    # Fewer reachable than requested -> keep all, no duplication.
    sub = cands[:4]
    sel = select_diverse(sub, cc.pose_count, seed_fwd=seed_T[:3, 2])
    assert sel == [0, 1, 2, 3]
    # Indices are unique, in range, and sorted (stable target naming).
    sel = select_diverse(cands, cc.pose_count, seed_fwd=seed_T[:3, 2])
    assert len(sel) == cc.pose_count == len(set(sel))
    assert sel == sorted(sel)
    assert all(0 <= i < len(cands) for i in sel)


if __name__ == "__main__":
    test_generated_poses_solve_well()
    test_select_diverse_beats_first_n()
    test_select_diverse_respects_count_and_membership()
    print("\nPose-generation conditioning check passed.")
