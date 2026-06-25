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
from tasni.core.geometry import Rt_to_T, compose, invert_T, transform_points  # noqa: E402
from tasni.modules.calibration import handeye, quality  # noqa: E402
from tasni.modules.calibration.handeye import CalibrationView  # noqa: E402
from tasni.modules.calibration.intrinsics_calib import solve_intrinsics  # noqa: E402
from tasni.modules.calibration.poses import (  # noqa: E402
    board_visible_fraction, frame_aim_offsets, generate_calibration_poses,
    projected_corner_coverage, select_diverse, select_diverse_with_coverage,
    viewing_angle_span)
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


def test_select_diverse_spreads_orientation_not_just_view():
    """Roll-aware (full-rotation) FPS must avoid keeping two near-identical
    ORIENTATIONS — a +Z-only pick can keep two poses that differ only in roll
    (≈0° apart on the viewing axis, yet distinct rotations the solve wants). So the
    closest pair of selected poses stays well separated in rotation space."""
    from tasni.modules.calibration.poses import _rotation_geodesic
    cc, seed_T, cands = _seed_and_candidates()
    sel = select_diverse(cands, cc.pose_count, seed_fwd=seed_T[:3, 2])
    Rsel = [np.asarray(cands[i], float)[:3, :3] for i in sel]
    mind = min(np.degrees(_rotation_geodesic(Rsel[a], Rsel[b]))
               for a in range(len(Rsel)) for b in range(a + 1, len(Rsel)))
    assert mind > 10.0, f"selected orientations too close: min pair {mind:.1f}°"


def test_board_visible_fraction_flags_off_aim():
    """The visibility predictor: ~1.0 when the camera frames the board, low when it
    aims off it, and 0.0 when the board sits behind the camera."""
    from tasni.core.geometry import transform_points
    from tasni.modules.calibration.poses import board_visible_fraction

    board_center = np.array([800.0, 0.0, 200.0])
    seed_pos = np.array([300.0, 0.0, 560.0])
    T_base_target = syn._look_at(board_center, seed_pos, 0.0)
    board_pts = transform_points(T_base_target, syn._make_board_points())
    size = (1920, 1080)                                   # matches syn.K (1080p)

    on_aim = syn._look_at(seed_pos, board_center, 0.0)
    off_aim = syn._look_at(seed_pos, board_center + np.array([0.0, 0.0, 800.0]), 0.0)
    looking_away = syn._look_at(seed_pos, 2 * seed_pos - board_center, 0.0)

    assert board_visible_fraction(on_aim, board_pts, syn.K, size) > 0.99
    assert board_visible_fraction(off_aim, board_pts, syn.K, size) < 0.5
    assert board_visible_fraction(looking_away, board_pts, syn.K, size) == 0.0


def test_frame_spread_targets_cover_intrinsic_grid():
    """Normal robot targets must supply the edge observations used by the implicit
    intrinsic solve, without relying on a separate camera-only workflow."""
    cc, seed_T, _ = _seed_and_candidates()
    board_center = np.array([800.0, 0.0, 200.0])
    seed_pos = np.array([300.0, 0.0, 560.0])
    look = float(np.linalg.norm(board_center - seed_pos))
    size = (1920, 1080)
    T_base_target = syn._look_at(board_center, seed_pos, 0.0)
    board_pts = transform_points(T_base_target, syn._make_board_points())
    candidates = generate_calibration_poses(
        seed_T, count=cc.pose_count, look_distance_mm=look,
        cone_half_angle_deg=cc.cone_half_angle_deg, roll_max_deg=cc.roll_max_deg,
        distance_jitter=cc.distance_jitter,
        aim_offsets=frame_aim_offsets(
            syn.K, size, edge_fraction=cc.intrinsics_edge_fraction))
    visible = [T for T in candidates
               if board_visible_fraction(
                   T, board_pts, syn.K, size,
                   margin_frac=cc.board_visible_margin_frac)
               >= cc.min_board_visible_frac]
    assert len(visible) >= cc.pose_count
    selected = select_diverse_with_coverage(
        visible, cc.pose_count, board_pts, syn.K, size,
        seed_fwd=seed_T[:3, 2])
    chosen = [visible[i] for i in selected]
    coverage, cells = projected_corner_coverage(chosen, board_pts, syn.K, size)
    assert coverage >= 0.9, (coverage, cells)
    assert viewing_angle_span(chosen, seed_T[:3, 2])[1] <= \
        cc.cone_half_angle_deg + 1e-6
    assert all(any(row[x] for row in cells) for x in range(4)), cells
    assert all(any(c for c in row) for row in cells), cells

    # Prove the generated views actually constrain the lens model, not merely the
    # coverage counter: synthesize distorted corners and recover the coefficients.
    obj = syn._make_board_points().astype(np.float32)
    dist_true = np.array([0.10, -0.15, 0.0015, -0.0008, 0.0])
    obj_views, img_views = [], []
    for cam in chosen:
        T_cam_target = compose(invert_T(cam), T_base_target)
        rvec, _ = cv2.Rodrigues(T_cam_target[:3, :3])
        corners, _ = cv2.projectPoints(
            obj, rvec, T_cam_target[:3, 3], syn.K, dist_true)
        obj_views.append(obj)
        img_views.append(corners.astype(np.float32))
    intr = solve_intrinsics(obj_views, img_views, size, syn.K, fix_k3=True)
    assert intr["coverage_pct"] >= 0.9
    assert np.allclose(intr["dist"][:4], dist_true[:4], atol=2e-3), intr["dist"]


def test_generation_orbits_detected_board_center_not_seed_axis_guess():
    """The operator may be approximately, not perfectly, centered. The detected
    ChArUco center is the geometric truth the generated camera poses must orbit."""
    cc, seed_T, _ = _seed_and_candidates()
    seed_axis_guess = seed_T[:3, 3] + 500.0 * seed_T[:3, 2]
    detected_center = seed_axis_guess + 55.0 * seed_T[:3, 0] - 30.0 * seed_T[:3, 1]
    poses = generate_calibration_poses(
        seed_T, count=cc.pose_count, look_distance_mm=500.0,
        cone_half_angle_deg=cc.cone_half_angle_deg, roll_max_deg=cc.roll_max_deg,
        distance_jitter=cc.distance_jitter, aim_offsets=[(0.0, 0.0)],
        target_center=detected_center)
    for T in poses:
        expected = detected_center - T[:3, 3]
        expected /= np.linalg.norm(expected)
        assert float(np.dot(T[:3, 2], expected)) > 1.0 - 1e-10


def test_large_board_preserves_perpendicular_plane_clearance():
    """Wide cone poses may increase radial range, but never approach the board
    plane closer than the A3 safety/occupancy floor."""
    cc, seed_T, _ = _seed_and_candidates()
    center = seed_T[:3, 3] + 450.0 * seed_T[:3, 2]
    normal = -seed_T[:3, 2]
    poses = generate_calibration_poses(
        seed_T, count=cc.pose_count, look_distance_mm=600.0,
        cone_half_angle_deg=cc.cone_half_angle_deg,
        roll_max_deg=cc.roll_max_deg, distance_jitter=0.08,
        target_center=center, target_normal=normal,
        min_perpendicular_mm=425.0)
    perpendicular = [
        abs(float(np.dot(T[:3, 3] - center, normal))) for T in poses
    ]
    assert min(perpendicular) >= 425.0 - 1e-6
    # The wide candidates must move farther away instead of violating the plane
    # floor; this locks in the radial-vs-perpendicular distinction.
    assert max(np.linalg.norm(T[:3, 3] - center) for T in poses) > 620.0


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
    test_select_diverse_spreads_orientation_not_just_view()
    test_board_visible_fraction_flags_off_aim()
    test_frame_spread_targets_cover_intrinsic_grid()
    test_generation_orbits_detected_board_center_not_seed_axis_guess()
    test_large_board_preserves_perpendicular_plane_clearance()
    test_select_diverse_respects_count_and_membership()
    print("\nPose-generation conditioning check passed.")
