"""Pure-math checks for the live aiming gate (core/aiming.py) — no camera, no RoboDK.

Builds ChArUco detections with known distance/tilt/offset and asserts the gate
lamps light exactly when the board is in the ideal band.

    py -3.10 tests/test_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core.aiming import (  # noqa: E402
    GateThresholds, board_tilt_deg, evaluate_gate)
from tasni.modules.calibration.charuco import ViewDetection  # noqa: E402

W, H = 1920, 1080
K = np.array([[1362.15, 0, W / 2], [0, 1362.21, H / 2], [0, 0, 1]], dtype=float)
TH = GateThresholds(min_corners=6, ideal_distance_mm=450.0,
                    distance_tol_mm=80.0, max_tilt_deg=25.0)


def _det(*, n=20, distance=450.0, tilt_deg=0.0, tx=0.0, ty=0.0) -> ViewDetection:
    """A detection whose board sits at ``distance`` mm, tilted ``tilt_deg`` about
    the camera X axis, centred unless tx/ty given (mm in the camera frame)."""
    rvec = np.array([np.deg2rad(tilt_deg), 0.0, 0.0]).reshape(3, 1)
    tz = float(np.sqrt(max(distance**2 - tx**2 - ty**2, 0.0)))
    tvec = np.array([tx, ty, tz]).reshape(3, 1)
    ids = np.arange(n).reshape(-1, 1).astype(np.int32)
    corners = np.zeros((n, 1, 2), np.float32)
    obj = np.zeros((n, 3), np.float32)
    return ViewDetection(corners, ids, obj, rvec, tvec)


def test_tilt_metric():
    assert board_tilt_deg(cv2.Rodrigues(np.array([0.0, 0, 0]))[0]) < 1e-6
    for deg in (10, 30, 60):
        R = cv2.Rodrigues(np.array([np.deg2rad(deg), 0, 0]))[0]
        assert abs(board_tilt_deg(R) - deg) < 1e-6
    # sign-agnostic: a board whose normal points back at the camera is still 0
    R180 = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    assert board_tilt_deg(R180) < 1e-6


def test_all_green_when_ideal():
    g = evaluate_gate(_det(distance=450, tilt_deg=5), K, (H, W), TH)
    assert g.detected and g.ok
    assert g.gates == {"detected": True, "distance": True, "angle": True}
    assert abs(g.distance_mm - 450) < 1e-6 and g.tilt_deg < 6


def test_distance_gate():
    assert not evaluate_gate(_det(distance=600), K, (H, W), TH).gates["distance"]
    assert not evaluate_gate(_det(distance=300), K, (H, W), TH).gates["distance"]
    assert evaluate_gate(_det(distance=450 + 79), K, (H, W), TH).gates["distance"]
    assert not evaluate_gate(_det(distance=450 + 81), K, (H, W), TH).ok


def test_angle_gate():
    assert evaluate_gate(_det(tilt_deg=24), K, (H, W), TH).gates["angle"]
    g = evaluate_gate(_det(tilt_deg=40), K, (H, W), TH)
    assert not g.gates["angle"] and not g.ok and g.tilt_deg > 39


def test_corner_and_none_gates():
    assert not evaluate_gate(_det(n=4), K, (H, W), TH).gates["detected"]
    none = evaluate_gate(None, K, (H, W), TH)
    assert not none.detected and not none.ok and none.distance_mm is None
    assert none.to_dict()["offset"] is None


def test_offset_sign():
    g = evaluate_gate(_det(tx=100), K, (H, W), TH)   # board to the right of centre
    assert g.offset[0] > 0.05 and abs(g.offset[1]) < 1e-3


def test_board_center_reference():
    # Board fronto-parallel with its CORNER origin on the optical axis at 450 mm.
    det = _det(distance=450, tilt_deg=0)             # tvec = [0,0,450]
    center = np.array([120.0, 90.0, 0.0])            # centre is 120mm right, 90mm down
    g0 = evaluate_gate(det, K, (H, W), TH)                          # corner reference
    gc = evaluate_gate(det, K, (H, W), TH, board_center_mm=center)  # centre reference
    # corner sits on-axis -> centred; the centre is offset right + down
    assert abs(g0.offset[0]) < 1e-6 and abs(g0.offset[1]) < 1e-6
    assert gc.offset[0] > 0.05 and gc.offset[1] > 0.05
    # jog deltas point at the centre; distance is to the centre (slightly > 450)
    assert abs(gc.move_cam[0] - 120) < 1e-6 and abs(gc.move_cam[1] - 90) < 1e-6
    assert gc.distance_mm > 450 and g0.distance_mm == 450


if __name__ == "__main__":
    test_tilt_metric()
    test_all_green_when_ideal()
    test_distance_gate()
    test_angle_gate()
    test_corner_and_none_gates()
    test_offset_sign()
    test_board_center_reference()
    print("All aiming-gate checks passed.")
