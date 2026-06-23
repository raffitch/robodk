"""depth_gate.py — standoff distance + surface tilt from a depth frame (pure numpy).

Renders synthetic depth of a plane at a known distance + tilt and asserts the gate
recovers them and lights the lamps correctly. No RoboDK / camera.

    py -3.10 tests/test_scan_depth_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.modules.scan.depth_gate import (  # noqa: E402
    ScanGateThresholds, evaluate_depth_gate)

W, H = 320, 240
K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])


def _render(normal, dist_mm):
    """Depth (uint16 mm) of a plane with camera-frame ``normal``, crossing the optical
    axis at ``dist_mm``."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    dirs = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, float)], -1)
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    d0 = n @ np.array([0, 0, dist_mm], float)
    denom = dirs @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        s = d0 / denom
    s[~np.isfinite(s) | (s <= 0)] = 0
    return s.astype(np.uint16)


def test_frontal_plane_all_green():
    th = ScanGateThresholds(ideal_distance_mm=500, distance_tol_mm=120, max_tilt_deg=20)
    r = evaluate_depth_gate(_render([0, 0, 1], 500), K, th)
    assert r.detected and r.ok, r.to_dict()
    assert abs(r.distance_mm - 500) < 5, r.distance_mm
    assert r.tilt_deg < 1.0, r.tilt_deg
    print("[frontal] distance", round(r.distance_mm, 1), "tilt", round(r.tilt_deg, 2), "OK")


def test_tilt_measured_and_gated():
    th = ScanGateThresholds(ideal_distance_mm=500, distance_tol_mm=120, max_tilt_deg=20)
    r = evaluate_depth_gate(_render([0, np.sin(np.deg2rad(30)), np.cos(np.deg2rad(30))], 500),
                            K, th)
    assert abs(r.tilt_deg - 30) < 1.5, r.tilt_deg          # tilt recovered
    assert r.gates["distance"] and not r.gates["angle"]    # 30deg > 20deg limit -> red
    assert not r.ok
    # Tilt is purely about the X axis (normal tilted in Y) -> correction is all C, no B.
    assert abs(abs(r.tilt_c_deg) - 30) < 1.5, r.tilt_c_deg
    assert abs(r.tilt_b_deg) < 1.5, r.tilt_b_deg
    print("[tilt] measured", round(r.tilt_deg, 1), "deg -> correct via C",
          round(r.tilt_c_deg, 1), "B", round(r.tilt_b_deg, 1))


def test_too_far_fails_distance():
    th = ScanGateThresholds(ideal_distance_mm=500, distance_tol_mm=120, max_tilt_deg=20)
    r = evaluate_depth_gate(_render([0, 0, 1], 800), K, th)
    assert r.detected and not r.gates["distance"] and not r.ok
    assert r.move_cam[2] > 0                                 # "too far" -> positive Z error
    print("[far] distance", round(r.distance_mm, 1), "-> distance lamp red")


def test_no_surface_not_detected():
    th = ScanGateThresholds(min_valid_depth_frac=0.5)
    r = evaluate_depth_gate(np.zeros((H, W), np.uint16), K, th)
    assert not r.detected and not r.ok and r.distance_mm is None
    assert evaluate_depth_gate(None, K, th).detected is False
    print("[empty] no depth -> not detected")


if __name__ == "__main__":
    test_frontal_plane_all_green()
    test_tilt_measured_and_gated()
    test_too_far_fails_distance()
    test_no_surface_not_detected()
    print("\ndepth_gate.py tests passed.")
