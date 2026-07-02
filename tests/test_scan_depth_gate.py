"""depth_gate.py — standoff distance + surface tilt from a depth frame (pure numpy).

Renders synthetic depth of a plane at a known distance + tilt and asserts the gate
recovers them and lights the lamps correctly. No RoboDK / camera.

    py -3.10 tests/test_scan_depth_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.modules.scan.depth_gate import (  # noqa: E402
    ScanGateThresholds, evaluate_depth_gate)
from tasni.core.config import ScanConfig  # noqa: E402
from tasni.modules.scan.service import (  # noqa: E402
    live_scan_telemetry_payload, stabilize_live_scan_payload)

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


def test_live_telemetry_uses_surface_appropriate_standoff():
    cfg = ScanConfig()
    raw = {
        "detected": True, "distance_mm": 500.0, "tilt_deg": 2.0,
        "tilt_b_deg": 1.0, "tilt_c_deg": -1.0, "valid_frac": 0.9,
        "fully_framed": True, "extent_mm": [300.0, 200.0],
        "rectangle_size_mm": [200.0, 300.0],
        "surface_mode": "full",
        "color_fit_standoff_per_margin_mm": 300.0,
    }
    p = live_scan_telemetry_payload(raw, cfg)
    expected = round((300.0 * cfg.frame_margin) / 10.0) * 10.0
    assert abs(p["ideal_distance_mm"] - expected) < 1e-6, p
    assert p["gates"]["angle"] is True
    assert p["gates"]["framed"] is True
    assert p["rectangle_size_mm"] == [200.0, 300.0]
    print("[telemetry] live target standoff derived from framed surface extent")


def test_live_target_is_continuous_across_color_frame_boundary():
    cfg = ScanConfig()
    base = {
        "detected": True, "distance_mm": 304.0, "tilt_deg": 1.0,
        "valid_frac": 0.9, "extent_mm": [363.0, 198.0],
        "surface_mode": "full",
        "color_fit_standoff_per_margin_mm": 326.0,
    }
    framed = live_scan_telemetry_payload({**base, "fully_framed": True}, cfg)
    clipped = live_scan_telemetry_payload({**base, "fully_framed": False}, cfg)
    expected = round((326.0 * cfg.frame_margin) / 10.0) * 10.0
    assert framed["ideal_distance_mm"] == clipped["ideal_distance_mm"] == expected
    chatter = live_scan_telemetry_payload(
        {**base, "color_fit_standoff_per_margin_mm": 318.0,
         "fully_framed": True}, cfg, previous_ideal_mm=expected)
    assert chatter["ideal_distance_mm"] == expected

    crop = live_scan_telemetry_payload({
        **base, "fully_framed": False, "surface_mode": "crop"}, cfg)
    assert crop["ideal_distance_mm"] == cfg.accurate_min_mm
    assert crop["crop_size_mm"] is not None
    print("[telemetry hysteresis] color clipping keeps target stable; true crop stays 300 mm")


def test_live_outline_uses_saved_color_calibration():
    cfg = ScanConfig()
    camera = type("Camera", (), {
        "K": np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]]),
        "dist": np.array([0.12, -0.25, -0.002, -0.0003, 0.0]).reshape(-1, 1),
        "size": (320, 240),
    })()
    corners = np.array([
        [-180.0, -110.0, 500.0],
        [180.0, -110.0, 500.0],
        [180.0, 110.0, 500.0],
        [-180.0, 110.0, 500.0],
    ])
    raw = {
        "detected": True, "distance_mm": 500.0, "tilt_deg": 0.0,
        "valid_frac": 1.0, "fully_framed": True,
        "depth_fully_framed": True, "surface_mode": "full",
        "extent_mm": [360.0, 220.0], "rectangle_size_mm": [360.0, 220.0],
        "rectangle_corners_color_mm": corners.tolist(),
        "outline_uv": [[0.0, 0.0]] * 4,
    }
    p = live_scan_telemetry_payload(raw, cfg, camera_cfg=camera)
    expected, _ = cv2.projectPoints(
        corners, np.zeros(3), np.zeros(3), camera.K, camera.dist)
    expected = expected.reshape(-1, 2) / np.array(camera.size)
    assert np.allclose(np.asarray(p["outline_uv"]), expected)
    assert not np.allclose(np.asarray(p["outline_uv"]), 0.0)
    print("[telemetry calibration] saved RGB K+distortion drives blue outline")


def test_live_scan_payload_stabilizes_static_jitter():
    cfg = ScanConfig()
    base = {
        "detected": True, "distance_mm": 500.0, "tilt_deg": 2.0,
        "tilt_b_deg": 1.0, "tilt_c_deg": -1.0, "valid_frac": 0.9,
        "fully_framed": True, "depth_fully_framed": True,
        "extent_mm": [300.0, 200.0], "rectangle_size_mm": [300.0, 200.0],
        "surface_mode": "full", "color_fit_standoff_per_margin_mm": 300.0,
        "surface_center_cam_mm": [4.0, -3.0, 500.0],
        "edge_angle_deg": 1.0,
        "outline_uv": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]],
    }
    prev = live_scan_telemetry_payload(base, cfg)
    noisy = live_scan_telemetry_payload({
        **base,
        "distance_mm": 530.0,
        "tilt_deg": 5.0,
        "tilt_b_deg": 5.0,
        "surface_center_cam_mm": [34.0, -23.0, 530.0],
        "edge_angle_deg": 4.0,
        "outline_uv": [[0.27, 0.24], [0.77, 0.26], [0.73, 0.76], [0.23, 0.74]],
    }, cfg, previous_ideal_mm=prev["ideal_distance_mm"])
    stable = stabilize_live_scan_payload(noisy, prev, cfg)
    assert stable["stabilized"] is True
    assert 500.0 < stable["distance_mm"] < 530.0, stable["distance_mm"]
    assert stable["distance_mm"] < 515.0, stable["distance_mm"]
    assert 2.0 < stable["tilt_deg"] < 5.0, stable["tilt_deg"]
    assert stable["move_cam"][0] < noisy["move_cam"][0], (stable["move_cam"], noisy["move_cam"])
    assert np.asarray(stable["outline_uv"])[0, 0] < np.asarray(noisy["outline_uv"])[0, 0]
    print("[telemetry smoothing] static frame jitter damped for live HUD")


def test_live_scan_payload_holds_mode_on_border_flicker():
    cfg = ScanConfig()
    prev = live_scan_telemetry_payload({
        "detected": True, "distance_mm": 500.0, "tilt_deg": 1.0,
        "valid_frac": 0.9, "fully_framed": True, "depth_fully_framed": True,
        "surface_mode": "full", "extent_mm": [500.0, 350.0],
        "rectangle_size_mm": [500.0, 350.0],
        "color_fit_standoff_per_margin_mm": 500.0,
        "surface_center_cam_mm": [0.0, 0.0, 500.0],
        "outline_uv": [[0.15, 0.15], [0.85, 0.15], [0.85, 0.85], [0.15, 0.85]],
    }, cfg)
    noisy_crop = live_scan_telemetry_payload({
        "detected": True, "distance_mm": 505.0, "tilt_deg": 1.2,
        "valid_frac": 0.9, "fully_framed": False, "depth_fully_framed": False,
        "surface_mode": "crop", "extent_mm": [510.0, 360.0],
        "rectangle_size_mm": [1000.0, 1000.0],
        "color_fit_standoff_per_margin_mm": 505.0,
        "surface_center_cam_mm": [5.0, -4.0, 505.0],
        "outline_uv": [[-0.1, -0.1], [1.1, -0.1], [1.1, 1.1], [-0.1, 1.1]],
    }, cfg, previous_ideal_mm=prev["ideal_distance_mm"])
    stable = stabilize_live_scan_payload(noisy_crop, prev, cfg)
    assert stable["surface_mode"] == "full", stable
    assert stable["fully_framed"] is True, stable
    assert stable["outline_uv"] == prev["outline_uv"], stable["outline_uv"]
    assert stable["stabilized"] is True
    print("[telemetry smoothing] border full/crop flicker held stable")


def test_live_scan_payload_aligns_rectangle_corner_order():
    cfg = ScanConfig()
    prev = live_scan_telemetry_payload({
        "detected": True, "distance_mm": 500.0, "tilt_deg": 1.0,
        "valid_frac": 0.9, "fully_framed": True, "depth_fully_framed": True,
        "surface_mode": "full", "extent_mm": [500.0, 350.0],
        "rectangle_size_mm": [500.0, 350.0],
        "color_fit_standoff_per_margin_mm": 500.0,
        "surface_center_cam_mm": [0.0, 0.0, 500.0],
        "outline_uv": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.7], [0.2, 0.7]],
    }, cfg)
    shifted = live_scan_telemetry_payload({
        "detected": True, "distance_mm": 501.0, "tilt_deg": 1.1,
        "valid_frac": 0.9, "fully_framed": True, "depth_fully_framed": True,
        "surface_mode": "full", "extent_mm": [501.0, 351.0],
        "rectangle_size_mm": [501.0, 351.0],
        "color_fit_standoff_per_margin_mm": 501.0,
        "surface_center_cam_mm": [1.0, -1.0, 501.0],
        "outline_uv": [[0.8, 0.2], [0.8, 0.7], [0.2, 0.7], [0.2, 0.2]],
    }, cfg, previous_ideal_mm=prev["ideal_distance_mm"])
    stable = stabilize_live_scan_payload(shifted, prev, cfg)
    assert stable["stabilized"] is True
    assert np.allclose(np.asarray(stable["outline_uv"])[0], [0.2, 0.2], atol=0.01)
    print("[telemetry smoothing] rectangle corner-order flip aligned")


def test_live_scan_payload_does_not_shrink_to_partial_depth():
    cfg = ScanConfig()
    prev = live_scan_telemetry_payload({
        "detected": True, "distance_mm": 800.0, "tilt_deg": 1.0,
        "valid_frac": 0.9, "fully_framed": True, "depth_fully_framed": True,
        "surface_mode": "full", "extent_mm": [800.0, 790.0],
        "rectangle_size_mm": [800.0, 790.0],
        "color_fit_standoff_per_margin_mm": 760.0,
        "surface_center_cam_mm": [0.0, 0.0, 800.0],
        "outline_uv": [[0.15, 0.15], [0.85, 0.15], [0.85, 0.85], [0.15, 0.85]],
    }, cfg)
    partial = live_scan_telemetry_payload({
        "detected": True, "distance_mm": 801.0, "tilt_deg": 1.1,
        "valid_frac": 0.9, "fully_framed": True, "depth_fully_framed": True,
        "surface_mode": "full", "extent_mm": [430.0, 400.0],
        "rectangle_size_mm": [430.0, 400.0],
        "color_fit_standoff_per_margin_mm": 430.0,
        "surface_center_cam_mm": [4.0, -3.0, 801.0],
        "outline_uv": [[0.32, 0.32], [0.68, 0.32], [0.68, 0.68], [0.32, 0.68]],
    }, cfg, previous_ideal_mm=prev["ideal_distance_mm"])
    stable = stabilize_live_scan_payload(partial, prev, cfg)
    assert stable["outline_uv"] == prev["outline_uv"], stable["outline_uv"]
    assert stable["extent_mm"] == prev["extent_mm"], stable["extent_mm"]
    assert stable["rectangle_size_mm"] == prev["rectangle_size_mm"], stable["rectangle_size_mm"]
    print("[telemetry smoothing] partial-depth rectangle shrink ignored")


def test_live_scan_payload_hysteresis_holds_green_gate():
    cfg = ScanConfig()
    prev = live_scan_telemetry_payload({
        "detected": True, "distance_mm": 500.0, "tilt_deg": 1.0,
        "valid_frac": 0.9, "fully_framed": True, "depth_fully_framed": True,
        "surface_mode": "full", "extent_mm": [300.0, 200.0],
        "rectangle_size_mm": [300.0, 200.0],
        "color_fit_standoff_per_margin_mm": 476.0,
        "surface_center_cam_mm": [25.0, 0.0, 500.0],
        "outline_uv": [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]],
    }, cfg)
    assert prev["gates"]["center"] is True, prev
    noisy = live_scan_telemetry_payload({
        "detected": True, "distance_mm": 501.0, "tilt_deg": 1.1,
        "valid_frac": 0.9, "fully_framed": True, "depth_fully_framed": True,
        "surface_mode": "full", "extent_mm": [300.0, 200.0],
        "rectangle_size_mm": [300.0, 200.0],
        "color_fit_standoff_per_margin_mm": 476.0,
        "surface_center_cam_mm": [42.0, 0.0, 501.0],
        "outline_uv": [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]],
    }, cfg, previous_ideal_mm=prev["ideal_distance_mm"])
    stable = stabilize_live_scan_payload(noisy, prev, cfg)
    assert stable["gates"]["center"] is True, stable
    print("[telemetry smoothing] live gate hysteresis holds near-threshold center")


def test_live_scan_near_square_skips_edge_gate():
    cfg = ScanConfig()
    raw = {
        "detected": True, "distance_mm": 800.0, "tilt_deg": 1.0,
        "valid_frac": 0.9, "fully_framed": True, "depth_fully_framed": True,
        "surface_mode": "full", "extent_mm": [800.0, 790.0],
        "rectangle_size_mm": [800.0, 790.0],
        "color_fit_standoff_per_margin_mm": 760.0,
        "surface_center_cam_mm": [0.0, 0.0, 800.0],
        "edge_angle_deg": 20.0,
    }
    p = live_scan_telemetry_payload(raw, cfg)
    assert "edge" not in p["gates"], p
    assert p["ok"] is True, p
    print("[telemetry square] EDGE A is advisory, not a lock gate")


if __name__ == "__main__":
    test_frontal_plane_all_green()
    test_tilt_measured_and_gated()
    test_too_far_fails_distance()
    test_no_surface_not_detected()
    test_live_telemetry_uses_surface_appropriate_standoff()
    test_live_target_is_continuous_across_color_frame_boundary()
    test_live_outline_uses_saved_color_calibration()
    test_live_scan_payload_stabilizes_static_jitter()
    test_live_scan_payload_holds_mode_on_border_flicker()
    test_live_scan_payload_aligns_rectangle_corner_order()
    test_live_scan_payload_does_not_shrink_to_partial_depth()
    test_live_scan_payload_hysteresis_holds_green_gate()
    test_live_scan_near_square_skips_edge_gate()
    print("\ndepth_gate.py tests passed.")
