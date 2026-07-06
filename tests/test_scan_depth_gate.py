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


def test_live_target_holds_framed_standoff_when_over_nudged_into_crop():
    # THE MOVING-GOALPOST BUG: framed -> target ~T; the operator moves a little too
    # close so the object overruns (surface_mode flips to crop). The target must HOLD T
    # (so they can back off to reframe), NOT collapse to accurate_min and drive them
    # even closer -- the reported "target says 592, I move toward it, it jumps to 300
    # and I can never reach it". Only a genuinely oversized surface (never framed ->
    # no latch) should work close at accurate_min.
    cfg = ScanConfig()
    framed = live_scan_telemetry_payload({
        "detected": True, "distance_mm": 658.0, "tilt_deg": 1.0, "valid_frac": 0.9,
        "fully_framed": True, "surface_mode": "full", "extent_mm": [360.0, 240.0],
        "rectangle_size_mm": [360.0, 240.0],
        "color_fit_standoff_per_margin_mm": 560.0,
    }, cfg)
    target = framed["ideal_distance_mm"]
    assert cfg.accurate_min_mm < target < cfg.accurate_max_mm, framed

    # Nudged too close -> overrun (crop), WITH a latched framed target: HOLD it.
    crop = live_scan_telemetry_payload({
        "detected": True, "distance_mm": 610.0, "tilt_deg": 1.0, "valid_frac": 0.9,
        "fully_framed": False, "surface_mode": "crop", "extent_mm": [360.0, 240.0],
        "rectangle_size_mm": [1000.0, 1000.0],
        "color_fit_standoff_per_margin_mm": 610.0,
    }, cfg, previous_ideal_mm=target)
    assert crop["ideal_distance_mm"] == target, crop            # held, not 300
    assert crop["ideal_distance_mm"] != cfg.accurate_min_mm, crop

    # Genuinely oversized surface: never framed -> no latch -> work close at 300.
    cold = live_scan_telemetry_payload({
        "detected": True, "distance_mm": 610.0, "tilt_deg": 1.0, "valid_frac": 0.9,
        "fully_framed": False, "surface_mode": "crop", "extent_mm": [900.0, 900.0],
        "rectangle_size_mm": [1000.0, 1000.0],
        "color_fit_standoff_per_margin_mm": 900.0,
    }, cfg)                                                     # no previous_ideal_mm
    assert cold["ideal_distance_mm"] == cfg.accurate_min_mm, cold
    print("[telemetry latch] over-nudge into crop holds framed target; cold crop = 300")


def test_recommended_standoff_sits_inside_the_frame_not_on_the_crop_edge():
    # The recommended aim standoff must leave enough border that the object stays framed
    # AT the target (so DISTANCE and FRAMED can both be green at one pose). With a fit-to-
    # frame standoff F, the recommendation is F*frame_margin > F, i.e. FURTHER than the
    # exact-fill distance -> the corner sits inside the view, not on the crop boundary.
    cfg = ScanConfig()
    fit = 560.0
    p = live_scan_telemetry_payload({
        "detected": True, "distance_mm": 640.0, "tilt_deg": 1.0, "valid_frac": 0.9,
        "fully_framed": True, "surface_mode": "full", "extent_mm": [360.0, 240.0],
        "rectangle_size_mm": [360.0, 240.0],
        "color_fit_standoff_per_margin_mm": fit,
    }, cfg)
    assert cfg.frame_margin >= 1.1, cfg.frame_margin           # comfortable border
    assert p["ideal_distance_mm"] > fit, p                     # target is outside exact-fill
    print("[telemetry margin] aim standoff sits inside the frame, off the crop edge")


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


def test_live_overlay_draws_trimmed_rectangle_when_server_sends_it():
    # Bug fix: the LIVE overlay must draw the density/colour-TRIMMED rectangle (the
    # same box lock/insert uses) instead of the RAW fitted rectangle, so the box does
    # not visibly shrink when the operator locks. The framing decision still uses the
    # RAW corners (full object extent). Falls back to raw when the server (pre-deploy)
    # sends no trimmed corners.
    cfg = ScanConfig()
    camera = type("Camera", (), {
        "K": np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]]),
        "dist": np.zeros((5, 1)), "size": (320, 240),
    })()
    raw_corners = np.array([[-180.0, -110.0, 500.0], [180.0, -110.0, 500.0],
                            [180.0, 110.0, 500.0], [-180.0, 110.0, 500.0]])
    trimmed_corners = np.array([[-150.0, -95.0, 500.0], [150.0, -95.0, 500.0],
                                [150.0, 95.0, 500.0], [-150.0, 95.0, 500.0]])
    base = {
        "detected": True, "distance_mm": 500.0, "tilt_deg": 0.0, "valid_frac": 1.0,
        "fully_framed": True, "depth_fully_framed": True, "surface_mode": "full",
        "extent_mm": [360.0, 220.0], "rectangle_size_mm": [300.0, 190.0],
        "rectangle_corners_color_mm": raw_corners.tolist(),
        "outline_uv": [[0.0, 0.0]] * 4,
    }
    proj = lambda c: (cv2.projectPoints(c, np.zeros(3), np.zeros(3),
                      camera.K, camera.dist)[0].reshape(-1, 2) / np.array(camera.size))
    # With trimmed corners present -> overlay is the TRIMMED projection.
    p = live_scan_telemetry_payload(
        {**base, "trimmed_corners_color_mm": trimmed_corners.tolist()},
        cfg, camera_cfg=camera)
    assert p["fully_framed"] is True, p          # framing from RAW corners
    assert np.allclose(np.asarray(p["outline_uv"]), proj(trimmed_corners)), p["outline_uv"]
    assert not np.allclose(np.asarray(p["outline_uv"]), proj(raw_corners))
    # Without trimmed corners (pre-deploy server) -> falls back to the RAW rectangle.
    p2 = live_scan_telemetry_payload(base, cfg, camera_cfg=camera)
    assert np.allclose(np.asarray(p2["outline_uv"]), proj(raw_corners)), p2["outline_uv"]
    print("[telemetry trim] live overlay draws the trimmed box; falls back to raw")


def _cam_320():
    return type("Camera", (), {
        "K": np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]]),
        "dist": np.zeros((5, 1)),
        "size": (320, 240),
    })()


def test_live_trusts_fitted_rectangle_over_strict_depth_border():
    # A well-margined block: the server's raw-pixel depth test tripped (a few stray
    # fringe points) -> it sent surface_mode="crop"/not framed, but the fitted
    # rectangle projects comfortably inside the frame. The host must TRUST the
    # rectangle: framed + full + draw the rectangle, not the generic square.
    cfg = ScanConfig()
    camera = _cam_320()
    corners = np.array([[-120.0, -90.0, 500.0], [120.0, -90.0, 500.0],
                        [120.0, 90.0, 500.0], [-120.0, 90.0, 500.0]])
    raw = {
        "detected": True, "distance_mm": 500.0, "tilt_deg": 1.0, "valid_frac": 1.0,
        "depth_fully_framed": False, "fully_framed": False, "surface_mode": "crop",
        "extent_mm": [240.0, 180.0], "rectangle_size_mm": [1000.0, 1000.0],
        "rectangle_corners_color_mm": corners.tolist(),
        "surface_center_cam_mm": [0.0, 0.0, 500.0],
        "outline_uv": [[-0.1, -0.1], [1.1, -0.1], [1.1, 1.1], [-0.1, 1.1]],
    }
    p = live_scan_telemetry_payload(raw, cfg, camera_cfg=camera)
    assert p["fully_framed"] is True, p
    assert p["surface_mode"] == "full", p
    assert p["gates"].get("framed") is True, p["gates"]
    expected = cv2.projectPoints(corners, np.zeros(3), np.zeros(3),
                                 camera.K, camera.dist)[0].reshape(-1, 2) / np.array(camera.size)
    assert np.allclose(np.asarray(p["outline_uv"]), expected), p["outline_uv"]
    print("[framing] fitted rectangle inside frame -> framed+full, hugging the object")


def test_live_overrun_rectangle_still_crops():
    # A surface that overruns the colour view: even if the server was optimistic
    # (full), the projected corners fall on/over the frame edge -> host corrects to
    # crop and keeps the generic square, not a frame-spanning rectangle.
    cfg = ScanConfig()
    camera = _cam_320()
    corners = np.array([[-280.0, -210.0, 500.0], [280.0, -210.0, 500.0],
                        [280.0, 210.0, 500.0], [-280.0, 210.0, 500.0]])
    raw = {
        "detected": True, "distance_mm": 500.0, "tilt_deg": 1.0, "valid_frac": 1.0,
        "depth_fully_framed": True, "fully_framed": True, "surface_mode": "full",
        "extent_mm": [560.0, 420.0], "rectangle_size_mm": [560.0, 420.0],
        "rectangle_corners_color_mm": corners.tolist(),
        "surface_center_cam_mm": [0.0, 0.0, 500.0],
        "outline_uv": [[-0.2, -0.2], [1.2, -0.2], [1.2, 1.2], [-0.2, 1.2]],
    }
    p = live_scan_telemetry_payload(raw, cfg, camera_cfg=camera)
    assert p["fully_framed"] is False, p
    assert p["surface_mode"] == "crop", p
    assert p["crop_size_mm"] is not None, p
    print("[framing] overrunning rectangle -> crop + generic square (no regression)")


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


def test_framed_rectangle_center_is_advisory_for_readiness():
    cfg = ScanConfig()
    raw = {
        "detected": True, "distance_mm": 500.0, "tilt_deg": 1.0,
        "valid_frac": 0.9, "fully_framed": True, "depth_fully_framed": True,
        "surface_mode": "full", "extent_mm": [300.0, 200.0],
        "rectangle_size_mm": [300.0, 200.0],
        "color_fit_standoff_per_margin_mm": 476.0,
        "surface_center_cam_mm": [95.0, 0.0, 500.0],
        "outline_uv": [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]],
    }
    p = live_scan_telemetry_payload(raw, cfg)
    assert p["gates"]["center"] is False, p
    assert p["ok"] is True, p
    print("[telemetry rectangle] framed surface center is guidance, not readiness")


def test_stable_rectangle_latches_center_jitter():
    cfg = ScanConfig()
    base = {
        "detected": True, "distance_mm": 500.0, "tilt_deg": 1.0,
        "valid_frac": 0.9, "fully_framed": True, "depth_fully_framed": True,
        "surface_mode": "full", "extent_mm": [300.0, 200.0],
        "rectangle_size_mm": [300.0, 200.0],
        "color_fit_standoff_per_margin_mm": 476.0,
        "surface_center_cam_mm": [0.0, 0.0, 500.0],
        "outline_uv": [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]],
    }
    prev = live_scan_telemetry_payload(base, cfg)
    noisy_1 = live_scan_telemetry_payload({
        **base,
        "distance_mm": 501.0,
        "surface_center_cam_mm": [400.0, -320.0, 501.0],
        "outline_uv": [[0.302, 0.301], [0.702, 0.300], [0.700, 0.701], [0.301, 0.700]],
    }, cfg, previous_ideal_mm=prev["ideal_distance_mm"])
    stable_1 = stabilize_live_scan_payload(noisy_1, prev, cfg)
    noisy_2 = live_scan_telemetry_payload({
        **base,
        "distance_mm": 502.0,
        "surface_center_cam_mm": [405.0, -315.0, 502.0],
        "outline_uv": [[0.301, 0.302], [0.701, 0.300], [0.700, 0.702], [0.300, 0.700]],
    }, cfg, previous_ideal_mm=stable_1["ideal_distance_mm"])
    stable_2 = stabilize_live_scan_payload(noisy_2, stable_1, cfg)
    assert stable_2["rect_stable_frames"] >= cfg.live_rect_latch_frames, stable_2
    assert stable_2["center_latched"] is True, stable_2
    assert stable_2["gates"]["center"] is True, stable_2
    assert stable_2["move_cam"][0] == stable_1["move_cam"][0], stable_2["move_cam"]
    print("[telemetry rectangle] stable rectangle latches noisy X/Y guidance")


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
    # Near-square platform: the edge yaw is ambiguous, so EDGE A reads OK (advisory
    # lamp is populated, not blank) and never blocks readiness even at 20deg off.
    assert p["gates"]["edge"] is True, p
    assert p["ok"] is True, p
    print("[telemetry square] EDGE A is advisory (lamp OK), not a lock gate")


def test_live_scan_elongated_edge_gate_is_advisory_lamp():
    # An elongated platform (aspect >= edge_gate_min_aspect) whose long edge is
    # misaligned lights EDGE A amber (gates.edge False) but must NOT block readiness
    # (ok stays True) — the lamp informs without making lock harder.
    cfg = ScanConfig()
    raw = {
        "detected": True, "distance_mm": 500.0, "tilt_deg": 1.0,
        "valid_frac": 0.9, "fully_framed": True, "depth_fully_framed": True,
        "surface_mode": "full", "extent_mm": [600.0, 200.0],
        "rectangle_size_mm": [600.0, 200.0],
        "color_fit_standoff_per_margin_mm": 480.0,
        "surface_center_cam_mm": [0.0, 0.0, 500.0],
        "edge_angle_deg": 20.0,     # long edge 20deg off axis -> misaligned
    }
    p = live_scan_telemetry_payload(raw, cfg)
    assert p["gates"]["edge"] is False, p     # meaningful + misaligned -> amber lamp
    assert p["ok"] is True, p                 # but advisory: does not block lock
    aligned = live_scan_telemetry_payload({**raw, "edge_angle_deg": 1.0}, cfg)
    assert aligned["gates"]["edge"] is True, aligned
    print("[telemetry elongated] EDGE A reflects real alignment but stays advisory")


def _jittery_pair(cfg):
    """A settled reading + the next, wildly-noisier live reading (static robot)."""
    base = {
        "detected": True, "distance_mm": 789.0, "tilt_deg": 2.4,
        "tilt_b_deg": 1.2, "tilt_c_deg": -2.6, "valid_frac": 0.9,
        "fully_framed": True, "depth_fully_framed": True,
        "extent_mm": [420.0, 300.0], "rectangle_size_mm": [420.0, 300.0],
        "surface_mode": "full", "color_fit_standoff_per_margin_mm": 780.0,
        "surface_center_cam_mm": [20.0, -35.0, 789.0], "edge_angle_deg": 5.0,
        "outline_uv": [[0.30, 0.30], [0.70, 0.30], [0.70, 0.70], [0.30, 0.70]],
    }
    prev = live_scan_telemetry_payload(base, cfg)
    prev["live"] = True
    # Same parked robot, but the RealSense plane fit swings hard (the measured bug:
    # X ~83 mm, edge angle ~73 deg, tilt a few deg — all with no real motion).
    noisy = live_scan_telemetry_payload({
        **base, "distance_mm": 789.6, "tilt_deg": 3.9, "tilt_b_deg": 3.4,
        "tilt_c_deg": -1.7, "surface_center_cam_mm": [-15.0, -48.0, 789.6],
        "edge_angle_deg": 40.0,
        "outline_uv": [[0.33, 0.27], [0.72, 0.31], [0.66, 0.73], [0.28, 0.69]],
    }, cfg, previous_ideal_mm=prev["ideal_distance_mm"])
    return prev, noisy


def test_pose_hold_settles_then_freezes_all_axes():
    cfg = ScanConfig()
    prev, noisy = _jittery_pair(cfg)
    settle = int(cfg.live_hold_settle_frames)
    cur, held, unheld = prev, None, 0
    for _ in range(settle + 3):
        out = stabilize_live_scan_payload(noisy, cur, cfg, robot_static=True)
        if out.get("held"):
            held = out
            break
        unheld += 1
        cur = out
    # A few projected frames settle first (building the rectangle) before locking.
    assert unheld >= 1, "should settle at least one frame before locking"
    assert held is not None and held.get("held") is True, held
    # Once locked, another static frame changes nothing — zero jitter on every axis.
    frozen = stabilize_live_scan_payload(noisy, held, cfg, robot_static=True)
    assert frozen.get("held") is True, frozen
    for key in ("distance_mm", "tilt_deg", "tilt_b_deg", "tilt_c_deg",
                "yaw_a_deg", "move_cam", "outline_uv", "extent_mm",
                "rectangle_size_mm", "gates", "ok"):
        assert frozen[key] == held[key], (key, frozen[key], held[key])
    print("[pose hold] settles a few frames, then locks rock-steady on all axes")


def test_pose_hold_releases_and_tracks_when_robot_moves():
    cfg = ScanConfig()
    prev, noisy = _jittery_pair(cfg)
    moved = stabilize_live_scan_payload(noisy, prev, cfg, robot_static=False)
    # Not held: the normal smoothing path runs so the HUD tracks the new pose.
    assert moved.get("held") is not True, moved
    assert moved.get("stabilized") is True, moved
    assert moved["tilt_deg"] != prev["tilt_deg"], moved["tilt_deg"]
    print("[pose hold] real motion -> released, readouts track again")


def test_pose_hold_release_is_debounced_against_single_noisy_frame():
    # The parked-arm jitter bug: a frozen hold must NOT break on ONE frame flagged
    # "moved" (a transient model-pose blip or a noisy plane fit). The release is
    # debounced (live_hold_release_frames), so a single such frame keeps holding the
    # frozen value and only SUSTAINED motion releases it. Directly reproduces "static,
    # then a sudden jump": before the debounce, that single frame re-latched a fresh
    # noisy sample.
    cfg = ScanConfig()
    prev, noisy = _jittery_pair(cfg)
    cur, held = prev, None
    for _ in range(int(cfg.live_hold_settle_frames) + 3):
        cur = stabilize_live_scan_payload(noisy, cur, cfg, robot_static=True)
        if cur.get("held"):
            held = cur
            break
    assert held is not None and held.get("held") is True, held

    # One "moved" frame -> STILL held (grace window), frozen value unchanged.
    grace = stabilize_live_scan_payload(noisy, held, cfg, robot_static=False)
    assert grace.get("held") is True, grace
    assert grace["tilt_b_deg"] == held["tilt_b_deg"], "frozen value must not jump on a blip"

    # Sustained motion (>= live_hold_release_frames) -> release, HUD tracks again.
    released = grace
    for _ in range(int(cfg.live_hold_release_frames)):
        released = stabilize_live_scan_payload(noisy, released, cfg, robot_static=False)
    assert released.get("held") is not True, released
    print("[pose hold] single noisy frame ridden out; sustained motion releases")


def test_pose_hold_vision_escape_releases_on_real_dolly():
    # Even if RoboDK is NOT mirroring the arm (pose gate wrongly says static), a real
    # dolly-in that drops the standoff far past the noise floor must release the hold
    # so the depth-derived rectangle keeps tracking.
    cfg = ScanConfig()
    prev, _ = _jittery_pair(cfg)          # settled at ~789 mm
    dolly = live_scan_telemetry_payload({
        "detected": True, "distance_mm": 729.0, "tilt_deg": 2.4,
        "tilt_b_deg": 1.2, "tilt_c_deg": -2.6, "valid_frac": 0.9,
        "fully_framed": True, "depth_fully_framed": True,
        "extent_mm": [420.0, 300.0], "rectangle_size_mm": [420.0, 300.0],
        "surface_mode": "full", "color_fit_standoff_per_margin_mm": 720.0,
        "surface_center_cam_mm": [20.0, -35.0, 729.0], "edge_angle_deg": 5.0,
        "outline_uv": [[0.30, 0.30], [0.70, 0.30], [0.70, 0.70], [0.30, 0.70]],
    }, cfg, previous_ideal_mm=prev["ideal_distance_mm"])
    out = stabilize_live_scan_payload(dolly, prev, cfg, robot_static=True)
    assert out.get("held") is not True, out
    print("[pose hold] real dolly releases the hold via the vision escape")


def test_camera_pose_moved_tolerances():
    import numpy as _np
    from tasni.modules.scan.service import camera_pose_moved
    ref = _np.eye(4)
    # Sub-tolerance sensor/encoder dither -> not moved.
    near = _np.eye(4); near[:3, 3] = [0.4, 0.0, 0.3]
    assert camera_pose_moved(near, ref, 0.8, 0.15) is False
    # A real jog past the translation tolerance -> moved.
    jog = _np.eye(4); jog[0, 3] = 5.0
    assert camera_pose_moved(jog, ref, 0.8, 0.15) is True
    # A real rotation past the angular tolerance -> moved.
    th = _np.radians(1.0)
    rot = _np.eye(4)
    rot[:3, :3] = [[_np.cos(th), -_np.sin(th), 0], [_np.sin(th), _np.cos(th), 0], [0, 0, 1]]
    assert camera_pose_moved(rot, ref, 0.8, 0.15) is True
    # Missing pose fails open (treated as moved -> falls back to smoothing).
    assert camera_pose_moved(None, ref, 0.8, 0.15) is True
    print("[pose hold] motion tolerances gate hold vs track correctly")


if __name__ == "__main__":
    test_frontal_plane_all_green()
    test_tilt_measured_and_gated()
    test_too_far_fails_distance()
    test_no_surface_not_detected()
    test_live_telemetry_uses_surface_appropriate_standoff()
    test_live_target_is_continuous_across_color_frame_boundary()
    test_live_outline_uses_saved_color_calibration()
    test_live_overlay_draws_trimmed_rectangle_when_server_sends_it()
    test_live_trusts_fitted_rectangle_over_strict_depth_border()
    test_live_overrun_rectangle_still_crops()
    test_live_scan_payload_stabilizes_static_jitter()
    test_live_scan_payload_holds_mode_on_border_flicker()
    test_live_scan_payload_aligns_rectangle_corner_order()
    test_live_scan_payload_does_not_shrink_to_partial_depth()
    test_live_scan_payload_hysteresis_holds_green_gate()
    test_framed_rectangle_center_is_advisory_for_readiness()
    test_stable_rectangle_latches_center_jitter()
    test_live_scan_near_square_skips_edge_gate()
    test_live_scan_elongated_edge_gate_is_advisory_lamp()
    test_pose_hold_settles_then_freezes_all_axes()
    test_pose_hold_releases_and_tracks_when_robot_moves()
    test_pose_hold_release_is_debounced_against_single_noisy_frame()
    test_pose_hold_vision_escape_releases_on_real_dolly()
    test_camera_pose_moved_tolerances()
    print("\ndepth_gate.py tests passed.")
