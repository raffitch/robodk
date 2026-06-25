"""test_scan_survey.py — full-frame surface survey for the scan planner (pure numpy).

Renders synthetic depth of a plane at a known camera-frame normal + distance and
asserts the survey recovers standoff / tilt / extent, lights the gates, and emits a
sane outline + adaptive metric grid. No RoboDK / camera.

    py -3.10 tests/test_scan_survey.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.modules.scan.survey import (  # noqa: E402
    SurveyThresholds, survey_surface)

W, H = 320, 240
K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])


def _render(normal_cam, dist_mm, W=W, H=H, K=K):
    """Depth (uint16) of a plane with given camera-frame normal at dist_mm along optical axis."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    dirs = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, float)], -1)
    n = np.asarray(normal_cam, float)
    n /= np.linalg.norm(n)
    d0 = n @ np.array([0, 0, dist_mm], float)
    denom = dirs @ n
    with np.errstate(divide='ignore', invalid='ignore'):
        s = d0 / denom
    s[~np.isfinite(s) | (s <= 0)] = 0
    return s.astype(np.uint16)


def _render_framed(normal_cam, dist_mm, W=W, H=H, K=K):
    """Like _render but the surface sits INSIDE the frame (a centered region surrounded
    by invalid depth) so its inliers never touch the image border -> fully_framed True.
    The valid region is the central 60% x 60% (= 36% of the frame, > 0.3 threshold)."""
    d = _render(normal_cam, dist_mm, W, H, K)
    mask = np.zeros((H, W), bool)
    x0, x1 = int(W * 0.2), int(W * 0.8)
    y0, y1 = int(H * 0.2), int(H * 0.8)
    mask[y0:y1, x0:x1] = True
    d[~mask] = 0
    return d


def test_frontal_plane_all_green():
    th = SurveyThresholds()  # accurate band 300..800, max tilt 6
    m = survey_surface(_render_framed([0, 0, 1], 500), K, th)
    assert m.detected and m.ok, m.to_dict()
    assert abs(m.standoff_mm - 500) < 5, m.standoff_mm
    assert m.tilt_deg < 1.0, m.tilt_deg
    assert m.fully_framed, "centered frontal plane should be fully framed (no border contact)"
    assert all(m.gates.values()), m.gates
    print("[frontal] standoff", round(m.standoff_mm, 1), "tilt", round(m.tilt_deg, 2),
          "framed", m.fully_framed, "OK")


def test_tilt_measured():
    th = SurveyThresholds()
    # 20deg about Y: normal = (sin20, 0, -cos20) (already faces the camera).
    a = np.deg2rad(20)
    m = survey_surface(_render([np.sin(a), 0, np.cos(a)], 500), K, th)
    assert abs(m.tilt_deg - 20) < 1.5, m.tilt_deg
    assert not m.gates["angle"], "20deg > 6deg limit -> angle red"
    assert not m.ok
    # Tilt about Y -> correction is all B, no C.
    assert abs(abs(m.tilt_b_deg) - 20) < 1.5, m.tilt_b_deg
    assert abs(m.tilt_c_deg) < 1.5, m.tilt_c_deg
    print("[tilt] measured", round(m.tilt_deg, 1), "deg -> correct via B",
          round(m.tilt_b_deg, 1), "C", round(m.tilt_c_deg, 1))


def test_too_far_reference_mode():
    th = SurveyThresholds()
    m = survey_surface(_render([0, 0, 1], 1200), K, th)
    assert m.detected
    assert not m.gates["distance"], "1200mm > accurate_max 800 -> distance red"
    assert not m.ok
    print("[far] standoff", round(m.standoff_mm, 1), "-> distance lamp red")


def test_no_depth_not_detected():
    th = SurveyThresholds()
    m = survey_surface(np.zeros((H, W), np.uint16), K, th)
    assert not m.detected and not m.ok
    assert m.standoff_mm is None
    assert survey_surface(None, K, th).detected is False
    print("[empty] no depth -> not detected")


def test_partial_surface_not_framed():
    th = SurveyThresholds()  # min_valid_depth_frac 0.3
    d = _render([0, 0, 1], 500)
    # Zero out only the bottom-right quadrant: 75% of the frame stays valid (> 0.3),
    # and the visible inliers still touch the top/left borders -> not fully framed.
    d[H // 2:, W // 2:] = 0
    m = survey_surface(d, K, th)
    assert m.detected, "three quadrants still have plenty of valid depth"
    assert not m.fully_framed, "inliers touch the top/left image borders"
    assert not m.gates["framed"]
    print("[partial] detected", m.detected, "framed", m.fully_framed)


def test_extent_approximate():
    th = SurveyThresholds()
    dist = 500.0
    m = survey_surface(_render([0, 0, 1], dist), K, th)
    fx, fy = K[0, 0], K[1, 1]
    expect_w = W * dist / fx        # real-world width spanned by the frame at 500mm
    expect_h = H * dist / fy
    longer, shorter = max(expect_w, expect_h), min(expect_w, expect_h)
    assert abs(m.extent_mm[0] - longer) / longer < 0.2, (m.extent_mm, longer)
    assert abs(m.extent_mm[1] - shorter) / shorter < 0.2, (m.extent_mm, shorter)
    print("[extent] measured", tuple(round(x, 1) for x in m.extent_mm),
          "expected ~", (round(longer, 1), round(shorter, 1)))


def test_grid_spacing_nice_number():
    th = SurveyThresholds()
    m = survey_surface(_render([0, 0, 1], 500), K, th)
    nice = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000}
    assert m.grid_spacing_mm in nice, m.grid_spacing_mm
    assert m.grid_uv is not None and len(m.grid_uv) > 0
    print("[grid] spacing", m.grid_spacing_mm, "mm,", len(m.grid_uv), "lines")


def test_outline_uv_normalized():
    th = SurveyThresholds()
    m = survey_surface(_render([0, 0, 1], 500), K, th)
    assert m.outline_uv is not None and len(m.outline_uv) == 4, m.outline_uv
    for u, v in m.outline_uv:
        assert 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0, (u, v)
    print("[outline]", [tuple(round(x, 3) for x in c) for c in m.outline_uv])


def test_surface_dots_are_a_stable_lattice():
    """points_uv is a fixed surface-anchored lattice, not a per-frame random pixel
    subsample — so the HUD dots hold still instead of 'dancing' every frame."""
    th = SurveyThresholds()
    d = _render_framed([0, 0, 1], 500)
    m1 = survey_surface(d, K, th)
    assert m1.points_uv is not None and len(m1.points_uv) > 20, m1.points_uv
    for u, v in m1.points_uv:
        assert 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0, (u, v)
    # Bounded count: a lattice (hundreds), not thousands of raw inlier pixels.
    assert len(m1.points_uv) <= 1000, len(m1.points_uv)
    # Deterministic: identical depth -> identical dots (proves it is NOT re-sampled).
    m2 = survey_surface(d.copy(), K, th)
    assert m1.points_uv == m2.points_uv, "same input must yield identical dots"
    # Steady under small depth noise: the centroid-anchored lattice barely moves, so
    # the projected dots do not jump (the anti-'dance' property the user asked for).
    rng = np.random.default_rng(0)
    noisy = d.astype(np.int32)
    nz = noisy > 0
    noisy[nz] += rng.integers(-2, 3, size=int(nz.sum()))
    mn = survey_surface(np.clip(noisy, 0, None).astype(np.uint16), K, th)
    A, B = np.asarray(m1.points_uv), np.asarray(mn.points_uv)
    nn = np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)).min(axis=1)
    assert float(np.median(nn)) < 0.01, float(np.median(nn))  # < 1% of the frame
    print("[surface dots] stable lattice:", len(m1.points_uv),
          "dots, median jitter", round(float(np.median(nn)) * 100, 3), "% of frame")


def test_to_dict_serializable():
    import json
    th = SurveyThresholds()
    d = survey_surface(_render([0, 0, 1], 500), K, th).to_dict()
    assert isinstance(d["normal_cam"], list)
    assert isinstance(d["centroid_cam_mm"], list)
    assert isinstance(d["extent_mm"], list)
    assert isinstance(d["fov_deg"], list)
    # The whole dict must be json-serializable (no numpy scalars/arrays leaking).
    s = json.dumps(d)
    assert isinstance(s, str) and len(s) > 0
    print("[to_dict] json length", len(s), "OK")


if __name__ == "__main__":
    test_frontal_plane_all_green()
    test_tilt_measured()
    test_too_far_reference_mode()
    test_no_depth_not_detected()
    test_partial_surface_not_framed()
    test_extent_approximate()
    test_grid_spacing_nice_number()
    test_outline_uv_normalized()
    test_surface_dots_are_a_stable_lattice()
    test_to_dict_serializable()
    print("\nsurvey.py tests passed.")
