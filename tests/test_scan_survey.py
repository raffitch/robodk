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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasni.modules.scan.survey import (  # noqa: E402
    SurveyThresholds, survey_surface)

import geometry_fixtures as gf  # noqa: E402

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
    m = survey_surface(_render_framed([0, 0, 1], 500), gf.aligned(K, (W, H)), K, None, th)
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
    m = survey_surface(_render([np.sin(a), 0, np.cos(a)], 500), gf.aligned(K, (W, H)), K, None, th)
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
    m = survey_surface(_render([0, 0, 1], 1200), gf.aligned(K, (W, H)), K, None, th)
    assert m.detected
    assert not m.gates["distance"], "1200mm > accurate_max 800 -> distance red"
    assert not m.ok
    print("[far] standoff", round(m.standoff_mm, 1), "-> distance lamp red")


def test_no_depth_not_detected():
    th = SurveyThresholds()
    m = survey_surface(np.zeros((H, W), np.uint16), gf.aligned(K, (W, H)), K, None, th)
    assert not m.detected and not m.ok
    assert m.standoff_mm is None
    assert survey_surface(None, gf.aligned(K, (W, H)), K, None, th).detected is False
    print("[empty] no depth -> not detected")


def test_partial_surface_not_framed():
    th = SurveyThresholds()  # min_valid_depth_frac 0.3
    d = _render([0, 0, 1], 500)
    # Zero out only the bottom-right quadrant: 75% of the frame stays valid (> 0.3),
    # and the visible inliers still touch the top/left borders -> not fully framed.
    d[H // 2:, W // 2:] = 0
    m = survey_surface(d, gf.aligned(K, (W, H)), K, None, th)
    assert m.detected, "three quadrants still have plenty of valid depth"
    assert not m.fully_framed, "inliers touch the top/left image borders"
    assert not m.gates["framed"]
    print("[partial] detected", m.detected, "framed", m.fully_framed)


def test_extent_approximate():
    th = SurveyThresholds()
    dist = 500.0
    m = survey_surface(_render([0, 0, 1], dist), gf.aligned(K, (W, H)), K, None, th)
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
    m = survey_surface(_render([0, 0, 1], 500), gf.aligned(K, (W, H)), K, None, th)
    nice = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000}
    assert m.grid_spacing_mm in nice, m.grid_spacing_mm
    assert m.grid_uv is not None and len(m.grid_uv) > 0
    print("[grid] spacing", m.grid_spacing_mm, "mm,", len(m.grid_uv), "lines")


def test_outline_uv_normalized():
    # A FULLY-FRAMED surface keeps the board rectangle, whose 4 corners are inside the
    # image -> normalized to [0,1]. (An overrunning surface intentionally projects a
    # generic square that exceeds the frame; that path is covered separately below.)
    th = SurveyThresholds()
    m = survey_surface(_render_framed([0, 0, 1], 500), gf.aligned(K, (W, H)), K, None, th)
    assert m.fully_framed, "centered surface should be fully framed (board-hug path)"
    assert m.outline_uv is not None and len(m.outline_uv) == 4, m.outline_uv
    for u, v in m.outline_uv:
        assert 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0, (u, v)
    print("[outline]", [tuple(round(x, 3) for x in c) for c in m.outline_uv])


def test_crop_mode_uses_generic_reticle_square():
    """When the surface overruns the view (inliers touch the border -> not fully
    framed), the work region is a GENERIC fixed square on the plane centred on the
    reticle (the optical axis), not the over-running board rectangle. extent_mm stays
    the raw measured extent (used to decide the surface is large)."""
    th = SurveyThresholds(work_crop_mm=(600.0, 600.0))
    m = survey_surface(_render([0, 0, 1], 500), gf.aligned(K, (W, H)), K, None, th)   # plane fills frame -> not framed
    assert m.detected and not m.fully_framed, (m.detected, m.fully_framed)
    c = np.asarray(m.corners_cam_mm, float)
    assert c.shape == (4, 3), c.shape
    centre = c.mean(axis=0)
    assert abs(centre[0]) < 1e-6 and abs(centre[1]) < 1e-6, centre   # on the optical axis
    assert abs(centre[2] - 500.0) < 6.0, centre                       # at the surface depth
    edges = np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)
    assert np.allclose(edges, 600.0, atol=2.0), edges                # the configured square
    # extent_mm stays the RAW measured rectangle (~the frame-spanned extent at 500 mm,
    # 320 px / fx 300 ≈ 533 mm), distinct from the 600 mm work square — i.e. it is the
    # real measurement used to decide the surface is large, not the crop.
    assert m.extent_mm is not None and 450.0 < m.extent_mm[0] < 580.0, m.extent_mm
    assert abs(m.extent_mm[0] - edges[0]) > 50.0, (m.extent_mm, edges[0])
    print("[crop] generic", round(float(edges[0]), 1), "mm reticle square; raw extent",
          tuple(round(x, 1) for x in m.extent_mm))


def test_dominant_plane_selection_prefers_the_aimed_at_surface_not_the_bigger_one():
    """2026-08-30 false-refusal investigation, cause confirmed live on the real
    cell (a coordinator re-measurement after the checkerboard-on-a-desk finding
    turned out not to generalize): a work platform (~450 mm standoff, ~248k
    depth points, 29% of the frame) shared a frame with an adjoining FLOOR
    (~1071 mm standoff, ~600k points, 71%) -- a genuine ~620 mm step, NOT
    coplanar. Plain maximal-consensus RANSAC selected the FLOOR purely by
    inlier count: standoff read ~1071 mm, the floor's enormous extent read
    ``fully_framed=False``, and the entire-platform refusal fired on a
    platform that was, in fact, fully and comfortably framed. The operator's
    objection was exactly right: "you should see the frame then the floor and
    realize that it's the frame we want to keep."

    No single-plane fixture (every other test in this file) can catch this --
    there is nothing to out-vote. This builds two genuinely disjoint,
    non-coplanar flat regions (platform centred in view, floor surrounding it
    with MORE points, mirroring the live cell's point-count ratio) and asserts
    the smaller, CENTRED platform -- not the bigger floor -- is what
    ``survey_surface`` locks onto, via the aiming-reticle seed
    (``fit_plane``'s ``seed_mask``, built from
    ``SurveyThresholds.center_patch_frac`` -- the same "where is the operator
    aiming" region the depth gate already uses).
    """
    th = SurveyThresholds()
    d = np.full((H, W), 1070, dtype=np.uint16)              # floor fills the whole frame...
    platform_frac = 0.4
    x0, x1 = int(W * (1 - platform_frac) / 2), int(W * (1 + platform_frac) / 2)
    y0, y1 = int(H * (1 - platform_frac) / 2), int(H * (1 + platform_frac) / 2)
    d[y0:y1, x0:x1] = 450                                    # ...platform sits centred in it
    platform_px = (y1 - y0) * (x1 - x0)
    floor_px = H * W - platform_px
    assert floor_px > platform_px, (
        "fixture must give the floor MORE points, matching the live report", platform_px, floor_px)

    m = survey_surface(d, gf.aligned(K, (W, H)), K, None, th)
    assert m.detected, m.to_dict()
    assert m.standoff_mm is not None and abs(m.standoff_mm - 450.0) < 10.0, (
        "RANSAC selected the bigger surrounding surface instead of the aimed-at "
        "platform -- the dominant-plane-selection regression is back", m.to_dict())
    assert m.fully_framed, (
        "the platform (well within the frame, once the right plane is selected) "
        "should read fully framed", m.to_dict())
    assert m.extent_mm is not None and max(m.extent_mm) < 600.0, (
        "extent this large means the FLOOR's whole-frame rectangle was measured, "
        "not the platform's", m.extent_mm)
    print("[plane selection] platform", round(m.standoff_mm, 1), "mm selected over the",
          "bigger floor (platform", platform_px, "px vs floor", floor_px, "px)")


def test_dominant_plane_selection_falls_back_to_global_ransac_with_one_surface():
    """Explicit degrade-sensibly check (companion to the two-plane test above):
    when the aiming-reticle seed region has no data of its own to offer beyond
    what a single-surface scene already provides, the seeded fit must recover
    the exact same result as plain global RANSAC -- proven here by comparing
    against every other single-plane test in this file still passing unchanged
    (test_frontal_plane_all_green etc.), and directly here against a tilted
    plane where the seed patch is real but there is only one surface to find."""
    th = SurveyThresholds()
    a = np.deg2rad(15)
    d = _render([np.sin(a), 0, np.cos(a)], 500)
    m = survey_surface(d, gf.aligned(K, (W, H)), K, None, th)
    assert m.detected, m.to_dict()
    assert abs(m.standoff_mm - 500) < 8, m.standoff_mm
    assert abs(m.tilt_deg - 15) < 1.5, m.tilt_deg
    print("[plane selection] single-surface scene unaffected: standoff",
          round(m.standoff_mm, 1), "tilt", round(m.tilt_deg, 2))


def test_fully_framed_survives_distortion_fold_back():
    """A ``gf.aligned`` fixture cannot exercise this: depth and colour share one
    K/size there, so a corner that leaves the colour frame also leaves the depth
    frame at the same normalized radius. Needs ``gf.offset`` with a depth FOV wide
    enough to backproject genuinely off-colour-frame 3D points, plus REAL
    (nonzero) colour distortion coefficients -- the fold-back only exists in the
    calibrated Brown-Conrady model, not in ``gf.aligned``'s implicit zero
    distortion.

    Root cause (2026-08-30 false-refusal investigation): ``_corners_in_frame``
    used to project the fitted rectangle's corners through the CALIBRATED
    distortion model for its in-frame test. cv2's forward radial polynomial is
    only monotonic inside its fitted domain; past it (normalized radius ~1.3-1.6
    with this camera's own measured coefficients) it folds back, mapping a point
    that is genuinely far outside the frame to a pixel that lands back inside.
    This fixture reproduces exactly that: a 1399x100mm strip at 500mm standoff
    (its colour FOV at 500mm spans only ~533x400mm) whose corners, before the
    fix, projected back inside the colour frame under the calibrated model and
    made ``fully_framed`` read True for an obviously overrunning rectangle.
    """
    # REAL colour distortion measured on the workstation D435i (cfg.camera.dist
    # in tasni.config.json) -- ties this regression to genuine hardware, not an
    # invented polynomial.
    dist_color = np.array([0.11480838, -0.23856355, -0.00182125, 0.00042104, 0.0])
    # Depth FOV much wider than colour's (K/size below), so a rectangle whose
    # short axis (Y) fills the depth frame while its long axis (X) reaches well
    # past the colour frame's edge is still backprojected in full -- exactly
    # protocol 2's depth-wider-than-colour geometry, exaggerated so the corners
    # land in the measured fold-back band.
    depth_K = np.array([[40.0, 0, 80.0], [0, 40.0, 60.0], [0, 0, 1.0]])
    geom = gf.offset(color_K=K, color_size=(W, H), depth_K=depth_K, depth_size=(160, 120))

    Z = 500.0
    xs = np.linspace(-700.0, 700.0, 400)     # normalized radius ~1.3-1.6 at Z=500mm:
    ys = np.linspace(-50.0, 50.0, 40)        # the measured fold-back band (see survey.py).
    XX, YY = np.meshgrid(xs, ys)
    plane = np.column_stack([XX.ravel(), YY.ravel(), np.full(XX.size, Z)])
    depth = gf.render_depth_in_depth_camera(plane, geom)
    assert np.count_nonzero(depth) > 500, "fixture is not discriminating -- too few depth points"

    # min_valid_depth_frac relaxed: the point of this fixture is the corner-
    # containment test, not the "enough of the frame has depth" gate -- the
    # exaggerated depth FOV needed to reach the fold-back band makes this thin
    # strip fill only a small fraction of the (much larger) depth image.
    th = SurveyThresholds(min_valid_depth_frac=0.01)
    m = survey_surface(depth, geom, K, dist_color, th)
    assert m.detected, m.to_dict()
    assert m.extent_mm is not None and m.extent_mm[0] > 1300.0, (
        "fixture must fit a genuinely oversized rectangle -- got", m.extent_mm)
    assert not m.fully_framed, (
        "a rectangle nearly 3x the colour FOV's width read as fully framed -- "
        "the distortion fold-back regression is back", m.to_dict())
    print("[fold-back] oversized rectangle", tuple(round(x, 1) for x in m.extent_mm),
          "mm correctly NOT framed despite calibrated-model fold-back")


def test_surface_dots_are_a_stable_lattice():
    """points_uv is the actual measured surface hits snapped to a FIXED image grid
    (one dot per occupied cell), not a per-frame random pixel subsample — so the HUD
    dots mark where depth truly landed and hold still instead of 'dancing'."""
    th = SurveyThresholds()
    d = _render_framed([0, 0, 1], 500)
    m1 = survey_surface(d, gf.aligned(K, (W, H)), K, None, th)
    assert m1.points_uv is not None and len(m1.points_uv) > 20, m1.points_uv
    for u, v in m1.points_uv:
        assert 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0, (u, v)
    # Bounded count: occupied cells on a fixed 1/180 image grid (capped at 4000),
    # not the full raw inlier cloud.
    assert len(m1.points_uv) <= 4000, len(m1.points_uv)
    # Deterministic: identical depth -> identical dots (proves it is NOT re-sampled).
    m2 = survey_surface(d.copy(), gf.aligned(K, (W, H)), K, None, th)
    assert m1.points_uv == m2.points_uv, "same input must yield identical dots"
    # Steady under small depth noise: the centroid-anchored lattice barely moves, so
    # the projected dots do not jump (the anti-'dance' property the user asked for).
    rng = np.random.default_rng(0)
    noisy = d.astype(np.int32)
    nz = noisy > 0
    noisy[nz] += rng.integers(-2, 3, size=int(nz.sum()))
    mn = survey_surface(np.clip(noisy, 0, None).astype(np.uint16), gf.aligned(K, (W, H)), K, None, th)
    A, B = np.asarray(m1.points_uv), np.asarray(mn.points_uv)
    nn = np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)).min(axis=1)
    assert float(np.median(nn)) < 0.01, float(np.median(nn))  # < 1% of the frame
    print("[surface dots] stable lattice:", len(m1.points_uv),
          "dots, median jitter", round(float(np.median(nn)) * 100, 3), "% of frame")


def test_to_dict_serializable():
    import json
    th = SurveyThresholds()
    d = survey_surface(_render([0, 0, 1], 500), gf.aligned(K, (W, H)), K, None, th).to_dict()
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
    test_crop_mode_uses_generic_reticle_square()
    test_dominant_plane_selection_prefers_the_aimed_at_surface_not_the_bigger_one()
    test_dominant_plane_selection_falls_back_to_global_ransac_with_one_surface()
    test_fully_framed_survives_distortion_fold_back()
    test_surface_dots_are_a_stable_lattice()
    test_to_dict_serializable()
    print("\nsurvey.py tests passed.")
