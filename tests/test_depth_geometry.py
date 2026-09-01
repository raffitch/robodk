# tests/test_depth_geometry.py
"""One back-projection for every consumer: colour-frame points from native depth."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geometry_fixtures as gf  # noqa: E402
from tasni.core import depth_geometry as dg  # noqa: E402
from tasni.core.geometry import transform_points  # noqa: E402

K_C = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
SIZE_C = (320, 240)
DIST0 = np.zeros((5, 1), np.float32)


def test_greeting_parses_and_rejects_protocol_1():
    g = gf.offset(color_K=K_C, color_size=SIZE_C)
    assert g.protocol == 2 and g.depth_unit_mm == 0.1 and g.depth_size == (160, 120)
    assert g.T_color_depth.shape == (4, 4) and g.T_color_depth[0, 3] == 14.7
    assert g.legacy is False
    assert g.to_dict()["depth_to_color"]["translation_mm"][0] == 14.7
    with pytest.raises(ValueError, match="protocol"):
        dg.CameraGeometry.from_greeting({**g.raw, "protocol": 1})


def test_from_greeting_carries_the_achieved_spatial_smooth_delta():
    """A take's provenance must say what the spatial filter ran AT, not just that it
    ran -- otherwise the two arms of a smooth_delta A/B are indistinguishable on disk
    (docs/inspection-roll-probe-handoff.md 3.1)."""
    base = gf.offset(color_K=K_C, color_size=SIZE_C).raw
    g = dg.CameraGeometry.from_greeting(
        {**base, "filter_options": {"spatial_smooth_delta": 4.0}})
    assert g.spatial_smooth_delta == 4.0
    # ...and it survives the round trip into the archive manifest.
    assert g.to_dict()["filter_options"]["spatial_smooth_delta"] == 4.0

    control = dg.CameraGeometry.from_greeting(
        {**base, "filters": ["threshold", "disparity", "temporal", "disparity_inv"],
         "filter_options": {"spatial_smooth_delta": None}})
    assert control.spatial_smooth_delta is None


def test_a_greeting_without_filter_options_still_parses():
    """Backward compatibility, and it is not optional: every greeting already on disk
    under runs/ predates this field. ``from_greeting`` reads those archives on every
    reprocess and every figure render -- a KeyError here would make the whole existing
    archive unreadable."""
    old = dict(gf.offset(color_K=K_C, color_size=SIZE_C).raw)
    old.pop("filter_options", None)
    assert "filter_options" not in old

    g = dg.CameraGeometry.from_greeting(old)               # must not raise
    assert g.protocol == 2 and g.depth_unit_mm == 0.1
    assert g.spatial_smooth_delta is None                  # unknown, not fabricated
    assert "filter_options" not in g.to_dict()             # nothing invented into raw


def test_legacy_aligned_geometry_reports_no_smooth_delta():
    g = gf.aligned(K_C, SIZE_C)
    assert g.spatial_smooth_delta is None


def test_every_greeting_already_on_disk_still_parses():
    """The same guard against the REAL archive rather than a fixture. Skips where
    runs/ is absent (it is git-ignored), so the synthetic test above is the one that
    always runs; this one is what catches a field the fixtures happen not to model."""
    import json

    root = Path(__file__).resolve().parents[1] / "runs" / "extrusion"
    manifests = sorted(root.glob("*/*/manifest.json")) if root.is_dir() else []
    if not manifests:
        pytest.skip("no extrusion archive on this machine (runs/ is git-ignored)")
    checked = 0
    for path in manifests:
        raw = (json.loads(path.read_text()).get("provenance") or {}).get("camera_geometry")
        if not raw:
            continue                                    # pre-protocol-2 take
        g = dg.CameraGeometry.from_greeting(raw)         # must not raise
        assert g.protocol == 2 and g.depth_unit_mm > 0
        assert g.spatial_smooth_delta is None            # these predate the field
        checked += 1
    assert checked, "the archive holds no protocol-2 greeting to check"


def test_backproject_applies_units_and_extrinsic():
    g = gf.offset(color_K=K_C, color_size=SIZE_C)
    truth = np.array([[0.0, 0.0, 400.0], [30.0, -20.0, 410.0], [-45.0, 25.0, 395.0]])
    depth = gf.render_depth_in_depth_camera(truth, g)
    pts, uv = dg.backproject(depth, g)
    assert len(pts) == 3 and uv.shape == (3, 2)
    # nearest match per truth point, within one depth pixel + one unit of quantisation
    for t in truth:
        err = np.linalg.norm(pts - t, axis=1).min()
        assert err < 3.5, (t, err)


def test_backproject_with_identity_geometry_is_the_old_formula():
    g = gf.aligned(K_C, SIZE_C)
    depth = np.zeros((240, 320), np.uint16); depth[120, 160] = 500; depth[0, 0] = 1000
    pts, uv = dg.backproject(depth, g)
    np.testing.assert_allclose(pts[uv[:, 0] == 160], [[0, 0, 500]])
    np.testing.assert_allclose(pts[uv[:, 0] == 0], [[-160 / 300 * 1000, -120 / 300 * 1000, 1000]])


def test_backproject_stride_recovers_original_pixel_coords_and_matches_stride1():
    g = gf.aligned(K_C, SIZE_C)
    depth = np.zeros((240, 320), np.uint16)
    depth[120, 160] = 500  # even (v, u): survives a stride=2 subsample
    depth[80, 200] = 700   # even (v, u): survives a stride=2 subsample
    pts1, uv1 = dg.backproject(depth, g, stride=1)
    pts2, uv2 = dg.backproject(depth, g, stride=2)
    # uv_depth must be ORIGINAL-image pixel coordinates (multiples of the
    # stride), not the subsampled array's own indices -- a bug here would
    # return (80, 60) instead of (160, 120).
    assert set(map(tuple, uv2.tolist())) == {(160, 120), (200, 80)}
    assert np.all(uv2 % 2 == 0)
    # the point recovered at a given pixel is the same regardless of stride
    for u, v in [(160, 120), (200, 80)]:
        p1 = pts1[(uv1[:, 0] == u) & (uv1[:, 1] == v)]
        p2 = pts2[(uv2[:, 0] == u) & (uv2[:, 1] == v)]
        assert len(p1) == 1 and len(p2) == 1
        np.testing.assert_allclose(p2, p1)


def test_backproject_mask_selects_exact_subset():
    g = gf.aligned(K_C, SIZE_C)
    depth = np.zeros((240, 320), np.uint16)
    depth[100, 50] = 300
    depth[150, 200] = 450
    depth[10, 10] = 900   # valid depth, but excluded by the mask below
    mask = np.zeros((240, 320), bool)
    mask[100, 50] = True
    mask[150, 200] = True
    pts, uv = dg.backproject(depth, g, mask=mask)
    assert len(pts) == 2
    assert set(map(tuple, uv.tolist())) == {(50, 100), (200, 150)}
    expected = {
        (50, 100): [(50 - 160) / 300 * 300, (100 - 120) / 300 * 300, 300],
        (200, 150): [(200 - 160) / 300 * 450, (150 - 120) / 300 * 450, 450],
    }
    for (u, v), want in expected.items():
        p = pts[(uv[:, 0] == u) & (uv[:, 1] == v)]
        np.testing.assert_allclose(p[0], want)


def test_depth_pose_composes_on_the_right():
    g = gf.offset(color_K=K_C, color_size=SIZE_C)
    T_base_color = np.eye(4); T_base_color[:3, 3] = [100, 200, 300]
    T_base_depth = dg.depth_pose(T_base_color, g)
    np.testing.assert_allclose(T_base_depth, T_base_color @ g.T_color_depth)


def test_ray_point_and_project_round_trip():
    p = np.array([[12.0, -7.0, 420.0]])
    uv = dg.project_to_color(p, K_C, DIST0)
    back = dg.ray_point(uv[0, 0], uv[0, 1], 420.0, K_C, DIST0)
    np.testing.assert_allclose(back, p[0], atol=1e-6)


def test_color_registered_selects_by_colour_region():
    g = gf.offset(color_K=K_C, color_size=SIZE_C)
    # a flat plane at z=400 in the colour frame, sampled on a grid
    # Sample finer than the depth pixel pitch: a 61x45 grid lights only 2745 of
    # the depth frame's 19200 pixels, so the plane would be a genuinely ~14%-valid
    # surface and valid_frac_in_center_patch would correctly report ~0.28. This
    # grid saturates the depth image (7222 lit px, the plane's whole footprint),
    # which is what "a fully valid depth image" in that metric's contract means.
    xs, ys = np.meshgrid(np.linspace(-150, 150, 150), np.linspace(-110, 110, 110))
    plane = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 400.0)])
    depth = gf.render_depth_in_depth_camera(plane, g)
    reg = dg.ColorRegistered.build(depth, g, K_C, DIST0)
    assert reg.pts_mm.shape[1] == 3 and reg.uv.shape == reg.pts_mm.shape[:1] + (2,)
    # centre patch: every registered point there projects inside the patch
    m = reg.in_center_patch(0.25)
    assert m.sum() > 20
    assert np.all(np.abs(reg.uv[m, 0] - 160) <= 40 + 0.5) and np.all(np.abs(reg.uv[m, 1] - 120) <= 30 + 0.5)
    assert 0.5 < reg.valid_frac_in_center_patch(0.25) <= 1.05
    # polygon: left half of the image, normalised coords
    left = reg.in_polygon([[0, 0], [0.5, 0], [0.5, 1], [0, 1]])
    assert np.all(reg.uv[left, 0] <= 160.5) and left.sum() > 0
    # near/median: the plane's depth in the colour frame is 400 at every pixel
    z = reg.median_z_near(160, 120, 6)
    assert abs(z - 400.0) < 1.0
    assert np.isnan(reg.median_z_near(5, 5, 0.1))


def test_valid_frac_in_center_patch_reads_near_1_for_full_coverage_r25():
    """R25: ``_density_ratio`` used to ESTIMATE the depth-px/colour-px ratio from
    the registered points' own footprint, which read 0.25 against the true
    0.1878 for this fixture's intrinsics -- a genuinely 100%-covered centre patch
    read ~0.73 instead of 1.0. The analytic ratio (fx_d*fy_d)/(fx_c*fy_c) fixes
    that: a fully covered patch must read close to 1.0 regardless of the
    registration's baseline/offset -- this is the property min_valid_depth_frac
    (tuned when depth WAS the colour image) relies on."""
    # legacy_aligned: depth_K == color_K -> ratio is exactly 1.0, and a fully
    # populated depth image maps 1:1 onto the colour image -> exactly 1.0.
    g = gf.aligned(K_C, SIZE_C)
    depth = np.full((SIZE_C[1], SIZE_C[0]), 500, dtype=np.uint16)
    reg = dg.ColorRegistered.build(depth, g, K_C, DIST0)
    assert reg.valid_frac_in_center_patch(0.25) == pytest.approx(1.0, abs=1e-6)

    # a real (offset) registration, depth image saturated -- same fixture as
    # test_color_registered_selects_by_colour_region, which the analytic ratio
    # must also put near 1.0, not the ~0.76 the old footprint estimate gave it.
    g2 = gf.offset(color_K=K_C, color_size=SIZE_C)
    xs, ys = np.meshgrid(np.linspace(-150, 150, 150), np.linspace(-110, 110, 110))
    plane = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 400.0)])
    depth2 = gf.render_depth_in_depth_camera(plane, g2)
    reg2 = dg.ColorRegistered.build(depth2, g2, K_C, DIST0)
    assert abs(reg2.valid_frac_in_center_patch(0.25) - 1.0) < 0.1


def test_density_ratio_divides_by_the_calibrated_colour_k_that_produced_uv():
    """The colour K in ``_density_ratio``'s denominator must be the one ``build``
    PROJECTED with, not the greeting's factory K.

    ``build`` computes ``uv = project_to_color(pts, K_color, dist_color)`` from the
    caller's CALIBRATED colour model, but used to hand ``_density_ratio`` the
    greeting's ``color_K_factory`` instead. ``uv`` is what ``in_center_patch``
    counts, so the density of registered points per colour pixel is governed by the
    calibrated K; the factory K is a flat multiplicative bias. On this D435i
    (factory fx,fy = 1362.15/1362.21 vs calibrated 1334.81/1336.21) it inflated
    ``valid_frac`` by x1.0403, so a fully covered patch read ~1.04 instead of 1.0
    and ``evaluate_depth_gate``'s ``min_valid_depth_frac = 0.5`` really tripped at a
    true coverage of 0.4806. Permissive, never a false refusal -- but the biased
    number is persisted into scan records where no later reader can undo it.

    The fixture reproduces exactly that, at this camera's real colour resolution
    and with its real two K's: a depth camera that IS the factory colour camera
    (same size, zero extrinsic) and a saturated depth image -- a 100%-covered
    surface by construction, whose honest reading is 1.0 and whose factory-K
    reading is 1.0403. Measured: 0.9974 with the fix, 1.0376 without (the 0.26%
    shortfall is the depth grid landing on integer colour pixels, an order of
    magnitude smaller than the bias being caught).
    """
    size = (1920, 1080)
    k_factory = np.array([[1362.15, 0, 960.0], [0, 1362.21, 540.0], [0, 0, 1.0]])
    k_calib = np.array([[1334.81, 0, 960.0], [0, 1336.21, 540.0], [0, 0, 1.0]])
    bias = (k_factory[0, 0] * k_factory[1, 1]) / (k_calib[0, 0] * k_calib[1, 1])
    assert bias == pytest.approx(1.0403, abs=5e-4), bias   # the fixture IS the defect

    g = gf.offset(color_K=k_factory, color_size=size, depth_K=k_factory,
                  depth_size=size, rot_deg=(0, 0, 0), t_mm=(0, 0, 0))
    depth = np.full((size[1], size[0]), 5000, np.uint16)   # 500 mm at 0.1 mm units
    reg = dg.ColorRegistered.build(depth, g, k_calib, DIST0)

    # depth_K == k_factory, so the calibrated denominator IS the bias factor and the
    # factory denominator would be exactly 1.0 -- the two answers cannot be confused.
    assert reg._density_ratio() == pytest.approx(bias, rel=1e-12)
    assert abs(reg._density_ratio() - 1.0) > 0.03
    assert reg.valid_frac_in_center_patch(0.25) == pytest.approx(1.0, abs=0.01)


def test_legacy_aligned_density_ratio_is_exactly_one_and_stride_still_divides_it():
    """Two properties the calibrated-K fix must not disturb.

    (1) EXACTLY 1.0 on the archive path. ``CameraGeometry.legacy_aligned`` (ARCHIVED
        takes only) is always fed the same config K the caller then hands ``build``
        -- ``extrusion/service.py``'s reprocess reads one ``intrinsics["K"]``,
        ``figures.py``'s ``geometry_for_take``/``_compute_stages`` one ``take.K``, and
        ``geometry_fixtures.aligned`` one ``K`` -- so ``(a*b)/(a*b)`` is 1.0 to the
        bit, preserving the pre-protocol-2 "one depth pixel == one colour pixel"
        convention rather than merely landing near it.
    (2) ``stride**2`` stays in the denominator. A ``stride=2`` build (what
        ``evaluate_depth_gate`` does on every live HUD tick) samples 1-in-4 native
        depth pixels, so the registered-point count a fully covered patch can reach
        drops by 4. Dropping the divisor would silently re-break every stride>1
        caller's DETECT lamp.
    """
    g = gf.aligned(K_C, SIZE_C)
    depth = np.full((SIZE_C[1], SIZE_C[0]), 500, dtype=np.uint16)

    reg = dg.ColorRegistered.build(depth, g, K_C, DIST0)
    assert reg._density_ratio() == 1.0                     # exact, not approx
    assert reg.valid_frac_in_center_patch(0.25) == pytest.approx(1.0, abs=1e-6)

    reg2 = dg.ColorRegistered.build(depth, g, K_C, DIST0, stride=2)
    assert reg2.stride == 2
    assert reg2._density_ratio() == 0.25                   # 1.0 / stride**2
    assert reg2.valid_frac_in_center_patch(0.25) == pytest.approx(1.0, abs=1e-6)


def test_legacy_geometry_flags_itself():
    g = gf.aligned(K_C, SIZE_C)
    assert g.legacy is True and g.depth_unit_mm == 1.0
    assert g.to_dict()["legacy_aligned"] is True


def test_color_registered_takes_no_factory_k_alias():
    """``ColorRegistered`` must ask for the colour K that PRODUCED ``uv``, by that
    name only -- no ``color_K_factory=`` alias, and no default.

    The alias existed for exactly one test's call site and is a live hazard: a
    parameter named for the greeting's factory K invites a caller to pass
    ``geom.color_K_factory``, which is the ~4% ``valid_frac`` bias the calibrated-K
    fix removed (see ``test_density_ratio_divides_by_the_calibrated_colour_k_that_
    produced_uv``). Wrong-but-plausible must not be spellable.
    """
    import inspect

    params = inspect.signature(dg.ColorRegistered.__init__).parameters
    assert "color_K_factory" not in params, list(params)
    assert "color_K" in params
    # Required, not defaulted: omitting it is a TypeError from Python itself rather
    # than a silently None colour model that only surfaces inside _density_ratio.
    assert params["color_K"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        dg.ColorRegistered(
            pts_mm=np.zeros((0, 3)), uv=np.zeros((0, 2)), uv_depth=np.zeros((0, 2), int),
            color_size=(320, 240), depth_size=(160, 120), stride=1, depth_K=np.eye(3))
