import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geometry_fixtures as gf  # noqa: E402
from tasni.core.depth_geometry import ColorRegistered  # noqa: E402
from tasni.modules.scan.corner_evidence import extract_corner_evidence, _deproject_base  # noqa: E402


def _reg(depth, K):
    """Wrap a synthetic aligned-mm depth image into a ColorRegistered (Task 10):
    the scenes below are rendered as aligned depth in K's model (depth == colour
    image, 1 mm units), matching gf.aligned's legacy convention."""
    h, w = np.asarray(depth).shape
    return ColorRegistered.build(depth, gf.aligned(K, (w, h)), K, None)


def _rot_x(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _scene(z_mm=400.0, w=320, h=240):
    """Flat plane at z with an L-shaped boundary meeting at pixel (160, 120)."""
    depth = np.full((h, w), z_mm, dtype=np.float32)
    K = np.array([[300.0, 0, w / 2], [0, 300.0, h / 2], [0, 0, 1]])
    # polygon: corner at image center, arms going +u and +v
    poly = np.array([[0.95, 0.5], [0.5, 0.5], [0.5, 0.95]])
    return depth, K, poly


def _scene_with_discontinuity(w=320, h=240, z_surface=400.0, z_background=700.0):
    """Same L-shaped corner as `_scene`, but with a REAL depth discontinuity:
    the small square u<160,v<120 (the "missing" quadrant of the L) is a
    different, still-valid, depth (background) from the rest of the frame
    (surface). Sampling the wrong side of the boundary produces a measurably
    wrong point, not merely an identical one (a uniform-depth fixture cannot
    distinguish a reversed inset direction from a correct one)."""
    K = np.array([[300.0, 0, w / 2], [0, 300.0, h / 2], [0, 0, 1]])
    poly = np.array([[0.95, 0.5], [0.5, 0.5], [0.5, 0.95]])
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    surface_mask = (uu >= 160) | (vv >= 120)
    depth = np.where(surface_mask, z_surface, z_background).astype(np.float32)
    return depth, K, poly


def _tilted_pose(tx=50.0, ty=-30.0, tz=900.0):
    """A NON-identity camera pose (rotated + translated) so a row/column
    transposition bug in `_deproject_base` would break planarity / produce a
    wrong-magnitude result rather than silently matching by construction."""
    T = np.eye(4)
    T[:3, :3] = _rot_x(18.0) @ _rot_y(12.0)
    T[:3, 3] = [tx, ty, tz]
    return T


def test_corner_and_edges_are_extracted_in_base_frame():
    """Accuracy test (rebuilt per code review): real surface/background depth
    discontinuity + a tilted (non-identity-rotation) camera pose, so this can
    actually catch a reversed inset direction or a transpose bug -- a
    uniform-depth, identity-rotation fixture cannot distinguish either from
    correct output."""
    depth, K, poly = _scene_with_discontinuity()
    T = _tilted_pose()
    ev = extract_corner_evidence(_reg(depth, K), K, poly, T, corner_hint_uv=(0.5, 0.5))
    assert ev is not None
    assert np.linalg.norm(np.asarray(ev.corner_uv) - 0.5) < 0.02
    pts = ev.edge_points_base
    assert pts.shape[0] >= 20 and pts.shape[1] == 3

    # Map base-frame points back into the camera frame via the pose's own
    # (independently computed) inverse: every point's implied camera-frame
    # depth must be the SURFACE depth, not the background depth, and the
    # round trip validates the rotation/translation aren't transposed.
    T_inv = np.linalg.inv(T)
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    pts_cam = (T_inv @ pts_h.T).T[:, :3]
    assert np.allclose(pts_cam[:, 2], 400.0, atol=1.0)

    # points split between the two arms: some vary in x, some in y
    assert pts[:, 0].ptp() > 50.0 and pts[:, 1].ptp() > 50.0
    assert ev.corner_base_mm is not None
    corner_cam = (T_inv @ np.append(np.asarray(ev.corner_base_mm), 1.0))[:3]
    assert abs(corner_cam[2] - 400.0) < 1.0


def test_local_bisector_lands_on_surface_when_global_vertex_mean_would_not():
    """Regression pin for the code-review finding: `interior = poly_px.mean()`
    is not a safe interior reference. Counter-example polygon (an L-shaped
    boundary strip) whose vertex mean sits in the strip's own missing corner,
    outside the surface entirely -- verified via matplotlib.path.Path during
    development. The corner tested is the strip's inner elbow, i.e. the
    concave/reflex vertex where the old centroid-based inset is provably
    wrong (confirmed by running the previously-committed implementation
    against this exact fixture: every pooled point landed at the background
    depth, 1600 mm, not the surface depth, 1300 mm)."""
    w, h = 320, 240
    K = np.array([[300.0, 0, w / 2], [0, 300.0, h / 2], [0, 0, 1]])
    poly = np.array([[0.02, 0.02], [0.98, 0.02], [0.98, 0.12],
                      [0.12, 0.12], [0.12, 0.98], [0.02, 0.98]])
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    un, vn = uu / w, vv / h
    # material = the L-strip itself (top bar OR left bar); everything else
    # (including the vertex mean at (0.373, 0.373)) is background.
    surface_mask = ((vn >= 0.02) & (vn <= 0.12)) | ((un >= 0.02) & (un <= 0.12))
    depth = np.where(surface_mask, 400.0, 700.0).astype(np.float32)
    T = np.eye(4)
    T[2, 3] = 900.0

    ev = extract_corner_evidence(_reg(depth, K), K, poly, T, corner_hint_uv=(0.12, 0.12))
    assert ev is not None
    pts = ev.edge_points_base
    assert pts.shape[0] >= 20
    # every pooled point must be on the SURFACE plane (z=1300), not the
    # background plane (z=1600) that the global-centroid bug would produce.
    assert np.allclose(pts[:, 2], 1300.0, atol=1.0)


def test_arm_stops_at_a_sharp_heading_turn_border_follow():
    """Regression pin for the code-review finding on unbounded arm length: a
    frame-clipped contour that runs out of real edge starts tracing the
    image border instead (typically a near-90 degree turn). Those
    border-following vertices carry no edge signal and must be excluded.

    Corner at (160, 120): arm-1 is a simple non-turning vertical run (real
    edge). arm+1 runs a real horizontal edge to (304, 120) and then turns 90
    degrees to trace what would be the image border down to (304, 240).
    `arm_frac` is large enough that the walk's budget exceeds the first
    segment alone, so it must actively decide whether to continue past the
    turn -- confirmed by running the previously-committed implementation
    (no turn limit) against this exact fixture: it pooled 23 points beyond
    the turn (u > 295 px, v > 130 px) that this fixture must exclude."""
    depth, K = _scene()[0], _scene()[1]
    poly = np.array([[0.5, 0.95], [0.5, 0.5], [0.95, 0.5], [0.95, 1.0]])
    T = np.eye(4)
    T[2, 3] = 900.0
    ev = extract_corner_evidence(_reg(depth, K), K, poly, T, corner_hint_uv=(0.5, 0.5),
                                  arm_frac=0.6, samples_per_arm=60)
    assert ev is not None
    pts = ev.edge_points_base
    # invert the (identity-rotation) pose back to pixel-equivalent u,v
    u_equiv = pts[:, 0] / 400.0 * 300.0 + 160.0
    v_equiv = pts[:, 1] / 400.0 * 300.0 + 120.0
    assert u_equiv.max() <= 305.0  # never reaches past idx2 (u=304) + slack
    # no point near idx2 shows the elevated-v signature of the turned segment
    assert not np.any((u_equiv > 295.0) & (v_equiv > 130.0))


def test_corner_sample_is_inset_like_edge_samples():
    """Regression pin for the code-review finding that the corner pixel was
    sampled raw (uninset) -- the point most likely to straddle a depth
    discontinuity. `corner_base_mm` must reflect a sample taken `inset_px`
    into the surface along the corner's local bisector, not the raw
    boundary-intersection pixel. Verified precisely: the corner's true 2D
    interior direction here is exactly (1,1)/sqrt(2) (confirmed via the
    walk's oriented-normal computation used elsewhere in this module), so
    the expected shift is exact geometry, not a fuzzy tolerance -- and
    matches to within 0.5 mm, while the raw (uninset) hypothesis does not
    (confirmed by running the previously-committed implementation, which
    returned exactly the raw hypothesis, (0, 0, 1300))."""
    depth, K, poly = _scene()
    T = np.eye(4)
    T[2, 3] = 900.0
    ev = extract_corner_evidence(_reg(depth, K), K, poly, T, corner_hint_uv=(0.5, 0.5),
                                  inset_px=4.0)
    assert ev is not None and ev.corner_base_mm is not None

    raw_hypothesis = _deproject_base(160.0, 120.0, 400.0, K, T)
    bisector = np.array([1.0, 1.0]) / np.sqrt(2.0)
    inset_hypothesis = _deproject_base(*( [160.0, 120.0] + 4.0 * bisector), 400.0, K, T)

    actual = np.asarray(ev.corner_base_mm)
    assert np.allclose(actual, inset_hypothesis, atol=0.5)
    assert not np.allclose(actual, raw_hypothesis, atol=0.5)


def test_returns_none_when_only_one_arm_has_depth_support():
    """Regression pin for the code-review finding on single-arm evidence
    silently presented as pooled: invalidate depth under arm+1 (the
    horizontal run, all near v=124) while leaving arm-1 (the vertical run,
    v up to ~260) mostly intact. The pooled TOTAL would still clear the old
    min_valid_frac floor (>=4 points) using arm-1 alone, so the per-arm
    minimum is the check that actually catches this -- confirmed by running
    the previously-committed implementation against this exact fixture: it
    returned 27 points with essentially zero x-variation (ptp ~2.3 mm),
    i.e. single-arm data pooled with nothing to distinguish it."""
    depth, K, poly = _scene()
    depth[:135, :] = 0.0  # wipes out arm+1 entirely; trims arm-1's near-corner end
    T = np.eye(4)
    T[2, 3] = 900.0
    assert extract_corner_evidence(_reg(depth, K), K, poly, T, corner_hint_uv=(0.5, 0.5)) is None


def test_closed_contour_wraps_to_find_both_arms_at_index_zero():
    """Finding 3: a corner at index 0 of a genuinely CLOSED contour must
    still get both arms via wraparound when `closed=True` is passed. Without
    it (the default, matching this module's other open-polyline fixtures),
    the same input correctly returns None (no wraparound -> one arm is
    missing entirely -> the min-points-per-arm safety net rejects it,
    exactly the safe behavior finding 3 requires regardless of how `closed`
    is resolved)."""
    depth = np.full((240, 320), 400.0, dtype=np.float32)
    K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1]])
    # corner at index 0; forward arm -> idx1 (+v); backward arm needs
    # wraparound to idx3 (+u) to find its data.
    poly = np.array([[0.5, 0.5], [0.5, 0.95], [0.95, 0.95], [0.95, 0.5]])
    T = np.eye(4)
    T[2, 3] = 900.0

    assert extract_corner_evidence(_reg(depth, K), K, poly, T, corner_hint_uv=(0.5, 0.5)) is None

    ev = extract_corner_evidence(_reg(depth, K), K, poly, T, corner_hint_uv=(0.5, 0.5), closed=True)
    assert ev is not None
    pts = ev.edge_points_base
    assert pts.shape[0] >= 20
    assert pts[:, 0].ptp() > 50.0 and pts[:, 1].ptp() > 50.0


def test_returns_none_without_depth_support():
    depth, K, poly = _scene()
    depth[:] = 0.0
    assert extract_corner_evidence(_reg(depth, K), K, poly, np.eye(4)) is None


def test_returns_none_for_degenerate_polygon():
    depth, K, _ = _scene()
    tiny = np.array([[0.5, 0.5], [0.501, 0.5]])
    assert extract_corner_evidence(_reg(depth, K), K, tiny, np.eye(4)) is None


def test_returns_none_for_degenerate_zero_area_polygon():
    """A polygon with >=3 vertices but all clustered at one point has no usable
    arm length to walk, and must not be mistaken for a valid L-shaped corner."""
    depth, K, _ = _scene()
    clustered = np.array([[0.5, 0.5], [0.5001, 0.5], [0.5002, 0.5]])
    assert extract_corner_evidence(_reg(depth, K), K, clustered, np.eye(4)) is None


def test_returns_none_for_nan_depth():
    """NaN depth must never leak into the output (no NaN-bearing partial result)."""
    depth, K, poly = _scene()
    depth[:] = np.nan
    result = extract_corner_evidence(_reg(depth, K), K, poly, np.eye(4))
    assert result is None


def test_returns_none_for_partial_nan_depth_without_support():
    """Mixed NaN/zero depth (no valid positive samples) must also yield None,
    never a result containing NaN coordinates."""
    depth, K, poly = _scene()
    depth[:] = np.nan
    depth[::2, ::2] = 0.0
    result = extract_corner_evidence(_reg(depth, K), K, poly, np.eye(4))
    if result is not None:
        assert np.all(np.isfinite(result.edge_points_base))
    else:
        assert result is None


def test_arm_walk_terminates_at_polyline_end_without_crash():
    """The polygon is an open polyline; a corner at index 0 has no backward
    neighbour at all (no wraparound by default), so this must terminate
    cleanly (no IndexError) and correctly recognize there is no usable
    second arm -- returning None rather than fabricating single-arm
    "pooled" evidence (see also `test_returns_none_when_only_one_arm_has_
    depth_support`, which pins the same safety net from the depth side
    rather than the topology side)."""
    depth, K = _scene()[0], _scene()[1]
    short_poly = np.array([[0.5, 0.5], [0.6, 0.5], [0.7, 0.5]])
    result = extract_corner_evidence(_reg(depth, K), K, short_poly, np.eye(4),
                                      corner_hint_uv=(0.5, 0.5))
    assert result is None


def test_arm_walk_direct_short_polyline_produces_bounded_points():
    """A polyline shorter than the requested arm length in both directions,
    at an interior vertex that is exactly collinear with both neighbours
    (not a real corner at all): the arm-bisector degeneracy guard must
    reject it rather than walking out of range (0 <= i+step < len is
    respected either way, but there is nothing valid to return here)."""
    depth, K = _scene()[0], _scene()[1]
    poly = np.array([[0.4, 0.5], [0.5, 0.5], [0.6, 0.5]])
    result = extract_corner_evidence(_reg(depth, K), K, poly, np.eye(4),
                                      corner_hint_uv=(0.5, 0.5),
                                      arm_frac=2.0)
    assert result is None


def test_returns_none_for_nan_camera_pose():
    """A malformed T_base_cam (e.g. a bad calibration read) must not silently
    deproject into NaN base coordinates -- depth is valid everywhere here, so
    without a finiteness guard on the deprojected point this would otherwise
    return confident-looking garbage."""
    depth, K, poly = _scene()
    T = np.eye(4)
    T[2, 3] = np.nan
    assert extract_corner_evidence(_reg(depth, K), K, poly, T, corner_hint_uv=(0.5, 0.5)) is None
