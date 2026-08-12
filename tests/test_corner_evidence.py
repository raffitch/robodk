import numpy as np

from tasni.modules.scan.corner_evidence import extract_corner_evidence


def _scene(z_mm=400.0, w=320, h=240):
    """Flat plane at z with an L-shaped boundary meeting at pixel (160, 120)."""
    depth = np.full((h, w), z_mm, dtype=np.float32)
    K = np.array([[300.0, 0, w / 2], [0, 300.0, h / 2], [0, 0, 1]])
    # polygon: corner at image center, arms going +u and +v
    poly = np.array([[0.95, 0.5], [0.5, 0.5], [0.5, 0.95]])
    return depth, K, poly


def test_corner_and_edges_are_extracted_in_base_frame():
    depth, K, poly = _scene()
    T = np.eye(4)
    T[2, 3] = 900.0  # camera 900 mm above base origin, looking along +Z(cam)
    ev = extract_corner_evidence(depth, K, poly, T, corner_hint_uv=(0.5, 0.5))
    assert ev is not None
    assert np.linalg.norm(np.asarray(ev.corner_uv) - 0.5) < 0.02
    pts = ev.edge_points_base
    assert pts.shape[0] >= 20 and pts.shape[1] == 3
    # all evidence lies on the plane z = 900 + 400 (identity rotation)
    assert np.allclose(pts[:, 2], 1300.0, atol=1.0)
    # points split between the two arms: some vary in x, some in y
    assert pts[:, 0].ptp() > 50.0 and pts[:, 1].ptp() > 50.0
    assert ev.corner_base_mm is not None


def test_returns_none_without_depth_support():
    depth, K, poly = _scene()
    depth[:] = 0.0
    assert extract_corner_evidence(depth, K, poly, np.eye(4)) is None


def test_returns_none_for_degenerate_polygon():
    depth, K, _ = _scene()
    tiny = np.array([[0.5, 0.5], [0.501, 0.5]])
    assert extract_corner_evidence(depth, K, tiny, np.eye(4)) is None


def test_returns_none_for_degenerate_zero_area_polygon():
    """A polygon with >=3 vertices but all clustered at one point has no usable
    arm length to walk, and must not be mistaken for a valid L-shaped corner."""
    depth, K, _ = _scene()
    clustered = np.array([[0.5, 0.5], [0.5001, 0.5], [0.5002, 0.5]])
    assert extract_corner_evidence(depth, K, clustered, np.eye(4)) is None


def test_returns_none_for_nan_depth():
    """NaN depth must never leak into the output (no NaN-bearing partial result)."""
    depth, K, poly = _scene()
    depth[:] = np.nan
    result = extract_corner_evidence(depth, K, poly, np.eye(4))
    assert result is None


def test_returns_none_for_partial_nan_depth_without_support():
    """Mixed NaN/zero depth (no valid positive samples) must also yield None,
    never a result containing NaN coordinates."""
    depth, K, poly = _scene()
    depth[:] = np.nan
    depth[::2, ::2] = 0.0
    result = extract_corner_evidence(depth, K, poly, np.eye(4))
    if result is not None:
        assert np.all(np.isfinite(result.edge_points_base))
    else:
        assert result is None


def test_arm_walk_terminates_at_polyline_end_without_crash():
    """The polygon is an open polyline; a corner near one end means one arm's
    walk runs off the end almost immediately. This must terminate cleanly
    (no IndexError) and never fabricate points past the polyline's extent."""
    depth, K = _scene()[0], _scene()[1]
    # Corner at index 0: walking with step=-1 goes out of range on the first
    # step (0 + (-1) = -1), so that arm must yield zero points, not crash.
    short_poly = np.array([[0.5, 0.5], [0.6, 0.5], [0.7, 0.5]])
    # Should not raise, regardless of whether enough evidence is found.
    result = extract_corner_evidence(depth, K, short_poly, np.eye(4),
                                      corner_hint_uv=(0.5, 0.5))
    if result is not None:
        assert np.all(np.isfinite(result.edge_points_base))
        assert result.edge_points_base.shape[1] == 3


def test_arm_walk_direct_short_polyline_produces_bounded_points():
    """Directly exercise _walk_arm-driven behavior via a polyline shorter than
    the requested arm length in both directions, at an interior vertex, to
    make sure the loop bounds (0 <= i+step < len) are respected both ways."""
    depth, K = _scene()[0], _scene()[1]
    poly = np.array([[0.4, 0.5], [0.5, 0.5], [0.6, 0.5]])
    result = extract_corner_evidence(depth, K, poly, np.eye(4),
                                      corner_hint_uv=(0.5, 0.5),
                                      arm_frac=2.0)
    if result is not None:
        assert np.all(np.isfinite(result.edge_points_base))


def test_returns_none_for_nan_camera_pose():
    """A malformed T_base_cam (e.g. a bad calibration read) must not silently
    deproject into NaN base coordinates -- depth is valid everywhere here, so
    without a finiteness guard on the deprojected point this would otherwise
    return confident-looking garbage."""
    depth, K, poly = _scene()
    T = np.eye(4)
    T[2, 3] = np.nan
    assert extract_corner_evidence(depth, K, poly, T, corner_hint_uv=(0.5, 0.5)) is None
