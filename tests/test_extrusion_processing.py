"""Pure centreline-processing helpers (no RoboDK/camera/Open3D device)."""
from __future__ import annotations

import numpy as np
import pytest

from tasni.modules.extrusion.processing import (_graph, _ordered_skeleton,
                                                 _prune_short_spurs, _rasterize,
                                                 _thin, depth_to_work_points)


def test_depth_backprojection_uses_explicit_work_transform():
    depth = np.array([[0, 1000], [1000, 0]], dtype=np.uint16)
    K = np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]], dtype=float)
    T = np.eye(4); T[:3, 3] = [10, 20, 30]
    points, raw = depth_to_work_points(depth, K, T)
    assert raw == 2
    np.testing.assert_allclose(points, [[11, 20, 1030], [10, 21, 1030]])


def test_graph_ordering_is_deterministic_and_rejects_branches():
    line = np.zeros((7, 9), np.uint8); line[3, 1:8] = 1
    first, branches, endpoints = _ordered_skeleton(line)
    second, *_ = _ordered_skeleton(line)
    np.testing.assert_array_equal(first, second)
    assert branches == 0 and endpoints == 2 and len(first) == 7

    tee = line.copy(); tee[1:4, 4] = 1
    with pytest.raises(RuntimeError, match="branch"):
        _ordered_skeleton(tee)


def test_short_raster_spurs_are_pruned_but_real_branch_is_rejected():
    theta = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    ring = np.column_stack((40 * np.cos(theta), 40 * np.sin(theta),
                            np.full_like(theta, 5.0)))
    mask, _, _ = _rasterize(ring, 1.0, 6.0, 0)
    raw = _thin(mask)
    assert any(len(links) > 2 for links in _graph(raw).values())
    cleaned, removed = _prune_short_spurs(raw, 6)
    _, branches, endpoints = _ordered_skeleton(cleaned)
    assert removed > 0 and branches == 0 and endpoints == 0

    tee = np.zeros((20, 20), np.uint8)
    tee[10, 2:18] = 1
    tee[2:11, 10] = 1
    cleaned_tee, _ = _prune_short_spurs(tee, 3)
    with pytest.raises(RuntimeError, match="branch"):
        _ordered_skeleton(cleaned_tee)
