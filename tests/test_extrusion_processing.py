"""Pure centreline-processing helpers (no RoboDK/camera/Open3D device)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry_fixtures as gf  # noqa: E402
from tasni.modules.extrusion.processing import (_graph, _ordered_skeleton,  # noqa: E402
                                                 _prune_short_spurs, _rasterize,
                                                 _thin, depth_to_work_points)


def test_depth_backprojection_uses_explicit_work_transform():
    depth = np.array([[0, 1000], [1000, 0]], dtype=np.uint16)
    K = np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]], dtype=float)
    T = np.eye(4); T[:3, 3] = [10, 20, 30]
    points, raw = depth_to_work_points(depth, gf.aligned(K, (2, 2)), T)
    assert raw == 2
    np.testing.assert_allclose(points, [[11, 20, 1030], [10, 21, 1030]])


def test_depth_backprojection_honours_units_and_the_depth_to_colour_extrinsic():
    K_c = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
    geom = gf.offset(color_K=K_c, color_size=(320, 240))          # 0.1 mm, non-identity extrinsic
    truth_color = np.array([[0.0, 0.0, 400.0], [40.0, -25.0, 405.0]])
    depth = gf.render_depth_in_depth_camera(truth_color, geom)
    T = np.eye(4); T[:3, 3] = [100, 200, 0]
    points, raw = depth_to_work_points(depth, geom, T)
    assert raw == 2
    for t in truth_color + [100, 200, 0]:
        assert np.linalg.norm(points - t, axis=1).min() < 3.5


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


# --------------------------------------- chroma gate on registered points (Task 9)

def test_chroma_gate_masks_points_by_their_colour_projection_not_depth_pixel():
    """Depth is not aligned to colour any more: a bead point must be kept because the
    colour pixel it PROJECTS TO is saturated, even though the depth pixel with the
    same (v, u) index looks at something else."""
    from tasni.core.config import ExtrusionConfig
    from tasni.core.depth_geometry import ColorRegistered
    from tasni.modules.extrusion.processing import chroma_gate_mask
    K_c = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
    geom = gf.offset(color_K=K_c, color_size=(320, 240), depth_size=(160, 120))
    # two colour-frame points at z=400: one lands on the saturated blob, one on grey
    pts = np.array([[0.0, 0.0, 400.0], [-80.0, 0.0, 400.0]])
    depth = gf.render_depth_in_depth_camera(pts, geom)
    color = np.full((240, 320, 3), 128, np.uint8)               # achromatic everywhere ...
    color[100:140, 140:180] = (20, 40, 220)                    # ... except a chromatic blob at the reticle
    reg = ColorRegistered.build(depth, geom, K_c, None)
    cfg = ExtrusionConfig(deposit_min_chroma_fraction=0.001)
    keep, applied = chroma_gate_mask(color, reg, cfg)
    assert applied and keep.shape == (len(reg),)
    on_blob = np.linalg.norm(reg.pts_mm - pts[0], axis=1) < 3.5
    assert keep[on_blob].all() and not keep[~on_blob].any()


def test_chroma_gate_abstains_on_an_achromatic_frame_or_size_mismatch():
    from tasni.core.config import ExtrusionConfig
    from tasni.core.depth_geometry import ColorRegistered
    from tasni.modules.extrusion.processing import chroma_gate_mask, deposit_floor_mm
    K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
    depth = np.full((240, 320), 4000, np.uint16)
    reg = ColorRegistered.build(depth, gf.aligned(K, (320, 240), depth_unit_mm=0.1), K, None)
    cfg = ExtrusionConfig()
    keep, applied = chroma_gate_mask(np.full((240, 320, 3), 128, np.uint8), reg, cfg)
    assert not applied and keep.all()
    assert deposit_floor_mm(cfg, applied) == 2.5
    keep, applied = chroma_gate_mask(np.zeros((100, 100, 3), np.uint8), reg, cfg)   # wrong size
    assert not applied and keep.all()


# ------------------------------------------------------- 1 mm voxel default (Task 9)

def test_default_voxel_is_1_mm_so_0_1_mm_depth_words_reach_the_ring_numbers():
    from tasni.core.config import ExtrusionConfig
    assert ExtrusionConfig().voxel_size_m == 0.001
