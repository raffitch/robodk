import math
import numpy as np
import pytest

from tasni.modules.scan.rect_fit import (
    fit_edge_line, fit_global_plane, lift_points_3d, project_points_2d,
    solve_constrained_rectangle,
)


def _edge_points(a, b, n=60, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.05, 0.95, n)[:, None]
    pts = np.asarray(a, float) + t * (np.asarray(b, float) - np.asarray(a, float))
    if noise:
        d = np.asarray(b, float) - np.asarray(a, float)
        nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        pts = pts + rng.normal(0, noise, (n, 1)) * nrm
    return pts


# 1600 x 1000 rectangle rotated 17 deg
_ANG = math.radians(17.0)
_R = np.array([[math.cos(_ANG), -math.sin(_ANG)], [math.sin(_ANG), math.cos(_ANG)]])
_CORNERS = (np.array([[0, 0], [1600, 0], [1600, 1000], [0, 1000]], float) - [800, 500]) @ _R.T


def _edges(noise=0.0):
    return [_edge_points(_CORNERS[i], _CORNERS[(i + 1) % 4], noise=noise, seed=i)
            for i in range(4)]


def test_perfect_rectangle_is_recovered_exactly():
    sol = solve_constrained_rectangle(_edges())
    assert sorted(sol.size_mm, reverse=True) == pytest.approx([1600.0, 1000.0], abs=1e-6)
    assert sol.parallelism_deg == pytest.approx(0.0, abs=1e-9)
    assert sol.perpendicularity_deg == pytest.approx(0.0, abs=1e-9)
    assert sol.discrepancy_mm == pytest.approx(0.0, abs=1e-6)
    # corners match the truth up to cyclic order
    diffs = [np.abs(np.roll(sol.corners2d, k, axis=0) - _CORNERS).max() for k in range(4)]
    assert min(diffs) < 1e-6


def test_noisy_rectangle_error_is_bounded():
    sol = solve_constrained_rectangle(_edges(noise=1.0))
    assert abs(max(sol.size_mm) - 1600.0) < 3.0
    assert abs(min(sol.size_mm) - 1000.0) < 3.0
    assert sol.discrepancy_mm < 5.0
    assert max(sol.edge_rms_mm) < 2.5


def test_biased_edge_shows_up_as_discrepancy():
    edges = _edges(noise=0.5)
    d = _CORNERS[1] - _CORNERS[0]
    nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    half = len(edges[0]) // 2
    edges[0][:half] += nrm * 30.0          # corrupt half of edge 0 by 30 mm
    sol = solve_constrained_rectangle(edges)
    assert sol.discrepancy_mm > 5.0 or max(sol.edge_rms_mm) > 5.0


def test_corner_agreement_reported():
    sol = solve_constrained_rectangle(_edges(), local_corners2d=_CORNERS + 2.0)
    assert sol.corner_agreement_mm == pytest.approx(2.0 * math.sqrt(2), abs=0.01)


def test_edge_line_direction_and_rms():
    line = fit_edge_line(_edge_points([0, 0], [100, 0], noise=0.5))
    assert abs(line.direction[1]) < 0.02 and line.rms_mm < 1.0


def test_global_plane_and_projection_roundtrip():
    rng = np.random.default_rng(1)
    sets = []
    for cx in (0.0, 800.0, -800.0):
        xy = rng.uniform(-200, 200, (300, 2)) + [cx, 0.0]
        z = rng.normal(0, 0.4, 300)
        sets.append(np.column_stack([xy, z + 500.0]))
    plane = fit_global_plane(sets)
    assert abs(plane.normal[2]) > 0.999
    assert plane.rms_mm < 1.0 and len(plane.per_set_rms_mm) == 3
    from tasni.modules.scan.plane import _plane_basis
    u, v = _plane_basis(np.asarray(plane.normal))
    p2 = project_points_2d(sets[0], np.asarray(plane.normal), np.asarray(plane.point), u, v)
    p3 = lift_points_3d(p2, np.asarray(plane.point), u, v)
    d = np.abs((p3 - np.asarray(plane.point)) @ np.asarray(plane.normal))
    assert p2.shape == (300, 2) and d.max() < 1e-9
