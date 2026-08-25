"""server/scan_overlay.py — pure-numpy live-rectangle math (density-cliff trim +
colour edge cross-check). No pyrealsense2/turbojpeg, so it imports on the host.

    py -3.10 tests/test_scan_overlay.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from scan_overlay import (  # noqa: E402
    density_extent_1d, edge_continues, reticle_plane_square, side_sample_points)


def test_density_trim_drops_sparse_halo():
    # Dense body in [0, 400] (~60 pts/bin) + a genuinely sparse halo in (400, 440]
    # (~4 pts/bin, well under the 20% body threshold, points spaced so they do not
    # alias together in a bin).
    body = np.repeat(np.linspace(0.0, 400.0, 81), 60)
    halo = np.repeat(np.linspace(406.0, 438.0, 5), 4)
    values = np.concatenate([body, halo])
    lo, hi = density_extent_1d(values, 0.0, 440.0)
    assert lo == 0.0, lo                 # dense from the start -> low side untouched
    assert 395.0 <= hi <= 415.0, hi      # pulled back to the board edge, not 440
    print("[overlay] halo trimmed: hi", round(hi, 1), "(body 400, halo->438)")


def test_density_trim_keeps_uniform():
    values = np.repeat(np.linspace(0.0, 500.0, 101), 40)   # uniform, no halo
    lo, hi = density_extent_1d(values, 0.0, 500.0)
    assert abs(lo - 0.0) < 7 and abs(hi - 500.0) < 7, (lo, hi)
    print("[overlay] uniform board preserved", round(lo, 1), round(hi, 1))


def test_density_trim_guard_caps_trim():
    # Almost all density at the far end: an unguarded cliff would collapse the span;
    # the 15% per-side cap must bound how far each side moves.
    values = np.repeat(np.linspace(420.0, 500.0, 17), 50)
    lo, hi = density_extent_1d(values, 0.0, 500.0)
    assert lo <= 0.0 + 0.15 * 500.0 + 1e-6, lo     # trimmed at most 15% off the low side
    print("[overlay] guard cap held: lo", round(lo, 1), "<= 75")


def test_density_trim_skips_when_too_few_points():
    values = np.linspace(0.0, 100.0, 20)            # < min_points
    assert density_extent_1d(values, 0.0, 100.0) == (0.0, 100.0)


def test_edge_continues_detects_real_edge():
    board = np.full(9, 205.0)        # bright board
    table = np.full(9, 45.0)         # dark table beyond the edge
    assert edge_continues(board, table) is False     # real edge -> trim applies


def test_edge_continues_vetoes_on_continuing_surface():
    inside = np.full(9, 130.0)
    outside = np.full(9, 128.0)      # same material -> surface continues
    assert edge_continues(inside, outside) is True   # veto -> hold (no amputation)


def test_edge_continues_no_veto_without_evidence():
    # Too few visible samples (edge projected off-frame) -> cannot confirm
    # continuation -> do not veto (trust the density trim).
    assert edge_continues(np.full(2, 100.0), np.full(9, 100.0)) is False


def test_reticle_square_matches_host_and_is_centred():
    """The server's reticle_plane_square mirror must be numerically identical to the
    workstation's plane.reticle_plane_square (so the live overlay and the locked box
    agree), and centre the square on the +Z optical axis."""
    sys.path.insert(0, str(ROOT))
    from tasni.modules.scan.plane import reticle_plane_square as host_square

    a = np.deg2rad(15.0)
    normal = np.array([np.sin(a), 0.0, -np.cos(a)])
    centroid = np.array([30.0, -20.0, 550.0])
    sc, su, sv, sr = reticle_plane_square(normal, centroid, (1000.0, 1000.0))
    hc, hu, hv, hr = host_square(normal, centroid, (1000.0, 1000.0))
    assert np.allclose(sc, hc) and np.allclose(sr, hr), "server/host squares diverge"
    assert abs(sr[0]) < 1e-9 and abs(sr[1]) < 1e-9, sr     # reticle on the optical axis
    assert np.allclose((sc - centroid) @ normal, 0.0, atol=1e-6)  # corners on the plane
    print("[overlay] reticle square matches host + centred on the optical axis")


def test_side_sample_points_geometry():
    pc = np.array([0.0, 0.0, 500.0])
    ax_along = np.array([1.0, 0.0, 0.0])
    ax_cross = np.array([0.0, 1.0, 0.0])
    pts = side_sample_points(pc, ax_along, ax_cross, 200.0, -100.0, 100.0, n=5)
    assert pts.shape == (5, 3)
    assert np.allclose(pts[:, 0], 200.0)             # fixed along-position
    assert np.isclose(pts[0, 1], -100.0) and np.isclose(pts[-1, 1], 100.0)
    assert np.allclose(pts[:, 2], 500.0)
    print("[overlay] side sample geometry ok")


if __name__ == "__main__":
    test_density_trim_drops_sparse_halo()
    test_density_trim_keeps_uniform()
    test_density_trim_guard_caps_trim()
    test_density_trim_skips_when_too_few_points()
    test_edge_continues_detects_real_edge()
    test_edge_continues_vetoes_on_continuing_surface()
    test_edge_continues_no_veto_without_evidence()
    test_reticle_square_matches_host_and_is_centred()
    test_side_sample_points_geometry()
    print("\nscan_overlay tests passed.")
