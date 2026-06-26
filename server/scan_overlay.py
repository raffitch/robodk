"""Pure-numpy overlay math for the live scan rectangle — NO pyrealsense2 / turbojpeg,
so the host test suite can import and unit-test it (the rest of
``server_unicast_syncronous.py`` cannot be imported off the Jetson).

Two pieces, used to make the live work-rectangle hug the real surface:

* ``density_extent_1d`` — the same density-cliff per-edge trim as the workstation's
  ``tasni.modules.scan.plane._density_extent_1d`` (kept textually identical so the
  live box and the locked/inserted box behave the same). Pulls a rectangle side in
  from a sparse coplanar halo just past the real edge to the dense board.

* ``edge_continues`` — a COLOR cross-check that vetoes a trim when the image shows the
  surface continuing across the proposed cliff (depth dropped out but the board goes
  on), so a trim cannot amputate a real edge the camera merely under-sampled. A real
  board edge shows a strong intensity contrast (board vs table); a depth hole on a
  continuing surface shows none.
"""
from __future__ import annotations

import numpy as np


def density_extent_1d(values, lo, hi, *, core_frac=0.20, max_trim_frac=0.15,
                      min_bin_mm=4.0, min_points=60):
    """Shrink a rectangle side's raw span ``[lo, hi]`` (1D positions along one axis,
    mm) inward to where point density rises to ``core_frac`` of the body's typical
    density. Drops a sparse coplanar halo just past the real edge that a symmetric
    quantile trim leaves in (so the box over-runs the board). A real edge is a sharp
    density CLIFF, so the body is kept. Guarded: trims at most ``max_trim_frac`` of
    the span per side, and is skipped below ``min_points`` points."""
    values = np.asarray(values, float)
    span = float(hi - lo)
    n = int(values.size)
    if span <= 0.0 or n < min_points:
        return float(lo), float(hi)
    n_bins = max(8, int(round(span / max(min_bin_mm, span / 80.0))))
    counts, edges = np.histogram(values, bins=n_bins, range=(float(lo), float(hi)))
    occupied = counts[counts > 0]
    if len(occupied) == 0:
        return float(lo), float(hi)
    core = float(np.median(occupied))            # typical density of a populated bin
    thresh = max(1.0, core_frac * core)          # a halo bin sits well below the body
    above = np.where(counts >= thresh)[0]
    if len(above) == 0:
        return float(lo), float(hi)
    new_lo = float(edges[int(above[0])])
    new_hi = float(edges[int(above[-1]) + 1])
    cap = max_trim_frac * span                   # never trim more than this per side
    new_lo = min(new_lo, lo + cap)
    new_hi = max(new_hi, hi - cap)
    return new_lo, new_hi


def edge_continues(inside, outside, *, min_contrast=22.0, min_samples=5) -> bool:
    """Does the COLOR image show the surface continuing across a proposed trim cliff?

    ``inside`` / ``outside`` are intensity samples (0-255) just inside the cliff (on
    the kept body) and just outside it (in the halo region about to be trimmed). A
    real board edge separates two materials -> a large mean-intensity contrast ->
    returns False (the trim is a real edge, apply it). A depth hole on a continuing
    surface looks the same on both sides -> small contrast -> returns True (VETO the
    trim, the surface goes on). Returns False unless there is positive evidence of
    continuation (enough samples on both sides AND low contrast), so the trim is only
    overridden when the colour genuinely shows more surface — never on missing data.
    """
    a = np.asarray(inside, float)
    b = np.asarray(outside, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < min_samples or b.size < min_samples:
        return False
    return abs(float(np.median(a)) - float(np.median(b))) < float(min_contrast)


def side_sample_points(pc, ax_along, ax_cross, along_pos, cross_lo, cross_hi,
                       n=9):
    """3D points across one rectangle side: ``n`` points at a fixed ``along_pos`` on
    the trimming axis, swept across ``[cross_lo, cross_hi]`` on the other axis. Pure
    geometry (returns an ``(n,3)`` array) so it is unit-testable; the caller projects
    these to colour pixels and samples intensity."""
    pc = np.asarray(pc, float)
    ax_along = np.asarray(ax_along, float)
    ax_cross = np.asarray(ax_cross, float)
    ts = np.linspace(float(cross_lo), float(cross_hi), int(n))
    return pc[None, :] + float(along_pos) * ax_along[None, :] + ts[:, None] * ax_cross[None, :]
