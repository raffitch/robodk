"""Tests for the THROWAWAY roll-probe readings (tools/probe_roll_readings.py).

Matched pair: delete this file when that probe is deleted. It exists because the
probe's output is a VERDICT that a large engineering decision rests on -- whether
the static depth noise decorrelates with pose, and therefore whether multi-view
can help at all (docs/inspection-roll-probe-handoff.md section 4). A silently
wrong frame convention here would answer that question confidently and wrongly,
which is the exact failure mode the whole handoff is written to prevent.

The decisive tests are the two pairs that plant a KNOWN camera-locked or
scene-locked pattern and require the probe to tell them apart:

    test_camera_locked_dropout_agrees_only_in_the_baseline_frame
    test_scene_locked_dropout_agrees_only_in_the_work_frame
    test_camera_locked_residual_correlates_only_after_baseline_expression
    test_scene_locked_residual_correlates_only_in_work_coordinates

Everything else is scaffolding those four depend on.
"""
import numpy as np
import pytest

from tools.probe_roll_readings import (
    _signed_delta, best_circular_shift_deg, correlate, detrend_polar,
    dropout_axis_deg, residual_polar, rotate_polar, rotate_to_baseline,
    sector_counts)


# --------------------------------------------------------------------------
# synthetic scenes
# --------------------------------------------------------------------------
def ring_points(*, centre=(0.0, 0.0), radius=43.0, gap=None, per_deg=40,
                jitter=0.0, seed=0):
    """Points on a ring, optionally missing an angular sector.

    ``gap`` is a (start_deg, end_deg) work-frame sector that is left EMPTY --
    the dropout this probe has to localise.
    """
    rng = np.random.default_rng(seed)
    # Offset off the whole degree so no point sits ON a 10 deg bin boundary:
    # a boundary point's angle survives the cos/sin/arctan2 round trip as
    # 139.9999997, which lands in the neighbouring bin and makes an exact-count
    # assertion meaningless. The offset is a property of the FIXTURE, not a
    # tolerance on the behaviour being tested.
    angles = np.repeat(np.arange(0.25, 360.0, 1.0), per_deg)
    if gap is not None:
        lo, hi = gap
        keep = ~((angles >= lo) & (angles < hi))
        angles = angles[keep]
    r = radius + (rng.normal(0.0, jitter, angles.size) if jitter else 0.0)
    theta = np.radians(angles)
    xy = np.column_stack([centre[0] + r * np.cos(theta),
                          centre[1] + r * np.sin(theta)])
    return xy


def annulus_points(*, centre=(0.0, 0.0), inner=48.0, outer=61.0, step=1.0):
    """A dense grid of points covering an annulus, for residual-field tests."""
    xs = np.arange(centre[0] - outer, centre[0] + outer + step, step)
    ys = np.arange(centre[1] - outer, centre[1] + outer + step, step)
    gx, gy = np.meshgrid(xs, ys)
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    r = np.hypot(xy[:, 0] - centre[0], xy[:, 1] - centre[1])
    return xy[(r >= inner) & (r <= outer)]


def angles_deg(xy, centre=(0.0, 0.0)):
    return np.degrees(np.arctan2(xy[:, 1] - centre[1],
                                 xy[:, 0] - centre[0])) % 360.0


# --------------------------------------------------------------------------
# sector counting
# --------------------------------------------------------------------------
def test_sector_counts_bins_a_full_ring_evenly():
    xy = ring_points()
    counts = sector_counts(xy, (0.0, 0.0), inner_mm=38.0, outer_mm=48.0)
    assert counts.shape == (36,)
    assert counts.sum() == len(xy)
    assert counts.min() > 0
    assert counts.max() - counts.min() <= 1


def test_sector_counts_excludes_points_outside_the_annulus():
    on_ring = ring_points(radius=43.0)
    far_away = ring_points(radius=90.0)
    xy = np.vstack([on_ring, far_away])
    counts = sector_counts(xy, (0.0, 0.0), inner_mm=38.0, outer_mm=48.0)
    assert counts.sum() == len(on_ring)


def test_sector_counts_collapses_where_the_ring_is_missing():
    xy = ring_points(gap=(140.0, 190.0))
    counts = sector_counts(xy, (0.0, 0.0), inner_mm=38.0, outer_mm=48.0)
    empty = np.flatnonzero(counts == 0)
    # 10 deg bins: 140-190 is bins 14..18
    assert empty.tolist() == [14, 15, 16, 17, 18]


def test_dropout_axis_points_at_the_middle_of_the_missing_sector():
    xy = ring_points(gap=(140.0, 190.0))
    counts = sector_counts(xy, (0.0, 0.0), inner_mm=38.0, outer_mm=48.0)
    assert dropout_axis_deg(counts) == pytest.approx(165.0, abs=3.0)


# --------------------------------------------------------------------------
# frame expression
# --------------------------------------------------------------------------
def test_rotate_to_baseline_shifts_a_profile_by_the_baseline_angle():
    profile = np.zeros(36)
    profile[14] = 1.0                      # a feature at work-frame 140-150 deg
    rotated = rotate_to_baseline(profile, 60.0)
    # relative to a baseline at 60 deg, that feature sits at 80-90 deg = bin 8
    assert np.flatnonzero(rotated).tolist() == [8]


def test_rotate_to_baseline_is_a_no_op_at_zero():
    profile = np.arange(36, dtype=float)
    assert np.array_equal(rotate_to_baseline(profile, 0.0), profile)


def test_signed_delta_takes_the_short_way_across_the_wrap():
    assert _signed_delta(10.0, 70.0) == pytest.approx(60.0)
    assert _signed_delta(70.0, 10.0) == pytest.approx(-60.0)
    assert _signed_delta(350.0, 10.0) == pytest.approx(20.0)
    assert _signed_delta(10.0, 350.0) == pytest.approx(-20.0)


# --------------------------------------------------------------------------
# whole-profile registration
#
# Measured on the real archive 2026-09-01: the layer-2 deficit is NOT one clean
# sector. Counts per 10 deg run [.. 200, 80, 6, 65, 122, 13, 118 ..] with further
# lows at 30, 240, 260 and 330 deg, so the circular mean of the deficit lands at
# 249 deg -- an average of everything low, not the collapse at 140-190. The
# verdict therefore cannot ride on an axis. It rides on registering the two
# WHOLE profiles against each other and asking how far one had to rotate.
# --------------------------------------------------------------------------
def test_best_shift_is_zero_for_identical_profiles():
    profile = sector_counts(ring_points(gap=(140.0, 190.0)), (0.0, 0.0),
                            inner_mm=38.0, outer_mm=48.0)
    assert best_circular_shift_deg(profile, profile) == pytest.approx(0.0, abs=1e-9)


def test_best_shift_recovers_a_known_rotation():
    profile = sector_counts(ring_points(gap=(140.0, 190.0)), (0.0, 0.0),
                            inner_mm=38.0, outer_mm=48.0)
    rotated = np.roll(profile, 6)          # +60 deg in work angle
    assert best_circular_shift_deg(profile, rotated) == pytest.approx(60.0, abs=1e-9)


def test_best_shift_is_signed_and_takes_the_short_way_round():
    profile = sector_counts(ring_points(gap=(140.0, 190.0)), (0.0, 0.0),
                            inner_mm=38.0, outer_mm=48.0)
    assert best_circular_shift_deg(profile, np.roll(profile, -3)) == pytest.approx(-30.0)


def test_best_shift_survives_a_multi_lobed_deficit():
    """The real profile has several lows. Registration must still find the
    rotation, which is exactly what the axis statistic fails to do."""
    measured = np.array([158, 238, 160, 28, 170, 266, 247, 107, 197, 278, 345,
                         455, 419, 200, 80, 6, 65, 122, 13, 118, 129, 250, 156,
                         141, 69, 100, 51, 147, 145, 106, 227, 282, 286, 54,
                         161, 149], float)
    assert best_circular_shift_deg(measured, np.roll(measured, 6)) == pytest.approx(60.0)
    assert dropout_axis_deg(measured) == pytest.approx(249.5, abs=1.0)


# --------------------------------------------------------------------------
# THE DECISIVE PAIR, dropout
# --------------------------------------------------------------------------
def test_camera_locked_dropout_registers_at_the_baseline_change():
    """A dropout that RIDES the camera: registering B against A recovers the
    change in baseline angle, not zero."""
    a = sector_counts(ring_points(gap=(140.0, 190.0)), (0.0, 0.0),
                      inner_mm=38.0, outer_mm=48.0)
    b = sector_counts(ring_points(gap=(200.0, 250.0)), (0.0, 0.0),
                      inner_mm=38.0, outer_mm=48.0)
    assert best_circular_shift_deg(a, b) == pytest.approx(60.0, abs=5.0)


def test_scene_locked_dropout_registers_at_zero():
    """A dropout that STAYS PUT while the camera rolls 60 deg."""
    a = sector_counts(ring_points(gap=(140.0, 190.0)), (0.0, 0.0),
                      inner_mm=38.0, outer_mm=48.0)
    b = sector_counts(ring_points(gap=(140.0, 190.0)), (0.0, 0.0),
                      inner_mm=38.0, outer_mm=48.0)
    assert best_circular_shift_deg(a, b) == pytest.approx(0.0, abs=5.0)


# --------------------------------------------------------------------------
# THE DECISIVE PAIR, dropout (axis statistic -- descriptive only)
# --------------------------------------------------------------------------
def test_camera_locked_dropout_agrees_only_in_the_baseline_frame():
    """A dropout that RIDES the camera: it must agree once each capture is
    expressed against its own baseline, and disagree in work coordinates."""
    baseline_a, baseline_b = 0.0, 60.0
    # the sector sits at a fixed 140-190 deg RELATIVE TO EACH BASELINE
    a = sector_counts(ring_points(gap=(140.0, 190.0)), (0.0, 0.0),
                      inner_mm=38.0, outer_mm=48.0)
    b = sector_counts(ring_points(gap=(200.0, 250.0)), (0.0, 0.0),
                      inner_mm=38.0, outer_mm=48.0)

    work_gap = abs(dropout_axis_deg(a) - dropout_axis_deg(b))
    rel_gap = abs(dropout_axis_deg(rotate_to_baseline(a, baseline_a))
                  - dropout_axis_deg(rotate_to_baseline(b, baseline_b)))

    assert rel_gap < 5.0
    assert work_gap > 45.0


def test_scene_locked_dropout_agrees_only_in_the_work_frame():
    """A dropout that STAYS PUT while the camera rolls."""
    baseline_a, baseline_b = 0.0, 60.0
    a = sector_counts(ring_points(gap=(140.0, 190.0)), (0.0, 0.0),
                      inner_mm=38.0, outer_mm=48.0)
    b = sector_counts(ring_points(gap=(140.0, 190.0)), (0.0, 0.0),
                      inner_mm=38.0, outer_mm=48.0)

    work_gap = abs(dropout_axis_deg(a) - dropout_axis_deg(b))
    rel_gap = abs(dropout_axis_deg(rotate_to_baseline(a, baseline_a))
                  - dropout_axis_deg(rotate_to_baseline(b, baseline_b)))

    assert work_gap < 5.0
    assert rel_gap > 45.0


# --------------------------------------------------------------------------
# residual fields
# --------------------------------------------------------------------------
def test_residual_polar_averages_into_radius_by_angle_cells():
    xy = annulus_points()
    residual = np.ones(len(xy)) * 0.4
    grid = residual_polar(xy, residual, (0.0, 0.0), inner_mm=48.0, outer_mm=61.0,
                          r_bins=4, theta_bins=36)
    assert grid.shape == (4, 36)
    assert np.nanmax(np.abs(grid - 0.4)) < 1e-9


def test_residual_polar_leaves_uncovered_cells_undefined():
    xy = annulus_points()
    keep = angles_deg(xy) < 180.0
    grid = residual_polar(xy[keep], np.zeros(int(keep.sum())), (0.0, 0.0),
                          inner_mm=48.0, outer_mm=61.0, r_bins=4, theta_bins=36)
    assert np.isnan(grid[:, 18:]).all()
    assert np.isfinite(grid[:, :18]).all()


def test_detrend_polar_removes_a_radially_symmetric_bowl():
    """A bowl carries NO angular information, so it must not survive to bias a
    correlation. An angular pattern on top of it must survive untouched."""
    theta = np.radians((np.arange(36) + 0.5) * 10.0)
    bowl = np.array([0.0, 0.5, 1.0, 1.5])[:, None] * np.ones(36)[None, :]
    pattern = np.cos(2 * theta)[None, :] * np.ones(4)[:, None]
    detrended = detrend_polar(bowl + pattern)
    assert np.nanmax(np.abs(detrended - pattern)) < 1e-9


def test_identical_fields_correlate_at_one():
    rng = np.random.default_rng(3)
    grid = rng.normal(size=(4, 36))
    assert correlate(grid, grid.copy()) == pytest.approx(1.0, abs=1e-9)


def test_independent_noise_does_not_correlate():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(8, 360))
    b = rng.normal(size=(8, 360))
    assert abs(correlate(a, b)) < 0.1


def test_correlate_ignores_cells_either_field_leaves_undefined():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(4, 36))
    b = a.copy()
    b[:, :6] = np.nan
    assert correlate(a, b) == pytest.approx(1.0, abs=1e-9)


def test_rotate_polar_shifts_along_the_angular_axis_only():
    grid = np.zeros((3, 36))
    grid[:, 14] = 1.0
    rotated = rotate_polar(grid, 60.0)
    assert np.flatnonzero(rotated[0]).tolist() == [8]
    assert np.flatnonzero(rotated[1]).tolist() == [8]


# --------------------------------------------------------------------------
# THE DECISIVE PAIR, residual field
# --------------------------------------------------------------------------
def _residual_grid(pattern_deg_offset, *, theta_bins=360):
    """A residual field whose angular pattern is offset by a known angle."""
    xy = annulus_points(step=0.5)
    theta = np.radians(angles_deg(xy) - pattern_deg_offset)
    residual = np.cos(2 * theta)
    return residual_polar(xy, residual, (0.0, 0.0), inner_mm=48.0, outer_mm=61.0,
                          r_bins=6, theta_bins=theta_bins)


def test_camera_locked_residual_correlates_only_after_baseline_expression():
    """The static residual pattern rides the camera: correlate the two captures
    in work coordinates and they disagree; express each against its own baseline
    first and they agree. This is the direct decorrelation test."""
    baseline_a, baseline_b = 0.0, 60.0
    a = _residual_grid(baseline_a)
    b = _residual_grid(baseline_b)

    work = correlate(detrend_polar(a), detrend_polar(b))
    relative = correlate(detrend_polar(rotate_polar(a, baseline_a)),
                         detrend_polar(rotate_polar(b, baseline_b)))

    assert relative > 0.95
    assert work < 0.0


def test_pair_probe_skirt_uses_the_shared_centre_it_is_given():
    """The 2026-09-01 review defect in tools/probe_roll_pair.py: each capture
    fitted its OWN ring centre, so the skirt annulus moved between the captures
    being compared and a biased fit in one arm could masquerade as an angular
    difference. A shared reference must override the take's own fit."""
    from tools.probe_roll_pair import skirt_histogram

    xy = annulus_points(inner=48.0, outer=61.0, step=1.0)
    points = np.column_stack([xy, np.zeros(len(xy))])
    take = {"points": points, "height": np.full(len(points), 5.0),
            "centre": np.array([25.0, 0.0]), "radius": 43.0, "floor": 1.0}

    own = skirt_histogram(take)
    shared = skirt_histogram(take, centre=np.array([0.0, 0.0]), radius=43.0)

    # the true centre puts the whole annulus inside the skirt; the take's own
    # (wrong) centre does not
    assert shared["band_points"] == len(points)
    assert own["band_points"] < shared["band_points"]


def test_scene_locked_residual_correlates_only_in_work_coordinates():
    """The pattern is a property of the board: it stays put while the camera
    rolls, so work coordinates agree and baseline coordinates do not."""
    baseline_a, baseline_b = 0.0, 60.0
    a = _residual_grid(0.0)
    b = _residual_grid(0.0)

    work = correlate(detrend_polar(a), detrend_polar(b))
    relative = correlate(detrend_polar(rotate_polar(a, baseline_a)),
                         detrend_polar(rotate_polar(b, baseline_b)))

    assert work > 0.95
    assert relative < 0.0
