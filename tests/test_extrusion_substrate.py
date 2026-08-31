"""PlaneSubstrate: the fitted-substrate reference (spec §3.2-§3.4).

Synthetic surfaces with a known bead: the estimator must recover the plane the
bead sits on, one-sidedly (deposit only ever contaminates from above), score its
own noise from the uncontaminated lower half, and do it all bit-identically."""
import math

import numpy as np
import pytest

from tasni.modules.extrusion.substrate import PlaneSubstrate


def _tilted_scene(*, tilt_deg=0.55, noise_mm=0.5, bead_height_mm=8.0,
                  bead_fraction=0.15, n=20_000, seed=7):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-120.0, 120.0, size=(n, 2))
    slope = math.tan(math.radians(tilt_deg))
    z = slope * xy[:, 0] - 1.2 + rng.normal(0.0, noise_mm, n)
    bead = rng.random(n) < bead_fraction
    z[bead] += bead_height_mm            # one-sided contamination, like a real bead
    return np.column_stack([xy, z]), slope


def test_recovers_the_plane_under_one_sided_contamination():
    pts, slope = _tilted_scene()
    fit = PlaneSubstrate.fit(pts)
    assert fit.a == pytest.approx(slope, abs=0.002)      # tilt recovered
    assert fit.c == pytest.approx(-1.2, abs=0.15)        # offset recovered
    # sigma runs ~8-13% high under this scene's own 15% bead (measured across
    # a 30-seed sweep); rel=0.20 leaves headroom for that while still catching
    # a real magnitude regression -- sigma now multiplies the bias correction,
    # so this bound is precisely what would notice one.
    assert fit.sigma_mm == pytest.approx(0.5, rel=0.20)  # sigma from the clean half
    clean = ~(pts[:, 2] - (fit.a * pts[:, 0] + fit.b * pts[:, 1] + fit.c) > 4.0)
    heights = fit.height(pts[clean])
    assert abs(float(np.median(heights))) < 0.1          # substrate sits at height 0


def test_fit_is_bit_identical_across_repeats():
    """The RANSAC failure mode, pinned (spec §3.2): a chain that cannot
    reprocess a frame to the same number twice is not a measurement chain."""
    pts, _ = _tilted_scene()
    first, second = PlaneSubstrate.fit(pts), PlaneSubstrate.fit(pts)
    assert (first.a, first.b, first.c, first.sigma_mm) \
        == (second.a, second.b, second.c, second.sigma_mm)


def test_derived_floor_is_clamped_both_ways():
    pts, _ = _tilted_scene(noise_mm=0.55)
    fit = PlaneSubstrate.fit(pts)
    assert 1.0 <= fit.floor_mm(3.0) <= 2.0
    assert fit.floor_mm(100.0) == 2.0        # ceiling: the k=4 cliff (spec §3.4)
    assert fit.floor_mm(0.0) == 1.0          # floor: never open to raw noise


def test_a_wall_is_refused_not_measured():
    rng = np.random.default_rng(3)
    n = 5000
    x = rng.uniform(-50, 50, n)
    z = rng.uniform(0, 100, n)
    pts = np.column_stack([x, 0.8 * z + rng.normal(0, 0.3, n), z])   # steep surface
    with pytest.raises(RuntimeError, match="substrate fit refused"):
        PlaneSubstrate.fit(pts)


def test_too_few_points_is_a_loud_error():
    with pytest.raises(RuntimeError, match="substrate fit needs"):
        PlaneSubstrate.fit(np.zeros((10, 3)))


def test_bias_correction_is_audited_in_the_report():
    """Task 4 review, Important 3: the correction that moves every measured
    height must not be invisible to the frame's own report (spec §4). A
    refactor that dropped both keys would still pass every other test here
    without this one."""
    pts, _ = _tilted_scene()
    fit = PlaneSubstrate.fit(pts)
    report = fit.to_report()
    assert "bias_correction_mm" in report
    assert "bias_correction_sigma" in report
    assert fit.bias_correction_mm == pytest.approx(
        fit.bias_correction_sigma * fit.sigma_mm)
    # to_report() rounds (4dp / 5dp) for display -- compare with matching tolerance.
    assert report["bias_correction_mm"] == pytest.approx(fit.bias_correction_mm, abs=1e-4)
    assert report["bias_correction_sigma"] == pytest.approx(fit.bias_correction_sigma, abs=1e-5)


def test_majority_deposit_is_refused_not_measured_as_perfect():
    """Task 4 review, Important 2: past ~50% deposit in the fit region the
    raw IRLS can converge onto the deposit's top instead of the table, and
    (before this guard) reported inlier_fraction=1.000 -- the one number an
    operator would trust -- while measuring height above the deposit, not
    the substrate. 75% deposit fraction reproduces that breakdown."""
    pts, _ = _tilted_scene(bead_fraction=0.75)
    with pytest.raises(RuntimeError, match="substrate fit refused"):
        PlaneSubstrate.fit(pts)
