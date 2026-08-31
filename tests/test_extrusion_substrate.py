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


def test_healthy_low_point_count_frames_are_not_falsely_refused():
    """Task 4 review round 3: the breakdown guard's clause (a) originally
    used a FIXED 1% fraction, which false-fired on healthy, uncontaminated
    frames at low n (measured up to 18% of trials at n=80 -- 1% of 50 points
    is half a point, so a single unlucky point tripped it). This is a
    supported input size (fit() explicitly accepts frames down to 50
    points), not a pathological one. Pins the fix: the n-adaptive
    (Poisson-tail) threshold must drive that rate to ~0 across n=50..500 --
    30 trials per size, no bead at all, budget generous enough not to be
    flaky (measured actual rate: 0-1% per size) but far below the ~10-18%
    the fixed-fraction version showed."""
    total_trials = 0
    total_false_fires = 0
    for n in (50, 60, 80, 100, 150, 200, 300, 500):
        for seed in range(30):
            pts, _ = _tilted_scene(n=n, bead_fraction=0.0, seed=1_000_000 + seed)
            total_trials += 1
            try:
                PlaneSubstrate.fit(pts)
            except RuntimeError:
                total_false_fires += 1
    assert total_false_fires <= max(3, round(0.05 * total_trials)), (
        f"{total_false_fires}/{total_trials} healthy low-n frames were refused")


def test_majority_deposit_is_refused_at_low_point_count_too():
    """Task 4 review round 3: a bead-locked sparse frame is exactly as wrong
    as a bead-locked dense one -- the n-adaptive threshold must not trade
    low-n sensitivity for its fixed false-fire fix. n=100, 75% deposit
    fraction (measured: caught 50/50 trials at this n)."""
    pts, _ = _tilted_scene(n=100, bead_fraction=0.75, seed=11)
    with pytest.raises(RuntimeError, match="substrate fit refused"):
        PlaneSubstrate.fit(pts)


def test_clause_b_margin_covers_noise_up_to_4mm():
    """Task 4 review round 4: clause (b)'s sigma multiplier
    (_BREAKDOWN_SIGMA_MULT) is a validated constant, not a round number --
    round 3 briefly tightened it 2.5x -> 2.0x to close a low-n catch gap in
    clause (a), which silently moved its false-fire cliff from noise_mm=5.0
    down to noise_mm=4.0 (measured 48.3% false-fire there) because that
    round only re-ran the sweep for the change it was making, not the sweep
    that had validated clause (b) in the first place. Reverted to 2.5x
    (controller ruling): real per-take sigma is 0.44-0.61mm, so 2.5x keeps
    ~8-11x headroom, and clause (b) is a BACKSTOP behind clause (a) -- it
    belongs far from any plausible operating point, not tuned flush against
    the detector it backs up. Pins the margin directly (10 seeds at the
    exact noise level round 3's tightening broke) so the next tightening
    trips this test instead of needing a reviewer to catch it."""
    n = 20_000
    for seed in range(10):
        pts, _ = _tilted_scene(n=n, noise_mm=4.0, bead_fraction=0.0,
                                seed=900_000 + seed)
        fit = PlaneSubstrate.fit(pts)   # must not raise
        assert fit.sigma_mm > 0.0


# ------------------------------------------ compactness: the gate's one real job

from tasni.modules.extrusion.substrate import compactness_filter


def _arc_points(radius_mm=41.0, span_deg=180.0, per_deg=6):
    angles = np.radians(np.linspace(0.0, span_deg, int(span_deg * per_deg)))
    return np.column_stack([radius_mm * np.cos(angles),
                            radius_mm * np.sin(angles),
                            np.full(len(angles), 3.0)])


def _patch_points(n, center=(60.0, 0.0)):
    rng = np.random.default_rng(11)
    xy = rng.uniform(-4.0, 4.0, size=(n, 2)) + np.asarray(center)
    return np.column_stack([xy, np.full(n, 3.0)])


def test_rejects_the_compact_patch_and_keeps_an_arc_of_equal_count():
    """The 22-point checker patch that exhausted the branch guard was COMPACT,
    not colourful -- an arc of the same pixel count is long and survives."""
    arc = _arc_points()
    patch = _patch_points(len(arc), center=(80.0, 0.0))
    counts = {}
    kept = compactness_filter(np.vstack([arc, patch]), mm_per_pixel=1.0,
                              bead_mm=9.0, min_length_beads=3.0,
                              min_points=10, counts=counts)
    assert counts["compactness_components"] == 2
    assert counts["compactness_kept_components"] == 1
    assert len(kept) == len(arc)
    assert np.allclose(np.sort(kept[:, 0]), np.sort(arc[:, 0]))


def test_fail_open_when_the_filter_would_starve_the_chain():
    """A thin or fragmented real ring must never be zeroed by topology alone:
    below min_points the cloud passes through untouched, recorded as bypassed."""
    patch = _patch_points(40)
    counts = {}
    kept = compactness_filter(patch, mm_per_pixel=1.0, bead_mm=9.0,
                              min_length_beads=3.0, min_points=10, counts=counts)
    assert counts["compactness_bypassed"] == 1
    assert len(kept) == len(patch)


def _bead_strip(x0, x1, width_mm, z=3.0, step=0.5):
    """A filled rectangle x in [x0, x1], y in [-width/2, width/2] -- mimics a
    real deposit's raster footprint, which has WIDTH (a mathematically 1px-wide
    line is degenerate for morphological closing: the erosion half of a close
    can never restore a bridge that thin, so it cannot exercise the weld
    reach at all)."""
    xs = np.arange(x0, x1 + step / 2, step)
    ys = np.arange(-width_mm / 2, width_mm / 2 + step / 2, step)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])


def test_weld_reach_is_about_half_a_bead_not_a_full_one():
    """Review round 1, Important 1: the first cut's close radius was a
    half-bead RADIUS, which a morphological close bridges at roughly TWICE
    ITS RADIUS -- i.e. a FULL bead width, silently switching the filter's
    rejection power off exactly at the scale of the 2026-08-29 checker patch
    (12 mm outside the ring -- close to a 0.6-bead gap here). Pins the fix
    (close radius = bead_mm / (4*mm_per_pixel), floor) against two
    bead-length, bead-width strips (bead_mm=20, mm_per_pixel=1.0), measured
    directly against `compactness_filter`'s own component count: a 2 mm
    sub-bead-scale gap (speckle-inside-a-bead scale) must still weld into
    one component, and a 0.6-bead (12 mm) gap -- the scale that mattered on
    2026-08-29 -- must not."""
    bead_mm, mm_per_pixel = 20.0, 1.0
    seg1 = _bead_strip(0.0, bead_mm, bead_mm)

    small_gap = _bead_strip(bead_mm + 2.0, bead_mm + 2.0 + bead_mm, bead_mm)
    counts = {}
    compactness_filter(np.vstack([seg1, small_gap]), mm_per_pixel=mm_per_pixel,
                       bead_mm=bead_mm, min_length_beads=100.0, min_points=1,
                       counts=counts)
    assert counts["compactness_components"] == 1        # 2 mm: still welds

    wide_gap = _bead_strip(bead_mm + 0.6 * bead_mm,
                           bead_mm + 0.6 * bead_mm + bead_mm, bead_mm)
    counts = {}
    compactness_filter(np.vstack([seg1, wide_gap]), mm_per_pixel=mm_per_pixel,
                       bead_mm=bead_mm, min_length_beads=100.0, min_points=1,
                       counts=counts)
    assert counts["compactness_components"] == 2         # 0.6 bead: separate


def _wide_arc_points(radius_mm, span_deg, width_mm, start_deg=0.0, per_deg=6,
                     per_width=2):
    """A deposit-width annular segment (unlike `_arc_points`, which is a bare
    1px-wide curve) -- the near-isotropic raster footprint that exposed the
    orientation bug: at a short span/radius its covariance eigenvalues sit
    close enough together that the eigenvector projection's "principal axis"
    became numerically unstable."""
    n = max(2, int(round(span_deg * per_deg)))
    angles = np.radians(np.linspace(start_deg, start_deg + span_deg, n))
    m = max(1, int(round(width_mm * per_width)))
    radii = np.linspace(radius_mm - width_mm / 2, radius_mm + width_mm / 2, m)
    aa, rr = np.meshgrid(angles, radii)
    x = rr.ravel() * np.cos(aa.ravel())
    y = rr.ravel() * np.sin(aa.ravel())
    return np.column_stack([x, y, np.full(len(x), 3.0)])


def test_six_identical_arcs_decide_consistently_regardless_of_grid_orientation():
    """Review round 1, Required (promoted from Minor): six geometrically
    identical 38-degree arcs (same radius/span/width, rotated only in start
    angle -- so the raster footprint is the same physical shape, just at a
    different orientation on the pixel grid) must reach the SAME keep/drop
    decision. Measured before this fix: the eigenvector-projection extent
    swung between the six rotations far enough (~10.8-12.2 mm) that a
    threshold of 11.0 mm split them 3 kept / 3 dropped -- purely from grid
    orientation, nothing physical changed. `cv2.minAreaRect`'s longer side
    is the true oriented caliper (no eigenvalue-tie ambiguity to swing on);
    at the same 11.0 mm threshold every rotation now measures inside a
    tight, sub-mm band (~9.9-10.7 mm) and all six decide the same way."""
    radius_mm, span_deg, width_mm = 10.0, 38.0, 9.0
    mm_per_pixel = 2.0
    bead_mm = 11.0                # with min_length_beads=1.0 -> threshold 11.0 mm,
    min_length_beads = 1.0        # inside the old eigen swing, outside the new one
    kept_counts = []
    for start_deg in (0, 15, 30, 45, 60, 75):
        arc = _wide_arc_points(radius_mm, span_deg, width_mm, start_deg=start_deg)
        counts = {}
        compactness_filter(arc, mm_per_pixel=mm_per_pixel, bead_mm=bead_mm,
                           min_length_beads=min_length_beads, min_points=0,
                           counts=counts)
        kept_counts.append(counts["compactness_kept_components"])
    assert len(set(kept_counts)) == 1, (
        f"decision flipped across rotations purely on grid orientation: {kept_counts}")
