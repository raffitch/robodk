import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core.characterize import (
    choose_dstar, known_length_error_mm, plane_metrics, summarize_distance_trial,
)
from tasni.modules.scan.plane import fit_plane
from tools.characterize_distance import latest_characterization


def _plane_set(z=400.0, sigma=0.3, n=500, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-150, 150, (n, 2))
    return np.column_stack([xy, np.full(n, z) + rng.normal(0, sigma, n)])


def test_plane_metrics_repeatability():
    sets = [_plane_set(z=400.0 + dz, seed=i) for i, dz in enumerate((0.0, 0.1, -0.1))]
    m = plane_metrics(sets)
    assert m["plane_rms_mm"] < 0.5
    assert m["height_repeat_mm"] < 0.3
    assert m["normal_repeat_deg"] < 0.5
    bad = plane_metrics(sets + [_plane_set(z=403.0, seed=9)])
    assert bad["height_repeat_mm"] > 1.0


def test_known_length_error():
    a = np.tile([0.0, 0.0, 400.0], (10, 1))
    b = np.tile([297.4, 0.0, 400.0], (10, 1))
    assert known_length_error_mm(a, b, 297.0) == pytest.approx(0.4, abs=1e-9)


def _trial(d, rms=0.3, length_err=0.3, coverage=0.9):
    sets = [_plane_set(z=d, sigma=rms, seed=i) for i in range(3)]
    a = np.tile([0.0, 0.0, float(d)], (5, 1))
    b = np.tile([297.0 + length_err, 0.0, float(d)], (5, 1))
    return summarize_distance_trial(d, sets, [(a, b, 297.0)], coverage)


# Generous, non-binding thresholds for the two gates added in the review fix round
# (plane_max_mm, length_spread_mm) so tests that are not exercising those specific
# gates don't accidentally trip on them. `_trial`'s captures have no outliers and its
# length samples are exact-duplicate rows (spread 0), so both are always well inside
# these limits unless a test deliberately contaminates the data.
_LOOSE_MAX_PLANE_MAX_MM = 2.0
_LOOSE_MAX_LENGTH_SPREAD_MM = 1.0


def test_choose_dstar_prefers_closest_passing_distance():
    trials = [_trial(300.0, rms=1.5), _trial(400.0), _trial(600.0)]
    best = choose_dstar(trials, max_rms_mm=1.0, max_height_repeat_mm=1.0,
                        max_normal_repeat_deg=1.0, max_length_err_mm=1.0,
                        min_coverage_frac=0.5, max_plane_max_mm=_LOOSE_MAX_PLANE_MAX_MM,
                        max_length_spread_mm=_LOOSE_MAX_LENGTH_SPREAD_MM)
    assert best is not None and best.distance_mm == 400.0
    none = choose_dstar([_trial(300.0, rms=2.0)], max_rms_mm=1.0,
                        max_height_repeat_mm=1.0, max_normal_repeat_deg=1.0,
                        max_length_err_mm=1.0, min_coverage_frac=0.5,
                        max_plane_max_mm=_LOOSE_MAX_PLANE_MAX_MM,
                        max_length_spread_mm=_LOOSE_MAX_LENGTH_SPREAD_MM)
    assert none is None


# --- Robustness beyond the brief (ambiguity resolution #3: non-finite inputs must
# never produce a confident-looking DistanceTrial, and choose_dstar must never be
# able to select one). Sibling modules in this codebase were previously found to
# silently accept NaN because `nan > tol` is False in a plain comparison.

def test_plane_metrics_nan_point_is_not_silently_dropped():
    """A single NaN point injected into one capture must not yield a falsely-clean
    metric — plane_metrics must surface non-finite contamination, not hide it behind
    a comparison that NaN always loses."""
    sets = [_plane_set(z=400.0 + dz, seed=i) for i, dz in enumerate((0.0, 0.1, -0.1))]
    sets[0] = sets[0].copy()
    sets[0][0] = [np.nan, np.nan, np.nan]
    m = plane_metrics(sets)
    assert not all(np.isfinite(v) for v in m.values())


def test_known_length_error_nan_input_is_nan_not_falsely_small():
    a = np.tile([0.0, 0.0, 400.0], (10, 1))
    a = a.copy()
    a[0] = [np.nan, 0.0, 400.0]
    b = np.tile([297.4, 0.0, 400.0], (10, 1))
    err = known_length_error_mm(a, b, 297.0)
    assert not np.isfinite(err)


def test_summarize_distance_trial_nan_metrics_do_not_look_confident():
    """A trial built from contaminated captures must report its metrics as
    non-finite, not as some small in-range number that would sneak past a gate."""
    sets = [_plane_set(z=400.0, seed=i) for i in range(3)]
    sets[0] = sets[0].copy()
    sets[0][0] = [np.nan, np.nan, np.nan]
    a = np.tile([0.0, 0.0, 400.0], (5, 1))
    b = np.tile([297.3, 0.0, 400.0], (5, 1))
    trial = summarize_distance_trial(400.0, sets, [(a, b, 297.0)], 0.9)
    finite = (np.isfinite(trial.plane_rms_mm) and np.isfinite(trial.plane_max_mm)
              and np.isfinite(trial.height_repeat_mm)
              and np.isfinite(trial.normal_repeat_deg))
    assert not finite


def test_choose_dstar_never_selects_a_nan_trial_even_if_closest():
    """A nearer trial with non-finite metrics must be excluded outright, not chosen
    just because plain `<=` comparisons against NaN are always False (which would
    normally make a NaN trial LOOK like it satisfies every upper-bound gate)."""
    contaminated = summarize_distance_trial(
        300.0,
        [_plane_set(z=300.0, seed=i) for i in range(3)],
        [(np.array([[np.nan, 0.0, 300.0]]), np.array([[297.0, 0.0, 300.0]]), 297.0)],
        0.9,
    )
    healthy = _trial(400.0)
    best = choose_dstar([contaminated, healthy], max_rms_mm=1.0,
                        max_height_repeat_mm=1.0, max_normal_repeat_deg=1.0,
                        max_length_err_mm=1.0, min_coverage_frac=0.5,
                        max_plane_max_mm=_LOOSE_MAX_PLANE_MAX_MM,
                        max_length_spread_mm=_LOOSE_MAX_LENGTH_SPREAD_MM)
    assert best is not None and best.distance_mm == 400.0


def test_choose_dstar_all_nan_trials_returns_none():
    contaminated = summarize_distance_trial(
        300.0,
        [_plane_set(z=300.0, seed=i) for i in range(3)],
        [(np.array([[np.nan, 0.0, 300.0]]), np.array([[297.0, 0.0, 300.0]]), 297.0)],
        0.9,
    )
    none = choose_dstar([contaminated], max_rms_mm=1.0, max_height_repeat_mm=1.0,
                        max_normal_repeat_deg=1.0, max_length_err_mm=1.0,
                        min_coverage_frac=0.5, max_plane_max_mm=_LOOSE_MAX_PLANE_MAX_MM,
                        max_length_spread_mm=_LOOSE_MAX_LENGTH_SPREAD_MM)
    assert none is None


# --- Review fix round: Finding 1 ----------------------------------------------
#
# The original version of this test used near-identical per-capture normals (a
# 500-point patch of noise with sigma=0.01mm, no deliberate tilt), so the mean
# normal and every per-capture normal were all ~(0, 0, 1) regardless of whether
# the implementation projects onto the mean or onto each capture's own normal —
# it could not tell the two apart. Independently confirmed by the reviewer: a
# hand-rolled "wrong" plane_metrics that projects each centroid onto its OWN
# normal passed that old test unchanged (0.8169 vs the expected 0.8165).
#
# This version gives each capture a GENUINELY DIVERGENT normal (real tilts of
# -20/0/+20 degrees about the X axis) on top of a real 50mm height ladder
# (400/450/350mm), and directly contrasts plane_metrics's real output against a
# hand-rolled per-capture-normal variant built the same way the reviewer's was.

def _tilted_plane_set(theta_deg, z, sigma=0.05, n=500, seed=0):
    """A flat noisy patch, tilted by theta_deg about the X axis, then raised to
    height z. For theta=0 this is `_plane_set`; for theta!=0 the patch's true
    normal is (0, -sin(theta), cos(theta)), not (0, 0, 1)."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-150, 150, (n, 2))
    local = np.column_stack([xy, np.zeros(n) + rng.normal(0, sigma, n)])
    theta = np.radians(theta_deg)
    rot_x = np.array([[1.0, 0.0, 0.0],
                      [0.0, np.cos(theta), -np.sin(theta)],
                      [0.0, np.sin(theta), np.cos(theta)]])
    rotated = local @ rot_x.T
    rotated[:, 2] += z
    return rotated


def test_plane_metrics_height_repeat_uses_mean_normal_not_per_capture_normal():
    """Ambiguity resolution #4: height repeatability must project each capture's
    centroid onto the MEAN normal, not its own per-capture normal.

    Three captures at genuinely different tilts (-20/0/+20 degrees) and heights
    (400/450/350mm, true std 40.8248mm). Because the tilts are symmetric the mean
    normal is exactly (0, 0, 1), so the correct (mean-normal) projection recovers
    the height ladder almost exactly; projecting each centroid onto its OWN normal
    instead scales each height by cos(its own tilt), which is a DIFFERENT factor
    per capture and measurably distorts the recovered spread.
    """
    thetas = (-20.0, 0.0, 20.0)
    zs = (400.0, 450.0, 350.0)
    sets = [_tilted_plane_set(t, z, seed=i) for i, (t, z) in enumerate(zip(thetas, zs))]
    true_std = float(np.std(zs))  # 40.8248mm

    m = plane_metrics(sets)
    assert m["height_repeat_mm"] == pytest.approx(true_std, abs=1.5)

    # Hand-rolled WRONG variant (mirrors the reviewer's): same fit_plane calls,
    # but each centroid projected onto its OWN normal instead of the mean.
    own_heights = []
    for pts in sets:
        n, c, _ = fit_plane(pts, distance=6.0)
        n = np.asarray(n, dtype=float)
        if n[2] < 0:
            n = -n
        own_heights.append(float(np.asarray(c, dtype=float) @ n))
    wrong_std = float(np.std(own_heights))

    # The wrong variant is NOT a close call — it misses truth by nearly the full
    # tilt-induced distortion, clearly separated from both the truth and from the
    # real (correct) implementation's result.
    assert abs(wrong_std - true_std) > 5.0
    assert abs(m["height_repeat_mm"] - true_std) < abs(wrong_std - true_std)

    # Verification that this test actually discriminates (per the review finding):
    # substituting the wrong variant's number in place of the real one fails the
    # first assertion above, since |wrong_std - true_std| > 1.5 tolerance.
    assert wrong_std != pytest.approx(true_std, abs=1.5)


# --- Review fix round: Finding 2 (plan-mandated, controller ruling) -----------
#
# choose_dstar previously gated on plane_rms_mm (a MEAN residual) but never on
# plane_max_mm (the WORST-CASE residual) — the exact metric this module computes
# for catching a single bad point that a mean can average away. One 15mm-off
# point in 500 barely moves the RMS but spikes the max.

def _plane_set_with_outlier(z, outlier_mm, sigma=0.3, n=500, seed=0):
    pts = _plane_set(z=z, sigma=sigma, n=n, seed=seed).copy()
    pts[0, 2] += outlier_mm
    return pts


def test_choose_dstar_gates_on_plane_max_mm_not_just_rms():
    contaminated_sets = [
        _plane_set_with_outlier(300.0, 15.0, seed=0),
        _plane_set(300.0, seed=1),
        _plane_set(300.0, seed=2),
    ]
    a = np.tile([0.0, 0.0, 300.0], (5, 1))
    b = np.tile([297.3, 0.0, 300.0], (5, 1))
    contaminated = summarize_distance_trial(300.0, contaminated_sets, [(a, b, 297.0)], 0.9)
    healthy = _trial(400.0)

    # Precondition: the contamination is invisible to the mean-based rms gate...
    assert contaminated.plane_rms_mm < 1.0
    # ...but glaring in the worst-case residual, which is the whole point of
    # tracking plane_max_mm at all.
    assert contaminated.plane_max_mm > 5.0

    best = choose_dstar([contaminated, healthy], max_rms_mm=1.0,
                        max_height_repeat_mm=1.0, max_normal_repeat_deg=1.0,
                        max_length_err_mm=1.0, min_coverage_frac=0.5,
                        max_plane_max_mm=_LOOSE_MAX_PLANE_MAX_MM,
                        max_length_spread_mm=_LOOSE_MAX_LENGTH_SPREAD_MM)
    assert best is not None and best.distance_mm == 400.0

    # Verification that this is really the plane_max_mm gate doing the work: with
    # that one gate removed (max_plane_max_mm effectively infinite), the nearer
    # contaminated trial passes every OTHER criterion and would win instead —
    # i.e. this is the exact "closest passing" failure mode Finding 2 named.
    best_without_max_gate = choose_dstar(
        [contaminated, healthy], max_rms_mm=1.0, max_height_repeat_mm=1.0,
        max_normal_repeat_deg=1.0, max_length_err_mm=1.0, min_coverage_frac=0.5,
        max_plane_max_mm=float("inf"), max_length_spread_mm=_LOOSE_MAX_LENGTH_SPREAD_MM)
    assert best_without_max_gate is not None
    assert best_without_max_gate.distance_mm == 300.0


# --- Review fix round: Finding 3 (plan-mandated, controller ruling) -----------
#
# known_length_error_mm reduces many repeat point-pair measurements to the bias
# of their MEAN distance from truth, so a genuinely scattered but symmetric set of
# repeats (-10mm and +10mm around the true length) reports a deceptively perfect
# 0.0mm error. DistanceTrial.length_spread_mm (appended field) and
# choose_dstar(..., max_length_spread_mm=...) close that gap.

def test_length_spread_mm_catches_scatter_that_known_length_error_mm_hides():
    a = np.array([[0.0, 0.0, 400.0], [0.0, 0.0, 400.0]])
    b = np.array([[287.0, 0.0, 400.0], [307.0, 0.0, 400.0]])  # distances 287, 307
    # The bias-only metric is fooled: mean distance is exactly the true 297.0mm.
    assert known_length_error_mm(a, b, 297.0) == pytest.approx(0.0, abs=1e-9)

    trial = summarize_distance_trial(
        400.0, [_plane_set(z=400.0, seed=i) for i in range(3)], [(a, b, 297.0)], 0.9)
    assert trial.length_err_mm == pytest.approx(0.0, abs=1e-9)
    # But the 10mm of real per-sample scatter is not invisible to length_spread_mm.
    assert trial.length_spread_mm == pytest.approx(10.0, abs=1e-9)


def test_choose_dstar_gates_on_length_spread_excludes_scattered_but_unbiased_trial():
    scattered_a = np.array([[0.0, 0.0, 300.0], [0.0, 0.0, 300.0]])
    scattered_b = np.array([[287.0, 0.0, 300.0], [307.0, 0.0, 300.0]])
    scattered = summarize_distance_trial(
        300.0, [_plane_set(z=300.0, seed=i) for i in range(3)],
        [(scattered_a, scattered_b, 297.0)], 0.9)
    assert scattered.length_err_mm == pytest.approx(0.0, abs=1e-9)  # bias gate alone can't see it
    tight = _trial(400.0)  # identical-row length samples -> spread 0.0

    best = choose_dstar([scattered, tight], max_rms_mm=1.0, max_height_repeat_mm=1.0,
                        max_normal_repeat_deg=1.0, max_length_err_mm=1.0,
                        min_coverage_frac=0.5, max_plane_max_mm=_LOOSE_MAX_PLANE_MAX_MM,
                        max_length_spread_mm=2.0)
    assert best is not None and best.distance_mm == 400.0

    # Same verification style as Finding 2: with the spread gate effectively
    # disabled, the nearer scattered-but-unbiased trial passes everything else
    # and wins — confirming length_spread_mm is what excludes it above.
    best_without_spread_gate = choose_dstar(
        [scattered, tight], max_rms_mm=1.0, max_height_repeat_mm=1.0,
        max_normal_repeat_deg=1.0, max_length_err_mm=1.0, min_coverage_frac=0.5,
        max_plane_max_mm=_LOOSE_MAX_PLANE_MAX_MM, max_length_spread_mm=float("inf"))
    assert best_without_spread_gate is not None
    assert best_without_spread_gate.distance_mm == 300.0


# --- Review fix round: bundled minors ------------------------------------------

def test_plane_metrics_empty_input_raises_domain_specific_error():
    """Previously bare numpy internals leaked out (`np.max` on an empty array:
    "zero-size array to reduction operation maximum which has no identity") —
    not an error an operator or Task 16's CLI could act on."""
    with pytest.raises(ValueError, match="at least one capture"):
        plane_metrics([])


def test_plane_metrics_all_nan_capture_raises():
    """Regression pin for the all-non-finite-capture case (previously verified
    manually, not under test): every point in a capture being non-finite must
    fail loudly via fit_plane's own RANSAC-failure error, not return a
    corrupted-but-finite-looking result."""
    all_nan = np.full((50, 3), np.nan)
    with pytest.raises(ValueError):
        plane_metrics([all_nan])


# --- Task 16: latest_characterization (the pure, headlessly-importable half of
# tools/characterize_distance.py -- everything touching the camera/robot lives
# behind main(), so this is testable with no hardware at all).

def test_latest_characterization_reads_newest(tmp_path):
    """The exact contract Task 16 is built to (spec §5/§10): given the
    characterization directory, return the newest dated file's contents by
    filename (YYYYMMDD sorts lexicographically), or None when there is none."""
    assert latest_characterization(tmp_path) is None
    (tmp_path / "characterization-20260101.json").write_text(json.dumps({"dstar_mm": 400}))
    (tmp_path / "characterization-20260812.json").write_text(json.dumps({"dstar_mm": 350}))
    assert latest_characterization(tmp_path)["dstar_mm"] == 350


def test_latest_characterization_missing_directory_returns_none(tmp_path):
    """The lock-side gate (modules/scan/service.py) calls this on every surface
    lock, before any characterization has ever been run on a fresh machine --
    a missing directory must read as "no characterization on file", not raise."""
    assert latest_characterization(tmp_path / "does_not_exist_yet") is None


def test_latest_characterization_skips_malformed_file(tmp_path):
    """Ambiguity resolution #1: a malformed/unreadable JSON file in the
    directory must not crash the caller. Here it is also the NEWEST file by
    name, so this pins the documented behaviour -- skip it and fall through to
    the next-newest readable file -- rather than raising or silently stopping
    at the corrupted file and reporting None."""
    (tmp_path / "characterization-20260101.json").write_text(
        json.dumps({"dstar_mm": 111}))
    (tmp_path / "characterization-20260810.json").write_text("{not valid json")
    result = latest_characterization(tmp_path)
    assert result is not None and result["dstar_mm"] == 111
