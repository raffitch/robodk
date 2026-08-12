import numpy as np
import pytest

from tasni.core.characterize import (
    choose_dstar, known_length_error_mm, plane_metrics, summarize_distance_trial,
)


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


def test_choose_dstar_prefers_closest_passing_distance():
    trials = [_trial(300.0, rms=1.5), _trial(400.0), _trial(600.0)]
    best = choose_dstar(trials, max_rms_mm=1.0, max_height_repeat_mm=1.0,
                        max_normal_repeat_deg=1.0, max_length_err_mm=1.0,
                        min_coverage_frac=0.5)
    assert best is not None and best.distance_mm == 400.0
    none = choose_dstar([_trial(300.0, rms=2.0)], max_rms_mm=1.0,
                        max_height_repeat_mm=1.0, max_normal_repeat_deg=1.0,
                        max_length_err_mm=1.0, min_coverage_frac=0.5)
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
    contaminated = _trial(300.0)
    contaminated = summarize_distance_trial(
        300.0,
        [_plane_set(z=300.0, seed=i) for i in range(3)],
        [(np.array([[np.nan, 0.0, 300.0]]), np.array([[297.0, 0.0, 300.0]]), 297.0)],
        0.9,
    )
    healthy = _trial(400.0)
    best = choose_dstar([contaminated, healthy], max_rms_mm=1.0,
                        max_height_repeat_mm=1.0, max_normal_repeat_deg=1.0,
                        max_length_err_mm=1.0, min_coverage_frac=0.5)
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
                        min_coverage_frac=0.5)
    assert none is None


def test_plane_metrics_height_repeat_uses_mean_normal_not_per_capture_normal():
    """Ambiguity resolution #4: height repeatability must project each capture's
    centroid onto the MEAN normal, not its own per-capture normal. Projecting a
    point onto its own plane's normal trivially gives (approximately) the same
    signed distance regardless of a rigid shift along that normal only if all
    normals agree; here we tilt each capture's sampled patch slightly differently
    (small in-plane rotation of noise does not change the normal materially, so
    instead we directly check that a height offset the mean-normal projection can
    see (different z-planes with identical, EXACTLY parallel normals) is reported
    non-zero, and that this is consistent with a hand-computed mean-normal
    projection.)"""
    sets = [_plane_set(z=400.0 + dz, sigma=0.01, seed=i)
            for i, dz in enumerate((0.0, 1.0, -1.0))]
    m = plane_metrics(sets)
    # Hand-compute expected: each capture is (near) exactly flat at its own z, so its
    # centroid is ~ (0, 0, z) and the mean normal is ~ (0, 0, 1). Height repeatability
    # should be very close to std([400.0, 401.0, 399.0]).
    expected = float(np.std([400.0, 401.0, 399.0]))
    assert m["height_repeat_mm"] == pytest.approx(expected, abs=0.05)
