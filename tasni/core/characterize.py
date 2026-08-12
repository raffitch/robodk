"""Distance-characterization metrics for selecting d* (spec §5, Phase 0).

Pure geometry, MILLIMETRES throughout, no hardware access. The CLI in
tools/characterize_distance.py (Task 16) feeds it captures; this module never
touches the camera or robot, at import time or otherwise.

d* is the CLOSEST camera-to-surface distance at which the whole measurement
chain (RealSense + hand-eye calibration + robot pose + processing) still
produces repeatable, accurate results. Depth error on a stereo camera grows
roughly with range squared, so closer is better *if* it still passes the
error budget — `choose_dstar` therefore returns the closest distance that
passes every criterion, not the best-scoring one.

Robustness: a non-finite (NaN/Inf) point anywhere in a capture must not
silently launder into a small, in-range-looking number — plain `<=`/`>=`
comparisons against NaN are always False, which would otherwise let a
contaminated trial sail through every upper-bound gate in `choose_dstar`.
Every metric here therefore either genuinely reflects the contamination (by
staying non-finite, since it is computed from the raw points) or is checked
for finiteness explicitly before being allowed to pass a gate.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from tasni.modules.scan.plane import fit_plane


@dataclass(frozen=True)
class DistanceTrial:
    distance_mm: float
    n_captures: int
    plane_rms_mm: float
    plane_max_mm: float
    height_repeat_mm: float
    normal_repeat_deg: float
    length_err_mm: float
    coverage_frac: float
    length_spread_mm: float

    def to_dict(self) -> dict:
        return asdict(self)


def plane_metrics(point_sets_mm) -> dict:
    """Per-capture plane fits -> repeatability metrics, in millimetres.

    ``plane_rms_mm`` / ``plane_max_mm``: mean/max of each capture's own RMS and
    max residual to its own fitted plane (a per-capture flatness/noise measure).

    ``height_repeat_mm``: how much the plane's HEIGHT varies capture-to-capture.
    Computed as the std of each capture's centroid projected onto the MEAN of
    all the per-capture normals (not each capture's own normal) — projecting
    onto a shared reference is what makes this a repeatability measure at all;
    projecting each centroid onto its own normal would trivially collapse
    toward the capture's own plane-fit residual regardless of any real height
    drift between captures.

    ``normal_repeat_deg``: the largest angle between any single capture's
    normal and the mean normal (worst-case orientation disagreement).

    Non-finite points inside a capture are NOT filtered out here: they flow
    straight into ``fit_plane`` (RANSAC + SVD refine), and a plane fit that
    ever touches a NaN/Inf coordinate produces a non-finite normal, centroid,
    or residual, which then contaminates the corresponding output the same
    way. That is deliberate — see the module docstring's robustness note.

    KNOWN CAVEAT (recorded, not fixed — Task 16 must not surface
    ``height_repeat_mm`` alone as a health number): if two captures' normals
    are exactly opposite (e.g. two near-vertical fits with genuinely opposite
    horizontal facings — ``fit_plane`` only flips a normal toward +Z when its
    Z component is *negative*, so two normals with Z==0 can come back on
    opposite sides), their mean has zero magnitude. The zero-norm guard below
    then leaves ``mean_n`` as the zero vector rather than dividing by zero, so
    every centroid projects to 0 and ``height_repeat_mm`` reports a
    misleadingly confident ``0.0`` even though the captures wildly disagree.
    ``normal_repeat_deg`` catches this correctly (it reports 90 degrees in
    that scenario: ``acos(abs(0)) == 90``), and ``choose_dstar`` always gates
    on both together, so selection stays safe — but a caller reading
    ``height_repeat_mm`` in isolation would be misled.
    """
    if not point_sets_mm:
        raise ValueError(
            "plane_metrics requires at least one capture (point_sets_mm was empty)")
    normals, centroids, rms, max_res = [], [], [], []
    for pts in point_sets_mm:
        p = np.asarray(pts, dtype=float).reshape(-1, 3)
        n, c, _ = fit_plane(p, distance=6.0)
        n = np.asarray(n, dtype=float)
        if n[2] < 0:
            n = -n
        res = (p - c) @ n
        normals.append(n)
        centroids.append(np.asarray(c, dtype=float))
        rms.append(float(np.sqrt(np.mean(res ** 2))))
        max_res.append(float(np.max(np.abs(res))))
    mean_n = np.mean(normals, axis=0)
    norm = np.linalg.norm(mean_n)
    mean_n = mean_n / norm if norm > 0 else mean_n
    # abs() folds a near-180-degree disagreement to near-0 degrees reported. This
    # domain is near-horizontal work surfaces at short standoff (table/board scans),
    # where every per-capture normal is already sign-oriented toward +Z by fit_plane
    # and real captures of the same surface sit within a tight cone of each other —
    # a genuine near-180-degree disagreement should not occur for a correctly aimed
    # capture, only for a degenerate/near-vertical fit (see the opposite-normals
    # caveat above, which abs() is also the direct cause of). It is safe here because
    # it only matters for that already-degenerate case, and normal_repeat_deg has a
    # companion path (via mean_n's own zero-magnitude guard) that still flags it.
    ang = [math.degrees(math.acos(min(1.0, max(-1.0, abs(float(n @ mean_n))))))
           for n in normals]
    heights_along_mean = [float(c @ mean_n) for c in centroids]
    return {
        "plane_rms_mm": float(np.mean(rms)),
        "plane_max_mm": float(np.max(max_res)),
        "height_repeat_mm": float(np.std(heights_along_mean)),
        "normal_repeat_deg": float(np.max(ang)),
    }


def known_length_error_mm(points_a_mm, points_b_mm, true_mm: float) -> float:
    """Abs error of the mean a->b point-pair distance against a known ``true_mm``
    length (e.g. a ruled/printed reference on the calibration board).

    This is a BIAS metric only: it reduces many repeat point-pair measurements
    to the error of their mean, so a symmetric scatter around the true length
    (e.g. half the repeats 10mm short, half 10mm long) reports a deceptive
    0.0mm error. See :func:`_length_spread_mm` for the companion repeatability
    metric that catches that scatter.
    """
    a = np.asarray(points_a_mm, dtype=float).reshape(-1, 3)
    b = np.asarray(points_b_mm, dtype=float).reshape(-1, 3)
    return abs(float(np.mean(np.linalg.norm(a - b, axis=1))) - float(true_mm))


def _length_spread_mm(points_a_mm, points_b_mm) -> float:
    """Std of the per-pair a->b distances — repeatability of a known-length
    measurement, independent of whether it is biased toward the true value.
    Companion to :func:`known_length_error_mm`, which only sees the bias."""
    a = np.asarray(points_a_mm, dtype=float).reshape(-1, 3)
    b = np.asarray(points_b_mm, dtype=float).reshape(-1, 3)
    return float(np.std(np.linalg.norm(a - b, axis=1)))


def summarize_distance_trial(distance_mm, plane_point_sets, length_samples,
                             coverage_frac) -> DistanceTrial:
    """Fold one distance's captures into a single :class:`DistanceTrial`.

    ``length_samples`` is ``[(points_a_mm, points_b_mm, true_mm), ...]``; the
    trial's ``length_err_mm`` and ``length_spread_mm`` are each the WORST (max)
    of their per-sample values, so a trial cannot pass by only reporting its
    best known-length measurement.
    """
    m = plane_metrics(plane_point_sets)
    errs = [known_length_error_mm(a, b, t) for a, b, t in length_samples] or [float("nan")]
    spreads = [_length_spread_mm(a, b) for a, b, _t in length_samples] or [float("nan")]
    return DistanceTrial(distance_mm=float(distance_mm), n_captures=len(plane_point_sets),
                         plane_rms_mm=m["plane_rms_mm"], plane_max_mm=m["plane_max_mm"],
                         height_repeat_mm=m["height_repeat_mm"],
                         normal_repeat_deg=m["normal_repeat_deg"],
                         length_err_mm=float(np.max(errs)),
                         coverage_frac=float(coverage_frac),
                         length_spread_mm=float(np.max(spreads)))


def choose_dstar(trials, *, max_rms_mm, max_plane_max_mm, max_height_repeat_mm,
                 max_normal_repeat_deg, max_length_err_mm, max_length_spread_mm,
                 min_coverage_frac) -> DistanceTrial | None:
    """The CLOSEST trial (smallest ``distance_mm``) that passes every criterion.

    This is not a "best score" search: depth quality improves monotonically as
    range decreases, so among the trials that clear the error budget the
    closest one is always the right pick, and a nearer trial that fails even
    one criterion is excluded outright rather than traded off against the
    others.

    ``max_plane_max_mm`` gates the WORST-CASE per-capture residual
    (``plane_max_mm``), separately from ``max_rms_mm``'s mean residual — a
    single bad point in an otherwise clean capture barely moves the mean but
    spikes the max, and without this gate such a trial could win purely on
    the strength of its (deceptively low) mean. ``max_length_spread_mm``
    gates the repeatability of the known-length check (``length_spread_mm``)
    separately from ``max_length_err_mm``'s bias-of-the-mean check, for the
    same reason: a symmetric scatter around the true length can average to a
    near-zero bias while still being poorly repeatable.

    A trial with any non-finite metric can never pass: plain ``<=``/``>=``
    comparisons against NaN are always False, which would make an all-NaN
    trial look like it satisfies every upper-bound gate (nothing is ever
    ``> max``) while still failing the ``>=`` lower-bound coverage gate only
    by luck. The explicit finiteness check makes that safe by construction
    instead of by accident.
    """
    def _passes(t: "DistanceTrial") -> bool:
        metrics = (t.plane_rms_mm, t.plane_max_mm, t.height_repeat_mm,
                  t.normal_repeat_deg, t.length_err_mm, t.length_spread_mm,
                  t.coverage_frac)
        if not all(math.isfinite(v) for v in metrics):
            return False
        return (t.plane_rms_mm <= max_rms_mm
                and t.plane_max_mm <= max_plane_max_mm
                and t.height_repeat_mm <= max_height_repeat_mm
                and t.normal_repeat_deg <= max_normal_repeat_deg
                and t.length_err_mm <= max_length_err_mm
                and t.length_spread_mm <= max_length_spread_mm
                and t.coverage_frac >= min_coverage_frac)

    passing = [t for t in trials if _passes(t)]
    return min(passing, key=lambda t: t.distance_mm) if passing else None
