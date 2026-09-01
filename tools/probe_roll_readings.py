"""THROWAWAY PROBE (2026-09-01): the two roll-pair readings that
``tools/probe_roll_pair.py`` does NOT perform.

That probe answers one question -- is the lifted-board HALO locked to the camera
-- by tracking the axis of a two-lobed distribution in the skirt annulus.
``docs/inspection-roll-probe-handoff.md`` section 4 identifies two further
questions it cannot answer, and this is their reader:

1. **The layer-2 dropout.** An ABSENCE, not a lifted shelf. Count ROI points per
   10 deg sector, find where they collapse, and express that sector twice: in
   work-frame angle and relative to that capture's own stereo baseline. Exactly
   one of the two should agree across captures.

2. **The static-noise decorrelation.** The question the whole multi-view case
   rests on, and the one neither angular statistic answers. Rasterise the
   substrate residual around the ring into polar cells, remove the fitted plane
   (already gone) and the radially symmetric component, then correlate the
   residual FIELDS -- within a pose, between the two same-roll control captures,
   and across rolls in both work and baseline coordinates.

Delete this file, and ``tests/test_probe_roll_readings.py``, once the answer is
in. They are a matched pair.

Usage::

    py -3.10 tools/probe_roll_readings.py <A1> <B> [<A2>]

with archived take directories from the A-B-A protocol (handoff section 3.2).
Two directories are accepted, but then there is no drift control and the probe
says so.

What it will NOT do
-------------------
Read a verdict off captures that do not share a noise floor. The comparability
gate is the same one ``probe_roll_pair.py`` enforces (substrate sigma within
25%, no floor pinned on its clamp), for the same reason: the 2026-08-31 pair
failed it, and the refusal is what stopped a confident wrong answer.

Frame conventions
-----------------
``baseline_deg`` is the camera's +X axis projected into the work plane -- the
D435i's depth/colour separation is essentially pure camera +X, so that is the
stereo baseline direction. Expressing a profile "relative to the baseline" means
rotating it by MINUS that angle, so a feature at work angle ``t`` lands at
``t - baseline``. Both readings share that convention and the tests pin it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tasni  # noqa: F401,E402  (pre-loads onnxruntime before robolink/Qt)
from tasni.core.config import ExtrusionConfig  # noqa: E402
from tasni.core.depth_geometry import CameraGeometry  # noqa: E402
from tasni.modules.extrusion.processing import (  # noqa: E402
    PlaneSubstrate, _deposit_clusters, _select_ring_cluster, depth_to_work_points,
    fit_circle_xy)

BIN_DEG = 10.0
NBINS = int(round(360.0 / BIN_DEG))

# Where each reading looks, as an offset from the FITTED RING RADIUS.
# The dropout lives on the ring itself; the residual field is read off the bare
# board OUTSIDE the bead's flank, which is what "the surrounding substrate" in
# the handoff means.
RING_BAND_MM = 10.0
SUBSTRATE_INNER_MM = 12.0
SUBSTRATE_OUTER_MM = 45.0

# A burst is SETTLED when its own first and last frames agree. Measured on the
# 2026-08-31 archive: the two repeat bursts scored +0.937 each, while the first
# burst at a fresh pose managed +0.045 -- the Jetson's temporal filter was still
# converging through it, so its residual field is transient filter state rather
# than the static structure this probe exists to measure. Nothing observed lands
# between those two populations; 0.5 separates them with room on both sides.
SETTLED_MIN_CORRELATION = 0.5


# ---------------------------------------------------------------------------
# the measured quantities (tested in tests/test_probe_roll_readings.py)
# ---------------------------------------------------------------------------
def sector_counts(points_xy, centre_xy, *, inner_mm: float, outer_mm: float,
                  bin_deg: float = BIN_DEG) -> np.ndarray:
    """Points per angular sector inside an annulus, in WORK-frame angle."""
    pts = np.asarray(points_xy, float)
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    nbins = int(round(360.0 / bin_deg))
    if not len(pts):
        return np.zeros(nbins)
    radius = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    band = (radius >= inner_mm) & (radius <= outer_mm)
    angle = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) % 360.0
    index = np.clip((angle / bin_deg).astype(int), 0, nbins - 1)
    return np.bincount(index[band], minlength=nbins).astype(float)


def best_circular_shift_deg(a, b, *, bin_deg: float = BIN_DEG) -> float:
    """How far ``b`` had to rotate to look like ``a``. Signed, -180 to +180.

    THE verdict statistic for the dropout, and deliberately not an axis.
    Measured against the real archive on 2026-09-01, the layer-2 deficit is
    multi-lobed -- a deep collapse at 140-190 deg plus lesser lows at 30, 240,
    260 and 330 -- so any "where is the dropout" summary averages them into a
    number that points at none of them. Registering the two WHOLE profiles
    sidesteps that: the question was never where the dropout is, only whether
    it moved with the camera.

    Returns the shift that best aligns the pair, so a scene-locked dropout
    registers near 0 and a camera-locked one near the change in baseline angle.
    """
    x = np.asarray(a, float)
    y = np.asarray(b, float)
    if x.shape != y.shape:
        raise ValueError("profiles must share a length to be registered")
    nbins = len(x)
    best_shift, best_corr = 0, -np.inf
    for shift in range(nbins):
        corr = correlate(np.roll(x, shift), y)
        if math.isfinite(corr) and corr > best_corr:
            best_corr, best_shift = corr, shift
    degrees = best_shift * bin_deg
    return degrees - 360.0 if degrees > 180.0 else degrees


def dropout_axis_deg(profile, *, bin_deg: float = BIN_DEG) -> float:
    """Circular mean of the deficit, 0-360. DESCRIPTIVE ONLY.

    Kept because it is a compact way to say "the missing weight sits over
    there", but it must not drive a verdict: on the real multi-lobed profile it
    returns 249 deg while the actual collapse is at 140-190. Use
    :func:`best_circular_shift_deg` to decide camera- versus scene-locked.
    """
    values = np.asarray(profile, float)
    if not np.isfinite(values).any():
        return float("nan")
    deficit = np.nanmax(values) - values
    deficit = np.where(np.isfinite(deficit), deficit, 0.0)
    if deficit.sum() <= 0:
        return float("nan")
    centres = np.radians((np.arange(len(values)) + 0.5) * bin_deg)
    return float(np.degrees(math.atan2(float((deficit * np.sin(centres)).sum()),
                                       float((deficit * np.cos(centres)).sum()))) % 360.0)


def rotate_to_baseline(profile, baseline_deg: float,
                       *, bin_deg: float = BIN_DEG) -> np.ndarray:
    """Re-express a per-sector profile against the capture's own baseline.

    Quantised to whole bins: at the default 10 deg that is up to 5 deg of
    rounding, which is far below the 60 deg the protocol rotates the baseline by.
    """
    shift = int(round(float(baseline_deg) / bin_deg))
    return np.roll(np.asarray(profile, float), -shift)


def residual_polar(points_xy, residual_mm, centre_xy, *, inner_mm: float,
                   outer_mm: float, r_bins: int, theta_bins: int) -> np.ndarray:
    """Mean residual per (radius, angle) cell. NaN where nothing was returned.

    Polar rather than Cartesian on purpose: expressing the field in a rolled
    camera's own frame is then an exact shift along one axis, with no
    interpolation to blur the very structure being measured.
    """
    pts = np.asarray(points_xy, float)
    values = np.asarray(residual_mm, float)
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    cells = int(r_bins) * int(theta_bins)
    if not len(pts):
        return np.full((r_bins, theta_bins), np.nan)
    radius = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    band = (radius >= inner_mm) & (radius <= outer_mm) & np.isfinite(values)
    angle = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) % 360.0
    span = max(float(outer_mm) - float(inner_mm), 1e-9)
    ri = np.clip(((radius - inner_mm) / span * r_bins).astype(int), 0, r_bins - 1)
    ti = np.clip((angle / 360.0 * theta_bins).astype(int), 0, theta_bins - 1)
    flat = ri * int(theta_bins) + ti
    total = np.bincount(flat[band], weights=values[band], minlength=cells)
    count = np.bincount(flat[band], minlength=cells)
    out = np.full(cells, np.nan)
    filled = count > 0
    out[filled] = total[filled] / count[filled]
    return out.reshape(int(r_bins), int(theta_bins))


def detrend_polar(grid) -> np.ndarray:
    """Remove the radially symmetric component of a polar field.

    A bowl, a residual tilt or a board warp that is the same at every angle
    carries NO angular information, so it cannot discriminate camera-locked from
    scene-locked -- but left in, it inflates every correlation toward 1. It goes.

    What deliberately does NOT go is low angular frequency. The stereo artifact
    this probe is hunting is itself a 2-cycle pattern (both ends of the
    baseline), so an angular high-pass would delete the signal along with the
    nuisance.
    """
    values = np.asarray(grid, float)
    finite = np.isfinite(values)
    count = finite.sum(axis=1, keepdims=True)
    total = np.where(finite, values, 0.0).sum(axis=1, keepdims=True)
    mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return values - mean


def rotate_polar(grid, baseline_deg: float) -> np.ndarray:
    """Express a polar field in the capture's own baseline coordinates."""
    values = np.asarray(grid, float)
    theta_bins = values.shape[1]
    shift = int(round(float(baseline_deg) / 360.0 * theta_bins))
    return np.roll(values, -shift, axis=1)


def correlate(a, b) -> float:
    """Pearson correlation over the cells BOTH fields define."""
    x = np.asarray(a, float).ravel()
    y = np.asarray(b, float).ravel()
    if x.shape != y.shape:
        raise ValueError("fields must share a shape to be correlated")
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3:
        return float("nan")
    x = x[ok] - x[ok].mean()
    y = y[ok] - y[ok].mean()
    denom = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if denom <= 0:
        return float("nan")
    return float(float((x * y).sum()) / denom)


def _signed_delta(a: float, b: float) -> float:
    """``b - a`` the short way round, -180 to +180."""
    return float(((float(b) - float(a) + 180.0) % 360.0) - 180.0)


def axial_gap(p: float, q: float, *, period: float = 360.0) -> float:
    """Smallest angular separation between two directions."""
    if not (math.isfinite(p) and math.isfinite(q)):
        return float("nan")
    d = abs((p - q) % period)
    return min(d, period - d)


# ---------------------------------------------------------------------------
# archive loading -- thin glue over the chain's own segmentation
# ---------------------------------------------------------------------------
def _provenance(take_dir: Path) -> tuple[dict, dict]:
    """Report + provenance, wherever this take kind happens to keep them.

    A characterization writes provenance into report.json; a layer measurement
    writes it into manifest.json. Both are read the same way here so the probe
    does not care which the operator captured.
    """
    report = json.loads((take_dir / "report.json").read_text(encoding="utf-8"))
    prov = report.get("provenance")
    if not prov:
        manifest_path = take_dir / "manifest.json"
        if manifest_path.is_file():
            prov = (json.loads(manifest_path.read_text(encoding="utf-8"))
                    .get("provenance"))
    if not prov:
        raise SystemExit(f"{take_dir}: no provenance in report.json or manifest.json")
    return report, prov


def load_take(take_dir: Path) -> dict:
    """One archived take, segmented the way the chain segments it.

    Deliberately stops BEFORE any layer-N assembly: the question is whether the
    camera returned points at a given angle at all, and running the deposit
    floor first would confound that with segmentation.
    """
    take_dir = Path(take_dir)
    report, prov = _provenance(take_dir)
    config = ExtrusionConfig.from_archive(prov["processing_config"])
    geom = CameraGeometry.from_greeting(prov["camera_geometry"])
    T = np.asarray(prov["T_work_camera"], float)
    depth = np.load(take_dir / "depth.npy")

    search = None
    for candidate in (report.get("search_center_mm"),
                      (report.get("coarse") or {}).get("center_mm"),
                      (report.get("metrics") or {}).get("measured_center_mm")):
        if candidate is not None:
            search = np.asarray(candidate, float)[:2]
            break
    if search is None:
        raise SystemExit(f"{take_dir}: no search centre, coarse fit or measured centre")

    points, _ = depth_to_work_points(depth, geom, T)
    radial = np.linalg.norm(points[:, :2] - search, axis=1)
    substrate = PlaneSubstrate.fit(
        points[radial <= config.substrate_fit_radius_mm],
        clamp_mm=tuple(config.substrate_floor_clamp_mm))
    height = substrate.height(points)
    floor = substrate.floor_mm(config.substrate_sigma_k)
    roi = ((height >= floor) & (height <= config.characterize_max_height_mm)
           & (radial <= config.characterize_search_radius_mm))

    clamp_hi = float(tuple(config.substrate_floor_clamp_mm)[1])
    baseline = np.asarray(T, float)[:3, 0]
    frames_path = take_dir / "depth-frames.npy"
    return {
        "dir": take_dir,
        "config": config,
        "geometry": geom,
        "T": T,
        "points": points,
        "height": height,
        "roi": roi,
        "floor": float(floor),
        "sigma": float(substrate.sigma_mm),
        "clamp_hi": clamp_hi,
        "floor_pinned": bool(abs(float(floor) - clamp_hi) < 1e-6),
        "search": search,
        "baseline_deg": float(np.degrees(math.atan2(baseline[1], baseline[0])) % 360.0),
        "frames": np.load(frames_path) if frames_path.is_file() else None,
        "valid": bool(report.get("valid")),
    }


def fit_ring(take: dict) -> tuple[np.ndarray, float]:
    """The chain's own coarse ring fit for this take."""
    counts: dict = {}
    clusters = _deposit_clusters(take["points"][take["roi"]], take["config"], counts)
    ring, _ = _select_ring_cluster(clusters, take["search"], counts)
    centre, radius = fit_circle_xy(ring)
    return np.asarray(centre, float)[:2], float(radius)


# ---------------------------------------------------------------------------
# CLI glue
# ---------------------------------------------------------------------------
def _comparable(takes: list[dict]) -> tuple[bool, str]:
    sigmas = [t["sigma"] for t in takes]
    ratio = max(sigmas) / max(min(sigmas), 1e-9)
    pinned = [tag for tag, t in zip("ABC", takes) if t["floor_pinned"]]
    detail = (f"substrate sigma " + ", ".join(f"{s:.3f}" for s in sigmas)
              + f" mm (ratio {ratio:.2f})")
    if pinned:
        detail += f"; floor PINNED on its {takes[0]['clamp_hi']:.1f} mm clamp in " \
                  + ", ".join(pinned)
    return (ratio <= 1.25 and not pinned), detail


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("takes", nargs="+",
                    help="archived take dirs in protocol order: A1 B [A2]")
    ap.add_argument("--r-bins", type=int, default=6)
    ap.add_argument("--theta-bins", type=int, default=360)
    args = ap.parse_args()

    if len(args.takes) < 2:
        raise SystemExit("need at least two takes (A1 B); three for a drift control")
    takes = [load_take(Path(p)) for p in args.takes]
    tags = ["A1", "B", "A2"][:len(takes)]

    # ONE shared centre and radius for every capture. Fitting each capture's own
    # ring would move the annulus being compared, so a biased fit in one arm
    # would masquerade as an angular difference -- the defect the 2026-09-01
    # review found in probe_roll_pair.py.
    shared_centre, shared_radius = fit_ring(takes[0])

    print("=" * 78)
    for tag, take in zip(tags, takes):
        own_centre, own_radius = fit_ring(take)
        drift = float(np.linalg.norm(own_centre - shared_centre))
        print(f"capture {tag}: {take['dir']}")
        print(f"  baseline {take['baseline_deg']:6.1f} deg   "
              f"substrate sigma {take['sigma']:.3f} mm   floor {take['floor']:.3f} mm"
              f"   {'VALID' if take['valid'] else 'aborted (raw frame is what we read)'}")
        print(f"  own ring fit: centre {own_centre.round(2).tolist()} r {own_radius:.2f} mm"
              f"   ({drift:.2f} mm from the shared reference)")
    print(f"  shared reference: centre {shared_centre.round(2).tolist()} "
          f"r {shared_radius:.2f} mm (from {tags[0]}, used for ALL captures)")

    ok, detail = _comparable(takes)
    print("=" * 78)
    print(f"comparability: {detail}")
    if not ok:
        print("VERDICT: NOT A CONTROLLED COMPARISON. The captures do not share a "
              "noise floor, so nothing below can be read as an answer. See the "
              "handoff's stop rule (section 3.2) before spending another excursion.")
        print("=" * 78)
        return 2

    r_in = shared_radius + SUBSTRATE_INNER_MM
    r_out = shared_radius + SUBSTRATE_OUTER_MM

    # ---- gate: was each burst SETTLED? -------------------------------------
    # Runs BEFORE the pair gate below: settling is a property of one capture, so
    # a capture that was never settled is worth naming even when the rolls also
    # failed to separate. Both are then on the same report.
    # Measured on the 2026-08-31 archive: a settled burst's first and last
    # frames correlate at +0.937, while the first burst at a fresh pose manages
    # +0.045 -- it does not even agree with ITSELF, because the Jetson's
    # temporal filter is still converging through it. An unsettled capture
    # cannot carry a static-structure verdict, so it is refused here rather
    # than quietly averaged into one.
    print("=" * 78)
    within = [_within_pose_correlation(t, shared_centre, r_in, r_out, args)
              for t in takes]
    for tag, value in zip(tags, within):
        print(f"  {tag}: within-pose frame-to-frame correlation "
              + ("n/a (no depth-frames.npy archived)" if value is None
                 else f"{value:+.3f}"))
    unsettled = [tag for tag, v in zip(tags, within)
                 if v is not None and v < SETTLED_MIN_CORRELATION]
    if unsettled:
        print(f"VERDICT: BURST NOT SETTLED in {', '.join(unsettled)} (below "
              f"{SETTLED_MIN_CORRELATION:+.2f}). The filter chain was still "
              "converging, so this capture's residual field is transient filter "
              "state, not static structure. Re-capture with more fusion frames "
              "and read the last five (handoff section 3.2).")
        print("=" * 78)
        return 2

    # ---- gate: did the rolls actually separate? ----------------------------
    baselines = [t["baseline_deg"] for t in takes]
    print("=" * 78)
    print(f"  baseline separation A1-B {axial_gap(baselines[0], baselines[1]):.1f} deg")
    if len(takes) == 3:
        print(f"  baseline return A1-A2    {axial_gap(baselines[0], baselines[2]):.1f} deg"
              "   (should be ~0: the control must come back to the same roll)")
    if axial_gap(baselines[0], baselines[1]) < 15.0:
        print("VERDICT: INCONCLUSIVE -- A1 and B are less than 15 deg apart, so a "
              "rolled pose probably fell through to roll 0.")
        print("=" * 78)
        return 2

    # ---- reading 1: the dropout ------------------------------------------
    inner = max(shared_radius - RING_BAND_MM, 0.0)
    outer = shared_radius + RING_BAND_MM
    profiles = [sector_counts(t["points"][t["roi"]][:, :2], shared_centre,
                              inner_mm=inner, outer_mm=outer) for t in takes]
    baseline_change = _signed_delta(baselines[0], baselines[1])
    print("=" * 78)
    print(f"DROPOUT -- ROI points per {BIN_DEG:.0f} deg sector, ring band "
          f"r {inner:.1f}-{outer:.1f} mm")
    for tag, profile in zip(tags, profiles):
        low = [int(i * BIN_DEG) for i in np.argsort(profile)[:4]]
        print(f"  {tag}: min {profile.min():6.0f}  max {profile.max():6.0f} pts/sector"
              f"   emptiest sectors {low} deg")
    shift = best_circular_shift_deg(profiles[0], profiles[1])
    print(f"  A1 -> B registers at {shift:+.0f} deg; the baseline moved "
          f"{baseline_change:+.0f} deg")
    if len(takes) == 3:
        control = best_circular_shift_deg(profiles[0], profiles[2])
        print(f"  A1 -> A2 (drift control) registers at {control:+.0f} deg "
              "(should be ~0)")
        if abs(control) > abs(baseline_change) / 2.0:
            print("  ** the control moved as far as the effect -- do NOT read a "
                  "verdict from this set **")
    camera_locked = abs(_signed_delta(shift, baseline_change)) < abs(shift)
    print("  -> " + ("CAMERA-LOCKED" if camera_locked else "SCENE-LOCKED")
          + " dropout, on this quantity alone")

    # ---- reading 2: the static residual field ------------------------------
    # Every point in the annulus, NOT just the sub-floor ones. The annulus
    # starts well outside the bead's flank, so there is no deposit in it to
    # exclude -- and cutting at the floor would censor exactly the above-floor
    # board the halo is made of, which is part of the field being measured.
    fields = [residual_polar(t["points"][:, :2], t["height"], shared_centre,
                             inner_mm=r_in, outer_mm=r_out,
                             r_bins=args.r_bins, theta_bins=args.theta_bins)
              for t in takes]
    work_fields = [detrend_polar(f) for f in fields]
    rel_fields = [detrend_polar(rotate_polar(f, t["baseline_deg"]))
                  for f, t in zip(fields, takes)]

    print("=" * 78)
    print(f"STATIC RESIDUAL FIELD -- board annulus r {r_in:.1f}-{r_out:.1f} mm, "
          f"{args.r_bins}x{args.theta_bins} polar cells")
    cross_work = correlate(work_fields[0], work_fields[1])
    cross_rel = correlate(rel_fields[0], rel_fields[1])
    print(f"  A1 vs B  in work coordinates      {cross_work:+.3f}")
    print(f"  A1 vs B  in baseline coordinates  {cross_rel:+.3f}")
    if len(takes) == 3:
        ceiling = correlate(work_fields[0], work_fields[2])
        print(f"  A1 vs A2 in work coordinates      {ceiling:+.3f}   "
              "(the repeatability ceiling: no cross-roll number can beat it)")
        if max(cross_work, cross_rel) > ceiling + 0.05:
            print("  ** a cross-roll correlation beat the same-roll ceiling, "
                  "which is impossible -- something is wrong with this set **")
    print("  -> " + ("CAMERA-LOCKED" if cross_rel > cross_work else "SCENE-LOCKED")
          + " static residual, i.e. it "
          + ("DOES" if cross_rel > cross_work else "does NOT")
          + " decorrelate with roll")
    print("=" * 78)
    print("Two independent quantities. Agreement between them is the result; a "
          "split verdict means neither is safe to build on.")
    print("=" * 78)
    return 0


def _within_pose_correlation(take: dict, centre, r_in: float, r_out: float,
                             args) -> float | None:
    """Correlation between the first and last frame of this take's own burst.

    The floor every cross-roll number is read against: structure that does not
    even repeat between two frames at ONE pose is not static structure.
    """
    frames = take.get("frames")
    if frames is None or len(frames) < 2:
        return None
    geom, T = take["geometry"], take["T"]
    grids = []
    for depth in (frames[0], frames[-1]):
        points, _ = depth_to_work_points(depth, geom, T)
        radial = np.linalg.norm(points[:, :2] - take["search"], axis=1)
        substrate = PlaneSubstrate.fit(
            points[radial <= take["config"].substrate_fit_radius_mm],
            clamp_mm=tuple(take["config"].substrate_floor_clamp_mm))
        grids.append(detrend_polar(residual_polar(
            points[:, :2], substrate.height(points), centre,
            inner_mm=r_in, outer_mm=r_out,
            r_bins=args.r_bins, theta_bins=args.theta_bins)))
    return correlate(grids[0], grids[1])


if __name__ == "__main__":
    raise SystemExit(main())
