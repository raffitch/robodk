"""THROWAWAY PROBE (2026-08-31): is the board halo locked to the camera's
stereo baseline, or is it real geometry on the board?

Background. The first live take under the geometry-only segmentation
(``runs/extrusion/20260831-163156-5bf38c80/characterize-01``) aborted with
``branch guard exhausted``. Diagnosed offline: a shelf of BOARD points -- HSV
saturation 16, indistinguishable from bare board at 19, against the ring's 104 --
sits 1.7-2.2 mm above the fitted plane (the cut is 1.77 mm), hugging the ring at
work-frame ~200 deg and ~353 deg. It is stable across all five fused frames, so
per-pixel median fusion cannot remove it, and it lands on the ring's LEFT and
RIGHT in the depth image -- the D435i's stereo baseline axis.

If that reading is right, the shelf is an artifact of WHERE THE CAMERA WAS, and
rolling the camera about its own optical axis must carry the shelf around with
it. If instead the shelf is something real on the board, it must stay put in
work coordinates no matter how the camera is rolled.

That is the whole question, and one rolled capture answers it. This script does
not fix anything and is not part of the chain; delete it once the answer is in.

Protocol
--------
Two TOP-DOWN characterizations of the SAME, UNTOUCHED ring, differing only in
``extrusion.inspection_roll_candidates_deg`` (tilt stays 0, so the plane noise
floor stays ~0.65 mm rather than the 3-4 mm a 15 deg tilt would cost):

  capture A   roll 0    -- the 16:31 take already on disk qualifies
  capture B   roll 90   -- set inspection_roll_candidates_deg [90, 60, 0],
                           restart the backend, run characterize again

Capture B is EXPECTED to abort with the branch guard as well. That is fine: the
raw RGB-D is archived on failure, which is all this probe reads.

Usage::

    py -3.10 tools/probe_roll_pair.py <take_dir_A> <take_dir_B>
    py -3.10 tools/probe_roll_pair.py A B --figure probe_roll.png

Reading the output. For each capture the script measures where the above-floor
board sits in the skirt annulus just outside the ring, and reports that
distribution twice: in WORK-frame angle, and in angle relative to that capture's
own camera baseline. Exactly one of the two should agree between captures.

  baseline-relative peaks agree, work-frame peaks rotate  -> BASELINE-LOCKED
      the shelf is a stereo edge artifact; a two-frame agreement test removes it
      at zero tilt cost, and the star's 15 deg tilt is the wrong tool for it.

  work-frame peaks agree, baseline-relative peaks rotate  -> REAL GEOMETRY
      something is actually on the board there; extra views will not help and
      the segmentation has to deal with it directly.
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
# The skirt: outside the bead's own flank, inside where the shelf was measured
# (the 2026-08-31 shelf ran to ~10 mm past the ring's outer edge).
SKIRT_INNER_MM = 5.0
SKIRT_OUTER_MM = 18.0


def _wrap180(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def _circular_peak_deg(weights: np.ndarray) -> float:
    """Weighted circular mean of a 36-bin histogram, in degrees."""
    centres = np.radians((np.arange(len(weights)) + 0.5) * BIN_DEG)
    if not weights.sum():
        return float("nan")
    return float(np.degrees(math.atan2(float((weights * np.sin(centres)).sum()),
                                       float((weights * np.cos(centres)).sum()))) % 360.0)


def _bimodal_axis_deg(weights: np.ndarray) -> float:
    """Axis of a two-lobed distribution, 0-180.

    The shelf appears at BOTH ends of the baseline (left and right of the ring),
    so its work-frame circular mean is near-degenerate -- the two lobes cancel.
    Doubling the angle folds the two lobes onto one before averaging, which is
    the standard axial statistic and the only one that can track this shape.
    """
    centres = np.radians((np.arange(len(weights)) + 0.5) * BIN_DEG)
    if not weights.sum():
        return float("nan")
    ang = math.atan2(float((weights * np.sin(2 * centres)).sum()),
                     float((weights * np.cos(2 * centres)).sum()))
    return float((np.degrees(ang) / 2.0) % 180.0)


def load_take(take_dir: Path) -> dict:
    """Everything this probe needs from one archived take, failed or not."""
    take_dir = Path(take_dir)
    report = json.loads((take_dir / "report.json").read_text(encoding="utf-8"))
    prov = report.get("provenance") or {}
    config = ExtrusionConfig.from_archive(prov["processing_config"])
    geom = CameraGeometry.from_greeting(prov["camera_geometry"])
    T = np.asarray(prov["T_work_camera"], float)
    depth = np.load(take_dir / "depth.npy")
    search_c = np.asarray(report["search_center_mm"], float)

    points, _ = depth_to_work_points(depth, geom, T)
    radial = np.linalg.norm(points[:, :2] - search_c, axis=1)
    substrate = PlaneSubstrate.fit(
        points[radial <= config.substrate_fit_radius_mm],
        clamp_mm=tuple(config.substrate_floor_clamp_mm))
    height = substrate.height(points)
    floor = substrate.floor_mm(config.substrate_sigma_k)

    # Find the ring exactly the way characterize_ring's coarse pass does, so the
    # centre this probe measures angles from is the chain's own centre.
    roi = ((height >= floor) & (height <= config.characterize_max_height_mm)
           & (radial <= config.characterize_search_radius_mm))
    counts: dict = {}
    clusters = _deposit_clusters(points[roi], config, counts)
    ring, _ = _select_ring_cluster(clusters, search_c, counts)
    centre, radius = fit_circle_xy(ring)

    # The camera's baseline is its own +X axis (depth and colour sensors are
    # separated along it: the archived depth_to_color translation is
    # [15.05, 0.07, 0.49] mm, i.e. essentially pure X). Project it into the work
    # plane -- the view is top-down, so that projection is well conditioned.
    baseline_work = np.asarray(T, float)[:3, 0]
    baseline_deg = float(np.degrees(math.atan2(baseline_work[1], baseline_work[0])) % 180.0)

    pose = report.get("inspection_pose") or {}
    return {
        "dir": take_dir,
        "roll_deg": float(pose.get("roll_deg", float("nan"))),
        "tilt_deg": float(pose.get("tilt_deg", float("nan"))),
        "valid": bool(report.get("valid")),
        "points": points,
        "height": height,
        "floor": float(floor),
        "sigma": float(substrate.sigma_mm),
        "centre": np.asarray(centre, float),
        "radius": float(radius),
        "baseline_deg": baseline_deg,
    }


def skirt_histogram(take: dict) -> dict:
    """Above-floor board in the annulus just outside the ring, by angle."""
    pts, h = take["points"], take["height"]
    cx, cy = take["centre"]
    r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    inner = take["radius"] + SKIRT_INNER_MM
    outer = take["radius"] + SKIRT_OUTER_MM
    band = (r >= inner) & (r <= outer)
    over = band & (h >= take["floor"])

    ang = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) % 360.0
    idx = np.clip((ang / BIN_DEG).astype(int), 0, NBINS - 1)
    total = np.bincount(idx[band], minlength=NBINS).astype(float)
    lifted = np.bincount(idx[over], minlength=NBINS).astype(float)
    # The FRACTION lifted, not the raw count: bins differ in how much surface the
    # camera actually returned, and a dropout-heavy bin would otherwise read as
    # clean simply for being empty.
    fraction = np.where(total > 0, lifted / np.maximum(total, 1), np.nan)
    return {
        "inner_mm": inner, "outer_mm": outer,
        "total": total, "lifted": lifted,
        "fraction": fraction,
        "band_points": int(band.sum()), "lifted_points": int(over.sum()),
        "lifted_fraction": float(over.sum() / band.sum()) if band.sum() else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take_a", help="archived take directory, capture A (e.g. roll 0)")
    ap.add_argument("take_b", help="archived take directory, capture B (e.g. roll 90)")
    ap.add_argument("--figure", default=None,
                    help="write a polar comparison figure to this path")
    args = ap.parse_args()

    takes = []
    for path in (args.take_a, args.take_b):
        take = load_take(Path(path))
        take["skirt"] = skirt_histogram(take)
        takes.append(take)

    print("=" * 78)
    for tag, take in zip("AB", takes):
        s = take["skirt"]
        print(f"capture {tag}: {take['dir']}")
        print(f"  commanded roll {take['roll_deg']:.1f} deg, tilt {take['tilt_deg']:.1f} deg"
              f"   run {'VALID' if take['valid'] else 'aborted (fine -- raw frame is what we read)'}")
        print(f"  ring centre {take['centre'].round(2).tolist()}  radius {take['radius']:.2f} mm")
        print(f"  substrate sigma {take['sigma']:.3f} mm, floor {take['floor']:.3f} mm")
        print(f"  camera baseline in the work frame: {take['baseline_deg']:.1f} deg (axis, 0-180)")
        print(f"  skirt annulus r {s['inner_mm']:.1f}-{s['outer_mm']:.1f} mm: "
              f"{s['lifted_points']} of {s['band_points']} points over the floor "
              f"({100 * s['lifted_fraction']:.1f}%)")

    weights = [np.nan_to_num(t["skirt"]["fraction"]) for t in takes]
    work_axis = [_bimodal_axis_deg(w) for w in weights]
    # Re-express each capture's lifted board relative to ITS OWN baseline, then
    # ask whether the two agree.
    rel_axis = [(work_axis[i] - takes[i]["baseline_deg"]) % 180.0 for i in range(2)]

    def axial_gap(p, q):
        d = abs((p - q) % 180.0)
        return min(d, 180.0 - d)

    work_gap = axial_gap(*work_axis)
    rel_gap = axial_gap(*rel_axis)
    baseline_gap = axial_gap(takes[0]["baseline_deg"], takes[1]["baseline_deg"])

    print("=" * 78)
    print("where the lifted board sits (axis of the two lobes, 0-180 deg)")
    print(f"  capture A   work frame {work_axis[0]:6.1f}    "
          f"relative to its baseline {rel_axis[0]:6.1f}")
    print(f"  capture B   work frame {work_axis[1]:6.1f}    "
          f"relative to its baseline {rel_axis[1]:6.1f}")
    print(f"  the two cameras' baselines differ by {baseline_gap:.1f} deg")
    print(f"  work-frame disagreement       {work_gap:6.1f} deg")
    print(f"  baseline-relative disagreement{rel_gap:6.1f} deg")

    print("=" * 78)
    if baseline_gap < 15.0:
        verdict = ("INCONCLUSIVE: the two captures' baselines are only "
                   f"{baseline_gap:.1f} deg apart. Check `rejected` in capture B's "
                   "report -- the rolled pose was probably refused and the run fell "
                   "back to roll 0. Lower max_tool_axis_spin_deg's obstacle or try 60 deg.")
    elif rel_gap < work_gap:
        verdict = ("BASELINE-LOCKED. The lifted board followed the camera. It is a "
                   "stereo edge artifact, so a two-frame agreement test at zero tilt "
                   "removes it -- and the star's 15 deg tilt is the wrong tool, since "
                   "it would add plane noise to fight a problem that needs none.")
    else:
        verdict = ("REAL GEOMETRY. The lifted board stayed put while the camera "
                   "rolled, so something is actually there. Extra views will not "
                   "remove it; the segmentation has to reject it on its own merits.")
    print("VERDICT: " + verdict)
    print("=" * 78)

    if args.figure:
        _draw(takes, Path(args.figure), work_axis, rel_axis)
    return 0


def _draw(takes, out: Path, work_axis, rel_axis) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), facecolor="white",
                            subplot_kw={"projection": "polar"})
    theta = np.radians((np.arange(NBINS) + 0.5) * BIN_DEG)
    width = np.radians(BIN_DEG) * 0.92
    colours = ("#0d8ba3", "#c1440e")
    for ax, mode in zip(axes, ("work", "baseline")):
        for i, (take, colour) in enumerate(zip(takes, colours)):
            frac = np.nan_to_num(take["skirt"]["fraction"])
            shift = 0.0 if mode == "work" else -np.radians(take["baseline_deg"])
            ax.bar(theta + shift, frac, width=width, bottom=0.0, alpha=.55,
                   color=colour, edgecolor=colour, linewidth=.6,
                   label=f"capture {'AB'[i]} — roll {take['roll_deg']:.0f}°")
        axis = work_axis if mode == "work" else rel_axis
        for i, colour in enumerate(colours):
            for sign in (0.0, 180.0):
                ax.axvline(np.radians(axis[i] + sign), color=colour, lw=1.6, ls="--", alpha=.9)
        ax.set_theta_zero_location("E")
        ax.set_title("angle in the WORK frame" if mode == "work"
                     else "angle relative to each camera's own baseline",
                     fontsize=12, pad=16)
        ax.set_rlabel_position(112)
        ax.grid(alpha=.3)
    axes[0].legend(loc="upper left", bbox_to_anchor=(-.16, 1.12), fontsize=10)
    fig.suptitle("Fraction of the skirt annulus lifted over the floor\n"
                 "dashed lines = the axis of the two lobes; "
                 "the panel where the two captures AGREE is the answer",
                 fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, .88))
    fig.savefig(out, dpi=130, facecolor="white")
    print(f"figure written to {out}")


if __name__ == "__main__":
    raise SystemExit(main())
