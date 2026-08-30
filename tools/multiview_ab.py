"""multiview_ab.py -- offline A/B: single-view vs star reconstruction of the
SAME archived take.

Every star take was already captured once, with every view's raw color/depth
archived under ``layer-*/views/`` (spec section 5.6, Task 7). That means the
comparison this script prints costs no arm time at all: it walks a measure-only
trial, and for every take that holds a ``views/`` directory it reprocesses that
ONE capture BOTH ways --

  * ``as_archived`` -- the star reconstruction (every surviving view merged), and
  * ``top_only``    -- the control arm: the identical physical ring placement,
                        rebuilt from the top view alone, as if it had only ever
                        been a single-view take.

-- via ``reprocess_saved_layer(..., views=...)``, and prints one row per take
comparing them. Because both reconstructions come from the SAME physical ring
placement, the comparison is paired -- something the operator cannot reproduce
by hand-placing a ring twice.

NOTE: the chain voxel-downsamples at 1 mm. Merging four views does NOT multiply
the surviving point count -- it multiplies the samples each voxel averages and
fills dropouts. Read after_work_roi (pre-voxel), never after_voxel.

Usage::

    py -3.10 tools/multiview_ab.py <trial_id>
    py -3.10 tools/multiview_ab.py <trial_id> --root runs/extrusion
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WARNING = (
    "NOTE: the chain voxel-downsamples at 1 mm. Merging four views does NOT multiply\n"
    "the surviving point count -- it multiplies the samples each voxel averages and\n"
    "fills dropouts. Read after_work_roi (pre-voxel), never after_voxel."
)


def _star_takes(root: Path, trial_id: str) -> list[tuple[int, int]]:
    """(layer_index, take) for every archived take that actually holds a
    views/ directory -- the only takes this A/B has anything to say about."""
    trial_dir = Path(root) / trial_id
    found = []
    for manifest_path in sorted(trial_dir.glob("layer-*/manifest.json")):
        if not (manifest_path.parent / "views").is_dir():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        found.append((int(manifest["layer_index"]), int(manifest.get("take", 1))))
    return found


def _one_row(root, trial_id: str, layer_index: int, take: int) -> dict:
    """Both reconstructions of one take. Never raises: a take that fails to
    reprocess one way (a branch guard exhausted, say) must not silence the
    report for every OTHER take in the trial.

    Both calls pass persist=False -- this is an ANALYSIS tool, not a rewrite
    tool, and reprocess_saved_layer(persist=True) (the default, used by the
    app's own reprocess endpoint) would overwrite manifest.json (and, for a
    session-backed trial, session.json too) with whichever reconstruction ran
    last. Since this function always runs "as_archived" then "top_only" on
    the SAME on-disk take, persist=True here would leave every star take this
    tool examined archived as capture.style == "single" -- silently undoing
    Task 8's whole point the next time paper_summary() reads that trial.
    """
    from tasni.modules.extrusion.service import reprocess_saved_layer
    row = {"layer_index": layer_index, "take": take, "star": None,
          "single": None, "error": None}
    try:
        row["star"] = reprocess_saved_layer(root, trial_id, layer_index, take,
                                            views="as_archived", persist=False)
        row["single"] = reprocess_saved_layer(root, trial_id, layer_index, take,
                                              views="top_only", persist=False)
    except Exception as exc:                                     # noqa: BLE001
        row["error"] = str(exc)
    return row


def _num(value, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _range(geometry: dict, lo_key: str, hi_key: str, digits: int = 1) -> str:
    lo, hi = geometry.get(lo_key), geometry.get(hi_key)
    return "-" if lo is None or hi is None else f"{lo:.{digits}f}-{hi:.{digits}f}"


_COLUMNS = ("layer", "take", "bead single", "bead star", "height single",
           "height star", "complete single", "complete star", "max gap single",
           "max gap star", "roi single (pre-voxel)", "roi star (pre-voxel)",
           "spread_before_mm", "residual_after_mm")


def build_report(root, trial_id: str) -> list[str]:
    """The whole printable report as a list of lines -- no I/O, so it is what
    the tests exercise directly."""
    from tasni.modules.extrusion.measure import centre_spread

    lines = [WARNING, ""]
    takes = _star_takes(Path(root), trial_id)
    if not takes:
        lines.append(f"no star takes found under {trial_id!r} "
                     "(no layer-*/views/ directory in this trial)")
        return lines
    rows = [_one_row(root, trial_id, layer_index, take) for layer_index, take in takes]
    lines.append(f"trial {trial_id}: {len(rows)} star take(s), "
                 "each reprocessed as_archived (star) and top_only (single) "
                 "from the SAME raw capture")
    lines.append(" | ".join(_COLUMNS))
    for row in rows:
        if row["error"]:
            lines.append(f"{row['layer_index']:>5} {row['take']:>4} : "
                         f"reprocess failed -- {row['error']}")
            continue
        star, single = row["star"], row["single"]
        star_geom, single_geom = star.get("geometry") or {}, single.get("geometry") or {}
        # Read from the reconstruction reprocess_saved_layer just returned, not
        # from report.json on disk -- persist=False deliberately never writes
        # it, and it would otherwise still hold whatever the archive's LAST
        # persisted reprocess wrote (stale, and possibly the OTHER style's).
        star_counts = ((star.get("processing") or {}).get("counts") or {})
        single_counts = ((single.get("processing") or {}).get("counts") or {})
        capture = star.get("capture") or {}
        lines.append(" | ".join((
            str(row["layer_index"]), str(row["take"]),
            _num(single_geom.get("bead_width_mean_mm")),
            _num(star_geom.get("bead_width_mean_mm")),
            _range(single_geom, "height_min_mm", "height_max_mm"),
            _range(star_geom, "height_min_mm", "height_max_mm"),
            f"{single['metrics']['path_completeness']:.0%}",
            f"{star['metrics']['path_completeness']:.0%}",
            _num(single["metrics"]["maximum_angular_gap_deg"]),
            _num(star["metrics"]["maximum_angular_gap_deg"]),
            str(single_counts.get("after_work_roi", "-")),
            str(star_counts.get("after_work_roi", "-")),
            _num(capture.get("spread_before_mm"), 2),
            _num(capture.get("residual_after_mm"), 2))))
    ok = [r for r in rows if not r["error"]]
    if len(ok) > 1:
        single_manifests = [{"metrics": r["single"]["metrics"]} for r in ok]
        star_manifests = [{"metrics": r["star"]["metrics"]} for r in ok]
        single_spread = centre_spread(single_manifests)
        star_spread = centre_spread(star_manifests)
        lines.append("")
        lines.append(f"centre spread across {len(ok)} repeats -- "
                     f"single: {_num(single_spread['rms_mm'], 2)} mm RMS "
                     f"(max {_num(single_spread['max_mm'], 2)} mm); "
                     f"star: {_num(star_spread['rms_mm'], 2)} mm RMS "
                     f"(max {_num(star_spread['max_mm'], 2)} mm)")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Offline A/B: single-view vs star reconstruction of the same "
                    "archived take (no robot; reprocesses the raw RGB-D already "
                    "on disk).")
    ap.add_argument("trial_id", help="measure-only trial id under the archive root")
    ap.add_argument("--root", default=None,
                    help="archive root (default: runs/extrusion under the repo)")
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else (ROOT / "runs" / "extrusion")
    for line in build_report(root, args.trial_id):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
