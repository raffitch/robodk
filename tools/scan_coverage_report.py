"""scan_coverage_report.py — measure how well a saved scan covered the work surface.

Reads a run under runs/scan/<stamp>/ and reports, in the work-plane's own frame:
  * the fitted rectangle size vs the DENSE board (2/98 percentile of points),
  * an occupancy grid (default 8 mm) — interior fill vs each edge band + the 4
    corners, which is what exposes a one-sided capture hole (e.g. an unseen +X
    edge that no single view framed).

This is the exact diagnostic that revealed the +X-corner hole in run
20260625-171346. Pure numpy; reads the run's preview.npz (flattened plane-inlier
points, mm) + report.json. No RoboDK / camera.

    py -3.10 tools/scan_coverage_report.py                 # newest run
    py -3.10 tools/scan_coverage_report.py 20260625-171346 # a specific run
    py -3.10 tools/scan_coverage_report.py --bin 6         # finer grid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "scan"


def _newest_run() -> str:
    runs = [p.name for p in RUNS.iterdir()
            if p.is_dir() and (p / "report.json").is_file()]
    if not runs:
        sys.exit(f"no scan runs with a report.json under {RUNS}")
    return sorted(runs)[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stamp", nargs="?", help="run stamp (default: newest)")
    ap.add_argument("--bin", type=float, default=8.0, help="occupancy cell mm (default 8)")
    ap.add_argument("--band", type=int, default=3, help="edge band width in cells (default 3)")
    args = ap.parse_args()

    stamp = args.stamp or _newest_run()
    run = RUNS / stamp
    rep = json.loads((run / "report.json").read_text())
    plane = rep["plane"]
    fT = np.asarray(plane["frame_T_mm"], float)
    ax_x, ax_y, origin = fT[:3, 0], fT[:3, 1], fT[:3, 3]

    npz = run / "preview.npz"
    if not npz.is_file():
        sys.exit(f"{npz} missing — this run saved no preview cloud")
    pts = np.asarray(np.load(npz)["points_mm"], float)
    px = (pts - origin) @ ax_x
    py = (pts - origin) @ ax_y

    sz = plane["size_mm"]
    tilt = float(np.degrees(np.arccos(np.clip(plane["normal"][2], -1, 1))))
    print(f"run {stamp}: {rep['n_views']} views, {len(pts)} surface pts, "
          f"inliers {plane['inlier_frac']:.0%}, tilt {tilt:.1f}°")
    print(f"fitted rectangle : {sz[0]:.1f} × {sz[1]:.1f} mm")
    for q in (0.5, 2.0, 5.0):
        xlo, xhi = np.percentile(px, [q, 100 - q])
        ylo, yhi = np.percentile(py, [q, 100 - q])
        print(f"  dense board {q:>4.1f}/{100-q:.1f}% : "
              f"{xhi-xlo:6.1f} × {yhi-ylo:6.1f} mm")

    # Occupancy grid in the plane frame.
    B = float(args.bin)
    ix = np.floor((px - px.min()) / B).astype(int)
    iy = np.floor((py - py.min()) / B).astype(int)
    gx, gy = int(ix.max()) + 1, int(iy.max()) + 1
    occ = np.zeros((gx, gy), int)
    np.add.at(occ, (ix, iy), 1)
    g = occ > 0
    b = max(1, int(args.band))
    print(f"\noccupancy {gx}×{gy} @ {B:.0f} mm — filled {g.mean():.0%} "
          f"(edge band = {b} cells / {b*B:.0f} mm):")
    print(f"  interior   {g[b:-b, b:-b].mean():.0%}")
    print(f"  X- edge    {g[:b, :].mean():.0%}    X+ edge {g[-b:, :].mean():.0%}")
    print(f"  Y- edge    {g[:, :b].mean():.0%}    Y+ edge {g[:, -b:].mean():.0%}")
    print(f"  corners    -X-Y {g[:b, :b].mean():.0%}  -X+Y {g[:b, -b:].mean():.0%}  "
          f"+X-Y {g[-b:, :b].mean():.0%}  +X+Y {g[-b:, -b:].mean():.0%}")
    worst = min(g[:b, :].mean(), g[-b:, :].mean(),
                g[:, :b].mean(), g[:, -b:].mean())
    if worst < 0.6:
        print(f"\n>> COVERAGE HOLE: weakest edge fill {worst:.0%} (< 60%) — a board "
              f"side was under-captured. Re-seed more central / widen the cone / "
              f"check predicted surface coverage on Create targets.")
    else:
        print(f"\n>> edges all >= {worst:.0%} filled — coverage looks balanced.")

    views = run / "views"
    if views.is_dir():
        n = len(list(views.glob("depth_*.png")))
        print(f"\n(per-view frames present: {n} under views/ — a camera-perspective "
              f"overlay can be built from views.json + the depth PNGs)")


if __name__ == "__main__":
    main()
