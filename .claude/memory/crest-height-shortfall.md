---
name: crest-height-shortfall
description: "The camera reads ring crest height ~1.5 mm LOW (r=0.97 on shape, verified against the operator's ruler); the spatial filter's smooth_delta=20 vs a ring spanning <4 disparity px is the prime suspect"
metadata:
  node_type: memory
  type: project
---

**First independent height reference we have.** 2026-08-31 the operator ruled five clock
positions on the characterization ring. Compared against
`runs/extrusion/20260831-173544-24d21bab/characterize-01`:

| clock | ruler | camera p95 | p99 | max | % over floor |
|---|---|---|---|---|---|
| 12h | 11.0 | 7.82 | 9.32 | 9.51 | 28% |
| 3h | 9.0 | 6.02 | 6.79 | 7.03 | 49% |
| 9h | 7.0 | 5.56 | 5.88 | 6.10 | 49% |
| 6h | 7.0 | 4.21 | 4.47 | 4.63 | 40% |
| 11h | 4.0 | 2.35 | 2.84 | 3.09 | **13%** |

`camera_max = 0.905 x ruler - 0.80 mm`, **r = 0.97**, mean shortfall **1.53 mm**. SHAPE is
excellent (every position in the right order); ABSOLUTE height is biased low. Not a choice
of statistic — that is each wedge's single highest point. Some may be the ruler reading to
the topmost fibres of a fuzzy rope, but not 1.5 mm of it.

**Why it matters:** the deposit floor is 1.78 mm and the board's own p99 is 1.648 — 0.13 mm
of headroom. Losing 1.5 mm of crest BEFORE the floor is applied is what made a continuous
ring report `completeness 0.875, max angular gap 44.9 deg`. The thin arc is a true 4 mm,
arrives as 2.8, and 13% of its band clears. **The ring was NOT open** — operator confirmed:
upper-left is thin but continuous, upper-RIGHT is the lifted part (12h, the tallest).

**Prime suspect:** `rs.spatial_filter()` runs in the DISPARITY domain with stock defaults,
`smooth_delta 20` — the edge-preservation threshold. At 300 mm with depth fx 637 and the
D435i's ~50 mm nominal baseline, 1 mm of relief is ~0.35 disparity px, so a 4-11 mm ring
spans 1.4-3.9. Under a fifth of the threshold: the filter cannot tell the rope from the
board and smooths across it twice at alpha 0.5. UNVERIFIED — needs the A/B.

**How to apply:**
- Levers shipped + deployed + live-verified 2026-08-31 (`f9c4a53`, Jetson on main, greeting
  re-read from the real camera: still the stock chain). `RS_SPATIAL=0` drops the filter
  (control arm); `RS_SPATIAL_SMOOTH_DELTA` lowers the threshold. **Both default to current
  behaviour** — every archived number was measured under the stock filter.
- Flip an arm with a systemd DROP-IN (`/etc/systemd/system/realsense-camera.service.d/`),
  NOT by editing `server/realsense-camera.service` — that file is version-controlled and
  `jetson_deploy bootstrap` reinstalls it, so an experiment committed there becomes the
  default. Remember [[jetson-wifi-network-ops]]: sudo over SSH needs `sudo -S` via stdin.
- The greeting's filter list USED to be hardcoded; it now derives from the same constants
  the chain is built from. That list is the ONLY record of which arm a take came from.
- A/B protocol: same UNTOUCHED ring, arm A = the 17:35 take (stock), arm B = `RS_SPATIAL=0`,
  optional arm C = `smooth_delta 4`. Compare crest per clock against the ruler above.

**A/B RESULT 2026-08-31 (arm A stock vs arm B spatial OFF, same untouched ring):**
crest raw-cloud p99 7.088 -> 8.494 mm, max 9.697 -> 10.576; mean shortfall vs ruler
1.53 -> 1.05 mm; 4 of 5 clocks gained ~0.9 mm, 3h went the OTHER way by 0.94 (unexplained);
substrate sigma 0.593 -> 0.624 (+5%, the positive control, weak). So the filter carries
roughly a THIRD of the deficit. Camera has been reverted to stock and live-verified.

**THE RING-OPEN FAILURE WAS NOT THE CAMERA — it was the 1 mm voxel.** Peer session
robodkclaude-73 proposed it; testing it split two ways. Arm A, camera untouched:
voxel 1.0 -> completeness 0.8752 not closed; voxel 0.5 -> 0.9925 closed. But crest max
moves <0.15 mm across 1.0/0.5/0.25, so the voxel is RULED OUT for the height shortfall.
(My ruler comparison reads the RAW back-projected cloud, which never touches the voxel.)

**Remaining suspect for the ~1.5 mm: the matcher.** Peer's as-found dump has a 9x9 census
window = 4.24 mm at 300 mm vs a 20.2 px ring = 45% coverage, which rounds convex ridges and
predicts a multiplicative SLOPE (observed 0.905) rather than an offset. Their 5-point
2026-08-13 refit gives `sigma = 0.662 + 2.196e-6 Z^2` -- a distance-INDEPENDENT 0.66 mm
floor that is 77% of the noise at 300 mm, which disparity quantisation (pure Z^2) cannot
produce. Two independent routes to "matcher-side". `param-texturecountthresh` and
`param-texturedifferencethresh` are both 0 (texture gating fully OFF) = first target;
census window second. **Their Z^1.72 flag is RETRACTED by them; do not carry it forward.**

Related: [[first-live-take-board-halo]], [[deposit-segmentation-spec-plan]],
[[cell-characterization-2026-08-13]], [[raster-free-centreline-beats-nksr]].
