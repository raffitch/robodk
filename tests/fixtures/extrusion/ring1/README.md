# Real ring fixtures

Each file holds the depth frame, camera intrinsics, camera-to-work transform and
search centre / recipe of one real capture.

**All three are pre-protocol-2 captures: 1 mm depth WORDS, already aligned to
colour.** That is not incidental. The measurement chain runs on 0.1 mm words
(protocol 2) since 2026-08; these frames are held here because each one carries a
failure mode worth pinning, and they are processed at the 2 mm voxel they were
captured under. Their coarse quantisation also inflates the fitted substrate's
own sigma (0.76–0.86 mm against 0.52–0.57 mm on a protocol-2 take), which is why
the derived deposit floor behaves differently here than on the 2026-08-30 archive
— see `ring1_take04_branchguard_20260829.npz` below. **Protocol-2 captures are
the supported path**; these are regression fixtures, not the reference geometry.

Segmentation reads NO COLOUR. `ring1_take04_branchguard_20260829.npz` still
carries its colour frame (JPEG-encoded under `color_jpeg`, ~240 KB; decode with
`cv2.imdecode`), and it is kept deliberately: it is the frame the 20:1
bead-vs-board saturation separation was measured on, and the evidence that the
separation had inverted by 2026-08-30 (bead median S 25, printed board 28). No
test reads it.

## `ring1_checkerboard_20260828.npz`

From trial `20260828-171615-f088cf48/characterize-01`. This frame contains both
the deposited ring and a larger above-plane residual on the ChArUco board. The
old largest-cluster rule selected the board residual and reported a 51 mm bead.
The fixture protects the ring-shape selection that rejects that residual.

## `ring1_low_relief_20260829.npz`

From trial `20260829-151445-acb42814/characterize-01`. A hand-placed dried ring
only 2-11 mm tall (median 2 mm) against a board whose own depth noise is +/-3 mm.
Its two thinnest arcs fall under the ROI height floor, so ONE ring reached DBSCAN
as two disconnected arcs: 48/72 angular bins and 25/72. Graded separately neither
cleared the 0.70 coverage gate, so characterization aborted on a ring that was in
fact captured all the way round -- the two arcs together cover 71/72.

The fixture protects arc assembly: completeness must be judged on the assembled
ring, never on one connected component.

## `ring1_take04_branchguard_20260829.npz`

From trial `20260829-165938-8bef1770/layer-001-take04` -- the noise-floor phase,
take 4 of 5, which aborted the run with `branch guard exhausted`.

A 22-point patch of BARE BLACK CHECKER, 12 mm outside the ring's +X flank,
welded itself to the bead. Height alone cannot reject it: the patch reads
2.9-5.9 mm, one to three of this capture's 1 mm LSBs over any floor a real bead
clears, while facing straight up like any bead crest. It survived the radial
trim, dilated (bead/2) and closed (bead/2) into skeleton arms of 22 and 17 px
against a 15 px spur limit, and the guard's retry ladder only ever GROWS the
mask, so all three attempts failed identically (mask 4362 -> 4471 -> 4643,
branch pixels stuck at 2).

Takes 1-3 of the same trial carried the same patch and did NOT crash -- their
skeleton topology happened to stay benign -- and reported radii biased 0.6-0.7 mm
large (43.02 and 42.78 against 42.2). The guard is topological, so it catches
this contamination only by luck. Loosening it would have turned the crash into a
silently wrong number, which is what the paired test asserts against.

### What this fixture proves NOW: the patch dies in geometry, not in colour

Measured 2026-08-31 running this frame through the fitted-substrate chain (task
7 step 5a of the 2026-08-30 segmentation design):

* **The patch is destroyed by SHAPE.** The compactness filter drops 5 of the 6
  connected components in the work ROI, without its fail-open bypass firing.
  Of the 1478 patch points that reach the ROI only **7** survive to the crest
  (99.5% gone), and the radius bias falls from the +0.6-0.7 mm takes 1-3 carried
  to +0.24 mm. With `deposit_min_length_beads = 0` the same frame reproduces the
  original cell abort exactly -- so compactness is measurably doing the job the
  saturation gate used to do.
* **The take is reported INVALID, and that is the honest answer.** Completeness
  0.846, maximum angular gap 55.3 deg. This capture's 1 mm quantisation gives the
  substrate a sigma of 0.759 mm, so `3 * sigma` saturates the configured clamp at
  2.0 mm -- about 2.54 mm in work-frame Z once the fitted plane's -1.32 mm offset
  is added. That is essentially the old 2.5 mm floor, and it costs the same
  low-relief sector (crest 2.9-4.9 mm) it always did: at 2.5 mm this ring measured
  completeness 0.87 with a 46 deg gap. The metrics say so instead of returning a
  ring that was not there.
* On a **protocol-2** capture (0.1 mm words, substrate sigma ~0.55 mm) the derived
  floor comes out at 1.65 mm and that sector is kept -- 8/8 valid at completeness
  0.992-0.993 across the 2026-08-30 archive (`tests/test_extrusion_golden.py`).

The pair of tests on this fixture therefore pins: contamination rejected by
geometry, and a frame whose own noise cannot support a low enough floor saying so
rather than measuring a partial ring as a whole one.
