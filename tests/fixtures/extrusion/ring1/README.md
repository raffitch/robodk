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
old largest-cluster rule selected the board residual and reported a 51 mm bead,
and the fixture was written to protect the ring-shape selection that rejects it.

The shape gate's own behaviour is unchanged -- the selected candidate's coarse
circle still reads r 41.13 mm against 41.12 under the colour-gate chain -- but
**the board no longer arrives as a competing candidate at all**: it now sits
below the derived floor rather than above the old constant one, so there is a
single candidate and `selected["points"] == largest["points"]`. "The ring is not
the largest blob" is therefore no longer what this frame exercises; that stays
pinned on the synthetic scene where the residual is placed on purpose
(`test_characterize_selects_ring_instead_of_larger_raised_patch`).

What this fixture pins now is the refined measurement on a real 1 mm-worded
frame: r 40.39, centre (218.56, 150.18), bead 9.82, top z 6.78 (re-measured
2026-08-31; previously 39.17 / (217.94, 150.44) / 13.26 / 6.14). Less board is
fused into the footprint, so the bead narrows and the crest-read radius moves
out to 0.74 mm inside the coarse circle where it used to sit 1.95 mm inside it.
There is no ground truth for this ring in the archive, so what is asserted is
self-consistency plus the unchanged selection.

## `ring1_low_relief_20260829.npz`

From trial `20260829-151445-acb42814/characterize-01`. A hand-placed dried ring
only 2-11 mm tall (median 2 mm) against a board whose own depth noise is +/-3 mm.
**Under the old constant 2.5 mm floor** its two thinnest arcs fell out, so ONE
ring reached DBSCAN as two disconnected arcs: 48/72 angular bins and 25/72.
Graded separately neither cleared the 0.70 coverage gate, so characterization
aborted on a ring that was in fact captured all the way round -- the two arcs
together cover 71/72. The fixture then protected arc assembly.

**What it protects now is the datum.** The derived floor removes the
fragmentation at its source: this capture's substrate fits at -1.81 mm with
sigma 0.836 mm, so the floor lands 2.0 mm above THAT -- about 0.2 mm in
work-frame Z, against 2.5 mm before. The thin arcs clear it and the ring arrives
as ONE complete cluster covering 72/72 (measured 2026-08-31: r 42.11, against
42.0 via assembly before). Measuring height above the surface the deposit rests
on, rather than above a nominal plane the surface is not on, is worth ~2.3 mm
here -- more than the ring's own thin arcs are tall.

Arc assembly itself is still covered, on the synthetic scene whose dips are cut
below the derived floor on purpose
(`test_characterize_assembles_one_ring_from_arcs_the_height_floor_broke`).

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

* **The patch is rejected with no colour input**, and the radius bias falls from
  the +0.6-0.7 mm takes 1-3 carried to +0.24 mm: of the 1478 patch points
  reaching the work ROI, 7 survive to the crest. **That cleaning is the
  downstream chain's, not the compactness filter's** -- the two facts must not
  be welded into one claim. With the filter ON and OFF (2026-08-31):

  | stage | ON | OFF |
  |---|---|---|
  | work_roi | 22226 (patch 1478) | 22226 (patch 1478) |
  | compactness | 21714 (patch 1386) | 22226 (patch 1478) |
  | deposit_cluster | 1266 (patch 105) | 1277 (patch 116) |
  | radial_trimmed | 1231 (patch **72**) | 1228 (patch **72**) |
  | top_surface | 432 (patch **7**) | 436 (patch **6**) |

  Compactness removes 92 of the 1471 patch points that go away -- 6%. DBSCAN,
  the radial trim about the FITTED circle and the crest filter reach the same
  endpoint either way.

* **What compactness decides on this frame is the branch-guard outcome.** It
  drops five compact components totalling 512 points, which changes the raster
  topology; with `deposit_min_length_beads = 0` the frame reproduces the
  original 2026-08-29 cell abort exactly. That role is real and load-bearing,
  and the paired tests pin it -- it is simply a different mechanism from
  "compactness cleans the patch off the crest". The filter's
  contamination-rejection value is separately evidenced on the 2026-08-30
  archive (spec 3.5).
* **The take is reported INVALID, and that is the honest answer.** Completeness
  0.846, maximum angular gap 55.3 deg. This capture's 1 mm quantisation gives the
  substrate a sigma of 0.759 mm, so `3 * sigma` saturates the configured clamp at
  2.0 mm -- about 2.54 mm in work-frame Z once the fitted plane's height AT THE
  RING CENTRE is added (`plane_offset_at_center_mm` = 0.542; the plane's -1.32 mm
  intercept is its value at the work-frame ORIGIN, and the 0.8 deg tilt carries
  it up to +0.54 by the time it reaches the ring). That is essentially the old
  2.5 mm floor, and it costs the same
  low-relief sector (crest 2.9-4.9 mm) it always did: at 2.5 mm this ring measured
  completeness 0.87 with a 46 deg gap. The metrics say so instead of returning a
  ring that was not there.
* On a **protocol-2** capture (0.1 mm words, substrate sigma ~0.55 mm) the derived
  floor comes out at 1.65 mm and that sector is kept -- 8/8 valid at completeness
  0.992-0.993 across the 2026-08-30 archive (`tests/test_extrusion_golden.py`).

The pair of tests on this fixture therefore pins: contamination rejected without
colour, compactness's real (topological) role separated from the downstream
chain's, and a frame whose own noise cannot support a low enough floor saying so
rather than measuring a partial ring as a whole one.
