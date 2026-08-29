# Real ring fixtures

Each file holds the depth frame, camera intrinsics, camera-to-work transform and
search centre / recipe of one real capture.

The first two omit the RGB image, on the rule that ring measurement is geometric
and must not depend on bead colour. The cell run of 2026-08-29 disproved that
rule -- see `ring1_take04_branchguard_20260829.npz`, which therefore carries its
colour frame (JPEG-encoded under `color_jpeg`, ~240 KB; decode with
`cv2.imdecode`). The older two still exercise the abstention path, where the
chroma gate stands down and the conservative deposit floor is restored, so both
behaviours stay covered.

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
welded itself to the bead. Height cannot reject it: depth is quantised at 1 mm
and the patch reads 2.9-5.9 mm, one to three LSB over the 2.5 mm deposit floor,
while facing straight up like any bead crest. It survived the radial trim,
dilated (bead/2) and closed (bead/2) into skeleton arms of 22 and 17 px against
a 15 px spur limit, and the guard's retry ladder only ever GROWS the mask, so
all three attempts failed identically (mask 4362 -> 4471 -> 4643, branch pixels
stuck at 2).

Takes 1-3 of the same trial carried the same patch and did NOT crash -- their
skeleton topology happened to stay benign -- and reported radii biased 0.6-0.7 mm
large (43.02 and 42.78 against 42.2). The guard is topological, so it catches
this contamination only by luck. Loosening it would have turned the crash into a
silently wrong number, which is what the paired test asserts against.

Measured over these four frames, saturation separates bead from board ~20:1:
bead S median 106-114 (0.74-0.80 above 60), the patch S median 15-16 (0.04),
bare board S median 5-25 (0.00). With the board gone the deposit floor can drop
from 2.5 to 1.5 mm, which recovers a 45 deg sector whose crest reads only
2.9-4.9 mm -- the reason ALL FOUR takes were invalid, not just the one that
crashed. Gated at floor 1.5 mm all four close: completeness 0.992, angular gap
2.7-2.9 deg, radius spread 0.32 mm (was 0.65 mm over the three that survived).

The fixture protects that pair: the colour gate, and the floor it earns.
