# Real ring characterization fixtures

Both files hold the depth frame, camera intrinsics, camera-to-work transform and
search centre of one real capture. The RGB image is deliberately omitted: ring
characterization is geometric, and the regressions must not depend on bead
colour.

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
