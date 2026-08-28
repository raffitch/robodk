# Real ring frame with ChArUco-board depth noise attached

`ring2_board_noise_20260828.npz` holds the depth frame, intrinsics, camera-to-work
transform and the applied recipe/centre of trial
`20260828-204846-5b455377/layer-001` (300 mm standoff, ring characterized and
applied minutes earlier: r 42.6 mm, bead 12.8 mm, centre (214.6, 146.7)). Colour is
omitted: the measurement is geometric and must not depend on bead colour.

On the cell this take FAILED with `branch guard exhausted`. Measured unfiltered,
the bare board's depth sits at z p50 0.8 / p99 4.8 mm, so 22.7% of it clears the
2.5 mm deposit floor; the patches that touch the ring join its DBSCAN cluster,
pass the upward-normal test (a flat board faces straight up) and dilate into a
lobe fused to the ring -- a skeleton T-junction with a 37 mm arm, longer than
the spur limit. Raising the floor cannot fix it (the bead's own z is p25 1.8 /
p50 3.8 mm) and produces confidently wrong radii instead. The fixture protects
the radial trim about the FITTED ring that separates bead from board by shape.
