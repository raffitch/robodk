# Real ring frame with ChArUco-board depth noise attached

`ring2_board_noise_20260828.npz` holds the depth frame, intrinsics, camera-to-work
transform and the applied recipe/centre of trial
`20260828-204846-5b455377/layer-001` (300 mm standoff, ring characterized and
applied minutes earlier: r 42.6 mm, bead 12.8 mm, centre (214.6, 146.7)).

There is no colour image, and none is needed: **segmentation reads depth only**
(design 2026-08-30). This fixture is where that is exercised end to end on a real
cell frame — the chain measures the applied ring from geometry alone, with no
colour input of any kind. Like the `../ring1/` fixtures it is a pre-protocol-2
capture (1 mm depth words, already aligned), kept as a regression frame rather
than as reference geometry.

On the cell this take FAILED with `branch guard exhausted`. Measured unfiltered,
the bare board's depth sits at z p50 0.8 / p99 4.8 mm above work-frame Z=0, so a
large share of it clears any fixed floor a real bead also clears; the patches that
touch the ring join its DBSCAN cluster, pass the upward-normal test (a flat board
faces straight up) and dilate into a lobe fused to the ring -- a skeleton
T-junction with a 37 mm arm, longer than the spur limit. Raising the floor cannot
fix it (the bead's own z is p25 1.8 / p50 3.8 mm) and produces confidently wrong
radii instead. The fixture protects the radial trim about the FITTED ring that
separates bead from board by shape.
