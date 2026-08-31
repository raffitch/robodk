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

---

# The stacked layer-2 frame layer 2 has never been measured from

`ring2_stacked_layer2_20260831.npz` holds the depth frame, the camera greeting
(native depth intrinsics + the depth→colour extrinsic — this is a **protocol-2**
capture, unlike the pre-protocol-2 frames above), the hand-eye pose, and the
applied recipe/centre/nominal Z of trial `20260831-195459-19838507/layer-002`
take 1: ring 2 hand-placed on ring 1, on MDF, r 42.2 mm, bead 8.4 mm, layer
height 4.6 mm, centre (208.48, 138.03), commanded Z 8.8 mm.

**On the cell this take measured 0.294 complete with a 254 deg angular gap, and
so did every layer-2 take ever recorded — 0 of 6 valid across 2026-08-30 and
2026-08-31.** Replayed stage by stage, the ring arrives whole (36/36 angular bins
in the work ROI) and leaves DBSCAN as five arcs, because the crest of a
hand-placed ring 2 swings ~10 mm around the circumference and the 3D
neighbourhood breaks where it steps. With arc assembly off above layer 1, the
largest arc — 110 deg of it — became the measurement.

This fixture is where the deposit floor under layer N is exercised on a real
stacked frame. It is deliberately a take that stays **INVALID**: the point of the
change is that the gate should reject it for what is wrong with the capture (a
contiguous ~50 deg sector with no usable depth return — a 19 mm stack seen from
one pose shadows itself) rather than because segmentation discarded five sixths
of a ring it had already found. Completeness moves 0.294 → 0.515, the gap
254 → 175 deg, and the fitted radius spread across this take's three repeats
collapses from 7.24 mm to 0.29 mm.

Its counterpart is `tests/test_extrusion_golden.py`, which holds the *opposite*
guard on the 2026-08-30 archive: those layer-2 takes must never be "recovered"
into a full ring, because there the circumference genuinely is not there.
