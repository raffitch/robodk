# Real ring characterization fixture

`ring1_checkerboard_20260828.npz` contains the depth frame, camera intrinsics,
camera-to-work transform and search centre from trial
`20260828-171615-f088cf48/characterize-01`. The RGB image is deliberately omitted:
ring characterization is geometric, and the regression must not depend on bead
colour.

This frame contains both the deposited ring and a larger above-plane residual on
the ChArUco board. The old largest-cluster rule selected the board residual and
reported a 51 mm bead. The fixture protects the ring-shape selection that rejects
that residual.
