---
name: scan-chain-audit-2026-08-14
description: Full scan-chain audit lives in docs/scan-audit-2026-08-14.md; headline finding is two disagreeing camera models (+2.05% lateral scale from depth)
metadata: 
  node_type: memory
  type: project
  originSessionId: 96cdd3b0-4f8f-4a39-b367-7a5a1d2cb711
  modified: 2026-08-14T04:56:35.540Z
---

End-to-end scan audit done 2026-08-14, committed as `docs/scan-audit-2026-08-14.md`
(`50fcab7`, pushed to origin/`calibration-improvements`). 16 ranked findings covering the
operator journey, host backend, and Jetson camera server. Read that doc before touching
scan accuracy, latency, or the camera server — it is the analysis of record.

The load-bearing finding (A1): the Jetson's `rs.align` maps depth into the colour frame
with the **factory** colour intrinsics (fx 908.10, zero distortion) while the host
back-projects that same aligned depth with the **ChArUco-calibrated** ones (fx 889.87,
k1 +0.115) and applies no undistortion at all. Net **+2.05 % lateral scale on every X/Y
measured from depth** (a 600 mm region measures ~612 mm) plus ~10–17 mm of uncorrected
radial error at the frame edge — where extents/corners are measured. The 2026-08-13
characterization structurally cannot see it: `length_err_mm` intersects undistorted
*colour* rays with the depth-fitted plane, so it validates the colour model and depth z
but never the depth grid's own lateral scale, and plane RMS is scale-invariant.

**Why:** every downstream tolerance in the scan module (survey_corner_agreement_mm 8 mm,
survey_rect_discrepancy_mm 6 mm) is tighter than this bias, so tuning them is meaningless
until one camera model is authoritative end to end.

**How to apply:** fix A1 first, and add a known-length check measured *from the depth
cloud alone* to `tools/characterize_distance.py` — that is the missing acceptance test.
Also note `tests/test_scan_job.py::test_generate_refuses_when_too_far` was FAILING at
audit time (finding A2: `distance_tol_mm` 50→150 is applied around a standoff already
clipped to [300, 800], so the gate accepts 150–950 mm). Related: [[scan-module-status]],
[[camera-intrinsics-distortion]], [[cell-characterization-2026-08-13]].
