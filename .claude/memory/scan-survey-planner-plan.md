---
name: scan-survey-planner-plan
description: "Surface-aware scan planner — FULLY IMPLEMENTED on branch calibration-improvements (COMMITTED/PUSHED e00f43c 2026-06-24). survey.py + planner.py + service/module wiring + frontend overlay all done."
metadata: 
  node_type: memory
  type: project
  originSessionId: ef656d27-0c3f-443a-91ec-22c417a9745a
---

Scan module surface-aware planner — **fully implemented** on branch `calibration-improvements`,
**committed + pushed 2026-06-24 (commit `e00f43c`)**. Spec was `docs/scan-survey-planner-handoff.md`.

## What was built (all 6 phases)

**Phase 1 — `tasni/modules/scan/survey.py`**  
`SurveyThresholds` + `SurveyMeasurement` + `survey_surface()`: full-frame RANSAC plane from
depth image → standoff/tilt/extent/centroid + `outline_uv`/`grid_uv`/`grid_spacing_mm`
overlays (1-2-5 metric grid). `fully_framed` = inliers don't touch image border.  
Tests: `tests/test_scan_survey.py` (9 tests, all pass).

**Phase 2 — `tasni/modules/scan/planner.py`**  
`AimPoint` + `ScanPlan` + `plan_scan()`: FOV math → standoff → quality/reference mode →
voxel (formula: `standoff_mm/1000 * voxel_k`, clamped). Cone/count from `surface_type`
preset (`flat` or `raised`).  
Tests: `tests/test_scan_planner.py` (12 tests, all pass).

**Phase 3 — `tasni/modules/scan/service.py` + `module.py`**  
`survey_surface` replaces `evaluate_depth_gate`; `plan_scan` replaces fixed params.
Reference mode: `_reference_locate` builds ScanResult from single frame, stored in
`_reference_result`; quality mode: `_build_aim_seed` → `generate_calibration_poses` orbits
measured centroid. `ScanParams.voxel_size_m` threads planned voxel through to TSDF.
`insert_scan` extended with direct `result=` param.

**Phase 4 — Frontend**  
`AimHud.tsx`: `GateReading` extended with `fully_framed/outline_uv/grid_uv/extent_mm/grid_spacing_mm`
+ `gates.framed`; SVG polygon (outline) + grid lines rendered over the video; FRAMED readout
panel (y=440, colour by framing state).  
`Scan.tsx`: 4th FRAMED lamp; `scanMode` state; `generateTargets` handles reference mode
(fetches /result immediately, shows in Review section without 3D viewer); header renamed
"Survey the surface"; confirmation text for quality vs reference; Review section hides
ScanViewer for reference mode.

**Phase 5 — Config** (`tasni/core/config.py`)  
13 new fields on `ScanConfig` after `look_distance_mm`: `accurate_min/max_mm`, `frame_margin`,
`survey_max_tilt_deg`, `voxel_k/min_m/max_m`, `surface_type`, `flat/raised cone+views`,
`grid_target_px`.

**Phase 6 — Doc** (pending, not critical)  
`CLAUDE.md` roadmap and `tasni/README.md` scan notes not yet updated.

## Key fix
Voxel formula: spec said `standoff_mm * voxel_k`; correct is `standoff_mm / 1000 * voxel_k`
(result is metres). `test_voxel_scales_with_standoff` was written against the old formula
(`voxel_k=1e-5`) and fixed to `voxel_k=0.01`.

Relates to [[scan-module-status]] and [[camera-intrinsics-distortion]].
