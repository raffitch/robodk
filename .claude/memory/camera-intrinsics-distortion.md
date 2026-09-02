---
name: camera-intrinsics-distortion
description: D435i RGB factory distortion is all-zeros; real distortion measured from a calib run lives in tasni.config.json
metadata: 
  node_type: memory
  type: project
  originSessionId: 7e502eed-e08a-4362-bed0-da3cd336c1c0
---

The RealSense **D435i color** stream reports factory distortion `[0,0,0,0,0]`
(model `inverse_brown_conrady`) even though the RGB lens physically distorts —
Intel calibrates depth/IR carefully but ships RGB distortion as zeros. The
factory **K is correct** (`fx 908.10, fy 908.14, cx 650.24, cy 366.65` @ 1280x720,
matches `_DEFAULT_INTRINSICS` in `tasni/core/config.py`).

Two ways intrinsics get calibrated now:

1. **Auto, on first hand-eye Run** (default, zero-touch): `CalibrationJob` derives
   K + distortion from its OWN captured `TasniCalib_*` views (`solve_intrinsics`,
   k3 fixed), applies them, and recomputes each view's board pose before the
   hand-eye solve. Gated on a marker (`runs.read_active("intrinsics")` via
   `service.intrinsics_present`) so it runs **only when missing**; toggle
   `calibration.auto_intrinsics`. Board stays centred in hand-eye poses ⇒ edge
   coverage thin ⇒ k3 fixed; the run logs/reports `intrinsics_auto` + a "run the
   dedicated capture" note when coverage <60%.
2. **Dedicated full-frame capture** (best accuracy, optional): backend only —
   `/intrinsics/*` routes + `IntrinsicCalibSession` (wave board, auto-capture on
   coverage, Solve+Apply). The **UI for this was REMOVED** (user wanted intrinsics
   fully under-the-hood, no buttons); the endpoints remain as an API escape hatch.
   Supersedes #1 if run.

Both call `service.apply_intrinsics` → writes `camera.intrinsics`/`dist_coeffs`
**live (no restart)** + persists to `tasni.config.json` + writes the
`runs/intrinsics/active.json` marker (source "auto"|"manual"). Backend:
`tasni/modules/calibration/intrinsics_calib.py` (`IntrinsicCalibSession`,
`solve_intrinsics`, `coverage_from_corners`) + `/intrinsics/*` routes; tests in
`tests/test_intrinsics_calib.py` + `test_calibration_job.py`. Workflow:
intrinsics (auto) → hand-eye.

The earlier **stopgap** in `tasni.config.json` `camera.dist_coeffs`
(`[0.1049, -0.3135, 0.00178, -0.00085, 4.327]`) came from a hand-eye run's
internal re-estimate (`intrinsics_check.dist_recovered` in `report.json`); k3=4.3
is overfit (14 partial-frame views). The Step-0 calibration supersedes it.

Tool: `python tools/jetson_intrinsics.py [WxH]` stops the `realsense-camera`
service, reads color intrinsics, restarts it (confirms factory K, shows zero dist).

Gotchas: (1) config is read **once at startup** — no live reload, so a
`tasni.config.json` change needs an app restart. (2) Calibration runs do NOT
persist raw per-pose corners/flange poses (only report.json + annotated JPEGs),
so an intrinsics change cannot be re-solved offline — you must re-Run (cheap: the
`TasniCalib_*` targets persist in the station, no re-aim/dry-tour). See
[[calibration-improvements-status]].
