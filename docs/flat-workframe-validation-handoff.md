# Handoff - flat workframe validation next step

Status: design recommendation captured after the flat scan mesh change. Implement next.
Branch: `calibration-improvements`.
Last updated: 2026-07-02.

## Current Capability

The scan flow can now:

- connect RoboDK, the real robot link, and the RealSense camera;
- guide the operator to a good surface survey pose;
- lock a surface and create `TasniScan_*` targets;
- run a multi-view scan;
- fit a flat work plane and rectangle;
- insert `Tasni Work Frame`, `Tasni Work Surface`, and `Tasni Scan Mesh` into RoboDK;
- preserve raw and measured scan artifacts for debugging.

For flat work surfaces, the inserted mesh is intentionally a fitted flat plane. The
measured TSDF topology is saved separately because ChArUco print/depth holes can make
the measured topology look patterned even after Z projection.

Current artifact convention:

- `mesh.obj` / `mesh.ply`: fitted flat surface used for preview and RoboDK insert.
- `measured_tsdf_mesh.obj` / `measured_tsdf_mesh.ply`: cleaned measured TSDF topology.
- `raw_tsdf_mesh.ply`: raw fused TSDF mesh.
- `work_surface_rect.obj`: generated rectangle reference.

Relevant recent commits:

- `6540e0a Avoid stale scan preview cache`
- `70d8286 Load scan previews by run id`
- `74e7673 Insert fitted flat scan mesh`

## Natural Next Step

Build a `Validate Workframe` step before adding more scanning complexity.

The main open risk is not whether the app can create a plausible frame. It is whether
the scan-derived frame is accurate enough for robot work. Validation should directly
measure that confidence on the cell.

Recommended flow:

1. Operator scans and inserts the flat workframe as today.
2. Operator starts `Validate Workframe`.
3. Robot visits or probes 4-6 validation points on the plane/corners/edges.
4. App reports:
   - plane Z error;
   - corner or edge error;
   - frame origin error;
   - normal/angle error;
   - pass/fail against configured tolerance.

This should answer: "Can I trust this scan-derived frame enough to place a model and
program robot work?"

## UX Direction

Split scan modes explicitly:

- `Flat Workframe Scan`: fit plane, insert clean flat mesh, intended for the current
  workframe/model-placement goal.
- `Measured Surface/Object Scan`: preserve real TSDF geometry, intended later for actual
  3D object shape.

The Review panel should state:

- inserted mesh type: `fitted flat plane`;
- measured scan quality: coverage, weakest edge, raw/measured artifacts saved;
- optional future toggle: preview `fitted plane` vs `measured TSDF`.

Add operator-facing warnings when useful:

- board/paper/printed texture was present on the scanned surface;
- surface is treated as flat and bumps/texture are intentionally removed;
- weak edge coverage requires rescan before trusting the frame.

## Why Not Neural Cleanup Next

Do not make neural cleanup the next milestone. For this phase, deterministic geometry is
the right tool: TSDF fusion, RANSAC plane fit, support/visibility checks, normal filtering,
connected-component cleanup, and then a fitted plane for a flat workframe.

Neural segmentation can be useful later as an assist layer, for example RGB masking of
the visible platform before fusion. It should not be the primary metrology authority for
the flat workframe.

## Implementation Pointers

Start in these files:

- `tasni/modules/scan/module.py`: add validation route(s).
- `tasni/modules/scan/service.py`: validation job orchestration and scan result access.
- `tasni/core/rdk_io.py`: RoboDK target/probe movement helpers if existing ones are not enough.
- `tasni/webui/src/pages/Scan.tsx`: add Validate button and result panel after Insert.
- `tasni/core/config.py`: add validation tolerances and point-count defaults.

Keep tests focused:

- mock RoboDK validation points and expected errors;
- fail when no active scan is inserted;
- verify pass/warn/fail thresholds;
- frontend typecheck/build.

## Validation Metrics Proposal

Config defaults to start:

- `validation_point_count = 5`
- `validation_plane_tolerance_mm = 1.0`
- `validation_edge_tolerance_mm = 2.0`
- `validation_normal_tolerance_deg = 0.2`
- `validation_warn_tolerance_scale = 2.0`

Report shape:

```json
{
  "status": "pass|warn|fail",
  "plane_z_rms_mm": 0.4,
  "plane_z_max_mm": 0.9,
  "edge_error_max_mm": 1.6,
  "origin_error_mm": 0.8,
  "normal_error_deg": 0.12,
  "points": [...]
}
```

## Important Tradeoff To Preserve

For the current goal, accuracy means the fitted workframe/plane is correct, not that the
inserted mesh preserves every measured bump. The fitted flat mesh is the operational
surface. The measured TSDF files are diagnostic evidence and future object-scan material.
