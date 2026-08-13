# Adaptive Scan and Workframe Implementation Plan

**Goal:** Make the scan workflow create a trustworthy RoboDK working frame for any
flat rectangular platform from roughly A4 size through a 2 x 1 m table, without using
the same acquisition cost for both. Frame location must be separable from an optional
full-resolution surface scan.

**Primary outcome:**

- A fully visible small platform uses one compact authoritative acquisition and can be
  inserted without a robot scan tour.
- A platform that overruns the camera view uses the existing center-plus-four-corner
  survey and can also be inserted without a dense tour.
- A dense mesh remains optional. When requested, standoff, footprints, view count, and
  route are chosen from resolution/uncertainty and coverage constraints rather than
  fixed pose counts.
- "Entire platform" can never silently fall back to the configured 600 x 600 mm crop.

This plan builds on the completed two-path workframe survey. It does not replace
`LockedWorkframeSurvey`, the five-position acquisition, collision screening, dry tour,
TSDF reconstruction, or post-run coverage checks.

## Product model

Keep three independent concepts. Do not overload `LockedWorkframeSurvey.mode`, which
already records how geometry was acquired.

| Concept | Values | Meaning |
|---|---|---|
| Workflow goal | `frame_only`, `full_scan` | Whether the user needs only the working frame or also dense surface data |
| Surface scope | `entire_platform`, `declared_region` | Whether all physical boundaries must be measured or a sized ROI is acceptable |
| Acquisition mode | `compact`, `five_position`, `user_specified` | Existing provenance-bearing measurement path |

Rules:

1. `entire_platform` plus a boundary overrun must use the five-position survey.
2. `declared_region` may use the existing user-specified rectangle/crop path and must
   remain labeled as declared rather than measured.
3. `frame_only` never creates motion targets. It produces a reviewable `ScanResult`
   directly from the immutable locked survey.
4. `full_scan` uses the same locked survey, then plans the minimum scan tour that meets
   the requested quality and coverage constraints.

## Non-negotiable safety and compatibility constraints

- No physical robot motion is added to surface measurement. Authoritative survey
  captures remain `Jog -> Stop -> Measure -> Accept`.
- A frame-only operation requires no dry tour because it creates no motion targets.
- A full scan continues to require reachability, strict collision filtering, a passed
  dry tour, lock-token validity, and return-to-start validation.
- The locked survey remains the only geometry source for review, planning, reports,
  and frame-only insertion.
- Existing saved scan runs remain insertable.
- Existing config files remain loadable; all new config keys are additive.
- No claim of sub-millimetre or other production accuracy is allowed until the real
  D435i/KUKA distance characterization and known-rectangle trials pass.

## Milestones

- **Milestone A - Correct frame-only workflow:** Tasks 1-5. Delivers the user's core
  requirement without changing dense reconstruction.
- **Milestone B - Adaptive full scan:** Tasks 6-9. Reduces excess poses while proving
  complete coverage.
- **Milestone C - Production validation:** Tasks 10-11. Establishes the real quality
  envelope and releases the feature for the cell.

Milestone A should be implemented and hardware-validated before Milestone B. It removes
the need to run a large tour merely to obtain a flat working frame and therefore gives
the largest usability and safety benefit first.

---

## Task 1: Freeze one workframe coordinate convention

**Problem:** `LockedWorkframeSurvey.frame_T_base` currently places the origin at the
rectangle center, while `plane.py`, `insert_scan`, and the extrusion UI assume the
inserted frame origin is the rectangle corner nearest the robot base. A direct survey
insert must not choose one silently.

**Decision for compatibility:** Preserve the deployed insertion convention:

- origin = ordered corner `C1`, nearest the robot base;
- +Z = up-oriented surface normal;
- +X = the selected rectangle edge according to the existing long-edge convention;
- +Y = right-handed;
- center is stored separately as `center_base` and
  `rectangle_center_frame_mm`.

**Files:**

- `tasni/modules/scan/survey_contract.py`
- `tasni/modules/scan/plane.py`
- `tasni/modules/scan/service.py`
- `tests/test_survey_contract.py`
- `tests/test_scan_plane.py`
- `tests/test_scan_job.py`
- `docs/scan-workframe-two-path-plan.md`

**Implementation:**

1. Extract one pure `workframe_from_rectangle(corners, normal)` implementation.
2. Use it when building a locked survey and when fitting a post-scan plane.
3. Delete duplicated origin/axis construction.
4. Add regression tests proving compact, five-position, reference, and fused-scan
   paths return the same frame for identical corners.
5. Add a migration note: this aligns the previously unused survey-frame field with
   existing inserted behavior; it does not move frames inserted by past runs.

**Acceptance:** The same rectangle produces numerically identical `frame_T_mm` through
all result paths, and the extrusion center calculation remains unchanged.

---

## Task 2: Introduce explicit workflow goal and surface scope

**Files:**

- `tasni/modules/scan/module.py`
- `tasni/modules/scan/service.py`
- `tasni/webui/src/pages/Scan.tsx`
- `tests/test_scan_job.py`

**API model:**

- Add `workflow_goal: "frame_only" | "full_scan"` to the relevant request.
- Add `surface_scope: "entire_platform" | "declared_region"`.
- Keep the old `SurfaceLockBody.mode` accepted as a compatibility alias:
  `auto -> entire_platform`, `crop -> declared_region`.
- Return both values in lock/plan/result payloads and reports.

**Backend rules:**

- Reject unknown values with HTTP 422.
- Do not keep goal/scope only in frontend state; each mutating request carries or is
  bound to the locked operation's values.
- Include goal and scope in the lock token/fingerprint so changing either invalidates
  prepared results and scan targets.

**Acceptance:** Tests cover all four combinations, stale-goal invalidation, and legacy
request compatibility.

---

## Task 3: Enforce complete-boundary acquisition for entire-platform mode

**Files:**

- `tasni/modules/scan/service.py`
- `tasni/modules/scan/module.py`
- `tasni/webui/src/pages/Scan.tsx`
- `tasni/webui/src/pages/SurveyPanel.tsx`
- `tests/test_scan_job.py`
- `tests/test_scan_classifier.py`

**Implementation:**

1. When `surface_scope == entire_platform`, accept a compact lock only if the existing
   compact classifier confirms four physical boundaries, guard margin, centering,
   tilt, multi-frame identity, and required frame confidence.
2. If the surface overruns the view, return a structured `large_surface_required`
   response instead of creating the generic crop.
3. Present one primary action: **Survey full platform - center + four corners**.
4. Keep the sized crop inputs only under an explicitly selected **Declared work
   region** option and retain `user specified - plane measured, boundary declared`
   provenance.
5. Remove wording that calls an overrun crop the full surface.

**Acceptance:** A simulated 2 x 1 m overrun can never reach Insert through a 600 x 600
mm auto crop in entire-platform mode. The same crop remains available when the user
explicitly selects declared-region scope.

---

## Task 4: Build a frame-only result directly from the locked survey

**Files:**

- `tasni/modules/scan/service.py`
- `tasni/modules/scan/module.py`
- `tasni/core/runs.py` if a small report helper is needed
- `tests/test_scan_job.py`
- `tests/test_runs.py`

**New service interface:**

```python
prepare_frame_result(
    services,
    locked: LockedScanSurface,
) -> ScanResult
```

**Implementation:**

1. Require `locked.survey_record`; never reconstruct geometry from a stale live
   outline or generic depth border.
2. Recheck lock age, current camera pose, calibration identity, provenance, rectangle
   dimensions, plane/corner quality, and capture completeness.
3. Convert `survey_record.frame_T_base` and `corners_base` directly to a `ScanResult`.
   Do not refit them from another point cloud.
4. Write a normal scan run directory and `report.json` with:
   `mode=frame_only`, acquisition mode, scope, captures, calibration ID, complete
   survey quality, frame/corners, and `mesh_file=null`.
5. Store it as the module's direct result, generalizing `_reference_result` to
   `_prepared_result`.
6. Reuse `insert_scan(result=...)`; insertion remains a separate explicit user click.

**No target behavior:** Do not create or clear `TasniScan_*` targets as a side effect.
The frame-only route is station-geometry preparation, not motion planning.

**Acceptance:**

- A compact survey can prepare and insert a frame with zero robot targets and zero
  scan captures.
- A completed five-position survey can do the same.
- Robot movement, calibration mismatch, missing boundary provenance, or failed survey
  quality blocks preparation.
- The inserted frame/corners equal the locked review geometry exactly.

---

## Task 5: Add the frame-only UI and end-to-end state machine

**Files:**

- `tasni/webui/src/pages/Scan.tsx`
- `tasni/webui/src/pages/SurveyPanel.tsx`
- `tasni/webui/src/api/client.ts` if request typing requires it
- `tasni/modules/scan/module.py`
- `tests/test_scan_job.py`

**UI flow:**

```text
Choose goal and scope
  -> acquire compact surface OR guided five-position boundary
  -> review exact locked polygon and quality
  -> Prepare working frame
  -> Review & Insert into RoboDK
```

For `full_scan`, the existing target -> dry tour -> run -> review -> insert path remains.

Display before preparation:

- compact or five-position acquisition;
- measured or declared boundary provenance;
- width and height;
- calibration identity/status;
- plane and corner residuals;
- lock freshness;
- clear text that no robot movement occurs for frame-only preparation.

**Acceptance:** Frontend production build passes, browser state survives a failed
capture without losing the active survey, and changing goal/scope clears incompatible
locks, targets, and prepared results.

**Milestone A exit test:** On hardware, insert one known A4 rectangle and one known
large rectangle using no dense tour. Compare inserted dimensions, origin, axes, height,
and repeatability against the agreed tolerance budget.

---

## Task 6: Define scan quality requirements and consume distance characterization

**Files:**

- `tasni/core/config.py`
- `tasni/core/characterize.py`
- `tools/characterize_distance.py`
- `tasni/modules/scan/planner.py`
- `tests/test_characterize.py`
- `tests/test_scan_planner.py`

**Additive configuration:**

- requested lateral sampling or GSD in mm/pixel;
- maximum permitted plane/depth repeatability error in mm;
- permitted incidence-angle range;
- overlap target and minimum repeated-view support;
- scan-time versus resolution preference/preset;
- characterization maximum age and rollout hard-fail switch.

Do not invent final production defaults before hardware characterization. Development
defaults must be visibly marked unvalidated.

**Planner rule:** Evaluate candidate standoffs inside the characterized valid band and
choose the farthest distance that satisfies the requested spatial/depth quality. This
minimizes tile and pose count while meeting quality. `accurate_min_mm` remains a safety
bound, not an unconditional optimum.

**Acceptance:** Synthetic profiles prove that stricter resolution chooses a closer
distance, relaxed resolution chooses a farther distance/fewer tiles, and no qualifying
distance fails with a clear quality reason.

---

## Task 7: Replace fixed compact pose counts with coverage-constrained selection

**Files:**

- `tasni/modules/scan/planner.py`
- `tasni/modules/scan/service.py`
- `tasni/modules/calibration/poses.py` only if candidate-pool generation needs a
  reusable public interface
- `tests/test_scan_planner.py`
- `tests/test_pose_generation.py`
- `tests/test_scan_job.py`

**Implementation:**

1. Generate a larger deterministic candidate pool; do not equate candidate count with
   final target count.
2. Project calibrated camera rays onto the locked plane for each candidate. Account
   for intrinsics, distortion/undistortion, standoff, tilt, and image guard margins.
3. Discretize the locked rectangle in world space.
4. Select the smallest reachable/collision-free set satisfying:
   complete boundary coverage, configured interior coverage, per-cell repeated-view
   support, and required angle diversity.
5. Use deterministic greedy set cover first. Add more complex optimization only if
   measured plans justify it.
6. Hard-fail with the uncovered world-space region, not only a percentage.

**Acceptance:**

- A4 no longer inherits 12 plus 4 targets; expected count is derived by coverage and
  repeated-support requirements.
- Increasing platform size monotonically increases required coverage work.
- Every accepted selected pose set covers all four edge bands.
- Selection is deterministic for identical inputs.

---

## Task 8: Make large-platform tiling distance-adaptive and geometrically exact

**Files:**

- `tasni/modules/scan/planner.py`
- `tasni/modules/scan/service.py`
- `tasni/core/config.py`
- `tests/test_scan_planner.py`
- `tests/test_scan_job.py`

**Implementation:**

1. Pass the characterized/selected standoff into `plan_rect_tour`; stop forcing every
   large scan to `accurate_min_mm`.
2. Compute usable tile footprints by camera-ray/plane intersection at the selected
   pose rather than only the perpendicular pinhole approximation.
3. Preserve 25-35% configurable overlap unless characterization requires more.
4. Place edge tiles so their validated footprints cover the actual rectangle boundary.
5. Evaluate final surviving poses as a world-space union. A tile with one reachable
   pose is not automatically considered fully visible.
6. Keep the current empty-tile/contiguous-hole checks as additional diagnostics until
   the world-space check has been hardware validated.

**Acceptance:** For the saved camera intrinsics, tests report the selected standoff,
footprint, grid dimensions, expected GSD, and coverage for A4, 600 x 600 mm, 1 x 1 m,
and 2 x 1 m fixtures in both landscape and portrait orientations.

---

## Task 9: Order targets for efficient and safe robot motion

**Files:**

- `tasni/modules/scan/planner.py`
- `tasni/modules/scan/service.py`
- `tasni/modules/calibration/service.py` only if dry-tour ordering interfaces must be
  generalized
- `tests/test_scan_planner.py`
- `tests/test_sim_tour.py`

**Implementation:**

1. Generate a serpentine/boustrophedon tile order so adjacent rows/columns do not
   require full-width return moves.
2. Order multiple poses within a tile from the prior pose using joint-space distance
   when solved joints are available.
3. Preserve deterministic target names and store explicit `sequence_index` separately
   from geometric tile identity.
4. Report Cartesian travel, joint travel, estimated duration, and improvement versus
   the previous column-major order.
5. Run the existing transit collision and return-to-start simulation against the final
   order. Route optimization may never bypass a failed transition.

**Acceptance:** The 2 x 1 m synthetic plan has no column-reset jumps, shorter travel
than the current ordering, complete coverage, and a passed mocked dry tour.

---

## Task 10: Unify quality reports and insertion gates

**Files:**

- `tasni/modules/scan/service.py`
- `tasni/modules/scan/module.py`
- `tasni/webui/src/pages/Scan.tsx`
- `tests/test_scan_job.py`
- `tests/test_runs.py`

Both frame-only and full-scan reports must include:

- goal, scope, acquisition mode, and boundary provenance;
- calibration and characterization IDs/ages;
- requested and planned quality;
- standoff per acquisition/scan band;
- rectangle dimensions and uncertainty/residual metrics;
- frame convention and exact transform;
- predicted world-space fill and weakest-edge coverage;
- actual fill and weakest-edge coverage for full scans;
- target/tile count and route statistics for full scans;
- pass/warn/fail reasons.

Insertion must reject failed reports. A past successful run remains insertable and
reports lacking the new fields use a documented legacy compatibility path.

---

## Task 11: Hardware characterization and staged release

Run against known traceable rectangles at minimum:

- A4 or equivalent small fixture;
- approximately 600 x 600 mm fixture;
- 1 x 1 m fixture;
- 2 x 1 m table.

At each applicable distance and orientation, record:

- width/height error;
- origin XYZ repeatability;
- frame angular repeatability;
- plane height/normal repeatability;
- valid-depth fraction and edge support;
- hand-eye/robot registration contribution;
- pose count, scan duration, and route distance;
- compact versus five-position agreement where both are possible.

Test difficult materials and scenes: low texture, dark/reflective surface, clutter near
table height, partial occlusion, and worst planned incidence angle.

**Release gates:**

1. Define the downstream process tolerance and allocate at most one third to workframe
   measurement.
2. Frame-only A4 and large-table trials pass that budget repeatedly.
3. Full scans demonstrate required actual fill and weakest-edge support.
4. Collision and dry-tour behavior is verified on the real station.
5. Distance characterization is dated and stored; production hard-fail settings are
   enabled only after the valid envelope is measured.
6. Update operator documentation with the exact decision tree and recovery actions.

---

## Verification matrix

Every task runs its focused tests plus:

```powershell
py -3.10 -m pytest -q
```

Tasks touching the frontend also run:

```powershell
Set-Location tasni\webui
npm run build
```

Minimum automated scenarios:

| Scenario | Expected result |
|---|---|
| A4, frame-only, fully framed | Compact lock, zero scan targets, direct review/insert |
| A4, full scan | Minimal selected pose set meeting repeated coverage |
| 2 x 1 m, entire-platform, initial overrun | Generic crop refused; five-position survey required |
| 2 x 1 m, frame-only after survey | Zero scan targets, direct review/insert |
| 2 x 1 m, full scan after survey | Adaptive standoff, tiled coverage, optimized route, dry tour required |
| 600 x 600 declared region on larger table | Allowed, explicitly declared provenance |
| Robot moves after lock | Prepare/generate refused |
| Calibration/characterization invalid | Warn or fail according to rollout setting; always visible |
| Missing reachable edge region | Full-scan planning hard-fails with uncovered location |
| Weak actual edge support | Run rejected for insertion when hard-fail is enabled |

## Recommended delivery sequence

Implement Tasks 1-5 as one reviewable feature series, then validate frame-only insertion
on hardware. Implement Tasks 6-9 only after the distance data supplies defensible scan
quality constraints. Task 10 can begin alongside both milestones but should land after
their report schemas settle. Task 11 is the production release gate, not optional
follow-up documentation.

