# Handoff: Post-Extrusion Cylinder Test in Tasni

> **Historical requirements document.** The module is now implemented. Read
> `docs/extrusion-current-handoff.md` first for current architecture, live-test
> status, the RoboDK status -5 diagnosis, and next steps.

## Purpose

Implement the existing PostExtrusionToolPath workflow as a Tasni module for a fast, repeatable cylinder experiment needed for the RoboArch / Prototypes for Humanity paper.

This is **not** a request to replace the working system with a smaller inspection demo. Preserve its end-to-end behavior, then improve its engineering: generate/receive a toolpath, print it through RoboDK, capture one RGB-D observation, reconstruct the deposited path, guard against bad skeletons, fit the measured spline, archive the trial, and retain the path-correction capability.

## Source material to inspect first

Legacy working logic:

- `C:\Users\User\Desktop\desktop\PostExtrusionToolpath\`
- Main integration candidates:
  - `rcv&scan.py`
  - `PostExtrusionToolpathRobodk.py`
  - `PostExtrusionToolpathSplineFitting.py`
  - `PostExtrusionToolpath03d.py`
  - `scanOnly.py`
  - `send_toolpath - 3D.html`
- The legacy `rcv&scan.py` configures RoboDK machining-program events named `AirOn` at path start and `AirOff` at path finish (`CallPathStart` / `CallPathFinish`). **These programs do not currently exist in `Tasni.rdk` and must be created as part of this task.** Inspect the associated legacy RoboDK station/controller setup to determine the real output/signal mapping; do not guess the I/O address or polarity.
- Earlier UI experiment: `C:\Users\User\Desktop\desktop\code test 1\code test 1\index.html`
- More developed standalone UI candidate: `C:\Users\User\Desktop\desktop\index_2.html`

Current target project:

- `D:\DesktopStuff\RAFFI NO TOUCH\backuprobodk\RoboDkClaude\`
- Read `CLAUDE.md`, `tasni/README.md`, and `docs/agent-debug-map.md` before changing code.
- Reuse Tasni's existing RoboDK session, camera connection, calibration/transforms, job/event handling, frontend patterns, run storage, safety checks, and testing conventions.

## Required workflow

For the paper experiment, the operator should be able to:

1. Open a simple **Cylinder Test** tool in Tasni.
2. Set radius, layer count, layer height, bead/extrusion diameter, robot speed, and extrusion rate using sliders or numeric inputs.
3. Generate and preview the circular layer toolpaths.
4. Dry-run the complete cylinder through the existing Tasni/RoboDK safety workflow before live printing.
5. Print the cylinder one layer at a time, with the extrusion valve/output enabled only for the extrusion path and disabled for approach, travel, inspection, completion, cancellation, and faults.
6. After each layer, move to the known inspection pose and save **one synchronized RGB-D frame** for analysis. The camera may stream beforehand to stabilize, but do not add a multi-pose scan or TSDF requirement.
7. Apply the existing point-cloud filtering, deposited-geometry extraction, 2D skeleton/branch guard, pixel-to-3D mapping, ordering, and spline fitting logic.
8. Compare the measured path with the nominal circle and archive the complete layer record.
9. Keep correction available: calculate a smoothed compensation from nominal versus measured geometry, generate a corrected path, and support a subsequent corrected print when enabled.

## Preserve before refactoring

- Single-frame capture and processing.
- Existing filters and their effective default parameters.
- Skeleton extraction and guarded reprocessing.
- Measured 3D spline returned to the interface.
- RoboDK program generation/execution.
- The existing path-start `AirOn` and path-finish `AirOff` extrusion actuation behavior, including the actual RoboDK/controller I/O mapping once verified from the station.
- Recipe/trial archival behavior.
- Correction behavior that can be verified in the legacy implementation.

First establish regression fixtures from saved legacy inputs and outputs. Refactoring is acceptable only when the new result is equivalent or the change is deliberately documented and tested.

## Engineering improvements

- Separate capture, segmentation/filtering, centreline reconstruction, comparison, correction, and archival into testable components.
- Replace hard-coded paths, thresholds, crop regions, and coordinate offsets with validated versioned configuration.
- Use Tasni's calibrated camera, work, and robot transforms; keep coordinate frames and units explicit.
- Vectorize point processing with NumPy/OpenCV where appropriate.
- Make skeleton ordering deterministic and graph-based.
- Bound branch-guard retries and record every processing attempt.
- Validate minimum point count, path completeness/continuity, plausible length/radius, and maximum gaps.
- Save the raw observation so every measurement can be rerun offline.
- Record stage timings, input/output point counts, warnings, configuration, calibration ID, and Git commit.
- Keep correction opt-in and visibly distinguish measured, nominal, commanded, and corrected paths.

## Extrusion valve and I/O control

Valve actuation is part of the working print system, not an optional UI detail. Preserve the legacy behavior while making it explicit and fail-safe:

- `Tasni.rdk` has no `AirOn` or `AirOff` programs at handoff. Create both programs in the Tasni station and ensure generated print programs call them at the correct path events.
- Determine the correct digital/analog controller output, signal name, polarity, and required timing from the legacy station or verified hardware configuration before defining their instructions. Never invent an I/O address.
- Make creation/setup repeatable where practical (for example, a checked setup utility or documented station-initialization step), rather than relying only on an unexplained manual edit to one `.rdk` file.
- Validate the new `AirOn` and `AirOff` programs first in simulation, then with an approved hardware I/O test before allowing a live print.
- Keep the output **off by default**. Turn it on immediately before the extrusion segment and off immediately after it.
- Always command/verify off on startup, normal completion, cancellation, timeout, exception, emergency stop, lost robot connection, and before moving to the inspection pose.
- Never energize the physical output during preview or dry run. Provide a clearly indicated simulation/mock state.
- Keep valve lead-in/lead-out or dwell times configurable if the real process requires pressure buildup or relief.
- Expose valve state in the Tasni job UI and record requested/confirmed state changes with timestamps in the layer archive.
- Do not conflate valve on/off with extrusion rate. If rate uses a separate analog output, pump command, or pressure setting, discover and model that separately.
- Require an explicit live-run confirmation/interlock before enabling physical extrusion.

## Dry-run mode

Dry run is a required first-class mode, not just a visual preview:

- Execute or simulate the complete generated robot path using the same poses, speeds, tool/frame selection, layer transitions, approach, retreat, and inspection motion intended for the live run.
- Run the normal reachability, joint-limit, collision, program-generation, and state checks.
- Show simulated `AirOn` / `AirOff` events in the job timeline, but block all physical valve, pump, pressure, and extrusion outputs at the lowest practical I/O layer.
- Clearly label the UI and archived result as `DRY_RUN`; it must never be mistaken for a printed trial or included in the paper's measured-print counts.
- Report failures with the layer/path location and prevent **Print & Record** until the most recently generated toolpath has passed dry run. Regenerating or materially editing the path invalidates that pass.
- Allow cancellation and confirm the robot returns to, or remains in, a known safe state.

## Archive requirement

Create one archive record per printed layer, grouped under one cylinder trial. The UI table should show thumbnails and headline values; full-resolution evidence should remain in the run directory.

Suggested structure:

```text
runs/extrusion/<trial-id>/
  trial.json
  layer-001/
    manifest.json
    color.png
    depth.*
    height-or-pointcloud.*
    segmentation.png
    skeleton.png
    comparison.png
    nominal_path.json
    measured_path.json
    corrected_path.json       # when correction is enabled
    report.json
```

Each layer record should contain:

- Trial, recipe, material components, and quantities.
- Layer index and timestamp.
- Nominal radius, layer height, extrusion diameter, robot speed, and extrusion rate.
- Raw RGB and depth evidence.
- Processing images: filtered/segmented result, skeleton, and nominal-versus-measured overlay.
- Nominal, commanded, measured, and corrected paths as applicable.
- Raw point count and point counts after major filtering stages.
- Branch-guard attempts/cycles and any warnings.
- Total and per-stage processing time.
- Mean absolute, RMS, and maximum centreline deviation in millimetres.
- Measured circle centre/radius, path completeness, and validity status.
- Calibration/configuration/software identifiers needed to reproduce the result.
- Valve/output identifier, verified polarity, commanded state transitions, timestamps, and any actuation fault.

The archive summary must make it easy to report: total recipes, total trials, total layers, and the metrics for the demonstrated trial.

## Scope control

The fast first milestone is a cylinder with only a few layers on a known, reasonably flat build surface. Do not make the implementation depend on:

- multi-view scanning or TSDF fusion;
- SAM or a trained segmentation model;
- a general-purpose replacement for Joanne's Mémoire toolpath generator;
- Firebase migration;
- arbitrary print geometry.

These exclusions constrain the paper-test UI, not the preserved PostExtrusionToolPath processing or correction workflow.

## Acceptance criteria

- A cylinder can be configured, previewed, dry-run, printed, and inspected from Tasni.
- Dry run covers the complete live path and safety checks while guaranteeing that no physical extrusion output can energize.
- A live print is enabled only after the current generated toolpath passes dry run; changing that path requires another dry run.
- Live printing reproduces the verified legacy `AirOn`/`AirOff` behavior; preview and dry run never energize the valve.
- `Tasni.rdk` contains the newly created, correctly mapped `AirOn` and `AirOff` programs, and their setup is documented or reproducible.
- The valve is forced off on every exit or fault path and before inspection motion.
- Each layer uses one saved RGB-D observation.
- The measured centreline/spline is displayed with the nominal path.
- Mean, RMS, and maximum deviation are reported in millimetres.
- Raw/filtered point counts, branch cycles, and processing time are reported.
- Every layer archives its raw imagery, derived imagery, paths, parameters, recipe, metrics, and provenance.
- A saved observation can be reprocessed offline without moving the robot.
- Existing Tasni tests remain passing and new processing/archive behavior has regression tests.
- Correction remains part of the workflow and is never claimed as executed unless a corrected path is actually printed and measured.

## Recommended implementation order

1. Trace and document the legacy pipeline precisely, including what its correction stage actually executes.
2. Create saved-frame regression fixtures and expected outputs.
3. Add typed Tasni models plus the trial/layer archive format.
4. Port the single-frame processing stages without changing behavior.
5. Integrate Tasni calibration, camera, RoboDK, jobs, and safety services.
6. Add the Cylinder Test UI and toolpath generator.
7. Add deviation metrics, overlays, timings, and the archive table.
8. Validate one complete few-layer cylinder trial, then validate correction if included in the paper claim.

## Original repository state at handoff (historical)

Before this original requirements document was added, the full Tasni test suite
completed successfully: **217 passed, 0 failed**, with two RoboDK import deprecation
warnings. No implementation work had been performed at that time. For the current
implemented state, read `docs/extrusion-current-handoff.md`.
