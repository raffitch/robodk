# tasni — robotic-fabrication control platform

One external app (Python, drives RoboDK over its API) that hosts **all** the
cell's robot workflows as pluggable **modules** on a shared **core**. Calibration
is module #1 and the proof-of-pattern; scan / ArUco-to-plane / define-targets /
3D-printing plug in the same way. Inspiration: Aibuild (ai-build.com).

## Run

```bash
py -3.10 -m tasni                 # web app -> http://127.0.0.1:8000
py -3.10 -m tasni.cli             # headless calibration (prints metrics)
py -3.10 -m tasni.cli --apply "TOOL"   # ...and write the result into a tool
py -3.10 tests/test_calibration_synthetic.py   # math checks (no hardware)
py -3.10 tests/test_calibration_job.py         # job checks (fake hardware)
```

Have RoboDK open with the station loaded (the `Target*` poses + the tool to
calibrate) before a real run — in `attach` mode the app binds your running
RoboDK; the Jetson camera server must be up on TCP 1024.

## Architecture

```
tasni/
  core/      shared services — nothing workflow-specific lives here
    session, rdk_io    RoboDK connection + item I/O (poses cross as numpy 4x4)
    camera             RealSense-over-TCP client (port-1024 wire format)
    config             layered dataclass config (+ optional tasni.config.json)
    jobrunner, events  run long robot jobs off-thread; stream progress/frames
    geometry, logging  rigid-transform helpers; per-run artifact dirs
  modules/
    base, registry     WorkflowModule ABC + ServiceContainer (DI) + registry
    calibration/       module #1 (see below)
  webapp/              FastAPI shell + static SPA that hosts the modules
  cli, __main__        headless + web entrypoints
```

**The module contract.** A module gets the core via a `ServiceContainer` and
contributes (a) a FastAPI `router()` and (b) a UI `panel_html()`/`panel_js()`.
It must not open sockets, import `robolink`, or spawn threads — those are the
core's job. That boundary is what lets new modules drop in as pure leaves.

## Calibration module (#1)

ChArUco eye-in-hand hand-eye calibration, refactored from `macros/AutoCalibrate.py`:

- **Solver: OpenCV `calibrateHandEye` TSAI** (kept per the research review), on a
  clean, explicit frame chain (replacing the macro's mixed-convention
  `pose_2_Rt`). Optional **post-solve refinement** minimizing reprojection error.
- **Quality metrics — the #1 research gap, previously unreported:**
  - **reprojection error (px)** on the solve poses,
  - **held-out validation-pose error (px)**,
  - **board-consistency (mm)** — spread of the board's recovered base-frame
    position across views (helps separate calibration error from D435i depth
    noise, an open question in `docs/best-practices-review.md`).
- **Review-then-apply**: the solved pose is shown with metrics; nothing is
  written to the tool until you click Apply.

Artifacts (report.json, summary.txt, annotated frames) land in `runs/calibration/<stamp>/`.

### Solver caveat (worth knowing)

OpenCV's TSAI implementation is numerically fragile as the **camera→flange mount
rotation approaches 180°** (its rotation parameterization degenerates there);
PARK/HORAUD/ANDREFF stay exact on identical data. We keep TSAI per the project
decision and rely on the new reprojection metric to make a bad solve **visible**
(it shows hundreds of px instead of silently applying a wrong calibration) and on
refinement to sharpen good solves. If a real mount turns out to sit near that
singularity, the cheapest robust fix is to seed the reprojection refinement from
the best linear method rather than switch the default solver — a one-line option
we can add if the metrics ever flag it.
