# Cylinder Test: Current Implementation and Live-Test Handoff

Last updated: 2026-08-12. Active branch: `calibration-improvements`.
Current implementation commits: `e36b4d5`, `a0670f1`, and the scan-surface placement
change below.

This is the authoritative current-state handoff for Tasni's extrusion/cylinder
module. `docs/HANDOFF_EXTRUSION_CYLINDER.md` is the original requirements document;
its final "not implemented" status is historical and no longer accurate.

## Current outcome

The Cylinder Test is implemented end to end in the Tasni app:

- Browser-only live analytic preview responds immediately to sliders. It does not
  create executable robot coordinates or approve a path.
- `Generate coordinates & fingerprint` freezes recipe, setup, and dense XYZ points
  into an exact backend plan.
- RoboDK receives one native XYZ+IJK curve per layer, not hundreds of targets.
- A native Curve Follow/Robot Machining Project owns interpolation,
  approach/retract, process/rapid speed, blending, and path start/finish events.
- The linked generated program is pinned to the selected fixed XYZRPW orientation
  and then validated with collision checking.
- Complete dry run uses mock `AirOn`/`AirOff` programs, never physical outputs.
- Live print remains locked until the exact current fingerprint passes dry run and
  the operator confirms the live run.
- Each printed layer captures one RGB-D observation, processes/archives it, and can
  produce an opt-in bounded radial correction with a new fingerprint.

Primary code:

- `tasni/modules/extrusion/module.py`: API, plan/preflight/dry/live interlocks.
- `tasni/modules/extrusion/service.py`: dry/live jobs and cleanup behavior.
- `tasni/modules/extrusion/toolpath.py`: deterministic layered circle coordinates.
- `tasni/modules/extrusion/surface.py`: scan → extrusion placement handoff and checks.
- `tasni/modules/extrusion/processing.py`: single-frame measurement pipeline.
- `tasni/core/rdk_io.py`: native curves, machining projects, IK, generated programs.
- `tasni/webui/src/pages/Extrusion.tsx`: controls, oblique bird's-eye preview, workflow.
- `tests/test_extrusion.py` and `tests/test_extrusion_job.py`: main regression coverage.

## Why RoboDK returned status -5

The operator reported:

```text
RoboDK could not generate the curve-follow program (status -5.0)
```

The brief curve appearance was real: `AddCurve` succeeded, but RoboDK could not
find a feasible start/path pose for the Curve Follow Project. It was not a RoboDK
license failure and was not related to the hardware-I/O approval.

The placement UI previously exposed center X/Y but implicitly fixed the build plane
at selected-frame Z=0. Selecting `World` therefore placed the first layer near world
Z = half the bead diameter. During the live diagnosis, the current spindle TCP was:

```text
Frame: World
XYZ:   [-5.084, -1505.704, 377.227] mm
RPW:   [-177.072, 58.385, 89.722] deg
```

A circle near World origin was consequently far from the current reachable work
area. Commit `a0670f1` fixed the placement workflow:

- `build_plane_z_mm` is explicit and fingerprinted.
- `POST /api/modules/extrusion/current-tcp` reads the selected TCP in the selected
  frame without robot motion.
- `Seed path start from current TCP` makes circle angle zero equal to the current
  TCP, derives center X/Y and build-plane Z, and captures the exact current RPW.
- Preflight samples fixed-orientation IK across the layers before enabling dry run.
- Selecting `World` shows a warning that all values are station-world coordinates.
- A negative Curve Follow setup error now reports tool, frame, XYZ bounds, and RPW.

## Placement on the scanned work surface (preferred workflow)

The intended flow is: **scan the surface → insert it → build on that frame**. The Scan
module's insert creates `Tasni Work Frame`, the `Tasni Work Surface` rectangle, and the
fused mesh, and records the applied run in `runs/scan/active.json`. The Cylinder Test
now reads that pointer.

- `GET /api/modules/extrusion/scan-surface` reports the applied surface (frame, size,
  run id, applied time) or that none is applied.
- **Center on scanned surface** sets the work frame to the scan frame, centers the
  cylinder on the measured rectangle, and sets build-plane Z = 0 (the scan frame's
  origin lies on the surface). It records the originating run in `setup.scan_run_id`.
- That run id is part of the plan fingerprint, so **re-scanning the table invalidates a
  surface-placed plan** exactly like editing the recipe does.
- Preflight then enforces the placement: same run, same frame, and the wall
  (`radius + bead/2`) must fit inside the measured rectangle, reporting signed per-edge
  margins.
- Manual placement is still legal. **Seed path start from current TCP** clears
  `scan_run_id`, and preflight reports `placement: "manual"` with an advisory when a
  scanned surface exists on another frame.

Why the centre cannot be derived from the recorded extents: the scan puts its frame
origin on the rectangle **corner** nearest the robot base, and frame +Y is `Z × X`,
which on this cell points *off* the rectangle (Y spans about −295..0 mm). So (0, 0) is a
corner and (size/2, size/2) has the wrong Y sign. The insert therefore publishes
`rectangle_corners_frame_mm` and `rectangle_center_frame_mm` in **frame** coordinates,
and extrusion centres on the corner mean. A surface applied before those fields existed
is recovered from its `report.json`; if neither source is available the module refuses
to centre rather than guess (`available: false`).

Primary code: `tasni/modules/extrusion/surface.py`,
`tasni/modules/scan/plane.py:rectangle_in_frame`, and the payload written by
`tasni/modules/scan/service.py:insert_scan`.

## Exact operator retry sequence

1. Refresh the Cylinder Test page and connect/refresh the RoboDK station.
2. Select the actual print tool, inspection tool, and inspection target.
3. Place the path, preferring the scanned surface:
   - **Preferred:** if the Scan surface row is not green, run the Scan module and insert
     its result, then click **Center on scanned surface**.
   - **Manual alternative:** jog the selected print TCP to the intended first point on
     the circle and click **Seed path start from current TCP**.
4. Review center X/Y, build-plane Z, and RPW. If using `World`, confirm these are
   deliberately world coordinates.
5. Generate coordinates and fingerprint.
6. Run geometry/station preflight. It must show all sampled IK poses reachable, and the
   placement section must not report an overhang or a stale scan.
7. Run the complete RoboDK dry run. Do not proceed live until collision validation,
   simulation, inspection motion, and return-to-start all pass.

Changing any recipe/setup value invalidates the fingerprint and prior checks, as does
re-scanning the surface a plan was centred on.

## Most recent live verification

A read-only current-TCP seed was tested with:

- print tool: `spindle`
- frame: `World`
- radius: 40 mm
- one layer
- inspection tool: `Realsense`
- inspection target: `NEUTRAL`

Results:

1. Geometry preflight passed.
2. All 9 sampled fixed-orientation path poses had IK solutions.
3. Native curve and Curve Follow program generation succeeded; status -5 did not
   recur.
4. Collision validation then stopped at 2.6%:

```text
Collision detected
Program: TasniCylinder_DRY_00b6aef849_L001
Instruction 5: MoveJ 1
```

This is the current next problem if the operator uses that same spindle/World seed:
inspect the generated approach move and the station collision pair. Do not bypass or
disable collision checking merely to obtain a pass. Likely adjustment points are the
chosen start placement, exact print-tool orientation, approach clearance, or an
incorrect/stale collision model. Confirm the physical cell before changing any
collision-map configuration.

The above audit ran immediately before the final failed-artifact retention change was
loaded, so that specific test artifact was cleaned. On the next failed dry run, the
native path artifacts remain for inspection as described below.

## Failed-artifact lifecycle

RoboDK items owned by this module use the `TasniCylinder_` prefix:

- `<program>_Curve`: dense native curve object.
- `<program>_Settings`: Curve Follow/Robot Machining Project.
- `<program>`: linked generated robot program.
- `<program>_Inspect`: inspection program, when created.

Behavior after `a0670f1`:

- Successful dry runs clean temporary native path artifacts.
- Failed curve generation retains curve/settings (and any linked program).
- Failed program/collision validation retains curve/settings/program for inspection.
- Dry-run mock I/O programs are always deleted, including on failure.
- **Reset / clean RoboDK path** removes stale `TasniCylinder_` artifacts.
- Existing user items outside that namespace are not removed.
- A retry with the same generated item names removes/replaces those exact stale items.

The curve may initially appear beneath `World` during `RDK.AddCurve`, then it is
parented to the selected frame. If the selected frame is `World`, remaining beneath
World is expected. The dense path is deliberately a curve, not a target list.

## Valve, approval, and license facts

The verified legacy mapping from `231006_RoboArchPaper.rdk` is:

```text
AirOn:  Set IO_508=1; Set IO_601=1
AirOff: Set IO_508=0; Set IO_601=0
```

The local ignored `tasni.config.json` currently has the operator's hardware-I/O
approval active. Approval is a local safety interlock and is distinct from mapping
discovery, RoboDK licensing, geometry feasibility, IK, and collision validation.

An earlier misleading free-license message came from a private RoboDK process launched
with `-SKIPINI`; it did not load the user's normal license settings. The setup tool was
fixed to use a licensed isolated instance without `-SKIPINI`. Do not diagnose status
-5 as a license issue.

Dry runs call generated mock programs whose instructions are comments only. Physical
outputs remain blocked. Live jobs additionally force `AirOff` on startup, around layer
boundaries/faults, and before inspection/return behavior.

## RoboDK API design decisions

Commit `e36b4d5` replaced target-per-point generation with the native manufacturing
workflow:

1. `RDK.AddCurve` receives Nx6 `[X,Y,Z,I,J,K]` vertices with no projection.
2. The curve is parented to the selected work frame.
3. `RDK.AddMachiningProject` creates a Curve Follow Project.
4. `setPoseFrame`, `setPoseTool`, project pose/joints, and
   `setMachiningParameters(part=curve)` define generation context.
5. Machining parameters configure process/rapid speeds, blending, approach/retract,
   and `CallPathStart`/`CallPathFinish`.
6. `UpdatePath` and project `Update(COLLISION_OFF)` generate the linked program.
7. Generated motion instructions are pinned to the exact requested fixed rotation.
8. Program `Update(COLLISION_ON)` is the authoritative complete validation.

Official references used during implementation:

- <https://robodk.com/doc/en/Robot-Machining-Curve-Follow-Project.html>
- <https://robodk.com/doc/en/PythonAPI/examples.html#points-to-curve>
- <https://robodk.com/doc/en/PythonAPI/examples.html#robot-machining-settings>
- <https://robodk.com/doc/en/Robot-Machining.html>

## Verification and runtime state

After the scan-surface placement change:

- Full Python suite: 244 passed.
- Extrusion-focused suite: 19 passed.
- Frontend TypeScript check and production build: passed (existing chunk warning only).

At commit `a0670f1`:

- Full Python suite: 234 passed.
- Extrusion-focused suite: 14 passed.
- Frontend TypeScript check: passed.
- Frontend production build: passed (only the existing chunk-size warning).
- Backend was restarted after the final changes and reported idle with no plan.
- Vite dev server remained running and picked up frontend changes.
- No `server/` files changed, so no Jetson deploy/restart was required.
- Commit `a0670f1` was pushed to `origin/calibration-improvements`.

Useful verification commands:

```powershell
py -3.10 -m pytest tests/test_extrusion.py tests/test_extrusion_job.py -q
py -3.10 -m pytest -q
cd tasni/webui
npm run typecheck
npm run build
```

## Safe next-agent priorities

1. Ask the operator to retry with the path placed on a freshly scanned surface
   (**Center on scanned surface**) rather than the `spindle`/`World` seed that produced
   the collision below. A circle centred on the measured table sits in the reachable
   work area by construction; world zero did not.
2. If preflight rejects a sampled coordinate, use the returned frame/XYZ and inspect
   the exact setup; do not start dry run.
3. If Curve Follow generation or collision validation fails, inspect the retained
   `TasniCylinder_*` curve/settings/program in RoboDK before Reset.
4. For the observed `MoveJ 1` collision, identify the collision pair and distinguish a
   real approach hazard from stale/oversized station geometry. Preserve fail-closed
   behavior.
5. Only after a complete dry-run pass should the operator enable the already-approved
   live workflow and explicitly confirm the run.

