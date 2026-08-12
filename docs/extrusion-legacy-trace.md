# Post-Extrusion Cylinder: legacy trace and deliberate changes

Trace date: 2026-08-12. Sources: `PostExtrusionToolpath/rcv&scan.py`,
`index_2.html`, and the legacy `231006_RoboArchPaper.rdk` station.

## Observed legacy sequence

1. The browser generates one circular layer and POSTs XYZ points to
   `/process_toolpath`.
2. `rcv&scan.py` copies `camTarg`, raises the copy by the largest filtered path Z,
   creates a RoboDK curve and machining project, and runs its generated program.
3. Machining events call `AirOn` at `CallPathStart` and `AirOff` at
   `CallPathFinish`.
4. The robot moves once to `camTarg_copy`; one RGB-D observation is received.
5. The depth cloud is cropped/downsampled, plane-subtracted, outlier-filtered,
   largest-cluster filtered, upward-normal clustered, rasterized, skeletonized,
   nearest-neighbour mapped back to 3D, nearest-neighbour ordered, and spline
   resampled to 40 points.
6. A branched skeleton causes the entire single-frame capture/processing sequence
   to repeat without a bound. An unbranched measured spline is returned to the UI.
7. The UI displays that measured path and uses its X/Z coordinates for the next
   layer, with a vertical bead offset.

## Verified valve mapping

The programs were read in a private, headless RoboDK session; neither was run.

- `AirOn`: `Set IO_508=1`, then `Set IO_601=1`.
- `AirOff`: `Set IO_508=0`, then `Set IO_601=0`.

This proves the legacy station mapping and polarity. It does **not** by itself
constitute an approved physical I/O test. Tasni therefore keeps the committed
default `hardware_io_test_approved=false`; an authorized operator can set the
interlock only in the ignored local `tasni.config.json` after approving the cell.
This workstation's operator explicitly approved it on 2026-08-12. The setup tool
reproduces the programs without executing them.

On 2026-08-12 the first setup attempt used Tasni's generic isolated-session helper,
which launches RoboDK with `-SKIPINI`. That private process did not load the user's
active license settings, displayed a misleading free-license message, and did not
persist the programs. This was **not evidence that the installed RoboDK license was
inactive**. The setup utility now uses a separate headless instance without
`-SKIPINI`, while retaining `-NEWINSTANCE` so it cannot attach to the open GUI. It
also reopens its output and fails if the instructions did not persist. Run:

```powershell
py -3.10 tools\setup_extrusion_station.py Tasni.rdk --inplace
```

The corrected setup was then run successfully. A second fresh private RoboDK
instance reopened `Tasni.rdk` and verified `AirOn` contains `Set IO_508=1` and
`Set IO_601=1`, while `AirOff` contains the corresponding `=0` instructions.
Neither program was executed by the setup utility; approval remains a separate,
operator-owned local interlock and is never inferred from station setup or licensing.

## Effective processing defaults

- Plane distance: 0.0025 m.
- Voxel: 0.002 m.
- Statistical outlier: 20 neighbours, standard-deviation ratio 2.0.
- Radius outlier: 16 neighbours in 0.1 m.
- Largest DBSCAN cluster: epsilon 0.005 m, minimum 10 points.
- Upward normal: Z greater than 0.92; secondary DBSCAN epsilon 0.02 m,
  minimum 10 points.
- Alpha-shape raster parameter: 0.1 after conversion to millimetres.
- Measured spline: exact `splprep(..., s=0)`, resampled to 40 points.

## Correction finding

The standalone UI calls the returned curve a deviation/correction path, but it
does not calculate `nominal - measured`. It copies the measured X/Z coordinates
into the following commanded layer and offsets height. That propagates the prior
shape rather than compensating its error.

Tasni deliberately replaces this with an opt-in, bounded radial compensation:
measure radial error against the nominal circle, cyclically smooth it, negate it,
apply a configurable gain, and clamp maximum correction. The original measurement
is always archived, and a correction is never marked executed merely because it
was calculated.

## Known unsafe or non-reproducible legacy behavior removed

- Live mode was a source-code constant (`RUNTYPE = 'LIVE'`).
- Branch retries were unbounded.
- Fault paths did not explicitly force the valve off.
- Paths, crop limits, thresholds, IP addresses, and station item names were hard-coded.
- The browser could request printing directly without a current-path dry run.
- Raw RGB-D evidence and per-stage provenance were not archived per layer.

## RoboDK curve-follow implementation

Tasni uses RoboDK's manufacturing abstraction rather than expanding sampled path
coordinates into station targets. For every active layer it creates one temporary
curve object with XYZ positions and IJK normals, attaches it to the selected work
frame, and links it to a Curve Follow/Robot Machining Project. The project owns
approach/retract, process and rapid speeds, path blending, and `CallPathStart` /
`CallPathFinish`. Its linked target-free robot program is then pinned to the exact
selected XYZRPW and collision-validated before simulation or live execution.

The curve, settings project, and linked generated program use the
`TasniCylinder_` prefix. They are deleted on normal completion, cancellation, or
fault, and the Cylinder Test reset action removes stale prefixed artifacts left by
an interrupted application process. Existing user curves, programs, and targets are
outside that namespace and are not touched.
