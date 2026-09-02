---
name: extrusion-platform-centring
description: "Extrusion centring resolves the middle of the platform from the RoboDK STATION (centre frame, else work-surface mesh) before the runs/scan disk pointer, expressed in whichever work frame is selected"
metadata:
  type: project
---

Shipped 2026-08-29 (`1cceb3c`, pushed to `main`). Centring used to read the platform
only from `runs/scan/active.json`, so clearing sessions produced "no scanned surface is
applied" while the geometry was still in the RoboDK tree.

`resolve_platform_center()` in `tasni/modules/extrusion/surface.py` now tries, in order:
1. **`Tasni Work Center`** — a frame the scan inserts at the rectangle's middle (new,
   child of `Tasni Work Frame`). Its origin IS the centre.
2. **`Tasni Work Surface`** — the quad object's own mesh, read with
   `Item.GetPoints(FEATURE_OBJECT_MESH)` + `PoseWrt(frame)` (`RdkIO.object_mesh_in_frame`).
3. **`runs/scan/active.json`** — and only for the frame it was recorded in.

The Work-frame dropdown chooses the **coordinate system, not the location**: verified
live on the cell station — `Tasni Work Frame` → (212.1, 149.7, z 0), `World` →
(-21.0, -1391.1, **z -166.3**), same 424 x 299 mm platform.

Traps this design encodes:
- **Bounds are only honest when the rectangle is axis-aligned in the asked-for frame**
  (checked via polygon area vs bounding-box area). In `Positioner Base` the rectangle is
  rotated, so extents are withheld and the fit abstains rather than passing a box bigger
  than the table. The *centre* (mean of corners) stays exact under any rotation.
- **`build_plane_z_mm` is the platform's height in the selected frame**, not a hardcoded
  0 — 0 was only ever true while that frame sat on the surface.
- **Only the disk source sets `scan_run_id`.** A station-derived centre has no run to
  claim; inventing one would arm the staleness check against a scan it never came from.
  Because that makes it read as "manual", `surface_check` now runs the overhang fit
  whenever the plan's frame IS the platform's frame, run id or not.
- **Do not add fields to `CylinderSetup`** — `plan_fingerprint` hashes the whole setup
  dump, so a new field invalidates every stored plan and in-flight measure session.

See [[scan-module-status]], [[pfh-paper-ring-stack-experiment]],
[[restart-tasni-backend-after-code-edits]], [[windows-python-and-encoding-traps]].

## TRAP (2026-08-30): renaming a station object makes it INVISIBLE, with a useless error

Operator hit "No platform is known for this work frame. Run the Scan module and insert
its result, or place a 'Tasni Work Center' frame in the middle of the platform
yourself." while a perfectly good platform was sitting in the station — the objects had
been renamed **`Tasni Work Surface old`** and **`Tasni Scan Mesh old`**. The lookup is an
EXACT name match, so an "old"/"v2"/any suffix makes the object invisible, and the error
message says nothing about the near-miss it just walked past. Renaming them back fixed
it instantly.

**Check the station object NAMES first** when centring says no platform is known — before
suspecting a missing scan or `runs/scan/active.json`. The message's two suggested
remedies (run Scan, or place a frame yourself) are both wrong advice in this case.
