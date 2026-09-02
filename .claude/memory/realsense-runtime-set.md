---
name: realsense-runtime-set
description: "The Jetson camera server now takes a runtime `SET` command for the filter chain (no deploy per A/B arm). MERGED + deployed + live-verified on the cell 2026-09-01. Device options stay pinned in the unit file, by design."
metadata: 
  node_type: memory
  type: project
  originSessionId: 301796e3-2dbd-4a52-9fe9-f44aa33275d1
  modified: 2026-09-01T11:56:55.108Z
---

`SET <key>=<value> ...` over the burst protocol changes the depth **filter chain**
at runtime — no unit-file edit, no `bootstrap`, no deploy per A/B arm. Merged to
`main` and deployed 2026-09-01. Design: `docs/superpowers/specs/2026-09-01-realsense-runtime-parameters-design.md`.

**Live-verified on the cell**, not just in tests:

```
SET                                  -> reads back all 11 achieved options
SET spatial_smooth_delta=8           -> ok, achieved 8.0, visible in a fresh greeting
SET laser_power=300                  -> REFUSED (unknown setting)
SET spatial_smooth_delta=500         -> clamped to 50.0 (the real SDK range)
SET spatial=0                        -> filters list drops "spatial", options go null
SET depth_min_m=1.0 depth_max_m=0.5  -> REFUSED (see below)
```

**The 12 settable keys are the FILTER CHAIN ONLY** — spatial on/off + its 3
options, temporal's 3, depth_min_m/max_m, hole_filling, decimation (pinned at 0
and refused, since enabling it would change the depth geometry the greeting
already declared). Device options (laser power, visual preset, depth units,
exposure) and advanced mode are deliberately NOT settable: they persist on the
device across a restart, which is how this cell once ran at 300 mW while a dated
characterization assumed 150. See [[jetson-laser-power-vs-characterization]].

**Send one with `py -3.10 tools/camera_set.py`** (added 2026-09-01 `41004fc`): bare
= read-only, `spatial=0` = one arm, `--restore` = the full spec-4.1 restore,
`--dry-run` to see the line. The merged runtime-parameters work was **server-side
only** — until that tool, every A/B had to be hand-rolled over a socket.

**How to apply:**
- A runtime override **dies on restart** — that impermanence is the safety
  argument, not a limitation. The unit file stays the boot default and the
  reviewable diff. There is deliberately no persist verb.
- A `SET` **bumps the camera generation**, retiring every session greeted before
  it. Swapping the chain mid-burst would otherwise fuse frames from two chains
  into one median, and the fusion guard cannot catch it (it compares *geometry*,
  and a filter swap does not change geometry).
- Every knob lands in the greeting's `filter_options` as an **achieved read-back**,
  except `spatial` on/off which shows in the `filters` list — so both arms of an
  A/B are always distinguishable on disk. See [[unsettled-burst-has-no-static-field]]
  for the other thing to control when running one.
- **The auto-pull timer (~2 min) restarts the camera when `server/` changes and
  will wipe a runtime override mid-sweep.** Check the greeting between arms.
- **Inverted depth thresholds:** the real SDK ACCEPTS `depth_min_m > depth_max_m`
  and then returns empty depth while reporting success — a take captured under it
  looks exactly like a take of an empty scene. Found on hardware, now refused in
  the server. Those two go through a constructor, not `set_option`, so the range
  clamp never saw them.
