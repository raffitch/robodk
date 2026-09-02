---
name: unsettled-burst-has-no-static-field
description: "The FIRST depth burst at a new pose does not correlate with itself (+0.045 vs +0.937 settled) - its residual field is transient filter state, not the scene. Measured 2026-09-01 on the 2026-08-31 archive."
metadata: 
  node_type: memory
  type: project
  originSessionId: 301796e3-2dbd-4a52-9fe9-f44aa33275d1
  modified: 2026-09-01T06:11:57.132Z
---

The Jetson's depth filter chain is built once at startup and advances **only when a
client pulls a frame** (`getFrames()`), so the first burst at a fresh pose runs a
temporal filter that is still converging. `settle_s` does not help: it elapses before
the host opens the depth stream.

Measured 2026-09-01 by correlating each burst's residual field (fitted plane removed,
rasterised into polar cells on the board annulus) on
`runs/extrusion/20260831-195459-19838507/layer-002*`:

| take | frame 0 vs frame 4 | vs the other takes |
|---|---|---|
| layer-002 (first burst at the pose) | **+0.045** | +0.053 / +0.052 |
| layer-002-take02 | **+0.937** | +0.960 vs take03 |
| layer-002-take03 | **+0.937** | +0.960 vs take02 |

**Why it matters:** two settled bursts reproduce each other's static field at **+0.960**,
so the static structure is real, highly repeatable, and measurable. The unsettled burst
is not merely noisier — it does not even reproduce *itself*, and is uncorrelated with
both settled takes. Its residual field is transient filter state, carrying almost none
of the scene. Nothing observed lands between the two populations.

The per-frame substrate sigma shows the same thing more weakly: first burst
0.695 -> 0.537 mm monotonic, repeat bursts flat at ~0.54.

**How to apply:**
- Any diagnostic that reads static structure must use a SETTLED burst. Either raise
  `measure_depth_fusion_frames` and analyse the last frames, or take a throwaway
  warm-up capture at each pose — a burst is already settled by the *second* one at
  the same pose.
- The production median still fuses every frame including unsettled ones, so a take's
  own `sigma_mm` does not describe the settled camera.
- `tools/probe_roll_readings.py` enforces this: it refuses any capture whose first and
  last frame correlate below +0.5.
- Do NOT confuse this with [[depth-pimples-census-and-preset-landmine]]'s finding that
  the noise is static rather than temporal. That holds — for a settled burst. See also
  [[roll-probe-camera-vs-scene]] and [[crest-height-shortfall]].
