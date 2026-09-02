---
name: depth-pimples-census-and-preset-landmine
description: "Why flat scans read pimply (static not temporal noise; texture gating off, 9x9 census), and the committed as-found preset that would silently undo R2's 0.1 mm depth words if loaded."
metadata: 
  node_type: memory
  type: project
  originSessionId: b02d334f-51f0-422b-ac1b-27adbe0cfeed
  modified: 2026-08-31T14:21:41.822Z
---

2026-08-31, from reading `server/presets/custom-as-found-2026-08-29.json` (88 params,
structure `['advanced_mode']['parameters']`) against the measured numbers in
`docs/realsense-quality-headroom-2026-08-30.md`.

**LANDMINE — do not `load_json()` the committed as-found preset.** It carries
`param-depthunits = 1000` / `param-zunits = 1000`, captured 2026-08-29 *before* R2
landed. Loading it to "restore as found" resets the device to 1 mm depth words and
silently un-does protocol 2. Nothing would complain: `depth_unit_mm` is read back
from the device, so the greeting would honestly report 1.0. Today the file is only
ever written (`server/rs_config.py:144`), never read, so nothing is broken — it is a
trap for whoever does R4.2. Prefer the typed advanced-mode setters over `load_json`
for any single-knob A/B.

**The pimples are STATIC, not jitter.** Measured: board plane RMS 1.176 mm (spatial,
single frame) vs temporal sigma 0.130 mm (frame-to-frame, same pixel) at ~447 mm.
9:1. So anything that averages *in time at one pose* cannot help — `frames_per_pose`,
dwell, and laser power (swept 90-360, RMS flat inside 1.16-1.19 mm) are all closed.

Mechanism candidate, now the prime suspect: `param-texturecountthresh = 0` and
`param-texturedifferencethresh = 0` — texture gating is **fully off**, so the matcher
must emit a disparity even where there is nothing to match, and settles the same way
every frame. Note the HA result does NOT close this: High Accuracy also raises
secondPeakDelta and the score thresholds, so a texture-only change is a different
experiment.

**Census window is the size of the ring.** `param-censususize/vsize = 9` (9x9), depth
fx = 637, so lateral px = Z/637: 9 px spans 4.24 mm at 300 mm and 6.32 mm at 447 mm
against a ~9.5 mm bead — 45% and 67% of its width. Window-based stereo rounds convex
ridges, which is a *multiplicative* crest bias and matches the observed regression
slope 0.905 (r=0.97, ~1.5 mm mean shortfall). Prediction that discriminates it:
fractional shortfall should grow with standoff by 1.49x (300 -> 447 mm). Rule out the
cheaper host-side explanation first, offline from the archive with no robot: the
extrusion voxel downsample (`tasni/core/config.py:855`) averages across a convex crest
— re-run at 0.5 mm / none and see if the shortfall shrinks.

Host-side amplifiers of the same noise: TSDF voxel = `standoff_mm * 0.003` clamped
[1,2] mm (`config.py:478`) = 1.34 mm at 447 mm, i.e. ~= the noise amplitude, so
marching cubes resolves the noise; and `fuse_views` (`modules/scan/reconstruct.py:145`)
integrates every view with uniform weight despite incidence costing ~4x what distance
costs.

**RESOLVED: there is a distance-INDEPENDENT noise floor of ~0.66 mm.** The
2026-08-13 characterization has FIVE standoffs, not the three the audit quotes:
310/400/498/599/795 -> 0.9343/0.9819/1.1152/1.5118/2.0491 mm plane RMS. Segment
exponents 0.195, 0.581, 1.648, 1.074 — not a power law, but a floor plus a Z^2 term.
Least squares over all five: `sigma = 0.662 + 2.196e-6 * Z^2` mm (residual RMS
~0.06), crossover at Z = 549 mm. **At 300 mm the floor is 77% of the noise**, so
moving closer cannot touch it below ~550 mm; 310 -> 400 mm costs only 5%.

Two things this kills. (1) The Z^1.72 flag (0.593@300 substrate sigma vs 1.176@447
plane RMS) was a METRIC artifact — the fit predicts 0.860 at 300, so the
post-segmentation substrate sigma is simply more optimistic than a RANSAC board RMS.
Do not treat it as a distance effect. (2) The hole_filling explanation for the
apparent flatness was wrong: the fit predicts 1.101 at 447 and the headroom study
measured 1.176 there under protocol 2 with no hole filling — 7% apart, so the
pipeline change barely moved plane RMS. R5's fabrication claim still stands
(`coverage_frac` is exactly 1.0 at all five distances) but it was not inflating the
flatness numbers.

Why this matters most: disparity quantisation is pure Z^2 and CANNOT produce a
distance-independent floor. A 0.66 mm term that ignores range is a matcher-side
artifact — independent corroboration of the 9:1 static/temporal split, from a
different pipeline 18 days earlier.

Also ruled out (peer tested it 2026-08-31): the voxel downsample is NOT the crest
shortfall — under 0.15 mm across 1.0/0.5/0.25 mm and the wrong sign, and the ruler
comparison never went through the voxel anyway. But finer voxel DOES rescue
completeness on a marginal ring (0.875 -> 0.993 at 1.0 -> 0.5 mm), so the
"ring reads open" failure is host-side, not the camera.

**No board take with a known-height feature at a second standoff exists** in either
characterization file (2026-08-13 is a flat board; 2026-08-29 is a single standoff,
one trial), so the census ridge-rounding discriminator needs a new capture: a known
step at ~300 and ~450 mm.

No zero-code way to read achieved depth exposure per capture: the greeting's
`achieved` (`server_unicast_syncronous.py:1180`) carries laser/preset/emitter/depth_units
only. The instrument is ~6 lines in `getFrames` (`:867-878`) reading
`actual_exposure` / `gain_level` off the frame metadata. Depth AE is ON
(`controls-autoexposure-auto = True`, sitting at 8500 us, gain 16).

Related: [[first-live-take-board-halo]], [[cell-characterization-2026-08-13]],
[[scan-chain-audit-2026-08-14]], [[extrusion-take-figures]]
