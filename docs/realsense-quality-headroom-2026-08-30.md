# RealSense depth-quality headroom study, 2026-08-30

**Question.** With protocol 2 live (raw unaligned depth, 0.1 mm words, 1280x720
depth / 1920x1080 colour, filter chain `threshold(0.15,1.5) -> disparity ->
spatial -> temporal -> disparity_inv`), is the D435i on this Jetson leaving
depth quality on the table at the operator's actual 300-500 mm ring-measurement
standoff, or is the current configuration (`laser_power=150`, `visual_preset=0`
Custom, `depth_units=0.0001`) already close to what the sensor can give here?

**Answer, in one line.** No usable headroom in laser power or depth resolution;
a real but modest and *unvalidated* candidate in `visual_preset=Medium Density`
for ring-crest fidelity specifically; disparity shift is actively harmful at
this standoff. Full numbers and reasoning below.

This is a measurement study only. No production config, `server/`, or systemd
unit was changed. The device was restored to its as-found state and the
read-back proving that is at the bottom of this document.

## Scene

The camera looked at a printed ChArUco calibration board with a hand-placed
extrusion-bead ring (~9.5 mm wide, rope-like, tan material, ~5 mm proud) and a
small metal washer resting on it, at **440-455 mm** standoff — squarely inside
the operator's stated 300-500 mm working range. A background wall/backdrop is
also visible around the board's edges at **~1050-1090 mm**; it plays no part
in the ring study and is excluded from every "board" and "ring" number below.

The fitted board plane's normal sits within ~0.8-1.4 degrees of the camera's
optical axis, i.e. the board was viewed almost fronto-parallel. Whole-frame
valid fraction at the as-found settings (laser 150, Custom preset) was 92.2%;
of the ~249,000 valid pixels in the 350-650 mm near band, 99.8% belong to one
flat plane at 1.18 mm RMS (see Method notes for why the near band matters).

**This is one scene at one standoff, captured once per configuration.** It is
not the operator's real print bed, lighting, or ring material, and the numbers
below should be read as *directionally* representative of the 300-500 mm
regime, not as a substitute for an on-cell A/B with calipers.

### Method notes

- **Board ROI.** The scene is bimodal: ~27% of the frame is the near board
  (~440-455 mm), ~70% is the far wall (~1050-1090 mm). An unrestricted
  whole-frame RANSAC plane fit locks onto the **wall** (bigger flat area, wins
  the inlier vote) — that was this study's first result and it was wrong for
  the question being asked, so the plane fit here is restricted to points in a
  350-650 mm z-band before RANSAC ever runs. The resulting mask (~248,600 px
  at 1280x720) is a rectangle matching the board almost exactly (measured
  z-range 441-455 mm), reused unchanged as the "board" ROI for every
  configuration at that resolution.
- **Ring ROI.** A fixed rectangular bounding box in pixel space
  (`y[249,439] x[509,710]` at 1280x720, hand-derived once from the
  colour/depth correspondence of the baseline capture — this scene does not
  move during the study). Within that box, a pixel is scored **"crest"** if
  its signed distance to the fitted board plane is more than 2.5 mm *closer*
  to the camera (roughly 2x the board's own noise floor). `ring_fill` is
  fill of the whole bounding box (mostly flat board, so it reads near 100%
  almost everywhere and is **not** a bead-fidelity metric by itself);
  `ring_crest_frac` (fraction of the box scored crest) and
  `ring_crest_height` (mean height of crest pixels, true bead ~5 mm) are the
  bead-fidelity metrics that actually matter for the paper.
- **Capture protocol**, identical for every row below: fresh filter-chain
  instances, ~3.0-3.5 s settle **and** >=40 processed warm-up frames (whichever
  is longer) for auto-exposure and the temporal filter to converge, then 10-12
  kept frames through the *same* production filter chain
  (`threshold(0.15,1.5) -> disparity -> spatial -> temporal -> disparity_inv`).
- **Exclusivity.** The camera service was stopped only for the duration of
  each single capture and restarted between every configuration (not held for
  the whole study), because another agent needed the camera mid-session for an
  unrelated scan-refusal diagnosis.

## Laser power sweep (board)

1280x720, Custom preset, 10 kept frames each. `plane RMS` = RANSAC-fit
residual on the board ROI; `temporal sigma` = per-pixel std across the 10 kept
(already temporal-filtered) frames, i.e. genuine frame-to-frame jitter, not
single-shot spatial noise.

| laser_power | valid frac (full frame) | board fill (ROI) | plane RMS (mm) | temporal sigma mean (mm) | centre-patch distinct values |
|---:|---:|---:|---:|---:|---:|
| 0 | 40.4% | 74.8% | 1.944 | 0.465 | 155 |
| 90 | 91.9% | 99.7% | 1.166 | 0.158 | 101 |
| **150 (current)** | **92.2%** | **99.9%** | **1.176** | **0.130** | **95** |
| 150 (repeat, end of session) | 92.2% | 99.9% | 1.171 | 0.133 | 95 |
| 240 | 92.3% | 99.8% | 1.161 | 0.108 | 93 |
| 300 | 92.4% | 99.8% | 1.179 | 0.102 | 100 |
| 360 | 92.4% | 99.7% | 1.186 | 0.096 | 100 |

## Laser power sweep (ring)

| laser_power | ring bbox fill | ring crest frac of bbox | ring crest mean height (mm) |
|---:|---:|---:|---:|
| 0 | 99.3% | 11.9% | 4.84 |
| 90 | 100% | 9.1% | 4.34 |
| **150 (current)** | **100%** | **9.4%** | **4.26** |
| 150 (repeat) | 100% | 9.2% | 4.24 |
| 240 | 100% | 9.2% | 4.17 |
| 300 | 100.0% | 9.1% | 4.15 |
| 360 | 100% | 9.0% | 4.17 |

**Reading it.** Between 90 and 360 the board plane RMS moves inside
1.16-1.19 mm — a 0.025 mm spread, i.e. noise-level (the 150-repeat differs from
the original 150 by 0.005 mm, that's the test-retest floor of this method).
Ring crest height likewise sits in a flat 4.15-4.34 mm band with no monotonic
trend. The **one real, monotonic laser effect** is temporal sigma on the
board: 0.130 mm at 150 down to 0.096 mm at 360, a genuine 26% reduction in
frame-to-frame jitter — but that is already two orders of magnitude below the
~5 mm bead scale, so it changes nothing for ring measurement. Below 150,
laser=0 breaks the board badly (RMS +65%, fill -25 points, valid fraction
less than half) but is *not* worse for the ring — crest height is actually
highest at 4.84 mm with laser off, consistent with the bead's own surface
texture giving passive stereo something to match, unlike the blank board.

## Visual preset sweep

`rs.rs400_visual_preset` on this build: `custom=0, default=1, hand=2,
high_accuracy=3, high_density=4, medium_density=5`. `laser_power` was set to
150 immediately before each preset load; "laser after" shows what the preset
itself left it at (presets can silently rewrite laser power).

| preset | laser after | valid frac (full) | board fill | board plane RMS (mm) | ring fill | ring crest frac | ring crest height (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Custom (0, current)** | 150 | 92.2% | 99.9% | **1.176** | 100% | 9.4% | **4.26** |
| High Accuracy (3) | 150 | 86.5% | 97.0% | 1.069 (-9%) | 99.9% | **8.5% (worst)** | 4.20 |
| High Density (4) | 150 | 92.8% | 98.7% | 1.866 (+59%) | 99.0% | 9.9% | **4.49 (best)** |
| Medium Density (5) | 150 | 91.2% | 99.4% | 1.278 (+9%) | 99.8% | **10.0% (best)** | 4.39 |

**Reading it.** This is the one factor that shows a real, non-noise-level
trade. High Accuracy gives the flattest board (best plane RMS, -9%) but the
**worst** ring — fewer, "surer" points is exactly the wrong direction for a
curved bead near the confidence floor (crest fraction drops from 9.4% to
8.5%, the lowest of every configuration tested). It also **stalled the camera
three times** ("Frame didn't arrive within 5000") immediately after the
preset load before frames resumed — a reliability cost beyond the metrics.
High Density and Medium Density both trade board flatness for ring fidelity
(exactly the "operator's real geometry" trade the coordinator flagged), and
since the paper measures the ring, the trade should be evaluated on the ring:
Medium Density gets nearly all of High Density's ring benefit (crest frac
10.0% vs 9.9%, height 4.39 mm vs 4.49 mm) for a much smaller board-flatness
cost (RMS +9% vs +59%). **Medium Density is the one candidate in this whole
study worth a dedicated on-cell trial** (measure a real ring against calipers,
dated, reversible) — not a default flip on this evidence alone.

## Depth resolution: 1280x720 vs 848x480

Both at laser=150, Custom preset. At ~447 mm the ~9.5 mm bead spans **~13.5
px** at 1280x720 (fx=637) versus **~9.0 px** at 848x480 (fx=422).

| depth mode | board plane RMS (mm) | ring crest frac | ring crest height (mm) | board ROI pixel count |
|---|---:|---:|---:|---:|
| **1280x720 (current)** | 1.176 | 9.4% | **4.26** | 248,635 |
| 848x480 | **1.073 (-9%)** | 9.7% | 3.94 (-8%) | 109,097 |

**Reading it.** Intel's claim that 848x480 is the noise-optimal D435 binning
mode holds here — flat-surface RMS is genuinely ~9% better. But it under-samples
the bead: crest height comes back 8% short of the 1280x720 reading (further
from the true ~5 mm), consistent with the pixel-count-across-the-bead math
above. This is the trade the task brief predicted, quantified: better on the
board, worse on the thing the paper measures. **Recommendation: keep
1280x720.**

## Advanced-mode `disparityShift` (optional sweep)

As-found value: 0. Tested at laser=150, Custom preset, 1280x720. Restored via
the exact original `depth_table` object after each trial, verified.

| disparityShift | valid frac (whole frame) | board plane RMS (mm) | ring fill | ring crest height (mm) |
|---:|---:|---:|---:|---:|
| **0 (current)** | 92.2% | 1.176 | 100% | 4.26 |
| 60 | 28.2% | 1.171 (unchanged) | 100% | 4.28 (unchanged) |
| 120 | 1.1% | 260.8 (broken) | 1.6% | not measurable |

**Reading it.** `disparityShift=60` changes nothing measurable on the near
surface — board RMS and ring crest height are statistically identical to
baseline — while it wipes out validity on the far background wall entirely
(92.2% -> 28.2% whole-frame, purely because the background at ~1070 mm falls
outside the shifted working range; harmless, since the operator never uses
that background). `disparityShift=120` is a hard failure at this standoff:
the board itself falls mostly outside the shifted disparity search window
(RMS balloons to 260 mm, fill collapses to ~1-2%). **No benefit found at
~447 mm, and a real breakage risk above ~60-120; not recommended.** The audit
doc's Z^2-error argument for disparity shift is strongest at closer standoffs
(e.g. 250-300 mm) than the ~447 mm this scene offered — untested here since
the camera was not moved, per the hard constraint on this study.

## Ranked recommendation

1. **Keep `laser_power=150`.** This was the single most likely source of
   headroom going in (150/360 = 42% of maximum) and the sweep closes that
   question: no measurable board or ring benefit anywhere in 90-360, only a
   small (0.03-0.04 mm) frame-to-frame jitter reduction that is irrelevant at
   the 5 mm bead scale. Raising it would cost the dated 2026-08-13
   characterisation baseline (measured at 150) and untested projector heat
   margin, for zero measured return. **Confirmed not headroom.**
2. **Trial `visual_preset=Medium Density (5)` as a dated, reversible
   experiment**, specifically scored against a real ring with calipers before
   any adoption. It is the only setting in this study that moved the ring's
   own crest-fidelity numbers in a direction that plausibly matters (crest
   frac +6%, crest height +3%, toward the true ~5 mm) for an acceptable board
   cost (RMS +9%). High Density pushes ring fidelity slightly further but at
   nearly 6x the board-flatness cost; High Accuracy should be avoided outright
   (worst ring result of all 12 configurations, plus repeated frame stalls on
   load).
3. **Keep depth resolution at 1280x720.** 848x480 is genuinely quieter on the
   flat board (as Intel's docs claim) but measurably worse on the bead itself
   (-8% crest height) because it halves the pixels sampling the 9.5 mm bead
   width. Not worth the lateral-resolution loss for a ring-measurement study.
4. **Leave `disparityShift` alone.** No benefit at this ~440 mm standoff, and
   values above ~60 risk breaking the measurement outright at this distance. A
   closer standoff is a separate, untested question.

## Already at (or effectively at) maximum — not worth pursuing further

- **`depth_units=0.0001` (0.1 mm words).** Centre-patch distinct-value counts
  now range 89-166 across every configuration (baseline 95), well above the
  legacy 25-value ceiling this protocol replaced, and already finer than the
  D435's own theoretical stereo granularity at this range (~0.19 mm at
  450 mm per the 2026-08-29 capability audit). The word width is no longer
  the limiting factor anywhere in this sweep. Note the caveat that *more*
  distinct values is not automatically *better* — the noisiest, most broken
  configurations (laser=0: 155, disparityShift=120: 5 valid samples) also
  show unusual counts, so this metric must always be read alongside plane RMS
  / temporal sigma, never alone.
- **`emitter_enabled=1`.** Already on; laser=0 in the sweep above is the
  emitter-off case and it is clearly worse, not better, for the board (though
  not for the ring — see above).
- **Laser power above 150** (see recommendation 1): tested up to the 360
  maximum, no return found.

## Device-state restore (proof)

Final read-back off the device at the end of the study, before the service
was restarted:

```
laser_power      = 150.0    (as found)
visual_preset    = 0.0      (Custom, as found)
emitter_enabled  = 1.0      (as found)
depth_units      = 9.999999747378752e-05   (0.0001 m = 0.1 mm, as found)
disparityShift   = 0        (as found; advanced-mode depth_table fully restored
                              after both disparity-shift trials)
```

Service verified after restart:

```
$ systemctl is-active realsense-camera
active
$ ss -tln | grep :1024
LISTEN 0  5  0.0.0.0:1024  0.0.0.0:*
```

Every laser/preset/disparity change during the sweep was applied, measured,
and reverted to the 150 / Custom / 0-shift baseline **before the next
configuration**, and the camera service was stopped only for the duration of
each single capture (not held for the whole study) so another agent's
concurrent scan-diagnosis session was not blocked for longer than one capture
at a time.

## Limitations

- One scene, one standoff (~447 mm), one capture pass per configuration (10
  kept frames after settle) — not a multi-session repeatability study.
- The ring ROI is a hand-placed rectangular bounding box around the visible
  bead in the baseline capture, not a segmented annulus; `ring_fill` is a
  bbox-level dropout metric, not a bead-shape metric on its own — read it
  together with `ring_crest_frac` / `ring_crest_height`.
- Sustained thermal effects of higher laser power were not tested (each
  capture was ~10-15 s); the "no headroom" finding is about depth quality,
  not about whether 360 is thermally safe to hold for a full scan.
- The 2026-08-13 historical characterisation (0.93 mm RMS @310 mm, 0.98 mm
  @400 mm, 1.12 mm @498 mm, interpolating to ~1.05 mm at this scene's ~447 mm)
  was measured under a different acquisition pipeline (aligned depth, 1 mm
  depth units, `hole_filling` in the chain — since shown to fabricate depth,
  per the 2026-08-29 capability audit R5). This study's 1.176 mm baseline is
  ~12% higher than that interpolation; given the pipeline difference this is
  read as a directional sanity check (same order of magnitude, same board),
  not a validated match.
