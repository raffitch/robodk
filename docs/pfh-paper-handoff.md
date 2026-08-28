# PFH paper — handoff (ring-stack measure-only evidence)

**Deadline: 1 September 2026.** Written 2026-08-28. This is the single page for the
paper task: what the paper is still missing, what already exists, and the exact order
to get the rest. Everything else is background:

- Design of the experiment: `docs/superpowers/specs/2026-08-27-ring-stack-measure-only-design.md`
- Module state and traps: `docs/extrusion-current-handoff.md`
- Cell-run status: `AGENTS.md`

## 1. What the paper needs

Prototypes for Humanity 2026 short paper, *"Real-time Error Mapping, Correction, and
Archival …"* (Tchakerian / Hayek / Daneluzzo). The co-author package
(`C:\Users\User\Desktop\desktop\RoboArch_to_PFH_Coauthor_Package\`) says the text is
written and compliant at 1,887 / 2,000 words and is waiting on numbers, under the
authors' rule: **do not invent or estimate values that were not measured.**

| # | Number | Status | Comes from |
|---|---|---|---|
| 1 | Count of recipes / trials in the legacy Firebase archive | **out of scope** — not in this repo | — |
| 2 | Geometric deviation nominal ↔ extracted centreline: mean abs / RMS / max | ❌ **needs the cell run** | `paper-summary` deviation columns |
| 2b | **Detection error**: how well a known introduced offset is recovered | ❌ **needs the cell run** | `paper-summary` *detection error* column |
| 3 | Scan-to-feedback time, mean ± sd over ~10 runs | ⚠️ one sample (3,124 ms); needs ≥ 10 | `acquisition_to_path_ms` |
| — | Path-extraction success X/N | ⚠️ 1/1 so far | `valid` count |
| — | Layer height / bead width variation | ✅ **have it** (6.1 mm mean, 2.9–11.0; bead 9.9 mm) | `geometry` in the manifest |

### The wording constraint — do not lose this

The rings are **hand-placed dried beads**, not a print. Nothing is extruded, no valve
switches, no material recipe is recorded. So the paper can claim a **controlled
validation of the sensing-and-comparison chain against a known introduced offset** —
**not** the deposition deviation of a printed cylinder. `paper_summary()` already
phrases it that way and the figure captions carry it too; keep it.

### The error floor — do not claim below it

Hand-eye calibration verdict is `borderline`, board consistency **1.26 mm**, work-plane
RMS 1.39 mm. Do not report sub-millimetre accuracy on top of that.

## 2. What already exists

**A first real capture succeeded 2026-08-28**: `runs/extrusion/20260828-192115-47fb78ea`
(300 mm standoff, collisions OFF). Characterize picked the ring over the ChArUco board
residual (r 40.5 mm, centre (197.5, 152.5), bead 10.4 mm) and a separate Measure re-found
it 0.5 mm away — that repeatability is itself evidence.

**Figures are done** (`tasni/modules/extrusion/figures.py`, merged `ba8d2b3`). Four per
take plus one per trial, 300 dpi PNG **and vector PDF**, rendered from the archive with
no robot. They draw automatically on every take from now on, and older takes render on
first view.

| Figure | Use in the paper |
|---|---|
| `plan` | the headline: cloud + extracted centreline + nominal ring, mm axes, scale bar |
| `heightmap` | bird's-eye relief with a z colourbar — shows the wavy bead at a glance |
| `iso` | oblique cloud + centreline, vertical exaggeration stated on the axis |
| `profile` | unrolled z(θ) and Δr(θ); a pure shift shows as a clean cosine |
| `stack` | per trial: every layer's latest take, plan + oblique — where offsets read |

Paths: `runs/extrusion/<trial>/layer-NNN/figures/*.pdf` and
`runs/extrusion/<trial>/figures/stack.pdf`.

## 3. What is left — the cell run

**The one thing that went wrong last time: *Apply to recipe & placement* was skipped**
between Characterize and Measure. Layer-001's 15.38 mm centre offset and 11.31 mm RMS
are therefore the ring measured against a stale plan — **an artifact, not a result.**
Do not put that number, or `layer-001`'s plan/profile figures, in the paper.

Restart the backend first (it caches imported modules — check `/api/health` →
`build.stale`), then, in the Extrusion page:

0. **Set Layers to 3** before anything else, and press **New session**. The 2026-08-28
   trial ran with `layer_count: 1`, so *Measure layer 2* would be refused
   (`layer_index must be 1..1`). *Apply to recipe & placement* keeps whatever layer
   count the recipe already has — it only rewrites radius, bead, layer height and centre.
1. Scan surface applied → **Center on scanned surface** → **Generate**.
2. Place ring 1 within ~50 mm of the table centre → **Characterize ring**.
3. **Apply to recipe & placement** → **Generate**. ← *the step that was missed*
4. **Noise floor**: Measure layer 1 **five times** without touching the ring.
5. **Placement repeatability**: lift and re-place ring 1 **three times**, measure each.
6. **Stack**: ring 2 placed true → Measure L2 (×3); ring 3 true → Measure L3 (×3).
7. **Introduced offsets — on the TOP ring only.** Type the value into *introduced offset
   X/Y* **before** pressing Measure, so ground truth is archived beside the result.
   Shift the top ring **10 mm** along frame +X → Measure ×3; then **15 mm** → Measure ×3.
   Optional: prop one side for a tilt case.
8. **Paper summary** → copy the Markdown block.

### Why displacements go on the top ring, and why three takes

- **Top ring only.** Layer N's ROI floor is the *latest take* of layer N−1
  (`MeasureSession.floor_profile`). Displacing a ring that something else is measured
  on top of corrupts that floor for every take above it. Displace the highest ring in
  the stack and nothing downstream is affected.
- **Three takes per condition, not one.** `paper-summary` reports mean ± sd per
  condition; a single take has no sd, and requirement #3 needs ≥ 12 measurements in
  total anyway.

### Getting the ground truth right — the weakest link

"Shift it 10 mm" done by eye is the least trustworthy number in the experiment, and
every detection-error figure is measured against it. The work frame came from the
board rectangle, so **its axes are parallel to the board edges and the ChArUco squares
are a ruler**: slide the ring by exactly one square pitch along an edge and type that
pitch. Take one throwaway measurement first to learn which edge is frame **+X** and
its sign (the reported `center_offset_mm` tells you), then re-take with the annotation
right — the summary groups by what you typed, so a sign error puts good data in a
mislabelled group.

### Watch this on layer 2's first take

Layer N keeps only points above the measured top of layer N−1 plus
`layer_floor_margin_mm` (2.0). These rings run **2.9–11.0 mm** tall, so where ring 2
sits only ~3 mm above ring 1 there is under 1 mm of margin and its low stretches can
be clipped — showing up as reduced `path_completeness` or an angular gap, i.e.
`valid: false`. The synthetic proof used a *uniform* ring, so this case is untested on
real geometry. **Check completeness on L2 take 1 before continuing.** If it clips,
lower `extrusion.layer_floor_margin_mm` and reprocess — every take archives its raw
RGB-D, so no cell time is lost.

Constraints and expectations:

- Keep every offset **≤ 25 mm** — the radial ROI is ±30 mm.
- For a pure shift `d`: centre offset ≈ `d`, max ≈ `d`, mean ≈ `0.64 d`, RMS ≈ `0.71 d`.
  A 10 mm shift should read ≈ 10 / 10 / 6.4 / 7.1 mm. **If it does not, stop and
  investigate — that relation is the built-in sanity check.** `paper-summary` now checks
  it for you and prints a `WARNING` naming which statistic disagrees.
- ≥ 12 measurements total gives requirement #3 its mean ± sd.
- 300 mm is the sensor floor at 1280×720 (MinZ ≈ 280 mm). Closer needs a lower depth
  profile on the Jetson — a separate, deliberate change, not something to try mid-run.
  The camera already **climbs with the stack**: it aims at the cylinder axis at the top
  of the layer being measured and holds 300 mm above *that*, so L1/L2/L3 are measured
  from the same standoff, not from a fixed Z.
- Collisions default OFF for measure-only: the hand-placed stack is not in the station
  model. IK/reachability screening still runs.

## 4. After the run

- Figures for every take appear automatically; pick the ones the paper wants and use
  the **PDF**. A take with an introduced offset also draws the **ground truth ring**
  (nominal + that offset, teal dash-dot) in `plan` and in the trial `stack`, so the
  figure shows the extracted centreline landing on where the ring was actually moved
  to rather than merely displaced from nominal.
- **Paper summary** gives the deviation table grouped by introduced offset, timing
  mean ± sd, height/bead stats and valid X/N, already worded correctly. It also reports
  the **detection error** — `|measured centre offset − the offset you typed|` — which is
  the claim the paper actually makes, and a sentence per condition ("a 10.0 mm introduced
  offset was recovered as 10.05 ± 0.28 mm"). A take archived before that field existed
  is left out of the score rather than counted as a perfect measurement.
- Copy the first good capture's `color.png` + `depth.npy` into
  `tests/fixtures/extrusion/ring1/` as a regression fixture if it is better than the
  one already there.
- Discard every `runs/extrusion/20260828-*-f088cf48` trial: those predate the ring
  selector fix and measure an empty board. Keep their raw frames as evidence only.

## 5. Known-good numbers you can already cite

From `20260828-192115-47fb78ea` (characterization + one measure):

- Ring, characterized from its own scan: **radius 40.5 mm**, bead **10.4 mm**,
  height **2.9–10.8 mm**.
- Capture-to-capture repeatability of the ring centre: **0.5 mm**.
- Acquisition → reconstructed 3-D path: **3,124 ms** (capture 2,874 + processing 250).
  *One sample — needs ~10 for mean ± sd.*
- Layer height along the ring: mean **6.1 mm**, min 2.9, max 11.0; bead footprint
  **9.9 mm**.
- Path extraction: **1/1 valid**, completeness 0.99, max angular gap 2.9°.
