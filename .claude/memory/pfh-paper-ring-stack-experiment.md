---
name: pfh-paper-ring-stack-experiment
description: "PFH paper ring-stack protocol (deadline 1 Sep 2026) — current run shape, the batching split, the side photo, and the traps"
metadata: 
  node_type: memory
  type: project
  originSessionId: 61e217e1-9a54-4c3c-a324-b23d36299d55
  modified: 2026-08-29T10:38:06.386Z
---

Paper deadline **1 Sep 2026**. Needs deviation/timing numbers from HAND-PLACED dried
rings (no print). Task page: `docs/pfh-paper-handoff.md`. Operator run card:
https://claude.ai/code/artifact/6bce70df-e8e7-408f-aac3-0b116bff8029

## The run as it stands (2026-08-29, `bd6364b` on main)

Fresh ring → characterize ring 1 → **noise floor** → **re-place** → ring 2 → ring 3 →
**ring 4 placed off-centre** → summary. Plan fixed at **4 layers**.

**Two kinds of repeat, bought differently** — this is the change that made the noise
floor worth keeping when the operator asked whether it was needed:
- `excursions` = whole trips out and back. Spread contains the **robot's re-approach**
  as well as the camera. The noise floor buys **5 on one press, unattended**.
- `repeats` = frames with the arm **PARKED** at the pose. Spread is the **sensing
  chain alone**. Every later condition buys **3 on one press**.
- Only a take that had a whole excursion to itself gets `inspection_cycle_ms`, so the
  paper's "what one inspection costs" is never divided by frames-taken-while-parked.
- `paper_summary` reports both pooled separately (`repeatability_mm`) and labels each
  condition row with **How** / **Centre spread**.
- Re-place stays one press per take — the hand has to move between them.

**Bottom-up, error on top.** The operator first asked for a top-down teardown (displace
ring 3, lift it off, displace ring 2 …) then **changed their mind and asked to
simplify**: stack three true, put a fourth on already off-centre, done. Nothing is
slid or lifted, so no ring is disturbed after it is measured.

**Consequence that needed code:** a ring placed already displaced has **no undisplaced
take of its own** to pair against, ever. `pre_shift_reference` falls back to the latest
zero-offset take of the layer **beneath** it — which is what the steel rule measured
anyway (how far ring 4 sits from ring 3). Same-layer stays first choice; layer 1 pairs
with nothing rather than inventing a reference. Scored the old way the stack's own
hand-placement error is charged to the chain (test: 0.00 paired vs 3.00 unpaired).

**Side photo** (`b94593c`): after each layer's capture, one RGB photo from the side via
**TAUGHT** targets — `neutral → TowardsSideCapture → SideCapture → photo →
TowardsSideCapture → neutral`. The approach target is mandatory in **both** directions
(the direct move bumps into cell objects; nothing in the station model knows they are
there). Once per press, filed as `side.png` with that press's last take. Measures
nothing, kept out of the cycle timing, and can never fail a measurement — a missing
target skips it *before any motion* with the reason archived.

## Traps

- **The ChArUco pitch is NOT a usable ruler here** — this A3 board's squares are 40 mm,
  past the 25 mm offset cap. Use a steel rule and type what you actually moved.
- **Offsets are conditions, not exact values.** "About 10" and "about 15" — a take falls
  to the nearer one. (Matching an exact millimetre was a bug: a step whose own text said
  "12 mm scores as well as 10" could never count that take, so it could never finish.)
- **Restart the backend before any cell run** — see [[restart-tasni-backend-after-code-edits]].
  After a restart press **Apply / "Use this ring"** first.
- Board depth noise fuses to the ring without `_radial_trim`; raising the floor gives
  confidently wrong radii. See [[extrusion-take-figures]].
- Don't run the full pytest suite — [[pytest-suite-too-slow-to-run-fully]].

Related: [[extrusion-a4-wrist-flip-fix]], [[multiview-inspection-spec]] (that plan's
derived side pose is superseded by the taught targets above).

## RUN-DAY HAZARD (2026-08-30): the branch guard is ~1 raster pixel from aborting

`a0fabca`'s bead clamp tightened the spur tolerance. On the archived takes,
`layer-001-take04` ran at `spur_limit` **15** before the clamp, **13** after — and
**12 raises `RuntimeError("branch guard exhausted")`**. The four takes of that ring
measured beads **8.308 / 8.956 / 9.004 / 8.626 mm** (0.70 mm spread), so a take
~0.35 mm below the lowest one already recorded ABORTS.

**Deliberately NOT loosened**, and don't loosen it: loosening moves toward false
ACCEPTS, which is what biased takes 1–3 of the 2026-08-29 run 0.6–0.7 mm large. An
abort is **recoverable** — the raw RGB-D is archived, so REPROCESS; do not re-run the
robot.

`da5f7a4` made the raise name the tolerance it gave up with, plus the clamped bead,
the recipe bead and the frame's measured bead. **Reading it:** measured bead close to
the recipe → genuine contamination spur, leave the guard alone; clamp pulled it well
below the recipe → you hit the margin, reprocess. Every archived take's
`measured_radius_mm` / `rms_mm` / `center_offset_norm_mm` / `bead_width_mean_mm` is
byte-identical with the clamp on or off — only `spur_guard_bead_mm` moves.

## Measurement reality at 448 mm (snapshot `runs/characterization/20260830-103249-snapshot-table`)

- Table plane RMS **1.037 mm**; temporal σ only **0.104 mm** → **spatial noise is ~10x
  temporal, so stacking more frames per pose buys nothing.** More viewpoints would.
- Noise is spatially CORRELATED: half-correlation ~7 mm → **blobs ~14 mm across**, the
  same scale as the 9.5 mm bead. It cannot be smoothed away without flattening the bead.
- Only **172° of 360°** of the ring is recovered above the 2 mm threshold. The colour
  frame shows a CLOSED ring — the gap is a SENSING gap, not a defective ring. (The
  earlier "32.4° gap + contamination spur" claim came from a different capture through
  the extrusion chain and is NOT corroborated by this snapshot — treat as open.)
- **Move to ~300 mm before capturing.** Depth noise barely improves (0.93 mm @310 vs
  1.12 mm @498, 2026-08-13), but 1 px goes 0.70 → 0.47 mm so the bead spans ~20 px not
  ~13, and the noise blobs shrink to ~9 mm.
- **Do not mix pre-protocol-2 archive takes with new ones** — the old aligned/1 mm/
  hole-filling chain carries the +2.05% scale error from [[scan-chain-audit-2026-08-14]].
- Report the CREST (height/centre/radius) + the measured noise floor, not a full bead
  cross-section. Lean on parked-vs-re-approach repeatability: relative, so the
  systematics cancel.
