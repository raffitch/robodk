---
name: first-live-take-board-halo
description: "2026-08-31 first LIVE take under geometry segmentation aborted on the branch guard; root cause is a stereo edge halo of BOARD points at 2.9-3.8 sigma that the deleted chroma gate used to remove"
metadata:
  node_type: memory
  type: project
---

The first live take under the merged geometry-only chain (`23d5bf3`) is
`runs/extrusion/20260831-163156-5bf38c80/characterize-01` and it **aborted**:
`branch guard exhausted ... spur_limit 16 px`. Diagnosed offline from the archive
(deterministic, no robot).

**Root cause (measured, not inferred):** 32 points in `top_surface` sit at r 50-52.5 mm
around a ring of r≈42, in two patches at ~200 deg and ~353 deg — the ring's LEFT and RIGHT
extremes in the depth image, i.e. the D435i's stereo-baseline axis. They are:
- **board, not bead**: HSV S median **16**, BGR (44,48,46) — identical to bare board
  (S 19, BGR (48,51,49)); the ring itself is S **104**, BGR (51,75,97).
- **at 2.9-3.8 sigma** above the fitted substrate (sigma 0.591, 3-sigma floor 1.773 mm) —
  i.e. sitting exactly on the floor, `substrate_p99_mm` 1.658 vs `floor_mm` 1.690.
- **temporally stable** (identical nonzero counts across all 5 fused frames), so the
  per-pixel-median fusion cannot remove them. A stereo edge artifact, not speckle.

`_rasterize` dilates by bead/2 (5 px) and welds them to the ring; thinning turns each into
a 17-18 px radial twig; `spur_limit = ceil(1.5 * 10.467) = 16 px`. **Three pixels short.**
The guard is RIGHT — it refused a frame whose bead was being inflated by board.

**Why:** the deleted chroma gate was removing exactly this halo. A/B on the SAME frame:
pre-merge chain (`c3832b0^`) → VALID first attempt, r 41.780, bead 8.113, completeness
0.993, `chroma_gate_kept_fraction` 0.128. The merge's premise (board S 28 > bead S 25 on
the 2026-08-30 frames) simply does not hold on this frame — separation here is ~6:1 the
RIGHT way. Colour auto-exposure makes it frame-dependent in both directions.

**How to apply:**
- Do NOT loosen the branch guard and do NOT re-run the robot — the frame is archived and
  reprocesses deterministically. See [[pfh-paper-ring-stack-experiment]].
- `substrate_sigma_k` is capped at 3.5 by config and 3.5 still fails; raising the floor
  clamp ceiling does not help either. A 3-sigma floor on a 0.56 mm-sigma plane structurally
  cannot separate a 1.7-2.4 mm edge halo.
- `radial_trim_schedule_mm` last band 10 -> 8 makes it pass. But the reported bead width is
  a strong function of that schedule (10.0 / 9.5 / 8.8 mm for [15,12,8] / [15,10,7] /
  [12,10,8,6]) converging toward the chroma answer 8.11 — so the halo biases the
  MEASUREMENT, not just the guard. Tuning the trim hides the bias; it does not remove it.
- Repro scripts and the skeleton/halo figures were scratchpad-only; the archive is the
  durable artifact.

**ROLL PROBE RAN 2026-08-31 17:12 — INCONCLUSIVE, and it surfaced something worse.**
`runs/extrusion/20260831-171203-24d21bab/characterize-01`, roll 90 refused at the wrist
→ fell back to 60 (baseline axis 179.1 → 119.1 deg). It came back **valid=True,
warnings []** — and that is the bad news, not the good news:

| | aborted take (16:31) | "successful" take (17:12) | golden |
|---|---|---|---|
| separation_margin_mm | 1.99 | **-0.119** | > 2 |
| substrate sigma_mm | 0.591 | **0.866** | 0.52-0.57 |
| floor_mm | 1.773 | **2.000 PINNED on clamp** | 1.55-1.70 |
| substrate_p99_mm | 1.658 | **3.535** | — |
| bead_width_mm | 10.47 | **11.46** (coarse 13.77) | chroma-era 8.11 |

It cleared the branch guard on attempt 1 because the mask got FATTER (6426 px vs 4502)
and everything welded into one blob — not because the scene was clean.

**Two findings:**
1. ~~`separation_margin_mm` gates nothing~~ **FIXED + pushed `95595a8`.**
   `substrate_health()` in `processing.py` now FAULTS (raises, like the branch guard)
   at margin <= `substrate_min_separation_mm` (0.0 — a definition, not a tuned value;
   ~2 mm headroom either side) and WARNS on margin < 1.5, sigma > 0.70, or a floor
   pinned on its clamp ceiling. Warnings join the report's existing `warnings` list.
   `None` margin is deliberately NOT a fault. Verified: goldens 8/8 still valid,
   measure suite 120 passed 1 xfailed, the 17:12 take now refused by name and number,
   the 16:31 take unchanged.
2. **Roll is not a single-variable change.** The IR projector shares the body with the
   stereo pair, so rolling turns the dot pattern against the printed board as well as
   the baseline. Skirt pattern went from two clean lobes to FOUR at ~3x amplitude. The
   probe's decision rule had no comparability precondition and printed a confident
   "REAL GEOMETRY"; fixed in `7185856` to refuse when sigma ratio > 1.25 or the floor
   is pinned. **The halo question is still open.**

`tools/probe_roll_pair.py` (`f620a9e` + `7185856`, pushed to main) is the probe; it is
throwaway and nothing imports it. `tasni.config.json` (git-ignored) was left with
`inspection_roll_candidates_deg: [90, 60, 0]` — REVERT to `[0, 180, 90, 270]`.

Related: [[deposit-segmentation-spec-plan]], [[extrusion-raster-free-centreline]],
[[camera-intrinsics-distortion]].
