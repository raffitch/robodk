---
name: paired-roll-capture
description: "One press now captures each frame twice from one trip out (vertical + camera rolled 90°), default ON across the whole sequence. Shipped 2026-09-01, run on the cell once, fixes unverified."
metadata: 
  node_type: memory
  type: project
  originSessionId: 29056919-c652-4e99-b9ca-e8ca5b163359
  modified: 2026-09-01T13:08:32.496Z
---

Task page: **`docs/paired-roll-capture-handoff.md`** (linked from the debug map).
Science: `docs/ring2-scan-handoff.md`. Shipped at `033c70b`/`20c186e`, 2026-09-01.

The Extrusion measure AND characterize paths take each frame twice from **one
trip out** — once as the pose generator chooses ("vertical"), once with the
camera rolled 90° about its optical axis ("horizontal"). UI checkbox defaults
**ON** and covers the whole guided sequence; the operator asked for it as a
property of the run, not a per-press choice.

**Acceptance check — a label is NEVER proof.** Read `provenance.T_work_camera`
per take. Valid only if: both views read **tilt 0.00°**, identical camera xyz,
and baselines differ by 90° (179.8 → 89.8 *or* 269.8 — the same axis).

**The failure mode to design against is a silent viewpoint substitution.** It
bit twice in one day: (1) the tilt ladder ran for a commanded roll, so a refused
`roll 90 / tilt 0` fell through to `tilt 10 / azimuth 270`, moving the camera
52 mm sideways and 10° off nadir and collapsing the substrate separation; (2)
`inspection_roll_candidates_deg` is a fallback list whose first entry (0°) always
wins, so a roll merely *listed* there is never used. A commanded roll now gets
tilt 0 / azimuth 0 and only `[θ, θ−180]` (same baseline axis, reversed).

**A commanded roll lands on A4, NOT A6 — the camera's optical axis is not the
flange Z.** This cost two rounds of "no reachable pose" that read like the arm
couldn't do it, when the arm rotates 90° about Z trivially and our own cap was
refusing it. Two separate filters, fixed in order: (1) `allow_wrist_flip` —
`solve_joints_on_neutral_branch` demanded the neutral `JointsConfig` triple
*including* the wrist-flip flag, which a 90° tool-axis roll legitimately changes
(`033c70b`); (2) the real one, the **|ΔA4| magnitude cap** (`a61dcda`).
`wrist_allowance_deg` capped at `|roll| + 15 = 105` on the premise that a roll
about a nadir optical axis costs ~`|roll|` of **A6**. Measured on the only rolled
pose that ever solved here (`165716-8c5ee676/characterize-02`, roll 90 at tilt 10/
az 270): **A4 −102.54°, A6 +3.97°** — ~1.15× the commanded angle, on A4, clearing
the cap by 2.5°. Tilt 0 costs more, so every branch was filtered out. That
relaxing the flip FLAG did not help is what PROVES the magnitude bound was
binding. Margin 15 → **60** (cap 150); a 180° flip is still refused, front/rear
and elbow stay locked, collisions + `program_neutral_wrist_report` unchanged.
**MEASURED 2026-09-01 17:43** (`174339-36ba48cf/layer-001-take02`, roll 90 at
tilt 0): **A4 -108.45, A5 -20.46, A6 +17.38** — 1.20x the commanded roll, on A4,
and only 3.45 deg past the old 105 cap, which is exactly why it read as "the arm
can't". The new cap (150) clears it by 41 deg; roll 90 now solves and ran 8/8.
`tasni.config.json` is git-ignored, so only the `core/config.py` default ships.

**Diagnosing this class of refusal:** the rejection string cannot tell you which
filter fired — "no IK solution on the neutral arm branch within ±N deg" covers
no-IK, wrong config triple, and the magnitude bound alike. The archived
`rejected[]` list plus `axis_4/5/6_rotation_deg` on any pose that DID solve is
the evidence; a private headless RoboDK is not — it stalled with flat CPU for
12 min, most likely contending with the operator's open GUI for the licence.

**A characterization seeds the recipe**, so a paired one applies its VERTICAL
view — applying the rolled one would derive radius/centre/height from the
orientation under test. Old sessions have no `orientation` and keep last-one.

Related: [[ring2-dropout-spatial-filter-cleared]], [[path-completeness-not-stable]],
[[measure-job-uncancellable-hang]], [[restart-tasni-backend-after-code-edits]].
