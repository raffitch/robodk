---
name: workframe-two-path-plan
description: Two-path workframe survey design reviewed+revised and 17-task implementation plan written; awaiting execution by another agent
metadata: 
  node_type: memory
  type: project
  originSessionId: a3f32bc8-1aa8-49d4-98dc-4ec2f7523707
  modified: 2026-08-12T12:30:46.505Z
---

2026-08-12 (`45314f6`, pushed to origin/`calibration-improvements`): reviewed
`docs/scan-workframe-two-path-plan.md` (compact vs five-position workframe survey) —
verdict: agree with corrections. Revised the doc in place: §9 premise fixed (RoboDK DOES
mirror pendant jogging when the KUKA driver monitors — two-mode guidance instead of
pessimistic step-and-measure-only), user-specified region promoted to first release
(labeled fast path replacing the fixed 1 m² crop), edge-fit conditioning answered
(direction fine via corner-to-corner baseline; ~1–2 mm cross-capture registration floor),
Phase 0 characterization must be an in-app tool with a stored `calibration_id`, new §17
records the review outcome.

Wrote `docs/scan-workframe-implementation-plan.md`: 17 TDD tasks, 3 milestones
(A: contract/`LockedWorkframeSurvey`+classifier+user-region+UI-latch-removal+hard gates;
B: rect_fit constrained-rectangle geometry + corner evidence + five-position state
machine + `plan_rect_tour` + routes + UI panel; C: `characterize.py` + CLI + docs).
Written against exact current code names (from a full scan-module map) for execution by
another agent via superpowers:subagent-driven-development or executing-plans.

**Why:** the plan supersedes the fixed crop / double-freeze / live-outline-after-lock
behaviors in [[scan-module-status]]; §11's single locked contract targets that recurring
bug class.

**How to apply:** when implementing, follow the implementation plan task-by-task; the
`survey_*` config defaults are engineering guesses to revisit after Phase 0 runs on the
cell. Deferred: server-side crop-square sync, Phase 5 cell validation, auto corner
viewpoints, edge-midpoint recovery.
