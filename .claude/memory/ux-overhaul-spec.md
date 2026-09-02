---
name: ux-overhaul-spec
description: "UX/UI overhaul of the tasni web app — design spec written 2026-08-28 on branch ux-overhaul, revised twice after a second-agent review, awaiting operator approval before writing-plans"
metadata: 
  node_type: memory
  type: project
  originSessionId: 95382621-a7f2-4314-a51f-8cb97ce33845
  modified: 2026-08-28T07:58:37.672Z
---

2026-08-28: operator said the app's UX/UI + user journey is "all over the place" and
asked for a design spec. Written + pushed as
`docs/superpowers/specs/2026-08-28-ux-workflow-shell-design.md` on branch
`ux-overhaul` (draft `c48f513`; review round 1 `f5b1b2a`; round 2 `b0c4530`; round 3 `3f26c7f`).
All 11 review findings verified in code + accepted (review log in spec §11).
**Status: APPROVED by the operator 2026-08-28 ("proceed with the implementation
plan"). Phase 0 plan written:
`docs/superpowers/plans/2026-08-28-ux-phase0-platform-foundation.md` (11 tasks incl.
5a CellArbiter, TDD, executes on branch `ux-phase0` off `ux-overhaul`). Revised
once after a second-agent plan review (R13–R19 in spec §11: running-before-worker
+ `/status` reconcile, stream resume, arbiter, strict manual-link check, truthful
`ready`, reconciler clears `running`, more tests, no page auto-connect, spec
status only after cell validation). Plan review round 2 also folded in (R20–R24:
`applied` calibration shown independent of job history + Re-apply via
`POST /apply {run_id}`, no-expiry policy, restart tests, link checks inside the
arbiter, camera lease through a re-entrant arbiter). Next: get the
reviewer's/operator's go, then execute (subagent-driven-development), then write
the phase 1 plan.**

Core design: stepped workflow shell per module (pure `deriveSteps` → `StepStatus`
+ separate `selectedStepId`, shared Rail/Step/RunControls/MotionConfirmDialog/
LogPanel/GuidePane), Dashboard readiness strip (recorded vs present), copy table.
Phase 0 = module-scoped job events (`module`, `job_id`, `kind`; live previews
`stream_id`) + job history per (module, kind) with `/status` returning workflow
fields separately from `running` + station-only `POST /api/rdk/connect` (409/lock) +
`POST /api/rdk/link` auto-called on Aim/Survey with a server-side live-pose gate on
`/poses/generate` + `/surface/lock` + `GET /api/readiness`.

**Why:** three modules were built as three different apps; gating shown only as
greyed buttons + hints; spec vocabulary leaked into UI; events untagged so pages
adopt foreign jobs' progress.

**How to apply:** Phase 3 (Extrusion rewire + ring-stack tab) must NOT start before
the paper's ring-stack cell run (deadline 1 Sep 2026) — see
[[pfh-paper-ring-stack-experiment]]. No job/motion logic changes in any phase.
**Lesson (R9):** the real-robot driver link is NOT a status convenience — it makes
the RoboDK model track the arm, so the Create-targets seed is the ACTUAL pose
(`config.py:132`, `calibration/service.py:303`, `scan/service.py:2286`). Never
propose dropping it; gate on `pose_live` instead.
