---
name: ring2-dropout-spatial-filter-cleared
description: "The ring-2 dropout is a stable anisotropy about the stereo baseline (along/across ~2.3); the spatial filter is falsified as its cause, and ScanAtRing2 turned out to BE the automatic pose."
metadata: 
  node_type: memory
  type: project
  originSessionId: 29056919-c652-4e99-b9ca-e8ca5b163359
  modified: 2026-09-01T12:16:30.791Z
---

2026-09-01 cell run, `docs/ring2-scan-handoff.md` §3.5 is the record.

**Two levers closed, no robot time left to spend on either:**

1. **Pose — dead.** `ScanAtRing2` is **bit-identical** to the automatic layer-2
   inspection pose: 0.000 mm, 0.000°, same 179.8° baseline. The taught target
   was never a new viewpoint, so the 13:13 layer-002 takes already were that
   capture. Check a taught target against the archived `T_work_camera` before
   planning an excursion around it.
2. **Spatial filter — falsified.** Proper A-B-A on an untouched stack via the
   runtime SET: `spatial=0` returned 4% FEWER points (7684 vs 8023), left the
   anisotropy intact (2.18–2.28 vs 2.25–2.35), and moved no dead sector. A1 vs
   A2 drift control passed. The `smooth_delta=20` vs "<4 disparity px" story is
   dead.

**What is real:** the dropout is a reproducible anisotropy about the stereo
baseline — yield along it ~2.3x yield across it — with dead sectors at 120–130°,
250–270°, 290–300°. Ring 1 alone on the same stack reads 0.91–0.95 (flat), so it
appears only when the stack gets tall. It reproduced across 4.5 hours, a backend
restart and 16 takes. In the dead sectors the camera still returns points; they
read at SUBSTRATE height, so the crest is unresolved rather than the data
missing. Brightness does not explain it (the deadest sector is one of the
brightest).

**Arming the roll: `py -3.10 tools/inspection_roll.py 90`** (added `12fcd11`;
`--disarm` restores the ladder). Three traps it guards, each of which yields a
run that looks fine and answers nothing: the default
`inspection_roll_candidates_deg` is a FALLBACK LADDER `[0,180,90,270]` and the
generator takes the first candidate that solves, so **adding 90 to the list does
nothing** — roll 0 always wins; a two-entry list like `[90, 0]` turns a refused
roll into a **silent roll-0 capture**; and the config is read at startup, so
arming does nothing until a restart. 90° also lands exactly ON
`max_tool_axis_spin_deg`'s default 90 (a nadir roll ≈ that much A6, gate is
`abs(delta) > limit`) — put headroom in the TRIAL's setup
(`maximum_tool_axis_spin_deg` ≈ 110), never the global default, or use 85°.

**Still open — and now the only decisive instrument:** the controlled camera
ROLL. The 2026-08-30 two-ring stack read ratio 1.09 at the same baseline, so the
baseline alone cannot own the anisotropy; nothing in the archive separates
camera-locked from scene-locked. Prediction to test: a roll of θ moves the dead
sectors by θ and holds the ratio at ~2.3 about the NEW baseline (camera-locked),
versus sectors staying put in the work frame and the ratio collapsing toward 1.0
(scene-locked). Needs `inspection_roll_candidates_deg` + a backend restart; keep
candidates well inside `max_tool_axis_spin_deg` (90 is ON the limit and was
refused before) — 60° is ample.

Related: [[path-completeness-not-stable]], [[roll-probe-camera-vs-scene]],
[[realsense-runtime-set]], [[measure-job-uncancellable-hang]].
