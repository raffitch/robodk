---
name: roll-probe-camera-vs-scene
description: The camera-roll probe separates a stereo/sensor artifact from real geometry; attempt 1 (2026-08-31) was REFUSED on an uncontrolled noise floor, and the protocol was redesigned to a SYMMETRIC +/-30 deg pair (82358c8). Handoff at docs/inspection-roll-probe-handoff.md.
metadata: 
  node_type: memory
  type: project
  originSessionId: 730828c5-1611-45cf-887d-7fccf7f3c54b
  modified: 2026-09-01T05:48:17.141Z
---

Rolling the camera about its own optical axis is the CLOSEST THING to a single-variable
move separating a stereo/sensor artifact from real geometry — but it is NOT perfectly
single-variable: the IR projector rolls too, and turning its dot pattern against the
printed board moved the noise floor itself (that is what refused attempt 1, per the
probe's own comparability comment). Task page: **`docs/inspection-roll-probe-handoff.md`**
(rewritten `82358c8`, 2026-09-01).

Three open problems ride on one answer: the layer-2 dropout (~50 deg sector at
work-frame 140-190 deg), the board halo, and — the one that matters most — whether
the **STATIC** depth noise (1.176 mm spatial vs 0.130 temporal) decorrelates with
pose. That last is the assumption the entire multi-view case rests on and it has
never been measured. See [[multiview-inspection-spec]], [[layer2-stack-segmentation]],
[[depth-pimples-census-and-preset-landmine]].

**Attempt 1 (0 vs 60 deg) FAILED ON CONTROL, NOT ON SCIENCE** — and probably not on
operator care either: B's substrate sigma 0.866 vs A's 0.591 (ratio 1.46 against a 25%
gate), floor PINNED at the 2.0 mm clamp, radius 44.10 vs 42.64 on the same untouched
ring, the skirt gone from two lobes to four at 3x amplitude. The pair is on disk
(A `runs/extrusion/20260831-163156-5bf38c80/characterize-01`, B `...-171203-24d21bab/...`).

**How to apply (protocol as tightened in `95f0fb0`, doc §3 — supersedes the plain
symmetric pair):**
- **Decision gate FIRST: the `RS_SPATIAL=0` A/B** on the same stack at roll 0 — the
  only untried lever with a measured mechanism for the dropout/low crest
  (smooth_delta 20 vs <4 disparity px of relief). Greeting archives the filter's
  PRESENCE but not `RS_SPATIAL_SMOOTH_DELTA` — no delta sweep until archived.
- **A-B-A: A1 at +30, B at -30, A2 at +30** (`[30,0]`/`[-30,0]`/`[30,0]`, negatives
  legal). A1-vs-A2 is the drift control; if they disagree, B is uninterpretable.
  Never command 90 — it sits exactly on `max_tool_axis_spin_deg`'s default limit and
  was refused. The roll-0 stack takes are the work-frame reference, NOT capture A.
- **The 2026-08-31 pair was confounded twice over**: recipe changed (bead 8.9 vs
  15.0 mm) so the camera sat 6.100 mm apart in work Z (313.159 vs 319.259, VERIFIED
  from the archives) — roll is a candidate cause of B's bad floor, not established.
  Freeze the whole applied plan/recipe, not just the standoff field.
- **Set `measure_depth_fusion_frames` 10 for the diagnostic, analyze the last 5**:
  the filter chain is global and advances only inside `getFrames()`, so the FIRST
  burst at a new pose settles across ~5 frames (VERIFIED: layer-002 first burst
  sigma 0.695→0.537 monotonic; take02/03 repeat bursts flat ~0.54). Restore after.
- **If the A-B-A set still fails the sigma gates, STOP** — the noise floor is
  roll-dependent on this board and no roll pair answers the question; back to the desk
  (matte board, or the tilted-star offline A/B), not more excursions.
- Achieved roll: just run `tools/probe_roll_pair.py` on the pair — its "baselines
  differ by" line is the check (characterize records carry no `roll_deg`).
- **Never quote the axis numbers the probe prints under its own refusal.**
- The probe measures LIFTED BOARD in the skirt (halo). The dropout is an ABSENCE —
  count ROI points per 10 deg sector; extend the probe or write a sibling.
- The doc's layer-002 baseline table is an offline post-`bd455a7` reprocess; the
  archived `report.json` still holds pre-fix numbers (compl 0.29-0.36, radii
  40.6-47.9) — the archive disagreeing with the table is not the ring moving.
