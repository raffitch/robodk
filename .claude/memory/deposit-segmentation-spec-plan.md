---
name: deposit-segmentation-spec-plan
description: "Geometry-based deposit segmentation SHIPPED and merged to main (23d5bf3) — chroma gate deleted, fitted substrate + derived floor + compactness filter; offline-validated only, no robot has moved"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1ec76a04-aa68-4f22-982e-e01da48bf117
  modified: 2026-08-31T12:28:59.477Z
---

The chroma gate is gone. Deposit segmentation now works on geometry: `PlaneSubstrate`
(deterministic one-sided IRLS, `tasni/modules/extrusion/substrate.py`) fits the actual
surface per frame, the floor is derived from that frame's own measured noise
(`clamp(k*sigma)`), and `compactness_filter` rejects contamination on shape. Merged to
`main` at **23d5bf3** on 2026-08-31 (8-task plan, 20 commits). Design of record:
`docs/superpowers/specs/2026-08-30-deposit-segmentation-design.md` (§11 and §13 record
what measurement changed in the design itself).

**Validation, and its limit:** on 8 archived layer-1 cell takes the new chain gives 8/8
valid, radius mean 40.980 mm (matching the design's independent prediction exactly),
radius sigma 0.0737 — better than the design prototype's own 0.107 — with no colour input
at all. **But everything is offline reprocessing of archived frames. NO ROBOT HAS MOVED
under this chain.**

**Why:** the saturation gate's premise had inverted (colour auto-exposure runs free, so the
printed board read MORE chromatic than the clay), and the work frame was never the datum
(board sits 1.2 mm low, tilted ~0.5 deg, against a 1.5 mm threshold).

**How to apply:**
- **First live take after this: read `report["substrate"]` against the frozen golden table**
  — sigma 0.52-0.57, floor 1.55-1.70, tilt < 0.9 deg, separation margin > 2. That block
  exists to make the first live take self-auditing.
- `tests/test_extrusion_golden.py` reprocesses 11 archived takes read-only against frozen
  baselines; it is the guard for any future change to this chain, and it SKIPS on a machine
  without `runs/`.
- Retiring an `ExtrusionConfig` field needs BOTH registries
  (`RETIRED_EXTRUSION_CONFIG_KEYS` for archives, `LEGACY_CONFIG_KEYS` for the user's
  config file — the latter RAISES a startup KeyError). A structural test enforces the
  subset relation.
- One live defect is pinned `xfail(strict=True)`: an overstated recipe bead defeats the
  branch guard through `_rasterize`'s dilation (r 39.04 for a 40.0 mm ring, VALID). The
  obvious fix was measured to regress a real frame and reverted. Fixing it XPASSes and
  fails the suite, which is the signal to delete the marker.

Related: [[extrusion-raster-free-centreline]], [[pfh-paper-ring-stack-experiment]],
[[paper-timeline-not-a-constraint]], [[pytest-suite-too-slow-to-run-fully]].
