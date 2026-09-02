---
name: paper-timeline-not-a-constraint
description: Operator explicitly deprioritized the PFH paper deadline — engineering correctness first; rescan/reprocess for the paper afterwards if time allows
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1ec76a04-aa68-4f22-982e-e01da48bf117
  modified: 2026-08-31T05:37:51.886Z
---

2026-08-31, on whether the deposit-segmentation plan should wait for the paper:
"i don't care about the paper timeline and precedents, once the scan is done we can
just rescan and use that data for the paper if we have time."

**Why:** The operator values the measurement chain being right over hitting the
1 Sep 2026 deadline with the old chain. Raw RGB-D is archived per take, so
already-captured data can always be re-derived by a newer chain (read-only golden
harness); nothing forces instrument-freeze sequencing.

**How to apply:** Do not sequence or defer engineering work around the paper
deadline, and do not re-raise the instrument-consistency objection as a blocker —
the resolution is "reprocess the archive + capture new takes on the new chain" so
the paper stands on ONE chain anyway. Measurement-integrity rules are NOT relaxed:
the layer-2 golden takes stay pinned invalid, the branch guard stays tight, and the
Task 7 hard stop on the 2026-08-29 crash fixture stands. Related:
[[deposit-segmentation-spec-plan]], [[pfh-paper-ring-stack-experiment]].
