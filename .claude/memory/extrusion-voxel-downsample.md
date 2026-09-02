---
name: extrusion-voxel-downsample
description: "The extrusion voxel went 1.0 -> 0.5 mm on 2026-08-31 because 1 mm was deleting marginal deposits and reporting continuous rings as open; neutral on good takes, worse on partial ones"
metadata:
  node_type: memory
  type: project
---

`ExtrusionConfig.voxel_size_m` is **0.5 mm** (was 1.0, was 2.0). Each step down was the same
argument: 2.0 -> 1.0 at `f662dae` when protocol 2 brought 0.1 mm depth words; 1.0 -> 0.5 on
2026-08-31 because 1 mm was also DELETING marginal deposits.

**The evidence, in full, including what it costs:**
- the marginal cell take: completeness 0.8752 -> **0.9925**, invalid -> valid, no camera change
- 8 valid layer-1 archive takes: +/-0.0006 completeness, both directions. Neutral.
- 3 partial layer-2 takes: 0.6252->0.5964, 0.6361->0.6325, 0.6137->0.5860. **Consistently
  worse** (3/3, not noise). They stay invalid either way. Likeliest reading: 1 mm was
  BRIDGING their genuine gaps, so 0.5 is the more honest -- inference on n=3, recorded not relied on.
- crest height max: <0.15 mm across 1.0/0.5/0.25 -- this is what rules the voxel OUT as an
  explanation of the 1.5 mm shortfall in [[crest-height-shortfall]]
- cost: +0.5 s/take (2.78 -> 3.32), after_voxel 3814 -> 10096

**How to apply:**
- The regression test is ARCHIVE-GATED (`test_extrusion_golden.py::test_a_marginal_deposit_
  survives_the_voxel_downsample`) against `runs/extrusion/20260831-173544-24d21bab`. It reads
  the archive for depth/pose but builds the config from SHIPPED DEFAULTS, because the archived
  payload carries the 1 mm voxel that caused the failure.
- **`tests/extrusion_synthetic.py` cannot reproduce this.** Tried a graded thin sector across
  several heights and noise levels; the rendered cloud is too clean and dense to lose raster
  connectivity. It is not a stand-in for sparse-marginal behaviour.
- The old pin `test_default_voxel_is_1_mm_...` asserted equality with its whole rationale in
  its NAME. It would have blocked this fix without saying what it protected. Replaced with the
  BOUND it was actually guarding (<= 0.001) plus the value. **Watch for other pins shaped
  like that in this repo.**

Related: [[crest-height-shortfall]], [[first-live-take-board-halo]],
[[deposit-segmentation-spec-plan]].
