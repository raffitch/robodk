---
name: radial-trim-fixed-point
description: "_radial_trim now iterates its tightest band to a fixed point; the schedule used to stop while the circle fit was still moving, letting a solid board patch ride the last band out to r+band"
metadata:
  node_type: memory
  type: project
---

`_radial_trim`'s schedule `[15, 12, 10]` **stopped iterating while its circle fit was still
moving**. Its own premise is "the first fit is biased by the contamination and walks onto
the ring as the bands tighten" — and nothing checked it had arrived. Measured: the last band
selected against a circle **2.3 mm off centre**, while the fit computed *from that very
selection* had already moved to 1.3 mm. A 10 mm band about the stale circle reaches r 73 on
a 60 mm ring; about the settled one, r 71.

Fixed (merged `50d0b34`): after the schedule finishes tightening, repeat the tightest band
until the selection **MASK** stops changing (mask, not size — selection is from the full
cloud each pass, so it is not monotone), bounded by `_RADIAL_TRIM_SETTLE_PASSES = 12`.
Settling runs only if the schedule completed, so an early bail-out keeps its old fallback.
New count `radial_trim_settle_passes`.

**Why the hole was invisible:** `_board_bias_patch` sampled at `step_mm=1.0` while
`RingSpec.surface_points` samples at 0.25. At 300 mm one colour pixel spans 0.34 mm, so the
1 mm patch filled only **12% of its own footprint** in the depth image — it rendered as a
sieve, and the chain rejected it for being holey. Solid (0.25 mm = 100% fill) it leaked at
BOTH voxel sizes. Fixture default is now 0.25 with a structural test pinning the fill.

**How to apply:**
- Settle passes measured: 2–5 on the synthetic, **0–1 on all eleven archived cell takes** —
  a take whose fit had arrived selects the same set and stops on the first pass.
- Golden archive: 7 of 11 bit-identical, 4 moved by <=1-2 points in ~3100; largest radius
  change **0.0101 mm** against a take-to-take sigma of 0.072. Spread got slightly TIGHTER.
- **RESIDUAL, verified independently and worse than first reported:** the leak returns at
  patch half-height **±22 mm** (r.max 72.09), not only ±30 as the subagent said. So the fix
  clears ±14 mm with ~0.9 mm margin and the boundary is ~±20 mm. Contamination big enough to
  move the FIXED POINT is `_radial_trim`'s contract, not a convergence bug.
- **Rejected on measurement, do not re-derive:** a per-angular-bin radial-span rule. Max
  span/bead is 1.88 on honest archived takes vs 2.02–2.06 on the leaking synthetic — a 7%
  gap across 17 clouds, so any threshold fires on real data first.
- Do NOT re-propose clamping `_rasterize`'s dilation; reverted earlier on a real frame.

Related: [[extrusion-voxel-downsample]], [[first-live-take-board-halo]],
[[deposit-segmentation-spec-plan]].
