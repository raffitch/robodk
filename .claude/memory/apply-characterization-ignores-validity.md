---
name: apply-characterization-ignores-validity
description: "measure/apply-characterization takes the VERTICAL view even when that view came back valid:false -- the 2026-09-01 17:43 run was scored against a 50.1 mm nominal ring that should have been 42.4, centred 9 mm off."
metadata:
  node_type: memory
  type: project
---

`module.py:739` picks the **vertical** characterization on purpose (do not seed the plan
from the view under test -- see [[paired-roll-capture]]) but **never checks `valid`**, and
`characterize_ring` returns an invalid result instead of raising, so it is applied.

Measured, `runs/extrusion/20260901-174339-36ba48cf`:
- `characterize-01` (vertical): **valid FALSE**, completeness 0.350, gap 233.8 deg,
  radius **50.05**, centre (203.03, 137.89), bead 8.17.
- `characterize-02` (horizontal): **valid TRUE**, completeness 0.992, closed, radius
  **42.40**, centre (203.91, 147.43), bead 10.11 -- and 42.4 is what every rolled take
  then measured (41.4-41.9).
- `session.json`: `applied characterization_index = 1`.

Consequence: every take in that run was scored against a nominal ring **8 mm too big,
centred 9 mm off**, and the camera aimed at that wrong centre. `mean_absolute_mm` (1.87)
and `center_offset_norm_mm` (9-11 mm) from that run are meaningless -- do not quote them.
The centre offset is the tell: a good run reads ~0.5 mm (13:13 run, recipe 42.0 vs
measured 42.01, offset 0.47).

Cause of the invalid vertical characterization is [[roll-verdict-vertical-flattens-bead]],
so the two defects compound: the view that cannot see the ring is the one that defines it.
