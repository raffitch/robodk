# Handoff — scan edge coverage + surface-dot visualization

**Status:** partially done, **not yet satisfying** — see §1. Diagnostic + coverage
selection are in; the *actual top-plane quality* (edges captured + rectangle that
hugs the real board) is committed-but-unvalidated and the rectangle trim is not done.
**Branch:** `calibration-improvements` (HEAD `2519898`). **Main:** `7eacf67`.
**Run the app:** `.\start.ps1` → Scan. Cell = KUKA + D435i on the Jetson (TCP 1024).

---

## 1. Why this isn't done yet (read first)

The user's original complaint (paraphrased): *"I should be able to get a pretty
accurate top plane, but the edges aren't being captured and the plane isn't adapted
properly."* Then, on the live dots: *"they change place every frame… I want to see
the full projection of dots on the uppermost surface."*

Two **different** problems got tangled together. Keep them separate:

| | The DIAGNOSTIC (seeing coverage) | The ACTUAL GOAL (a good top plane) |
|---|---|---|
| What | live/lock dots showing where depth landed | edges captured + rectangle fits the real board |
| State | mostly done (lattice + accumulation) | **coverage-selection committed but NOT validated on the cell; rectangle over-run NOT fixed** |

Most of the last few sessions went into the **diagnostic** (dots dancing → stable
lattice → frame accumulation). That is necessary but **does not by itself fix the
scan**. The dissatisfaction is almost certainly the ACTUAL GOAL: a real run still
needs to (a) capture all four board edges and (b) produce a rectangle that matches
the physical board instead of over-running it. Those are §4.

**Do not keep polishing the dots** unless a concrete coverage hole is still invisible.
The dots already revealed the problem (see §3); the next work is closing it.

---

## 2. What's committed (precise)

On `calibration-improvements`, newest first:

- `2519898` **frontend coverage accumulation** (branch only, NOT merged — frontend-only,
  needs no deploy). `Scan.tsx` unions the last 18 live frames' `points_uv` (deduped to a
  ~1/180 grid), resets on a >3.5%-of-frame camera move + on start/stop/lock. `AimHud.tsx`
  gained an optional `coverageDots` prop. Fills per-frame RealSense dropouts so the whole
  board shows.
- `3766fb0` **stable dot lattice** (merged → main `7eacf67`). `server/…syncronous.py`
  `scan_plane_telemetry` + `tasni/modules/scan/survey.py` `survey_surface` emit `points_uv`
  as a fixed metric lattice in the surface's rectangle frame, anchored at the CENTROID, one
  dot per occupied cell center. Replaces the per-frame random subsample that made dots dance.
- `6d68c2d` **coverage-aware target selection + dots + save_views** (merged → main `01b85c1`).
  `tasni/modules/scan/service.py::generate_scan_targets` now uses
  `select_diverse_with_coverage` + `projected_corner_coverage` (was azimuth-blind
  `select_diverse`); warns below `scan.min_surface_coverage` (0.85); returns `surface_coverage`.
  `scan.save_views` persists per-pose color+depth+pose under `<run>/views/`.

165 Python tests green; `tsc` + `vite build` clean.

### RealSense facts established (so nobody re-derives them)
- D435i is **stereo**; output is a **dense depth raster** aligned to color **@1280×720
  (≈920k px, ~840k valid** on the close table — see the run logs' "…depth px").
- WHICH pixels are valid **fluctuates every frame** (stereo dropouts at edges / low
  texture / glare). So coverage is best judged over **several frames** (hence §2's
  accumulation), not one.

---

## 3. The evidence — run `20260625-171346`

Plane fit is *fine* (87% inliers, 1.5° tilt). The failure is a **one-sided capture
hole**. Reproduce with the new tool:

```
py -3.10 tools/scan_coverage_report.py 20260625-171346
```

→ interior 100%, **X+ edge 44%, Y+ edge 40%, both +X corners 0%**; fitted rectangle
436×317 mm vs dense board ~422×305 (2/98%). So: (a) the +X side was never captured,
and (b) the rectangle over-runs the dense board by ~14 mm into sparse fringe.

The `(a)` one-sidedness is exactly what `select_diverse → select_diverse_with_coverage`
(commit `6d68c2d`) targets — but **this run predates that fix**, so it has NOT been
proven to close the hole on hardware.

---

## 4. The real work, prioritized

### P1 — Validate the coverage fix on the cell (do this FIRST; it may be enough)
1. App on latest (`.\start.ps1`), `scan.save_views = true` in `tasni.config.json`.
2. Run a scan. On Create targets, note the new `predicted surface coverage NN%` log line.
3. `py -3.10 tools/scan_coverage_report.py` (newest run). If every edge ≥ ~70% and no
   0% corner, the capture problem is solved — move to P2. If an edge is still starved,
   the seed view / cone need work (P3).

### P2 — Rectangle that hugs the real board (the "plane not adapted" half)
`tasni/modules/scan/plane.py::_oriented_rectangle` currently trims a **fixed 0.5%
quantile** then takes the min-area box → it over-runs the dense board by ~14 mm. Replace
the fixed quantile with a **per-edge density drop-off**: along each rectangle axis, walk in
from the extreme until point density crosses a threshold, set that as the edge. Keep the
min-area *orientation*; only tighten the *extents*. Pure post-processing, no robot motion;
add a `test_scan_plane.py` case (fringe points beyond a dense core must be trimmed).

### P3 — Edge-biased capture (only if P1 still leaves starved edges)
Every view aims at the centroid inside an 18° cone (`scan.flat_cone_deg`), so edges are
always peripheral / foreshortened. Levers, cheapest first:
- raise `scan.frame_margin` (each view frames more around the surface),
- widen `scan.flat_cone_deg` / `cone_half_angle_deg`,
- spread **aim points** toward the edges (calibration already has
  `poses.frame_aim_offsets` — a 3×3 aim spread; reuse it for scan so some views look at
  edges, not just the centroid). This is the highest-leverage change for true edge depth.

### P4 — Make coverage a GATE, not just a log
`generate_scan_targets` already computes `surface_coverage`. Consider refusing / strongly
warning Run when predicted (or post-run actual) coverage is below threshold, so a known
hole can't silently ship.

---

## 5. Deploy gotcha (live lattice still pending)

The stable-lattice **server** (`3766fb0`, on main `7eacf67`) reaches the Jetson via
auto-pull — but the timer **defers while a client is connected on :1024**. So while the
app is streaming, the Jetson keeps the OLD random-subsample server. To deploy: **stop the
camera in the app for ~2 min**, let auto-pull hard-reset + restart, then reconnect.
Verify (read-only) with `tools/jetson_deploy.py status` or check Jetson `git rev-parse
HEAD` == `7eacf67`. The frontend accumulation (§2) masks the dance regardless, so this is
not blocking — but the per-frame dots are only truly stable once it lands.

NB: a manual `git reset --hard` + `systemctl restart` over SSH was **denied by the safety
classifier** as out-of-scope; the sanctioned path is re-enabling auto-pull and letting it
self-deploy. Don't try to force it.

---

## 6. Diagnostic tooling
- `tools/scan_coverage_report.py` (NEW) — per-edge/corner occupancy + dense-board vs fitted
  rectangle for any run. This is the objective coverage check; use it before/after every
  experiment instead of eyeballing.
- Top-down dot-map artifact (built 2026-06-25):
  https://claude.ai/code/artifact/9581744f-e010-4ff6-bb6c-205b0b6e0559 — depth-density /
  true-color / occupancy views of a run's `preview.npz`. Regenerate by feeding a run's
  points to a similar canvas if needed.
- `scan.save_views` → `<run>/views/` (color jpg + 16-bit depth png + `views.json` poses):
  the data for a **camera-perspective** overlay (project the fused cloud back into each
  view to see coverage from the camera's eye). Not built yet — candidate if P1–P3 still
  leave doubt about which views missed an edge.

## 7. Verify end-to-end
- Tests: `py -3.10 -m pytest -q` (expect 165). Frontend: `cd tasni/webui && npx tsc
  --noEmit && npx vite build`.
- Live: `.\start.ps1` → Scan → Start camera → aim (no lock) → red dots should accumulate
  into a steady coverage field; gaps = real holes. Create targets → read `predicted
  surface coverage`. Run → `tools/scan_coverage_report.py` to grade the result.
