# Handoff — calibration aiming guidance (Part 2: board tilt-direction guidance)

**Status:** ✅ §1–4 DONE (2026-06-24). The calibration gate now emits
`tilt_b_deg`/`tilt_c_deg` and the HUD's ROTATE-TOOL panel shows board-level
direction. `board_tilt_bc_deg()` in `core/aiming.py` reuses the scan gate's
normal→B/C decomposition (signs consistent by construction); `evaluate_gate`
carries the two fields and `to_dict()` emits them (both live-preview and
authoritative-grab paths ride along, no other change). Frontend copy tweak (§3b)
skipped — `TiltFix` has no "surface" wording, reads fine for the board. Test
`test_tilt_direction_bc` added; 125 pytest green + `vite build` green.
**Remaining:** real-board sign verification (§3b note / §7 last bullet — needs the
cell), and the secondary ideas in §5 and the collision follow-ups in §6.
**Branch:** `calibration-improvements` (HEAD `725e9dd` as of 2026-06-24).
**Context:** follow-up to the calibration collision-safety work (commit `725e9dd`).
The user asked to improve "how we position the robot, what info we display, and how
we guide the viewer to position the robot" in the calibration section, and (via a
scoping question) **prioritized rotation guidance: tell the operator which way to
rotate to level the board** — the calibration HUD shows the tilt *magnitude* but not
the *direction* to fix it. The scan HUD already has this; calibration doesn't.

---

## 1. The goal

When aiming the camera at the ChArUco board, the calibration HUD (`AimHud`, mode
`"calibration"`) shows **RANGE** and **TILT** readouts and an X/Y/Z **JOG** bar, but
TILT only shows the angle (e.g. "18°") — not which way to rotate the tool to make the
board fronto-parallel. The scan module already solves this: its HUD shows a
**"LEVEL — ROTATE TOOL  B ◀ 7°  /  C ▼ 4°"** panel (KUKA A/B/C convention). We want
the same guidance for the calibration board aim.

**Good news:** the rendering already exists and is mode-independent. `AimHud`'s
`TiltFix` component (`tasni/webui/src/pages/AimHud.tsx`) renders **whenever
`gate.tilt_b_deg`/`gate.tilt_c_deg` are non-null** — it's drawn for scan today only
because only the scan gate emits those fields. So Part 2 is mostly: **make the
calibration gate emit `tilt_b_deg`/`tilt_c_deg`.**

---

## 2. How the scan gate computes it (the pattern to reuse)

`tasni/modules/scan/depth_gate.py` (~lines 116–127): it has the surface **normal in
the camera frame**, orients it to face the camera (so its Z is negative), then:

```python
if normal[2] > 0:           # face the camera (toward -Z optical axis)
    normal = -normal
nx, ny, nz = normal
denom = max(-nz, 1e-9)
tilt_b_deg = degrees(atan2(nx, denom))   # rotate about camera/TOOL Y  -> KUKA B (left/right)
tilt_c_deg = degrees(atan2(ny, denom))   # rotate about camera/TOOL X  -> KUKA C (fwd/back)
```

A rotation about Z (KUKA A) does NOT change tilt, so only B and C are guided.
`ScanGateReading` carries `tilt_b_deg`/`tilt_c_deg` and `to_dict()` emits them.

## 3. The change for calibration

### 3a. Backend — `tasni/core/aiming.py`
The calibration gate is `evaluate_gate(det, K, image_shape, th, board_center_mm)`.
The board's normal in the camera frame is **column 2 of `det.R_target2cam`** (the
helper `board_tilt_deg()` already uses it for the magnitude). Add the direction:

1. Add `tilt_b_deg: float | None = None` and `tilt_c_deg: float | None = None` to
   the `GateReading` dataclass, and to `to_dict()`.
2. In `evaluate_gate`, after computing `R`/tilt, compute the normal decomposition
   with the **same formula as the scan gate** (orient the board normal toward the
   camera first — `R[:, 2]`, flip if its Z > 0). Set the two fields on the returned
   `GateReading`. The `None`-board branch leaves them `None`.

Note: the board normal sign depends on the board definition; orienting it toward the
camera (negative Z) before the `atan2` makes B/C signs consistent with the scan gate
and the HUD's ◀▶ / ▲▼ arrows. Verify the sign on the real board (a board tilted so
its top leans away should ask to rotate the same way scan does).

### 3b. Frontend — `tasni/webui/src/pages/AimHud.tsx`
- `GateReading` already declares `tilt_b_deg?`/`tilt_c_deg?` and `TiltFix` already
  renders when either is non-null — so once the backend emits them, the calibration
  HUD shows the panel **with no frontend change required**.
- Optional polish: `TiltFix`'s copy says "surface" / "LEVEL". For the board aim that
  still reads fine, but consider passing the `mode` to `TiltFix` and switching the
  word "surface" → "board" in calibration mode. Low priority.

### 3c. Where the gate is emitted (no logic change, just confirm it flows)
- Live preview: `module.py` `live_start()` → `analyze()` calls `evaluate_gate(...)`
  and ships `reading.to_dict()` as the `gate` event. New fields ride along.
- Authoritative grab: `service.py` `generate_calibration_targets` also calls
  `evaluate_gate` and publishes a `gate` event. Same.
So emitting the fields from `evaluate_gate` is sufficient for both paths.

### 3d. Test
Extend `tests/test_gate.py`: build a `ViewDetection` whose `R_target2cam` tilts the
board a known amount about camera X (and Y), call `evaluate_gate`, and assert
`tilt_c_deg` (resp. `tilt_b_deg`) has the right magnitude **and sign**, and that the
other axis is ~0. Mirror the scan gate's tilt tests if present.

---

## 4. Files to touch
- `tasni/core/aiming.py` — compute + carry `tilt_b_deg`/`tilt_c_deg` (the real work).
- `tasni/webui/src/pages/AimHud.tsx` — optional copy tweak only.
- `tests/test_gate.py` — assert the B/C decomposition.
- `vite build` + `py -3.10 -m pytest` to verify.

---

## 5. Broader Part-2 ideas (the user's wider ask — secondary, not prioritized)
The user also mentioned "how we position the robot / what info we display / how we
guide the viewer." Beyond tilt direction, candidate improvements (confirm before
building):
- **Seed-quality hint before Create targets**: warn at aim time (not only after) if
  the current view is at a workspace edge where the reachable cone will be starved
  (the effective-cone warning already exists post-generation; surface a *pre*-warning).
- **A roll/centring cue** if the board is far off-centre (the lock-box + JOG bar
  cover translation; tilt B/C covers angle; roll about the optical axis is not gated
  for ChArUco and probably needn't be).
- Live 3D preview of the generated tour (bigger; the `ConeDiagram` is a schematic).

---

## 6. Related collision follow-ups (from commit `725e9dd`, capture so they're not lost)
- **Target-9 "never created" (optional):** `725e9dd` models the platform (keep-out
  box) and the **dry tour now flags** low-transit bumps (e.g. 8→9), and generation
  drops any pose whose *seed→pose* approach enters the box — but generation screens
  seed→pose, NOT consecutive tour transits, so a low pose like target 9 may still be
  *created* (then flagged by the dry tour). To prevent creation entirely: after
  `select_diverse` + ordering in `generate_calibration_targets`, screen the
  consecutive tour (seed→t1→…→tN→seed) with `rdk.path_new_collisions(prev, dest,
  baseline, samples)` against the keep-out box and DROP any target whose tour transit
  introduces a new collision (re-link to the next kept target); refuse if too few
  remain (consistent with the chosen refuse-with-guidance behaviour).
- **Keep-out margin:** `calibration.board_keepout_margin_mm` defaults to **300 mm**
  (live-validated to catch the 8→9 dip in this cell). It models the platform's
  overhang beyond the board — reduce for a small pedestal, increase for a big table.
- **Intrinsics stay implicit:** normal `TasniCalib_*` generation now aims the board
  across a 3x3 frame pattern and selects targets for both rotation diversity and
  4x3 image-grid coverage. The first Run solves K + distortion from those same
  captures; every later well-covered Run refreshes the fit rather than trusting a
  stale marker. There is no separate operator step.

---

## 7. Verify
- `py -3.10 -m pytest -q` (currently 124 green on `725e9dd`).
- `cd tasni/webui && npm run build`.
- On the cell: start the app, aim at the board tilted off-square, confirm the HUD
  shows **ROTATE TOOL B/C** arrows pointing the way that, when followed, drives the
  TILT lamp green.
