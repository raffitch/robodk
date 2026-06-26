# Handoff — scan dots + work-rectangle that hugs the board

**Status:** four fixes landed this session; **all committed + pushed**, the two
server-side ones **deployed to the Jetson and verified healthy**. **None eyeballed on
the cell yet** — on-cell confirmation is the next step (§6).
**Branch:** `calibration-improvements` (HEAD `0c83bbc`).
**Main:** `f88b44d` (the two server commits were cherry-picked here for the Jetson).
**Run the app:** `.\start.ps1` → Scan. Cell = KUKA + D435i on the Jetson (TCP 1024).

---

## 1. What the user reported (in order)

1. *(snap.png)* "I should get an accurate top plane but the edges aren't captured and
   the rectangle isn't adapted / over-runs the board."
2. "The dots change place every frame even though the camera is static."
3. "'Surface ready' keeps alternating with 'hold position'."
4. *Their own idea:* "For the dots/rectangle, keep the majority (dense) and filter
   whatever falls below the majority as an outlier."

These are **four different problems**. They got addressed as four commits. Keep them
separate — only #1's *real* half (edges actually captured on a run) is still open.

---

## 2. The four fixes (precise)

Newest first. **B = branch only, M = also on main (cherry-picked), J = on the Jetson.**

| Commit (branch) | On main | What | Where it runs |
|---|---|---|---|
| `0c83bbc` | `f88b44d` (M, **J**) | **Live work-box trim + colour edge veto** | Jetson server |
| `1743d11` | branch-only (B) | **Host rectangle trim** (lock/insert box) | host (PC) |
| `93c15fc` | branch-only (B) | **Surface-ready flicker debounce** | frontend |
| `667e19b` | `3f0a967` (M, **J**) | **Coverage dots = real measured depth** | Jetson server + host survey |

### 2a. `667e19b` / `3f0a967` — dots mark REAL depth (problem #2)
The dots were one dot per occupied **cell-centre** of a metric lattice that was
**re-derived every frame** from the surface estimate (centroid + size). That estimate
drifts with depth noise, so the lattice slid frame-to-frame: accumulated dots landed
*beside* each other, and a gap was ambiguous.
**Fix:** emit the **actual plane-inlier pixels** (where depth truly landed) snapped to
a **fixed image grid** (`GRID=180`, cap 4000 cells) as the occupied cells. The grid
never moves → a static camera yields static dots; an empty cell is a genuine hole.
Done in **both** producers: server `scan_plane_telemetry` (live) and host
`survey_surface` (the lock snapshot), so live and locked show the same marker.
*Trade-off:* the dots are now image-anchored, so under camera **motion** they smear in
image space until the frontend's move-reset fires — fine for a static-aim diagnostic.

### 2b. `93c15fc` — surface-ready debounce (problem #3)
"Surface ready" required an **unbroken 1 s** where every gate stayed ok AND frames
stayed fresh. At ~1.5–2 fps a single noisy/late frame reset the timer → the readout
bounced "Surface ready" ↔ "Hold position".
**Fix:** `GATE_GRACE_MS = 1000` in `Scan.tsx` (`lastValidRef`): only a sustained loss
of validity longer than ~one slow frame breaks the streak. Frontend-only.

### 2c. `1743d11` — host rectangle trim (problem #1 "over-run" + user idea #4)
The work rectangle's extent came from a symmetric **0.5 % / 99.5 % quantile** trim of
the plane inliers (`plane._oriented_rectangle`). That cuts the same tiny fraction off
every edge, so a **sparse coplanar halo** just past the real board edge (flying TSDF
pixels / depth spill — more than the 0.5 % removed) survived and **inflated the box by
~14 mm** into the fringe + empty corners.
**Fix:** new `plane._density_extent_1d`. After the min-area box fixes orientation, pull
each of the 4 sides in to the **density cliff** — the first bin (per side, along the
rectangle axis) whose density rises to `core_frac=0.20` of the body's median populated
-bin density. A real edge is a sharp cliff (body kept); the sub-threshold halo is
dropped. **Guarded:** ≤ `max_trim_frac=0.15` of the span per side, skipped below 60
points, so it can't eat a real fully-sampled edge. Improves **both** the inserted
RoboDK work frame and the lock-time survey review box (survey reuses the function).

### 2d. `0c83bbc` / `f88b44d` — live box trim + COLOUR veto (the "do a and b")
**(a)** Ported the density-cliff trim to the **Jetson live rectangle** so the operator's
overlay box hugs the surface *while aiming*, matching lock/insert.
**(b)** Added a **colour cross-check** so a trim can't amputate a real-but-under-captured
edge: before trimming a side, sample the colour image **just inside vs just outside**
the cliff. Low contrast = surface continues across it (a depth hole, not a real edge) →
**VETO** the trim. High contrast (board vs table) → trim applies.

**Architecture (important):** the pure decision logic is in a **new numpy-only
`server/scan_overlay.py`** (`density_extent_1d`, `edge_continues`,
`side_sample_points`) so the host suite unit-tests it — the rest of the server can't be
imported off the Jetson (pyrealsense2/turbojpeg). The server passes the colour frame
into `scan_plane_telemetry` and orchestrates the 4-edge confirm; any colour-sampling
failure **abstains** → degrades to density-only.

**SAFE SPLIT — this is the key design guarantee:** the **raw** rectangle still drives
**every gate unchanged** — FRAMED, the standoff recommendation
(`color_fit_standoff_per_margin_mm`), the work-frame corners
(`rectangle_corners_color_mm`), and `extent_mm` are all computed from the **untrimmed**
extent. Only the **display** fields changed: `outline_uv` (the overlay polygon) and the
reported `rectangle_size_mm` are now the **trimmed** box. So gating/standoff cannot
regress from the trim.

---

## 3. Tests

- **175 pytest green** (was 165): +2 in `test_scan_plane.py` (halo trimmed / uniform
  board preserved), +8 in new `test_scan_overlay.py` (density cliff, guard cap,
  too-few-points skip, colour veto detects real edge / vetoes on continuing surface /
  no-veto-without-evidence, side-sample geometry).
- `test_scan_survey.py` dot-count bound updated 1000 → 4000 (fixed-grid is finer).
- `test_scan_telemetry_server.py`: the distortion-projector assertion now allows a
  multiple of 4 corner-projector calls (raw + trimmed corners) and adds `server/` to
  `sys.path` so it imports `scan_overlay` regardless of test order.
- `vite build` green.

---

## 4. Deploy state (Jetson)

The Jetson tracks `origin/main` and **auto-pulls every ~2 min** (root systemd timer),
restarting the camera only when `server/` changed **and no client is on :1024**.

- **Both server commits are deployed + verified:** after `3f0a967` →
  Jetson HEAD flipped, `real_uv` marker present; after `f88b44d` → Jetson `HEAD=f88b44d`,
  `scan_overlay.py` present, `density_extent_1d` in the server, service **active +
  listening**, restarted 13:10:37, **`NRestarts=0`** (no `import scan_overlay`
  crash-loop). `realsense-camera.service` runs the script by path with
  `WorkingDirectory=…/server`, so the import resolves.
- **Deploy gotchas (relearned this session):**
  - Auto-pull **defers while a client holds :1024**. The user's live preview
    *flap-connects* (connect→broken-pipe every frame), so a no-client gap eventually
    coincides with a tick and it deploys — but it can take several minutes. If you need
    it now, **stop the camera in the app** for ~2 min.
  - **`tools/jetson_deploy.py deploy` is unreliable** — its `sudo systemctl restart`
    needs `JETSON_SUDO_PASSWORD`, which looks unset (the status `journalctl` prompted
    for a password). A failed restart would leave old code running with no future
    auto-restart. **Prefer the root auto-pull.** Read-only SSH checks (`git rev-parse`,
    `ss`, `systemctl is-active`, `test -f`) work fine over the key.
  - SSH: `ssh -i ~/.ssh/jetson_robodk jetson@10.12.171.70`.

---

## 5. What needs the app restarted vs. not

- **Live dots + live box trim** → already live on the Jetson; just refresh the preview.
- **Surface-ready debounce (`93c15fc`)** and **host rectangle trim (`1743d11`)** are
  **branch-only** (frontend + host). To see them, run the app on
  `calibration-improvements` and restart `.\start.ps1` (the debounce is frontend; the
  host trim affects the lock-review and the inserted frame).

---

## 6. What to verify ON THE CELL (not yet done)

1. **Dots hold still** on a static camera and fill the board; a gap = a real hole.
2. **"Surface ready" stops flickering** while holding a valid pose.
3. **Live box hugs the board** while aiming (no ~14 mm over-run into the fringe).
4. **Lock-review box and the inserted RoboDK frame hug the board** (host trim).
5. **The colour veto works:** deliberately under-capture one edge (so depth is sparse
   there but the board is clearly visible in RGB) and confirm the box does **not**
   amputate that edge. Conversely, the real board/table boundary **is** trimmed.

---

## 7. The bigger picture — DON'T lose the real goal

Everything this session improved the **diagnostic + display**: honest dots, a steady
readout, a box that *reports/draws* the real board. That is real progress and directly
serves the user's "rectangle over-runs" complaint. **But the other half of report #1 —
the scan actually CAPTURING all four edges on a run — is still open.** See
`docs/scan-coverage-dots-handoff.md` §4 / P1–P4:

- **P1** validate `select_diverse_with_coverage` (`6d68c2d`) on the cell with
  `scan.save_views=true`, grade with `tools/scan_coverage_report.py` (never run on HW).
- **P3** edge-biased capture (reuse calibration's `poses.frame_aim_offsets` 3×3 aim
  spread; raise `frame_margin`/cone) — a centroid-only aim + 18° cone foreshortens
  edges, which is *why* corners come back at 0 % occupancy.
- **P4** make `surface_coverage` a hard gate.

Note the **tension** (flagged to the user): the density trim *removes* sparse edges,
edge-capture *adds* them. The colour veto is the reconciliation — it stops the trim from
hiding an under-captured edge — but the durable fix is P3 (actually capture the edge).

---

## 8. Known limitations / follow-ups

- **Colour veto is heuristic.** `edge_continues` uses median-intensity contrast
  (default `min_contrast=22`/255). A **low-contrast** scene (pale object on pale table)
  finds no edge to confirm → no veto → density-only trim still applies. For the cell's
  white-board-on-dark-table this is fine. A more robust version would segment the board
  in RGB rather than threshold a gradient.
- **Image-anchored dots smear under motion** (§2a) until the move-reset fires. If clean
  dots *during* motion are ever needed, anchor the fixed grid to the surface (board
  frame) and lock it once at aim start.
- **Tunables** all live at call sites / module defaults: dot `GRID=180`/cap 4000;
  trim `core_frac=0.20`, `max_trim_frac=0.15`, `min_bin_mm=4`, `min_points=60`; veto
  `min_contrast=22`, `min_samples=5`; debounce `GATE_GRACE_MS=1000`.

---

## 9. File map (this session)

- `server/scan_overlay.py` **(new)** — pure trim + colour-veto logic (host-testable).
- `server/server_unicast_syncronous.py` — dots → real depth; live box trim + colour
  veto wired into `scan_plane_telemetry` (raw rect still feeds gates).
- `tasni/modules/scan/plane.py` — `_density_extent_1d` + per-edge trim in
  `_oriented_rectangle` (host lock/insert box).
- `tasni/modules/scan/survey.py` — survey dots → real depth (reuses `_oriented_rectangle`).
- `tasni/webui/src/pages/Scan.tsx` — `GATE_GRACE_MS` debounce.
- `tests/test_scan_overlay.py` **(new)**, `tests/test_scan_plane.py`,
  `tests/test_scan_survey.py`, `tests/test_scan_telemetry_server.py` — updated/added.
