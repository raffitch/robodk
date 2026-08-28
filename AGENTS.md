# AGENTS.md — start here (any coding agent)

Entry point for **any** agent working in this repo — Codex, Cursor, Copilot, a fresh
Claude session. It carries the operational knowledge that is otherwise only in one
assistant's private memory store, so nothing here depends on which tool you are.

**[CLAUDE.md](CLAUDE.md) is the full project brief** — what this station is, the RoboDK
extract/sync loop, the `tasni/` platform, the north star, the roadmap. Read it after this
file. Everything in it applies to you too; the filename is historical.

---

## 1. Working agreement (non-negotiable)

**Commit AND push every change.** The operator reviews progress from the pushed history
and the Jetson deploys from it, so an unpushed local commit is invisible to them and can
strand the robot on old code. Say which commit hashes you pushed in your summary.

- Current working branch: **`main`** (`calibration-improvements` was merged in `51849b1`
  on 2026-08-25 and the Jetson was re-pointed at `main`).
- If a change touches **`server/`**, it runs on the Jetson: push, then
  `python tools/jetson_deploy.py deploy`, and report the restart status.
- This is a **real 150 kg industrial robot** with a real pneumatic extruder. Code here
  moves it. Never weaken a safety gate, a valve fail-safe, or a collision check to make
  something pass.

## 2. Environment gotchas that will waste your day

| Trap | What to do |
|---|---|
| There is **no `python` on PATH** | Use **`py -3.10`**. `python` fails with "command not found". |
| **The full pytest suite is too slow** | It has been interrupted repeatedly. Use `py -3.10 -m pytest tests/test_<specific>.py -q`, plus import/syntax checks. For the frontend, `npm run build` in `tasni/webui`. |
| **PowerShell mangles this repo's UTF-8** | Never round-trip source through `Get-Content`/`Set-Content` — it silently mojibakes the em-dashes and degree signs this codebase is full of. Use the editor tools, or `py -3.10` with explicit `encoding="utf-8"`. |
| **The app caches imported modules** | After editing backend code you MUST restart the app or you are testing stale code. `/api/health` reports `build.stale`. Check it **before** asking the operator for a cell test — this has burned multiple cell runs. |
| **Loading `Tasni.rdk` takes 1–2 minutes** | The station is ~117 MB. Expected, not a hang. |
| **RoboDK's API attaches to ANY running instance** | `rdk_extract.py` / `rdk_sync.py` use `rdk_session.connect()` which spawns a private headless instance. The **app** deliberately uses `attach` mode to bind the operator's open GUI. Do not mix them up. |
| **`nmcli` on the Jetson over SSH** | polkit denies a non-console session; pipe the password via `sudo -S` from `secrets/jetson.env` (**git-ignored, never commit**). |

## 3. Where to look first

1. **[docs/agent-debug-map.md](docs/agent-debug-map.md)** — the index. Fast orientation
   before opening any long handoff doc. Start here.
2. **[docs/live-print-dispatch-handoff-2026-08-28.md](docs/live-print-dispatch-handoff-2026-08-28.md)**
   — the resolved API-dispatch blocker and cell evidence (see §4).
3. **[tasni/README.md](tasni/README.md)** — the app's architecture.
4. `docs/extrusion-current-handoff.md`, `docs/jetson-scanner.md`,
   `docs/scan-workframe-two-path-plan.md` — per-area depth.

## 4. What is open right now (2026-08-28)

**Live print: RoboDK's `RunCode()` API path dispatches nothing; the item-Start fix is
cell-validated through a complete Characterize motion/capture/return.** The old path dispatches a
layer program, RoboDK accepts it, the cell clicks once, the arm does not move — and
right-clicking the *same* program in RoboDK afterwards *does* move it.

Measured and therefore **dead** — do not re-chase:

- `RunCode()` returns **195 of 195** instructions; nothing is being refused.
- Station run mode reads back as **6** (`RUN_ROBOT`) on every dispatch.
- The trivial **2-instruction valve program dispatches identically** and is equally
  silent, so this is **not** about the layer program's contents. Stop looking at the
  toolpath, the Curve Follow machining project, and the valve mapping.
- The direct-driver bisect `dispatch_bisect.py jog` **physically moved A6 by 2 deg**.
  Immediately afterwards the bare one-MoveJ `trivial` program returned 1/1 accepted,
  never became busy, left the driver READY, and did not move the physical arm. This
  measures the fault inside RoboDK's API program-execution path; the driver,
  KUKAVARPROXY, KRL loop, pendant state, and app-specific timing are not the cause.
- Also dead: stale work frame/centre, wrong inspection tool, a bad scan, the
  `RoboDKsync570.src` driver version (the two copies are byte-identical), `$OUT[0]`.

The bare item-Start bisect physically moved the arm, and the app now uses that path for
real-robot programs while retaining `RunCode()` for simulation. On the first app retry,
Characterize moved to the overhead pose, captured RGB-D, archived it, and returned; this
validates app dispatch without involving a valve. Python `robodk>=6.0.1` is required.

The first archived frame exposed a separate processing bug: the largest DBSCAN cluster
was a broad ChArUco-board depth residual, not the ring. The new selector uses angular
coverage and radial compactness; offline replay of that exact frame reads radius 39.17
mm, centre (217.94, 150.44) mm, bead footprint 13.26 mm and top Z 6.14 mm.

**2026-08-28 evening — first successful measure-only cell run** (session
`runs/extrusion/20260828-192115-47fb78ea`, 300 mm, collisions OFF): Characterize picked the
ring over the board residual (selector candidate 2: coverage 0.97, radial-span ratio 0.39;
r 40.5 mm, centre (197.5, 152.5), bead 10.4 mm, top Z 6.0 mm) and a separate Measure layer
1 re-found it at r 39.9, centre (197.0, 152.4) — 0.5 mm apart — shape RMS 1.9 mm, capture
2.7 s, acquisition→path 3.1 s. Its 10 mm mean |dev| is the ring sitting 15 mm from the
un-applied plan centre, not measurement error. **Next action:** press **Apply to recipe &
placement**, re-measure layer 1 as the zero-offset baseline, then run the introduced-offset
protocol (spec §3) for the paper. **Closer is not available on this cell:** depth streams at
the D435i's native maximum 1280×720, whose MinZ is ~280 mm, so the existing 300 mm
`inspection_min_mm` already sits at the sensor floor. (An earlier change clamped
measure-only to 175 mm on a wrong MinZ reading; `measure_close_range_min_mm` is now 300 mm
and the operator checkbox is gone. Real close-range headroom needs a LOWER depth profile
in `server_unicast_syncronous.py` — MinZ scales with depth width — which is a separate,
deliberate change.) The optical fit at 135 mm is what the recipe *wants*, not what the
sensor can measure. Failed characterization now archives raw RGB-D instead of losing the
only diagnostic frame.

**2026-08-28 — figures per take.** `tasni/modules/extrusion/figures.py` renders `plan`,
`heightmap`, `iso` and `profile` (300 dpi PNG + vector PDF) plus a per-trial `stack`,
from the archive alone: no robot, no RoboDK, no camera. Every take draws them
automatically; serving is render-if-missing, so takes archived earlier — including
`20260828-192115-47fb78ea` — produce figures with zero cell time. Click a take in the
measurement table to see them. Needs `pip install -e .[figures]`; without matplotlib the
measurement is unchanged and only the figures are skipped. Details and the two
correctness traps (deposit-band colour range; FITTED not averaged nominal centre) are in
`docs/extrusion-current-handoff.md`.

> **Picking this up? Read [docs/live-print-next-session.md](docs/live-print-next-session.md)**
> — the continuation handoff: current state, the full decision tree for each bisect
> outcome, where the code is, and the fallback options with their real costs.

**Also pending:** the PFH paper's ring-stack cell run (deadline 1 Sep 2026) — **read
[docs/pfh-paper-handoff.md](docs/pfh-paper-handoff.md) first**: it is the single page for
that task (what the paper still needs, the exact operator order, and the wording
constraint). Background in `docs/superpowers/` and `docs/extrusion-current-handoff.md`. Discard every
`runs/extrusion/20260828-*-f088cf48` trial made before the real-ring capture: they are
measurements of an empty board and must not reach the paper. Trial
`20260828-171615-f088cf48/characterize-01` does contain the real ring, but its archived
52.77 mm radius / 51.12 mm bead result selected the board residual and is invalid. Keep
the raw frame as regression evidence; do not use its metrics in the paper. The following
300 mm retry was rejected by the ring-shape gate and predated failed-frame archiving, so
it produced no `characterize-02` directory and is also not evidence.

## 5. How to work here

- **Measure before theorising.** Most of the time lost on this project has gone to
  confident diagnoses reasoned from partial evidence. The measurements that actually
  resolved things were quick: a depth grab at the parked pose, a TCP comparison, reading
  the driver `.src`, logging a return value that was being discarded.
- **Keep test fakes physically coherent.** Fakes here have modelled impossible cells (a
  camera 6 mm from its aim point serving a 500 mm depth frame; a 2 s program finishing
  instantly; a driver that never leaves READY) and then passed the very guard meant to
  catch the real fault.
- **State what is measured vs inferred** in any handoff you write. An inference presented
  as a rule-out costs the next agent a day.
- **The flange camera is the only independent witness of real motion.** RoboDK's model
  advances to the target whether or not the controller executed anything.
