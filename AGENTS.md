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
   — the **currently open blocker** (see §4).
3. **[tasni/README.md](tasni/README.md)** — the app's architecture.
4. `docs/extrusion-current-handoff.md`, `docs/jetson-scanner.md`,
   `docs/scan-workframe-two-path-plan.md` — per-area depth.

## 4. What is open right now (2026-08-28)

**Live print: the arm does not move.** The app dispatches a layer program, RoboDK accepts
it, the cell clicks once, the arm does not move — and right-clicking the *same* program in
RoboDK afterwards *does* move it.

Measured and therefore **dead** — do not re-chase:

- `RunCode()` returns **195 of 195** instructions; nothing is being refused.
- Station run mode reads back as **6** (`RUN_ROBOT`) on every dispatch.
- The trivial **2-instruction valve program dispatches identically** and is equally
  silent, so this is **not** about the layer program's contents. Stop looking at the
  toolpath, the Curve Follow machining project, and the valve mapping.
- Also dead: stale work frame/centre, wrong inspection tool, a bad scan, the
  `RoboDKsync570.src` driver version (the two copies are byte-identical), `$OUT[0]`.

**Next action:** `py -3.10 tools/dispatch_bisect.py jog` — a direct driver `MoveJ`, no
program at all. If the arm moves, the fault is RoboDK's program executor; if it does not,
the fault is below RoboDK and `ConnectedState() == READY` only ever meant the socket was
up. Full reasoning in the handoff doc.

**Also pending:** the PFH paper's ring-stack cell run (deadline 1 Sep 2026) — see
`docs/superpowers/` and `docs/extrusion-current-handoff.md`. Discard every
`runs/extrusion/20260828-*-f088cf48` trial: they are measurements of an empty board and
must not reach the paper.

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
