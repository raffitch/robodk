# Tasni UX overhaul — workflow shell, cell-level connect, journey dashboard (design)

Date: 2026-08-28. Status: **draft for operator review** — revised twice the same day
after a second-agent review (all findings verified in code; resolutions in §11).
Branch `ux-overhaul`.
Audit basis: `tasni/webui` on `main` @ `9101aa1` (every file read; refs below are to
that revision).

## 1. Goal

One operator journey across the three modules, and one way of doing each thing.
After this work:

- every module page is a **stepped workflow** — a step rail, one expanded step, one
  primary action per step — built from the **same components** for connect,
  simulate, run, confirm-motion, errors and log;
- the cell is **connected once, from the topbar**, and every module reads that state;
- the **Dashboard shows the journey** (calibrated → work surface → print plan),
  distinguishes what is *recorded* from what is *present in the open station*, and
  links into a named module step (`/m/{id}?step=`);
- explanatory prose leaves the step bodies and lives in a **per-module guide pane**;
- **nothing the cell can do today is lost** — Appendix A maps every existing control
  to its new home.

**Non-goals.** A new visual identity or branding; a 3D viewport; mobile/tablet
layouts; any change to job logic in `tasni/modules/*/service.py` or to robot
motion; a general config editor (phase 4 adds a read/write page for a fixed subset
only, §9).

## 2. Problems this fixes (audit summary)

1. **Three modules, three apps.** Calibration is two-column with a sticky guide
   (`pages/Calibration.tsx:439`, `pages/CalibrationGuide.tsx`); Scan is one column
   of seven cards with no guide; Extrusion is a form grid plus a "Safety workflow"
   card that is the only page with a step tracker (`pages/Extrusion.tsx:678`).
   Vocabulary differs for the same act: *Dry run (simulate)* vs *Geometry & station
   preflight* + *Complete validated dry run*; *Apply to tool* / *Insert into RoboDK*
   / *Apply to recipe & placement*; *Create targets* / *Lock & create targets* /
   *Generate coordinates & fingerprint*. Motion confirmation is a modal with a
   cell-clear ack in Calibration and Scan, inline checkboxes with no modal in
   Extrusion (`Extrusion.tsx:741`), and a third checkbox on the ring-stack card
   (`:755`). The Extrusion module is titled "Cylinder Test"
   (`modules/extrusion/module.py:85`).
2. **Connect three times.** Three `/connect` routes (`calibration/module.py:148`,
   `scan/module.py:528`, `extrusion/module.py:204`), three page-local `conn`
   states, three banners — for one shared RoboDK session. A platform
   `GET /api/rdk/status` already exists (`webapp/server.py:95`) and is what
   Calibration/Scan hydrate from; there is just no platform `connect`.
3. **The journey is invisible.** Calibrate → Scan → Print is the real sequence
   (Extrusion says "run the Scan module first", `Extrusion.tsx:621`; Scan warns
   without a calibration) but the Dashboard shows only calibration provenance and
   "Recent runs" is a dead list (`pages/Home.tsx:73-85`).
4. **Wizards flattened into long pages.** Gating is expressed as greyed buttons plus
   hints that point elsewhere ("Create targets (above) to enable Run",
   `Calibration.tsx:622`; "Lock the surface above…"; a card whose whole content is
   "Not used for a working frame", `Scan.tsx:1251`). In Extrusion the step tracker
   and the buttons for its first two steps are in different cards (`:677` vs `:670`).
5. **Spec language in the UI.** "Boundary DECLARED, not measured", "boundary
   provenance", "acquisition mode", "GENERATED · FINGERPRINTED"
   (`Scan.tsx:1212-1217`, `Extrusion.tsx:664`). Cards open with 3–5 lines of
   uncollapsible prose.
6. **Redundant status.** Scan's Survey card states "surface ready" three ways (lamp
   row `Scan.tsx:1151`, `scan-ready` bar `:1163`, `SurfaceGuide` header `:1732`);
   the topbar pills (`components/Layout.tsx:26`) duplicate the Dashboard "Cell
   status" card (`Home.tsx:45`) with different labels.
7. **Extrusion specifics.** "Station & motion" is 4 selects + 9 numbers + 3 buttons
   + 4 conditional warnings with no grouping; the research-only ring-stack card is
   permanently on the production page (`Extrusion.tsx:745`); the primary CTA is at
   the bottom of the preview card (`:670`).
8. **No settings surface.** The UI tells the operator to hand-edit
   `tasni.config.json` (`Calibration.tsx:490` jog inversion; camera IP, board,
   tolerances).
9. **Job events and status are not module-scoped** (found in review). `JobEvent`
   is `type` + `payload` (`core/events.py:16`); only `status`/`result`/`error`
   carry a job `name` (`core/jobrunner.py:88-104`), while `progress`/`log`/`frame`
   (`:41-49`) and `gate` (`core/livepreview.py:118`) carry nothing, and
   Calibration/Scan consume them unfiltered (`Calibration.tsx:241-247`,
   `Scan.tsx:414-420`). Both modules name their dry run `sim_tour`, so each page's
   `name` check would accept the other's tour result. Every module `/status`
   returns the shared runner's job (`calibration/module.py:409`). A Scan page open
   during a Calibration run shows calibration progress as its own.

## 3. Approaches considered

**A. Stepped workflow shell (recommended).** Each module page becomes a rail of
steps derived from module state by a pure function; one step expanded, one primary
action, shared run/confirm/log components, a guide pane for the prose. *Pro:* fixes
1, 4, 5, 6 structurally, and the "why is this greyed out" question disappears
because the rail states it. *Con:* every page is re-laid-out (not re-written — the
state logic and the HUD/viewer components are reused as-is).

**B. Tabs per module (Setup / Run / Review).** *Pro:* cheap. *Con:* tabs don't
express gating, so the disabled-button-plus-hint pattern stays; three tabs is too
coarse for Scan's six states and too fine for Calibration.

**C. Keep the long pages; unify components and cut copy.** *Pro:* least churn.
*Con:* leaves problem 4 untouched — the operator still infers position from
greyed buttons.

A is chosen. B's simplicity is preserved inside A by keeping the rail on one route
(no per-step URLs), so the live camera and job subscriptions never remount.

## 4. Architecture

### 4.1 Frontend layout

```
tasni/webui/src/
  platform/
    PlatformProvider.tsx   health poll + rdk status + connect() + job event bus
                           (absorbs api/events.tsx and api/useHealth.ts)
    usePlatform.ts
  components/
    shell/      Layout, Topbar (+ CellConnect), Sidebar, GuidePane
    workflow/   WorkflowRail, Step, PrereqChip, RunControls, TourResult,
                RunError, MotionConfirmDialog, LogPanel, TargetsSummary
    (existing, reused unchanged) AimHud, StreamStats, CollisionPanel,
                ConeDiagram, ScanViewer, SurveyPanel, BirdseyeStack
  modules/
    registry.ts            id -> { Page, guide }
    calibration/  Page.tsx  useCalibrationState.ts  steps.ts  guide.ts
    scan/         Page.tsx  useScanState.ts         steps.ts  guide.ts
    extrusion/    Page.tsx  useExtrusionState.ts    steps.ts  guide.ts
                  RingStackTab.tsx
  pages/
    Home.tsx (Dashboard)   RunDetail.tsx (drawer)
```

`pages/Calibration.tsx`, `Scan.tsx`, `Extrusion.tsx` are split, not rewritten: the
hooks/effects/handlers move to `use<Module>State.ts` verbatim, the JSX is
re-expressed as steps. Sub-components already extracted (`SurveyPanel`,
`FramePrepPanel`, `SurfaceGuide`, `Metrics`, `BirdseyeStack`, …) are reused.

### 4.2 The step model (pure, testable)

```ts
type StepStatus = "hidden" | "locked" | "ready" | "done";   // derived, pure
interface Step {
  id: string; title: string; status: StepStatus;
  summary?: string;      // one line shown when collapsed/done
  lockReason?: string;   // shown in the rail tooltip and the step body when locked
  attention?: boolean;   // a dismissible error is pending in this step
}
deriveSteps(state: ModuleState): Step[]   // one per module, in steps.ts
// UI selection is separate state, never derived:
selectedStepId: string | null            // null => the current step
```

- `deriveSteps` is a pure function of the state the page already fetches (rdk
  ready, config, targets, gate, tour, job status, result, applied). **Rendering
  never computes gating itself**; every disabled primary action has a `lockReason`.
- **Workflow status and UI selection are separate.** The *current* step is the
  first step whose status is `ready`; the rail highlights it while
  `selectedStepId` is null. Clicking a step sets `selectedStepId` (locked steps
  open read-only with their reason); it clears when the current step advances past
  it, and it is seeded from `/m/{id}?step=<stepId>` on entry (ignored if that step
  is hidden).
- Steps with no server action (Board, Intent, Placement) complete on their
  Confirm/Continue button. Those flags and the other client-only inputs live in a
  per-module store inside `PlatformProvider`, so they survive navigation between
  modules within a session. **Physical assertions (print-scale ack, placement
  confirmation) reset on reload on purpose** — today's `scaleOk` behaviour — and
  clear whenever the step's inputs change. Non-safety inputs (held-out count,
  refine, selected layer, intent toggles) persist in `sessionStorage`.
- `hidden` steps are not rendered at all (Scan "Run" under *Working frame only*).
- State changes that invalidate later steps (clearing targets, repositioning a lock,
  changing intent, editing a generated plan) are already enforced server-side; the
  rail simply re-derives, so later steps fall back to `ready`/`locked` and their
  summaries clear.

### 4.3 Rail and step behaviour

- **Rail**: horizontal at the top of the main column; `n · title · state glyph`
  (`○ locked · ● ready · ✓ done`, plus `⚠` when `attention`); the current step is
  highlighted and the selected one outlined. Click selects; locked steps open
  read-only with the reason. Under 760 px the rail stacks vertically.
- **Only the selected step is expanded.** A `done` step collapses to its summary
  line with a *Change* affordance that selects it (e.g. Aim: "15 targets
  (TasniCalib_*) — Change").
- **Expanding/collapsing never moves hardware.** Entering Aim/Survey does establish
  the real-robot monitoring link (§4.5) — a driver connection, not motion — and Scan
  starts its surface feed as today; collapsing stops nothing. The camera-live state
  is shown in the step header ("● camera live · Stop") even when collapsed; Run
  stops the live preview exactly as today (`Calibration.tsx:299`).
- **Exactly one primary action per step**, right-aligned in the step footer;
  secondary actions sit left of it. The primary action's disabled state always
  carries a reason (footer text = `lockReason`).
- **Advisory simulation is not a step; gating simulation is** (decision 10.4). The
  tour dry run in Calibration and Scan is advisory (the operator may run without
  it), so it is the Run step's secondary action and skipping it surfaces in the
  confirm dialog as "No dry run performed" — exactly today's nag. Extrusion's layer
  simulation is a hard gate for `live_print_enabled` (`extrusion/module.py:626`),
  so it is its own step (§5.3).

### 4.4 Shared workflow components (contracts)

| Component | Props (essentials) | Replaces |
|---|---|---|
| `WorkflowRail` | `steps, currentId, onSelect` | Extrusion `workflow-steps` |
| `Step` | `step, expanded, onToggle, primary: {label, onClick, disabled, reason, danger?}, secondary?: Action[], children` | every hand-built card |
| `PrereqChip` | `label, state: ok/warn/missing, detail, action?` | Extrusion `surface-row`, Scan calibration hint, `CollisionPanel` header |
| `RunControls` | `onSimulate?, onRun, onCancel, running, runKind, simulateDisabled/reason, runDisabled/reason, progressPct, statusLine, tour?, error?, thumbs?` | the ~60 duplicated lines in `Calibration.tsx:610-676` and `Scan.tsx:1270-1320` |
| `TourResult` | `tour` | both copies |
| `RunError` | `message, onDismiss` | both copies |
| `MotionConfirmDialog` | `title, robot, body, tour: TourResult \| "none", checks: string[], ackLabel, level: "motion" \| "live-process", confirmLabel, onConfirm, onCancel` | both modals; **and** Extrusion's inline Print confirmation and, after the paper run, the ring-stack `confirmMotion` checkbox |
| `LogPanel` | `lines, collapsed default true, badge = line count, auto-expands on ERROR` | three log cards |
| `TargetsSummary` | `count, prefix, onClear` | two `ok-text` blocks |
| `GuidePane` | `guide: GuideSpec, currentStepId, ready` | `CalibrationGuide` (generalised) |

`MotionConfirmDialog` keeps today's safety semantics verbatim: `role=dialog`,
Cancel is `autoFocus` so Enter cannot fire motion, the ack checkbox is required, and
Escape closes. `level: "live-process"` adds the material/valve line and uses the red
button; `"motion"` is the amber tour dialog.

### 4.5 Cell-level connect

**Backend.** `POST /api/rdk/connect` in `webapp/server.py`, next to
`/api/rdk/status`: the calibration/scan polling implementation lifted (open the
station if RoboDK came up empty, poll up to `robodk.connect_timeout_s`, reset the
session on a socket error mid-load, check robot + camera tool). Response = the
`/api/rdk/status` shape. Two deliberate differences from today's module routes:

- **Connect is station-only; the real-robot link is a separate, gated concern.**
  The link is *not* a status convenience: with the driver connected and monitoring,
  the RoboDK model tracks the physical arm, and that is what makes the seed pose
  Create-targets reads the arm's ACTUAL pose (`core/config.py:132-138`;
  calibration seeds from `camera_pose_T()` + `current_joints()`,
  `calibration/service.py:303-308`; scan's `annotate_pose_liveness` documents the
  stale-model-pose case, `scan/service.py:2286-2298`). Today the link is attempted
  at connect (calibration/scan) or never (extrusion), and neither target creation
  nor surface locking checks it — calibration generates around whatever pose the
  model holds. The design therefore:
  1. keeps platform **Connect station-only** (open/attach, robot + tool present);
  2. adds `POST /api/rdk/link` — `link_real_robot` (`core/rdk_io.py:65`) and the
     raising `ensure_real_robot_link` (`calibration/service.py:198`) are
     consolidated into one `core.rdk_io.ensure_robot_link(rdk, cfg, *, strict)`
     returning `{connected, monitoring, message, ip, configured}`; nothing is
     deleted;
  3. **auto-links when the Aim (Calibration) or Survey (Scan) step is entered**:
     the step calls `/api/rdk/link` and shows a `PrereqChip` "Real robot · ONLINE
     (monitoring)" / "OFFLINE — power the controller · Retry"; every `gate`
     reading carries `pose_live` (scan already does; calibration's gate gains it);
  4. **hard-gates server-side**: `POST /poses/generate` (calibration) and
     `POST /surface/lock` (scan, every lock — the plane is measured in the camera
     pose) return 409 unless `pose_live`; the rail's `lockReason` is "Real robot
     not linked — the model pose may be stale". `robodk.require_live_pose: bool =
     True` is the explicit escape hatch for bench work without a controller
     (set together with `connect_robot_on_connect = False`, as
     `tests/test_extrusion_job.py:143` already does). `connect_robot_on_connect`
     keeps its name but its docstring (`config.py:132-138`) is rewritten: it now
     governs auto-linking at Aim/Survey and before motion, not at Connect;
  5. **rechecks immediately before real motion** exactly as today: every run calls
     `ensure_robot_link(strict=True)` right before moving — strict mode *verifies*
     an existing (manual) link even when `connect_robot_on_connect` is off, and
     never attempts one then (`calibration/service.py:1038`,
     `scan/service.py:3079`, `extrusion/service.py:546`, `extrusion/measure.py:201`)
     and fails clearly if it cannot; the `MotionConfirmDialog` shows the current
     link state as one of its `checks`.
  `/api/rdk/status` keeps reporting the link state via `robot_connected()`, so the
  topbar Link pill stays truthful without linking on its own.
- **Concurrency.** One non-blocking `CellArbiter` (`core/arbiter.py`) gates every
  cell state transition — connect, link, job start, live-preview start — so
  Connect's busy check and a start can never interleave. Connect holds it for its
  whole duration and returns **409** while `jobs.running`, `live.running` or the
  camera lease is held (its error path resets the session —
  `calibration/module.py:186` — which must never happen under a running job); a
  concurrent second call gets 409 "connect in progress"; a job or preview start
  during a connect fails fast with 409 "cell is busy: connect". Link holds the
  arbiter too (up to `robot_connect_timeout_s`), so pages start the camera
  *before* linking.

The three module `/connect` routes are deleted in phase 4 once every page uses the
platform one (the web UI is their only client; `tasni.cli` does not call them).

**Job events and status are module-scoped, and every execution has an id.**
`JobEvent` gains `module`, `job_id` and `kind` (`core/events.py:16`): `job_id` is
unique per execution (uuid4, minted by `jobs.start`), `kind` is today's name
(`sim_tour`, `calibration`, `scan`, `extrusion-print`, …). `jobs.start(job, kind=,
module=)` stamps all three on `status`/`result`/`error` **and**, through the
`JobContext`, on `progress`/`log`/`frame` (`jobrunner.py:41-49`);
`LivePreview.start(..., owner=module)` stamps `module` on `frame`/`gate`/`boundary`/
`survey` (`livepreview.py:118`; the camera lease already tracks an owner string,
`camera_lease.py:29`). `jobs.start()` **returns the `job_id`**, and every start
endpoint (`/run`, `/poses/simulate`, `/quick-sim`, `/dry-run`, `/print`,
`/measure/characterize`, `/measure/layer`) returns it to the browser as
`{job_id}`. The runner keeps **history per (module, kind)** —
`last_jobs[module][kind] = {job_id, kind, status, result, error, started_at,
finished_at}` — because a module has several kinds (calibration: solve + tour;
extrusion: quick-sim, dry-run, print, characterize, measure) and today's single
global `status/result/error` (`jobrunner.py:66-69`) lets a later `sim_tour` erase
the Calibration solve a returning page hydrates its Apply button from. Each module
`/status` returns `running: {module, kind, job_id} | null` (whoever's job it is,
so a foreign one renders "Calibration job running — controls locked"),
`jobs: {kind: record}` for its own kinds, and its **authoritative workflow
fields** (calibration `applied`, scan lock/prepared state, extrusion
`fingerprint`/`preflight`/`dry_run_passed`/`live_print_enabled`) exactly as today —
workflow state is never inferred from "the last job". Pages act only on job events
whose `job_id` matches a job they started or hydrated, so delayed events from two
consecutive runs of one kind cannot mix. The runner publishes `status: running`
*before* the worker starts, and a page reconciles from its module `/status` right
after every start response and whenever its socket (re)connects — so a job that
finishes before the browser learns its id is still settled, and `running` never
sticks.

**Live-preview events are not job events.** `live.start(owner=module)` mints a
`stream_id` — or keeps one passed in for an *internal resume* (scan restarting its
preview after a five-position capture) — `/live/start` returns it, and
`frame`/`gate`/`boundary` events carry `module` + `stream_id` (no `job_id`);
`survey` and the locked `frame`/`gate` pair are request-path events (module only).
A page accepts id-less events of its own module and only the stream it started,
which also drops frames from a previous preview after a restart. A running
preview is owned by its module: another module's `/live/start` is refused (409).
**This is what makes the navigation/rehydration promise in §7 true**; it ships in
phase 0.

**Frontend.** `PlatformProvider` exposes
`{ health, rdk, connect(), connecting, subscribe(module, handler) }`.
`subscribe(module, …)` delivers a module only its own events (the topbar's Link
pill subscribes to all). `health` polls `/api/health` every 4 s as now. `rdk`
(`/api/rdk/status`) is refreshed: on mount; after `connect()`; on **every terminal
job event** (`result`, `error`, `status: cancelled` — completion does not publish
`status`, `jobrunner.py:97-104`); and by a slow poll (10 s) **that pauses while a
job is running** (its four RoboDK API calls must not contend with the job). When
`health.robodk.ok` turns false, `rdk.ready` drops immediately without waiting for
the poll. **Topbar** = brand · RoboDK pill ("connected — KR150 · Realsense tool ·
real robot ONLINE (10.x)") · Camera pill · Link pill · **Connect** button (label
*Reconnect* when ready; disabled while connecting or while a job/live preview runs,
with the reason as tooltip; errors shown as a topbar toast, not a page banner).
Pages call `usePlatform()`; a step that needs the cell is `locked` with
`lockReason: "Connect the cell (top right)"`. **No per-page banner, no per-page
Connect button.**

Module-specific prerequisites stay module-side and render as `PrereqChip`s in the
step that needs them: Scan — "Calibration on file · 2026-08-14 · PASS" (warn when
none, never blocks, as today); Extrusion — "Scanned surface · Frame_x · 1200 × 800
mm · inserted 2 h ago" or "none — run Scan first →" (link). Calibration has none
beyond the cell.

### 4.6 Guide pane

Every module page uses the same two-column layout: main column (rail + steps) and a
360 px sticky right pane, collapsing to a *Guide* toggle button under 1100 px.
`guide.ts` per module:

```ts
interface GuideSpec {
  intro: string;                       // one paragraph: what this module does
  steps: Record<string, {              // keyed by step id
    what: string;                      // what to do
    why?: string;                      // the explanation moved out of the step body
    physical?: string[];               // manual checkboxes: session-scoped, gate nothing
    slot?: ComponentType;              // module-specific widget (board print tools)
  }>;
}
```

The pane opens on the current step's section (others collapsed). Manual checkboxes
are for setup-only physical facts ("Board mounted rigidly, no glare", "Ring
placed"); they are session-scoped (in memory), reset on reload, and gate nothing.
Per-run safety assertions — cell clear, hands out — are **not** checklist items:
the `MotionConfirmDialog` ack is the only place they are asserted, every time. App
steps are not checkboxes — the rail is their state. Every paragraph currently inline
in a card (`Scan.tsx:1212-1217`, `:1029-1071` hints, `Calibration.tsx:465-472`,
`:545-561`, `Extrusion.tsx:626-633`, …) moves into `why`. **Step bodies keep at most
one instruction sentence.** Calibration's printable-board tools become the `slot` of
its *Board* step and are also linked from the guide.

### 4.7 Dashboard

1. **Readiness strip** — three linked cards in journey order, replacing the "Cell
   status" card (the topbar owns cell status now). They are fed by one new
   `GET /api/readiness` that distinguishes **recorded** (what `active.json` says)
   from **present** (what the open station contains now; `null` when no session is
   open or a job is running, so the check never contends with a run):
   - *Calibration*: recorded = `read_active("calibration")` (date, verdict, val px,
     tool); present = the camera tool exists **and** its pose matches the applied
     run's `X_cam2gripper` (an unsaved station reload silently loses an applied
     calibration while `active.json` still claims it). "Calibrated 2026-08-14 10:22
     · PASS · 0.9 px val" / "Not calibrated — Calibrate →". Links
     `/m/calibration?step=review`.
   - *Work surface*: recorded = `read_active("scan")` (frame, size, provenance,
     inserted date — scan writes it on insert, `scan/service.py:3435`); present =
     the frame and rectangle items exist in the station (`item_exists_as`, as
     extrusion's preflight already checks, `extrusion/service.py:94`). "Frame_x ·
     1200 × 800 mm · measured by camera · inserted 2026-08-27" / "No surface —
     Scan →". Links `/m/scan?step=review`.
   - *Print plan*: `GET /api/modules/extrusion/status` (in-memory module state that
     resets with the app — which is the truth the card should show) → "Plan a1b2c3…
     · preflight ✓ · simulated ✓ · dry run ✓ · live print enabled" / "No plan —
     Extrusion →". Links `/m/extrusion`.

   A card whose `present` is false says so ("recorded 2026-08-27 · **not in the
   current station** — re-insert"); `null` renders as "connect to verify".
2. **Modules grid** — unchanged (registry-driven).
3. **Recent runs** — rows become clickable and open a `RunDetail` drawer from a new
   `GET /api/runs/{module}/{stamp}` (`core/runs.load_meta` + `load_report`) showing
   module, stamp, artifact path (copy button), the run's `summary.txt` if present,
   and top-level scalar fields of `report.json` as a generic key/value table.

### 4.8 Copy and vocabulary

Rules: one instruction sentence per step body; imperative labels; the robot moves
only behind a `MotionConfirmDialog`; state words are shared across modules.

| Today | New |
|---|---|
| Connect & open Tasni station / Connect / Refresh items | **Connect** (topbar) |
| Dry run (simulate) / Geometry & station preflight / Complete validated dry run — collisions ON | **Simulate** (tour) · **Preflight** · **Full simulation (collisions on)** |
| Run calibration / Run scan / Print & record — LIVE ROBOT | **Run on robot** (dialog title says what: "Calibration tour", "Scan tour", "Live print") |
| Apply to tool / Insert into RoboDK / Apply to recipe & placement | **Apply** · **Insert into RoboDK** · **Use measurement** |
| Create targets / Lock & create targets / Generate coordinates & fingerprint | **Create targets** · **Lock surface** · **Generate plan** |
| boundary provenance: "camera measured – complete boundary" / "user specified – …" | Boundary: **measured by camera** / **entered by you** |
| acquisition mode | How it was measured: one view / five positions / entered |
| GENERATED · FINGERPRINTED / LIVE DRAFT · NOT ROBOT-READY | **Plan locked** / **Draft — generate to lock** |
| SURFACE READY / Surface ready / Hold position… (×3) | one step-header chip: POSITION → HOLD n % → READY → LOCKED |
| Cylinder Test (module) | **Extrusion** module; "Cylinder test" is the recipe preset name |
| Ring stack — measure only (no extrusion) | **Ring stack (measure only)** tab under Extrusion |

### 4.9 Visual system

`index.css` gains spacing (`--s1..--s5` = 4/8/12/16/24) and type (`--t-xs..--t-xl` =
11/12/13/15/20) tokens; no new colours. Rule for new code: no `style={{}}` except
computed values (progress width, opacity ramps). CSS is reorganised by component;
the page one-offs (`calib-*`, `extrusion-*`, `birdseye-*`, `survey-*`) stay until
their page is rewired, then are deleted or moved under the owning component.

## 5. Module step definitions

Summaries are what the collapsed step shows. Primary action in **bold**.

### 5.1 Calibration — 4 steps

| # | Step | Contents | Primary / done summary |
|---|---|---|---|
| 1 | Board | board preview, page select (A4/A3/Letter), Open PDF / Download, measured-square check, **scale ack** (hard gate, unchanged) | **Confirm print scale** · "8×6 @ 30 mm · A4 · scale verified" |
| 2 | Aim & targets | `PrereqChip` real-robot link (auto-linked on entry, §4.5); live frame + `AimHud` + `StreamStats`, lamps + LOCK, jog-frame note (one line; details in guide), `CollisionPanel` as a `PrereqChip` + expandable pairs, `ConeDiagram` in the guide slot | **Create targets** (locked until LOCK **and** `pose_live`) · "15 targets (TasniCalib_*)" — Change → Clear |
| 3 | Run | kv (robot/tool/camera/board), held-out count + refine toggle, `holdoutInvalid` warning, `RunControls` (Simulate → `TourResult` → Run on robot → dialog), progress, thumbnails | **Run on robot** · "solved 2026-08-28 14:02 · 15 poses" |
| 4 | Review & apply | `Metrics` + verdict banner | **Apply** · "applied · PASS · 0.9 px val" |

The cell-connected requirement locks step 2 (not step 1 — printing needs no robot).

### 5.2 Scan — 5 steps (full scan) / 4 steps (working frame only)

| # | Step | Contents | Primary / done summary |
|---|---|---|---|
| 1 | Intent | goal toggle, scope toggle, region W×H inputs inline when *Declared region* | **Continue** · "Full scan · entire platform" |
| 2 | Survey | `PrereqChip` real-robot link (auto-linked on entry, §4.5; lock is 409 without `pose_live`); live frame + `AimHud` (coverage dots, live boundary) + `StreamStats`; step-header chip POSITION/HOLD/READY/LOCKED; `SurfaceGuide` chips; lamps (mandatory + advisory); **one Boundary line** replacing `SurfaceModeNotice` + `LargeSurfaceNotice` + the extra survey button: "Boundary: measured (1200 × 800 mm, all edges in view)" / "platform overruns the view → [Survey five positions] [Declare region]" / "declared 1000 × 1000 mm [edit]"; `CollisionPanel` as chip; provenance chip once locked; `SurveyPanel` replaces the controls during a five-position survey (as today) | **Lock surface** · "Locked · 1200 × 800 mm · measured by camera · 12 s ago" — Change → Reposition |
| 3a | Targets *(full scan)* | accept region → targets; reference mode auto-completes this step ("Reference surface — rectangle placed directly") and hides step 4 | **Create targets** · "14 targets (TasniScan_*)" |
| 3b | Prepare frame *(frame only)* | `FramePrepPanel` (provenance chip, quality table, freshness) | **Prepare working frame** |
| 4 | Run *(full scan)* | kv (robot/camera/fusion), `RunControls` | **Run on robot** · "fused 14 views · 2026-08-28" |
| 5 | Review & insert | `ScanViewer` (full) or kv (frame-only/reference), mode/boundary/coverage kv, artifacts path | **Insert into RoboDK** · "inserted Frame_x" |

Changing intent after a lock re-derives: step 2 returns to `ready` (the backend
already invalidates the lock token). The five-position survey stays a sub-flow
inside step 2.

### 5.3 Extrusion — two tabs

**Print tab — 6 steps**

| # | Step | Contents | Primary / done summary |
|---|---|---|---|
| 1 | Placement | `PrereqChip` scanned surface; groups: *Station items* (print tool, work frame, inspection tool, inspection target + derive toggle) · *Position on surface* (center X/Y, build plane Z, **Center on scanned surface**) · *Orientation* (A/B/C, capture neutral, seed from TCP, max wrist rotation) · *Clearances* (approach/retreat); the World-frame and frame-mismatch warnings | **Continue** · "Tool_x in Frame_x · centred on scan 2026-08-27" |
| 2 | Recipe & plan | sliders + numbers, correction toggle, `BirdseyeStack` + layer rail, draft/locked chip, path length | **Generate plan** · "12 layers · r 60 mm · plan a1b2c3" — Change → Reset |
| 3 | Preflight | placement fit, inspection-pose note, IK sample (today's `io-note`s, as a kv) | **Run preflight** · "preflight ✓ · 48/48 reachable" |
| 4 | Simulate | layer picker (current/all/toggles), representative-layers approval, quick-sim result, **Full simulation (collisions on)** as secondary, live-collision-check toggle | **Simulate layers** · "L0,L5,L11 simulated · approved · full sim ✓" |
| 5 | Print | valve/hardware-approval note; `RunControls` with `level: "live-process"` dialog (material line + cell-clear ack) | **Run on robot** (red) · "printed 12 layers · trial t123" |
| 6 | Review | measured-layers table, **Create corrected plan** (returns the rail to step 3) | — |

The live-print hard gate is unchanged: `live_print_enabled` (quick-sim approval +
hardware I/O approval) gates step 5, and the dialog's ack replaces the inline
`confirmLive` checkbox.

**Ring stack (measure only) tab** — today's card moved verbatim (session, note,
characterize, measure layer with introduced offset, records table, paper summary).
Its own motion checkbox stays until the paper run is done (§9), then it adopts
`MotionConfirmDialog` with `level: "motion"`.

## 6. Data flow and state

Each module's `use<Module>State.ts` is the page's existing hooks, effects and
handlers lifted out of the render: fetches (`/config`, `/status`, `/targets`,
`/result`, …), the job-event subscription, live-preview handling, and the actions.
It returns `{ state, actions }`; `Page.tsx` does `const steps = deriveSteps(state)`
and renders `WorkflowRail` + one `Step` per non-hidden entry. New platform
endpoints: `POST /api/rdk/connect`, `POST /api/rdk/link` (§4.5),
`GET /api/readiness` and `GET /api/runs/{module}/{stamp}` (§4.7); one new module
gate on `/poses/generate` and `/surface/lock` (§4.5). The module-scoping of events and status
(§4.5) touches `core/events.py`, `core/jobrunner.py`, `core/livepreview.py` and each
module's `/status` route and `jobs.start` / `live.start` calls; no job logic changes.

## 7. Error handling

- API/job errors render as `RunError` inside the step that issued the action
  (dismissible), are appended to `LogPanel` (which auto-expands), and set
  `attention` on that rail step until dismissed.
- Connection lost mid-job: `health.robodk` fails → `rdk.ready` drops immediately
  (§4.5), the topbar pill goes red, the active step keeps its frozen progress and
  the Cancel button; the job's terminal event or the resumed poll restores state.
- Navigating away and back re-hydrates from the module's own `/status`, `/targets`
  and `/api/rdk/status`. Because events and status are module-scoped (§4.5), a
  running job is shown running in its own module's step, and another module's page
  shows "Calibration job running — controls locked" instead of adopting its
  progress. A job whose terminal event was missed (fast job, socket reconnect) is
  settled from the same `/status` record, so `running` never sticks.
- **Hard gates preserved verbatim**: print-scale ack, cell-clear ack, `holdoutInvalid`,
  gate-green for target creation (also re-checked server-side), lock freshness,
  `can_prepare_frame`, `can_insert`, `live_print_enabled`, all server-side
  fingerprint invalidation. **One gate is added** (§4.5): target creation and
  surface locking require a live pose (driver linked + monitoring), server-side.

## 8. Testing

- **Frontend gets its first test runner**: `vitest` + `@testing-library/react`
  (dev deps; `npm test`). Unit tests: `deriveSteps` for each module (table-driven
  state → expected rail, including hidden/locked reasons and intent changes);
  `MotionConfirmDialog` (Cancel focused, Enter does not confirm, ack required,
  "none" tour wording); `RunControls` gating; `WorkflowRail` state glyphs.
- **Integration tests** (vitest + testing-library with a fake `WebSocket` and
  mocked `fetch`), because the riskiest behaviour is between the pages and the
  platform, not inside a component: (a) events from another module are ignored and
  the page shows "other job running"; (b) reload/navigation during a job
  rehydrates the running step from `/status`; (c) a `health.robodk` failure drops
  readiness before the next rdk poll; (d) a terminal `result`/`error` refreshes
  `rdk`; (e) `?step=` seeding, including a hidden step.
- **Backend** (`pytest -k`, never the full suite): `test_platform_connect.py`
  (fake rdk: ready / tool missing / timeout / socket error → session reset; 409
  during a job, during live preview, and for a concurrent call; Connect never
  calls `connect_robot`), the existing `tests/test_robot_link.py` extended
  (`ensure_robot_link` consolidation; `/poses/generate` and `/surface/lock` return
  409 when the driver is not monitoring and pass when it is;
  `require_live_pose=False` bypass),
  `test_job_events_scope.py` (every job event carries `module`/`job_id`/`kind`, every
  live event `module`/`stream_id`; `jobs.start` returns the id every start endpoint
  echoes; two consecutive runs of one kind get distinct ids; the calibration solve
  record survives a later `sim_tour` and another module's run; `running` shape),
  `test_readiness.py`
  (recorded-vs-present for a deleted frame and a moved tool; `null` while a job
  runs), `test_runs_api.py` (run detail 200 / 404 / rejected segment).
- `npm run typecheck && npm run build` green at the end of every phase.
- **Cell validation** per phase (Appendix B). No motion code changes, so the checks
  are UI-only walk-throughs of each module.

## 9. Phasing

| Phase | Scope | Risk / notes |
|---|---|---|
| 0 | Backend: module-scoped events with `job_id`/`kind` + per-module `/status` records (§4.5), station-only `POST /api/rdk/connect` with 409/lock, `POST /api/rdk/link` + the live-pose gate on target creation / surface lock, `GET /api/readiness`. Frontend: `PlatformProvider` (filtered `subscribe`, rdk refresh rules), topbar Connect + pills; pages switch to `usePlatform()` and the filtered subscription and drop their banners (no other layout change) | Low–medium. Ships alone; it is the foundation every later promise rests on, so it gets the integration tests first. |
| 1 | `components/workflow/*`, `GuidePane`, tokens; **Calibration** rewired as the reference implementation; Dashboard readiness strip | Medium. Calibration is the best-understood page and the reference for the other two. |
| 2 | **Scan** rewired (largest page); Boundary line merge; `RunDetail` drawer + endpoint | Medium-high: most conditional states. `deriveScanSteps` tests carry it. |
| 3 | **Extrusion** rewired + Ring-stack tab; module retitled | **Not before the paper's cell run** (deadline 1 Sep 2026): the ring-stack flow is the paper's evidence path and must stay byte-identical until those numbers are captured. |
| 4 | Settings page (`GET/PUT /api/config` for: camera ip/port/resolution, board geometry + page, gate tolerances, jog inversion, pose counts/cone; saved with a "restart to apply" notice, since config is read at start-up); delete module `/connect` routes; final copy pass | Low. |

Each phase is a separate branch off `ux-overhaul` merged when its checklist passes.

## 10. Decisions taken (defaults; override at review)

1. Rail is horizontal above the steps (vertical would compete with the sidebar).
2. Only the selected step is expanded (an accordion invites the old "everything
   visible" page back).
3. Guide pane is always present on desktop (a help drawer would hide the setup
   checklist).
4. Advisory tour simulation (Calibration, Scan) lives inside the Run step as the
   secondary action, with the confirm dialog nagging when it was skipped; gating
   simulation (Extrusion) is its own step.
5. Module `/connect` routes are removed in phase 4 rather than kept as aliases.
6. The ring-stack experiment stays inside Extrusion as a tab rather than becoming a
   module.
7. Extrusion module title becomes "Extrusion"; "Cylinder test" names the recipe
   preset.
8. Scan's `scan-ready` bar and `SurfaceGuide` header are removed in favour of the
   step-header chip; the `SurfaceGuide` axis chips and the lamps stay.
9. The platform connect is station-only; the real-robot link is established
   automatically on entering Aim/Survey, hard-gates target creation and surface
   locking (server-side, `require_live_pose` escape hatch), and is rechecked by
   every run right before motion as today.
10. Guide checklist items are session-scoped and gate nothing; per-run safety is
    asserted only in the motion dialog.
11. Readiness cards show *recorded* vs *present* rather than trusting `active.json`.
12. Every job execution has a unique `job_id` (returned by every start endpoint);
    `kind` is separate; the runner keeps history per (module, kind); live previews
    carry a `stream_id` instead and are exempt from job-id filtering.
13. No page auto-connects: connecting the cell is only the topbar action (Scan's
    silent auto-connect on entry is removed).

## 11. Review log

2026-08-28 — second-agent review of the first draft. Every finding was verified
against the code and accepted; one remedy was changed (R3).

| # | Finding | Resolution |
|---|---|---|
| R1 | Events and `/status` are not module-scoped, so a page open during another module's job adopts its progress; the §7 rehydration promise was false | §4.5 *Job events and status are module-scoped*; §2.9; phase 0 |
| R2 | `rdk` state goes stale: completion publishes `result`/`error`, not `status`; `/api/health` is a TCP probe | §4.5 refresh rules (terminal events, paused slow poll, immediate drop on health failure) |
| R3 | Connect has no concurrency rules; linking at connect silently changes Extrusion's semantics | 409 + lock; Connect is station-only (first revision wrongly dropped linking altogether — see R9) |
| R4 | `active` mixed derived status with UI selection; flag persistence undefined | §4.2 `StepStatus` + `selectedStepId`; per-module store, physical acks reset on reload |
| R5 | "simulation is never a step" contradicted Extrusion's Simulate step | rule scoped to advisory vs gating simulation (§4.3, decision 4) |
| R6 | "links into the right step" was not designed | `/m/{id}?step=` seeds `selectedStepId` (§4.2, §4.7) |
| R7 | physical checks persisted in `localStorage` could appear ticked in a later session | session-scoped, gate nothing, cell-clear lives only in the dialog (§4.6) |
| R8 | the surface card trusts `active.json`, which records the last insertion, not station presence | `GET /api/readiness` recorded vs present, for calibration too (§4.7) |
| — | unit tests do not cover navigation-during-job, stale events, reload, connection loss, reconnect races | integration tests + backend scope/readiness tests (§8) |

Second round, same day — one blocker and two clarifications, all verified and
accepted:

| # | Finding | Resolution |
|---|---|---|
| R9 | "No linking at Connect" rested on a false premise: the link is what makes the model track the arm, so the Create-targets seed is the arm's actual pose (`config.py:132-138`, `calibration/service.py:303`, `scan/service.py:2286`); nothing gates on it today | Connect stays station-only; `POST /api/rdk/link` auto-called on entering Aim/Survey; server-side live-pose gate on `/poses/generate` and `/surface/lock`; recheck before motion unchanged; `link_real_robot` + `ensure_real_robot_link` consolidated, not deleted (§4.5, §5.1, §5.2, §7) |
| R10 | a single current/last-job record lets a Scan run erase the Calibration result a returning page hydrates from | per-module records (superseded by R12: per (module, kind), and `running` instead of `busy_with`) |
| R11 | `job` as a name lets delayed events from two consecutive runs of one kind mix | unique `job_id` per execution, `kind` separate; pages act only on their own `job_id` (§4.5) |

Third round, same day:

| # | Finding | Resolution |
|---|---|---|
| R12 | one last-job record per module: a later `sim_tour` would erase the Calibration solve, and Extrusion has five job kinds | history per (module, kind) + `/status` returns authoritative workflow fields separately from `running` (§4.5) |
| — | clarify: `jobs.start()` returns `job_id` and start endpoints echo it; live-preview events need their own id; "never touches hardware" is false once Aim links the driver; decision numbering ran 9, 12, 10, 11 | all four applied (§4.5, §4.3, §10) |

Fourth round, same day — review of the phase 0 implementation plan
(`docs/superpowers/plans/2026-08-28-ux-phase0-platform-foundation.md`); all accepted:

| # | Finding | Resolution |
|---|---|---|
| R13 | job events can precede the browser knowing `job_id` (fast job) → UI stuck | "running" published before the worker; post-start + reconnect reconcile from `/status` (§4.5, §7) |
| R14 | scan's internal preview restart would mint a new `stream_id`; boundary/survey unstamped | `live.start(stream_id=)` resume; boundary stamped; survey is request-path (§4.5) |
| R15 | Connect's busy check was not atomic with job/preview starts | `CellArbiter` gates connect/link/job/live transitions (§4.5) |
| R16 | strict link skipped verification when auto-link is off; error text conflated link with motion | strict verifies a manual link; text corrected (§4.5) |
| R17 | `ready` could go stale-green after a health failure; Connect ignored preview/lease | `ready = rdk.ready && health.robodk.ok !== false`; Connect disabled while the camera is in use (§4.5) |
| R18 | rehydration never cleared stale `running` | the module `/status` reconciler settles finished own jobs (§7) |
| R19 | tests missed the strongest promises; Scan auto-connected silently; spec marked "implemented" before validation | tests added (§8); decision 13; status changes only after cell validation |

## Appendix A — control mapping (today → new home)

**Shell / Dashboard**: topbar pills → topbar (RoboDK/Camera/Link) · Dashboard "Cell
status" card → removed (readiness strip + topbar) · calib-stamp → readiness strip ·
Recent runs → clickable → `RunDetail`.

**Calibration**: conn banner → topbar · Aim card (Start/Stop camera, lamps, jog note,
CollisionPanel, Create/Clear targets) → step 2 · Target spread card → guide slot of
step 2 · Run card (kv, holdout, refine, scale gate, warning, Dry run, Run, Cancel,
error, tour, progress, thumbs) → step 3 (scale gate lives in step 1; the Run step
shows "scale not verified" via `lockReason` until it is) · Quality metrics + Apply →
step 4 · Log → `LogPanel` · guide checklist + board tools → `GuidePane` + step 1
slot · confirm modal → `MotionConfirmDialog`.

**Scan**: conn banner → topbar · What do you need? → step 1 · Survey card (Start/
Stop/Refresh camera, HUD, SurfaceModeNotice, LargeSurfaceNotice, survey button,
SurfaceGuide, lamps, scan-ready, provenance chip, warnings, CollisionPanel,
SurveyPanel, Lock/Reposition/Accept region/Clear targets) → step 2 (+ step 3a) ·
Prepare working frame → step 3b · Run scan (kv, warning, Dry run/Run/Cancel,
error, tour, progress, thumbs; "Not used" text) → step 4 or hidden · Review &
insert → step 5 · Log → `LogPanel` · modal → `MotionConfirmDialog`.

**Extrusion**: conn banner → topbar · Station & motion → step 1 (grouped) · Recipe +
Live bird's-eye draft + Generate/Preflight/Reset → step 2 (Generate, Reset) and step
3 (Preflight) · Safety workflow (steps chips → rail; valve note, placement note,
inspection note, IK note → step 3; quick-sim layer picker + approval + quick-sim
result → step 4; live collision toggle → step 4; confirmLive + Print → step 5
dialog; log → `LogPanel`) · Ring stack card → Ring-stack tab · Measured layers +
Create corrected plan → step 6.

## Appendix B — cell validation checklist per phase

- **Phase 0**: Connect from the topbar with RoboDK closed (station opens, pills turn
  green, the Link pill reports the driver state; Connect itself does not link);
  switch between the three modules — none asks to connect again; enter Calibration
  Aim with the controller OFF — the link chip reads OFFLINE and Create targets is
  locked with "Real robot not linked"; power the controller, Retry — chip ONLINE
  (monitoring), Create targets unlocks; same for Scan Lock surface; start a
  Calibration dry run and switch to Scan — Scan shows "Calibration job running"
  and no foreign progress, and Connect is disabled until it ends; return to
  Calibration afterwards — its solve result and Apply are still shown, and they
  survive a further dry run too; kill RoboDK mid-session —
  pill goes red within one health tick, Reconnect works; delete the inserted frame
  in RoboDK — the Dashboard surface card says "not in the current station".
- **Phase 1**: Calibration end-to-end on the cell (board → aim → targets → simulate →
  run → apply) with the rail advancing on its own; Clear targets sends the rail back
  to step 2; Dashboard shows the new calibration in the strip.
- **Phase 2**: Scan in all three intents (frame-only compact, full scan, declared
  region) + one five-position survey; step 4 absent under frame-only; Insert updates
  the Dashboard surface card; RunDetail opens for the run.
- **Phase 3**: Extrusion dry path (placement → generate → preflight → simulate → full
  simulation) on the model; Ring-stack tab produces a measurement identical to the
  pre-refactor card (compare `paper-summary` output of one measure).
- **Phase 4**: change jog inversion in Settings, restart, HUD hints flip.
