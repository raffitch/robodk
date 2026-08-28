# Tasni UX overhaul — workflow shell, cell-level connect, journey dashboard (design)

Date: 2026-08-28. Status: **draft for operator review**. Branch `ux-overhaul`.
Audit basis: `tasni/webui` on `main` @ `9101aa1` (every file read; refs below are to
that revision).

## 1. Goal

One operator journey across the three modules, and one way of doing each thing.
After this work:

- every module page is a **stepped workflow** — a step rail, one expanded step, one
  primary action per step — built from the **same components** for connect,
  simulate, run, confirm-motion, errors and log;
- the cell is **connected once, from the topbar**, and every module reads that state;
- the **Dashboard shows the journey** (calibrated → work surface → print plan) with
  links into the right module step;
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
type StepState = "hidden" | "locked" | "ready" | "active" | "done";
interface Step {
  id: string; title: string; state: StepState;
  summary?: string;      // one line shown when collapsed/done
  lockReason?: string;   // shown in the rail tooltip and the step body when locked
  attention?: boolean;   // a dismissible error is pending in this step
}
deriveSteps(state: ModuleState): Step[]   // one per module, in steps.ts
```

- `deriveSteps` is a pure function of the state the page already fetches (rdk
  ready, config, targets, gate, tour, job status, result, applied). **Rendering
  never computes gating itself**; every disabled primary action has a `lockReason`.
- The current step is the first step that is neither `done` nor `hidden`, unless
  the operator has pinned another by clicking it (the pin clears when the derived
  current step advances).
- Steps with no server action (Board, Intent, Placement) complete on their
  Confirm/Continue button, recorded as a flag in the module state that clears
  whenever that step's inputs change (the scale ack already works this way).
- `hidden` steps are not rendered at all (Scan "Run" under *Working frame only*).
- State changes that invalidate later steps (clearing targets, repositioning a lock,
  changing intent, editing a generated plan) are already enforced server-side; the
  rail simply re-derives, so later steps fall back to `ready`/`locked` and their
  summaries clear.

### 4.3 Rail and step behaviour

- **Rail**: horizontal at the top of the main column; `n · title · state glyph`
  (`○ locked · ● ready · ◉ active · ✓ done`, plus `⚠` when `attention`). Click
  selects; locked steps open read-only with the reason. Under 760 px the rail
  stacks vertically.
- **One step expanded at a time.** A `done` step collapses to its summary line with a
  *Change* affordance that re-expands it (e.g. Aim: "15 targets (TasniCalib_*)
  — Change").
- **Expanding/collapsing never touches hardware.** The camera-live state is shown in
  the Aim/Survey step header ("● camera live · Stop") even when collapsed; Run stops
  the live preview exactly as today (`Calibration.tsx:299`).
- **Exactly one primary action per step**, right-aligned in the step footer;
  secondary actions sit left of it. The primary action's disabled state always
  carries a reason (footer text = `lockReason`).
- **Simulate is never a step of its own** (decision 10.4): it is the secondary
  action of the Run step, and running without it surfaces in the confirm dialog as
  "No dry run performed" — exactly today's nag.

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
`/api/rdk/status`: the calibration/scan polling implementation lifted verbatim
(open the station if RoboDK came up empty, poll up to `robodk.connect_timeout_s`,
reset the session on a socket error mid-load, check robot + camera tool,
best-effort `link_real_robot` — never blocks readiness, never moves the robot).
Response = the `/api/rdk/status` shape. `link_real_robot` already lives in
`core/rdk_io.py:65` (imported by both calibration and scan), so nothing moves.
Extrusion's variant ("without linking") is subsumed: linking is a driver connect on
the same shared session that the other two modules already perform.
The three module `/connect` routes are deleted in phase 4 once every page uses the
platform one (the web UI is their only client; `tasni.cli` does not call them).

**Frontend.** `PlatformProvider` exposes
`{ health, rdk, connect(), connecting, subscribe }`. `health` polls `/api/health`
every 4 s as now; `rdk` is hydrated on mount, after `connect()`, and on every job
`status` event. **Topbar** = brand · RoboDK pill ("connected — KR150 · Realsense
tool · real robot ONLINE (10.x)") · Camera pill · Link pill · **Connect** button
(label *Reconnect* when ready, disabled while connecting; errors shown as a topbar
toast, not a page banner). Pages call `usePlatform()`; a step that needs the cell is
`locked` with `lockReason: "Connect the cell (top right)"`. **No per-page banner, no
per-page Connect button.**

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
    physical?: string[];               // manual checkboxes (persisted per module in localStorage)
    slot?: ComponentType;              // module-specific widget (board print tools)
  }>;
}
```

The pane opens on the current step's section (others collapsed). Manual checkboxes
are for the physical world only ("Mount the board rigidly", "Clear the cell"); app
steps are not checkboxes — the rail is their state. Every paragraph currently inline
in a card (`Scan.tsx:1212-1217`, `:1029-1071` hints, `Calibration.tsx:465-472`,
`:545-561`, `Extrusion.tsx:626-633`, …) moves into `why`. **Step bodies keep at most
one instruction sentence.** Calibration's printable-board tools become the `slot` of
its *Board* step and are also linked from the guide.

### 4.7 Dashboard

1. **Readiness strip** — three linked cards in journey order, replacing the "Cell
   status" card (the topbar owns cell status now):
   - *Calibration*: `GET /api/runs/active?module=calibration` (exists) →
     "Calibrated 2026-08-14 10:22 · PASS · 0.9 px val · tool Realsense" /
     "Not calibrated — Calibrate →". Links `/m/calibration`.
   - *Work surface*: `GET /api/runs/active?module=scan` (scan writes it on insert,
     `scan/service.py:3435`) → "Frame_x · 1200 × 800 mm · measured by camera ·
     inserted 2026-08-27" / "No surface — Scan →". Links `/m/scan`.
   - *Print plan*: `GET /api/modules/extrusion/status` (the current session's plan —
     it is in-memory module state and resets with the app, which is the truth the
     card should show) → "Plan a1b2c3… · preflight ✓ · simulated ✓ · dry run ✓ ·
     live print enabled" / "No plan — Extrusion →". Links `/m/extrusion`.
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
| 2 | Aim & targets | live frame + `AimHud` + `StreamStats`, lamps + LOCK, jog-frame note (one line; details in guide), `CollisionPanel` as a `PrereqChip` + expandable pairs, `ConeDiagram` in the guide slot | **Create targets** (locked until LOCK) · "15 targets (TasniCalib_*)" — Change → Clear |
| 3 | Run | kv (robot/tool/camera/board), held-out count + refine toggle, `holdoutInvalid` warning, `RunControls` (Simulate → `TourResult` → Run on robot → dialog), progress, thumbnails | **Run on robot** · "solved 2026-08-28 14:02 · 15 poses" |
| 4 | Review & apply | `Metrics` + verdict banner | **Apply** · "applied · PASS · 0.9 px val" |

The cell-connected requirement locks step 2 (not step 1 — printing needs no robot).

### 5.2 Scan — 5 steps (full scan) / 4 steps (working frame only)

| # | Step | Contents | Primary / done summary |
|---|---|---|---|
| 1 | Intent | goal toggle, scope toggle, region W×H inputs inline when *Declared region* | **Continue** · "Full scan · entire platform" |
| 2 | Survey | live frame + `AimHud` (coverage dots, live boundary) + `StreamStats`; step-header chip POSITION/HOLD/READY/LOCKED; `SurfaceGuide` chips; lamps (mandatory + advisory); **one Boundary line** replacing `SurfaceModeNotice` + `LargeSurfaceNotice` + the extra survey button: "Boundary: measured (1200 × 800 mm, all edges in view)" / "platform overruns the view → [Survey five positions] [Declare region]" / "declared 1000 × 1000 mm [edit]"; `CollisionPanel` as chip; provenance chip once locked; `SurveyPanel` replaces the controls during a five-position survey (as today) | **Lock surface** · "Locked · 1200 × 800 mm · measured by camera · 12 s ago" — Change → Reposition |
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
and renders `WorkflowRail` + one `Step` per non-hidden entry. No new module
endpoints; new platform endpoints are only `POST /api/rdk/connect` (§4.5) and
`GET /api/runs/{module}/{stamp}` (§4.7).

## 7. Error handling

- API/job errors render as `RunError` inside the step that issued the action
  (dismissible), are appended to `LogPanel` (which auto-expands), and set
  `attention` on that rail step until dismissed.
- Connection lost mid-job: the topbar RoboDK pill goes red; the active step keeps its
  frozen progress and the Cancel button; reconnect + job `status` events restore
  state (today's behaviour, now in one place).
- Navigating away and back re-hydrates from `/status`, `/targets`,
  `/api/rdk/status` (today's `hydrateConnection`, now via the shared hook), so a
  running job is shown running in its step.
- **Hard gates preserved verbatim**: print-scale ack, cell-clear ack, `holdoutInvalid`,
  gate-green for target creation (also re-checked server-side), lock freshness,
  `can_prepare_frame`, `can_insert`, `live_print_enabled`, all server-side
  fingerprint invalidation.

## 8. Testing

- **Frontend gets its first test runner**: `vitest` + `@testing-library/react`
  (dev deps; `npm test`). Unit tests: `deriveSteps` for each module (table-driven
  state → expected rail, including hidden/locked reasons and intent changes);
  `MotionConfirmDialog` (Cancel focused, Enter does not confirm, ack required,
  "none" tour wording); `RunControls` gating; `WorkflowRail` state glyphs.
- **Backend** (`pytest -k`, never the full suite): `test_platform_connect.py`
  (fake rdk: ready / tool missing / timeout / socket error → session reset),
  `test_runs_api.py` (run detail 200 / 404 / rejected segment).
- `npm run typecheck && npm run build` green at the end of every phase.
- **Cell validation** per phase (Appendix B). No motion code changes, so the checks
  are UI-only walk-throughs of each module.

## 9. Phasing

| Phase | Scope | Risk / notes |
|---|---|---|
| 0 | `POST /api/rdk/connect`, `PlatformProvider`, topbar Connect + pills; pages switch to `usePlatform()` for `ready` and drop their banners (no other layout change) | Low. Ships alone; visible win immediately. |
| 1 | `components/workflow/*`, `GuidePane`, tokens; **Calibration** rewired as the reference implementation; Dashboard readiness strip | Medium. Calibration is the best-understood page and the reference for the other two. |
| 2 | **Scan** rewired (largest page); Boundary line merge; `RunDetail` drawer + endpoint | Medium-high: most conditional states. `deriveScanSteps` tests carry it. |
| 3 | **Extrusion** rewired + Ring-stack tab; module retitled | **Not before the paper's cell run** (deadline 1 Sep 2026): the ring-stack flow is the paper's evidence path and must stay byte-identical until those numbers are captured. |
| 4 | Settings page (`GET/PUT /api/config` for: camera ip/port/resolution, board geometry + page, gate tolerances, jog inversion, pose counts/cone; saved with a "restart to apply" notice, since config is read at start-up); delete module `/connect` routes; final copy pass | Low. |

Each phase is a separate branch off `ux-overhaul` merged when its checklist passes.

## 10. Decisions taken (defaults; override at review)

1. Rail is horizontal above the steps (vertical would compete with the sidebar).
2. One step expanded at a time (an accordion invites the old "everything visible"
   page back).
3. Guide pane is always present on desktop (a help drawer hides the
   physical-checklist items that matter for safety).
4. Dry run/simulate lives inside the Run step as the secondary action, not as its
   own step; the confirm dialog nags when it was skipped.
5. Module `/connect` routes are removed in phase 4 rather than kept as aliases.
6. The ring-stack experiment stays inside Extrusion as a tab rather than becoming a
   module.
7. Extrusion module title becomes "Extrusion"; "Cylinder test" names the recipe
   preset.
8. Scan's `scan-ready` bar and `SurfaceGuide` header are removed in favour of the
   step-header chip; the `SurfaceGuide` axis chips and the lamps stay.

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
  green, real-robot link note appears); switch between the three modules — none asks
  to connect again; kill RoboDK mid-session — pill goes red, Reconnect works.
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
