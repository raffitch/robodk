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
5. **[docs/superpowers/specs/2026-08-30-deposit-segmentation-design.md](docs/superpowers/specs/2026-08-30-deposit-segmentation-design.md)**
   — deposit segmentation is geometry now, not colour: the chroma gate is deleted,
   replaced end to end by a per-frame fitted substrate
   (`tasni/modules/extrusion/substrate.py`) with a derived threshold and a
   compactness filter (see §4). `docs/deposit-segmentation-handoff-2026-08-30.md`
   is the superseded diagnosis that motivated it — background only.

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
`heightmap`, `mesh`, `iso`, `profile` and `pipeline` (300 dpi PNG + vector PDF) plus a
per-trial `stack` and `tube`, from the archive alone: no robot, no RoboDK, no camera.
`mesh` (2026-08-29) is the surfaced view the old paper had — the frame meshed, from
above and rotated — which used to exist only as an interactive Open3D window in
`macros/3DScan.py` that wrote no file. Since 2026-08-30 they render on first view or
Word-draft generation, so plotting cannot delay robot return; serving is
render-if-missing, so takes archived earlier — including
`20260828-192115-47fb78ea` — produce figures with zero cell time. Click a take in the
measurement table to see them. Needs `pip install -e .[figures]`; without matplotlib the
measurement is unchanged and only the figures are skipped. Details and the two
correctness traps (deposit-band colour range; FITTED not averaged nominal centre) are in
`docs/extrusion-current-handoff.md`.

**2026-08-30 — top-view capture and live-job latency corrected.** A ring take now
captures **five consecutive RGB-D frames at the stationary inspection pose** and
processes their per-pixel nonzero depth median; the five raw depth frames are archived
as `depth-frames.npy`, the fused observation remains `depth.npy`, and the full burst
time is reported as `capture_ms`. The burst uses one held-open stream; changing greeting
telemetry such as device temperature is not mistaken for changed camera geometry, while
resolution, depth scale, intrinsics and extrinsics remain guarded. Layer 1 now uses the
same guarded ring-arc assembly as
Characterize (only layer 1: there is no lower ring to fuse into it). Session
`20260830-190622-925d7178` measured each trip in 7.6–8.6 s but then held the arm at the
inspection pose for 90–108 s of Matplotlib work per take; figures are now on-demand.
Each archived take appears in the UI immediately, and an invalid unattended take
returns home and stops the remaining trips. Median replay of the five old single frames
still reaches only 68.5 % completeness, so this is an evidence-backed improvement,
**not yet a cell proof that a single top view can measure this thin reflective ring**.
Restart the backend before the next take and check `/api/health` → `build.stale: false`.

**2026-08-30 19:56 session:** all four characterizations in
`20260830-195630-925d7178` correctly failed the shape gate although RGB shows a complete
pale ring on the green cutting mat. Offline replay measured 31–33 % maximum angular
coverage in each of the five raw depth frames, 30.6 % in their median and 30.6 % in a
closest-of-five composite: fusion did not discard a recoverable ring. Disabling the
legacy saturation gate raised the best candidate to only 56.9 % while admitting 883,087
background depth points, so removing the gate is not a safe fix. Measured conclusion:
the current colour/depth selection does not support this pale-ring/green-mat combination;
the 70 % completeness gate must remain. A replacement colour model needs evidence from
these archived frames, not a relaxed shape threshold.

> **Picking this up? Read [docs/live-print-next-session.md](docs/live-print-next-session.md)**
> — the continuation handoff: current state, the full decision tree for each bisect
> outcome, where the code is, and the fallback options with their real costs.

**2026-08-30 — multi-view inspection: DESIGNED (revamped), not built.** The mock rings are
thin and one top-down frame under-samples them, so the operator wants — as an optional
toggle, default OFF — a top view plus three **15°**-tilted views at 120° azimuths merged
into one work-frame cloud. Spec:
`docs/superpowers/specs/2026-08-30-multiview-inspection-design.md`; plan:
`docs/superpowers/plans/2026-08-30-multiview-inspection.md`. This **supersedes** the
2026-08-29 pair (`a1cafa0` / `c31c720`), retired because protocol 2 removed the ≈2 % scale
mismatch their registration existed to correct, the voxel moved 2 mm → 1 mm, and the side
photo half **shipped** meanwhile with taught targets.

Three things to know before touching it. (1) **Do not start before the PFH paper's cell run
is done** — it edits the shared capture path and redefines `acquisition_to_path_ms`.
(2) `measure.py:203 depth_plane_check` assumes a straight-down view and **rejects tilted
frames above ~18° at 300 mm, lower at longer standoffs**; the fix reads the incidence off
`T_work_camera` (`cos = -T[2,2]`) so tilt 0 reduces exactly. (3) **ChArUco is out of scope
by operator decision** — it belongs to hand-eye calibration only; registration is
ring-first (level on the surface annulus, then a gauge-fixed joint solve against one
shared circle).

**2026-08-29 — detection error is PAIRED.** A top ring "placed true" by eye sits 1–3 mm
from the plan centre before anything is introduced, so scoring a shift against the plan
centre folds the operator's placement error into what the paper calls the chain's error.
`paper_summary` now also reports the **paired detection error** — the measured shift
relative to the same layer's last valid zero-offset take *before* the displacement,
against the typed vector — and the prose/Word draft quote that one ("measured against
the ring's own last measured position before it was moved"). The Run guide's "which way
is +X" throwaway is recorded with phase `axis check` and is never a pairing reference;
the 15 mm condition is 15 mm from the *original* marks. The draft's timing gap now counts
LIVE measurements (an offline reprocess owes one more). See
`docs/pfh-paper-handoff.md` §3 *Paired detection error*.

**2026-08-28 — the summary is scored against ground truth.** `paper_summary` now reports
**detection error** (`|measured centre offset − the offset the operator typed|`) per
condition and machine-checks the pure-shift relation (`max = d`, `mean = 2d/pi`,
`RMS = d/sqrt(2)`), printing a `WARNING` naming any statistic that disagrees instead of
averaging a bad condition in. A take archived before the offset *vector* existed is
excluded rather than counted as perfect. Figures draw the **ground truth ring** (nominal
+ introduced offset) in `plan` and `stack`. Replaying the 2026-08-28 capture: 15.38 mm
detection error against its "no offset" annotation (the skipped *Apply*), and 0.002 mm
once re-labelled with the displacement that was really there.

**2026-08-28 night — board depth noise fused to the ring, fixed.** The first paper
take after a correct Apply (`20260828-204846-5b455377/layer-001`) failed with `branch
guard exhausted`: 22.7 % of the bare ChArUco board clears the 2.5 mm deposit floor at
300 mm, joins the ring's cluster, faces up like any flat surface, and dilates into a
lobe with a 37 mm skeleton arm. Raising the floor is NOT a fix (3 mm read r 36.7 for a
42.6 mm ring, "valid"). `processing._radial_trim` now keeps only points within a
tightening band of the FITTED circle (`radial_trim_schedule_mm = [15, 12, 10]`, the only
schedule that passed every real and synthetic case). `reprocess_saved_layer` now scores
a take against its own recipe / archived nominal centre / manifest provenance instead of
the pre-Apply `trial.json`; the failed take reprocessed to r 42.31, offset 1.28 mm —
the first zero-offset baseline. Fixture: `tests/fixtures/extrusion/ring2/`.
**After the restart press Apply to recipe & placement first** (the plan is in-memory
only; Center-on-surface → Generate would rebuild the stale pre-Apply plan).

**2026-08-28 — the measure-only journey now defends itself** (`8ed25fd`, `b7c746b`,
pushed to `main`; 140 extrusion tests green). Reprocessing a take rejoins the session
(`session.json` gains the record and `tops[N]`, so the next layer keeps its floor —
the paper trial had a valid layer-001 and an empty session); offline runs keep the
capture time they were measured with, never claim an acquisition-to-path they did not
produce, and are excluded from the cycle statistic; a session is **bound** to the plan
it applied, and measuring against any other is refused by name; that plan is **restored
after a backend restart** (and Apply now applies onto the session's own trial rather
than config defaults, which do not name a tool here — the documented "press Apply
first" used to raise a validation error); layer N refuses until layer N−1 has a
measured top; invalid takes stay visible with a **Reprocess** button and are never
averaged; conditions group by **layer + phase + offset**, so the noise floor is not
pooled with placement repeatability. The Extrusion page carries a live **Run guide**.
**Trap corrected:** the ChArUco square pitch is NOT a usable ruler for the introduced
offset on this cell — the board is A3 8×6 at **40 mm** squares and 40 mm is past the
25 mm cap (the ring leaves the ±30 mm search band and the take is invalid). Use a steel
rule along a board edge and type what was actually achieved.

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

**Sensor layer (opened 2026-08-29, nothing started):**
[docs/realsense-capability-audit-2026-08-29.md](docs/realsense-capability-audit-2026-08-29.md)
measured that the Jetson service loads an **unoptimised, non-CUDA librealsense 2.53.1 debug
build** (not the apt 2.55.1), delivers depth quantised to **1 mm** where the sensor resolves
~0.2 mm at the working standoff, discards ~50 % of the depth field by aligning to colour on
the Jetson, and runs an **unrecorded Custom preset** that nobody can reproduce. Twelve
findings R1-R12 with a dependency-ordered sequence; R4.1 (record the as-found advanced-mode
JSON) is read-only and should go first because it makes the 2026-08-13 characterisation
reproducible. Nothing else there lands on the cell before the 1 Sep paper deadline.

**Deposit segmentation: the chroma gate is DELETED, replaced by geometry (opened
2026-08-30, shipped 2026-08-31, branch `deposit-segmentation`):** the saturation gate that
inverted on the cell (bead 25 vs board 28) and `deposit_floor_mm`'s constant work-frame-Z
floor are both gone, along with `floor_profile` (referencing layer N-1's own measured top
was tried and measured WORSE -- completeness 0.62 -> 0.50 -- and deleted, not kept as a
fallback). Segmentation is now geometric and per-frame: `substrate.PlaneSubstrate`
(`tasni/modules/extrusion/substrate.py`) fits the surface the deposit rests on in THIS
frame, the height floor is `clamp(substrate_sigma_k x sigma_mm, substrate_floor_clamp_mm)`
-- derived from that frame's own measured noise, not a constant -- and `compactness_filter`
takes over the gate's one defensible job (rejecting contamination on shape, not colour).
The colour frame is still captured and archived as evidence; nothing in the decision path
reads it. Design:
`docs/superpowers/specs/2026-08-30-deposit-segmentation-design.md`; the original diagnosis is
now background history at `docs/deposit-segmentation-handoff-2026-08-30.md`. Validated
offline on the full 2026-08-30 archive (`tests/test_extrusion_golden.py`): 8/8 layer-1 takes
valid, radius mean 40.980 mm / sigma 0.074 mm (better than the design prototype's own 0.107),
completeness 0.9923-0.9928, with no colour input at all; the 3 layer-2 takes remain correctly
invalid. Merge to `main` is a separate decision, still pending.

**Layer >= 2 was ALSO a segmentation artifact, fixed 2026-08-31 (deposit floor under
layer N).** The line above used to add "a badly stacked physical ring, not a segmentation
artifact" about the layer-2 takes. That was half right and the wrong half was load-bearing.
Layer 2 had never once been measured validly -- **0 of 6 takes** across both cell sessions --
and replaying the 2026-08-31 archive stage by stage put the loss in one place: the ring
reaches DBSCAN whole (36/36 angular bins in the work ROI) and leaves it as 5-7 arcs, because
a hand-placed ring 2's crest swings ~10 mm around the circumference and the 3D neighbourhood
breaks where it steps. Layer 1 fragments the same way and was rescued by arc assembly;
above layer 1 assembly was off, so the **largest arc alone -- 110 deg of it -- became the
ring**. Completeness 0.294 was never a statement about that ring.

The fix is a **deposit floor under layer N at the top of layer N-1**, applied to the deposit
population after the compactness filter (NOT to the ROI band -- measured, that starves
compactness and makes the take worse). With the layer beneath removed, arc assembly is safe
to enable everywhere, which is what recovers the ring. Two traps live here:

- The two archives move in OPPOSITE directions and both are correct. 2026-08-31 goes
  0.294 -> 0.515 (it was reporting one arc); 2026-08-30 goes 0.62 -> 0.50 (it was padding
  the ring with ring 1's crest where ring 2 was thin). The confirmation is the fitted radius:
  spread across three repeat takes of ONE physical ring collapses 7.24 mm -> 0.29 mm on the
  2026-08-31 stack, and lands within 0.3 mm of layer 1's own radius on 2026-08-30.
- **Layer 2 is still INVALID and must stay that way.** `tests/test_extrusion_golden.py`
  holds `LAYER2_MAX_COMPLETENESS = 0.75` for exactly this: on the 2026-08-30 stack the
  circumference genuinely is not there, and a change that "recovers" it is measuring
  something that does not exist. Do not raise that ceiling.

What is NOT fixed, and is the real blocker for measuring a stack from one pose: the
2026-08-31 layer-2 crest still carries a contiguous ~50 deg sector the chain cannot use.
Counting ROI points per 10 deg sector in the bead annulus of that frame, the 140-190 deg band
falls to 22-121 against 250-466 everywhere else in the SAME frame (layer 1 of the same
archive: 60-377). A 19 mm stack seen from one top-down pose shadows itself. That is a
CAPTURE problem and it is what `docs/multiview-inspection-*` exists for.

(Do not confuse that with the 2026-08-30 archive, where `test_extrusion_golden.py` records
200-530 **valid depth pixels** per 10 deg sector -- a different measurement on a different
frame, and the reason those takes are the false-positive guard rather than this one.)

**Also fixed on the way:** offline reprocess scored takes against `trial.json`'s **pre-Apply**
setup, so its `build_plane_z_mm` was 4.259 mm where every applied path says 0.0 -- the whole
height band sat 4.26 mm above where the live run had it. Harmless until now (the ROI margin
is 15 mm) but fatal to a floor derived from `nominal_z_mm` inside a 4.6 mm layer.
`reprocess_saved_layer` now shares `plan_for_archived_take` with the figures and the golden
harness instead of keeping its own drifted copy.

**NEXT on the cell: spatial-filter A/B, then the controlled A-B-A roll probe if needed.**
**[docs/inspection-roll-probe-handoff.md](docs/inspection-roll-probe-handoff.md)** is the
task page. First use the already-built `RS_SPATIAL=0` control on the untouched two-ring
stack: the stock spatial filter is the only untried lever with a measured direct mechanism
for the ~1.5 mm crest-height shortfall. If it does not recover the sector, capture
**+30 / -30 / +30** (three excursions, ten raw frames each, analyze the last five), freezing
the complete applied plan and reading achieved roll off `T_work_camera`. The old pair was
not merely a mismatched noise floor: its recipes used 8.9 vs 15.0 mm bead diameter, which
moved the absolute camera centre **6.1 mm**, so it was not one viewpoint. The existing
`tools/probe_roll_pair.py` reads the board halo only; a sibling must count the layer-2
dropout by 10 deg sector and directly correlate substrate residual maps before anyone can
claim the static noise decorrelates. The probe can be written after capture because
`depth-frames.npy` preserves the evidence, but no halo verdict substitutes for it. Also:
90 deg was requested last time and the planner delivered 60; never weaken the spin limit.

**Built, never used, never merged: multi-view inspection.**
`origin/worktree-multiview-inspection` @ `96a17f6` (worktree at
`.claude/worktrees/multiview-inspection`) is **17 commits and +3659 lines** -- the star
capture, the shared-circle registration, `tools/multiview_ab.py` (offline A/B, no robot
time), 1356 lines of tests and a written cell A/B protocol. It is 45 commits behind main
and predates the halved voxel, the radial-trim fixed point and the layer-N deposit floor,
so its "1 mm voxel" count trap is stale. It also carries a fix **main still lacks**:
`depth_plane_check` taking incidence from the actual pose instead of assuming a
straight-down view -- without it main's arrival gate rejects any frame tilted past roughly
18 deg at 300 mm, so no tilted capture works on main at all. Worth cherry-picking on its
own merits. Do not rebuild any of this from scratch.

**OPEN, not diagnosed: the backend hard-crashes.** Seven times across 2026-08-30/31 -- five
sharing one ntdll access violation (WER bucket `755b2d74...`, fault offset `0x2f6a3`, inside
the NT heap manager = native heap corruption), once as BEX64. Tonight's was pid 2680 at
19:58:56, ~2.5 s after the measure job wrote `session.json`. It is NOT the segmentation code
(66 replays of the archived takes on a worker thread with Qt loaded, no crash). `start.ps1`
captures stdout/stderr, but an access violation unwinds no Python frames, so the log just
stops mid-poll and the UI shows "Backend not responding". `faulthandler` is now armed in
`tasni/__main__.py` -> **`%TEMP%\tasni-backend.crash.log`** (append; `TASNI_CRASH_LOG`
overrides): it dumps every thread's Python stack on the fault. **Read that file first after
the next crash** -- the process holds Open3D, OpenCV, onnxruntime, PySide2/Qt (via robolink),
numpy/scipy and matplotlib, and the faulting module alone cannot say which was on the stack.

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
