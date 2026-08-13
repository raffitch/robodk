# Two-path workframe survey — live-cell defects

**Context:** first run of the five-position survey against the real KUKA + D435i,
2026-08-13, branch `calibration-improvements`. Two defects reported by the operator.
Defect 1 is **FIXED AND PUSHED**. Defect 2 is **NOT STARTED** — that is the work for a
fresh session.

---

## Defect 1 — centre capture crashed — FIXED (`b8aa4fd`, pushed)

**Symptom.** Jogged to roughly the surface centre, clicked *Measure*, got:

```
RoboDK/camera unavailable: only size-1 arrays can be converted to Python scalars
```

**Root cause.** `rdk.current_joints()` returns a `robomath.Mat` wrapping an **N×1 column
vector** on real RoboDK, but a plain flat list in every test fake. `Mat.__len__` reports
the **column** count (1 for a column vector), so
`np.asarray(mat, dtype=float)` builds a `(1, N)` array instead of the flat `(N,)` the code
assumed. Iterating that single row then hands `float()` the whole 6-element joint vector in
one call — numpy's "only size-1 arrays can be converted to Python scalars".

Location: `refresh_robot_state` in `tasni/modules/scan/survey_contract.py`.
Fix: `.reshape(-1)` on both joint reads, which normalises `(N,)`, `(1, N)` and `(N, 1)`
identically.

**Why 408 tests missed it.** Every fake `RdkIO` in the suite returns a plain Python list.
This is a fakes-diverged-from-reality defect, not a logic error. The new regression test
uses the **real** `robomath.Mat` class and is still hardware-free.

**Wider than it looked.** `refresh_robot_state` is also called by `lock_scan_surface`, so
the **compact lock path would have failed identically** on the cell — this was not
survey-specific.

**Also fixed (same commit).** Unexpected exceptions on the `/survey/*` routes were caught
by a generic handler that reported them as `RoboDK/camera unavailable` — blaming hardware
for a code bug — and swallowed the traceback entirely, which is why the message was
useless for diagnosis. They now log the full traceback server-side and surface as a 500
naming the real exception type. `/poses/generate` already had this treatment; the survey
routes did not.

**Verification:** `py -3.10 -m pytest -q` → 412 passed, 11 pre-existing warnings.

### Worth checking on the next cell run
Search for other places that consume RoboDK return values as if they were flat sequences.
`RdkIO` methods returning `Mat` (poses, joints) are the risk surface, and the same
list-shaped fakes hide the same class of bug everywhere. A targeted sweep of
`np.asarray(rdk.…)` and `float(…)`/`tuple(…)` over RoboDK returns is worth one pass.

---

## Defect 2 — live target readout lags ~10 s behind the robot — INSTRUMENTED, NOT DIAGNOSED

**Symptom.** Jogging toward the indicated Z target (300 mm), the on-screen feedback took
about ten seconds to reflect the new position. Aiming is impractical at that latency.

**Status.** The instrumentation described under *How to diagnose* is now implemented and on
by default. **No cause is established yet** — the next cell run produces the measurement that
picks between the candidates below. Do not fix before reading it.

### What a code read established (no hardware needed)

These are facts about the data path, and they reshape the candidate list:

1. **Telemetry rides a second, independent socket.** `MODE TELEMETRY` is served by
   `stream_telemetry` (`server/server_unicast_syncronous.py:572`) and consumed host-side by
   `_TelemetryReader` (`tasni/core/camera.py:305`) on its **own background thread** that keeps
   only `self._latest`. It is **latest-wins**, so telemetry can never queue up stale behind a
   slow consumer. A Jetson bottleneck can only present as a *low update rate*, never as a
   backlog of old payloads.
2. **Video also cannot backlog.** `stream.read(drain=True)` (`camera.py:361`) skips every
   buffered frame and returns the newest (64-frame safety cap). So a slow `analyze()` yields
   *low FPS with fresh frames*, not delayed frames.
3. **Consequence — a host stall and a slow Jetson produce the same visible symptom by
   different routes.** `analyze()` samples `telemetry_reader.latest()` once per iteration, so
   if the loop iterates every 10 s the readout updates every 10 s even though the telemetry
   itself is current. Both candidates stay live; only measurement separates them.
4. **The staleness check is a cross-machine wall-clock subtraction.**
   `live_scan_telemetry_payload` drops telemetry when `time.time() - stamp > 2.0`
   (`service.py:1309`), where `stamp` is `time.time()` **on the Jetson**
   (`server_unicast_syncronous.py:274`). A Jetson Nano has no RTC battery. If its clock runs
   more than 2 s *behind* the host, every payload is discarded, `metrics` comes back `{}`,
   and — because `LivePreview._loop` only publishes a gate `if metrics` (`livepreview.py:139`)
   — **the HUD stops updating entirely while the video keeps streaming**. This candidate was
   not on the original list and costs nothing to measure.
5. **Telemetry is computed inline on the Jetson's frame feeder**
   (`server_unicast_syncronous.py:844-920`), nominally every `SCAN_TELEMETRY_PERIOD_S = 1.0`.
   It is not on a separate thread there, so a slow plane-fit/trim/grid pass throttles the
   video feeder too.

### Cell finding 2026-08-13: TWO CONCURRENT CAMERA CLIENTS starve each other

Observed while running `tools/characterize_distance.py` with the Tasni live preview open
at the same time: the HUD's fps element dropped to **"no signal"** intermittently and read
**1-2 fps** when it recovered.

The server's own comments are **stale and misleading** here. `handle_client` says "this
server is single-threaded with `listen(1)`"
(`server/server_unicast_syncronous.py:715`), and the module is literally named
`server_unicast_syncronous`. The actual `main()` does `listen(5)` and spawns a
**daemon thread per client** (`:1076-1082`). So a second client is *accepted*, not
refused — and both threads then pull from the **one** RealSense pipeline via
`getFrames(pipeline, align, depth_filters)`.

Consequences on a Nano, all of which match the symptom:

- `wait_for_frames()` from two threads splits the stream, roughly halving each client's rate;
- `align` + the depth filter chain + `scan_plane_telemetry`'s plane fit run **per client**,
  doubling CPU on a board that was already the suspected bottleneck;
- `characterize_distance.py` uses `CameraClient.grab()`, which connects/reads/closes **per
  frame** (5 per capture), so each capture is a burst of new connections contending with the
  preview's long-lived stream.

**Operational rule until this is fixed: only one camera client at a time.** Stop the Tasni
live preview before running the sweep.

**This contaminates any latency measurement taken with both running** — check whether the
original ~10 s observation was made with a second client attached before trusting it as a
baseline. If a fix is wanted server-side, the honest one is to refuse or queue a second
client rather than silently degrading both, and per the standing constraints that needs the
Jetson proven as the cause first.

### Candidates and their signatures

| Candidate | Signature in the log line |
|---|---|
| Host RoboDK RPC stall (`camera_pose_T`, `robot_connected`) | `pose`/`driver` p50 or max high **and** `loop` interval high |
| Jetson telemetry cadence | `loop` and RPCs normal, `telemetry` rate far below 1 Hz |
| Clock skew tripping the 2 s drop | `dropped` climbing, `age` far from 0 (negative = Jetson ahead) |
| Hold/freeze logic | `held` ≈ frame count while the arm is being jogged |

Corrections to the original analysis, worth keeping straight:

- `camera_pose_T()` is **not** called per video frame. Both RoboDK calls sit inside
  `if metrics:` (`module.py:594`), and `live_scan_telemetry_payload` returns `{}` for frames
  with no fresh telemetry, so they fire at roughly the ~1 Hz telemetry cadence.
- Both calls are **more expensive than one RPC each**: every `RdkIO` helper re-resolves the
  robot via `self.rdk.Item(...)` first (`rdk_io.py:69`), making `camera_pose_T()` three
  round-trips (Item + `Pose` + `PoseTool`) and `robot_connected()` two.
- "At 6 fps this should be sub-second" was wrong arithmetic: `live_hold_release_frames=2`
  counts **telemetry** frames at ~1 Hz, so the release debounce is worth ~2-3 s, not
  sub-second. The conclusion (can't explain 10 s alone, could compound) still holds.
- Even a perfectly healthy loop refreshes the Z readout at only ~1 Hz. Judge any fix against
  that ceiling, not against video framerate.

**How to diagnose.** Start the scan live preview on the cell, jog the arm toward the Z
target, and read the `live-latency` lines from the app console (`tasni.scan`, INFO):

```
live-latency 5.0s: 30 frames (6.0 fps) | loop p50/max 165/180 ms | driver p50/max 2/40 ms |
pose p50/max 3/12 ms | telemetry 5 upd (1.00 Hz) age p50 120 ms | dropped 0 held 25 no-tel 0
```

Match it against the signature table. Implementation: `tasni/modules/scan/live_diag.py`
(`LiveLatencyProbe`), wired into `live_start`'s `analyze()`; cadence/off-switch is
`scan.live_latency_log_s` (default 5.0, `0` disables recording and logging entirely).

**Constraint on the fix.** If the answer is to move the RoboDK probes off the hot path
(background thread, longer cache, non-blocking probe), **preserve the behaviour Task 7 added**:
`pose_live` must still correctly report `false` when the driver is not mirroring the arm.
Do not fix the latency by deleting the liveness signal — that signal exists because
pose-derived X/Y readouts are otherwise presented as real-time when they are not.

If the Jetson is the proven bottleneck, the fix is not in the host — **report it and stop
rather than editing `server/`**.

---

## Standing constraints for either fix

- `py -3.10 -m pytest -q` from the repo root must stay green (426 now, 11 pre-existing
  deprecation warnings expected). `cd tasni\webui; npm run build` must stay clean if the
  frontend is touched.
- Tests must run without hardware.
- Do not touch `Tasni.rdk`, `macros/`, or `server/` unless the Jetson is *proven* to be the
  cause.
- Commit and push to `calibration-improvements`; the Jetson follows that branch.

## Related

- Design: [scan-workframe-two-path-plan.md](scan-workframe-two-path-plan.md) — see its
  Hardware validation TODO (§18) for items already known to be unverified on the cell,
  including the corner-aiming plane-selection risk and the survey-diagram chirality check.
- Build log: [scan-workframe-implementation-plan.md](scan-workframe-implementation-plan.md)
