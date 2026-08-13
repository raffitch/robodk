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

## Defect 2 — live target readout lags ~10 s behind the robot — NOT STARTED

**Symptom.** Jogging toward the indicated Z target (300 mm), the on-screen feedback took
about ten seconds to reflect the new position. Aiming is impractical at that latency.

**Leading hypothesis (must be verified, not assumed).** Task 7 added a driver-liveness
probe into the live preview loop:

- `tasni/modules/scan/module.py` ~line 603-606, inside `live_start`'s loop:
  `driver_ok = bool(services.rdk.robot_connected()[0])`, refreshed every 2 s
  (`last_driver_check`).
- `RdkIO.robot_connected()` (`tasni/core/rdk_io.py:165`) makes a **synchronous RoboDK API
  round-trip** (`ConnectedState()`).

If that RPC blocks while RoboDK is busy driving/monitoring a real KUKA, it stalls the whole
preview loop every 2 seconds. That would be a regression introduced by this work, not a
pre-existing condition.

`services.rdk.camera_pose_T()` is called **per frame** in the same loop (~line 594) and is
likewise a synchronous RoboDK RPC — that one predates this work, but under a live driver it
may now be far more expensive than it was in simulation, so it belongs in the same
measurement.

**Other candidates to eliminate before settling on a cause:**
- The payload hold/freeze in `stabilize_live_scan_payload` (`service.py`). Release needs
  `live_hold_release_frames` consecutive *moving* frames; if the driver is not mirroring,
  release depends entirely on the vision escape thresholds
  `live_hold_vision_distance_mm` (12 mm) / `live_hold_vision_tilt_deg` (10°). At 6 fps this
  should be sub-second, so it is unlikely to explain 10 s on its own — but it could compound
  a stall.
- The 2-second staleness drop in `live_scan_telemetry_payload`.
- Jetson-side telemetry rate: the server computes a plane fit, rectangle, density trim and a
  180×180 coverage grid per telemetry frame. If the Jetson is the bottleneck, the fix is not
  in the host — **report it and stop rather than editing `server/`**.

**How to diagnose.** Instrument the live loop: time each RoboDK call and the end-to-end
publish interval, and log them. That distinguishes "host RPC stall" from "Jetson telemetry
rate" from "freeze logic" in one run. Do not guess between them.

**Constraint on the fix.** If the answer is to move the RoboDK probes off the hot path
(background thread, longer cache, non-blocking probe), **preserve the behaviour Task 7 added**:
`pose_live` must still correctly report `false` when the driver is not mirroring the arm.
Do not fix the latency by deleting the liveness signal — that signal exists because
pose-derived X/Y readouts are otherwise presented as real-time when they are not.

---

## Standing constraints for either fix

- `py -3.10 -m pytest -q` from the repo root must stay green (412 now, 11 pre-existing
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
