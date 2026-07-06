# Live robot testing & debugging (RoboDK + real KUKA + Jetson)

How to drive the **real** KUKA from a script to exercise/verify the app (e.g. move
the camera and read the live HUD), without the traps that cost a lot of time on
2026-07-06. Read this before writing any script that moves the robot.

If you only need to *read* state (joints, gate telemetry) do the read-only checks in
§2–§4 and never touch run mode.

## 1. The stack at a glance

| Piece | Where | Notes |
|---|---|---|
| Backend (FastAPI) | `http://127.0.0.1:8000` | `py -3.10 -m tasni --port 8000` (dev) or `serve.ps1` (prod) |
| Frontend (Vite dev) | `http://127.0.0.1:5173` | proxies to :8000; browser reconnects on backend restart |
| Event stream | `ws://127.0.0.1:8000/ws` | JSON `{type, payload}`; `gate` events carry the HUD readout |
| RoboDK API | attach, `:20500` | the app binds the running GUI (`RdkSession` mode `attach`) |
| Jetson camera | `10.12.171.70:1024` | H.264 + scan-telemetry stream; flaky Wi-Fi |

Health/probe (read-only):

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/rdk/status
```

Restart just the backend (host code changes; browser/Vite survive):

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='py.exe'" |
  Where-Object { $_.CommandLine -like '*-m tasni*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
# then relaunch (background):  py -3.10 -m tasni --port 8000
```

## 2. Golden rules (the traps that bit us)

1. **SIMULATE vs RUN_ROBOT decides model-only vs real motion.** In `SIMULATE` (run
   mode 1) `MoveJ` moves *only the RoboDK model* — the real arm does **not** move.
   You can burn an entire test thinking the robot moved when it didn't. The tell:
   a Z move that should change standoff but the distance readout doesn't budge.
   Real motion needs `RUN_ROBOT` (6) **and** a driver in `ROBOTCOM_READY`.
2. **The model can drift from the real arm in SIMULATE.** Model-only moves leave the
   *model* at a phantom pose while the *real* arm sits still. `connect_robot` (driver
   link + position monitoring) **re-syncs the model to the real arm** — the model can
   suddenly "snap" to the true joints. Always trust the real arm, not a drifted model.
3. **Guard every mover with `if __name__ == "__main__":`.** A helper that calls
   `main()` at module scope will run the whole move sequence the moment another script
   `import`s it (e.g. to reuse its `TESTS` list). This actually happened.
4. **The live gate can be a STALE frozen value.** The anti-jitter hold freezes the
   HUD readout while the arm is parked. Right after a move, a naive "grab the first
   `held` frame" can return the *pre-move* frozen value. Read fresh (§4).
5. **Scan telemetry stalls on Wi-Fi.** Symptom: no `gate` events at all (`/ws` silent
   for scan). Fix: restart the preview (`/live/stop` then `/live/start`). It tends to
   stall right after robot-state changes, so **set run mode + driver first, restart
   the preview, then move**.
6. **Always return to the exact start joints** in a `finally`, at reduced speed, after
   verifying reachability. **Restore the run mode you found** (usually SIMULATE) when done.

## 3. Read-only pre-flight (NO motion)

Before moving anything, confirm the state and that every target pose is reachable and
sane. This never commands motion.

```python
from robolink import Robolink, RUNMODE_SIMULATE, RUNMODE_RUN_ROBOT
RL = Robolink()                                   # attaches the running GUI
robot = RL.Item("KUKA KR150 R2700")
print("joints:", [round(v,3) for v in robot.Joints().list()])
print("run mode:", RL.RunMode())                 # 1=SIMULATE, 6=RUN_ROBOT
print("driver:", robot.ConnectedState())         # (0,'Ready') == ROBOTCOM_READY
```

**Harness / IK correctness check (no motion).** Prove `solve_joints_for_pose` actually
places the *camera* where you command, by forward-kinematic-ing the solved joints back:

```python
from tasni.core.config import load_config
from tasni.core.session import RdkSession
from tasni.core.rdk_io import RdkIO, pose_to_T
rdk = RdkIO(RdkSession(load_config().robodk))
mount = rdk.use_camera_tool("Realsense")          # flange->camera 4x4
J0, T0 = rdk.current_joints(), rdk.camera_pose_T()
def camera_fk(j): return pose_to_T(rdk.robot().SolveFK(j)) @ mount   # base->camera
T1 = T0 @ delta                                   # delta in the CAMERA/tool frame
j  = rdk.solve_joints_for_pose(T1, J0)
Tfk = camera_fk(j)
# pos_err and rot_err should both be ~0 before you trust a real move.
```

On 2026-07-06 this reported 0.0 mm / 0.00° for all test poses — so muddy HUD results
were the *surface fit*, not the mover.

## 4. Reading the HUD live (`/ws` gate stream)

Gate `payload` fields that matter for aiming:

- `detected`, `held`, `live`, `ok`, `surface_mode` (`full`/`crop`), `fully_framed`
- `distance_mm`, `ideal_distance_mm`, `tilt_deg`, `tilt_b_deg`, `tilt_c_deg`, `yaw_a_deg`
- `move_cam = [Δx, Δy, Δz]` — surface centroid X/Y (camera frame) + `distance - ideal`
- `gates = {detected, distance, angle, center, framed, edge}`
- `extent_mm`, `rectangle_size_mm`, `crop_size_mm`, `outline_uv`, `points_uv`

Notes: `center`/`yaw_a`/`framed` are only meaningful for a **framed finite platform**
(`surface_mode == "full"`, `fully_framed == true`). In `crop` mode (surface overruns
the view — e.g. a big/close/white object) `move_cam` X/Y read 0, so you **cannot** test
centering there.

**`ideal_distance_mm` is a STABLE target, not a live recompute** (fixed 2026-07-06). It
is the standoff that frames the *physical* surface with a border (`frame_margin`, 1.12),
which depends on the object's size — **not** on where the camera is now — so a parked
operator sees a steady number. In `crop` mode the target **HOLDS** the value latched
while the surface was framed (so a small over-nudge into overrun doesn't move the goal);
it only falls to `accurate_min_mm` for a **genuinely oversized surface that was never
framed**. If you see the RANGE target jump around as you dolly in/out, that is a
regression — it should barely move. (The old bug: target `~592` → move toward it →
overrun flips to `crop` → target collapses to `300` → drives you even closer, unreachable.)

**Fresh read after a move** (avoids the stale frozen value): wait to *see the hold
release* (`held == False`, the move registered) then take the next `held == True`
(re-settled at the new pose).

**Stale projection escape hatch (operator + scripts): `POST /api/modules/scan/live/refresh`.**
The hold releases automatically when RoboDK mirrors the arm (pose gate) or on a
dolly/tilt (vision escape). The one gap is **driver-not-monitoring + a pure lateral
jog** — neither trips, so the frozen overlay keeps projecting the old pose. Refresh
drops the hold + re-anchors so the reading re-settles at the current pose (video keeps
streaming; distance target stays continuous — no flash to `accurate_min`). In the UI
it's the **"Refresh view"** button next to Stop camera. Returns 409 if no preview is
running. Watch it work: `static_frames` resets to 0 and `held` drops, then climbs back
to a fresh `held == True` at the current pose.

**Best pattern — continuous monitor + move in an executor** (this is the one that
worked; `focused_hud_test.py`):

```python
async def main():
    async with websockets.connect("ws://127.0.0.1:8000/ws", max_size=None) as ws:
        rec = asyncio.create_task(recorder(ws))          # appends every gate frame
        loop = asyncio.get_event_loop()
        await asyncio.sleep(6)                            # baseline
        await loop.run_in_executor(None, move_to, delta) # REAL move (blocking MoveJ)
        await asyncio.sleep(8)                            # watch it release->track->re-hold
        await loop.run_in_executor(None, go_home)
        ...
```

You will see the readout hold steady when parked, drop to `held=False` and smoothly
track during the move, then re-settle and re-hold at the new pose. That *is* the
anti-jitter hold working correctly.

## 5. Driving the real robot safely

```python
rdk = RdkIO(RdkSession(load_config().robodk))
rdk.use_camera_tool("Realsense")
rdk.robot().setSpeed(-1, -1, 12, -1)                 # ~12 deg/s joints, gentle
mode = rdk.apply_run_mode("run_robot")               # REAL motion
ready, msg = rdk.connect_robot(cfg.robodk.robot_ip)  # driver link + monitoring
assert mode == "run_robot" and ready                 # else abort, do NOT move
J0 = rdk.current_joints()                             # exact home for return
T0 = rdk.camera_pose_T()                              # base->camera now
try:
    for delta in deltas:                             # delta = tool/camera-frame 4x4
        T1 = T0 @ delta
        if not rdk.is_reachable(T1): continue
        j = rdk.solve_joints_for_pose(T1, J0)
        if j is None: continue
        rdk.move_j_joints(j)                          # moves the REAL arm (blocks)
        ... read fresh HUD ...
        rdk.move_j_joints(J0)                          # return home each test
finally:
    rdk.move_j_joints(J0)                              # always return
    rdk.apply_run_mode("simulate")                     # restore what you found
```

- **Camera-tool IK:** use `RdkIO.solve_joints_for_pose(T, seed=J0)` — it passes the
  camera tool to `SolveIK` explicitly and seeds the branch. Raw `robolink.SolveIK` has
  historically ignored the seed / solved for the flange (wrong by ~130°).
- **Tool/camera-frame move:** `T1 = T0 @ delta` (post-multiply). `delta` = translation
  `trans(dx,dy,dz)` or rotation about a camera axis. The jog hints are the camera
  optical frame: **X right, Y down, Z forward (toward surface)**.
- **Safe directions:** `−Z` (away from surface) and small tilts are always safe; be
  careful with `+Z` (toward the surface) and large lateral moves near fixtures.

## 6. Verified feedback semantics (2026-07-06, on the real arm)

- **RANGE / distance:** exact. Camera −100 mm (away) → `distance_mm` +98 mm; return →
  baseline. `move_cam[2] = distance - ideal`; jog toward the sign to correct. The
  `ideal` (target) is a stable framing standoff — it should hold steady as you dolly,
  and it does NOT collapse to `accurate_min` when you nudge slightly too close (see §4).
- **TILT / level:** correct magnitude/axis. Camera +10° about tool-Y → `tilt_deg`
  ~2→9.6°, `tilt_b_deg` ~0→9.2°; standoff unchanged during a pure rotation.
- **HOLD:** freezes when parked, releases + tracks on real motion, re-settles — good.

**Unverified — the sign/direction mapping to the operator's jog.** The rotations above
were in the *camera optical* frame, so only magnitude is confirmed. Because of the
~180° camera→flange mount, the "X right / Y down" jog hints may be inverted vs the
KUKA **TOOL** jog frame. `ScanConfig.jog_invert_x/y/z` (all `False` today) exist to
flip them. **Spot-check on the pendant:** jog one axis, confirm the HUD guides *back
toward centre*; if an axis points the wrong way, flip its `jog_invert` flag.

**Measurement caveat:** a large low-texture **white** surface filling the frame gives
patchy RealSense depth → a partial/unstable plane fit (`extent_mm` jumps, centroid
doesn't track). Centering/framing feedback is only trustworthy on a **framed** surface
with a stable fit (a smaller platform, more overhead, lower tilt).

## 7. Checklist for a live move test

- [ ] `/api/health` ok; `/api/rdk/status` ready; driver `Ready`.
- [ ] Read-only pre-flight: joints, run mode, all target poses reachable, IK/FK ~0 err.
- [ ] Set `run_robot` + `connect_robot`; **abort if not ready**.
- [ ] Restart the preview (`/live/stop`+`/live/start`); confirm `gate` events flow.
- [ ] Continuous `/ws` monitor; move via executor; read fresh (release→re-hold).
- [ ] Return to exact `J0` in `finally`; restore the original run mode.
- [ ] Confirm final joints == start joints; report what physically moved.

## 8. Related

- `docs/agent-debug-map.md` — overall navigation.
- `docs/jetson-scanner.md` — camera server / Wi-Fi reality.
- `tasni/modules/scan/service.py` — `stabilize_live_scan_payload` (the hold),
  `live_scan_telemetry_payload` (the gate the HUD reads).
- `tasni/core/rdk_io.py` — `solve_joints_for_pose`, `camera_pose_T`, `apply_run_mode`,
  `connect_robot`.
