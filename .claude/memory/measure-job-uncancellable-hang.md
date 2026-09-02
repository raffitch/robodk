---
name: measure-job-uncancellable-hang
description: "A measure run on a taught inspection target wedged the backend's shared RoboDK connection; /cancel could not stop it because _wait_program has no timeout and only checks cancel between polls."
metadata: 
  node_type: memory
  type: project
  originSessionId: 29056919-c652-4e99-b9ca-e8ca5b163359
  modified: 2026-09-01T10:41:21.367Z
---

2026-09-01: the first ring-2 capture from the taught target `ScanAtRing2`
(manual inspection mode, `inspection_auto=false`) hung in the robot move and
could not be cancelled. Symptoms, in the order they distinguish causes:

- `/api/health` keeps answering in ~200 ms and reports `robodk: ok` — it does
  NOT prove RoboDK is usable.
- `/api/modules/extrusion/station-options` **blocks** (20 s timeout). That is
  the real test: it calls the shared `services.rdk`, so a wedged job thread
  takes the whole backend's RoboDK path with it.
- `POST /measure/cancel` returns `{"status":"cancelling"}` and nothing happens.
- The Jetson has **no ESTABLISHED connection on 1024** (`ss -tn state
  established '( sport = :1024 )'`) even though health still says
  `camera: in_use` — the lease flag outlives the socket, so "in_use" is not
  evidence of a live capture.
- RoboDK's process CPU barely advances (~1.8 s over 10 min) and
  `Responding` stays True: it is waiting, not computing.

**Why cancel cannot help:** `_wait_program` (`modules/extrusion/service.py:295`)
loops `while _busy()` with **no timeout**, and checks `ctx.cancelled` only
*between* polls. If a single robolink call never returns, the cancel check is
never reached. `stop_program` is only issued from inside that same loop.

**How to read it next time:** camera socket closed + station-options blocked
= stuck on the RoboDK/robot side, not the camera. Do not kill RoboDK or the
backend on your own: a program already dispatched with `real_robot=True` runs
on the KUKA controller, so killing the host process does not stop the arm, and
the 117 MB station is open in the GUI. Ask the operator to check the robot's
mode/state at the cell first.

Related: [[restart-tasni-backend-after-code-edits]],
[[pfh-paper-ring-stack-experiment]].
