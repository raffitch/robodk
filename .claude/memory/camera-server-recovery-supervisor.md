---
name: camera-server-recovery-supervisor
description: "The Jetson camera server now supervises its own pipeline (read_frames) — what it guarantees, and the two bounds that exist to avoid hammering USB."
metadata: 
  node_type: memory
  type: project
  originSessionId: b2138c9d-8ae1-41fe-a2bd-3209d64d7d12
  modified: 2026-08-29T11:13:32.837Z
---

`267cf71` (main, deployed + live-verified 2026-08-29). Before it, `pipeline`/
`align` were built once under `if __name__ == '__main__'` and **nothing ever
rebuilt them**, so any stall was permanent until a human restarted the service.

All three acquisition loops — `getFrames`, the colour-only fast path, and the
H.264 `feeder` — now go through `read_frames()`, so one stall detector sees the
timeouts from every client thread. It rides out a brief stall
(`FRAME_WEDGE_THRESHOLD = 3`), rebuilds the pipeline once for a persistent one,
and calls `_give_up()` (which `os._exit(1)`s, letting `Restart=always` re-open
from scratch) rather than serving clients it cannot feed.

**Why:** the failure being prevented is not a dead camera — software can't fix
that — it's the server *pretending to serve*. See [[jetson-usb-camera-failure]].

**Two bounds, both there because hammering USB is what precedes a dead host
controller** (`tegra-xusb: HC died`, same cell, same day):
- reopen attempts back off `RECOVERY_BACKOFF_S = (1.0, 5.0, 15.0)`;
- a rebuild is credited **only once frames actually flow again**
  (`HEALTHY_FRAMES_AFTER_REBUILD = 30`), so a device that re-opens cleanly but
  never streams isn't rebuilt every ~20 s forever. I hit that infinite loop for
  real — the test for it *hung* before the bound existed.

`getFrames` lost its unused `pipeline` parameter — signature is now
`getFrames(align, depth_filters)`. Only `server_unicast_syncronous.py` matters;
the `asyncio`/`dynamicRes` variants are unused and untouched.

Tests: `tests/test_camera_recovery.py` (15). Stub `pyrealsense2`/`turbojpeg` in
`sys.modules` and import the server on the host — the pattern from
`test_server_client_lifecycle.py`.
