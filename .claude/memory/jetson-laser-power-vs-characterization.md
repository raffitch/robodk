---
name: jetson-laser-power-vs-characterization
description: "RESOLVED 2026-08-25: laser_power pinned to 150 via Environment= in realsense-camera.service, verified by device read-back; RealSense option state lives on the DEVICE, so 'leave alone' preserves whatever was last written."
metadata: 
  node_type: memory
  type: project
  originSessionId: 3372a85a-dae9-45ab-8cfc-d5be9e713794
  modified: 2026-08-25T10:42:01.213Z
---

**The durable lesson: RealSense option state lives on the DEVICE and survives a
service restart.** So "leave the option alone" does not mean "the option is at its
factory default" — it means "whatever the last process wrote is still there."

That is exactly how this bit. An earlier build defaulted `RS_LASER_POWER=300`, which
was written into the sensor. Changing the code default to `-1` (leave-alone) stopped
*future* silent changes but did not undo the past one: the camera kept reporting
`laser_power left as-is at 300` while the 2026-08-13 depth envelope
([[cell-characterization-2026-08-13]]) had been measured at **150**.

**Resolved 2026-08-25** (`b697f3b`) by declaring it explicitly rather than implicitly:
`Environment=RS_LASER_POWER=150` in `server/realsense-camera.service` (version
controlled; applied with `python tools/jetson_deploy.py bootstrap`). Verified by
read-back on the cell — `laser_power -> requested 150, set 150, device reports 150
(range 0..360)`. `RS_VISUAL_PRESET` is still leave-alone and the device still sits
at 0 (Custom).

**Trade-off deliberately accepted:** 300 was originally chosen to put MORE IR texture
on blank surfaces, the most direct lever on the ~20 mm of missing depth at panel
edges. 150 may cost some of that coverage. It is still the right default — an accuracy
envelope is only meaningful if the device matches it — and raising the power is now an
explicit experiment that must carry its own before/after measurement
(`tools/characterize_distance.py`).

**Always read the option back; never trust the write.** The visual preset logged
"High Accuracy" for over a month while the device actually sat at 0.
