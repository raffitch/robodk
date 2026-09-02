---
name: jetson-usb-camera-failure
description: "Diagnosing \"camera connected but no image\" — the D435i USB failure chain (EPROTO -71 → tegra-xusb HC death), and that dmesg floods away the evidence."
metadata: 
  node_type: memory
  type: project
  originSessionId: b2138c9d-8ae1-41fe-a2bd-3209d64d7d12
  modified: 2026-08-29T11:13:20.328Z
---

2026-08-29: "camera connection problem" = the app connects fine and shows nothing.
The server was `active` and LISTENING on 1024 the whole time; only the *frames*
were gone. Port-open is not camera-alive — never diagnose this from `is-active`.

**The chain, in order.** Camera streamed 2 h healthy, then at 14:44:04 stopped
delivering: `uvcvideo: Failed to query (SET_CUR) UVC control 1 on unit 3: -71`
(EPROTO — device stopped answering control transfers) → every acquisition threw
`Frame didn't arrive within 5000` → restarting the service tore down the wedged
device and killed the whole USB controller: `tegra-xusb: HC died; cleaning up`,
`hcd_reinit is disabled`, every device on both buses disconnected, process out on
SIGSEGV. After a reboot the controller came back but the camera would not
enumerate: `device not accepting address N, error -71` ×4 including a hub power
cycle → `unable to enumerate USB device`. Fixed only by the operator physically
replugging (it came back on a different hub port, Dev 007, 5000M, all 6
interfaces bound).

**Order of checks** (each one distinguishes a layer):
`journalctl -u realsense-camera` → `lsusb | grep 8086` (device there at all?) →
`lsusb -t` (SuperSpeed 5000M? which hub port?) → `ls /dev/video*` →
`dmesg -T | grep -iE "usb|xusb|uvc"`.

**Two traps.**
1. The EPROTO spam **floods the dmesg ring buffer** and destroys the
   moment-of-failure history — by the time I looked, the entire buffer was
   nothing but that one repeated line. Grab dmesg EARLY. The journal is also
   volatile here (`--list-boots` shows only the current boot), so cross-boot
   history is gone too.
2. A reboot revives a dead `tegra-xusb`, but **cannot** revive a camera that
   won't enumerate — that needs hands on the cable. Wi-Fi is PCIe (Intel 8265),
   so SSH survives a dead USB controller and a remote reboot is safe.

**Standing hardware suspicion:** the D435i hangs off a cheap generic Realtek hub
(`0bda:0411`), not the Jetson directly, and its cable is flexed by the KUKA on
every move. Two wedges in two days (2026-08-28, 2026-08-29). If it recurs, suspect
cable/connector and the hub before any software.

Server-side half is fixed in `267cf71` — see [[camera-server-recovery-supervisor]].
Related: [[jetson-laser-power-vs-characterization]], [[scan-module-status]].
