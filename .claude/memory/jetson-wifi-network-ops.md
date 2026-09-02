---
name: jetson-wifi-network-ops
description: Jetson Wi-Fi facts — Intel 8265 (no dongle), In5 network details, polkit blocks nmcli over SSH, and the NM back-to-back switch race
metadata:
  type: project
---

Operational facts for changing the Jetson's network remotely (learned 2026-08-28).

**Hardware:** the Jetson has an **Intel Wireless 8265/8275 M.2 card** (`iwlwifi`, PCI 01:00.0),
dual-band 802.11ac — NOT a USB dongle. Any USB Wi-Fi dongle (e.g. TL-WN821N, 2.4 GHz-only
802.11n) would be a downgrade. `lsusb` shows only the RealSense (8086:0b3a) and hubs.

**In5 Innovation Center** (the cell's corporate Wi-Fi) is **WPA/WPA2-PSK, not 802.1X**, with
dozens of APs on 2.4 + 5 GHz. It hands the Jetson `10.12.171.70/16`, gw `10.12.0.1`.
No captive portal, no client isolation — the Windows host (`10.12.172.19`, same /16) can
ping, SSH, and reach camera tcp/1024 directly. In5 is now `autoconnect-priority 10`
(both iPhone hotspot profiles are 0), so it wins on boot.

**`nmcli` over SSH fails with "Not authorized to control networking"** — polkit denies
network control to a non-console session. You MUST run it as root:
`echo "$PW" | sudo -S -p "" nmcli ...`, with the password piped over **stdin** (never in the
command line). Password is `JETSON_SUDO_PASSWORD` in git-ignored `secrets/jetson.env`.

**Don't switch Wi-Fi networks back-to-back.** Issuing `nmcli con up <other-wifi>` while still
associated to another network on the same radio races: the old network's DEAUTH_LEAVING lands
after the new activation starts and NM misreports it as
`Connection activation failed: 802.1X supplicant failed` — with **no `SME: Trying to
authenticate` line for the target at all**, which is how you tell it apart from a real auth
failure. Retry once and it connects. Always run remote network switches as a **detached,
self-reverting script** (`nohup setsid`) with a watchdog that restores a known-good profile
if there's no internet after N seconds — otherwise a failed switch strands the box.

Two latent issues seen while debugging: `/var/log/syslog` grows ~100 MB/day (342 MB in
/var/log; disk 64% full), and syslog timestamps are **non-monotonic across boots** because the
Nano has no RTC battery — relevant to [[scan-module-status]]'s Jetson-vs-host clock-skew hazard.
