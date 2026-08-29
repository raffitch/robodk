"""One-command morning readiness check for the scanning cell.

    py -3.10 tools/cell_health.py                 # full check (opens one camera connection)
    py -3.10 tools/cell_health.py --skip-camera   # no camera traffic (Jetson + host only)

Prints one line per check: OK / WARN / FAIL, and for anything not OK, the exact
remedy. Exit code 0 when nothing FAILed, 1 otherwise.

Written after the 2026-08-29/30 "sensor layer at full fidelity" programme moved the
D435i to protocol 2 (raw unaligned depth, 0.1 mm words, 1280x720 depth / 1920x1080
colour). Most of what can silently regress here is silent BY NATURE -- a stale
factory K still projects, a stale backend still serves, a 1 mm depth word still
looks like a number -- so every check below proves its value rather than assuming it.

Device options are read from the service's own journal READ-BACK lines rather than by
opening the device: the camera service holds the D435i exclusively, and stopping it to
answer a question the journal already answers would be the riskiest thing this script
does. Those lines log what the DEVICE reported, not what was requested.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# The Jetson journal carries localised month names; never let an encoding surprise
# take the report down on a cp1252 console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OK, WARN, FAIL = "OK", "WARN", "FAIL"
_COLOR = {OK: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m"}
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

START_CMD = r".\start.ps1"
RESULTS: list[tuple[str, str, str, str]] = []   # (status, title, detail, remedy)


def record(status, title, detail, remedy=""):
    RESULTS.append((status, title, detail, remedy))
    tag = f"{_COLOR[status]}{status:4s}\033[0m" if _USE_COLOR else f"{status:4s}"
    print(f"  [{tag}] {title}: {detail}")
    if status != OK and remedy:
        print(f"         -> {remedy}")


def section(name):
    print(f"\n{name}")


def _ascii(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


# --------------------------------------------------------------------------- host

def check_repo():
    section("Repository")

    def git(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True,
                              text=True, timeout=90).stdout.strip()
    try:
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        head = git("rev-parse", "--short", "HEAD")
        subprocess.run(["git", "-C", str(ROOT), "fetch", "origin", "--quiet"],
                       capture_output=True, timeout=180)
        local, remote = git("rev-parse", "HEAD"), git("rev-parse", f"origin/{branch}")
        if local and remote and local == remote:
            record(OK, "branch vs origin", f"{branch} {head} == origin/{branch}")
        else:
            record(WARN, "branch vs origin", f"{branch} {head} differs from origin/{branch}",
                   f"git pull origin {branch}   (or push, if the local side is ahead)")
        dirty = git("status", "--porcelain")
        if dirty:
            record(WARN, "working tree", f"{len(dirty.splitlines())} uncommitted change(s)",
                   "git status -- commit or stash so the run is reproducible")
        else:
            record(OK, "working tree", "clean")
        return head
    except Exception as e:                                   # noqa: BLE001
        record(FAIL, "repository", f"could not read git state: {e}",
               "run this from the repo root with git on PATH")
        return None


def check_config():
    section("Host config (tasni.config.json)")
    # Protocol 2 streams colour at 1920x1080. The host picks K by this string, and
    # nothing validates it against the greeting at config-load time -- a stale 720p
    # value projects every point into the top-left ~44% of the frame with no error
    # and no telemetry. check_camera_protocol() cross-checks it against the stream.
    try:
        from tasni.core.config import load_config
        cfg = load_config()
    except Exception as e:                                   # noqa: BLE001
        record(FAIL, "config loads", f"{e}", "fix tasni.config.json, then re-run")
        return
    res = cfg.camera.resolution
    if res == "1920x1080":
        record(OK, "camera.resolution", res)
    else:
        record(FAIL, "camera.resolution", f"{res} (the server streams 1920x1080)",
               'remove the "resolution" key from tasni.config.json -- the default is now '
               "1920x1080, and a 720p K misprojects the chroma gate SILENTLY")

    fx = float(cfg.camera.K[0, 0])
    expected = 1334.8113          # calibrated 720p fx 889.8742 x 1.5
    if abs(fx - 1362.15) < 0.5:
        record(FAIL, "1080p intrinsics", f"fx={fx:.4f} is the FACTORY value",
               "the x1.5 migration did not fire. Confirm tasni.config.json still holds the "
               "calibrated 1280x720 K (fx 889.8742); load_config() migrates on next start and "
               'logs "config: migrated calibrated 1280x720 intrinsics to 1920x1080 (x1.5)"')
    elif abs(fx - expected) < 0.5:
        record(OK, "1080p intrinsics", f"fx={fx:.4f} (calibrated 720p x 1.5)")
    else:
        record(WARN, "1080p intrinsics", f"fx={fx:.4f}, expected ~{expected}",
               "neither the factory K nor the expected migration -- confirm this was a "
               "deliberate re-calibration")

    raw = {}
    p = ROOT / "tasni.config.json"
    if p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                    # noqa: BLE001
            raw = {}
    if "depth_scale" in (raw.get("scan") or {}):
        record(WARN, "legacy scan.depth_scale", "still present in the file",
               "harmless (load_config drops it on read), but re-save the config to clean it up")
    else:
        record(OK, "legacy scan.depth_scale", "absent (removed with protocol 2)")

    dist = cfg.camera.dist.ravel()
    if float(abs(dist).sum()) == 0.0:
        record(WARN, "colour distortion", "all zeros",
               "the D435i ships zero RGB distortion, so a real calibration should have "
               "replaced this. Re-run calibration if bead/board classification looks off")
    else:
        record(OK, "colour distortion", f"k1={dist[0]:.4f} k2={dist[1]:.4f} (calibrated)")


def _newest_source():
    newest, newest_t = None, 0.0
    for pat in ("tasni/**/*.py", "tasni/webui/src/**/*.ts", "tasni/webui/src/**/*.tsx"):
        for f in ROOT.glob(pat):
            if "node_modules" in f.parts:
                continue
            t = f.stat().st_mtime
            if t > newest_t:
                newest, newest_t = f, t
    return newest, newest_t


def check_backend():
    section("Tasni backend")
    newest, newest_t = _newest_source()
    try:
        # Match on the process NAME as well as the command line: this query's own
        # powershell.exe carries the pattern in its arguments and would otherwise
        # match itself -- reporting a fresh "backend" whenever none is running,
        # which is precisely the failure this check exists to catch.
        ps = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python|py|"
             "pythonw)' -and $_.CommandLine -match 'uvicorn|tasni.webapp' } | "
             "Select-Object ProcessId,CreationDate | ConvertTo-Json"],
            capture_output=True, text=True, timeout=120)
        out = (ps.stdout or "").strip()
        procs = json.loads(out) if out else []
        if isinstance(procs, dict):
            procs = [procs]
    except Exception as e:                                   # noqa: BLE001
        record(WARN, "backend process", f"could not enumerate processes: {e}",
               "check by hand that the backend was restarted after the last code change")
        return
    if not procs:
        record(WARN, "backend process", "not running",
               f"start it with  {START_CMD}   -- it must be started AFTER the latest code "
               "change, because this backend caches imported modules")
        return
    # CreationDate arrives as /Date(ms)/ or an ISO string depending on the PS version.
    start = 0.0
    for p in procs:
        m = re.search(r"(\d{10,})", str(p.get("CreationDate")))
        if m:
            start = max(start, int(m.group(1)) / 1000.0)
    if start == 0.0:
        record(WARN, "backend process", f"running ({len(procs)}), start time unreadable",
               f"if you changed code since starting it, restart with  {START_CMD}")
        return
    started = time.strftime("%H:%M:%S", time.localtime(start))
    if newest is not None and start < newest_t:
        age = (newest_t - start) / 60.0
        record(FAIL, "backend freshness",
               f"started {started}, but {_ascii(str(newest.relative_to(ROOT)))} changed "
               f"{age:.0f} min later",
               f"RESTART IT:  {START_CMD}   -- you are running STALE code")
    else:
        record(OK, "backend freshness", f"started {started}, newer than every source file")


# ------------------------------------------------------------------------- jetson

# Each entry proves one device option from the service's own read-back line, i.e.
# the value the DEVICE reported after the set, not the value that was requested.
JOURNAL_CHECKS = [
    ("depth units (0.1 mm)",
     r"depth_units -> requested 0\.0001, set 0\.0001, device reports 0\.0001",
     "the service is not applying 0.1 mm depth words: py -3.10 tools/jetson_deploy.py deploy"),
    ("depth_unit_mm read-back", r"depth_unit_mm = 0\.0999",
     "the device did not accept 0.1 mm units. Restart the service and re-read the journal"),
    ("laser power 150", r"laser_power -> requested 150, set 150, device reports 150",
     "laser power is not the characterised 150. The unit pins it via "
     "Environment=RS_LASER_POWER=150 -- check /etc/systemd/system/realsense-camera.service"),
    ("emitter on", r"emitter_enabled -> requested 1, set 1, device reports 1",
     "the IR projector is off; depth on blank surfaces will be poor"),
    ("visual preset 0 (Custom)", r"visual_preset left as-is at 0\.0",
     "the device is not on the Custom (0) preset the 2026-08-13 characterisation was "
     "measured under. Check for an RS_VISUAL_PRESET override in the unit"),
    ("stream resolutions", r"depth \(1280, 720\) colour \(1920, 1080\)",
     "the streams are not at the sensor maximums protocol 2 requires"),
]


def check_jetson(host_head):
    section("Jetson camera service")
    try:
        import jetson_deploy as jd
        j = jd.Jetson(jd.load_env(jd.SECRETS))
    except SystemExit as e:
        record(FAIL, "ssh", f"{e}",
               "check the Jetson is powered and on the network: "
               "ssh -i ~/.ssh/jetson_robodk jetson@10.12.171.70")
        return
    except Exception as e:                                   # noqa: BLE001
        record(FAIL, "ssh", f"{e}", "check secrets/jetson.env and network reachability")
        return

    _, out, _ = j.run("systemctl is-active realsense-camera", quiet=True)
    state = out.strip()
    if state == "active":
        record(OK, "service", "active")
    else:
        record(FAIL, "service", state or "unknown",
               "py -3.10 tools/jetson_deploy.py deploy   (pulls and restarts)")

    _, out, _ = j.run("ss -tln | grep ':1024' || true", quiet=True)
    if "LISTEN" in out:
        record(OK, "port 1024", "LISTENING")
    else:
        record(FAIL, "port 1024", "not listening",
               "the service is up but not accepting clients -- restart it and read the journal")

    _, out, _ = j.run("ss -tn state established sport = :1024 | tail -n +2 || true", quiet=True)
    n_conn = len([ln for ln in out.splitlines() if ln.strip()])
    if n_conn == 0:
        record(OK, "camera clients", "none connected (camera free)")
    else:
        record(WARN, "camera clients", f"{n_conn} connection(s) already open",
               "another client holds the camera; close it before starting a capture run")

    _, out, _ = j.run("cd ~/robodk && git rev-parse --short HEAD && "
                      "git rev-parse --abbrev-ref HEAD", quiet=True)
    parts = out.split()
    if len(parts) >= 2:
        jhead, jbranch = parts[0], parts[1]
        if host_head and jhead == host_head:
            record(OK, "jetson code", f"{jbranch} {jhead} (same commit as this checkout)")
        else:
            record(WARN, "jetson code", f"{jbranch} {jhead} vs host {host_head}",
                   "the auto-pull timer runs every ~2 min; for an immediate sync: "
                   "py -3.10 tools/jetson_deploy.py deploy")

    _, out, _ = j.run("journalctl -u realsense-camera --since '-24h' --no-pager | tail -400",
                      quiet=True)
    log = out
    if not log.strip():
        record(WARN, "journal", "no entries in the last 24 h",
               "restart the service so it re-logs its startup read-back: "
               "py -3.10 tools/jetson_deploy.py deploy")
        return
    for title, pattern, remedy in JOURNAL_CHECKS:
        if re.search(pattern, log):
            record(OK, title, "confirmed by the service's journal read-back")
        else:
            record(FAIL, title, "not found in the last 400 journal lines", remedy)

    if re.search(r"extrinsic layout check failed", log):
        record(FAIL, "depth->colour extrinsic", "layout check FAILED",
               "the greeting's extrinsic would be wrong for every client -- DO NOT CAPTURE. "
               "See server/rs_geometry.py:extrinsic_row_major")
    else:
        record(OK, "depth->colour extrinsic", "no layout-check failure logged")

    stalls = len(re.findall(r"Camera stalled; rebuilding", log))
    if stalls == 0:
        record(OK, "pipeline stalls", "none in the last 400 journal lines")
    else:
        record(WARN, "pipeline stalls", f"{stalls} rebuild(s) logged",
               "the recovery supervisor handled it, but repeated stalls usually mean the USB "
               "link is degrading -- if captures fail, physically replug the camera "
               "(docs/jetson-scanner.md)")


# ------------------------------------------------------------------------- camera

def check_camera_protocol():
    section("Camera protocol (opens one connection)")
    import numpy as np
    from tasni.core.config import load_config
    cam = load_config().camera

    # A client that did not restart after the protocol change must be REFUSED at the
    # handshake, not left to misread the JSON greeting as a frame length and hang.
    try:
        s = socket.create_connection((cam.ip, cam.port), timeout=10)
        try:
            s.sendall(b"MODE FULL\n")            # deliberately stale: no V2 token
            s.settimeout(10)
            reply = s.recv(128)
        finally:
            s.close()
    except Exception as e:                                   # noqa: BLE001
        record(FAIL, "stale-client refusal", f"{e}",
               f"could not reach the camera at {cam.ip}:{cam.port}")
        return
    if reply.startswith(b"ERR protocol 2 required"):
        record(OK, "stale-client refusal", _ascii(reply.decode("utf-8", "replace").strip()))
    else:
        record(FAIL, "stale-client refusal", f"got {reply[:60]!r}",
               "an old client will HANG instead of failing loudly. Confirm the Jetson runs "
               "the current server: py -3.10 tools/jetson_deploy.py deploy")

    try:
        from tasni.core.camera import CameraClient
        frame = CameraClient(cam).grab(with_depth=True, timeout=25)
    except Exception as e:                                   # noqa: BLE001
        record(FAIL, "depth grab", f"{e}",
               "the camera is listening but did not deliver a protocol-2 frame. Read the "
               "journal: py -3.10 tools/jetson_deploy.py status")
        return

    g, d = frame.geometry, frame.depth
    if abs(float(g.depth_unit_mm) - 0.1) < 0.005:
        record(OK, "depth unit", f"{g.depth_unit_mm:.4f} mm/word")
    else:
        record(FAIL, "depth unit", f"{g.depth_unit_mm} mm/word, expected 0.1",
               "the greeting is not reporting 0.1 mm words -- protocol 2 is not fully live")

    # The one cross-check nothing else performs: the K the host projects with is
    # chosen by camera.resolution, and a mismatch is silent in every downstream number.
    if tuple(g.color_size) == tuple(cam.size):
        record(OK, "greeting vs config", f"colour {g.color_size} matches camera.resolution")
    else:
        record(FAIL, "greeting vs config",
               f"stream is {g.color_size} but config says {cam.size}",
               "this misprojects every point into the wrong part of the frame SILENTLY. "
               "Set camera.resolution in tasni.config.json to match the stream")

    cy, cx = d.shape[0] // 2, d.shape[1] // 2
    patch = d[cy - 60:cy + 60, cx - 60:cx + 60]
    vals = np.unique(patch[patch > 0])
    if vals.size < 2:
        record(FAIL, "depth quantisation", "no depth in the centre patch",
               "point the camera at a surface 0.3-1.5 m away and re-run")
        return
    step = float(np.diff(vals).min()) * float(g.depth_unit_mm)
    valid = float((d > 0).sum()) / d.size
    if step <= 0.15:
        record(OK, "depth quantisation",
               f"min step {step:.2f} mm, {vals.size} distinct values, {valid:.1%} valid")
    else:
        record(FAIL, "depth quantisation", f"min step {step:.2f} mm (expected 0.10)",
               "depth is still quantised at the old 1 mm word -- the Jetson is serving a "
               "pre-protocol-2 stream: py -3.10 tools/jetson_deploy.py deploy")


def main() -> int:
    ap = argparse.ArgumentParser(description="Morning readiness check for the scanning cell.")
    ap.add_argument("--skip-camera", action="store_true",
                    help="do not open a camera connection (Jetson + host checks only)")
    args = ap.parse_args()

    print("=" * 74)
    print(" Tasni cell health check   " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 74)

    head = check_repo()
    check_config()
    check_backend()
    check_jetson(head)
    if args.skip_camera:
        section("Camera protocol")
        print("  (skipped: --skip-camera)")
    else:
        try:
            check_camera_protocol()
        except Exception as e:                               # noqa: BLE001
            record(FAIL, "camera checks", f"unexpected error: {e}",
                   "re-run with --skip-camera to get the rest of the report")

    fails = [r for r in RESULTS if r[0] == FAIL]
    warns = [r for r in RESULTS if r[0] == WARN]
    print("\n" + "=" * 74)
    if not fails and not warns:
        print(" READY -- every check green.")
    elif not fails:
        print(f" READY WITH NOTES -- {len(warns)} warning(s); nothing blocks a capture run.")
    else:
        print(f" NOT READY -- {len(fails)} failure(s), {len(warns)} warning(s). "
              "Fix the FAIL lines above:")
        for _, title, _, remedy in fails:
            print(f"   * {title}: {remedy or 'see above'}")
    print("=" * 74)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
