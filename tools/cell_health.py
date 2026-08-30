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
# (status, title, detail, remedy, blocks_capture)
RESULTS: list[tuple[str, str, str, str, bool]] = []


def record(status, title, detail, remedy="", blocks_capture=False):
    """Record and print one check line.

    ``blocks_capture`` marks a WARN the operator must still ACT on before a capture
    run is possible at all -- the backend is not started, another client holds the
    unicast camera. Those are deliberately not FAILs: at 8 am "the backend is not
    running yet" is the normal state of the world, not a broken cell. But the
    summary must not then tell the operator that nothing blocks a capture run,
    which is what it used to do.
    """
    RESULTS.append((status, title, detail, remedy, blocks_capture))
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


# What the RUNNING backend has cached, and therefore what makes it stale:
#   * everything it imports (``tasni/**``) and the webui sources built into dist;
#   * ``tasni.config.json`` -- ``create_app()`` calls ``load_config()`` ONCE at
#     startup and hands that AppConfig to ServiceContainer, so an edited config
#     (a new K, a new camera.resolution) is not picked up until a restart, exactly
#     like an edited module. It belongs here for the same reason the .py files do.
# Deliberately NOT included, so the check keeps meaning what it says:
#   * ``server/**`` runs on the JETSON, not in this backend -- nothing under
#     ``tasni/`` imports it. Staleness there is what the "jetson code" commit
#     check covers; counting it here would demand a pointless host restart after
#     every server edit and train the operator to ignore this line.
#   * ``tools/**`` is not imported by the backend either -- and THIS script lives
#     there, so counting it would make the check fire on its own edits.
BACKEND_SOURCE_GLOBS = ("tasni/**/*.py", "tasni/webui/src/**/*.ts",
                        "tasni/webui/src/**/*.tsx", "tasni.config.json")


def _source_files():
    """Every file whose mtime can make the running backend stale."""
    for pat in BACKEND_SOURCE_GLOBS:
        for f in ROOT.glob(pat):
            if "node_modules" in f.parts or not f.is_file():
                continue
            yield f


def _newest_source():
    newest, newest_t = None, 0.0
    for f in _source_files():
        t = f.stat().st_mtime
        if t > newest_t:
            newest, newest_t = f, t
    return newest, newest_t


def _oldest_start(procs):
    """``(oldest start epoch, how many start times were unreadable)``.

    MIN, not max. The failure this check exists to catch is a STALE backend still
    holding :8000 while some newer python process (a second launch that lost the
    port, a tool, a notebook) also matches the pattern. Reducing with max() reports
    the newest of them and calls the cell fresh -- it green-lights precisely the
    situation it was written to catch. The OLDEST matching process is the one whose
    imports may predate the last edit, so it is the one to report.
    """
    starts, unreadable = [], 0
    for p in procs:
        # CreationDate arrives as /Date(ms)/ or an ISO string depending on the PS version.
        m = re.search(r"(\d{10,})", str(p.get("CreationDate")))
        if m:
            starts.append(int(m.group(1)) / 1000.0)
        else:
            unreadable += 1
    return (min(starts) if starts else 0.0), unreadable


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
               "check by hand that the backend is running and was started AFTER the last "
               "code change", blocks_capture=True)
        return
    if not procs:
        record(WARN, "backend process", "not running",
               f"start it with  {START_CMD}   -- it must be started AFTER the latest code "
               "change, because this backend caches imported modules", blocks_capture=True)
        return
    start, unreadable = _oldest_start(procs)
    n = len(procs)
    if start == 0.0:
        record(WARN, "backend process", f"running ({n}), start time unreadable",
               f"if you changed code since starting it, restart with  {START_CMD}")
        return
    started = time.strftime("%H:%M:%S", time.localtime(start))
    which = f"oldest of {n} matching process(es) started {started}" if n > 1 else \
            f"started {started}"
    if newest is not None and start < newest_t:
        age = (newest_t - start) / 60.0
        record(FAIL, "backend freshness",
               f"{which}, but {_ascii(str(newest.relative_to(ROOT)))} changed "
               f"{age:.0f} min later",
               f"RESTART IT:  {START_CMD}   -- you are running STALE code")
    elif unreadable:
        record(WARN, "backend freshness",
               f"{which} (newer than every source file), but {unreadable} of {n} "
               "process(es) had an unreadable start time",
               f"one of them may predate the last edit -- if in doubt, restart: {START_CMD}")
    else:
        record(OK, "backend freshness", f"{which}, newer than every source file")


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
    # Matched against the server's startup banner, which prints the two tuples
    # COMMA-SEPARATED ("... depth (1280, 720), colour (1920, 1080), protocol 2").
    # The comma is load-bearing: without it this pattern never matched anything the
    # server has ever printed, so the check reported FAIL on a perfectly healthy
    # cell and sent the operator chasing a resolution problem that did not exist.
    # tests/test_cell_health.py now matches every pattern here against log text
    # generated by the production code itself, so it cannot drift again.
    ("stream resolutions", r"depth \(1280, 720\), colour \(1920, 1080\)",
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
        # The server is UNICAST: while someone else holds it, a capture run cannot
        # start at all, so this is a to-do for the operator, not a note.
        record(WARN, "camera clients", f"{n_conn} connection(s) already open",
               "another client holds the camera; close it before starting a capture run",
               blocks_capture=True)

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

# The server clips EVERY depth frame it ships through
# ``rs.threshold_filter(RS_DEPTH_MIN_M, RS_DEPTH_MAX_M)`` -- it is the FIRST link of
# setup_depth_filters()'s chain in server/server_unicast_syncronous.py, and the
# greeting advertises "threshold" at the head of that chain. So every non-zero word
# on the wire is, in TRUE metric, between 150 mm and 1500 mm, whatever the wire word
# size happens to be. THAT is what makes a wire/greeting scale mismatch detectable:
# multiply the words by the greeting's mm/word and the population must land back
# inside the clip. A pre-protocol-2 1 mm stream read with a 0.1 mm greeting lands at
# a tenth of it -- 15..150 mm, i.e. nearer than the camera is allowed to report.
DEPTH_CLIP_MIN_MM, DEPTH_CLIP_MAX_MM = 150.0, 1500.0
DEPTH_BAND_TOL = 0.10            # slack for the smoothing that runs AFTER the clip
DEPTH_MIN_VALID_PX = 1000        # below this the scene carries no scale evidence
DEPTH_MIN_VALID_FRAC = 0.02
DEPTH_SCALE_FAIL_FRAC = 0.50     # a MAJORITY outside the clip is a scale error
DEPTH_SCALE_WARN_FRAC = 0.05     # a minority is something else, and worth saying so

_AIM_REMEDY = ("point the camera at a surface 0.3-1.5 m away and re-run -- until it sees "
               "one, the wire word size is UNVERIFIED (not OK)")


def evaluate_depth_scale(depth, depth_unit_mm):
    """``(status, detail, remedy)`` for "do the wire words and the greeting agree?".

    This replaced a min-step "quantisation" check that could not fail. It computed
    ``min(diff(unique(words))) * depth_unit_mm``; on any real noisy patch the
    smallest gap between distinct words is 1 word REGARDLESS of the word size, so
    the expression collapsed to ``depth_unit_mm`` -- re-testing a number an earlier
    check already asserted, and reporting a confident OK on the exact condition its
    remedy named (1 mm words behind a 0.1 mm greeting: the host then reads 45 mm for
    a 450 mm surface). See tests/test_cell_health.py.

    Uncertainty is reported as uncertainty: an empty scene, or one entirely beyond
    the server's 1.5 m clip, yields no valid words and therefore no evidence either
    way -- that is a WARN saying NOT CHECKED, never an OK.
    """
    import numpy as np

    d = np.asarray(depth)
    if d.ndim != 2 or d.size == 0:
        return (WARN, "NOT CHECKED - the frame carried no depth image", _AIM_REMEDY)
    try:
        unit = float(depth_unit_mm)
    except (TypeError, ValueError):
        unit = float("nan")
    if not (unit > 0.0):
        return (FAIL, f"greeting depth_unit_mm is {depth_unit_mm!r}",
                "the greeting carries no usable mm/word, so nothing downstream can scale "
                "depth: py -3.10 tools/jetson_deploy.py status")

    words = d[d > 0]
    frac = words.size / float(d.size)
    if words.size < DEPTH_MIN_VALID_PX or frac < DEPTH_MIN_VALID_FRAC:
        return (WARN,
                f"NOT CHECKED - only {words.size} valid depth pixel(s) ({frac:.2%} of the "
                f"frame); an empty scene, or one entirely beyond the server's "
                f"{DEPTH_CLIP_MAX_MM:.0f} mm clip, proves nothing about the word size",
                _AIM_REMEDY)

    mm = words.astype(np.float64) * unit
    lo = DEPTH_CLIP_MIN_MM * (1.0 - DEPTH_BAND_TOL)
    hi = DEPTH_CLIP_MAX_MM * (1.0 + DEPTH_BAND_TOL)
    med = float(np.median(mm))
    near, far = float((mm < lo).mean()), float((mm > hi).mean())
    seen = f"median {med:.0f} mm, {frac:.0%} of the frame valid"

    if near >= DEPTH_SCALE_FAIL_FRAC:
        return (FAIL,
                f"{seen} -- {near:.0%} of valid depth reads NEARER than {lo:.0f} mm, which "
                f"the server's {DEPTH_CLIP_MIN_MM:.0f} mm threshold filter makes impossible",
                f"the wire words are LARGER than the greeting's {unit:.4f} mm/word claims, so "
                "every distance is read short (a pre-protocol-2 1 mm stream reads exactly 10x "
                "short: 45 mm for a 450 mm surface). The Jetson is not serving protocol-2 "
                "depth: py -3.10 tools/jetson_deploy.py deploy")
    if far >= DEPTH_SCALE_FAIL_FRAC:
        return (FAIL,
                f"{seen} -- {far:.0%} of valid depth reads FARTHER than {hi:.0f} mm, which "
                f"the server's {DEPTH_CLIP_MAX_MM:.0f} mm threshold filter makes impossible",
                f"the greeting's {unit:.4f} mm/word is too LARGE for the words on the wire, so "
                "every distance is read long: py -3.10 tools/jetson_deploy.py status")
    if near + far >= DEPTH_SCALE_WARN_FRAC:
        return (WARN,
                f"{seen} -- {near + far:.1%} of valid depth outside the server's "
                f"{DEPTH_CLIP_MIN_MM:.0f}-{DEPTH_CLIP_MAX_MM:.0f} mm clip",
                "not the 10x signature of a wire/greeting scale mismatch, but the frame "
                "disagrees with the server's own threshold filter -- check the filter chain "
                "reported in the greeting before trusting depth")
    return (OK, f"{seen}, all inside the server's {DEPTH_CLIP_MIN_MM:.0f}-"
                f"{DEPTH_CLIP_MAX_MM:.0f} mm clip -- the wire words and the greeting's "
                f"{unit:.4f} mm/word agree", "")


# Every camera check that needs a live connection, in report order. Naming them
# here is what lets an earlier failure report the rest as NOT CHECKED instead of
# returning and leaving them silently missing from the report.
CAMERA_CHECK_TITLES = ("stale-client refusal", "depth grab", "depth unit",
                       "greeting vs config", "depth scale")


def _not_checked(titles, reason):
    """Record the checks we could not reach, rather than dropping them.

    A check that vanishes from the report reads as a check that passed. Every one of
    these lines is a thing we do NOT know about the cell, so it says so."""
    for title in titles:
        record(WARN, title, f"NOT CHECKED ({reason})",
               "re-run once the failure above is fixed -- this check has not run, so its "
               "condition is unknown, not good")


def _refusal_probe(connect, timeout, seen):
    """A ``probe(host, port)`` for :meth:`CameraClient.resolve_via` that doubles as
    the stale-client refusal check.

    The refusal check has to open a connection anyway, and the server is unicast, so
    making the refusal handshake BE the reachability probe costs no extra connection
    and resolves the route exactly the way a real capture would.
    """
    def probe(host, port):
        try:
            s = connect((host, port), timeout=timeout)
        except Exception as e:                               # noqa: BLE001
            seen["errors"].append(f"{host}:{port} ({e})")
            return False
        try:
            s.sendall(b"MODE FULL\n")           # deliberately stale: no V2 token
            s.settimeout(timeout)
            seen["reply"] = s.recv(128)
        except Exception as e:                               # noqa: BLE001
            seen["errors"].append(f"{host}:{port} handshake ({e})")
            seen["reply"] = b""
        finally:
            try:
                s.close()
            except Exception:                                # noqa: BLE001
                pass
        return True
    return probe


def check_camera_protocol(client=None, *, connect=socket.create_connection, grab=None):
    """The checks that need a live connection. ``connect``/``grab`` are seams for
    tests; the defaults are the real socket and a real one-shot depth grab."""
    section("Camera protocol (opens one connection)")
    from tasni.core.camera import CameraClient
    from tasni.core.config import load_config
    from tasni.core.health import connection_route

    if client is None:
        client = CameraClient(load_config().camera)
    cam = client.config
    todo = list(CAMERA_CHECK_TITLES)

    def done(title):
        if title in todo:
            todo.remove(title)

    # Resolve the host the way CameraClient does. This used to
    # ``socket.create_connection((cam.ip, cam.port))`` -- cam.ip is the TAILSCALE
    # address, while CameraClient tries cam.lan_ip FIRST, so with Tailscale down on
    # the host this check FAILED (and returned) against a cell that captures fine.
    # resolve_via() walks the same ladder and CACHES the winner on the client, so
    # the grab below reuses the resolved host instead of re-laddering.
    seen = {"errors": [], "reply": None}
    host, reachable = client.resolve_via(
        _refusal_probe(connect, cam.connect_probe_timeout_s, seen))
    if not reachable:
        # Mirror _connect(): the short budget is for failing over fast, so give the
        # ladder one more pass at the full timeout before calling the camera dead.
        host, reachable = client.resolve_via(_refusal_probe(connect, cam.timeout_s, seen))

    if not reachable:
        tried = "; ".join(dict.fromkeys(seen["errors"])) or "no candidate hosts configured"
        record(FAIL, "camera route", f"no route answered: {tried}",
               f"neither {cam.lan_ip or '(no lan_ip)'} (Direct/LAN) nor {cam.ip} (Tailscale) "
               f"accepted a connection on {cam.port}. Check the Jetson is powered and on the "
               "network: py -3.10 tools/jetson_deploy.py status")
        _not_checked(todo, "camera unreachable")
        return
    record(OK, "camera route", f"{host}:{cam.port} ({connection_route(host)})")

    # A client that did not restart after the protocol change must be REFUSED at the
    # handshake, not left to misread the JSON greeting as a frame length and hang.
    reply = seen["reply"] or b""
    done("stale-client refusal")
    if reply.startswith(b"ERR protocol 2 required"):
        record(OK, "stale-client refusal", _ascii(reply.decode("utf-8", "replace").strip()))
    else:
        record(FAIL, "stale-client refusal", f"got {reply[:60]!r}",
               "an old client will HANG instead of failing loudly. Confirm the Jetson runs "
               "the current server: py -3.10 tools/jetson_deploy.py deploy")

    done("depth grab")
    try:
        frame = (grab or _grab_one)(client)
    except Exception as e:                                   # noqa: BLE001
        record(FAIL, "depth grab", f"{e}",
               "the camera is listening but did not deliver a protocol-2 frame. Read the "
               "journal: py -3.10 tools/jetson_deploy.py status")
        _not_checked(todo, "no frame")
        return
    record(OK, "depth grab", f"one protocol-2 frame from {host}")

    g, d = frame.geometry, frame.depth
    if g is None:
        _not_checked(todo, "the frame carried no greeting")
        return

    done("depth unit")
    if abs(float(g.depth_unit_mm) - 0.1) < 0.005:
        record(OK, "depth unit", f"{g.depth_unit_mm:.4f} mm/word")
    else:
        record(FAIL, "depth unit", f"{g.depth_unit_mm} mm/word, expected 0.1",
               "the greeting is not reporting 0.1 mm words -- protocol 2 is not fully live")

    # The one cross-check nothing else performs: the K the host projects with is
    # chosen by camera.resolution, and a mismatch is silent in every downstream number.
    done("greeting vs config")
    if tuple(g.color_size) == tuple(cam.size):
        record(OK, "greeting vs config", f"colour {g.color_size} matches camera.resolution")
    else:
        record(FAIL, "greeting vs config",
               f"stream is {g.color_size} but config says {cam.size}",
               "this misprojects every point into the wrong part of the frame SILENTLY. "
               "Set camera.resolution in tasni.config.json to match the stream")

    done("depth scale")
    status, detail, remedy = evaluate_depth_scale(d, g.depth_unit_mm)
    record(status, "depth scale", detail, remedy)


def _grab_one(client):
    return client.grab(with_depth=True, timeout=25)


def summarise(results):
    """``(exit_code, lines)`` for the closing block.

    The old version bucketed only FAIL vs WARN, so a backend that is NOT RUNNING --
    recorded as a WARN, because at 8 am that is the normal state of the world --
    printed "READY WITH NOTES ... nothing blocks a capture run". You cannot capture
    without a backend, so that line was simply false.

    A warning the operator must ACT on before capturing is therefore separated from
    a warning that is only worth reading. It stays a WARN (exit 0, no scary red on a
    cell that is fine) but it gets its own headline and its own to-do list, which is
    what the operator actually needs at 8 am: is the cell good, and what is left for
    me to do?
    """
    fails = [r for r in results if r[0] == FAIL]
    warns = [r for r in results if r[0] == WARN]
    todo = [r for r in warns if len(r) > 4 and r[4]]

    def bullets(rows):
        return [f"   * {r[1]}: {r[3] or 'see above'}" for r in rows]

    out = ["=" * 74]
    if fails:
        out.append(f" NOT READY -- {len(fails)} failure(s), {len(warns)} warning(s). "
                   "Fix the FAIL lines above:")
        out += bullets(fails)
        if todo:
            out.append(" And still to do before you can capture:")
            out += bullets(todo)
    elif todo:
        n = len(todo)
        left = "1 thing still needs you" if n == 1 else f"{n} things still need you"
        out.append(f" CELL OK -- no failures, but {left} before you can capture:")
        out += bullets(todo)
        others = len(warns) - len(todo)
        if others:
            out.append(f"   ({others} other warning(s) above -- worth reading, "
                       "none of them a blocker.)")
    elif warns:
        out.append(f" READY WITH NOTES -- {len(warns)} warning(s) above; "
                   "none of them blocks a capture run.")
    else:
        out.append(" READY -- every check green.")
    out.append("=" * 74)
    return (1 if fails else 0), out


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
        # Recorded, not merely printed: otherwise a --skip-camera run with a green
        # host and Jetson ends on "READY -- every check green", which claims five
        # checks that never ran.
        section("Camera protocol")
        record(WARN, "camera protocol", f"NOT CHECKED (--skip-camera): {len(CAMERA_CHECK_TITLES)} "
               "check(s) not run", "re-run without --skip-camera before a capture run")
    else:
        try:
            check_camera_protocol()
        except Exception as e:                               # noqa: BLE001
            record(FAIL, "camera checks", f"unexpected error: {e}",
                   "re-run with --skip-camera to get the rest of the report")

    code, lines = summarise(RESULTS)
    print("")
    for line in lines:
        print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
