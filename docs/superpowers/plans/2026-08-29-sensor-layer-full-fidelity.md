# Sensor Layer at Full Fidelity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship raw, unaligned, 0.1 mm depth from the Jetson behind a versioned handshake with a JSON greeting; delete `align`/`hole_filling`/the unused IR stream; move colour to 1080p; give the host ONE back-projection module that every consumer uses; record the ASIC config and temperatures; rebuild librealsense as Release+CUDA first.

**Architecture:** The server (`server/`) gains two pure helper modules (`rs_config.py`, `rs_geometry.py`, plus a pure `handshake.py` parser) and stops aligning depth to colour. The per-frame wire header is unchanged; a one-line JSON greeting per connection carries depth intrinsics, the depth->colour extrinsic (row-major, asserted at start), `depth_unit_mm`, temps and device facts. The host gains `tasni/core/depth_geometry.py` (`CameraGeometry`, `backproject`, `ColorRegistered`, `depth_pose`) and every consumer of `Frame.depth` goes through it. Back-projected points are expressed in the **colour camera frame**, so the hand-eye (`RoboDKIO.camera_pose_T()`, a colour-camera pose) and every existing "camera frame" convention stay untouched; only the pixel->point step changes. Only Open3D TSDF and per-view support counting use the depth image directly and compose `depth_pose`.

**Tech Stack:** Python 3.10 (`py -3.10` on Windows — there is no `python` on PATH), numpy, OpenCV, scipy (cKDTree, already a dependency), Open3D (scan extra), pyrealsense2 2.55.1 on the Jetson (Ubuntu 18.04 / JetPack 4.6, CUDA 10.2), systemd, pytest (`py -3.10 -m pytest tests/<file> -k <name>` — NEVER the full suite, it is too slow).

**Spec:** `docs/superpowers/specs/2026-08-29-sensor-layer-full-fidelity-design.md`

## Global Constraints

- Per-frame wire header stays byte-identical: `<I depth_len><I color_len><d timestamp>` + lz4(`np.save`) depth + JPEG colour (spec 4.1).
- Depth clients MUST send `MODE FULL V2\n` (burst: `MODE BURST V2\n`); anything else asking for depth is refused with one line `ERR protocol 2 required; send MODE FULL V2\n` and a close (spec 4.1). `MODE COLOR ...` and `MODE TELEMETRY` are unchanged.
- The greeting is ONE newline-terminated JSON line, `protocol: 2`, sent before any frame (spec 4.1).
- `depth_units` = `0.0001` m; the greeting's `depth_unit_mm` is read back from `get_depth_scale()` AFTER the option is set, never assumed (spec 4.2).
- Colour stream 1920x1080 bgr8 @30; depth 1280x720 z16 @30; no infrared stream (spec 4.2).
- Filter chain exactly `threshold -> disparity -> spatial -> temporal -> disparity_inv`; NO `hole_filling` (spec 4.2).
- `CameraConfig.resolution` default `"1920x1080"`; `intrinsics["1920x1080"]` migrates to 1.5 x the calibrated 720p entry; `dist_coeffs` unchanged (spec 4.3).
- `ScanConfig.depth_scale` is DELETED. No code may default a depth unit to 1000 or 1 mm. The only fallback is the archive read-side `CameraGeometry.legacy_aligned(...)` for takes without `camera_geometry` (spec 4.4).
- `ExtrusionConfig.voxel_size_m` default 0.002 -> 0.001 (spec 4.4).
- The RoboDK-embedded macros (`macros/3DScan.py`, `AutoCalibrate.py`, `3DScanParam.py`, `ArucoToPlane.py`) and `server/robodk_*.py` speak the old wire and WILL be refused. They are superseded by the app (CLAUDE.md north star). Do not port them; say so in the docs (Task 13).
- **Branch: `sensor-layer-v2`, cut from `main` before Task 1.** The Jetson auto-pulls `main` every ~2 min and restarts the camera whenever `server/` changed, so a server commit on `main` deploys ITSELF. Every task commits and pushes to `sensor-layer-v2`; Task 13 merges to `main` and is the ONE deploy. Never push `server/` changes to `main` before Task 13.
- Windows traps: use `py -3.10`; never round-trip source files through PowerShell `Get-Content`/`Set-Content` (mojibake); after host code changes the Tasni backend must be restarted before any cell test (it caches imports).
- Order is fixed: Task 1 (done) -> 2 -> 3 -> (4..12 in order) -> 13. Task 2 MUST precede any change to `depth_units` (the depth table is part of the as-found JSON). Task 3 MUST be verified on the OLD protocol before Task 13 deploys the new one.

---

## File Structure

**Server (runs on the Jetson, Python 3.10 venv `~/EtherSenseServer/ethenv`):**
- Create `server/handshake.py` — pure parser `parse_handshake(req: bytes) -> dict`. Importable on the host (no pyrealsense2).
- Create `server/rs_config.py` — device option setup with read-back, `depth_units`, temps, `global_time_enabled`, advanced-mode as-found JSON dump. Takes the `rs` module as a parameter so it is testable with a fake.
- Create `server/rs_geometry.py` — intrinsics/extrinsics extraction, row-major transpose + assert, greeting builder. Same `rs`-as-parameter pattern.
- Create `server/presets/custom-as-found-2026-08-29.json` — the ASIC configuration the 2026-08-13 characterisation was measured under (captured in Task 2).
- Modify `server/server_unicast_syncronous.py` — remove `align`, V2 handshake + greeting + refusal, filter chain, 1080p colour, `DEPTH_SIZE`/`COLOR_SIZE`, feeder uses `rs_geometry`.
- Create `tests/test_handshake.py`, `tests/test_rs_config.py`, `tests/test_rs_geometry.py`.

**Host core:**
- Create `tasni/core/depth_geometry.py` — `CameraGeometry`, `backproject`, `depth_pose`, `project_to_color`, `ray_point`, `ColorRegistered`.
- Modify `tasni/core/camera.py` — `Frame.geometry`, V2 hello, greeting read, refusal error, burst greeting.
- Modify `tasni/core/config.py` — delete `ScanConfig.depth_scale`; `resolution` default; `migrate_camera_intrinsics`; legacy-key tolerance in `load_config`.
- Create `tests/test_depth_geometry.py`, `tests/geometry_fixtures.py` (shared synthetic geometries); modify `tests/test_camera_wire.py`, `tests/test_scan_config.py`.
- Create `tools/probe_depth_quantisation.py` — the audit §6 centre-patch probe as a tool (acceptance).

**Extrusion:**
- Modify `tasni/modules/extrusion/processing.py`, `measure.py`, `service.py`, `figures.py`; `tasni/core/config.py` (voxel).
- Modify `tests/test_extrusion_processing.py`, `tests/test_extrusion_measure.py`, `tests/test_extrusion_figures.py`, `tests/test_extrusion_job.py`, `tests/extrusion_synthetic.py`.

**Scan:**
- Modify `tasni/modules/scan/depth_gate.py`, `survey.py`, `corner_evidence.py`, `reconstruct.py`, `service.py`.
- Modify `tests/test_scan_depth_gate.py`, `tests/test_scan_survey.py`, `tests/test_corner_evidence.py`, `tests/test_scan_reconstruct.py`, `tests/test_scan_job.py`, `tests/test_five_position.py`.

**Tools/docs:**
- Modify `tools/characterize_distance.py`, `tasni/webui/src/pages/AimHud.tsx` (comment only), `docs/jetson-scanner.md`, `docs/realsense-capability-audit-2026-08-29.md`, `AGENTS.md`, `CLAUDE.md`, `docs/agent-debug-map.md`.

---

### Task 1: Commit the in-flight extrusion work — DONE (`aba0d9e` on `main`, 2026-08-29; 3 tests green). Skip to Task 2.

**Files:**
- Modify (commit as-is): `tasni/modules/extrusion/processing.py`, `tests/test_extrusion_measure.py`, `tests/fixtures/extrusion/ring1/README.md`
- Add: `tests/fixtures/extrusion/ring1/ring1_low_relief_20260829.npz`

The working tree holds an unfinished-looking but complete feature (ring-arc assembly + low-relief capture, 3 tests) on the same file Task 9 rewrites. It must land on its own commit first so the big-bang diff is reviewable.

- [ ] **Step 1: Confirm with the operator that this work is theirs and ready** (it was not written in this plan's session). If they say no, `git stash push -m "extrusion ring arcs (parked)"` and continue; Task 9 then applies on top of `main`.

- [ ] **Step 2: Run the three new tests**

Run: `py -3.10 -m pytest tests/test_extrusion_measure.py -k "arcs or low_relief or measured_only_in_part" -v`
Expected: 3 PASS

- [ ] **Step 3: Commit + push**

```bash
git add tasni/modules/extrusion/processing.py tests/test_extrusion_measure.py tests/fixtures/extrusion/ring1/README.md tests/fixtures/extrusion/ring1/ring1_low_relief_20260829.npz
git commit -m "feat(extrusion): assemble a ring from arcs the height floor split apart"
git push origin sensor-layer-v2
```

---

### Task 2: R4.1 — capture the as-found advanced-mode JSON

**Files:**
- Create: `tools/jetson_dump_asfound.py`
- Create: `server/presets/custom-as-found-2026-08-29.json` (its content comes off the device)

**Interfaces:**
- Produces: the JSON file that Task 4's `dump_advanced_mode_json` writes in the same format at every service start, and that Task 13's acceptance reloads.

Read-only on the device. Must run BEFORE Task 4 changes `depth_units` (the depth table, including `depthUnits`, is inside this JSON).

- [ ] **Step 1: Write the dump tool**

```python
# tools/jetson_dump_asfound.py
"""Dump the D435i's advanced-mode configuration AS FOUND, over SSH, into the repo.

Read-only on the camera. Run once BEFORE any depth_units/preset change so the
configuration the 2026-08-13 characterisation was measured under is on record.

    py -3.10 tools/jetson_dump_asfound.py            # -> server/presets/custom-as-found-<date>.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SSH = ["ssh", "-i", str(Path.home() / ".ssh" / "jetson_robodk"), "jetson@10.12.171.70"]
VENV_PY = "/home/jetson/EtherSenseServer/ethenv/bin/python"

REMOTE = r'''
import json, sys
import pyrealsense2 as rs
ctx = rs.context()
devs = ctx.query_devices()
if len(devs) == 0:
    sys.exit("no RealSense device")
dev = devs[0]
adv = rs.rs400_advanced_mode(dev)
if not adv.is_enabled():
    sys.exit("advanced mode is not enabled on this device")
out = {"serial": dev.get_info(rs.camera_info.serial_number),
       "firmware": dev.get_info(rs.camera_info.firmware_version),
       "librealsense": rs.__version__,
       "advanced_mode": json.loads(adv.serialize_json())}
print(json.dumps(out))
'''


def main() -> int:
    # The service holds the device; the dump needs it stopped for the ~2 s read.
    subprocess.run(SSH + ["sudo -n systemctl stop realsense-camera"], check=True)
    try:
        proc = subprocess.run(SSH + [f"{VENV_PY} -c {json.dumps(REMOTE)}"],
                              capture_output=True, text=True, check=True)
    finally:
        subprocess.run(SSH + ["sudo -n systemctl start realsense-camera"], check=True)
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    payload["captured_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = REPO / "server" / "presets" / f"custom-as-found-{time.strftime('%Y-%m-%d')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note: `sudo -n` needs passwordless sudo for `systemctl` on the Jetson. If it prompts, run the two systemctl lines by hand with `echo '<pw>' | sudo -S ...` (password in `secrets/jetson.env`, `JETSON_SUDO_PASSWORD`) and run only the middle SSH command from the tool.

- [ ] **Step 2: Run it with the arm parked and no client connected**

Run: `py -3.10 tools/jetson_dump_asfound.py`
Expected: `wrote ...server/presets/custom-as-found-2026-08-29.json`; the file's `advanced_mode.parameters` contains a `depth-table` block with `depthUnits: 1000` and a `depth-control` block. If `depthUnits` is not 1000, STOP and report — the device is not in the state the audit measured.

- [ ] **Step 3: Verify the service is back**

Run: `py -3.10 tools/jetson_deploy.py status`
Expected: `active`, `LISTENING`

- [ ] **Step 4: Commit + push**

```bash
git add tools/jetson_dump_asfound.py server/presets/custom-as-found-2026-08-29.json
git commit -m "feat(realsense): record the as-found ASIC configuration (audit R4.1)"
git push origin sensor-layer-v2
```

---

### Task 3: R1 — rebuild librealsense 2.55.1 as Release + CUDA + OpenMP

**Files:**
- Create: `docs/jetson-librealsense-rebuild.md` (the runbook, with the measured before/after numbers filled in)

**Interfaces:**
- Produces: `~/librealsense/build_cuda/` on the Jetson; the venv's `pyrealsense2` re-pointed at it; `~/librealsense/build_py310/` kept intact as rollback.

Infra only; zero app code. The OLD wire protocol must still serve identically afterwards — that is how this task is proven independent of Task 13.

- [ ] **Step 1: Record the BEFORE numbers**

```sh
ssh -i ~/.ssh/jetson_robodk jetson@10.12.171.70 '
  top -bn1 | grep -E "python|Cpu" | head -3
  /home/jetson/EtherSenseServer/ethenv/bin/python -c "import pyrealsense2 as rs; print(rs.__version__, rs.__file__)"
  ldd $(/home/jetson/EtherSenseServer/ethenv/bin/python -c "import pyrealsense2 as m;print(m.__file__)") | grep realsense'
```
Then from the workstation, per-frame time on the OLD protocol (10 grabs):
```
py -3.10 -c "import time; from tasni.core.config import load_config; from tasni.core.camera import CameraClient; c=CameraClient(load_config().camera); [c.grab(with_depth=True, timeout=20) for _ in range(2)]; t=time.perf_counter(); [c.grab(with_depth=True, timeout=20) for _ in range(10)]; print('ms/grab', (time.perf_counter()-t)*100)"
```
Write both into the runbook under "Before".

- [ ] **Step 2: Build (about 1 h on the Nano; add swap first)**

```sh
ssh -i ~/.ssh/jetson_robodk jetson@10.12.171.70 '
  set -e
  free -m | head -2
  # 4 GB RAM: CUDA translation units OOM at -j4. 4 GB of swap makes -j2 safe.
  if [ ! -f /swapfile ]; then echo "<pw>" | sudo -S sh -c "fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile"; fi
  cd ~/librealsense && git fetch --tags && git checkout v2.55.1
  mkdir -p build_cuda && cd build_cuda
  cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_CUDA=true -DBUILD_WITH_OPENMP=true \
    -DBUILD_PYTHON_BINDINGS=true \
    -DPYTHON_EXECUTABLE=/home/jetson/EtherSenseServer/ethenv/bin/python \
    -DBUILD_EXAMPLES=false -DBUILD_GRAPHICAL_EXAMPLES=false -DFORCE_RSUSB_BACKEND=false
  grep -E "^BUILD_WITH_CUDA|^BUILD_WITH_OPENMP|^CMAKE_BUILD_TYPE" CMakeCache.txt
  nohup make -j2 > build.log 2>&1 &
  echo started'
```
Poll `tail -3 ~/librealsense/build_cuda/build.log` until `[100%]`. Expected CMakeCache lines: `BUILD_WITH_CUDA:BOOL=true`, `BUILD_WITH_OPENMP:BOOL=true`, `CMAKE_BUILD_TYPE:STRING=Release`.

- [ ] **Step 3: Re-point the venv binding (keep the old one)**

```sh
ssh -i ~/.ssh/jetson_robodk jetson@10.12.171.70 '
  set -e
  SP=$(/home/jetson/EtherSenseServer/ethenv/bin/python -c "import pyrealsense2,os;print(os.path.dirname(pyrealsense2.__file__))")
  echo "site dir: $SP"; ls -la $SP | grep -i realsense
  mkdir -p ~/pyrealsense2_rollback && cp -a $SP/pyrealsense2* ~/pyrealsense2_rollback/
  NEW=$(ls ~/librealsense/build_cuda/wrappers/python/pyrealsense2*.so | head -1)
  echo "new binding: $NEW"
  cp -f $NEW $SP/
  cp -f ~/librealsense/build_cuda/librealsense2.so.2.55* $SP/ 2>/dev/null || true
  /home/jetson/EtherSenseServer/ethenv/bin/python -c "import pyrealsense2 as rs; print(rs.__version__); print(rs.__file__)"
  ldd $SP/pyrealsense2*.so | grep realsense'
```
Expected: version prints `2.55.1`, `ldd` resolves `librealsense2.so.2.55` from `build_cuda` (or the copied one), NOT `build_py310`. If the `.so` name the binding links is not found, `patchelf --set-rpath ~/librealsense/build_cuda` on the binding, or add `Environment=LD_LIBRARY_PATH=/home/jetson/librealsense/build_cuda` to the unit (then `bootstrap`).

Rollback (one line): `cp -f ~/pyrealsense2_rollback/pyrealsense2* $SP/ && sudo systemctl restart realsense-camera`.

- [ ] **Step 4: Restart and prove the OLD protocol still serves identically**

```
py -3.10 tools/jetson_deploy.py restart
py -3.10 tools/jetson_deploy.py status
```
Journal must show `RealSense: laser_power ...`, `visual_preset left as-is at 0`, no tracebacks. Then re-run the Step 1 ms/grab probe and `top`. Expected: idle CPU well under 110 %; ms/grab lower. Write into the runbook under "After".

- [ ] **Step 5: Plane RMS sanity on the characterisation board (same pose as the last dated run)**

Run: `py -3.10 tools/characterize_distance.py --help` and run one stop at the usual standoff; plane RMS within noise of the 2026-08-13 record. CUDA align rounds slightly differently; a change is expected, a worsening is a bug.

- [ ] **Step 6: Write the runbook + commit + push**

`docs/jetson-librealsense-rebuild.md`: the exact commands above, the CMakeCache lines observed, Before/After table (idle CPU %, ms/grab, plane RMS), the rollback line, and the swapfile note.

```bash
git add docs/jetson-librealsense-rebuild.md
git commit -m "docs(jetson): librealsense 2.55.1 Release+CUDA rebuild runbook with before/after (audit R1)"
git push origin sensor-layer-v2
```

---

### Task 4: `server/handshake.py` + `server/rs_config.py` (pure, host-testable)

**Files:**
- Create: `server/handshake.py`
- Create: `server/rs_config.py`
- Test: `tests/test_handshake.py`, `tests/test_rs_config.py`

**Interfaces:**
- Produces: `parse_handshake(req: bytes) -> dict` with keys `mode` (`"full"|"color"|"burst"|"telemetry"`), `v2: bool`, `codec` (`"jpeg"|"h264"`), `quality: int|None`, `bitrate: int`, `scan_telemetry: bool`, `depth_requested: bool` (True for full/burst).
- Produces: `rs_config.DEPTH_UNITS_M = 0.0001`; `configure_depth_sensor(sensor, rs, *, laser_power: float, visual_preset: int, log=print) -> dict` (achieved values by option name, incl. `"depth_unit_mm"`); `read_temperatures(sensor, rs) -> dict`; `read_global_time_enabled(sensor, rs) -> bool|None`; `dump_advanced_mode_json(device, rs, out_dir: str, log=print) -> str|None`; `librealsense_version(rs) -> str`.

The existing test files stub `pyrealsense2` with `sys.modules.setdefault("pyrealsense2", SimpleNamespace())`, so anything importing it at module level must tolerate a bare namespace. Both new modules take `rs` as a parameter and never import it.

- [ ] **Step 1: Write the failing handshake tests**

```python
# tests/test_handshake.py
"""The server's handshake parse is the version gate: a depth client without V2 is refused."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from server.handshake import parse_handshake  # noqa: E402


def test_full_v2_is_the_only_accepted_depth_stream():
    assert parse_handshake(b"MODE FULL V2\n") == {
        "mode": "full", "v2": True, "codec": "jpeg", "quality": None,
        "bitrate": 4000, "scan_telemetry": False, "depth_requested": True}


def test_no_handshake_and_old_full_are_depth_requests_without_v2():
    for req in (b"", b"MODE FULL\n", b"garbage"):
        p = parse_handshake(req)
        assert p["mode"] == "full" and p["depth_requested"] and not p["v2"], req


def test_burst_needs_v2_too():
    assert parse_handshake(b"MODE BURST\n")["v2"] is False
    p = parse_handshake(b"MODE BURST V2\n")
    assert p["mode"] == "burst" and p["v2"] and p["depth_requested"]


def test_color_and_telemetry_are_unchanged_and_never_depth():
    p = parse_handshake(b"MODE COLOR H264 B6000 SCAN\n")
    assert p["mode"] == "color" and p["codec"] == "h264" and p["bitrate"] == 6000
    assert p["scan_telemetry"] and not p["depth_requested"]
    assert parse_handshake(b"MODE COLOR Q5\n")["quality"] == 10        # clamped low
    assert parse_handshake(b"MODE COLOR Q999\n")["quality"] == 100     # clamped high
    assert parse_handshake(b"C")["mode"] == "color"
    assert parse_handshake(b"MODE TELEMETRY\n")["mode"] == "telemetry"
    assert parse_handshake(b"MODE TELEMETRY\n")["depth_requested"] is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `py -3.10 -m pytest tests/test_handshake.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.handshake'`

- [ ] **Step 3: Write `server/handshake.py`**

```python
"""Client handshake parsing for the camera server (pure; importable on the host).

One line, sent right after connect, declares the stream a client wants:

    MODE FULL V2                  depth+colour, protocol 2 (greeting, raw depth)
    MODE BURST V2                 burst capture of protocol-2 frames
    MODE COLOR [Q<n>] [H264 [B<kbps>]] [SCAN]   colour-only paths (unchanged)
    MODE TELEMETRY                scan telemetry side-channel (unchanged)

Anything else -- including NO line and the pre-V2 "MODE FULL" -- parses as a
depth request WITHOUT v2. The server refuses those: a host that did not restart
after the protocol change must fail loudly at the handshake, not misread the JSON
greeting as a frame length and hang.
"""
from __future__ import annotations

DEFAULT_H264_BITRATE_KBPS = 4000


def parse_handshake(req: bytes) -> dict:
    req = bytes(req).strip().upper()
    tokens = req.split()
    mode = "full"
    if req.startswith(b"MODE BURST"):
        mode = "burst"
    elif req.startswith(b"MODE TELEMETRY"):
        mode = "telemetry"
    elif req.startswith(b"MODE COLOR") or req == b"C":
        mode = "color"
    codec, quality, bitrate = "jpeg", None, DEFAULT_H264_BITRATE_KBPS
    for tok in tokens:
        if tok == b"H264":
            codec = "h264"
        elif tok.startswith(b"Q") and tok[1:].isdigit():
            quality = max(10, min(100, int(tok[1:])))
        elif tok.startswith(b"B") and tok[1:].isdigit():
            bitrate = max(500, min(20000, int(tok[1:])))
    return {
        "mode": mode,
        "v2": b"V2" in tokens,
        "codec": codec,
        "quality": quality,
        "bitrate": bitrate,
        "scan_telemetry": b"SCAN" in tokens,
        "depth_requested": mode in ("full", "burst"),
    }
```

- [ ] **Step 4: Run the handshake tests**

Run: `py -3.10 -m pytest tests/test_handshake.py -v`
Expected: 4 PASS

- [ ] **Step 5: Write the failing rs_config tests (fake sensor + fake rs namespace)**

```python
# tests/test_rs_config.py
"""Device option setup must READ BACK what stuck, set depth_units to 0.1 mm, and
never raise on an unsupported option (the unit is Restart=always: an exception
here is a crash-loop with the camera dark)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from server import rs_config  # noqa: E402


class FakeSensor:
    def __init__(self, supported, ranges, initial=None):
        self._supported = set(supported)
        self._ranges = ranges
        self.values = dict(initial or {})
        self.depth_scale = 0.001

    def supports(self, opt): return opt in self._supported
    def get_option_range(self, opt): return SimpleNamespace(min=self._ranges[opt][0], max=self._ranges[opt][1])
    def set_option(self, opt, v):
        self.values[opt] = float(v)
        if opt == "depth_units":
            self.depth_scale = float(v)
    def get_option(self, opt): return self.values.get(opt, 0.0)
    def get_depth_scale(self): return self.depth_scale


FAKE_RS = SimpleNamespace(option=SimpleNamespace(
    emitter_enabled="emitter_enabled", laser_power="laser_power",
    visual_preset="visual_preset", depth_units="depth_units",
    auto_exposure_priority="auto_exposure_priority",
    asic_temperature="asic_temperature", projector_temperature="projector_temperature",
    global_time_enabled="global_time_enabled"), __version__="2.55.1")


def _sensor():
    return FakeSensor(
        supported={"emitter_enabled", "laser_power", "visual_preset", "depth_units",
                   "asic_temperature", "projector_temperature", "global_time_enabled"},
        ranges={"emitter_enabled": (0, 1), "laser_power": (0, 360), "visual_preset": (0, 5),
                "depth_units": (0.00001, 0.01)},
        initial={"laser_power": 150.0, "visual_preset": 0.0, "asic_temperature": 41.5,
                 "projector_temperature": 38.0, "global_time_enabled": 1.0})


def test_depth_units_are_set_and_read_back_as_0_1_mm():
    s = _sensor()
    achieved = rs_config.configure_depth_sensor(s, FAKE_RS, laser_power=-1, visual_preset=-1,
                                                log=lambda *_: None)
    assert s.values["depth_units"] == 0.0001
    assert achieved["depth_unit_mm"] == 0.1
    assert achieved["emitter_enabled"] == 1.0
    assert achieved["laser_power"] == 150.0          # left alone, still reported
    assert achieved["visual_preset"] == 0.0


def test_unsupported_option_is_skipped_not_fatal():
    s = FakeSensor(supported={"emitter_enabled"}, ranges={"emitter_enabled": (0, 1)})
    achieved = rs_config.configure_depth_sensor(s, FAKE_RS, laser_power=300, visual_preset=4,
                                                log=lambda *_: None)
    assert achieved["emitter_enabled"] == 1.0
    assert "depth_units" not in s.values
    assert achieved["depth_unit_mm"] == 1.0          # from get_depth_scale(), whatever it is


def test_temperatures_and_global_time_read_back():
    s = _sensor()
    assert rs_config.read_temperatures(s, FAKE_RS) == {"asic_c": 41.5, "projector_c": 38.0}
    assert rs_config.read_global_time_enabled(s, FAKE_RS) is True


def test_as_found_dump_writes_dated_json(tmp_path):
    class FakeAdv:
        def __init__(self, dev): pass
        def is_enabled(self): return True
        def serialize_json(self): return json.dumps({"parameters": {"depth-table": {"depthUnits": 1000}}})
    rs = SimpleNamespace(rs400_advanced_mode=FakeAdv, camera_info=SimpleNamespace(
        serial_number="serial_number", firmware_version="firmware_version"), __version__="2.55.1")
    dev = SimpleNamespace(get_info=lambda k: {"serial_number": "S1", "firmware_version": "5.16"}[k])
    path = rs_config.dump_advanced_mode_json(dev, rs, str(tmp_path), log=lambda *_: None)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["advanced_mode"]["parameters"]["depth-table"]["depthUnits"] == 1000
    assert data["serial"] == "S1" and data["librealsense"] == "2.55.1"
    assert Path(path).name.startswith("asfound-")
```

- [ ] **Step 6: Run to verify they fail**

Run: `py -3.10 -m pytest tests/test_rs_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'rs_config'`

- [ ] **Step 7: Write `server/rs_config.py`**

```python
"""RealSense device configuration with READ-BACK, plus the read-only facts the
greeting carries (temperatures, global time, library version, as-found JSON).

Every function takes the ``rs`` module as a parameter: the host test suite stubs
``pyrealsense2`` with a bare namespace, and this module must import there.

Option state lives on the DEVICE and survives restarts, so laser power and the
visual preset default to leave-alone (-1) -- a silent default would invalidate the
dated depth characterisation. ``depth_units`` is the exception: 0.1 mm words are
the whole point (audit R2), so it is always set, and always read back.
"""
from __future__ import annotations

import json
import os
import time

DEPTH_UNITS_M = 0.0001            # 0.1 mm per uint16 step; 6.55 m ceiling


def _opt(rs, name):
    return getattr(getattr(rs, "option", None), name, None)


def _set_with_readback(sensor, name, option, value, log) -> float | None:
    if option is None or not sensor.supports(option):
        log(f"RealSense: {name} unsupported on this device/build - skipped")
        return None
    try:
        rng = sensor.get_option_range(option)
        clamped = min(max(float(value), rng.min), rng.max)
        sensor.set_option(option, clamped)
        got = float(sensor.get_option(option))
        log(f"RealSense: {name} -> requested {value:g}, set {clamped:g}, device reports "
            f"{got:g} (range {rng.min:g}..{rng.max:g})")
        return got
    except Exception as e:  # noqa: BLE001 - never take the service down over one option
        log(f"WARNING: could not set {name}={value}: {e}")
        return None


def _read(sensor, name, option, log) -> float | None:
    if option is None or not sensor.supports(option):
        return None
    try:
        return float(sensor.get_option(option))
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: could not read {name}: {e}")
        return None


def configure_depth_sensor(sensor, rs, *, laser_power: float, visual_preset: int,
                           log=print) -> dict:
    """Set emitter on, optional laser/preset, depth_units, AE priority off; return
    the ACHIEVED values (read back) keyed by option name, plus ``depth_unit_mm``."""
    achieved: dict = {}
    if visual_preset >= 0:
        achieved["visual_preset"] = _set_with_readback(
            sensor, "visual_preset", _opt(rs, "visual_preset"), float(visual_preset), log)
    else:
        achieved["visual_preset"] = _read(sensor, "visual_preset", _opt(rs, "visual_preset"), log)
        log(f"RealSense: visual_preset left as-is at {achieved['visual_preset']} "
            "(set RS_VISUAL_PRESET to change it)")
    if laser_power >= 0:
        achieved["laser_power"] = _set_with_readback(
            sensor, "laser_power", _opt(rs, "laser_power"), float(laser_power), log)
    else:
        achieved["laser_power"] = _read(sensor, "laser_power", _opt(rs, "laser_power"), log)
        log(f"RealSense: laser_power left as-is at {achieved['laser_power']} "
            "(set RS_LASER_POWER to change it)")
    achieved["emitter_enabled"] = _set_with_readback(
        sensor, "emitter_enabled", _opt(rs, "emitter_enabled"), 1.0, log)
    achieved["depth_units"] = _set_with_readback(
        sensor, "depth_units", _opt(rs, "depth_units"), DEPTH_UNITS_M, log)
    # Frame rate is a contract for every client of the shared pipeline; AE priority
    # lets the sensor drop below 30 fps in dim light and stalls wait_for_frames.
    achieved["auto_exposure_priority"] = _set_with_readback(
        sensor, "auto_exposure_priority", _opt(rs, "auto_exposure_priority"), 0.0, log)
    try:
        achieved["depth_unit_mm"] = float(sensor.get_depth_scale()) * 1000.0
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: could not read depth scale: {e}")
        achieved["depth_unit_mm"] = None
    log(f"RealSense: depth_unit_mm = {achieved['depth_unit_mm']}")
    return achieved


def read_temperatures(sensor, rs, log=print) -> dict:
    return {"asic_c": _read(sensor, "asic_temperature", _opt(rs, "asic_temperature"), log),
            "projector_c": _read(sensor, "projector_temperature",
                                 _opt(rs, "projector_temperature"), log)}


def read_global_time_enabled(sensor, rs, log=print) -> bool | None:
    v = _read(sensor, "global_time_enabled", _opt(rs, "global_time_enabled"), log)
    return None if v is None else bool(v)


def librealsense_version(rs) -> str:
    return str(getattr(rs, "__version__", "unknown"))


def dump_advanced_mode_json(device, rs, out_dir: str, log=print) -> str | None:
    """Write the ASIC configuration AS FOUND to ``<out_dir>/asfound-<stamp>.json``.
    Read-only on the device. Returns the path, or None if advanced mode is absent."""
    adv_cls = getattr(rs, "rs400_advanced_mode", None)
    if adv_cls is None:
        log("RealSense: rs400_advanced_mode not available in this build - no as-found dump")
        return None
    try:
        adv = adv_cls(device)
        if not adv.is_enabled():
            log("RealSense: advanced mode not enabled - no as-found dump")
            return None
        payload = {
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "serial": device.get_info(rs.camera_info.serial_number),
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "librealsense": librealsense_version(rs),
            "advanced_mode": json.loads(adv.serialize_json()),
        }
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: as-found dump failed: {e}")
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"asfound-{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    log(f"RealSense: as-found advanced-mode JSON written to {path}")
    return path
```

- [ ] **Step 8: Run the rs_config tests**

Run: `py -3.10 -m pytest tests/test_rs_config.py tests/test_handshake.py -v`
Expected: 8 PASS

- [ ] **Step 9: Commit + push**

```bash
git add server/handshake.py server/rs_config.py tests/test_handshake.py tests/test_rs_config.py
git commit -m "feat(camera server): pure handshake parser + device config with read-back, depth_units 0.1 mm"
git push origin sensor-layer-v2
```

---

### Task 5: `server/rs_geometry.py` — intrinsics, asserted row-major extrinsic, greeting

**Files:**
- Create: `server/rs_geometry.py`
- Test: `tests/test_rs_geometry.py`

**Interfaces:**
- Consumes: nothing from Task 4 except the greeting fields it is handed.
- Produces: `intrinsics_dict(video_profile) -> dict`; `extrinsic_row_major(depth_profile, color_profile, rs) -> tuple[np.ndarray, np.ndarray]` (R 3x3 row-major, t in **mm**; raises `RuntimeError` if the transpose does not reproduce `rs2_transform_point_to_point`); `StaticGeometry` dataclass (`depth`, `color`, `R_dc`, `t_dc_mm`, `depth_size`, `color_size`); `static_geometry(profile, rs) -> StaticGeometry`; `build_greeting(static: StaticGeometry, *, depth_unit_mm, filters, temps, global_time_enabled, achieved, device) -> dict`; `greeting_line(g: dict) -> bytes`.

librealsense documents `rs2_extrinsics.rotation` as column-major. `rs2_transform_point_to_point` computes `out[i] = rot[i] * p0 + rot[i+3] * p1 + rot[i+6] * p2 + t[i]`, so row-major `R = reshape(3,3).T`. The assert makes that a checked fact, not a belief.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rs_geometry.py
"""The greeting's extrinsic must be row-major and PROVEN against the SDK's own
transform; the server used to re-derive this empirically every telemetry frame."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from server import rs_geometry  # noqa: E402


def _rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


R_TRUE = _rot_z(3.0) @ np.array([[1, 0, 0], [0, 0.999, -0.0447], [0, 0.0447, 0.999]])
T_TRUE_M = np.array([0.0147, -0.0002, 0.0003])


def _fake_rs(column_major: bool):
    """A pyrealsense2 stand-in whose rs2_transform_point_to_point is the ground
    truth, and whose .rotation layout is column-major (correct) or not (bug)."""
    rot = (R_TRUE.T if column_major else R_TRUE).reshape(-1).tolist()
    ext = SimpleNamespace(rotation=rot, translation=T_TRUE_M.tolist())
    def transform(e, p):
        return (R_TRUE @ np.asarray(p, float) + T_TRUE_M).tolist()
    return ext, SimpleNamespace(rs2_transform_point_to_point=transform)


def _profiles(ext):
    intr_d = SimpleNamespace(width=1280, height=720, fx=640.0, fy=640.0, ppx=640.0, ppy=360.0,
                             model=SimpleNamespace(name="brown_conrady"), coeffs=[0, 0, 0, 0, 0])
    intr_c = SimpleNamespace(width=1920, height=1080, fx=1362.0, fy=1362.0, ppx=975.0, ppy=550.0,
                             model=SimpleNamespace(name="brown_conrady"), coeffs=[0, 0, 0, 0, 0])
    depth = SimpleNamespace(intrinsics=intr_d, get_extrinsics_to=lambda other: ext)
    color = SimpleNamespace(intrinsics=intr_c)
    return depth, color


def test_column_major_rotation_is_transposed_into_row_major_and_verified():
    ext, rs = _fake_rs(column_major=True)
    depth, color = _profiles(ext)
    R, t_mm = rs_geometry.extrinsic_row_major(depth, color, rs)
    np.testing.assert_allclose(R, R_TRUE, atol=1e-12)
    np.testing.assert_allclose(t_mm, T_TRUE_M * 1000.0, atol=1e-9)


def test_wrong_layout_is_refused_not_guessed():
    ext, rs = _fake_rs(column_major=False)
    depth, color = _profiles(ext)
    try:
        rs_geometry.extrinsic_row_major(depth, color, rs)
    except RuntimeError as e:
        assert "extrinsic" in str(e)
    else:
        raise AssertionError("a mismatching rotation layout must raise")


def test_greeting_is_one_json_line_with_protocol_2():
    ext, rs = _fake_rs(column_major=True)
    depth, color = _profiles(ext)
    static = rs_geometry.StaticGeometry(
        depth=rs_geometry.intrinsics_dict(depth.intrinsics),
        color=rs_geometry.intrinsics_dict(color.intrinsics),
        R_dc=R_TRUE, t_dc_mm=T_TRUE_M * 1000.0, depth_size=(1280, 720), color_size=(1920, 1080))
    g = rs_geometry.build_greeting(
        static, depth_unit_mm=0.1, filters=["threshold", "disparity", "spatial", "temporal",
                                            "disparity_inv"],
        temps={"asic_c": 41.5, "projector_c": 38.0}, global_time_enabled=True,
        achieved={"visual_preset": 0.0, "laser_power": 150.0},
        device={"serial": "S1", "fw": "5.16.00.01", "librealsense": "2.55.1"})
    line = rs_geometry.greeting_line(g)
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    back = json.loads(line.decode("utf-8"))
    assert back["protocol"] == 2 and back["aligned"] is False
    assert back["depth_unit_mm"] == 0.1
    assert back["depth"]["width"] == 1280 and back["color"]["width"] == 1920
    np.testing.assert_allclose(back["depth_to_color"]["rotation_row_major"], R_TRUE, atol=1e-12)
    assert back["depth_to_color"]["translation_mm"][0] == 14.7
    assert back["device"]["visual_preset"] == 0.0 and back["temps"]["asic_c"] == 41.5
```

- [ ] **Step 2: Run to verify they fail**

Run: `py -3.10 -m pytest tests/test_rs_geometry.py -v`
Expected: FAIL — `ImportError: cannot import name 'rs_geometry'`

- [ ] **Step 3: Write `server/rs_geometry.py`**

```python
"""Depth/colour camera models and the depth->colour extrinsic, for the greeting.

Takes the ``rs`` module as a parameter (host tests stub pyrealsense2). The
extrinsic is transposed from librealsense's column-major layout to row-major and
CHECKED against ``rs2_transform_point_to_point`` on a test point; a mismatch
raises. The server used to resolve the layout empirically on every telemetry
frame by projecting eight sample points both ways (see git history of
``stream_h264``); the host must never inherit that guess.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


def intrinsics_dict(intr) -> dict:
    model = getattr(intr, "model", None)
    return {
        "width": int(intr.width), "height": int(intr.height),
        "fx": float(intr.fx), "fy": float(intr.fy),
        "ppx": float(intr.ppx), "ppy": float(intr.ppy),
        "model": str(getattr(model, "name", model)) if model is not None else "none",
        "coeffs": [float(c) for c in (getattr(intr, "coeffs", None) or [0, 0, 0, 0, 0])],
    }


def extrinsic_row_major(depth_profile, color_profile, rs) -> tuple[np.ndarray, np.ndarray]:
    ext = depth_profile.get_extrinsics_to(color_profile)
    R = np.asarray(ext.rotation, dtype=float).reshape(3, 3).T      # column-major -> row-major
    t_m = np.asarray(ext.translation, dtype=float)
    probe = [0.12, -0.05, 0.45]                                       # metres, in front of the camera
    expected = np.asarray(rs.rs2_transform_point_to_point(ext, probe), dtype=float)
    got = R @ np.asarray(probe) + t_m
    if not np.allclose(got, expected, atol=1e-6):
        raise RuntimeError(
            f"depth->colour extrinsic layout check failed: transposed rotation maps the "
            f"probe to {got.tolist()} but the SDK says {expected.tolist()}; refusing to "
            f"serve a geometry that would be wrong for every client")
    return R, t_m * 1000.0


@dataclass(frozen=True)
class StaticGeometry:
    depth: dict
    color: dict
    R_dc: np.ndarray
    t_dc_mm: np.ndarray
    depth_size: tuple[int, int]
    color_size: tuple[int, int]


def static_geometry(profile, rs) -> StaticGeometry:
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    R, t_mm = extrinsic_row_major(depth_profile, color_profile, rs)
    d, c = intrinsics_dict(depth_profile.intrinsics), intrinsics_dict(color_profile.intrinsics)
    return StaticGeometry(depth=d, color=c, R_dc=R, t_dc_mm=t_mm,
                          depth_size=(d["width"], d["height"]),
                          color_size=(c["width"], c["height"]))


def build_greeting(static: StaticGeometry, *, depth_unit_mm: float, filters: list,
                   temps: dict, global_time_enabled, achieved: dict, device: dict) -> dict:
    return {
        "protocol": 2,
        "aligned": False,
        "depth_unit_mm": float(depth_unit_mm),
        "depth": dict(static.depth),
        "color": dict(static.color),
        "depth_to_color": {
            "rotation_row_major": np.asarray(static.R_dc, float).round(12).tolist(),
            "translation_mm": np.asarray(static.t_dc_mm, float).round(6).tolist(),
        },
        "filters": list(filters),
        "device": {**dict(device),
                   "visual_preset": achieved.get("visual_preset"),
                   "laser_power": achieved.get("laser_power")},
        "temps": dict(temps),
        "global_time_enabled": global_time_enabled,
    }


def greeting_line(greeting: dict) -> bytes:
    return json.dumps(greeting, separators=(",", ":")).encode("utf-8") + b"\n"
```

- [ ] **Step 4: Run the tests**

Run: `py -3.10 -m pytest tests/test_rs_geometry.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit + push**

```bash
git add server/rs_geometry.py tests/test_rs_geometry.py
git commit -m "feat(camera server): depth/colour geometry with an asserted row-major extrinsic, greeting builder"
git push origin sensor-layer-v2
```

---

### Task 6: The server — V2 handshake, greeting, refusal, no `align`, new filters, 1080p colour

**Files:**
- Modify: `server/server_unicast_syncronous.py` (imports `:1-25`; `scan_plane_telemetry` unchanged; supervision globals `:640-650`; `_rebuild_pipeline` `:711-760`; `getFrames` `:807-825`; `width/height` `:827-828`; `_env_number`/env `:848-870`; `set_high_accuracy_preset` `:873-936` DELETED; `openPipeline` `:939-947`; `handle_client` `:950-1076`; `stream_h264` `:1089-1257`; `stream_burst` `:1276-1356`; `setup_depth_filters` `:1359-1366`; `__main__` `:1410-1422`)
- Test: `tests/test_server_env.py` (still imports the module with stubs — must keep passing), `tests/test_scan_telemetry_server.py` (same)

**Interfaces:**
- Consumes: `parse_handshake`, `rs_config.*`, `rs_geometry.*` from Tasks 4-5.
- Produces (wire): `MODE FULL V2` -> greeting line -> frames; `MODE BURST V2` -> `BURST READY\n` -> greeting line -> CAP/GET/CLEAR loop; refusal line `ERR protocol 2 required; send MODE FULL V2\n`.

Module-level state after this task: `pipeline`, `depth_unit_mm`, `depth_filters`, `STATIC_GEOMETRY` (a `StaticGeometry`), `ACHIEVED_OPTIONS` (dict), `DEVICE_INFO` (dict). `align` is gone everywhere.

- [ ] **Step 1: Imports and constants**

Replace `width = 1280;` / `height = 720;` (`:827-828`) with:

```python
DEPTH_SIZE = (1280, 720)      # the D435i's top depth mode
COLOR_SIZE = (1920, 1080)     # audit R7: ChArUco corner precision bounds every downstream number
```

Add after the existing imports:

```python
from handshake import parse_handshake
import rs_config
import rs_geometry

ASFOUND_DIR = os.environ.get("RS_ASFOUND_DIR", "/home/jetson/robodk-characterization")
# Work-volume clip ahead of the spatial filter (audit R5): background depth must not
# be smoothed into surface edges, and nothing may fabricate depth.
RS_DEPTH_MIN_M = 0.15
RS_DEPTH_MAX_M = 1.5
```

(`rs_config`/`rs_geometry` are imported as top-level names because the service runs with `WorkingDirectory=/home/jetson/robodk/server`; the host tests put `server/` on `sys.path` the same way `scan_overlay` is found.)

- [ ] **Step 2: Replace the whole `set_high_accuracy_preset` + `openPipeline` block (`:873-947`)**

```python
STATIC_GEOMETRY = None      # rs_geometry.StaticGeometry, set by openPipeline
ACHIEVED_OPTIONS = {}       # read-back values, set by openPipeline
DEVICE_INFO = {}


def openPipeline():
    """Start depth 720p + colour 1080p (no infrared: nobody reads it), configure the
    depth sensor with read-back, record the as-found ASIC JSON, and extract the
    static geometry the greeting carries. Returns the pipeline."""
    global STATIC_GEOMETRY, ACHIEVED_OPTIONS, DEVICE_INFO
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, DEPTH_SIZE[0], DEPTH_SIZE[1], rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, COLOR_SIZE[0], COLOR_SIZE[1], rs.format.bgr8, 30)
    pipeline = rs.pipeline()
    profile = pipeline.start(cfg)
    device = profile.get_device()
    # As-found FIRST: the depth table (incl. depthUnits) is part of this record.
    rs_config.dump_advanced_mode_json(device, rs, ASFOUND_DIR, log=_log)
    sensor = device.first_depth_sensor()
    ACHIEVED_OPTIONS = rs_config.configure_depth_sensor(
        sensor, rs, laser_power=RS_LASER_POWER, visual_preset=RS_VISUAL_PRESET, log=_log)
    DEVICE_INFO = {
        "serial": device.get_info(rs.camera_info.serial_number),
        "fw": device.get_info(rs.camera_info.firmware_version),
        "librealsense": rs_config.librealsense_version(rs),
    }
    STATIC_GEOMETRY = rs_geometry.static_geometry(profile, rs)     # raises on a bad extrinsic
    gte = rs_config.read_global_time_enabled(sensor, rs, log=_log)
    _log(f"RealSense: global_time_enabled = {gte}; temps = "
         f"{rs_config.read_temperatures(sensor, rs, log=_log)}; "
         f"depth {STATIC_GEOMETRY.depth_size} colour {STATIC_GEOMETRY.color_size}; "
         f"depth->colour t = {STATIC_GEOMETRY.t_dc_mm.round(2).tolist()} mm")
    return pipeline


def _log(msg):
    print(msg, flush=True)


def make_greeting() -> dict:
    """The per-connection greeting: static geometry + LIVE temps + achieved options."""
    sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
    return rs_geometry.build_greeting(
        STATIC_GEOMETRY, depth_unit_mm=depth_unit_mm,
        filters=["threshold", "disparity", "spatial", "temporal", "disparity_inv"],
        temps=rs_config.read_temperatures(sensor, rs, log=_log),
        global_time_enabled=rs_config.read_global_time_enabled(sensor, rs, log=_log),
        achieved=ACHIEVED_OPTIONS, device=DEVICE_INFO)
```

Keep `_env_number`, `RS_VISUAL_PRESET`, `RS_LASER_POWER` exactly as they are (the env tests depend on them).

- [ ] **Step 3: `getFrames` and the filter chain**

Replace `getFrames(align, depth_filters)` (`:807-825`) with:

```python
def getFrames(depth_filters):
    """One native depth frame (filtered) + the colour frame. NOT aligned: the host
    back-projects with the depth intrinsics it received in the greeting."""
    frames = read_frames()
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    if not depth or not color:
        return None, None, None
    for f in depth_filters:
        depth = f.process(depth)
    return (np.asanyarray(depth.get_data()), np.asanyarray(color.get_data()),
            frames.get_timestamp())
```

Replace `setup_depth_filters` (`:1359-1366`) with:

```python
def setup_depth_filters():
    """threshold -> disparity -> spatial -> temporal -> disparity_inv, on NATIVE depth.
    No decimation (full-resolution scan data) and NO hole filling: a filled pixel is
    fabricated depth, and it was fabricated exactly where the metrology cares
    (surface edges). Threshold first so background is never smoothed into an edge."""
    threshold = rs.threshold_filter(RS_DEPTH_MIN_M, RS_DEPTH_MAX_M)
    return [threshold, rs.disparity_transform(True), rs.spatial_filter(),
            rs.temporal_filter(), rs.disparity_transform(False)]
```

- [ ] **Step 4: Remove `align` from the supervision path**

In the globals (`:640-642`) delete `align = None`. In `_rebuild_pipeline`: `global pipeline, depth_unit_mm` (drop `align`); `new_pipeline = openPipeline()`; `pipeline = new_pipeline`. In `_release_pipeline` nothing changes.

- [ ] **Step 5: `handle_client` — parse, refuse, greet**

Replace the handshake block (`:985-1013`, from `color_only = False` through the `print(f"Connection from ...")`) with:

```python
    req = b""
    try:
        conn.settimeout(0.5)
        req = conn.recv(64)
    except (socket.timeout, OSError):
        pass
    finally:
        conn.settimeout(10.0)     # see the comment above about the send timeout
    hs = parse_handshake(req)
    color_only, codec, quality = hs["mode"] == "color", hs["codec"], hs["quality"]
    h264_bitrate, burst = hs["bitrate"], hs["mode"] == "burst"
    telemetry_only, scan_telemetry = hs["mode"] == "telemetry", hs["scan_telemetry"]
    print(f"Connection from {addr} ({hs})", flush=True)

    if hs["depth_requested"] and not hs["v2"]:
        # Big-bang protocol change: a host that did not restart must fail HERE,
        # loudly, not misread the JSON greeting as a frame length and hang.
        print(f"client {addr[0]} did not request V2; this server speaks protocol 2 only "
              f"(got {req!r})", flush=True)
        try:
            conn.sendall(b"ERR protocol 2 required; send MODE FULL V2\n")
        except OSError:
            pass
        conn.close()
        return
```

Then, just before the `while True:` frame loop, add the greeting for the full stream:

```python
    if not color_only:
        conn.sendall(rs_geometry.greeting_line(make_greeting()))
```

In the loop replace `depth, color, timestamp = getFrames(align, depth_filters)` with `getFrames(depth_filters)`.

- [ ] **Step 6: `stream_burst` — greeting after READY, no align**

After `conn.sendall(b'BURST READY\n')` (`:1300`) add `conn.sendall(rs_geometry.greeting_line(make_greeting()))` inside the same `try`. Replace `getFrames(align, depth_filters)` (`:1315`) with `getFrames(depth_filters)`. Update the docstring's protocol block: `(server first sends BURST READY\n then the protocol-2 greeting line)`.

- [ ] **Step 7: `stream_h264` — sizes and the geometry**

Signature `stream_h264(conn, addr, bitrate_kbps, scan_telemetry=False)`; inside use `width, height = COLOR_SIZE` for the gst caps and `frame_bytes`. The call site in `handle_client` becomes `stream_h264(conn, addr, h264_bitrate, scan_telemetry=scan_telemetry)`.

In the feeder replace the block from `depth_profile = depth.profile.as_video_stream_profile()` through the `R_vec = ...` selection (`:1141-1187`) with:

```python
                            depth_profile = depth.profile.as_video_stream_profile()
                            color_profile = color.profile.as_video_stream_profile()
                            intr = depth_profile.intrinsics
                            color_intr = color_profile.intrinsics
                            depth_to_color = depth_profile.get_extrinsics_to(color_profile)
                            R_vec = STATIC_GEOMETRY.R_dc          # row-major, asserted at start
                            t_dc_mm = STATIC_GEOMETRY.t_dc_mm
                            pointcloud = rs.pointcloud()
                            depth_points = pointcloud.calculate(depth)
                            depth_vertices_mm = (
                                np.asanyarray(depth_points.get_vertices())
                                .view(np.float32)
                                .reshape(depth.get_height(), depth.get_width(), 3)
                                * 1000.0)

                            def overlay_project(p):
                                color_point = rs.rs2_transform_point_to_point(
                                    depth_to_color, [float(x) for x in p])
                                return rs.rs2_project_point_to_pixel(color_intr, color_point)

                            def project_color_points(points_color):
                                cp = np.asarray(points_color, float)
                                zc = cp[:, 2]
                                return np.column_stack([
                                    cp[:, 0] * float(color_intr.fx) / zc + float(color_intr.ppx),
                                    cp[:, 1] * float(color_intr.fy) / zc + float(color_intr.ppy)])
```

and keep the three closures that follow (`overlay_project_points`, `overlay_transform_points`, `depth_deproject_points`) as they are — they now use `R_vec`/`t_dc_mm` from the static geometry. Note `overlay_project_points` must be `points @ R_vec.T + t_dc_mm` (row-major R applied to row vectors), so change its body and `overlay_transform_points` to use `R_vec.T`. `scan_plane_telemetry(..., depth_unit_mm, ...)` already takes the unit — it is now 0.1 and nothing else changes.

- [ ] **Step 8: `__main__`**

```python
if __name__ == '__main__':
    print(f"Initiating Jetson-Realsense Wi-Fi Server: depth {DEPTH_SIZE}, colour {COLOR_SIZE}, protocol 2")
    try:
        pipeline = openPipeline()
        depth_unit_mm = (pipeline.get_active_profile().get_device()
                         .first_depth_sensor().get_depth_scale() * 1000.0)
        depth_filters = setup_depth_filters()
        main()
    except Exception as e:
        print(f"Unexpected error: {e}")
```

Also fix the `depth_unit_mm` refresh inside `_rebuild_pipeline` to the same expression (it already is; just drop `align`).

- [ ] **Step 9: Grep for leftovers**

Run: `grep -n "align\|hole_filling\|infrared\|width, height\|set_high_accuracy_preset" server/server_unicast_syncronous.py`
Expected: only comments that describe history (rewrite any that describe current behaviour: the "Fast path: skip align" comment in the colour-only branch becomes "Fast path: skip the depth filters entirely"; the module docstring/handshake comment lists `MODE FULL V2`).

- [ ] **Step 10: Host-side import tests still pass**

Run: `py -3.10 -m pytest tests/test_server_env.py tests/test_scan_telemetry_server.py tests/test_scan_overlay.py -v`
Expected: all PASS (the module must still import with `pyrealsense2`/`turbojpeg` stubbed; `import rs_config`/`rs_geometry` resolve because those tests put `server/` on `sys.path`).

- [ ] **Step 11: Extend the wire test's server mirror**

In `tests/test_camera_wire.py` replace `_server_parse_handshake` and `test_server_parses_quality_handshake` with a direct use of the real parser:

```python
from server.handshake import parse_handshake  # noqa: E402  (sys.path has server/ via tests/test_handshake.py's insert; add the same two inserts at the top of this file)


def test_server_parses_quality_handshake():
    assert parse_handshake(b"MODE COLOR\n")["quality"] is None
    assert parse_handshake(b"MODE COLOR Q60\n")["quality"] == 60
    assert parse_handshake(b"MODE COLOR Q5\n")["quality"] == 10
    assert parse_handshake(b"MODE COLOR Q999\n")["quality"] == 100
    assert parse_handshake(b"")["depth_requested"] and not parse_handshake(b"")["v2"]
```

Run: `py -3.10 -m pytest tests/test_camera_wire.py -v` — expected: PASS (the client-side tests there are updated in Task 8; if `test_full_frame_roundtrip` etc. still pass now, fine — they read raw frames without a greeting, which `_read_frame` still supports).

- [ ] **Step 12: Commit + push (NOT deployed yet — Task 13 deploys)**

```bash
git add server/server_unicast_syncronous.py tests/test_camera_wire.py
git commit -m "feat(camera server): protocol 2 - raw unaligned 0.1 mm depth behind MODE FULL V2 with a JSON greeting; align, hole_filling and IR removed; colour 1080p"
git push origin sensor-layer-v2
```

---

### Task 7: `tasni/core/depth_geometry.py` — the one back-projection

**Files:**
- Create: `tasni/core/depth_geometry.py`
- Create: `tests/geometry_fixtures.py` (shared synthetic geometries for every later test)
- Test: `tests/test_depth_geometry.py`

**Interfaces (every later task uses exactly these names):**

```python
@dataclass(frozen=True)
class CameraGeometry:
    protocol: int
    depth_unit_mm: float
    depth_K: np.ndarray            # 3x3
    depth_size: tuple[int, int]    # (w, h)
    depth_dist: np.ndarray         # (5,), ignored for back-projection (depth is rectified)
    color_size: tuple[int, int]
    color_K_factory: np.ndarray    # 3x3 (the Jetson's colour model; the HOST projects with its calibrated K)
    T_color_depth: np.ndarray      # 4x4, translation mm: p_color = T @ p_depth
    temps: dict
    device: dict
    raw: dict                      # the greeting as received (archives store this)
    legacy: bool                   # True only for CameraGeometry.legacy_aligned

    @classmethod
    def from_greeting(cls, d: dict) -> "CameraGeometry"      # raises ValueError unless protocol == 2
    @classmethod
    def legacy_aligned(cls, K_color, size, *, depth_unit_mm=1.0) -> "CameraGeometry"   # archives only
    def to_dict(self) -> dict

def backproject(depth, geom, *, stride=1, mask=None) -> tuple[np.ndarray, np.ndarray]
    # (pts_mm (N,3) in the COLOUR camera frame, uv_depth (N,2) int [u, v] source pixels); depth==0 dropped
def depth_pose(T_x_color, geom) -> np.ndarray              # T_x_depth = T_x_color @ T_color_depth
def project_to_color(pts_color_mm, K_color, dist_color) -> np.ndarray   # (N,2) float colour pixels (cv2.projectPoints)
def ray_point(u, v, z_mm, K_color, dist_color) -> np.ndarray            # (3,) colour-frame point on the undistorted ray at depth z

class ColorRegistered:
    pts_mm: np.ndarray        # (N,3) colour camera frame
    uv: np.ndarray            # (N,2) float colour pixels
    uv_depth: np.ndarray      # (N,2) int
    color_size: tuple[int,int]
    depth_size: tuple[int,int]
    @classmethod
    def build(cls, depth, geom, K_color, dist_color, *, stride=1) -> "ColorRegistered"
    def in_polygon(self, polygon_uv_norm) -> np.ndarray[bool]   # polygon in NORMALISED 0-1 colour coords
    def in_center_patch(self, frac) -> np.ndarray[bool]          # central frac x frac of the COLOUR image
    def valid_frac_in_center_patch(self, frac) -> float          # points found / points a full depth image would put there
    def near(self, u_px, v_px, radius_px) -> np.ndarray[int]     # cKDTree on uv
    def median_z_near(self, u_px, v_px, radius_px) -> float      # NaN when nothing is near
```

Convention (repeat it in the module docstring): points are in the **colour camera frame**, so `RoboDKIO.camera_pose_T()` (the hand-eye, a colour-camera pose) applies to them directly. `depth_pose` exists only for consumers that work on the depth *image* (Open3D TSDF, per-view support counts).

- [ ] **Step 1: Write the shared fixtures module**

```python
# tests/geometry_fixtures.py
"""Synthetic CameraGeometry objects shared by the depth-geometry, scan and
extrusion tests. ``aligned(K, size)`` is the legacy identity registration (depth
image == colour image, 1 mm units) so existing synthetic renders keep their maths;
``offset(...)`` is a real registration: a different depth K/size, 0.1 mm units and a
non-identity depth->colour extrinsic, so a test can prove the mapping is applied."""
from __future__ import annotations

import numpy as np

from tasni.core.depth_geometry import CameraGeometry
from tasni.core.geometry import Rt_to_T


def aligned(K, size, *, depth_unit_mm: float = 1.0) -> CameraGeometry:
    return CameraGeometry.legacy_aligned(np.asarray(K, float), tuple(size),
                                         depth_unit_mm=depth_unit_mm)


def _rot(axis, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    if axis == "x": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def offset(*, color_K, color_size, depth_K=None, depth_size=(160, 120),
           depth_unit_mm: float = 0.1, rot_deg=(0.4, -0.3, 0.2),
           t_mm=(14.7, -0.2, 0.3)) -> CameraGeometry:
    depth_K = np.array([[130.0, 0, 80.0], [0, 130.0, 60.0], [0, 0, 1.0]]) if depth_K is None \
        else np.asarray(depth_K, float)
    R = _rot("x", rot_deg[0]) @ _rot("y", rot_deg[1]) @ _rot("z", rot_deg[2])
    greeting = {
        "protocol": 2, "aligned": False, "depth_unit_mm": depth_unit_mm,
        "depth": {"width": depth_size[0], "height": depth_size[1], "fx": depth_K[0, 0],
                  "fy": depth_K[1, 1], "ppx": depth_K[0, 2], "ppy": depth_K[1, 2],
                  "model": "brown_conrady", "coeffs": [0, 0, 0, 0, 0]},
        "color": {"width": color_size[0], "height": color_size[1], "fx": float(color_K[0][0]),
                  "fy": float(color_K[1][1]), "ppx": float(color_K[0][2]),
                  "ppy": float(color_K[1][2]), "model": "brown_conrady",
                  "coeffs": [0, 0, 0, 0, 0]},
        "depth_to_color": {"rotation_row_major": R.tolist(), "translation_mm": list(t_mm)},
        "filters": ["threshold", "disparity", "spatial", "temporal", "disparity_inv"],
        "device": {"serial": "synthetic", "fw": "0", "librealsense": "0", "visual_preset": 0,
                   "laser_power": 150},
        "temps": {"asic_c": 40.0, "projector_c": 37.0}, "global_time_enabled": True}
    return CameraGeometry.from_greeting(greeting)


def render_depth_in_depth_camera(points_color_mm, geom: CameraGeometry) -> np.ndarray:
    """Splat colour-frame points into the DEPTH camera's image (uint16 in geom units,
    nearest-wins). The inverse of backproject(), for round-trip tests."""
    from tasni.core.geometry import invert_T, transform_points
    p = transform_points(invert_T(geom.T_color_depth), np.asarray(points_color_mm, float))
    w, h = geom.depth_size
    K = geom.depth_K
    z = p[:, 2]
    ok = z > 1e-6
    u = np.rint(K[0, 0] * p[ok, 0] / z[ok] + K[0, 2]).astype(int)
    v = np.rint(K[1, 1] * p[ok, 1] / z[ok] + K[1, 2]).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    raw = np.rint(z[ok][inside] / geom.depth_unit_mm).astype(np.uint16)
    depth = np.zeros((h, w), np.uint16)
    order = np.argsort(-raw)                     # nearest wins: write far first
    depth[v[inside][order], u[inside][order]] = raw[order]
    return depth
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_depth_geometry.py
"""One back-projection for every consumer: colour-frame points from native depth."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geometry_fixtures as gf  # noqa: E402
from tasni.core import depth_geometry as dg  # noqa: E402
from tasni.core.geometry import transform_points  # noqa: E402

K_C = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
SIZE_C = (320, 240)
DIST0 = np.zeros((5, 1), np.float32)


def test_greeting_parses_and_rejects_protocol_1():
    g = gf.offset(color_K=K_C, color_size=SIZE_C)
    assert g.protocol == 2 and g.depth_unit_mm == 0.1 and g.depth_size == (160, 120)
    assert g.T_color_depth.shape == (4, 4) and g.T_color_depth[0, 3] == 14.7
    assert g.legacy is False
    assert g.to_dict()["depth_to_color"]["translation_mm"][0] == 14.7
    with pytest.raises(ValueError, match="protocol"):
        dg.CameraGeometry.from_greeting({**g.raw, "protocol": 1})


def test_backproject_applies_units_and_extrinsic():
    g = gf.offset(color_K=K_C, color_size=SIZE_C)
    truth = np.array([[0.0, 0.0, 400.0], [30.0, -20.0, 410.0], [-45.0, 25.0, 395.0]])
    depth = gf.render_depth_in_depth_camera(truth, g)
    pts, uv = dg.backproject(depth, g)
    assert len(pts) == 3 and uv.shape == (3, 2)
    # nearest match per truth point, within one depth pixel + one unit of quantisation
    for t in truth:
        err = np.linalg.norm(pts - t, axis=1).min()
        assert err < 3.5, (t, err)


def test_backproject_with_identity_geometry_is_the_old_formula():
    g = gf.aligned(K_C, SIZE_C)
    depth = np.zeros((240, 320), np.uint16); depth[120, 160] = 500; depth[0, 0] = 1000
    pts, uv = dg.backproject(depth, g)
    np.testing.assert_allclose(pts[uv[:, 0] == 160], [[0, 0, 500]])
    np.testing.assert_allclose(pts[uv[:, 0] == 0], [[-160 / 300 * 1000, -120 / 300 * 1000, 1000]])


def test_depth_pose_composes_on_the_right():
    g = gf.offset(color_K=K_C, color_size=SIZE_C)
    T_base_color = np.eye(4); T_base_color[:3, 3] = [100, 200, 300]
    T_base_depth = dg.depth_pose(T_base_color, g)
    np.testing.assert_allclose(T_base_depth, T_base_color @ g.T_color_depth)


def test_ray_point_and_project_round_trip():
    p = np.array([[12.0, -7.0, 420.0]])
    uv = dg.project_to_color(p, K_C, DIST0)
    back = dg.ray_point(uv[0, 0], uv[0, 1], 420.0, K_C, DIST0)
    np.testing.assert_allclose(back, p[0], atol=1e-6)


def test_color_registered_selects_by_colour_region():
    g = gf.offset(color_K=K_C, color_size=SIZE_C)
    # a flat plane at z=400 in the colour frame, sampled on a grid
    xs, ys = np.meshgrid(np.linspace(-150, 150, 61), np.linspace(-110, 110, 45))
    plane = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 400.0)])
    depth = gf.render_depth_in_depth_camera(plane, g)
    reg = dg.ColorRegistered.build(depth, g, K_C, DIST0)
    assert reg.pts_mm.shape[1] == 3 and reg.uv.shape == reg.pts_mm.shape[:1] + (2,)
    # centre patch: every registered point there projects inside the patch
    m = reg.in_center_patch(0.25)
    assert m.sum() > 20
    assert np.all(np.abs(reg.uv[m, 0] - 160) <= 40 + 0.5) and np.all(np.abs(reg.uv[m, 1] - 120) <= 30 + 0.5)
    assert 0.5 < reg.valid_frac_in_center_patch(0.25) <= 1.05
    # polygon: left half of the image, normalised coords
    left = reg.in_polygon([[0, 0], [0.5, 0], [0.5, 1], [0, 1]])
    assert np.all(reg.uv[left, 0] <= 160.5) and left.sum() > 0
    # near/median: the plane's depth in the colour frame is 400 at every pixel
    z = reg.median_z_near(160, 120, 6)
    assert abs(z - 400.0) < 1.0
    assert np.isnan(reg.median_z_near(5, 5, 0.1))


def test_legacy_geometry_flags_itself():
    g = gf.aligned(K_C, SIZE_C)
    assert g.legacy is True and g.depth_unit_mm == 1.0
    assert g.to_dict()["legacy_aligned"] is True
```

- [ ] **Step 3: Run to verify they fail**

Run: `py -3.10 -m pytest tests/test_depth_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: tasni.core.depth_geometry`

- [ ] **Step 4: Write `tasni/core/depth_geometry.py`**

```python
"""Depth-frame geometry: the ONE place raw RealSense depth becomes 3D points.

The Jetson streams NATIVE depth (1280x720, 0.1 mm units, not aligned to colour)
and, once per connection, a greeting with the depth intrinsics and the
depth->colour extrinsic. Everything on the host that reads ``Frame.depth`` comes
through here. Points are returned in the **colour camera frame**: the hand-eye
(``RoboDKIO.camera_pose_T()``) is the colour camera's pose, so it applies to them
directly and every existing "camera frame" convention downstream is unchanged --
only the pixel->point step moved. ``depth_pose`` exists for the two consumers that
work on the depth *image* itself (Open3D TSDF integration, per-view support).

Colour-space selections (the aiming reticle, a survey rectangle, a ChArUco corner)
are answered by :class:`ColorRegistered`: back-project every valid depth pixel,
project into the calibrated colour model, keep what lands inside. There is no
inverse mapping and no resampling -- nothing here invents a depth value.

``CameraGeometry.legacy_aligned`` reproduces the pre-protocol-2 convention
(depth == colour image, 1 mm) for ARCHIVED takes only; live code never builds one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import transform_points

_ZERO_DIST = np.zeros((5, 1), np.float32)


@dataclass(frozen=True)
class CameraGeometry:
    protocol: int
    depth_unit_mm: float
    depth_K: np.ndarray
    depth_size: tuple[int, int]
    depth_dist: np.ndarray
    color_size: tuple[int, int]
    color_K_factory: np.ndarray
    T_color_depth: np.ndarray
    temps: dict
    device: dict
    raw: dict
    legacy: bool = False

    @staticmethod
    def _K(block: dict) -> np.ndarray:
        return np.array([[float(block["fx"]), 0.0, float(block["ppx"])],
                         [0.0, float(block["fy"]), float(block["ppy"])],
                         [0.0, 0.0, 1.0]])

    @classmethod
    def from_greeting(cls, d: dict) -> "CameraGeometry":
        if int(d.get("protocol", 0)) != 2:
            raise ValueError(f"camera greeting protocol {d.get('protocol')!r} is not 2")
        for key in ("depth_unit_mm", "depth", "color", "depth_to_color"):
            if key not in d:
                raise ValueError(f"camera greeting is missing {key!r}")
        dc = d["depth_to_color"]
        R = np.asarray(dc["rotation_row_major"], float).reshape(3, 3)
        t = np.asarray(dc["translation_mm"], float).reshape(3)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        depth, color = d["depth"], d["color"]
        coeffs = np.asarray(depth.get("coeffs") or [0, 0, 0, 0, 0], float)[:5]
        return cls(
            protocol=2, depth_unit_mm=float(d["depth_unit_mm"]),
            depth_K=cls._K(depth), depth_size=(int(depth["width"]), int(depth["height"])),
            depth_dist=coeffs, color_size=(int(color["width"]), int(color["height"])),
            color_K_factory=cls._K(color), T_color_depth=T,
            temps=dict(d.get("temps") or {}), device=dict(d.get("device") or {}),
            raw=dict(d), legacy=False)

    @classmethod
    def legacy_aligned(cls, K_color, size, *, depth_unit_mm: float = 1.0) -> "CameraGeometry":
        """The pre-protocol-2 convention, for ARCHIVED takes without camera_geometry:
        depth was aligned to colour on the Jetson, so depth K == colour K, depth
        size == colour size, and the extrinsic is the identity."""
        K = np.asarray(K_color, float).reshape(3, 3)
        size = (int(size[0]), int(size[1]))
        raw = {"legacy_aligned": True, "protocol": 1, "depth_unit_mm": float(depth_unit_mm),
               "K": K.tolist(), "size": list(size)}
        return cls(protocol=1, depth_unit_mm=float(depth_unit_mm), depth_K=K, depth_size=size,
                   depth_dist=np.zeros(5), color_size=size, color_K_factory=K,
                   T_color_depth=np.eye(4), temps={}, device={}, raw=raw, legacy=True)

    def to_dict(self) -> dict:
        return dict(self.raw)


def backproject(depth, geom: CameraGeometry, *, stride: int = 1, mask=None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Valid depth pixels -> (N,3) mm in the COLOUR camera frame, plus their (N,2)
    [u, v] depth-image pixels. ``stride`` subsamples rows/cols; ``mask`` (depth-image
    shape) restricts the pixels. The depth image is rectified, so its distortion
    coefficients are not applied."""
    d = np.asarray(depth)
    if d.ndim != 2 or d.size == 0:
        return np.zeros((0, 3), float), np.zeros((0, 2), int)
    stride = max(1, int(stride))
    sub = d[::stride, ::stride]
    valid = sub > 0
    if mask is not None:
        valid &= np.asarray(mask, bool)[::stride, ::stride]
    vs, us = np.nonzero(valid)
    if len(vs) == 0:
        return np.zeros((0, 3), float), np.zeros((0, 2), int)
    u = us * stride
    v = vs * stride
    z = sub[vs, us].astype(np.float64) * float(geom.depth_unit_mm)
    K = geom.depth_K
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pts_depth = np.column_stack([x, y, z])
    if geom.legacy:
        pts_color = pts_depth
    else:
        pts_color = transform_points(geom.T_color_depth, pts_depth)
    return pts_color, np.column_stack([u, v]).astype(int)


def depth_pose(T_x_color, geom: CameraGeometry) -> np.ndarray:
    """The depth camera's pose from the colour camera's: T_x_depth = T_x_color @ T_color_depth."""
    return np.asarray(T_x_color, float).reshape(4, 4) @ geom.T_color_depth


def project_to_color(pts_color_mm, K_color, dist_color) -> np.ndarray:
    """Colour-frame points (mm) -> (N,2) float colour pixels through the CALIBRATED model."""
    import cv2
    p = np.asarray(pts_color_mm, np.float64).reshape(-1, 3)
    if len(p) == 0:
        return np.zeros((0, 2), float)
    dist = _ZERO_DIST if dist_color is None else np.asarray(dist_color, np.float64).reshape(-1, 1)
    uv, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), np.asarray(K_color, np.float64), dist)
    return uv.reshape(-1, 2)


def ray_point(u, v, z_mm, K_color, dist_color) -> np.ndarray:
    """The colour-frame point at depth ``z_mm`` on the UNDISTORTED ray through (u, v)."""
    import cv2
    dist = _ZERO_DIST if dist_color is None else np.asarray(dist_color, np.float64).reshape(-1, 1)
    xy = cv2.undistortPoints(np.array([[[float(u), float(v)]]], np.float64),
                             np.asarray(K_color, np.float64), dist).reshape(2)
    return np.array([xy[0] * z_mm, xy[1] * z_mm, float(z_mm)])


class ColorRegistered:
    """One frame's valid depth points with their positions in the colour image."""

    def __init__(self, pts_mm, uv, uv_depth, color_size, depth_size, stride):
        self.pts_mm = np.asarray(pts_mm, float)
        self.uv = np.asarray(uv, float)
        self.uv_depth = np.asarray(uv_depth, int)
        self.color_size = (int(color_size[0]), int(color_size[1]))
        self.depth_size = (int(depth_size[0]), int(depth_size[1]))
        self.stride = int(stride)
        self._tree = None

    @classmethod
    def build(cls, depth, geom: CameraGeometry, K_color, dist_color, *, stride: int = 1
              ) -> "ColorRegistered":
        pts, uv_depth = backproject(depth, geom, stride=stride)
        uv = project_to_color(pts, K_color, dist_color)
        d = np.asarray(depth)
        return cls(pts, uv, uv_depth, geom.color_size, (d.shape[1], d.shape[0]), stride)

    def __len__(self) -> int:
        return len(self.pts_mm)

    def in_polygon(self, polygon_uv_norm) -> np.ndarray:
        import cv2
        w, h = self.color_size
        poly = np.asarray(polygon_uv_norm, float).reshape(-1, 2) * [w, h]
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.rint(poly).astype(np.int32)], 1)
        u = np.clip(np.rint(self.uv[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.rint(self.uv[:, 1]).astype(int), 0, h - 1)
        inside = (self.uv[:, 0] >= 0) & (self.uv[:, 0] < w) & (self.uv[:, 1] >= 0) & (self.uv[:, 1] < h)
        return inside & (mask[v, u] > 0)

    def _center_patch_bounds(self, frac):
        w, h = self.color_size
        pf = float(np.clip(frac, 0.05, 1.0))
        cw, ch = max(2, int(w * pf)), max(2, int(h * pf))
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        return x0, y0, x0 + cw, y0 + ch

    def in_center_patch(self, frac) -> np.ndarray:
        x0, y0, x1, y1 = self._center_patch_bounds(frac)
        return ((self.uv[:, 0] >= x0) & (self.uv[:, 0] < x1)
                & (self.uv[:, 1] >= y0) & (self.uv[:, 1] < y1))

    def valid_frac_in_center_patch(self, frac) -> float:
        """Points found in the colour centre patch / the points a FULLY valid depth
        image would put there. The denominator is the patch's area in colour pixels
        scaled by the depth image's angular pixel density relative to colour --
        exact for a fronto-parallel surface, and a ratio, so units cancel."""
        x0, y0, x1, y1 = self._center_patch_bounds(frac)
        area_color = float((x1 - x0) * (y1 - y0))
        if area_color <= 0:
            return 0.0
        return float(self.in_center_patch(frac).sum()) / (area_color * self._density_ratio())

    def _density_ratio(self) -> float:
        # depth pixels per colour pixel, from the two images' sizes and FOVs is not
        # known here; use the registered points' own footprint: total depth pixels
        # (after stride) over the colour-image area they cover.
        w_c, h_c = self.color_size
        w_d, h_d = self.depth_size
        n_depth = (w_d // self.stride) * (h_d // self.stride)
        if len(self.uv) == 0:
            return n_depth / float(w_c * h_c)
        span_u = max(1.0, float(np.ptp(self.uv[:, 0])))
        span_v = max(1.0, float(np.ptp(self.uv[:, 1])))
        cover = min(float(w_c * h_c), span_u * span_v * n_depth / max(len(self.uv), 1))
        return n_depth / max(cover, 1.0)

    def near(self, u_px, v_px, radius_px) -> np.ndarray:
        if len(self.uv) == 0:
            return np.zeros(0, int)
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.uv)
        return np.asarray(self._tree.query_ball_point([float(u_px), float(v_px)], float(radius_px)), int)

    def median_z_near(self, u_px, v_px, radius_px) -> float:
        idx = self.near(u_px, v_px, radius_px)
        if len(idx) == 0:
            return float("nan")
        return float(np.median(self.pts_mm[idx, 2]))
```

The `_density_ratio` estimate: for the centre-patch valid fraction the gate only needs a threshold-quality number (0.5 today). It uses the registered points' own footprint so a partially valid frame does not inflate the ratio. Keep the docstring honest about that.

- [ ] **Step 5: Run the tests**

Run: `py -3.10 -m pytest tests/test_depth_geometry.py -v`
Expected: 7 PASS. If `test_color_registered_selects_by_colour_region` fails on `valid_frac_in_center_patch` bounds, loosen the upper bound to 1.2 and note why in the test (the footprint estimate is approximate) — do NOT change the gate semantics.

- [ ] **Step 6: Commit + push**

```bash
git add tasni/core/depth_geometry.py tests/geometry_fixtures.py tests/test_depth_geometry.py
git commit -m "feat(core): depth_geometry - one back-projection from native depth into the colour camera frame"
git push origin sensor-layer-v2
```

---

### Task 8: `CameraClient` V2 + `Frame.geometry`; config: delete `depth_scale`, 1080p default, K migration; probe tool

**Files:**
- Modify: `tasni/core/camera.py` (`Frame` `:40-45`; `_read_raw`/`_read_frame` `:167-190`; `grab` `:219-238`; `stream` `:240-277`; `burst` `:283-315`; `_BurstSession.fetch_all` `:337-343`; `_CameraStream.read` `:415-433`)
- Modify: `tasni/core/config.py` (`CameraConfig` `:64-66`; `ScanConfig.depth_scale` `:647` DELETE; `load_config` `:942-955`)
- Create: `tools/probe_depth_quantisation.py`
- Test: `tests/test_camera_wire.py`, `tests/test_scan_config.py`

**Interfaces:**
- Consumes: `CameraGeometry.from_greeting` (Task 7).
- Produces: `Frame.geometry: CameraGeometry | None`; `CameraClient.geometry` (last greeting); `CameraClient._read_line(sock) -> bytes`; `CameraClient._read_greeting(sock) -> CameraGeometry`; `CameraError` text contains `"restart the Tasni backend"` on refusal; `config.migrate_camera_intrinsics(cam: CameraConfig) -> bool`; `config.LEGACY_CONFIG_KEYS`.

- [ ] **Step 1: Write the failing client tests (append to `tests/test_camera_wire.py`)**

```python
import json  # noqa: E402  (add to the imports at the top)
from tasni.core.depth_geometry import CameraGeometry  # noqa: E402

GREETING = {
    "protocol": 2, "aligned": False, "depth_unit_mm": 0.1,
    "depth": {"width": 64, "height": 48, "fx": 60.0, "fy": 60.0, "ppx": 32.0, "ppy": 24.0,
              "model": "brown_conrady", "coeffs": [0, 0, 0, 0, 0]},
    "color": {"width": 64, "height": 48, "fx": 80.0, "fy": 80.0, "ppx": 32.0, "ppy": 24.0,
              "model": "brown_conrady", "coeffs": [0, 0, 0, 0, 0]},
    "depth_to_color": {"rotation_row_major": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                       "translation_mm": [14.7, 0.0, 0.0]},
    "filters": [], "device": {"serial": "S"}, "temps": {"asic_c": 40.0, "projector_c": 37.0},
    "global_time_enabled": True}


def _greeting_bytes() -> bytes:
    return json.dumps(GREETING, separators=(",", ":")).encode() + b"\n"


def _patched_socket(sock):
    import tasni.core.camera as camera_mod
    orig = camera_mod.socket.socket
    camera_mod.socket.socket = lambda *a, **k: sock
    return camera_mod, orig


def test_grab_with_depth_sends_v2_and_reads_the_greeting_first():
    client = CameraClient(CameraConfig())
    color = _make_color()
    depth = (np.arange(64 * 48, dtype=np.uint16).reshape(48, 64) * 3)
    frame_bytes, _ = _server_encode(color, depth, 7.0)
    sock = FakeSocket(_greeting_bytes() + frame_bytes)
    camera_mod, orig = _patched_socket(sock)
    try:
        frame = client.grab(with_depth=True)
    finally:
        camera_mod.socket.socket = orig
    assert bytes(sock.sent) == b"MODE FULL V2\n"
    assert isinstance(frame.geometry, CameraGeometry)
    assert frame.geometry.depth_unit_mm == 0.1 and frame.geometry.T_color_depth[0, 3] == 14.7
    assert client.geometry is frame.geometry
    np.testing.assert_array_equal(frame.depth, depth)
    print("[v2] grab sent MODE FULL V2, parsed the greeting, attached geometry")


def test_refusal_is_a_clear_error_not_a_hang():
    client = CameraClient(CameraConfig())
    sock = FakeSocket(b"ERR protocol 2 required; send MODE FULL V2\n")
    camera_mod, orig = _patched_socket(sock)
    try:
        try:
            client.grab(with_depth=True)
        except CameraError as e:
            assert "restart the Tasni backend" in str(e)
        else:
            raise AssertionError("a refusal must raise CameraError")
    finally:
        camera_mod.socket.socket = orig


def test_color_only_grab_has_no_geometry_and_no_v2_token():
    client = CameraClient(CameraConfig())
    frame_bytes, _ = _server_encode(_make_color(), None, 1.0)
    sock = FakeSocket(frame_bytes)
    camera_mod, orig = _patched_socket(sock)
    try:
        frame = client.grab(color_only=True)
    finally:
        camera_mod.socket.socket = orig
    assert bytes(sock.sent) == b"MODE COLOR\n" and frame.geometry is None


def test_burst_reads_ready_then_greeting():
    client = CameraClient(CameraConfig())
    color = _make_color()
    depth = np.full((48, 64), 4000, np.uint16)
    frame_bytes, _ = _server_encode(color, depth, 2.0)
    payload = (b"BURST READY\n" + _greeting_bytes()
               + struct.pack("<I", 0) + struct.pack("<I", 5) + b"thumb"     # CAP reply
               + struct.pack("<I", 1) + frame_bytes                          # GET reply
               + struct.pack("<I", 0))                                       # CLEAR ack
    sock = FakeSocket(payload, chunk=64)
    camera_mod, orig = _patched_socket(sock)
    try:
        with client.burst() as bs:
            assert bs.capture() == b"thumb"
            frames = bs.fetch_all()
            bs.clear()
    finally:
        camera_mod.socket.socket = orig
    assert bytes(sock.sent).startswith(b"MODE BURST V2\n")
    assert len(frames) == 1 and frames[0].geometry.depth_unit_mm == 0.1
    np.testing.assert_array_equal(frames[0].depth, depth)
```

Also update `test_grab_sends_mode_color_handshake` if it asserts anything about depth frames without a greeting — it uses `color_only=True`, so it is unchanged.

- [ ] **Step 2: Run to verify they fail**

Run: `py -3.10 -m pytest tests/test_camera_wire.py -v -k "v2 or refusal or no_geometry or burst_reads"`
Expected: FAIL (`MODE FULL V2` not sent / `Frame` has no `geometry`)

- [ ] **Step 3: Implement in `tasni/core/camera.py`**

`Frame`:
```python
@dataclass
class Frame:
    color: np.ndarray              # HxWx3 BGR
    depth: np.ndarray | None       # HxW NATIVE depth (uint16, geometry.depth_unit_mm per step) or None
    timestamp: float
    telemetry: dict | None = None
    geometry: "CameraGeometry | None" = None   # the connection's greeting; None on colour-only paths
```
with `from .depth_geometry import CameraGeometry` at the top. Module docstring: replace the wire description with the protocol-2 one (header unchanged; `MODE FULL V2` -> greeting line -> frames; depth is native and unaligned).

Constants + helpers in `CameraClient`:
```python
_HELLO_FULL = b"MODE FULL V2\n"
_HELLO_BURST = b"MODE BURST V2\n"

    @staticmethod
    def _read_line(sock: socket.socket, maxlen: int = 65536) -> bytes:
        buf = bytearray()
        while len(buf) < maxlen:
            ch = sock.recv(1)
            if not ch:
                raise CameraError("connection closed by camera server before the greeting")
            buf.extend(ch)
            if ch == b"\n":
                return bytes(buf)
        raise CameraError("camera greeting exceeded 64 KB")

    def _read_greeting(self, sock: socket.socket) -> CameraGeometry:
        """Protocol 2: one JSON line before any frame. A refusal line means the
        server changed protocol under a host that never restarted."""
        try:
            line = self._read_line(sock)
        except socket.timeout as e:
            raise CameraError("camera timeout waiting for the protocol-2 greeting") from e
        if line.startswith(b"ERR"):
            raise CameraError(
                f"camera server refused the depth stream: {line.decode(errors='replace').strip()} "
                "- restart the Tasni backend (it is speaking an older protocol)")
        try:
            geom = CameraGeometry.from_greeting(json.loads(line.decode("utf-8")))
        except (ValueError, json.JSONDecodeError) as e:
            raise CameraError(f"invalid camera greeting: {e}") from e
        self.geometry = geom
        return geom
```
(`self.geometry: CameraGeometry | None = None` in `__init__`.)

`_read_frame(self, sock, with_depth, geometry=None)` returns `Frame(..., geometry=geometry)`.

`grab`:
```python
        with self._connect(timeout)[0] as s:
            geometry = None
            if color_only:
                self._request_color_only(s, quality)
            else:
                s.sendall(_HELLO_FULL)
                geometry = self._read_greeting(s)
            try:
                return self._read_frame(s, with_depth, geometry=geometry)
            finally: ...
```

`stream`: after the `if color_only or h264:` branch add `else: s.sendall(_HELLO_FULL); geometry = self._read_greeting(s)` (initialise `geometry = None` before), and construct `_CameraStream(self, s, telemetry_reader=..., geometry=geometry)`; `_CameraStream.__init__` stores it and `read` passes `geometry=self._geometry` into the `Frame`.

`burst`: send `_HELLO_BURST`, read `BURST READY\n` as today, then `geometry = self._read_greeting(s)`; `_BurstSession(self, s, geometry)`; `fetch_all` passes `geometry=self._geometry` to `_read_frame`. Docstring: a server that predates V2 would answer with the refusal line, which the READY check turns into `CameraError`.

- [ ] **Step 4: Run the client tests**

Run: `py -3.10 -m pytest tests/test_camera_wire.py -v`
Expected: all PASS

- [ ] **Step 5: Config — failing migration tests (append to `tests/test_scan_config.py`)**

```python
def test_depth_scale_is_gone_and_a_stale_override_is_dropped_with_a_warning(tmp_path, capsys):
    from tasni.core.config import ScanConfig, load_config
    assert "depth_scale" not in ScanConfig.model_fields
    p = tmp_path / "tasni.config.json"
    p.write_text('{"scan": {"depth_scale": 1000.0, "voxel_size_m": 0.002}}', encoding="utf-8")
    cfg = load_config(p)
    assert cfg.scan.voxel_size_m == 0.002
    assert "scan.depth_scale" in capsys.readouterr().out


def test_calibrated_720p_intrinsics_migrate_to_1080p_by_exact_scale():
    from tasni.core.config import CameraConfig, migrate_camera_intrinsics, _DEFAULT_INTRINSICS
    cam = CameraConfig()
    assert cam.resolution == "1920x1080"
    cal = [[889.8742, 0.0, 648.9804], [0.0, 890.8099, 362.0046], [0.0, 0.0, 1.0]]
    cam.intrinsics = {**cam.intrinsics, "1280x720": cal}
    assert migrate_camera_intrinsics(cam) is True
    K = cam.K
    assert abs(K[0, 0] - 889.8742 * 1.5) < 1e-6 and abs(K[1, 2] - 362.0046 * 1.5) < 1e-6
    assert migrate_camera_intrinsics(cam) is False                      # idempotent
    fresh = CameraConfig()
    assert migrate_camera_intrinsics(fresh) is False                    # factory 720p: nothing to carry
    assert fresh.intrinsics["1920x1080"] == _DEFAULT_INTRINSICS["1920x1080"]
```

- [ ] **Step 6: Run to verify they fail**

Run: `py -3.10 -m pytest tests/test_scan_config.py -v -k "depth_scale or migrate"`
Expected: FAIL

- [ ] **Step 7: Implement in `tasni/core/config.py`**

- `CameraConfig.resolution: str = "1920x1080"` with the comment: `# The server streams colour at 1920x1080 (server_unicast_syncronous.py COLOR_SIZE); K is picked by this key. Depth has its own model, carried in the per-connection greeting (core/depth_geometry.py).`
- Delete `ScanConfig.depth_scale` (`:647`) and its comment.
- `ExtrusionConfig.voxel_size_m` default `0.001` (spec 4.4; Task 9 tests it).
- Add:
```python
LEGACY_CONFIG_KEYS = {("scan", "depth_scale")}   # removed with protocol 2; the greeting carries the unit


def migrate_camera_intrinsics(cam: CameraConfig) -> bool:
    """Carry a CALIBRATED 720p K into the 1080p slot when that slot is still factory.
    The D435i's 1080p RGB mode is an exact 1.5x scale of 720p (factory table:
    1362.15/908.10 = 1.5000), and distortion coefficients are normalised, so this
    is a pure rescale -- the hand-eye is a physical transform and stays valid."""
    factory = _DEFAULT_INTRINSICS
    cur720 = cam.intrinsics.get("1280x720")
    cur1080 = cam.intrinsics.get("1920x1080")
    if cur720 is None or cur1080 is None:
        return False
    if np.allclose(cur720, factory["1280x720"]) or not np.allclose(cur1080, factory["1920x1080"]):
        return False
    K = np.asarray(cur720, float).copy()
    K[0, 0] *= 1.5; K[1, 1] *= 1.5; K[0, 2] *= 1.5; K[1, 2] *= 1.5
    cam.intrinsics = {**cam.intrinsics, "1920x1080": K.tolist()}
    return True
```
- In `load_config`, before `_merge`, strip legacy keys:
```python
        for section, key in LEGACY_CONFIG_KEYS:
            if isinstance(data.get(section), dict) and key in data[section]:
                data[section].pop(key)
                print(f"config: dropped legacy key {section}.{key} (removed with camera protocol 2)")
        _merge(cfg, data)
        if migrate_camera_intrinsics(cfg.camera):
            print("config: migrated calibrated 1280x720 intrinsics to 1920x1080 (x1.5)")
            save_overrides({"camera": {"intrinsics": cfg.camera.intrinsics}})
```
(`save_overrides` writes the repo-root file; when `path` was given explicitly and differs from `config_file_path()`, skip the save — add `if Path(path).resolve() == config_file_path().resolve()` around it.)

- [ ] **Step 8: Fix every reader of `scfg.depth_scale` to import-clean**

Run: `grep -rn "depth_scale" --include=*.py tasni/ tools/ tests/`
Expected after this task: hits only in `tasni/modules/scan/*`, `tasni/modules/extrusion/*`, `tools/characterize_distance.py` and their tests — those are Tasks 9-11. The app must still IMPORT: `py -3.10 -c "import tasni.core.config, tasni.core.camera"` prints nothing.

- [ ] **Step 9: The acceptance probe tool**

```python
# tools/probe_depth_quantisation.py
"""Audit R2 acceptance: depth word granularity in a centre patch, from the workstation.

    py -3.10 tools/probe_depth_quantisation.py [--patch 120]

Before protocol 2 (2026-08-29, arm parked): uint16 (720,1280) valid 0.999, unique 25,
min step 1 (mm). After: unit 0.1 mm, expect >= 200 unique values in the same patch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core.camera import CameraClient  # noqa: E402
from tasni.core.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=120)
    args = ap.parse_args()
    frame = CameraClient(load_config().camera).grab(with_depth=True, timeout=20)
    d = frame.depth
    g = frame.geometry
    v = d[d > 0]
    print(f"{d.dtype} {d.shape} unit {g.depth_unit_mm} mm valid {v.size / d.size:.3f} "
          f"range {v.min() * g.depth_unit_mm:.1f}..{v.max() * g.depth_unit_mm:.1f} mm "
          f"temps {g.temps} preset {g.device.get('visual_preset')}")
    cy, cx = d.shape[0] // 2, d.shape[1] // 2
    r = args.patch // 2
    p = d[cy - r:cy + r, cx - r:cx + r]
    pv = np.unique(p[p > 0])
    print(f"centre {args.patch}x{args.patch}: unique {pv.size}, min step "
          f"{(np.diff(pv).min() if pv.size > 1 else 0) * g.depth_unit_mm:.2f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 10: Run the touched tests**

Run: `py -3.10 -m pytest tests/test_camera_wire.py tests/test_scan_config.py tests/test_camera_failover.py tests/test_camera_lease.py -v`
Expected: all PASS

- [ ] **Step 11: Commit + push**

```bash
git add tasni/core/camera.py tasni/core/config.py tools/probe_depth_quantisation.py tests/test_camera_wire.py tests/test_scan_config.py
git commit -m "feat(core): camera client speaks protocol 2 (V2 hello, greeting, Frame.geometry); config drops depth_scale, defaults 1080p, migrates K"
git push origin sensor-layer-v2
```

---

### Task 9: Extrusion — geometry-driven back-projection, archive provenance, figures fallback, 1 mm voxel

**Files:**
- Modify: `tasni/modules/extrusion/processing.py` (`depth_to_work_points` `:39-49`; `process_observation` signature `:577-583` + call `:611-612`; `characterize_ring` signature `:791-794` + call `:806` + inner `process_observation(...)` `:837`)
- Modify: `tasni/modules/extrusion/measure.py` (`_capture_at_pose` returns the frame — `:284-320`; provenance base dict `:700-708`; `process_observation(...)` call `:712-714`; `characterize_ring(...)` call `:833-840`)
- Modify: `tasni/modules/extrusion/service.py` (provenance `:1020-1026`; `process_observation` `:1034-1037`; reprocess `:1131-1176`)
- Modify: `tasni/modules/extrusion/figures.py` (`TakeData` `:96-110`; `load_take` `:203-220`; `_scene_points` `:264-268`; `_compute_stages` `:488-493`; `TakeData.label` `:141-143`)
- Modify: `tests/extrusion_synthetic.py` (docstring only: depth is what `backproject` expects with an ALIGNED legacy geometry)
- Test: `tests/test_extrusion_processing.py`, `tests/test_extrusion_measure.py`, `tests/test_extrusion_figures.py`, `tests/test_extrusion_job.py`

**Interfaces:**
- Consumes: `CameraGeometry`, `backproject` (Task 7); `Frame.geometry` (Task 8).
- Produces: `depth_to_work_points(depth, geometry: CameraGeometry, T_work_camera) -> tuple[np.ndarray, int]`; `process_observation(*, color, depth, geometry, T_work_camera, K, dist, plan, layer, config, floor_profile=None, stages=None, assemble_arcs=False)`; `characterize_ring(*, color, depth, geometry, T_work_camera, K, dist, search_center_mm, work_frame, config, inspection_tool=..., print_tool=...)`; `chroma_gate_mask(color, registered: ColorRegistered, config, counts=None) -> tuple[np.ndarray, bool]` (per-POINT keep mask + whether the gate applied; replaces the image-space `chroma_gated_depth`); manifest `provenance.camera_geometry` (the greeting dict); `figures.TakeData.geometry: CameraGeometry | None`; `figures.geometry_for_take(manifest: dict, K: np.ndarray | None, depth: np.ndarray | None) -> CameraGeometry | None`.

`K`/`dist` STAY in the processing signatures, but as the CALIBRATED COLOUR model, used for one thing only: projecting registered depth points into the colour image for the chroma gate (`041ad1b`, 2026-08-29: bead vs board is decided by saturation, not height). `T_work_camera` stays the COLOUR camera's pose (`camera_pose_T()`), which `backproject`'s colour-frame points need.

**Why the gate must move (read this before Step 3b).** `chroma_gated_depth(color, depth, config)` blanks depth pixels where the colour pixel at the SAME (v, u) is achromatic -- an aligned-depth assumption. With protocol 2 the colour frame is 1920x1080 and depth 1280x720, so its `image.shape[:2] != depth.shape[:2]` check trips and the gate ABSTAINS silently, the deposit floor falls back to 2.5 mm, and every take goes back to the 0/4-valid + branch-guard-crash state that commit fixed. The fix is not to mask depth pixels but to mask registered POINTS: project each depth point into the calibrated colour model (`ColorRegistered.uv`) and read the saturation mask there. With the legacy aligned geometry the projection is the identity, so the archived-frame tests (`ring1_take04_branchguard_20260829.npz`, with and without the gate) keep their exact meaning.

- [ ] **Step 1: Failing processing tests (replace `test_depth_backprojection_uses_explicit_work_transform` in `tests/test_extrusion_processing.py`)**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry_fixtures as gf  # noqa: E402


def test_depth_backprojection_uses_explicit_work_transform():
    depth = np.array([[0, 1000], [1000, 0]], dtype=np.uint16)
    K = np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]], dtype=float)
    T = np.eye(4); T[:3, 3] = [10, 20, 30]
    points, raw = depth_to_work_points(depth, gf.aligned(K, (2, 2)), T)
    assert raw == 2
    np.testing.assert_allclose(points, [[11, 20, 1030], [10, 21, 1030]])


def test_depth_backprojection_honours_units_and_the_depth_to_colour_extrinsic():
    K_c = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
    geom = gf.offset(color_K=K_c, color_size=(320, 240))          # 0.1 mm, non-identity extrinsic
    truth_color = np.array([[0.0, 0.0, 400.0], [40.0, -25.0, 405.0]])
    depth = gf.render_depth_in_depth_camera(truth_color, geom)
    T = np.eye(4); T[:3, 3] = [100, 200, 0]
    points, raw = depth_to_work_points(depth, geom, T)
    assert raw == 2
    for t in truth_color + [100, 200, 0]:
        assert np.linalg.norm(points - t, axis=1).min() < 3.5
```

- [ ] **Step 2: Run to verify they fail**

Run: `py -3.10 -m pytest tests/test_extrusion_processing.py -v -k backprojection`
Expected: FAIL (signature mismatch: `K` positional is not a `CameraGeometry`)

- [ ] **Step 3: Implement `processing.py`**

```python
from ...core.depth_geometry import CameraGeometry, backproject


def depth_to_work_points(depth: np.ndarray, geometry: CameraGeometry,
                         T_work_camera: np.ndarray) -> tuple[np.ndarray, int]:
    """Native depth -> work-frame mm. ``T_work_camera`` is the COLOUR camera's pose
    (the hand-eye); ``backproject`` already returns colour-frame points."""
    camera, _uv = backproject(depth, geometry)
    return transform_points(T_work_camera, camera), int(len(camera))
```

`process_observation(*, color, depth, geometry: CameraGeometry, T_work_camera, plan, layer, config, ...)` — the call becomes `depth_to_work_points(depth, geometry, T_work_camera)`. `characterize_ring(*, color, depth, geometry: CameraGeometry, T_work_camera, search_center_mm, ...)` — same, and its inner `process_observation(color=color, depth=depth, geometry=geometry, T_work_camera=T_work_camera, ...)`. Remove every `depth_scale` mention from this file; `K` is kept as the colour model (see Step 3b).

- [ ] **Step 3b: Port the chroma gate to registered points**

Failing tests first (append to `tests/test_extrusion_processing.py`):

```python
def test_chroma_gate_masks_points_by_their_colour_projection_not_depth_pixel():
    """Depth is not aligned to colour any more: a bead point must be kept because the
    colour pixel it PROJECTS TO is saturated, even though the depth pixel with the
    same (v, u) index looks at something else."""
    from tasni.core.config import ExtrusionConfig
    from tasni.core.depth_geometry import ColorRegistered
    from tasni.modules.extrusion.processing import chroma_gate_mask
    K_c = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
    geom = gf.offset(color_K=K_c, color_size=(320, 240), depth_size=(160, 120))
    # two colour-frame points at z=400: one lands on the saturated blob, one on grey
    pts = np.array([[0.0, 0.0, 400.0], [-80.0, 0.0, 400.0]])
    depth = gf.render_depth_in_depth_camera(pts, geom)
    color = np.full((240, 320, 3), 128, np.uint8)               # achromatic everywhere ...
    color[100:140, 140:180] = (20, 40, 220)                    # ... except a chromatic blob at the reticle
    reg = ColorRegistered.build(depth, geom, K_c, None)
    cfg = ExtrusionConfig(deposit_min_chroma_fraction=0.001)
    keep, applied = chroma_gate_mask(color, reg, cfg)
    assert applied and keep.shape == (len(reg),)
    on_blob = np.linalg.norm(reg.pts_mm - pts[0], axis=1) < 3.5
    assert keep[on_blob].all() and not keep[~on_blob].any()


def test_chroma_gate_abstains_on_an_achromatic_frame_or_size_mismatch():
    from tasni.core.config import ExtrusionConfig
    from tasni.core.depth_geometry import ColorRegistered
    from tasni.modules.extrusion.processing import chroma_gate_mask, deposit_floor_mm
    K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
    depth = np.full((240, 320), 4000, np.uint16)
    reg = ColorRegistered.build(depth, gf.aligned(K, (320, 240), depth_unit_mm=0.1), K, None)
    cfg = ExtrusionConfig()
    keep, applied = chroma_gate_mask(np.full((240, 320, 3), 128, np.uint8), reg, cfg)
    assert not applied and keep.all()
    assert deposit_floor_mm(cfg, applied) == 2.5
    keep, applied = chroma_gate_mask(np.zeros((100, 100, 3), np.uint8), reg, cfg)   # wrong size
    assert not applied and keep.all()
```

Run `py -3.10 -m pytest tests/test_extrusion_processing.py -k chroma_gate -v` -> FAIL (`ImportError: chroma_gate_mask`). Then implement, REPLACING `chroma_gated_depth` (delete it -- its image-space contract is the aligned assumption):

```python
def chroma_gate_mask(color, registered: "ColorRegistered", config,
                     counts: dict | None = None) -> tuple[np.ndarray, bool]:
    """Per REGISTERED POINT: True where the colour frame says "bead", not "board".

    Depth is native and not aligned to colour (camera protocol 2), so the gate
    cannot blank depth pixels in place; each depth point is projected into the
    calibrated colour model (``registered.uv``) and the saturation mask is read
    there. Everything else is the 2026-08-29 gate unchanged: saturation > threshold
    (bead ~20:1 over the printed board), a chroma-fraction abstention for RGB
    dropouts and depth-only fixtures, and a closing so speckle inside the bead
    does not punch holes. Points that project OUTSIDE the colour image (the depth
    field is wider than colour) have no colour evidence and are dropped while the
    gate applies -- they are far outside any ring ROI anyway. Abstains as
    ``(all True, False)``, which ``deposit_floor_mm`` turns into the 2.5 mm floor.
    """
    def note(key: str, value: int) -> None:
        if counts is not None:
            counts[key] = value

    n = len(registered)
    threshold = int(getattr(config, "deposit_min_saturation", 0) or 0)
    if threshold <= 0 or color is None:
        note("chroma_gate_applied", 0)
        return np.ones(n, bool), False
    image = np.asarray(color)
    w, h = registered.color_size
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != (h, w):
        note("chroma_gate_applied", 0)
        return np.ones(n, bool), False
    saturation = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV)[:, :, 1]
    keep = (saturation > threshold).astype(np.uint8)
    note("chroma_gate_kept_pixels", int(keep.sum()))
    if float(keep.mean()) < float(getattr(config, "deposit_min_chroma_fraction", 0.0)):
        note("chroma_gate_applied", 0)
        return np.ones(n, bool), False
    k = max(3, int(round(5 * w / 1280.0)))          # the 5x5 close was tuned at 720p
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    u = np.rint(registered.uv[:, 0]).astype(int)
    v = np.rint(registered.uv[:, 1]).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    mask = np.zeros(n, bool)
    mask[inside] = keep[v[inside], u[inside]] > 0
    note("chroma_gate_applied", 1)
    note("chroma_gate_outside_colour", int((~inside).sum()))
    return mask, True
```

In `process_observation` (and identically in `characterize_ring`) the first stage becomes:

```python
    reg = ColorRegistered.build(depth, geometry, K, dist)
    keep, chroma_gated = chroma_gate_mask(color, reg, config, counts)
    points = transform_points(T_work_camera, reg.pts_mm[keep])
    counts["raw_depth_pixels"] = int(keep.sum())
```

(`depth_to_work_points` stays as the ungated helper for `figures._scene_points`.) Rewrite the two direct `chroma_gated_depth(...)` tests in `tests/test_extrusion_measure.py` (`test_chroma_gate_keeps_the_chromatic_bead_and_blanks_the_achromatic_board`, `test_chroma_gate_abstains_on_an_achromatic_frame_and_restores_the_floor`) against `chroma_gate_mask(color, ColorRegistered.build(depth, gf.aligned(K, (w, h)), K, None), config)` with the same expectations stated on points instead of pixels. The paired take-04 tests (`..._clears_the_board_lobe_...` / `..._still_fails_with_the_chroma_gate_disabled`) need only the new kwargs: `geometry=gf.aligned(f["K"], (1280, 720)), K=f["K"], dist=None` -- with the identity registration they must keep passing unchanged, which is the proof the port preserved the gate.

- [ ] **Step 4: Callers in `measure.py` and `service.py`**

`measure.py:712` -> `process_observation(color=frame.color, depth=frame.depth, geometry=frame.geometry, T_work_camera=T_work_camera, K=services.config.camera.K, dist=services.config.camera.dist, plan=..., layer=..., config=ecfg, floor_profile=floor)`. `measure.py:835` -> `characterize_ring(color=frame.color, depth=frame.depth, geometry=frame.geometry, T_work_camera=captured["T_work_camera"], K=services.config.camera.K, dist=services.config.camera.dist, ...)`. `service.py:1034` same shape. Guard at both capture sites right after the grab: `if frame.geometry is None: raise RuntimeError("depth frame arrived without a protocol-2 greeting")`.

Provenance: in the per-take manifest base dict (`measure.py:700-708` and `service.py:1020-1026`) add `"camera_geometry": frame.geometry.to_dict()`. `_provenance(services)` (`measure.py:76`) is unchanged (the calibrated colour K is still a fact worth recording).

Reprocess (`service.py:1131-1176`):
```python
    from ...core.depth_geometry import CameraGeometry
    geom_dict = provenance.get("camera_geometry")
    depth = np.load(depth_path, allow_pickle=False)
    if geom_dict is not None and not geom_dict.get("legacy_aligned"):
        geometry = CameraGeometry.from_greeting(geom_dict)
    else:
        # Pre-protocol-2 archive: depth was aligned to colour, 1 mm units.
        geometry = CameraGeometry.legacy_aligned(np.asarray(intrinsics["K"], float),
                                                 (depth.shape[1], depth.shape[0]))
    processed = process_observation(color=color, depth=depth, geometry=geometry,
                                    T_work_camera=np.asarray(transform, dtype=float),
                                    K=np.asarray(intrinsics["K"], dtype=float),
                                    dist=intrinsics.get("dist_coeffs"), ...)
```

- [ ] **Step 5: Figures — the one read-side fallback**

In `figures.py`:
```python
from ...core.depth_geometry import CameraGeometry


def geometry_for_take(manifest: dict, K, depth) -> "CameraGeometry | None":
    """The take's depth geometry: protocol-2 greeting from provenance, else the
    legacy aligned model (this is the ONLY place the old convention survives)."""
    raw = (manifest.get("provenance") or {}).get("camera_geometry")
    if raw and not raw.get("legacy_aligned"):
        return CameraGeometry.from_greeting(raw)
    if K is None or depth is None:
        return None
    d = np.asarray(depth)
    return CameraGeometry.legacy_aligned(np.asarray(K, float), (d.shape[1], d.shape[0]))
```
`TakeData` gains `geometry: CameraGeometry | None`; `load_take` sets `geometry=geometry_for_take(manifest, K, depth)` (compute `depth` and `K` first). `_scene_points`: `if take.depth is not None and take.geometry is not None and take.T_work_camera is not None: points, _ = depth_to_work_points(take.depth, take.geometry, take.T_work_camera)`. `_compute_stages` (`:488-493`): `process_observation(color=image, depth=take.depth, geometry=take.geometry, T_work_camera=..., K=take.K, dist=None, ...)` with the same `geometry is not None` guard as `take.K` had (a zero image makes the gate abstain, as today). `TakeData.label`: append `" · depth: legacy aligned 1 mm"` when `self.geometry is not None and self.geometry.legacy`.

- [ ] **Step 6: Tests for the archive paths (append to `tests/test_extrusion_figures.py`)**

```python
def test_a_take_without_camera_geometry_renders_as_legacy_aligned(tmp_path):
    from tasni.modules.extrusion import figures
    layer_dir = write_take(tmp_path)                       # write_take records no camera_geometry
    take = figures.load_take(layer_dir)
    assert take.geometry is not None and take.geometry.legacy
    assert take.geometry.depth_size == (1280, 720)
    assert "legacy aligned" in take.label
    assert figures._scene_points(take) is not None


def test_a_protocol_2_take_uses_its_recorded_geometry(tmp_path):
    import geometry_fixtures as gf
    from tasni.modules.extrusion import figures
    geom = gf.offset(color_K=syn.K_720P, color_size=syn.SIZE_720P,
                     depth_K=syn.K_720P * [[1, 1, 1], [1, 1, 1], [1, 1, 1]], depth_size=syn.SIZE_720P)
    layer_dir = write_take(tmp_path, trial_id="t-v2")
    manifest_file = layer_dir / "manifest.json"
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    payload["provenance"]["camera_geometry"] = geom.to_dict()
    manifest_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    take = figures.load_take(layer_dir)
    assert take.geometry.legacy is False and take.geometry.depth_unit_mm == 0.1
    assert "legacy" not in take.label
```
(`write_take` takes `trial_id` — check its signature at `:52`; add the kwarg if it lacks one.) Add `sys.path.insert(0, str(Path(__file__).resolve().parent))` + imports at the top of the file as the other test files do.

Ring fixtures (`tests/test_extrusion_measure.py` `:334-345`, `:409-420`, and `:290`, `:318`, `:386`): ADD `geometry=gf.aligned(<the same K>, (1280, 720)), dist=None` (keeping `K=`) for the real-capture `.npz` fixtures (`fixture["K"]`; `ring1_take04_branchguard_20260829.npz` also carries `color_jpeg` -- decode with `cv2.imdecode`), and `geometry=gf.aligned(syn.K_720P, syn.SIZE_720P), dist=None` for synthetic renders. Fake cameras in `tests/test_extrusion_job.py:150-163` and `tests/test_extrusion_measure.py:1990-1998` must return `Frame(..., geometry=gf.aligned(syn.K_720P, (16, 16)))` — the 16x16 frames there are only ever median-checked, so any aligned geometry of that size works.

- [ ] **Step 7: Voxel default test (append to `tests/test_extrusion_processing.py`)**

```python
def test_default_voxel_is_1_mm_so_0_1_mm_depth_words_reach_the_ring_numbers():
    from tasni.core.config import ExtrusionConfig
    assert ExtrusionConfig().voxel_size_m == 0.001
```

- [ ] **Step 8: Run the extrusion tests**

Run: `py -3.10 -m pytest tests/test_extrusion_processing.py tests/test_extrusion_measure.py tests/test_extrusion_figures.py tests/test_extrusion_job.py tests/test_extrusion_standoff.py -q`
Expected: all PASS. `grep -rn "depth_scale\|K=services.config.camera.K" tasni/modules/extrusion/` returns nothing.

- [ ] **Step 9: Commit + push**

```bash
git add tasni/modules/extrusion tasni/core/config.py tests/test_extrusion_processing.py tests/test_extrusion_measure.py tests/test_extrusion_figures.py tests/test_extrusion_job.py tests/extrusion_synthetic.py
git commit -m "feat(extrusion): back-project native depth through the greeting geometry; archive it; figures keep one legacy fallback; 1 mm voxel"
git push origin sensor-layer-v2
```

---

### Task 10: Scan — gate, survey, corner evidence, depth-only TSDF, support counts, saved views

**Files:**
- Modify: `tasni/modules/scan/depth_gate.py` (`evaluate_depth_gate` `:76-130`)
- Modify: `tasni/modules/scan/survey.py` (`survey_surface` `:138-200` + the overlay projection at `:230-275`)
- Modify: `tasni/modules/scan/corner_evidence.py` (`_median_depth` `:59-68`, `extract_corner_evidence` `:162-300`)
- Modify: `tasni/modules/scan/reconstruct.py` (`ScanView` `:26-32`; `fuse_views` `:60-104`; `look_point_from_views` `:107-131`; `_view_support_counts` `:252-300`; `clean_measured_surface_mesh` signature `:306-316` + call `:365`)
- Modify: `tasni/modules/scan/service.py` (`_authoritative_acquisition` `:406-432`; `_plane_rms_mm` `:209-255`; `_backproject_depth` `:948-958`; `_deproject_plane_points_mm` `:961-1040`; five-position `:1299-1440`; `_save_views` `:1481-1500`; `_reference_locate` `:1517-1525`; fusion `:3104-3186`; `_capture`/`_capture_burst` `:3294`, `:3365`)
- Test: `tests/test_scan_depth_gate.py`, `tests/test_scan_survey.py`, `tests/test_corner_evidence.py`, `tests/test_scan_reconstruct.py`, `tests/test_scan_job.py`, `tests/test_five_position.py`

**Interfaces:**
- Consumes: Task 7 (`backproject`, `ColorRegistered`, `depth_pose`), Task 8 (`Frame.geometry`).
- Produces: `evaluate_depth_gate(depth, geometry, K_color, dist_color, th, *, max_samples=3000)`; `survey_surface(depth, geometry, K_color, dist_color, thresholds)`; `extract_corner_evidence(registered: ColorRegistered, K, polygon_uv, T_base_cam, *, ..., window_px=3, ...)`; `ScanView(color, depth, pose_T, geometry)`; `fuse_views(views, *, voxel_size_m, sdf_trunc_m, depth_min_m, depth_max_m)`; `clean_measured_surface_mesh(mesh, views, wp, K, width, height, *, ...)` loses `depth_scale` (K/width/height stay for the colour-model callers already there; support counts use the view geometry); `service._backproject_depth(depth, geometry) -> np.ndarray`; `service._registered(frame, cfg, stride=1) -> ColorRegistered`.

Semantics kept: every "camera frame" quantity (gate tilt normal, `measurement.normal_cam`, `centroid_cam_mm`, `T_base_cam` = `camera_pose_T()`) stays the COLOUR camera frame. Overlays stay normalised colour coordinates (the HUD already draws them that way).

- [ ] **Step 1: Failing gate/survey tests**

In `tests/test_scan_depth_gate.py` and `tests/test_scan_survey.py` the synthetic `_render` produces an aligned frame in `K`'s model; keep every existing assertion and change the calls to `evaluate_depth_gate(depth, gf.aligned(K, (W, H)), K, None, th)` / `survey_surface(depth, gf.aligned(K, (W, H)), K, None, th)`. Add ONE registration test to `tests/test_scan_depth_gate.py`:

```python
def test_gate_reads_the_reticle_through_the_colour_model_not_the_depth_centre():
    """With a real (offset) registration the depth image centre is NOT the colour
    reticle. Put the plane only under the colour reticle and prove the gate sees it."""
    import geometry_fixtures as gf
    geom = gf.offset(color_K=K, color_size=(W, H), depth_size=(160, 120))
    xs, ys = np.meshgrid(np.linspace(-60, 60, 41), np.linspace(-45, 45, 31))
    plane = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 500.0)])   # colour frame
    depth = gf.render_depth_in_depth_camera(plane, geom)
    r = evaluate_depth_gate(depth, geom, K, None, ScanGateThresholds(center_patch_frac=0.2,
                                                                     min_valid_depth_frac=0.2))
    assert r.detected and abs(r.distance_mm - 500.0) < 1.0 and r.tilt_deg < 1.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `py -3.10 -m pytest tests/test_scan_depth_gate.py tests/test_scan_survey.py -q`
Expected: FAIL (signature)

- [ ] **Step 3: Implement `depth_gate.py`**

```python
from ...core.depth_geometry import CameraGeometry, ColorRegistered


def evaluate_depth_gate(depth, geometry: CameraGeometry, K_color, dist_color,
                        th: ScanGateThresholds, *, max_samples: int = 3000) -> ScanGateReading:
    """Gate one NATIVE depth frame. The reticle is a COLOUR-image region, so depth
    points are registered into the calibrated colour model and the centre patch is
    selected there; distance/tilt are computed on colour-frame points."""
    if depth is None or np.asarray(depth).size == 0 or geometry is None:
        return _not_detected(th, 0.0)
    reg = ColorRegistered.build(depth, geometry, K_color, dist_color, stride=2)
    in_patch = reg.in_center_patch(th.center_patch_frac)
    valid_frac = min(1.0, reg.valid_frac_in_center_patch(th.center_patch_frac))
    if valid_frac < th.min_valid_depth_frac or in_patch.sum() < 8:
        return _not_detected(th, valid_frac)
    pts = reg.pts_mm[in_patch]
    distance_mm = float(np.median(pts[:, 2]))
    if len(pts) > max_samples:
        pts = pts[::int(np.ceil(len(pts) / max_samples))]
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vt[2] / max(float(np.linalg.norm(vt[2])), 1e-9)
    ...   # the tilt/gates code from here on is UNCHANGED (normal orientation, tilt_b/c, gates dict)
```
Delete the `depth_scale` parameter and the "raw mm" docstring sentences.

- [ ] **Step 4: Implement `survey.py`**

`survey_surface(depth, geometry, K_color, dist_color, thresholds)`: FOV from `K_color` and `geometry.color_size` (the image the HUD shows). Replace the manual back-projection (`:169-181`) with `pts_mm, _ = backproject(depth, geometry, stride=<stride so that len <= th.max_samples>)`; `valid_frac = np.count_nonzero(depth) / depth.size` (fraction of the DEPTH frame with valid depth — same meaning as before). Replace every projection to normalised uv (`:236-237`, `:267-268`, and the grid/outline code that uses `fx, fy, cx, cy, W, H`) with `project_to_color(points, K_color, dist_color) / [W, H]` where `W, H = geometry.color_size`. Remove `depth_scale`.

- [ ] **Step 5: `corner_evidence.py`**

`extract_corner_evidence(registered, K, polygon_uv, T_base_cam, *, corner_hint_uv=(0.5, 0.5), arm_frac=0.35, samples_per_arm=40, inset_px=4.0, window_px=3, ...)`: `w, h = registered.color_size`; delete `depth = np.asarray(depth)` checks (replace with `if registered is None or len(registered) == 0: return None`); every `_median_depth(depth, px, py, window_px)` becomes `registered.median_z_near(px, py, window_px)` with the NaN check `if not np.isfinite(z) or z <= 0: continue`. `_deproject_base` unchanged (pinhole with the colour K at the sample pixel, depth = the registered median — same convention as before). Delete `_median_depth`.

Tests (`tests/test_corner_evidence.py`): the scenes are rendered as aligned depth in `K`'s model; wrap each call: `extract_corner_evidence(ColorRegistered.build(depth, gf.aligned(K, (w, h)), K, None), K, poly, T, ...)`. Add a helper `_reg(depth)` at the top of the file to keep the diff small. All existing assertions stay.

- [ ] **Step 6: `reconstruct.py` — depth-only TSDF**

```python
@dataclass
class ScanView:
    color: np.ndarray         # HxWx3 BGR (diagnostics / saved views only)
    depth: np.ndarray         # native depth image, geometry.depth_unit_mm per step
    pose_T: np.ndarray        # 4x4 base->COLOUR camera, mm (the hand-eye pose)
    geometry: "CameraGeometry"


def fuse_views(views, *, voxel_size_m=0.004, sdf_trunc_m=0.02,
               depth_min_m=0.2, depth_max_m=1.5) -> FusionResult:
    """Integrate every posed DEPTH view into a TSDF volume (no colour: the scan mesh
    is neutral by contract, and the depth image is not registered to colour)."""
    import open3d as o3d
    from ...core.depth_geometry import depth_pose

    if not views:
        raise ValueError("no views to fuse")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_size_m), sdf_trunc=float(sdf_trunc_m),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    n_used = 0
    for v in views:
        g = v.geometry
        w, h = g.depth_size
        depth = np.ascontiguousarray(np.asarray(v.depth, np.uint16))
        units_per_m = 1000.0 / float(g.depth_unit_mm)
        d = depth.copy()
        d[d < float(depth_min_m) * units_per_m] = 0
        grey = np.zeros((h, w, 3), np.uint8)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(grey), o3d.geometry.Image(d),
            depth_scale=units_per_m, depth_trunc=float(depth_max_m),
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, _intrinsic(g.depth_K, w, h),
                         _pose_to_extrinsic_m(depth_pose(v.pose_T, g)))
        n_used += 1
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return FusionResult(mesh=mesh, cloud=volume.extract_point_cloud(), n_views=n_used)
```
`look_point_from_views`: the central patch is now the DEPTH image's centre, so the ray is the depth camera's axis: `T = depth_pose(v.pose_T, v.geometry)`; `d_mm = median(valid) * v.geometry.depth_unit_mm`; `pts.append(T[:3, 3] + d_mm * T[:3, 2])`.
`_view_support_counts(vertices_m, views, *, tolerance_m, depth_min_m, depth_max_m)`: per view `g = v.geometry; fx, fy, cx, cy from g.depth_K; w, h = g.depth_size; T = depth_pose(v.pose_T, g)`; `d_m = depth[vv, u] * g.depth_unit_mm / 1000.0`. `clean_measured_surface_mesh(...)`: drop the `depth_scale` kwarg and pass nothing extra to `_view_support_counts`.

`tests/test_scan_reconstruct.py`: `rc.ScanView(*_render(T), pose_T=T, geometry=gf.aligned(K, (W, H)))`; `fuse_views(views, voxel_size_m=..., ...)` without K/width/height. Add one assertion that a 0.1 mm-unit view fuses to the same plane: render, multiply the depth by 10 (`(depth.astype(np.uint32) * 10).astype(np.uint16)`), `geometry=gf.aligned(K, (W, H), depth_unit_mm=0.1)`, fuse, plane normal +Z within 1 deg and extent within 5 %.

- [ ] **Step 7: `service.py`**

- `_registered(frame, cfg, stride=1)`: `ColorRegistered.build(frame.depth, frame.geometry, cfg.camera.K, cfg.camera.dist, stride=stride)`.
- `_authoritative_acquisition` (`:425-428`): `evaluate_depth_gate(frame.depth, frames[-1].geometry, K, cfg.camera.dist, scan_gate_thresholds(scfg))`; `survey_surface(frame.depth, frames[-1].geometry, K, cfg.camera.dist, _survey_thresholds(scfg))`. Attach `geometry=frames[-1].geometry` to the `SimpleNamespace` frame it returns (the median-combined frame must carry the geometry of its sources; they share one connection).
- `_plane_rms_mm(frame, cfg, *, stride=8, outline_uv=None)`: `reg = _registered(frame, cfg, stride)`; `pts = reg.pts_mm[reg.in_polygon(shrunk)]` if `outline_uv` else `reg.pts_mm`; the 3 % shrink toward the centroid is kept on the normalised polygon; rest unchanged. Update its two callers (`:1420`, and grep for `_plane_rms_mm(`).
- `_backproject_depth(depth, geometry) -> backproject(depth, geometry)[0]` (colour-frame mm). Callers: `:1520` (`_reference_locate`) becomes `_backproject_depth(frame.depth, frame.geometry)`.
- `_deproject_plane_points_mm(depth, geometry, T_base_cam, *, plane_normal_cam, plane_point_cam, band_mm, stride=6)`: `pts_cam, uv = backproject(depth, geometry, stride=stride)`; `n_grid_total = (h // stride) * (w // stride)` of the depth image; band test on `pts_cam` (colour frame, as `measurement.normal_cam` is); transform survivors with `T_base_cam`. Rewrite the docstring paragraph that says "this module does NOT reuse `_backproject_depth`" — it now does, and the units are explicit through the geometry.
- Five-position (`:1344-1352`, `:1420`, `:1437`): `survey_surface(frame.depth, frame.geometry, K, cfg.camera.dist, corner_th)`; `_deproject_plane_points_mm(frame.depth, frame.geometry, T_base_cam, ...)`; `plane_rms_mm=_plane_rms_mm(frame, cfg)`; `extract_corner_evidence(_registered(frame, cfg), K, polygon_uv, T_base_cam, corner_hint_uv=(0.5, 0.5), closed=True)`.
- `_save_views(views, run_dir, *, log)`: `meta = {"camera_geometry": views[0].geometry.to_dict(), "views": [...]}`; PNG depth stays raw uint16 (now 0.1 mm steps — say so in the docstring). Caller `:3104`.
- Fusion (`:3110-3112`): `fuse_views(views, voxel_size_m=voxel_m, sdf_trunc_m=scfg.sdf_trunc_m, depth_min_m=scfg.depth_min_m, depth_max_m=scfg.depth_max_m)`; `clean_measured_surface_mesh(...)` without `depth_scale=`.
- `_capture` (`:3294`) and `_capture_burst` (`:3365`): `ScanView(color=color, depth=depth, pose_T=pose, geometry=frames[0].geometry)` (burst: `g["frames"][0].geometry`). Skip a pose whose frames have `geometry is None` with a log line.
- `grep -n "depth_scale" tasni/modules/scan/` must return nothing.

- [ ] **Step 8: Fake cameras in the scan tests**

`tests/test_scan_job.py:113`: `SimpleNamespace(color=color, depth=depth, timestamp=FRAME_TIMESTAMP, geometry=gf.aligned(K, (W, H)))` — one line; `FakeBurst.fetch_all` returns those same objects so they carry it. `tests/test_five_position.py`: same pattern wherever a frame is built (grep `SimpleNamespace(color=`).

- [ ] **Step 9: Run the scan tests**

Run: `py -3.10 -m pytest tests/test_scan_depth_gate.py tests/test_scan_survey.py tests/test_corner_evidence.py tests/test_scan_reconstruct.py tests/test_scan_job.py tests/test_five_position.py tests/test_survey_contract.py tests/test_scan_plane.py tests/test_rect_fit.py -q`
Expected: all PASS (open3d-dependent ones skip cleanly if it is absent, as today).

- [ ] **Step 10: Commit + push**

```bash
git add tasni/modules/scan tests/test_scan_depth_gate.py tests/test_scan_survey.py tests/test_corner_evidence.py tests/test_scan_reconstruct.py tests/test_scan_job.py tests/test_five_position.py
git commit -m "feat(scan): native depth through the greeting geometry - colour-registered gate/survey/corner evidence, depth-only TSDF, saved views carry camera_geometry"
git push origin sensor-layer-v2
```

---

### Task 11: `tools/characterize_distance.py` — colour corners, depth discs

**Files:**
- Modify: `tools/characterize_distance.py` (`_backproject_valid_mm` `:175-210`; `_corner_point_mm` `:255-275`; `_board_region_mask` (grep); the loop `:317-390`)
- Test: `tests/test_characterize.py`

**Interfaces:**
- Consumes: `ColorRegistered`, `ray_point` (Task 7); `Frame.geometry` (Task 8).
- Produces: `_board_points_mm(registered, board_polygon_uv_norm) -> np.ndarray`; `_corner_point_mm(registered, u, v, K, dist, *, disc_px) -> np.ndarray | None`; CLI flag `--corner-disc-px` (default 6).

This tool is what PROVES the audit's win (acceptance rows "plane RMS" and "length_err_mm"), so it must not carry a hidden 1 mm or aligned assumption.

- [ ] **Step 1: Failing test (append to `tests/test_characterize.py`)**

```python
def test_corner_point_reads_depth_through_the_colour_disc():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import geometry_fixtures as gf
    from tasni.core.depth_geometry import ColorRegistered
    from tools.characterize_distance import _corner_point_mm, _board_points_mm
    K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
    geom = gf.offset(color_K=K, color_size=(320, 240))
    xs, ys = np.meshgrid(np.linspace(-120, 120, 97), np.linspace(-90, 90, 73))
    plane = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 450.0)])
    depth = gf.render_depth_in_depth_camera(plane, geom)
    reg = ColorRegistered.build(depth, geom, K, None)
    p = _corner_point_mm(reg, 200.0, 150.0, K, None, disc_px=6)
    assert p is not None and abs(p[2] - 450.0) < 1.0
    np.testing.assert_allclose(p[:2], [(200 - 160) / 300 * p[2], (150 - 120) / 300 * p[2]], atol=1e-6)
    board = _board_points_mm(reg, [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
    assert len(board) > 100 and np.all(np.abs(board[:, 2] - 450.0) < 1.5)
```
(`tools/` needs an `__init__.py` for `from tools.characterize_distance import ...`; if it lacks one, add an empty file.)

- [ ] **Step 2: Run to verify it fails**

Run: `py -3.10 -m pytest tests/test_characterize.py -v -k colour_disc`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implement**

```python
from tasni.core.depth_geometry import ColorRegistered, ray_point


def _board_points_mm(registered: ColorRegistered, board_polygon_uv_norm) -> np.ndarray:
    """Every registered depth point whose COLOUR projection lies inside the detected
    board polygon (normalised colour coords), colour-frame mm."""
    return registered.pts_mm[registered.in_polygon(board_polygon_uv_norm)]


def _corner_point_mm(registered: ColorRegistered, u: float, v: float, K, dist, *,
                     disc_px: float = 6.0) -> "np.ndarray | None":
    """One ChArUco corner (colour pixel) -> colour-frame mm: the median depth of the
    registered points within ``disc_px`` of it, placed on the corner's undistorted ray."""
    z = registered.median_z_near(u, v, disc_px)
    if not np.isfinite(z) or z <= 0:
        return None
    return ray_point(u, v, z, K, dist)
```
In the loop: `reg = ColorRegistered.build(frame.depth, frame.geometry, K, cfg.camera.dist)`; `_board_region_mask(...)` (a pixel mask over the aligned image) is replaced by the polygon it was built from — pass the detected corners' convex hull in normalised coords (`hull / [W, H]`, `W, H = cfg.camera.size`) to `_board_points_mm`; `pts = _subsample_for_plane_fit(_board_points_mm(reg, hull_uv))`. Corner fallbacks (`:380-383`) call `_corner_point_mm(reg, *px_by_id[id], K, cfg.camera.dist, disc_px=args.corner_disc_px)`. Delete `_backproject_valid_mm`, `depth_scale = cfg.scan.depth_scale` (`:317`) and the docstring lines that call the division "an identity". Record `frame.geometry.to_dict()` (temps, unit, preset) in the trial JSON under `"camera_geometry"` so a characterisation file states the configuration it was measured under.

- [ ] **Step 4: Run**

Run: `py -3.10 -m pytest tests/test_characterize.py -q` and `py -3.10 tools/characterize_distance.py --help`
Expected: PASS; help shows `--corner-disc-px`.

- [ ] **Step 5: Commit + push**

```bash
git add tools/characterize_distance.py tools/__init__.py tests/test_characterize.py
git commit -m "feat(characterize): colour-detected corners read depth through registered discs; trial records camera_geometry"
git push origin sensor-layer-v2
```

---

### Task 12: HUD check + frontend build

**Files:**
- Modify: `tasni/webui/src/pages/AimHud.tsx` (comment `:4` and the `W, H` comment `:67`)

The HUD draws `outline_uv`/`points_uv`/`grid_uv` as NORMALISED 0-1 coordinates scaled into its own 1280x720 SVG viewBox (`AimHud.tsx:159-163`), and the server normalises by `overlay_size` (`server_unicast_syncronous.py:375-376`). 1080p colour is 16:9 like 720p, so nothing moves. This task is the proof, not a change.

- [ ] **Step 1: Verify no pixel-space assumption survives**

Run: `grep -n "1280\|720\|1920\|1080" tasni/webui/src/pages/AimHud.tsx tasni/webui/src/pages/Scan.tsx tasni/webui/src/pages/StreamStats.tsx`
Expected: only the viewBox constants in `AimHud.tsx` and comments. If any component divides telemetry by a literal 1280/720, replace with the viewBox constants `W`/`H` (the coordinates are normalised, so the viewBox is the only pixel space).

- [ ] **Step 2: Update the two comments**

Line 4: `// the preview box is 16:9, so the 1280x720 viewBox is a normalised canvas over the image (the stream itself is 1920x1080 since camera protocol 2; all overlay coords arrive normalised 0-1).` Line 67 gets `// SVG canvas only - NOT the camera resolution`.

- [ ] **Step 3: Build**

Run: `cd tasni/webui && npm run build`
Expected: build succeeds with no type errors.

- [ ] **Step 4: Commit + push**

```bash
git add tasni/webui/src/pages/AimHud.tsx
git commit -m "docs(hud): the viewBox is a normalised canvas, not the camera resolution"
git push origin sensor-layer-v2
```

---

### Task 13: Deploy, restart, cell acceptance, docs

**Files:**
- Modify: `docs/realsense-capability-audit-2026-08-29.md` (append "## 7. Results" with the measured table), `docs/jetson-scanner.md` (`:30` librealsense row; the streams/filters paragraph), `AGENTS.md` (open items + the macros note), `CLAUDE.md` (the stale "High-Accuracy preset" sentence under Roadmap; the macros table note), `docs/agent-debug-map.md` (protocol 2 pointer)

**Preconditions:** Tasks 1-12 pushed on `main`; Task 3's rebuild in place; the cell free; the operator present (the arm moves for the hand-eye validation).

- [ ] **Step 0: Merge to main**

```bash
git checkout main && git pull origin main && git merge --no-ff sensor-layer-v2 -m "feat(realsense): sensor layer at full fidelity (protocol 2)" && git push origin main
```
The Jetson's auto-pull will now pick this up within ~2 min on its own; the explicit deploy below just makes it immediate and restarts the service.

- [ ] **Step 1: Deploy the server and restart the backend**

```
py -3.10 tools/jetson_deploy.py deploy
py -3.10 tools/jetson_deploy.py status
```
Journal must show, in order: the as-found JSON path, `depth_units -> requested 0.0001 ... device reports 0.0001`, `depth_unit_mm = 0.1`, `global_time_enabled = True`, temps, `depth (1280, 720) colour (1920, 1080)`, and NO `extrinsic layout check failed`. Then stop and restart the Tasni backend (`.\start.ps1`) and confirm its process StartTime is newer than every file changed today (the module-cache trap).

- [ ] **Step 2: A stale client is refused, not hung**

From the workstation with the OLD client code (`git stash` is not needed — use raw sockets):
```
py -3.10 -c "import socket; s=socket.create_connection(('10.12.171.70',1024),5); s.settimeout(5); print(s.recv(128))"
```
Expected: `b'ERR protocol 2 required; send MODE FULL V2\n'` within a second; journal logs `did not request V2`.

- [ ] **Step 3: R2 acceptance — quantisation**

Run: `py -3.10 tools/probe_depth_quantisation.py` (arm parked, same pose as the audit).
Pass: `unit 0.1 mm`, `unique >= 200`, `min step 0.10 mm`. Record the line.

- [ ] **Step 4: R3/R5 acceptance — points per view and validity**

Run the probe's first line twice more and note `valid`: it must be BELOW 0.999 (real support; hole filling is gone). Points per view: `np.count_nonzero(frame.depth)` at the same pose vs the audit's aligned frame (921,600 x 0.999): with the full field expect a count near the same number of PIXELS but covering ~2x the solid angle — the honest comparison is the scan's `captured (... depth px)` log line at a fixed `TasniScan_*` pose before/after; record both.

- [ ] **Step 5: A1 acceptance — the known length from the depth cloud**

Run: `py -3.10 tools/characterize_distance.py` at three standoffs on the ChArUco board (300 / 450 / 600 mm), with the as-found JSON reloaded on the device if any preset experiment happened in between (none should have). Pass: plane RMS not worse than the 2026-08-13 record at each stop; `length_err_mm` within the plane-noise band of the true pair distance where the 08-14 audit reported +2 %.

- [ ] **Step 6: R7 validation — hand-eye at 1080p**

In the app: Calibration -> Create targets -> dry tour -> Run. Pass: held-out reprojection in the same band as the last PASS (about 0.9 px), board consistency in the same band (about 0.8-1.3 mm). This is a validation, not a recalibration; if the verdict is worse than `borderline`, inspect `intrinsics["1920x1080"]` in `tasni.config.json` (it should be exactly 1.5x the 720p entry) before anything else.

- [ ] **Step 7: HUD**

Scan page: aim at the platform. LEVEL/TILT readouts and the blue rectangle behave as on 2026-07-06 (parked: no jitter; a real move tracks). Extrusion page: a measure-only take completes and its manifest carries `provenance.camera_geometry` with `depth_unit_mm: 0.1` and both temps.

- [ ] **Step 8: Write the results into the audit doc**

Append `## 7. Results (2026-08-3x)` to `docs/realsense-capability-audit-2026-08-29.md`: a table with rows R1 (idle CPU, ms/grab before/after from Task 3), R2 (unique values / step), R3 (depth px per view before/after), R5 (validity fraction), A1 (length error at three standoffs), R7 (hand-eye verdict + numbers), and the as-found JSON path. Every number measured, none estimated.

- [ ] **Step 9: Docs**

- `docs/jetson-scanner.md:30`: librealsense row -> `2.55.1 Release+CUDA+OpenMP source build in ~/librealsense/build_cuda (rollback: build_py310 + ~/pyrealsense2_rollback)`; streams paragraph -> depth 1280x720 z16, colour 1920x1080 bgr8, no IR, unaligned, `depth_units` 0.1 mm, filters `threshold -> disparity -> spatial -> temporal -> disparity_inv`; wire -> `MODE FULL V2` + greeting.
- `CLAUDE.md`: in the Roadmap bullet "The RealSense options + filter chain live in `server/server_unicast_syncronous.py` (device runs a Custom preset ...)" append "; since protocol 2 (2026-08-3x) depth is streamed native/unaligned at 0.1 mm behind `MODE FULL V2` with a JSON greeting — see `docs/superpowers/specs/2026-08-29-sensor-layer-full-fidelity-design.md`". In the Macros table add one line under it: "The embedded macros speak the pre-V2 wire and are refused by the current camera server; they are superseded by the `tasni/` app."
- `AGENTS.md`: same macros note under environment traps; open items: R4.2/R10/R6/R8/R11/chrony are the next specs.
- `docs/agent-debug-map.md`: one pointer line "Camera wire: protocol 2 — `server/handshake.py`, greeting in `server/rs_geometry.py`, host side `tasni/core/depth_geometry.py`".

- [ ] **Step 10: Commit + push, mention the Jetson state**

```bash
git add docs/realsense-capability-audit-2026-08-29.md docs/jetson-scanner.md AGENTS.md CLAUDE.md docs/agent-debug-map.md
git commit -m "docs(realsense): protocol-2 results on the cell; scanner/agent docs updated; macros marked superseded"
git push origin main    # Task 13 runs on main after the Step 0 merge
```
Report: the pushed hashes, `jetson_deploy status` output, and the results table.

---

## Self-review notes (done while writing; kept so the executor sees the reasoning)

- **Spec coverage.** 4.1 wire/gate/greeting -> Tasks 4, 6, 8. 4.2 server (rs_config, rs_geometry, align removal, filters, 1080p, sizes, startup log) -> Tasks 4, 5, 6. 4.3 host core (`depth_geometry`, client, `depth_scale` deletion, K migration) -> Tasks 7, 8. 4.4 consumers (extrusion + archive + figures fallback + voxel; scan TSDF/gate/survey/service/`_save_views`; characterize; calibration no-op; HUD) -> Tasks 9, 10, 11, 12. 4.5 sequencing -> Tasks 1-3 first, 13 last. 5 error handling: refusal (6, 8), greeting validation (7, 8), extrinsic assert (5, 6), archive fallback (9), `depth_units` achieved-not-requested (4, 6). 6 tests: every task has its file. 7 acceptance -> Task 13. 9 task list -> mapped 1:1 (spec task 12 "HUD check" is Task 12; spec task 13 is Task 13).
- **Late change absorbed (`041ad1b`, pushed after the plan was written):** extrusion now reads the COLOUR frame -- a saturation gate separates bead from board -- written against aligned depth (same-pixel indexing). Task 9 Step 3b ports it to registered points; without that port the gate abstains silently on protocol-2 frames and the 2026-08-29 ring failure returns.
- **Consumers the spec did not list, found while planning and covered:** `corner_evidence.extract_corner_evidence` and `service._plane_rms_mm` (colour-pixel depth reads; Task 10), `look_point_from_views` (Task 10), the live-preview `depth_probe` interleave (`core/livepreview.py:163`) — verified UNUSED in production (no module passes `depth_probe=`; scan uses the telemetry side-channel), so it is left alone; the extrusion readiness grabs (`measure.py:367`, `service.py:769`) only check `depth is not None` and need nothing.
- **Type consistency.** `CameraGeometry`, `backproject(depth, geom, *, stride, mask) -> (pts, uv_depth)`, `depth_pose(T_x_color, geom)`, `ColorRegistered.build(depth, geom, K_color, dist_color, *, stride)` are used with those exact shapes in Tasks 9, 10, 11. `ScanView(color, depth, pose_T, geometry)` in Task 10 matches its `fuse_views`/`_view_support_counts` readers. `Frame.geometry` (Task 8) is what Tasks 9-11 read.
- **Known approximation, stated in code:** `ColorRegistered.valid_frac_in_center_patch` estimates the expected point density from the registered footprint; it feeds a 0.5 threshold, not a measurement.
