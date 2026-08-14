# Scan Chain Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 13 actionable findings of the 2026-08-14 scan-chain audit — one camera model end to end, honest depth in the metrology chain, real-time aiming telemetry, and matching live/lock behaviour.

**Architecture:** Three phases. Phase 1 = small, independent accuracy/hygiene fixes (each own commit). Phase 2 = the two structural fixes (camera-model split A1, Jetson telemetry off the video thread B1) plus the fixes that ride on them (C1, C2, A3). Phase 3 = capture-chain efficiency (A5, B2, A7). Host code is `tasni/`; camera-server code is `server/` (runs on the Jetson, auto-deployed by `tools/jetson_deploy.py deploy`).

**Tech Stack:** Python 3.10 (`py -3.10`), numpy, OpenCV 4.7 (contrib), pydantic v2, FastAPI, pyrealsense2 (Jetson only), Open3D 0.17, pytest.

**Spec:** `docs/scan-audit-2026-08-14.md` — read it first; every task cites its finding ID (A1…C6). The three corrections from the post-audit review are already folded in: A6 masks RMS to the *measured plane's inliers* (not `outline_uv`, which is a fabricated square for corner captures); A5 flushes with **bare** `wait_for_frames()` (never `getFrames`, which costs ~1 s/frame on the Nano); B2 spaces kept frames by capture timestamp (back-to-back frames share temporal-filter state — the `length_spread==0` lesson already recorded in `tools/characterize_distance.py`).

## Global Constraints

- Repo root: `D:\DesktopStuff\RAFFI NO TOUCH\backuprobodk\RoboDkClaude`. Branch: `calibration-improvements`. Python: **`py -3.10`**.
- **NEVER run the full pytest suite** (user rule — it is too slow and gets interrupted). Run only the test files/`-k` selections named in each task. `tests/test_scan_job.py` alone takes ~2–3 min; that is expected.
- **Commit + push after every task** (CLAUDE.md working agreement). Plain `git push` on this branch.
- Any task touching `server/` ends with: `py -3.10 tools/jetson_deploy.py deploy`, then `py -3.10 tools/jetson_deploy.py status` (confirm `active` + `LISTEN … :1024`). The Jetson follows this branch and restarts the service when `server/` changed.
- `server/*.py` must stay **stdlib + numpy only** (the Jetson venv has numpy, pyrealsense2, lz4, turbojpeg — no scipy, no cv2, no tasni deps). `server/scan_overlay.py` must stay importable with **no** pyrealsense2/turbojpeg at all (host tests import it).
- `server/server_unicast_syncronous.py` must stay importable on the host with `pyrealsense2`/`turbojpeg` stubbed as `SimpleNamespace()` — `tests/test_scan_telemetry_server.py` relies on this; do not add module-level code that calls into those stubs.
- The repo pattern for host/Jetson shared math is **textual duplication with a parity test** (see `_density_extent_1d` in `tasni/modules/scan/plane.py` vs `server/scan_overlay.py`). Follow it; do not make `tasni/` import from `server/` in production code.
- Steps tagged **[CELL]** need the physical robot/camera cell and cannot be completed headless. Do every other step, leave [CELL] boxes unchecked, and say so in the task summary.
- Do not touch `macros/`, `Tasni.rdk`, or anything under `tasni/modules/calibration/` except where a task names it.
- Do not run `pip install`/upgrade anything; Task 7 only edits `pyproject.toml` text.

## Orientation for a fresh session (read once)

- Scan backend: `tasni/modules/scan/service.py` (3.3k lines — the lock/generate/run/insert logic), routes in `tasni/modules/scan/module.py`, pure geometry in `survey.py` / `plane.py` / `depth_gate.py`, camera client in `tasni/core/camera.py`, config in `tasni/core/config.py` (`ScanConfig` ~line 351, `CameraConfig` ~line 44).
- Jetson server: `server/server_unicast_syncronous.py` (one file). Telemetry math: `scan_plane_telemetry` (~line 229). H.264 path + feeder thread: `stream_h264` (~line 850). Handshake parsing: `handle_client` (~line 711). Burst capture: `stream_burst` (~line 1034).
- Line numbers in this plan are from commit `50fcab7`; re-grep if they drift.

---

# Phase 1 — small accuracy + hygiene fixes

### Task 1: Clamp the standoff accept window to the accurate band (A2)

**Files:**
- Modify: `tasni/modules/scan/service.py` (lock gate ~465–472; live gate ~1750–1754; helper near `scan_gate_thresholds` ~723)
- Test: `tests/test_scan_job.py` (one existing failing test + one new)

**Interfaces:**
- Produces: `standoff_accept_window_mm(ideal_mm: float, scfg) -> tuple[float, float]` in `tasni/modules/scan/service.py` (module level). Payload key `"distance_window_mm": [lo, hi]` added to both gate payloads (additive; no consumer changes).

- [ ] **Step 1: Run the already-failing test to confirm the defect**

Run: `py -3.10 -m pytest tests/test_scan_job.py::test_generate_refuses_when_too_far -q`
Expected: FAIL (`expected refusal — surface out of the standoff band`). This is the red step: a 900 mm standoff passes because ideal clips to 800 and |900−800| ≤ 150.

- [ ] **Step 2: Add the helper**

In `service.py`, directly above `scan_gate_thresholds` (~line 723):

```python
def standoff_accept_window_mm(ideal_mm: float, scfg) -> tuple[float, float]:
    """Distance-gate accept window: ideal ± distance_tol_mm, CLAMPED to the
    camera's accurate depth band (audit A2).

    distance_tol_mm was widened 50 -> 150 on 2026-08-13 (plane RMS is flat
    across ±150 mm), but the tolerance is applied around an ideal that is
    itself clipped to [accurate_min_mm, accurate_max_mm] — without this clamp
    the gate accepts 150..950 mm: past the characterized envelope on one end
    and below the D435i's MinZ on the other."""
    lo = max(float(ideal_mm) - float(scfg.distance_tol_mm), float(scfg.accurate_min_mm))
    hi = min(float(ideal_mm) + float(scfg.distance_tol_mm), float(scfg.accurate_max_mm))
    return lo, hi
```

- [ ] **Step 3: Use it in the lock gate**

In `lock_scan_surface` (~line 462, right after `gate_payload["distance_tol_mm"] = ...`):

```python
    lo_mm, hi_mm = standoff_accept_window_mm(ideal_distance, scfg)
    gate_payload["distance_window_mm"] = [lo_mm, hi_mm]
```

and change the `final_gates` distance entry (~line 467) from the `abs(...) <= tol` form to:

```python
        "distance": (
            reading.distance_mm is not None
            and lo_mm <= float(reading.distance_mm) <= hi_mm
        ),
```

- [ ] **Step 4: Use it in the live gate**

In `live_scan_telemetry_payload`, after the `ideal_distance` value is FINAL (after the crop-hold / deadband logic, just before the `gates = {` dict ~line 1750):

```python
    lo_mm, hi_mm = standoff_accept_window_mm(ideal_distance, scfg)
```

change `"distance": abs(distance - ideal_distance) <= th.distance_tol_mm,` to
`"distance": lo_mm <= distance <= hi_mm,` and add `"distance_window_mm": [lo_mm, hi_mm],` next to `"distance_tol_mm"` in the returned dict.

- [ ] **Step 5: Add the live-gate regression test**

Append to `tests/test_scan_job.py` (it already imports `scan_service`; use the same ScanConfig source as neighbouring live-payload tests — grep `live_scan_telemetry_payload(` in the file and mirror the fixture style):

```python
def test_live_distance_gate_clamped_to_accurate_band():
    """Audit A2: a crop-latched ideal of 800 must not accept 930 mm just because
    |930-800| <= distance_tol_mm — the window's top is accurate_max_mm."""
    import time as _time
    from tasni.core.config import ScanConfig
    scfg = ScanConfig()
    raw = {"detected": True, "distance_mm": 930.0, "tilt_deg": 1.0,
           "valid_frac": 0.9, "surface_mode": "crop",
           "_received_at": _time.time(), "timestamp": _time.time()}
    out = scan_service.live_scan_telemetry_payload(raw, scfg, previous_ideal_mm=800.0)
    assert out["ideal_distance_mm"] == 800.0
    assert out["gates"]["distance"] is False, out
    ok = dict(raw, distance_mm=780.0)
    out2 = scan_service.live_scan_telemetry_payload(ok, scfg, previous_ideal_mm=800.0)
    assert out2["gates"]["distance"] is True, out2
```

- [ ] **Step 6: Run both tests**

Run: `py -3.10 -m pytest tests/test_scan_job.py -q -k "too_far or clamped_to_accurate"`
Expected: 2 passed.

- [ ] **Step 7: Sanity-run the file's gate/lock tests**

Run: `py -3.10 -m pytest tests/test_scan_job.py -q`
Expected: all pass (the file was 100% green before except `too_far`).

- [ ] **Step 8: Commit + push**

```bash
git add tasni/modules/scan/service.py tests/test_scan_job.py
git commit -m "Clamp the standoff accept window to the accurate band (audit A2)"
git push
```

---

### Task 2: Drop hole-filling from the capture filter chain (A4)

**Files:**
- Modify: `server/server_unicast_syncronous.py:1117-1125` (`setup_depth_filters`)

**Interfaces:**
- Produces: `setup_depth_filters()` returns 4 filters (no `hole_filling_filter`). No signature change.

- [ ] **Step 1: Edit `setup_depth_filters`**

Replace the function body so the chain is disparity → spatial → temporal → depth, and document why:

```python
def setup_depth_filters():
    # Full-resolution scan data: do NOT decimate. Work in disparity space for the
    # filters that benefit from roughly uniform stereo noise, then return to depth.
    #
    # NO hole_filling_filter (audit A4): it FABRICATES depth by copying from
    # neighbours — a visualisation filter, not a measurement one. In this chain it
    # corrupted exactly what the scan measures: the density-cliff edge trim (it
    # smooths over the cliff), the fully_framed border test (filled pixels reach
    # the border), the coverage dots (documented as REAL measured depth), and the
    # TSDF (fuses invented geometry). A hole in the data is information.
    depth_to_disparity = rs.disparity_transform(True)
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    disparity_to_depth = rs.disparity_transform(False)
    return [depth_to_disparity, spatial, temporal, disparity_to_depth]
```

- [ ] **Step 2: Compile check**

Run: `py -3.10 -m py_compile server/server_unicast_syncronous.py`
Expected: silent success.

- [ ] **Step 3: Host-side server tests still pass**

Run: `py -3.10 -m pytest tests/test_scan_telemetry_server.py tests/test_scan_overlay.py -q`
Expected: all pass (telemetry math doesn't touch the filter chain).

- [ ] **Step 4: Commit + push + deploy**

```bash
git add server/server_unicast_syncronous.py
git commit -m "Drop hole_filling from the Jetson capture filter chain (audit A4)"
git push
py -3.10 tools/jetson_deploy.py deploy
py -3.10 tools/jetson_deploy.py status
```
Expected: `active`, `LISTEN … 0.0.0.0:1024`.

- [ ] **Step 5 [CELL]: Watch the next real lock/scan for coverage-gate behaviour**

Removing fabricated pixels can lower `valid_frac`/mesh edge coverage slightly. On the next cell run, if locks that used to pass now fail `detected`, revisit `scan.min_valid_depth_frac` (0.5) — do NOT pre-tune it blind.

---

### Task 3: Five-position capture RMS against the measured plane (A6)

**Files:**
- Modify: `tasni/modules/scan/service.py` (new helper near `_plane_rms_mm` ~line 209; call site in `five_position_capture` ~line 1333)
- Test: `tests/test_scan_job.py`

**Interfaces:**
- Produces: `_plane_band_rms_mm(depth, K, *, plane_normal_cam, plane_point_cam, stride=8, depth_scale=1000.0, band_mm=25.0) -> float` in `service.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scan_job.py`:

```python
def test_five_position_rms_measures_the_surveyed_plane_not_the_scene():
    """Audit A6: a corner capture is majority background; RMS must be computed
    against the plane survey_surface measured, not a fresh whole-frame fit
    (which locks onto the dominant background and reports ITS flatness)."""
    rng = np.random.default_rng(0)
    K = np.array([[600.0, 0, 80], [0, 600.0, 60], [0, 0, 1.0]])
    depth = np.zeros((120, 160), np.float64)
    depth[:, 48:] = 1350.0                       # dominant flat wall, sigma ~0
    depth[:, :48] = 600.0 + rng.normal(0, 2.0, (120, 48))   # table, sigma 2 mm
    rms = scan_service._plane_band_rms_mm(
        depth, K, plane_normal_cam=[0.0, 0.0, -1.0],
        plane_point_cam=[0.0, 0.0, 600.0])
    assert 1.4 <= rms <= 2.8, rms                # the TABLE's 2 mm, not the wall's 0
    whole_scene = scan_service._plane_rms_mm(depth, K)
    assert not (1.4 <= whole_scene <= 2.8), whole_scene   # documents the old defect
```

- [ ] **Step 2: Run it to verify it fails**

Run: `py -3.10 -m pytest tests/test_scan_job.py -q -k five_position_rms_measures`
Expected: FAIL with `AttributeError: ... has no attribute '_plane_band_rms_mm'`.

- [ ] **Step 3: Implement the helper**

In `service.py`, directly after `_plane_rms_mm` (~line 253):

```python
def _plane_band_rms_mm(depth, K, *, plane_normal_cam, plane_point_cam,
                       stride: int = 8, depth_scale: float = 1000.0,
                       band_mm: float = 25.0) -> float:
    """Plane-fit RMS (mm) against an ALREADY-MEASURED plane (audit A6).

    _plane_rms_mm fits a fresh whole-frame plane, which for a five-position
    CORNER capture (majority background by design) locks onto the background
    and reports its flatness. And masking by survey outline_uv is no better
    there: when not fully framed, survey_surface has already replaced the
    outline with a fabricated reticle square. So: take the plane
    survey_surface measured for THIS capture, select pixels within a loose
    ``band_mm`` of it (drops background + spurious stereo pixels, same 25 mm
    convention as _plane_rms_mm), and report the RMS of those residuals."""
    d = np.asarray(depth, dtype=float)[::stride, ::stride]
    v, u = np.nonzero(d > 0)
    if len(v) < 50:
        return float("nan")
    z = d[v, u] / float(depth_scale) * 1000.0
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    pts = np.stack([(u * stride - cx) / fx * z, (v * stride - cy) / fy * z, z], axis=1)
    n = np.asarray(plane_normal_cam, dtype=float)
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    res = (pts - np.asarray(plane_point_cam, dtype=float)) @ n
    res = res[np.abs(res) <= float(band_mm)]
    if len(res) < 50:
        return float("nan")
    return float(np.sqrt(np.mean(res ** 2)))
```

- [ ] **Step 4: Switch the call site**

In `five_position_capture` (~line 1333), change
`plane_rms_mm=_plane_rms_mm(frame.depth, K),` to:

```python
        plane_rms_mm=_plane_band_rms_mm(
            frame.depth, K, plane_normal_cam=measurement.normal_cam,
            plane_point_cam=measurement.centroid_cam_mm),
```

- [ ] **Step 5: Run tests**

Run: `py -3.10 -m pytest tests/test_scan_job.py -q -k five_position` then `py -3.10 -m pytest tests/test_five_position.py -q`
Expected: all pass (`test_five_position.py` tests the pure survey class, not this capture path — it must stay green).

- [ ] **Step 6: Commit + push**

```bash
git add tasni/modules/scan/service.py tests/test_scan_job.py
git commit -m "Five-position capture RMS: residuals vs the measured plane, not the scene (audit A6)"
git push
```

---

### Task 4: Delete the dead border-framing computation (C4)

**Files:**
- Modify: `tasni/modules/scan/survey.py:221-230`

- [ ] **Step 1: Verify it is dead**

Run: `py -3.10 -m pytest tests/test_scan_job.py -q --collect-only >NUL 2>&1 & grep -rn "depth_within_border" tasni/ tests/ server/`
Expected: exactly one hit — `tasni/modules/scan/survey.py:227`.

- [ ] **Step 2: Delete it**

In `survey.py`, in step "6. Framed test", delete the `margin = th.border_margin_px` line and the whole `depth_within_border = not (...)` statement (lines ~226–230). Keep the `_corners_in_frame` closure and `fully_framed = _corners_in_frame(corners3d)` — those are the live rule. Update the step-6 comment to state the corner-based test is the only rule (the pixel-border test was superseded and removed per audit C4). Leave the `border_margin_px` field on `SurveyThresholds` (constructing it with the field must keep working).

- [ ] **Step 3: Run tests**

Run: `py -3.10 -m pytest tests/test_scan_job.py -q`
Expected: all pass.

- [ ] **Step 4: Commit + push**

```bash
git add tasni/modules/scan/survey.py
git commit -m "Remove the dead pixel-border framing computation (audit C4)"
git push
```

---

### Task 5: Correct the agent debug map (C5)

**Files:**
- Modify: `docs/agent-debug-map.md` (the `classifier.py` bullet ~lines 35–46, the Compact bullet ~lines 15–20, the "Existing Long Docs" list ~line 319, `Last updated` line 6)

- [ ] **Step 1: Rewrite the stale claims**

1. Replace the `classifier.py` bullet's text from "**Not wired into any production caller as of Task 17**…" through "…actually protecting a real lock." with:

```
  scan-coverage (predict-only planner run). Returns `CompactEligibility`.
  **Wired into `lock_scan_surface` since Task 18** (`service.py` — grep
  `classify_compact`): a compact lock is only recorded as MEASURED provenance
  when `classify_compact` passes; failures surface as gate warnings (with a
  guard-band backoff hint) and the lock completes record-less.
```

2. In the **Compact** bullet, delete the parenthetical "(see the `classifier.py` bullet below — **known gap**)" and the "NOT via `classify_compact`" clause; state that the pre-existing `surface_mode` full/crop heuristic decides crop-vs-full while `classify_compact` gates whether a full lock's boundary counts as measured.
3. Add to the "Existing Long Docs" list: `- docs/scan-audit-2026-08-14.md: end-to-end scan-chain audit (accuracy/latency findings A1-C6) — read before touching scan accuracy or the camera server.`
4. Bump `Last updated:` to 2026-08-14.

- [ ] **Step 2: Commit + push**

```bash
git add docs/agent-debug-map.md
git commit -m "docs: agent map — classify_compact IS wired; link the scan audit (audit C5)"
git push
```

---

### Task 6: Make `jetson_deploy.py` never block on a sudo prompt (C6)

**Files:**
- Modify: `tools/jetson_deploy.py:127-129` (`sudo` method), `status` function (~line 277)

- [ ] **Step 1: Harden the sudo helper**

Replace the `sudo` method (line 127) with:

```python
    def sudo(self, cmd, check=False, quiet=False):
        """Run a remote command under sudo without EVER being able to prompt
        (audit C6): with a password on file, feed `sudo -S` via stdin; without
        one (or if it is wrong) `sudo -n` fails fast instead of hanging the
        whole tool on `[sudo] password for jetson:`."""
        if self.sudo_pw:
            full = (f"echo '{self.sudo_pw}' | sudo -S -p '' bash -c {shq(cmd)} "
                    f"|| sudo -n bash -c {shq(cmd)}")
        else:
            full = f"sudo -n bash -c {shq(cmd)}"
        return self.run(full, check=check, quiet=quiet)
```

(Adapt the final `return` line to whatever the current body calls — read the method first; only the command construction changes.)

- [ ] **Step 2: Degrade `status` gracefully**

In `status(j)` (~line 284), wrap the `journalctl` call: if its return code is non-zero, print `--- recent logs unavailable (sudo needs a valid JETSON_SUDO_PASSWORD in secrets/jetson.env) ---` instead of failing.

- [ ] **Step 3: Verify end to end**

Run: `py -3.10 tools/jetson_deploy.py status`
Expected: completes without any `[sudo] password` prompt appearing; either logs or the graceful notice.

- [ ] **Step 4: Commit + push**

```bash
git add tools/jetson_deploy.py
git commit -m "jetson_deploy: sudo -n fallback so status/deploy can never hang on a prompt (audit C6)"
git push
```

---

### Task 7: Pin the validated dependency versions (tooling)

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Pin**

In `[project] dependencies`, change `"opencv-contrib-python"` to `"opencv-contrib-python>=4.7,<4.8"` and add `"numpy<2"` (cv2 4.7 wheels are numpy-1-only). In `[project.optional-dependencies]`, change `scan = ["open3d"]` to `scan = ["open3d>=0.17,<0.18"]`. Add one comment line above each: `# pinned to the validated set — the 4.8+ ChArUco migration is a deliberate, separate task (audit: Tooling / Deferred)`.

- [ ] **Step 2: Verify the file parses and the pins match reality**

Run: `py -3.10 -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); import cv2, numpy, open3d; print(cv2.__version__, numpy.__version__, open3d.__version__)"`
Expected: parses; prints `4.7.0 1.24.x 0.17.0` (the installed set satisfies the new pins — nothing to install).

- [ ] **Step 3: Commit + push**

```bash
git add pyproject.toml
git commit -m "Pin opencv<4.8, numpy<2, open3d 0.17 to the validated set (audit: tooling)"
git push
```

---

# Phase 2 — one camera model, real-time telemetry

### Task 8: Split the depth-grid intrinsics from the colour model (A1, host)

**Files:**
- Modify: `tasni/core/config.py` (`CameraConfig`, ~line 44), `tasni/modules/scan/service.py` (K sources at ~386, ~428, ~1212, ~2944, and inside `_reference_locate` ~1421)
- Test: `tests/test_scan_job.py`

**Interfaces:**
- Produces: `CameraConfig.depth_grid_intrinsics: dict[str, list[list[float]]]` (defaults = the factory `_DEFAULT_INTRINSICS`) and property `CameraConfig.K_depth -> np.ndarray`. Consumed by Tasks 9 and 13.

**The rule (memorise before editing):** `rs.align` on the Jetson builds the aligned depth image by projecting through the device-stored **factory** colour intrinsics with **zero distortion**. Inverting that grid back to 3D must therefore use the factory model (`K_depth`, pinhole, no undistort) — that recovers the true 3D regardless of the physical lens. The ChArUco-calibrated `K` + `dist` stay authoritative for real colour **pixels**: ChArUco, SAM boundary rays, `cv2.projectPoints`/`undistortPoints`, and image-framing math (how big something looks in the actual image).

Per-site decision table for `service.py`:

| Site | Uses K for | Verdict |
|---|---|---|
| `_authoritative_acquisition` ~386 | `evaluate_depth_gate` + `survey_surface` (aligned depth) | → `K_depth` |
| `lock_scan_surface` ~428 | `_plane_rms_mm`, `_survey_outline_history`, `_crop_gate_payload` | → `K_depth` (uniform; the crop overlay shifts ≤2 %, consistent with the survey overlays) |
| `five_position_capture` ~1212 | deproject + corner evidence + Task 3's RMS helper | → `K_depth` |
| `ScanCaptureJob.__call__` ~2944 | `fuse_views` (TSDF), `clean_measured_surface_mesh`, `_save_views` | → `K_depth` |
| `generate_scan_targets` ~2512 | `plan_scan` framing standoff, coverage prediction (image-framing) | **keep `K`** |
| `_generate_tiled_scan_targets` ~2313 | `plan_rect_tour` / `_tile_grid_dims` (image-framing) | **keep `K`** |
| `_project_color_corners_uv` ~311, `_corners_from_boundary_on_plane` ~1104 | colour-pixel projection/rays | **keep `K` + `dist`** |

- [ ] **Step 1: Write the failing wiring test**

Append to `tests/test_scan_job.py`:

```python
def test_lock_backprojects_with_depth_grid_intrinsics():
    """Audit A1: extents measured from ALIGNED depth must come from the factory
    depth-grid model — skewing the ChArUco-calibrated colour K by 10% must not
    move the locked extent at all."""
    services, _state = _build_fakes()
    ext0 = scan_service.lock_scan_surface(services).gate_payload.get("extent_mm")
    services2, _s2 = _build_fakes()
    cam = services2.config.camera
    skewed = [row[:] for row in cam.intrinsics[cam.resolution]]
    skewed[0][0] *= 1.10
    skewed[1][1] *= 1.10
    cam.intrinsics = {**cam.intrinsics, cam.resolution: skewed}
    ext1 = scan_service.lock_scan_surface(services2).gate_payload.get("extent_mm")
    assert ext0 is not None and ext1 is not None
    assert ext1 == pytest.approx(ext0, rel=1e-6), (ext0, ext1)
```

(If `_build_fakes` locks in crop mode with no `extent_mm`, use whichever existing lock test in the file produces a framed lock and mirror its fake setup — the assertion stays identical.)

- [ ] **Step 2: Run it to verify it fails**

Run: `py -3.10 -m pytest tests/test_scan_job.py -q -k depth_grid_intrinsics`
Expected: FAIL — the extents differ by ~10% (today both locks back-project with the skewed colour K).

- [ ] **Step 3: Add the config field + property**

In `CameraConfig` (`tasni/core/config.py`), after `dist_coeffs`:

```python
    # Intrinsics of the ALIGNED-DEPTH pixel grid (audit A1). rs.align on the
    # Jetson builds that grid by projecting through the DEVICE-STORED (factory)
    # colour intrinsics with zero distortion, so inverting the grid back to 3D
    # must use the same model — regardless of how good the ChArUco-calibrated
    # `intrinsics` above are as a model of the physical lens. `K`/`dist` stay
    # authoritative for real colour PIXELS (ChArUco, SAM rays, projectPoints).
    # Defaults = the factory values probed off the device (2026-06); re-probe
    # with tools/jetson_intrinsics.py if the camera is ever swapped.
    depth_grid_intrinsics: dict[str, list[list[float]]] = Field(
        default_factory=lambda: {k: [row[:] for row in v]
                                 for k, v in _DEFAULT_INTRINSICS.items()})

    @property
    def K_depth(self) -> np.ndarray:
        """Camera matrix of the aligned-depth grid (factory model, no distortion)."""
        return np.array(self.depth_grid_intrinsics[self.resolution], dtype=np.float64)
```

Note: `tasni.config.json` overrides only `intrinsics`, so `depth_grid_intrinsics` keeps the factory defaults — exactly right.

- [ ] **Step 4: Apply the table**

In `service.py`, change `K = cfg.camera.K` to `K = cfg.camera.K_depth` at the four → sites (386, 428, 1212, 2944). Then read `_reference_locate` (~1421–1484) and apply the rule inside it: any `survey_surface`/backprojection call → `K_depth`; any colour projection with `dist` → unchanged. Then verify coverage:

Run: `grep -n "camera.K\b" tasni/modules/scan/service.py`
Expected: remaining plain-`K` hits are only the keep-sites (2512, 2313, 311, 1104 and any colour-projection line inside `_reference_locate`).

- [ ] **Step 5: Run the tests**

Run: `py -3.10 -m pytest tests/test_scan_job.py -q`
Expected: all pass, including the new wiring test. (Fakes construct default configs where `K == K_depth` numerically for their synthetic frames, so existing expectations hold.)

- [ ] **Step 6: Run the neighbouring suites**

Run: `py -3.10 -m pytest tests/test_five_position.py tests/test_scan_planner.py tests/test_survey_contract.py -q`
Expected: all pass. (`camera_calibration_id` still hashes `K`+`dist` only — leave it; changing it would orphan the 2026-08-13 characterization's `calibration_id`.)

- [ ] **Step 7: Commit + push**

```bash
git add tasni/core/config.py tasni/modules/scan/service.py tests/test_scan_job.py
git commit -m "One camera model per job: factory K_depth for the aligned-depth grid, calibrated K for colour pixels (audit A1)"
git push
```

- [ ] **Step 8 [CELL]: Confirm the factory defaults against the device**

Run: `py -3.10 tools/jetson_intrinsics.py` and compare the reported 1280×720 colour intrinsics to `_DEFAULT_INTRINSICS["1280x720"]` (fx 908.100, cx 650.236). If they differ, update `depth_grid_intrinsics` in `tasni.config.json` with the probed values.

---

### Task 9: Depth-grid known-length metric in the characterization sweep (A1, acceptance test)

**Files:**
- Modify: `tasni/core/characterize.py` (`DistanceTrial` ~line 48), `tools/characterize_distance.py` (capture loop ~lines 370–425)
- Test: `tests/test_characterize.py`

**Interfaces:**
- Consumes: `CameraConfig.K_depth` (Task 8).
- Produces: `DistanceTrial.depth_length_err_mm: float | None = None` (recorded, NOT gated — no `budget` change, `choose_dstar` untouched).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_characterize.py` (mirror the file's existing `DistanceTrial` construction style):

```python
def test_distance_trial_depth_length_field_optional_and_serialized():
    """Audit A1: the depth-grid length error is recorded alongside the
    colour-ray length error; old records without it must still load."""
    t = DistanceTrial(distance_mm=400.0, n_captures=5, plane_rms_mm=1.0,
                      plane_max_mm=4.0, height_repeat_mm=0.02,
                      normal_repeat_deg=0.03, length_err_mm=0.3,
                      coverage_frac=1.0, length_spread_mm=0.02)
    assert t.depth_length_err_mm is None
    assert "depth_length_err_mm" in t.to_dict()
    t2 = replace(t, depth_length_err_mm=8.4)
    assert t2.to_dict()["depth_length_err_mm"] == 8.4
```

(Add `from dataclasses import replace` to the test file imports if absent.)

- [ ] **Step 2: Run it to verify it fails**

Run: `py -3.10 -m pytest tests/test_characterize.py -q -k depth_length`
Expected: FAIL — unexpected keyword / missing attribute.

- [ ] **Step 3: Add the field**

In `tasni/core/characterize.py`, `DistanceTrial`, after `length_spread_mm`:

```python
    # Audit A1: |mean depth-grid diagonal − true| — the SAME two ChArUco corners
    # as length_err_mm, but measured from the ALIGNED DEPTH GRID alone
    # (K_depth pinhole, no colour model anywhere). length_err_mm validates the
    # colour model + depth z; THIS validates the depth grid's lateral scale —
    # the axis the 2026-08-13 sweep could not see (VDI/VDE 2634-2
    # sphere-spacing-error analogue). Optional so pre-2026-08-14 records load.
    depth_length_err_mm: float | None = None
```

Run the test again — Expected: PASS.

- [ ] **Step 4: Measure it in the tool**

In `tools/characterize_distance.py`, in the per-frame loop where the plane-ray corner pair is computed (~line 377, the `pa = _corner_on_plane_mm(...)` branch): also always compute the depth-grid pair using the existing raw-depth fallback helper, with the factory model and NO undistort:

```python
            K_depth = cfg.camera.K_depth
            da = _corner_point_mm(frame.depth, K_depth, *px_by_id[id_a], depth_scale,
                                  dist=None)
            db = _corner_point_mm(frame.depth, K_depth, *px_by_id[id_b], depth_scale,
                                  dist=None)
            if da is not None and db is not None:
                depth_pair_lengths.append(float(np.linalg.norm(da - db)))
```

(Initialise `depth_pair_lengths: list[float] = []` next to `length_samples`; read `_corner_point_mm` at ~line 255 first — its `dist` handling already supports `dist=None` = pinhole; if it doesn't, add that branch there.) After `trial = summarize_distance_trial(...)` (~line 425):

```python
    if depth_pair_lengths:
        depth_err = abs(float(np.mean(depth_pair_lengths)) - true_length_mm)
        trial.depth_length_err_mm = depth_err
        delta = abs(depth_err - trial.length_err_mm)
        if delta > 2.0:
            print(f"   !! depth-grid length error {depth_err:.1f} mm vs colour-ray "
                  f"{trial.length_err_mm:.1f} mm — the two camera models disagree "
                  f"on lateral scale (audit A1). Expect ~2% if the calibrated K "
                  f"was used for the aligned grid.")
```

(`true_length_mm` = whatever variable the file already compares `length_err` against — grep `length_err` in the summarize call chain to get its exact name.)

- [ ] **Step 5: Headless-import check + tests**

Run: `py -3.10 -c "import importlib.util,sys; spec=importlib.util.spec_from_file_location('cd','tools/characterize_distance.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('import ok')"`
Expected: `import ok` (the tool must stay importable with no hardware — everything above `main()` is pure).
Run: `py -3.10 -m pytest tests/test_characterize.py -q` — Expected: all pass.

- [ ] **Step 6: Commit + push**

```bash
git add tasni/core/characterize.py tools/characterize_distance.py tests/test_characterize.py
git commit -m "Characterization: depth-grid known-length metric — the missing lateral-scale acceptance test (audit A1)"
git push
```

- [ ] **Step 7 [CELL]: Re-run the sweep on the cell**

`py -3.10 tools/characterize_distance.py` at 3+ standoffs. Record whether `depth_length_err_mm` ≈ `length_err_mm` (models agree → A1's magnitude was overstated) or ~2% of the diagonal (A1 confirmed). Either result goes in the audit doc.

---

### Task 10: Vectorised `center_connected_mask` in `scan_overlay` (B1, part 1)

**Files:**
- Modify: `server/scan_overlay.py` (add function), `server/server_unicast_syncronous.py:33-77` (replace def with alias)
- Test: `tests/test_scan_overlay.py`

**Interfaces:**
- Produces: `scan_overlay.center_connected_mask(mask) -> np.ndarray[bool]` — same name/signature/semantics as the current server BFS. The server re-exports it (`tests/test_scan_telemetry_server.py` imports it from the server module — that import must keep working).

- [ ] **Step 1: Write the parity test (this is the spec)**

Append to `tests/test_scan_overlay.py`:

```python
def _bfs_reference(mask):
    """The original pure-Python BFS, kept verbatim as the semantic oracle."""
    from collections import deque
    src = np.asarray(mask, dtype=bool)
    h, w = src.shape
    padded = np.pad(src, 1, constant_values=False)
    neighbours = sum(padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    bridged = src | (neighbours >= 5)
    cy, cx = h // 2, w // 2
    ry, rx = max(2, h // 10), max(2, w // 10)
    seeds = np.argwhere(bridged[max(0, cy - ry):min(h, cy + ry + 1),
                                max(0, cx - rx):min(w, cx + rx + 1)])
    if len(seeds):
        seeds[:, 0] += max(0, cy - ry)
        seeds[:, 1] += max(0, cx - rx)
    else:
        pts = np.argwhere(bridged)
        if not len(pts):
            return np.zeros_like(src)
        seeds = pts[np.argmin((pts[:, 0] - cy) ** 2
                              + (pts[:, 1] - cx) ** 2)].reshape(1, 2)
    out = np.zeros_like(src)
    q = deque()
    for y, x in seeds:
        if not out[y, x]:
            out[y, x] = True
            q.append((int(y), int(x)))
    while q:
        y, x = q.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dx or dy):
                    continue
                yy, xx = y + dy, x + dx
                if (0 <= yy < h and 0 <= xx < w
                        and bridged[yy, xx] and not out[yy, xx]):
                    out[yy, xx] = True
                    q.append((yy, xx))
    return out


def test_center_connected_mask_matches_bfs_reference():
    from scan_overlay import center_connected_mask
    rng = np.random.default_rng(7)
    for _ in range(40):
        h, w = rng.integers(8, 60), rng.integers(8, 60)
        mask = rng.random((h, w)) < rng.uniform(0.15, 0.75)
        got = center_connected_mask(mask)
        want = _bfs_reference(mask)
        assert np.array_equal(got, want), (h, w)
    assert not center_connected_mask(np.zeros((20, 20), bool)).any()
```

- [ ] **Step 2: Run to verify it fails**

Run: `py -3.10 -m pytest tests/test_scan_overlay.py -q -k bfs_reference`
Expected: FAIL — `ImportError: cannot import name 'center_connected_mask'`.

- [ ] **Step 3: Implement in `scan_overlay.py`**

```python
def center_connected_mask(mask):
    """Keep the 8-connected plane component crossing the image-center reticle.

    Vectorised replacement for the server's original per-cell BFS (audit B1):
    that flood fill cost ~112 ms per frame on a workstation and an estimated
    ~1-1.7 s on the Nano — the dominant term in the measured 3.4 s telemetry
    cadence. Same semantics (parity-tested against the BFS): bridge isolated
    invalid-depth pinholes (>=5 of 8 neighbours), seed from the reticle window
    (or the nearest bridged cell), then grow to the fixpoint by masked
    dilation — each pass is 9 shifted ORs over a bool array, microseconds at
    the strided grid size."""
    src = np.asarray(mask, dtype=bool)
    h, w = src.shape
    padded = np.pad(src, 1, constant_values=False)
    neighbours = sum(padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    bridged = src | (neighbours >= 5)
    out = np.zeros_like(src)
    cy, cx = h // 2, w // 2
    ry, rx = max(2, h // 10), max(2, w // 10)
    win = (slice(max(0, cy - ry), min(h, cy + ry + 1)),
           slice(max(0, cx - rx), min(w, cx + rx + 1)))
    out[win] = bridged[win]
    if not out.any():
        pts = np.argwhere(bridged)
        if not len(pts):
            return out
        y0, x0 = pts[np.argmin((pts[:, 0] - cy) ** 2 + (pts[:, 1] - cx) ** 2)]
        out[y0, x0] = True
    while True:
        p = np.pad(out, 1, constant_values=False)
        grown = (p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:]
                 | p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:]
                 | p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:])
        grown &= bridged
        if int(grown.sum()) == int(out.sum()):
            return grown & bridged if out.any() else out
        out = grown
```

Note the return: `grown` at fixpoint equals the connected component (already `& bridged`); returning it directly is fine — simplify to `return out` after the loop breaks with `out = grown` if you prefer, but keep parity green.

- [ ] **Step 4: Replace the server's def with the import**

In `server_unicast_syncronous.py`, delete the whole `def center_connected_mask...` (lines 33–77) and, below the `import scan_overlay` line, add:

```python
# Vectorised in scan_overlay (audit B1) — re-exported under the old name so the
# host test suite's `from server.server_unicast_syncronous import ...` holds.
center_connected_mask = scan_overlay.center_connected_mask
```

Also delete `from collections import deque` if now unused (grep `deque` — `stream_burst` doesn't use it; only the old BFS did).

- [ ] **Step 5: Run the tests**

Run: `py -3.10 -m pytest tests/test_scan_overlay.py tests/test_scan_telemetry_server.py -q`
Expected: all pass (parity + the existing center-connected behaviour tests through the server import).

- [ ] **Step 6: Commit + push + deploy**

```bash
git add server/scan_overlay.py server/server_unicast_syncronous.py tests/test_scan_overlay.py
git commit -m "Vectorise center_connected_mask (~1 s/frame on the Nano -> ms) with a BFS parity test (audit B1)"
git push
py -3.10 tools/jetson_deploy.py deploy
py -3.10 tools/jetson_deploy.py status
```

---

### Task 11: Crop-square size over the handshake (C1)

**Files:**
- Modify: `server/server_unicast_syncronous.py` (`scan_plane_telemetry` signature ~229 + WORK_CROP_MM uses ~486/538; handshake ~753–768; `stream_h264` signature ~850 + telemetry call ~962), `tasni/core/camera.py` (`_request_color_only` ~119, `stream` ~177), `tasni/core/livepreview.py` (`start` ~50), `tasni/modules/scan/module.py` (live_start kwargs ~757, stale comment ~132)
- Test: `tests/test_scan_telemetry_server.py`, `tests/test_scan_job.py`

**Interfaces:**
- Produces: handshake tokens `CW<mm> CH<mm>` (ints, clamped 100–4000, ignored by old servers); `scan_plane_telemetry(..., work_crop_mm=None)` kwarg; `CameraClient.stream(..., crop_mm=None)`; `LivePreview.start(..., crop_mm=None)`. Task 12 must preserve `work_crop_mm` when it moves the telemetry call into the worker.

- [ ] **Step 1: Write the failing server-side test**

Append to `tests/test_scan_telemetry_server.py`, mirroring the file's existing overrun-scene fixture (grep `surface_mode` in the file for the test that produces `"crop"`):

```python
def test_crop_square_size_follows_work_crop_mm():
    """Audit C1: the live generic square must be the host's configured region
    size, not a hardcoded 1000 mm."""
    depth, intr = _overrun_scene()   # reuse/extract the existing overrun fixture
    p = scan_plane_telemetry(depth, intr, work_crop_mm=(600.0, 600.0))
    assert p["surface_mode"] == "crop"
    assert p["rectangle_size_mm"] == [600.0, 600.0]
```

If no reusable overrun fixture exists, build one exactly like the file's existing full-frame-plane test but with the plane filling the whole frame (touching borders ⇒ `depth_fully_framed` False).

- [ ] **Step 2: Run to verify it fails**

Run: `py -3.10 -m pytest tests/test_scan_telemetry_server.py -q -k work_crop`
Expected: FAIL — unexpected keyword `work_crop_mm`.

- [ ] **Step 3: Server — parameter + handshake**

1. `scan_plane_telemetry(..., color_image=None, work_crop_mm=None)`; first lines of the body:

```python
    crop_w, crop_h = ((float(work_crop_mm[0]), float(work_crop_mm[1]))
                      if work_crop_mm else (WORK_CROP_MM, WORK_CROP_MM))
```

Replace `(WORK_CROP_MM, WORK_CROP_MM)` at ~486 with `(crop_w, crop_h)` and `[WORK_CROP_MM, WORK_CROP_MM]` at ~538 with `[crop_w, crop_h]`. Keep the module constant as the default + update its comment (now a fallback for clients that don't send CW/CH).
2. In `handle_client`'s token loop (~760), add (AFTER the existing `if tok == b'H264'` branch, so `H264` can never match a prefix rule):

```python
            elif tok.startswith(b'CW') and tok[2:].isdigit():
                crop_w_req = max(100, min(4000, int(tok[2:])))
            elif tok.startswith(b'CH') and tok[2:].isdigit():
                crop_h_req = max(100, min(4000, int(tok[2:])))
```

with `crop_w_req = crop_h_req = None` initialised beside `color_only = False`, and include both in the connection `print`.
3. Thread it: `stream_h264(conn, addr, width, height, h264_bitrate, scan_telemetry=scan_telemetry, work_crop_mm=(float(crop_w_req or WORK_CROP_MM), float(crop_h_req or WORK_CROP_MM)))`; add the parameter to `stream_h264` and pass `work_crop_mm=work_crop_mm` into the `scan_plane_telemetry(...)` call inside the feeder.

- [ ] **Step 4: Host — thread `crop_mm` to the handshake**

1. `camera.py` `_request_color_only(sock, quality=None, codec="jpeg", bitrate=None, scan_telemetry=False, crop_mm=None)` — after the `if scan_telemetry: msg += b" SCAN"` line:

```python
        if scan_telemetry and crop_mm is not None:
            msg += (f" CW{int(round(float(crop_mm[0])))}"
                    f" CH{int(round(float(crop_mm[1])))}").encode()
```

2. `camera.py` `stream(..., scan_telemetry: bool = False, crop_mm=None)` — pass `crop_mm=crop_mm` into its `_request_color_only` call.
3. `livepreview.py` `start(..., scan_telemetry: bool = False, crop_mm=None)` — forward it exactly where `scan_telemetry` is forwarded into `CameraClient.stream` (grep `scan_telemetry` in the file for both spots).
4. `module.py` live_start kwargs dict (~757): add `crop_mm=tuple(float(v) for v in sc.work_crop_mm),`. Replace the stale comment at ~132 ("display-only and intentionally left out of sync") with: the Jetson square now follows `scan.work_crop_mm` via the CW/CH handshake; a region change via `POST /surface/region` applies to the overlay on the next live-preview start.

- [ ] **Step 5: Host-side handshake test**

Append to `tests/test_scan_job.py` (or a more fitting camera test file if one imports `CameraClient` — grep `_request_color_only` in tests first):

```python
def test_color_handshake_carries_crop_size():
    import socket as _socket
    from tasni.core.camera import CameraClient
    a, b = _socket.socketpair()
    try:
        CameraClient._request_color_only(a, 60, codec="h264", bitrate=4000,
                                         scan_telemetry=True, crop_mm=(600.0, 600.0))
        sent = b.recv(128)
    finally:
        a.close(); b.close()
    assert sent == b"MODE COLOR H264 B4000 SCAN CW600 CH600\n", sent
```

(If `socketpair` is unavailable on this Windows Python, use a localhost `socket.create_server`/connect pair — assert the same bytes.)

- [ ] **Step 6: Run the tests**

Run: `py -3.10 -m pytest tests/test_scan_telemetry_server.py -q` and `py -3.10 -m pytest tests/test_scan_job.py -q -k handshake_carries_crop`
Expected: all pass.

- [ ] **Step 7: Commit + push + deploy**

```bash
git add server/server_unicast_syncronous.py tasni/core/camera.py tasni/core/livepreview.py tasni/modules/scan/module.py tests/test_scan_telemetry_server.py tests/test_scan_job.py
git commit -m "Live crop square follows scan.work_crop_mm via CW/CH handshake — no more 1000 mm lie (audit C1)"
git push
py -3.10 tools/jetson_deploy.py deploy
py -3.10 tools/jetson_deploy.py status
```

---

### Task 12: Telemetry off the video thread + JPEG-path telemetry (B1 + C2)

**Files:**
- Modify: `server/server_unicast_syncronous.py` (`stream_h264` feeder ~891–989; `handle_client` color-only loop ~801–813; `SCAN_TELEMETRY_PERIOD_S` ~18)

**Interfaces:**
- Consumes: `scan_plane_telemetry(..., work_crop_mm=...)` (Task 11 — preserve it), `scan_overlay.center_connected_mask` (Task 10).
- Produces: `ScanTelemetryWorker` class (server module level) with `submit(depth_np, color_np)` / `stop()`; `_build_scan_calib(pipeline, work_crop_mm) -> dict`. The published payload schema is UNCHANGED — the host needs no edits.

- [ ] **Step 1: Build the static calibration once (kills the per-cycle `rs.pointcloud`)**

Add at module level (below `publish_scan_telemetry`):

```python
def _build_scan_calib(pipeline, work_crop_mm):
    """Everything scan telemetry needs that NEVER changes while the pipeline
    runs (audit B1): intrinsics, depth->color extrinsics, and the projection
    closures. Built ONCE per stream instead of per cycle — the old feeder
    recomputed all of it every second, including a full-frame rs.pointcloud
    (an 11 MB vertex array) whose only job was an orientation self-check."""
    prof = pipeline.get_active_profile()
    depth_profile = prof.get_stream(rs.stream.depth).as_video_stream_profile()
    color_profile = prof.get_stream(rs.stream.color).as_video_stream_profile()
    intr = depth_profile.intrinsics
    color_intr = color_profile.intrinsics
    depth_to_color = depth_profile.get_extrinsics_to(color_profile)
    R_dc = np.asarray(depth_to_color.rotation, dtype=float).reshape(3, 3)
    t_dc_mm = np.asarray(depth_to_color.translation, dtype=float) * 1000.0

    if any(abs(c) > 1e-9 for c in intr.coeffs):
        print("WARNING: depth stream reports nonzero distortion; pinhole "
              "deprojection below is approximate", flush=True)

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

    # Row/column-major self-check ONCE, against the SDK's exact transform,
    # on synthetic camera-frame points (no pointcloud needed).
    sample = np.array([[0.0, 0.0, 500.0], [100.0, 50.0, 600.0],
                       [-80.0, 40.0, 700.0], [30.0, -60.0, 800.0]])
    sdk_px = np.asarray([overlay_project(p) for p in sample], float)
    cand_a = project_color_points(sample @ R_dc.T + t_dc_mm)
    cand_b = project_color_points(sample @ R_dc + t_dc_mm)
    R_vec = R_dc.T if (np.nanmean(np.linalg.norm(cand_a - sdk_px, axis=1))
                       <= np.nanmean(np.linalg.norm(cand_b - sdk_px, axis=1))) else R_dc

    def overlay_project_points(points):
        return project_color_points(np.asarray(points, float) @ R_vec + t_dc_mm)

    def overlay_transform_points(points):
        return np.asarray(points, float) @ R_vec + t_dc_mm

    def depth_deproject_points(pixels, depths_mm):
        uv = np.asarray(pixels, float).reshape(-1, 2)
        z = np.asarray(depths_mm, float).reshape(-1)
        return np.column_stack([
            (uv[:, 0] - float(intr.ppx)) / float(intr.fx) * z,
            (uv[:, 1] - float(intr.ppy)) / float(intr.fy) * z, z])

    return dict(intr=intr, overlay_project=overlay_project,
                overlay_project_points=overlay_project_points,
                overlay_transform_points=overlay_transform_points,
                depth_deproject_points=depth_deproject_points,
                overlay_size=(color_profile.width(), color_profile.height()),
                work_crop_mm=work_crop_mm)
```

- [ ] **Step 2: The worker**

```python
class ScanTelemetryWorker:
    """Compute scan telemetry OFF the video feeder thread (audit B1). The
    feeder submits the newest frames; if a computation is still running the
    slot is simply overwritten, so the worker always works on the freshest
    frame and self-paces to what the Nano sustains — the feeder never blocks
    and the encoder never starves."""

    def __init__(self, calib):
        self._calib = calib
        self._lock = threading.Lock()
        self._slot = None
        self._event = threading.Event()
        self._halt = threading.Event()
        self._thread = threading.Thread(target=self._loop,
                                        name="scan-telemetry-worker", daemon=True)
        self._thread.start()

    def submit(self, depth_np, color_np):
        with self._lock:
            self._slot = (depth_np, color_np)
        self._event.set()

    def stop(self):
        self._halt.set()
        self._event.set()

    def _loop(self):
        while not self._halt.is_set():
            self._event.wait(timeout=1.0)
            self._event.clear()
            with self._lock:
                item, self._slot = self._slot, None
            if item is None:
                continue
            depth_np, color_np = item
            c = self._calib
            try:
                payload = scan_plane_telemetry(
                    depth_np, c['intr'], depth_unit_mm,
                    overlay_project=c['overlay_project'],
                    overlay_project_points=c['overlay_project_points'],
                    overlay_transform_points=c['overlay_transform_points'],
                    depth_deproject_points=c['depth_deproject_points'],
                    overlay_size=c['overlay_size'],
                    color_image=color_np,
                    work_crop_mm=c['work_crop_mm'])
                publish_scan_telemetry(payload)
            except Exception as e:
                publish_scan_telemetry({"detected": False, "valid_frac": 0.0,
                                        "error": str(e), "timestamp": time.time()})
```

(`depth_unit_mm` is the existing module global set in `__main__` — same one the old inline call used.)

- [ ] **Step 3: Rewrite the feeder's telemetry branch**

In `stream_h264`, before starting the feeder: `worker = ScanTelemetryWorker(_build_scan_calib(pipeline, work_crop_mm)) if scan_telemetry else None`. Replace the entire `if scan_telemetry and ...:` block inside `feeder()` (lines ~899–975 — everything from `depth = frames.get_depth_frame()` through the exception handler) with:

```python
                if (worker is not None
                        and time.monotonic() - last_telemetry >= SCAN_TELEMETRY_PERIOD_S):
                    depth = frames.get_depth_frame()
                    if depth:
                        worker.submit(np.asanyarray(depth.get_data()).copy(),
                                      np.asanyarray(color.get_data()).copy())
                    last_telemetry = time.monotonic()
```

In `stream_h264`'s `finally:` add `if worker is not None: worker.stop()`. Change `SCAN_TELEMETRY_PERIOD_S = 1.0` to `0.25` with the comment: `# submit cadence; the worker self-paces below this if the Nano can't keep up (audit B1)`.

- [ ] **Step 4: Wire the JPEG preview path (C2)**

In `handle_client`, before the `while True:` stream loop: `worker = ScanTelemetryWorker(_build_scan_calib(pipeline, (float(crop_w_req or WORK_CROP_MM), float(crop_h_req or WORK_CROP_MM)))) if (scan_telemetry and color_only) else None` and `last_telemetry = 0.0`. Inside the `if color_only:` branch, after `color = np.asanyarray(color_frame.get_data())`:

```python
            if (worker is not None
                    and time.monotonic() - last_telemetry >= SCAN_TELEMETRY_PERIOD_S):
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    worker.submit(np.asanyarray(depth_frame.get_data()).copy(),
                                  color.copy())
                last_telemetry = time.monotonic()
```

Wrap the loop in `try: ... finally: (worker.stop() if worker else None); conn.close()` (the function currently ends with a bare `conn.close()`). This makes `MODE COLOR Q60 SCAN` (the `preview_codec="jpeg"` config) produce telemetry instead of silently nothing.

- [ ] **Step 5: Host-side import + telemetry tests**

Run: `py -3.10 -m pytest tests/test_scan_telemetry_server.py tests/test_scan_overlay.py -q`
Expected: all pass (the stubs tolerate the new module-level class — it references `rs` only inside functions).
Run: `py -3.10 -m py_compile server/server_unicast_syncronous.py` — Expected: silent.

- [ ] **Step 6: Commit + push + deploy**

```bash
git add server/server_unicast_syncronous.py
git commit -m "Jetson: scan telemetry in a worker thread at 4 Hz submit; JPEG preview gets telemetry too (audit B1 + C2)"
git push
py -3.10 tools/jetson_deploy.py deploy
py -3.10 tools/jetson_deploy.py status
```

- [ ] **Step 7: End-to-end cadence check against the live camera (no robot needed)**

Save + run this probe (scratchpad, not the repo):

```python
# probe_telemetry.py — count distinct telemetry stamps over 20 s
import time
from tasni.core.config import load_config
from tasni.core.camera import CameraClient
cfg = load_config()
cam = CameraClient(cfg.camera)
stamps = set()
with cam.stream(color_only=True, quality=60, scan_telemetry=True,
                crop_mm=tuple(cfg.camera and cfg.scan.work_crop_mm)) as s:
    t0 = time.time()
    while time.time() - t0 < 20:
        f = s.read()
        t = getattr(f, "telemetry", None)
        if t and t.get("timestamp"):
            stamps.add(t["timestamp"])
print(f"{len(stamps)} distinct payloads in 20 s -> {len(stamps)/20:.1f} Hz")
```

(Adjust the config import to the repo's actual loader — grep `def load_config` in `tasni/core/config.py`; `cfg.scan.work_crop_mm` is the crop tuple.) Expected: **≥ 2 Hz** (was ~0.3 Hz). This also proves C2 (telemetry over the JPEG path). If the camera is busy (app running), stop the app's preview first.

---

### Task 13: One plane-selection rule — reticle-seeded, connected (A3)

**Files:**
- Modify: `tasni/modules/scan/survey.py` (`SurveyThresholds`, step 3 of `survey_surface`), `tasni/modules/scan/service.py` (`_survey_thresholds` ~734)
- Test: `tests/test_scan_job.py` (or wherever `survey_surface` unit tests live — grep `survey_surface(` in `tests/` first and use that file)

**Interfaces:**
- Consumes: `scan_overlay.center_connected_mask` semantics (Task 10) — duplicated textually per repo pattern, with a parity test.
- Produces: `SurveyThresholds.center_patch_frac: float = 0.25`; `survey.py` module functions `_fit_nearest_plane_mm(points_mm, *, distance_mm, min_inlier_frac=0.12, iterations=160, seed=7)` and `_center_connected_mask(mask)` (textual copies of the server's `fit_nearest_plane` / vectorised `center_connected_mask`).

- [ ] **Step 1: Write the failing behaviour test**

```python
def test_survey_surface_anchors_to_the_reticle_plane():
    """Audit A3: a small platform under the reticle must win over a dominant
    background plane — the live Jetson rule, now shared by the lock."""
    from tasni.modules.scan.survey import SurveyThresholds, survey_surface
    K = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]])
    depth = np.full((720, 1280), 1200, np.uint16)          # dominant floor
    depth[252:468, 448:832] = 600                          # platform under reticle
    m = survey_surface(depth, K, SurveyThresholds())
    assert m.detected
    assert abs(m.standoff_mm - 600.0) < 15.0, m.standoff_mm   # NOT the floor's 1200
    assert max(m.extent_mm) < 330.0, m.extent_mm  # platform (~256x144mm), not room-sized
```

- [ ] **Step 2: Run to verify it fails**

Run: `py -3.10 -m pytest <chosen test file> -q -k anchors_to_the_reticle`
Expected: FAIL — standoff ≈ 1200 (the dominant floor wins today).

- [ ] **Step 3: Implement**

In `survey.py`:
1. Add `center_patch_frac: float = 0.25` to `SurveyThresholds` with the comment `# reticle window used to SEED plane selection (audit A3) — mirrors ScanConfig.center_patch_frac`.
2. Add `_fit_nearest_plane_mm` — copy `fit_nearest_plane` from `server/server_unicast_syncronous.py:181-219` verbatim (it is already pure numpy, mm units), renamed, with the header comment `# Kept textually identical to server/server_unicast_syncronous.fit_nearest_plane (audit A3) — the live aiming plane and the lock plane must be the same plane.` Use `distance_mm=th.ransac_distance_mm` at the call site.
3. Add `_center_connected_mask` — copy the vectorised function from `server/scan_overlay.py` (Task 10) verbatim, same "kept textually identical" comment.
4. In `survey_surface` step 3, replace the whole-frame `fit_plane` call:

```python
    # 3. Reticle-anchored plane selection (audit A3): fit the NEAREST coherent
    # plane through the CENTER PATCH, then keep only inliers 8-connected to the
    # reticle — the same rule as the Jetson live server, so the plane the
    # operator aims at is the plane the lock measures. Whole-frame largest-plane
    # RANSAC remains only as the fallback when the patch has too little depth.
    patch = float(th.center_patch_frac)
    in_patch = ((np.abs(xs - W / 2.0) <= W * patch / 2.0)
                & (np.abs(ys - H / 2.0) <= H * patch / 2.0))
    try:
        if int(in_patch.sum()) >= 60:
            normal, centroid, _ = _fit_nearest_plane_mm(
                pts_mm[in_patch], distance_mm=th.ransac_distance_mm)
        else:
            normal, centroid, _ = fit_plane(pts_mm, distance=th.ransac_distance_mm)
    except ValueError:
        return _not_detected(th, fov_deg)
```

5. After the existing inlier re-selection (`inlier_mask = dist < th.ransac_distance_mm`), add connectivity before the `< 8` count check:

```python
    # Connectivity: drop coplanar-but-disjoint regions (another table at the
    # same height across the room stays out of the extent).
    GRIDC_W, GRIDC_H = 160, 120
    gx = np.clip((xs / W * GRIDC_W).astype(int), 0, GRIDC_W - 1)
    gy = np.clip((ys / H * GRIDC_H).astype(int), 0, GRIDC_H - 1)
    occ = np.zeros((GRIDC_H, GRIDC_W), bool)
    occ[gy[inlier_mask], gx[inlier_mask]] = True
    connected = _center_connected_mask(occ)
    inlier_mask &= connected[gy, gx]
```

In `service.py` `_survey_thresholds` (~734), add `center_patch_frac=float(scfg.center_patch_frac),`.

- [ ] **Step 4: Add the cross-file parity tests**

In the same test file:

```python
def test_survey_plane_helpers_match_server_textually():
    """The repo's duplication contract: host copy == server copy, behaviourally."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
    import scan_overlay
    from tasni.modules.scan import survey as ssur
    rng = np.random.default_rng(3)
    for _ in range(20):
        m = rng.random((40, 50)) < 0.4
        assert np.array_equal(ssur._center_connected_mask(m),
                              scan_overlay.center_connected_mask(m))
```

- [ ] **Step 5: Run the affected suites**

Run: `py -3.10 -m pytest tests/test_scan_job.py tests/test_five_position.py tests/test_scan_planner.py -q`
Expected: all pass. Single-plane fakes are unaffected (the nearest plane through the patch IS the dominant plane there). If a multi-plane test that previously asserted background-plane selection now fails, that test was asserting the A3 defect — update its expectation and say so in the commit body.

- [ ] **Step 6: Commit + push**

```bash
git add tasni/modules/scan/survey.py tasni/modules/scan/service.py tests/
git commit -m "Lock measures the plane the operator aims at: reticle-seeded + connected selection (audit A3)"
git push
```

- [ ] **Step 7 [CELL]: Corner-capture sanity**

The five-position corner steps also route through `survey_surface`; the reticle sits ON the corner, so the nearest patch plane = the table (this FIXES the documented "RANSAC locks onto the floor" hazard in `five_position_capture`'s precondition comment — update that comment if the cell run confirms). Verify one guided survey end to end.

---

# Phase 3 — capture-chain efficiency

### Task 14: Post-motion frame flush in burst capture (A5)

**Files:**
- Modify: `server/server_unicast_syncronous.py` (`stream_burst` ~1034–1114)

**Interfaces:**
- Produces: no API change; CAP behaviour gains a pre-capture flush + per-pose fresh temporal filter.

- [ ] **Step 1: Edit `stream_burst`**

At the top of the function body add `filters = setup_depth_filters()` and `last_cap = 0.0`. In the `if cmd == b'CAP':` branch, BEFORE the grab loop:

```python
            if cmd == b'CAP':
                # Audit A5: the pipeline holds the most recent frame — possibly
                # captured before or during the robot's move — and the temporal
                # filter carries history from the PREVIOUS pose's scene. On the
                # first CAP of a new pose (>1.5 s since the last), (a) discard a
                # few RAW framesets (bare wait_for_frames only — getFrames runs
                # align+filters at ~1 s/frame on the Nano; a raw discard is
                # ~33 ms), and (b) rebuild the filters so the temporal history
                # is empty instead of a blend with the previous pose.
                if time.monotonic() - last_cap > 1.5:
                    for _ in range(4):
                        try:
                            pipeline.wait_for_frames()
                        except Exception:
                            break
                    filters = setup_depth_filters()
                last_cap = time.monotonic()
```

and change the grab loop's `getFrames(pipeline, align, depth_filters)` to `getFrames(pipeline, align, filters)` (the module-global `depth_filters` stays for the plain-stream paths).

- [ ] **Step 2: Compile + host tests**

Run: `py -3.10 -m py_compile server/server_unicast_syncronous.py` then `py -3.10 -m pytest tests/test_scan_telemetry_server.py -q`
Expected: clean / all pass.

- [ ] **Step 3: Commit + push + deploy**

```bash
git add server/server_unicast_syncronous.py
git commit -m "Burst CAP: flush raw frames + fresh temporal filter per pose (audit A5)"
git push
py -3.10 tools/jetson_deploy.py deploy
py -3.10 tools/jetson_deploy.py status
```

---

### Task 15: One connection per authoritative measurement (B2)

**Files:**
- Modify: `tasni/core/camera.py` (new `grab_many` after `grab` ~line 174), `server/server_unicast_syncronous.py` (`handle_client` SNAP token + full loop), `tasni/modules/scan/service.py` (`_authoritative_acquisition` ~392–400)
- Test: `tests/test_scan_job.py` (or the camera test file found in Task 11 Step 5)

**Interfaces:**
- Produces: `CameraClient.grab_many(n: int, *, with_depth: bool = True, timeout: float | None = None, min_interval_s: float = 0.35) -> list[Frame]`; handshake line `MODE FULL SNAP<n>` (old servers ignore it and just stream — still compatible). `_authoritative_acquisition` uses `grab_many` when present (`getattr`), so test fakes without it keep working.

- [ ] **Step 1: Server — SNAP token**

In `handle_client`: initialise `snap_n = None` beside `color_only`; in the token loop add `elif tok.startswith(b'SNAP') and tok[4:].isdigit(): snap_n = max(1, min(64, int(tok[4:])))`. In the full-stream `while True:` loop, add `sent = 0` before the loop and after a successful `conn.sendall(frame_data)`:

```python
        sent += 1
        if snap_n is not None and sent >= snap_n:
            print(f"SNAP: sent {sent} frame(s) to {addr}; closing cleanly")
            break
```

Comment on the handshake doc block: `MODE FULL SNAP<n>` = full stream that stops after n frames, so a one-shot client's disconnect stops being a logged broken pipe (audit B2).

- [ ] **Step 2: Client — `grab_many`**

In `camera.py`, after `grab`:

```python
    def grab_many(self, n: int, *, with_depth: bool = True,
                  timeout: float | None = None,
                  min_interval_s: float = 0.35) -> "list[Frame]":
        """Read up to ``n`` depth+color frames over ONE connection (audit B2).

        Sends ``MODE FULL SNAP<n>`` so a new server stops after n frames and
        the socket closes cleanly (the five separate connect/close cycles per
        lock filled the Jetson journal with broken-pipe lines); an old server
        ignores the line and streams — we read n and close, which is exactly
        the old behaviour.

        ``min_interval_s`` keeps KEPT frames at least this far apart by
        capture timestamp (RealSense stamps are ms), discarding in-between
        ones: back-to-back frames share temporal-filter state, so a median
        over them buys almost nothing — the length_spread==0 lesson recorded
        in tools/characterize_distance.py. Over cell Wi-Fi a full frame takes
        ~6-11 s to transfer, so this normally discards nothing; on a fast
        link it does the decorrelation the slow connects used to do by
        accident. May return fewer than ``n`` frames."""
        cfg = self.config
        frames: list[Frame] = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(cfg.timeout_s if timeout is None else timeout)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            _set_nodelay(s)
            try:
                s.connect((cfg.ip, cfg.port))
                s.sendall(b"MODE FULL SNAP%d\n" % int(n))
            except socket.timeout as e:
                raise CameraError(f"camera timeout ({cfg.ip}:{cfg.port})") from e
            except OSError as e:
                raise CameraError(f"camera socket error: {e}") from e
            last_ts = None
            for _ in range(int(n)):
                fr = self._read_frame(s, with_depth)
                if (last_ts is None
                        or fr.timestamp - last_ts >= min_interval_s * 1000.0):
                    frames.append(fr)
                    last_ts = fr.timestamp
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if not frames:
            raise CameraError("no frames received from the camera server")
        return frames
```

- [ ] **Step 3: Use it in the acquisition core**

In `_authoritative_acquisition` (service.py ~392), replace the grab loop inside `_camera_hold` with:

```python
    frames = []
    with _camera_hold(services, owner):
        grab_many = getattr(services.camera, "grab_many", None)
        if grab_many is not None:
            try:
                frames = [fr for fr in grab_many(
                              measure_frames, timeout=scfg.grab_timeout_s)
                          if fr.depth is not None]
            except CameraError:
                frames = []
        if not frames:            # legacy fallback: fakes without grab_many, or
            for _ in range(measure_frames):     # a transient one-connection failure
                fr = services.camera.grab(with_depth=True, timeout=scfg.grab_timeout_s)
                if fr.depth is not None:
                    frames.append(fr)
```

- [ ] **Step 4: Protocol test with a fake server**

Append to the test file chosen in Task 11 Step 5:

```python
def test_grab_many_single_connection_snap_handshake():
    import socket as _socket
    import struct as _struct
    import threading as _threading
    import io as _io
    import lz4.frame as _lz4
    import cv2 as _cv2
    from tasni.core.camera import CameraClient
    from tasni.core.config import CameraConfig

    depth = np.full((8, 8), 500, np.uint16)
    buf = _io.BytesIO(); np.save(buf, depth)
    depth_raw = _lz4.compress(buf.getvalue())
    ok, jpg = _cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))
    assert ok
    color_raw = jpg.tobytes()

    got = {}
    srv = _socket.create_server(("127.0.0.1", 0))
    port = srv.getsockname()[1]

    def serve():
        conn, _ = srv.accept()
        got["handshake"] = conn.recv(64)
        for i in range(3):
            ts = float(i) * 1000.0        # 1 s apart -> all kept
            conn.sendall(_struct.pack("<I", len(depth_raw))
                         + _struct.pack("<I", len(color_raw))
                         + _struct.pack("<d", ts) + depth_raw + color_raw)
        conn.close(); srv.close()

    t = _threading.Thread(target=serve, daemon=True); t.start()
    cam = CameraClient(CameraConfig(ip="127.0.0.1", port=port, timeout_s=5.0))
    frames = cam.grab_many(3)
    t.join(timeout=5)
    assert got["handshake"] == b"MODE FULL SNAP3\n", got
    assert len(frames) == 3 and frames[0].depth is not None
```

- [ ] **Step 5: Run the tests**

Run: `py -3.10 -m pytest tests/test_scan_job.py -q` (the fakes lack `grab_many` → legacy path exercised; plus the new protocol test)
Expected: all pass.

- [ ] **Step 6: Commit + push + deploy**

```bash
git add tasni/core/camera.py tasni/modules/scan/service.py server/server_unicast_syncronous.py tests/
git commit -m "grab_many + MODE FULL SNAP<n>: one clean connection per measurement (audit B2)"
git push
py -3.10 tools/jetson_deploy.py deploy
py -3.10 tools/jetson_deploy.py status
```

- [ ] **Step 7 [CELL]: Journal check**

After the next real lock: `py -3.10 tools/jetson_deploy.py status` — the log should show ONE `SNAP: sent 5 frame(s) … closing cleanly` line per lock instead of five broken-pipe pairs.

---

### Task 16: Flat surfaces stop using the calibration cone (A7, interim)

**Files:**
- Modify: `tasni/core/config.py` (`ScanConfig.flat_cone_deg` ~469, `distance_jitter` ~446)
- Test: run-only

- [ ] **Step 1: Change the defaults + comments**

```python
    # Audit A7 (interim): 18°/±15% were inherited from the HAND-EYE pose
    # generator, where pose diversity is the objective. For a FLAT surface the
    # 2026-08-13 characterization measured the opposite: incidence costs ~4x
    # what distance costs, and the jitter walks the standoff off d*. Near-
    # fronto-parallel views win. Full fix (deferred, audit §5): route flat
    # surfaces through plan_rect_tour's fronto-parallel tiling; the cone stays
    # for raised/3D objects (raised_cone_deg unchanged).
    flat_cone_deg: float = 6.0           # was 18.0
```

and `distance_jitter: float = 0.0` (was 0.15) with `# audit A7: hold the planned standoff; jitter belonged to calibration`.

- [ ] **Step 2: Run the pose/planner suites; fix any literal expectations**

Run: `py -3.10 -m pytest tests/test_scan_planner.py tests/test_scan_job.py -q`
Expected: pass. If a test asserts the literal 18.0/0.15 (grep `18.0` / `0.15` in those files), update the expectation to the new defaults — the commit body must name each updated test.

- [ ] **Step 3: Commit + push**

```bash
git add tasni/core/config.py tests/
git commit -m "Flat-surface tours: near-fronto-parallel, no distance jitter (audit A7 interim)"
git push
```

- [ ] **Step 4 [CELL]: Compare fused plane RMS before/after on one table scan**

The report JSON (`runs/scan/<stamp>/report.json`, `plane.inlier_frac` + mesh stats) is the comparison artifact.

---

## Deferred — decisions recorded, deliberately NOT tasks

- **B3 (Jetson frame broker):** one capture thread owning the pipeline, per-client latest-frame queues. Correct end-state, but `rs.frame.keep()` semantics on the ancient L4T pyrealsense2 build cannot be validated headless, and a wedged camera server strands the cell. Do it after B1 has soaked on-cell; B1 + the host `CameraLease` cover the practical contention meanwhile. Also fix the stale "single-threaded with listen(1)" comment at `server:770` when this lands.
- **C3 (consolidate the five "framed" tests):** each margin fixed a documented symptom; collapsing them is regression risk with no accuracy payoff. Revisit only after A2 + A3 have settled behaviour on-cell.
- **OpenCV 4.12 / numpy 2 / Open3D 0.19 upgrade:** one coupled migration — `cv2.aruco.CharucoDetector` (`detector.detectBoard(gray)` replaces `detectMarkers` + `interpolateCornersCharuco`; pose via `board.matchImagePoints` + `cv2.solvePnP` replaces the removed `estimatePoseCharucoBoard`), then re-run a full calibration to revalidate. Own session, own plan; Task 7's pins keep fresh installs on today's validated set until then.
- **MobileSAM default (licence):** EdgeSAM weights are S-Lab non-commercial. The Apache-2.0 MobileSAM drops in via `scan.sam_encoder_file`/`sam_decoder_file` + `tools/download_sam.py`, but flipping the default requires the green-mat low-contrast scene on the cell to revalidate `sam_min_score`. Do it in the first on-cell session after this plan.
- **Jetson Nano EOL:** no software fix exists. B1 buys real-time aiming on the Nano; an Orin Nano (or Pi 5 + USB3) removes the ceiling structurally. Hardware decision, user's call.

## Self-review record

- Spec coverage: A1→T8+T9, A2→T1, A3→T13, A4→T2, A5→T14, A6→T3, A7→T16, B1→T10+T12, B2→T15, B3→deferred, C1→T11, C2→T12, C3→deferred, C4→T4, C5→T5, C6→T6, tooling pins→T7, OpenCV/MobileSAM/Nano→deferred. All 16 findings accounted for.
- Post-audit corrections folded in: A6 uses plane-inlier residuals (T3), A5 uses bare `wait_for_frames` + fresh filters (T14), B2 spaces kept frames by timestamp (T15), C3 softened to deferred.
- Known intentional couplings: T12 must preserve T11's `work_crop_mm`; T13 duplicates T10's function textually (parity-tested); T3's helper receives whatever K its caller passes, so T8 automatically upgrades it.
