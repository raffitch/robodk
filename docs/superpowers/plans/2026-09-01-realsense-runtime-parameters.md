# RealSense Settings as Runtime Parameters — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `SET key=value` command on the Jetson camera server's burst protocol that changes safe-tier depth-filter settings at runtime — no deploy, no restart — with every achieved value recorded in the per-connection greeting and every override dying on restart.

**Architecture:** The filter chain becomes a pure function of one mutable module-level dict (`FILTER_SETTINGS`, initialized from env). `SET` validates all-or-nothing, rebuilds the chain from the dict under `_camera_lock`, bumps `_camera_generation` LAST (the existing staleness machinery then retires every session greeted under the old chain), and replies with one JSON line of achieved values. The greeting's `filter_options` grows from one key to the full safe-tier inventory.

**Tech Stack:** Python 3.10 (host tests) / 3.6 on the Jetson (server code — no walrus, no `X | Y` type unions in server/), pytest, pyrealsense2 (faked in tests), systemd service `realsense-camera`.

**Spec:** `docs/superpowers/specs/2026-09-01-realsense-runtime-parameters-design.md` — read it first; every task below cites its sections.

## Global Constraints

- **Python invocation on this Windows box:** `py -3.10` (there is no `python` on PATH). Tests: `py -3.10 -m pytest tests/<file> -v`. NEVER run the full pytest suite — it is too slow; run only the named test files.
- **server/ code runs on the Jetson's Python 3.6.** No f-string `=` specifiers, no `dict | dict` merges, no `str.removeprefix`, no PEP 604 unions in `server/*.py`. (`tasni/` and `tests/` are host 3.10 and may use modern syntax.)
- **Never round-trip source files through PowerShell Get-Content/Set-Content** — it silently mojibakes the repo's UTF-8. Use the Edit/Write tools only.
- **Commit AND push after every task** (repo working agreement — the user reviews from pushed history and the Jetson auto-pulls `main` every ~2 min). This means the Jetson will pick up server changes as you push; that is fine — every intermediate state in this plan is behavior-preserving until Task 5, and Task 7 does the explicit verified deploy.
- **The unit file `server/realsense-camera.service` is NOT edited.** Boot defaults are unchanged (threshold 0.15/1.5 m are today's constants; the new env vars merely make them overridable). The unit file stays the reviewable boot default per spec §3.5.
- **The `-1.0` sentinel means "leave the SDK default alone"** — the existing convention of `RS_SPATIAL_SMOOTH_DELTA`/`RS_LASER_POWER`. Every new optional knob uses it.
- **Provenance rule (spec §3.3):** a knob that can be changed but not recorded in the greeting must not ship. Every task that adds a knob also lands its greeting field in the same task.
- **Do not touch** `_rebuild_pipeline`, `read_frames`, `openPipeline`, or anything in the camera-recovery supervisor. `SET` reuses `_camera_generation` from the outside only.

---

### Task 1: `FILTER_SETTINGS` registry + threshold env vars (behavior-preserving)

The chain config moves from scattered module constants into one dict; thresholds become env-overridable (spec §6.4, test 9). After this task the server behaves byte-identically to today under an empty environment.

**Files:**
- Modify: `server/server_unicast_syncronous.py` (lines ~20-24 threshold constants; ~946-947 spatial env block; ~1682-1711 `setup_depth_filters`)
- Test: `tests/test_server_env.py`

**Interfaces:**
- Consumes: existing `_env_number(name, default) -> float`, existing `setup_depth_filters()`.
- Produces: module global `FILTER_SETTINGS: dict[str, float]` (keys listed below — later tasks depend on these exact names); env-fed `RS_DEPTH_MIN_M`/`RS_DEPTH_MAX_M` module floats; `setup_depth_filters()` reads `FILTER_SETTINGS` instead of the `RS_*` constants.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server_env.py` (the `_chain` helper at line 111 already reloads the module under env and swaps in `_FakeRs`):

```python
# ------------------------------------------------- runtime-parameters, Task 1
# spec 6.4: threshold min/max join the safe tier, with env vars so the knob has
# all three precedence layers (unit file / env / runtime SET) instead of being
# runtime-only.

def test_threshold_env_vars_reach_the_filter(monkeypatch):
    try:
        chain = _chain(monkeypatch, RS_DEPTH_MIN_M="0.2", RS_DEPTH_MAX_M="0.9")
        thr = next(f for f in chain if f.kind == "threshold")
        assert thr.options == {"min_distance": 0.2, "max_distance": 0.9}
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_thresholds_default_to_todays_constants(monkeypatch):
    """No env set must clip exactly as today: every archived take was measured
    under 0.15..1.5 m."""
    try:
        chain = _chain(monkeypatch)
        thr = next(f for f in chain if f.kind == "threshold")
        assert thr.options == {"min_distance": 0.15, "max_distance": 1.5}
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_filter_settings_is_fed_by_env(monkeypatch):
    """FILTER_SETTINGS is the single source the chain is built from; env feeds it
    at import. -1 keeps the existing 'leave the SDK default alone' sentinel."""
    try:
        _chain(monkeypatch, RS_SPATIAL="0", RS_SPATIAL_SMOOTH_DELTA="4")
        assert srv.FILTER_SETTINGS["spatial"] == 0
        assert srv.FILTER_SETTINGS["spatial_smooth_delta"] == 4.0
        assert srv.FILTER_SETTINGS["depth_min_m"] == 0.15
        assert srv.FILTER_SETTINGS["hole_filling"] == -1.0
        assert srv.FILTER_SETTINGS["decimation"] == 0.0
    finally:
        monkeypatch.undo()
        importlib.reload(srv)
```

This also needs the fake's `threshold_filter` to record its constructor args. In `tests/test_server_env.py` replace `_FakeFilter.__init__` and `_FakeRs.threshold_filter`:

```python
class _FakeFilter:
    def __init__(self, kind, options=None):
        self.kind = kind
        self.options = dict(options or {})
```

```python
    @staticmethod
    def threshold_filter(lo, hi):
        return _FakeFilter("threshold", {"min_distance": lo, "max_distance": hi})
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `py -3.10 -m pytest tests/test_server_env.py -v`
Expected: the three new tests FAIL (`RS_DEPTH_MIN_M` env ignored / `FILTER_SETTINGS` missing); all pre-existing tests PASS (the fake change is backward-compatible).

- [ ] **Step 3: Implement**

In `server/server_unicast_syncronous.py`:

3a. Delete lines 21-24 (the `RS_DEPTH_MIN_M`/`RS_DEPTH_MAX_M` constants and their comment — keep `ASFOUND_DIR` on line 20). They must move below `_env_number` (defined ~line 901), which they now call.

3b. Directly after the existing `RS_SPATIAL_SMOOTH_DELTA` line (~947), add:

```python
# Work-volume clip ahead of the spatial filter (audit R5): background depth must
# not be smoothed into surface edges, and nothing may fabricate depth. Env vars
# since 2026-09-01 (runtime-parameters spec 6.4) so the knob has all three
# precedence layers; the defaults are the constants every archived take was
# clipped under.
RS_DEPTH_MIN_M = _env_number('RS_DEPTH_MIN_M', 0.15)
RS_DEPTH_MAX_M = _env_number('RS_DEPTH_MAX_M', 1.5)

# The single source the depth-filter chain is built from. Env feeds it once at
# import (the unit file's boot defaults); a runtime SET (stream_burst) may
# rewrite entries later, which cannot survive a restart -- that impermanence is
# the safety argument of the runtime-parameters spec (4.2). -1.0 = leave the
# SDK's own default alone, the same sentinel RS_SPATIAL_SMOOTH_DELTA uses.
# `decimation` is recorded but pinned at 0: enabling it would change the depth
# geometry the greeting already declared (spec 2.2/2.5), so a SET refuses it.
FILTER_SETTINGS = {
    "spatial":                float(RS_SPATIAL),
    "spatial_smooth_delta":   RS_SPATIAL_SMOOTH_DELTA,
    "spatial_magnitude":      -1.0,
    "spatial_smooth_alpha":   -1.0,
    "spatial_holes_fill":     -1.0,
    "temporal_smooth_alpha":  -1.0,
    "temporal_smooth_delta":  -1.0,
    "temporal_persistency":   -1.0,
    "depth_min_m":            RS_DEPTH_MIN_M,
    "depth_max_m":            RS_DEPTH_MAX_M,
    "hole_filling":           -1.0,
    "decimation":             0.0,
}
```

3c. In `setup_depth_filters()` (~line 1695), replace the two constant reads with the dict — only these two lines change in this task:

```python
    threshold = rs.threshold_filter(FILTER_SETTINGS["depth_min_m"],
                                    FILTER_SETTINGS["depth_max_m"])
```
and
```python
    if FILTER_SETTINGS["spatial"]:
```
and inside the spatial block:
```python
        if FILTER_SETTINGS["spatial_smooth_delta"] >= 0:
            spatial.set_option(rs.option.filter_smooth_delta,
                               FILTER_SETTINGS["spatial_smooth_delta"])
```

3d. `DEPTH_FILTER_NAMES` (~line 953) is derived from `RS_SPATIAL` at import; leave it for now (Task 2 makes it live). But its derivation must now read the dict so import order stays coherent:

```python
DEPTH_FILTER_NAMES = (["threshold", "disparity"]
                      + (["spatial"] if FILTER_SETTINGS["spatial"] else [])
                      + ["temporal", "disparity_inv"])
```

(Place the `FILTER_SETTINGS` block ABOVE this derivation.)

3e. Sanity-check nothing else reads the moved constants: `py -3.10 -m py_compile server/server_unicast_syncronous.py`, then grep the repo for `RS_DEPTH_MIN_M|RS_DEPTH_MAX_M` — expected hits are this file and the new tests only.

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.10 -m pytest tests/test_server_env.py -v`
Expected: ALL PASS (new and pre-existing).

- [ ] **Step 5: Commit + push**

```bash
git add server/server_unicast_syncronous.py tests/test_server_env.py
git commit -m "feat(server): filter chain built from a FILTER_SETTINGS registry; threshold env vars"
git push
```

---

### Task 2: Full safe-tier knob set, achieved read-back, live `DEPTH_FILTER_NAMES`

`setup_depth_filters()` grows the remaining §2.2 knobs (spatial magnitude/alpha/holes, temporal trio, hole-filling filter), applies values through a clamp-to-SDK-range helper, and publishes what it ACTUALLY built into two rebindable globals the greeting will use.

**Files:**
- Modify: `server/server_unicast_syncronous.py` (`setup_depth_filters` and the `SPATIAL_SMOOTH_DELTA` global block ~lines 957-972)
- Test: `tests/test_server_env.py`

**Interfaces:**
- Consumes: `FILTER_SETTINGS` (Task 1).
- Produces: module global `FILTER_OPTIONS: dict` — achieved value per wire key, `float` or `None` (None = filter absent from the chain, or SDK refused the read-back); `DEPTH_FILTER_NAMES: list[str]` rebound by every `setup_depth_filters()` call; helpers `_apply_option(filt, option_name, value)` and `_achieved_filter_options(by_kind) -> dict`; module constant `_OPTION_MAP`. The global `SPATIAL_SMOOTH_DELTA` is REMOVED (replaced by `FILTER_OPTIONS["spatial_smooth_delta"]`).

- [ ] **Step 1: Extend the fake, update the one displaced assertion, write the failing tests**

In `tests/test_server_env.py`, replace `SDK_DEFAULTS` and extend the fake:

```python
# librealsense's own defaults, as a freshly constructed filter reports them. The
# spatial filter's smooth_delta is 20 -- the number the whole archive was measured
# under and the one an "unset" env var silently inherits. Temporal: alpha 0.4,
# delta 20, persistency index 3 (exposed by the SDK as the `holes_fill` option on
# the temporal filter -- three knobs ride that one option name, disambiguated by
# which filter object they are set on).
SDK_DEFAULTS = {
    "spatial":  {"filter_smooth_delta": 20.0, "filter_magnitude": 2.0,
                 "filter_smooth_alpha": 0.5, "holes_fill": 0.0},
    "temporal": {"filter_smooth_alpha": 0.4, "filter_smooth_delta": 20.0,
                 "holes_fill": 3.0},
}

# The option ranges librealsense advertises, for the clamp path (spec test 5).
SDK_RANGES = {
    "spatial":  {"filter_smooth_delta": (1.0, 50.0), "filter_magnitude": (1.0, 5.0),
                 "filter_smooth_alpha": (0.25, 1.0), "holes_fill": (0.0, 5.0)},
    "temporal": {"filter_smooth_alpha": (0.0, 1.0), "filter_smooth_delta": (1.0, 100.0),
                 "holes_fill": (0.0, 8.0)},
}
```

Add to `_FakeFilter` (keep `set_option`/`get_option` as they are):

```python
    def get_option_range(self, option):
        lo, hi = SDK_RANGES[self.kind][option]   # KeyError -> caller's except path
        return SimpleNamespace(min=lo, max=hi)
```

Extend `_FakeRs.option` and add the hole-filling constructor:

```python
    class option:
        filter_smooth_delta = "filter_smooth_delta"
        filter_magnitude = "filter_magnitude"
        filter_smooth_alpha = "filter_smooth_alpha"
        holes_fill = "holes_fill"
        min_distance = "min_distance"
        max_distance = "max_distance"
```

```python
    @staticmethod
    def hole_filling_filter(mode):
        return _FakeFilter("hole_filling", {"holes_fill": float(mode)})
```

Update the read-back-failure test (line ~275): `assert srv.SPATIAL_SMOOTH_DELTA is None` becomes `assert srv.FILTER_OPTIONS["spatial_smooth_delta"] is None`.

Append the new tests:

```python
# ------------------------------------------------- runtime-parameters, Task 2
# spec 2.2: every safe-tier knob applied when set, SDK-default when not, and the
# ACHIEVED values published for the greeting (spec 3.3: a knob that can be
# changed but not recorded must not ship).

def test_every_new_knob_reaches_its_filter(monkeypatch):
    try:
        chain = _chain(monkeypatch, RS_SPATIAL="1")
        srv.FILTER_SETTINGS.update({
            "spatial_magnitude": 3.0, "spatial_smooth_alpha": 0.6,
            "spatial_holes_fill": 1.0, "temporal_smooth_alpha": 0.2,
            "temporal_smooth_delta": 40.0, "temporal_persistency": 5.0,
            "hole_filling": 2.0})
        chain = srv.setup_depth_filters()
        spatial = next(f for f in chain if f.kind == "spatial")
        temporal = next(f for f in chain if f.kind == "temporal")
        hole = next(f for f in chain if f.kind == "hole_filling")
        assert spatial.options == {"filter_magnitude": 3.0, "filter_smooth_alpha": 0.6,
                                   "holes_fill": 1.0}
        assert temporal.options == {"filter_smooth_alpha": 0.2,
                                    "filter_smooth_delta": 40.0, "holes_fill": 5.0}
        assert hole.options == {"holes_fill": 2.0}
        assert [f.kind for f in chain] == ["threshold", "disparity", "spatial",
                                          "temporal", "disparity_inv", "hole_filling"]
        assert srv.DEPTH_FILTER_NAMES == [f.kind for f in chain]
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_untouched_knobs_leave_the_sdk_defaults_alone(monkeypatch):
    """-1 everywhere must reproduce today's chain EXACTLY -- options untouched,
    no hole_filling filter -- because every number in the archive was measured
    under it."""
    try:
        chain = _chain(monkeypatch)
        assert [f.kind for f in chain] == ["threshold", "disparity", "spatial",
                                          "temporal", "disparity_inv"]
        assert next(f for f in chain if f.kind == "spatial").options == {}
        assert next(f for f in chain if f.kind == "temporal").options == {}
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_achieved_options_are_read_back_not_echoed(monkeypatch):
    """FILTER_OPTIONS reports what the filters are AT (SDK defaults when
    untouched), never the -1 sentinel -- the greeting archives these."""
    try:
        _chain(monkeypatch)
        assert srv.FILTER_OPTIONS == {
            "spatial_smooth_delta": 20.0, "spatial_magnitude": 2.0,
            "spatial_smooth_alpha": 0.5, "spatial_holes_fill": 0.0,
            "temporal_smooth_alpha": 0.4, "temporal_smooth_delta": 20.0,
            "temporal_persistency": 3.0,
            "depth_min_m": 0.15, "depth_max_m": 1.5,
            "hole_filling": None, "decimation": 0.0}
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_absent_spatial_reports_none_for_its_options(monkeypatch):
    try:
        _chain(monkeypatch, RS_SPATIAL="0")
        assert srv.FILTER_OPTIONS["spatial_smooth_delta"] is None
        assert srv.FILTER_OPTIONS["spatial_magnitude"] is None
        assert "spatial" not in srv.DEPTH_FILTER_NAMES
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_out_of_range_values_are_clamped_to_the_sdk_range(monkeypatch):
    """spec test 5. The SDK advertises smooth_delta 1..50; a request of 500 must
    land at 50 and the ACHIEVED 50 is what gets recorded."""
    try:
        _chain(monkeypatch)
        srv.FILTER_SETTINGS["spatial_smooth_delta"] = 500.0
        srv.setup_depth_filters()
        assert srv.FILTER_OPTIONS["spatial_smooth_delta"] == 50.0
    finally:
        monkeypatch.undo()
        importlib.reload(srv)
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `py -3.10 -m pytest tests/test_server_env.py -v`
Expected: new tests FAIL (`FILTER_OPTIONS`/knobs missing); the pre-existing greeting tests may also fail on the removed `SPATIAL_SMOOTH_DELTA` only if implementation started — at this point everything pre-existing still PASSES.

- [ ] **Step 3: Implement**

In `server/server_unicast_syncronous.py`:

3a. Replace the `SPATIAL_SMOOTH_DELTA = None` global and its comment block (~lines 957-972) with:

```python
# The ACHIEVED value of every safe-tier knob, read back by setup_depth_filters()
# off the objects that actually process frames; None = that filter is absent from
# the chain, or the SDK would not report the option. Archived in the greeting's
# `filter_options` -- it is the ONLY record of which arm of an A/B a take came
# from (docs/inspection-roll-probe-handoff.md 3.1). Read-back rather than an echo
# of FILTER_SETTINGS because the -1 sentinel names no number at all: the filter
# then runs at librealsense's default, which a future SDK is free to change.
# Rebound (with DEPTH_FILTER_NAMES) by every setup_depth_filters() call -- at
# boot, and on every runtime SET -- always under _camera_lock or before any
# client thread exists, and always BEFORE _camera_generation moves, so a greeting
# can never pair old names with new options.
FILTER_OPTIONS = {}

# wire key -> (chain filter kind, rs.option attribute name). Temporal persistency
# and both hole-fill knobs ride the SDK's one `holes_fill` option, disambiguated
# by which filter object they are set on. threshold and hole_filling take their
# values through their CONSTRUCTORS (handled in setup_depth_filters), so this map
# only drives set_option for spatial/temporal -- but it drives READ-BACK for all.
_OPTION_MAP = {
    "spatial_smooth_delta":  ("spatial",      "filter_smooth_delta"),
    "spatial_magnitude":     ("spatial",      "filter_magnitude"),
    "spatial_smooth_alpha":  ("spatial",      "filter_smooth_alpha"),
    "spatial_holes_fill":    ("spatial",      "holes_fill"),
    "temporal_smooth_alpha": ("temporal",     "filter_smooth_alpha"),
    "temporal_smooth_delta": ("temporal",     "filter_smooth_delta"),
    "temporal_persistency":  ("temporal",     "holes_fill"),
    "depth_min_m":           ("threshold",    "min_distance"),
    "depth_max_m":           ("threshold",    "max_distance"),
    "hole_filling":          ("hole_filling", "holes_fill"),
}
```

3b. Add the two helpers above `setup_depth_filters`:

```python
def _apply_option(filt, option_name, value):
    """Set one filter option, clamped to the SDK's advertised range when it
    exposes one (spec test 5; mirrors rs_config._set_with_readback's discipline:
    a bad value must never take the service down -- the achieved read-back in
    FILTER_OPTIONS is what gets archived, so a clamp is visible, not silent)."""
    option = getattr(rs.option, option_name)
    try:
        rng = filt.get_option_range(option)
        value = min(max(float(value), float(rng.min)), float(rng.max))
    except Exception:
        pass                       # no range API on this build: try the raw value
    try:
        filt.set_option(option, value)
    except Exception as exc:
        print("WARNING: could not set {}={:g} ({}); the filter keeps its "
              "previous value".format(option_name, value, exc), flush=True)


def _achieved_filter_options(by_kind):
    """One achieved value per wire key, read off the filters that will run.
    Guarded per option: provenance is never worth the camera, so a refused
    read-back records None (unknown) and the service keeps serving.
    `decimation` is a constant 0.0 -- this tier cannot enable it (it would
    change the depth geometry the greeting already declared, spec 2.2/2.5), and
    recording the 0 makes "we do not decimate, on purpose" a value instead of an
    absence someone has to infer."""
    def read(kind, option_name):
        filt = by_kind.get(kind)
        if filt is None:
            return None
        try:
            return float(filt.get_option(getattr(rs.option, option_name)))
        except Exception as exc:
            print("WARNING: could not read back {}.{} ({}) -- takes captured now "
                  "will not record it".format(kind, option_name, exc), flush=True)
            return None
    achieved = {}
    for key in _OPTION_MAP:
        kind, option_name = _OPTION_MAP[key]
        achieved[key] = read(kind, option_name)
    achieved["decimation"] = 0.0
    return achieved
```

3c. Rewrite `setup_depth_filters` (replacing the whole function, keeping its chain-order semantics and docstring intent):

```python
def setup_depth_filters():
    """threshold -> disparity -> [spatial] -> temporal -> disparity_inv ->
    [hole_filling], on NATIVE depth, built ENTIRELY from FILTER_SETTINGS.

    No decimation ever (it changes the depth geometry the greeting declares --
    restart path only, spec 2.5) and hole filling only by explicit request: a
    filled pixel is fabricated depth, fabricated exactly where the metrology
    cares (surface edges). Threshold first so background is never smoothed into
    an edge; hole filling last, in the depth domain, per Intel's recommended
    order. -1.0 anywhere = leave that SDK default alone.

    Also rebinds DEPTH_FILTER_NAMES and FILTER_OPTIONS (the achieved values,
    read back off these very objects) so the greeting always describes the chain
    that is actually installed -- see the FILTER_OPTIONS comment block for the
    ordering contract with _camera_generation."""
    global DEPTH_FILTER_NAMES, FILTER_OPTIONS
    s = FILTER_SETTINGS
    by_kind = {"threshold": rs.threshold_filter(float(s["depth_min_m"]),
                                                float(s["depth_max_m"]))}
    chain = [by_kind["threshold"], rs.disparity_transform(True)]
    names = ["threshold", "disparity"]
    if s["spatial"]:
        by_kind["spatial"] = rs.spatial_filter()
        chain.append(by_kind["spatial"])
        names.append("spatial")
    by_kind["temporal"] = rs.temporal_filter()
    chain += [by_kind["temporal"], rs.disparity_transform(False)]
    names += ["temporal", "disparity_inv"]
    if s["hole_filling"] >= 0:
        by_kind["hole_filling"] = rs.hole_filling_filter(int(s["hole_filling"]))
        chain.append(by_kind["hole_filling"])
        names.append("hole_filling")
    for key in _OPTION_MAP:
        kind, option_name = _OPTION_MAP[key]
        if kind in ("threshold", "hole_filling"):
            continue               # constructed with their values above
        if kind in by_kind and s[key] >= 0:
            _apply_option(by_kind[kind], option_name, s[key])
    DEPTH_FILTER_NAMES = names
    FILTER_OPTIONS = _achieved_filter_options(by_kind)
    print("RealSense: depth filters {}, options {}".format(
        DEPTH_FILTER_NAMES, FILTER_OPTIONS), flush=True)
    return chain
```

3d. `make_greeting` (~line 1195) still passes `spatial_smooth_delta=SPATIAL_SMOOTH_DELTA`, which no longer exists — change that ONE argument to `spatial_smooth_delta=FILTER_OPTIONS.get("spatial_smooth_delta")` for now (Task 3 replaces the whole parameter). Grep the file for any other `SPATIAL_SMOOTH_DELTA` stragglers.

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.10 -m pytest tests/test_server_env.py tests/test_rs_geometry.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit + push**

```bash
git add server/server_unicast_syncronous.py tests/test_server_env.py
git commit -m "feat(server): full safe-tier knob set with clamped apply + achieved read-back"
git push
```

---

### Task 3: The greeting carries the full `filter_options`

`build_greeting` takes the whole achieved dict instead of one float, and the host proves old archives still parse (spec §3.3, tests 3 & 8).

**Files:**
- Modify: `server/rs_geometry.py:64-95` (`build_greeting`)
- Modify: `server/server_unicast_syncronous.py` (`make_greeting`, ~line 1192)
- Test: `tests/test_rs_geometry.py`, `tests/test_server_env.py`, `tests/test_depth_geometry.py`

**Interfaces:**
- Consumes: `FILTER_OPTIONS` (Task 2).
- Produces: `build_greeting(static, *, depth_unit_mm, filters, temps, global_time_enabled, achieved, device, filter_options)` — `filter_options: dict` REQUIRED, values float-or-None, emitted verbatim (floated) into the greeting's `filter_options` object. Host `CameraGeometry.from_greeting` is UNCHANGED (it already lifts only `spatial_smooth_delta` and carries the rest in `raw`).

- [ ] **Step 1: Update the two direct-caller tests and add the new ones**

In `tests/test_rs_geometry.py`:
- Line 78: `spatial_smooth_delta=20.0,` → `filter_options={"spatial_smooth_delta": 20.0},`
- Line 115: `_greeting(spatial_smooth_delta=4.0)` → `_greeting(filter_options={"spatial_smooth_delta": 4.0})`
- Line 124: `_greeting(spatial_smooth_delta=None, filters=[...])` → `_greeting(filter_options={"spatial_smooth_delta": None}, filters=[...])`
- Append:

```python
def test_the_greeting_emits_every_safe_tier_knob():
    """Runtime-parameters spec 3.3: a knob that can be changed but not recorded
    must not ship. The greeting emits the whole achieved dict verbatim."""
    full = {"spatial_smooth_delta": 8.0, "spatial_magnitude": 2.0,
            "spatial_smooth_alpha": 0.5, "spatial_holes_fill": 0.0,
            "temporal_smooth_alpha": 0.4, "temporal_smooth_delta": 20.0,
            "temporal_persistency": 3.0, "depth_min_m": 0.15, "depth_max_m": 1.5,
            "hole_filling": None, "decimation": 0.0}
    g = _greeting(filter_options=full)
    assert g["filter_options"] == full
    back = json.loads(rs_geometry.greeting_line(g).decode("utf-8"))
    assert back["filter_options"]["hole_filling"] is None
    assert back["filter_options"]["decimation"] == 0.0
```

In `tests/test_server_env.py`, extend the greeting-path tests — after the existing `test_the_greeting_records_the_smooth_delta...` assertions, add one test:

```python
def test_the_greeting_carries_the_full_achieved_options(monkeypatch):
    try:
        greeting = _greeting(monkeypatch)
        assert greeting["filter_options"] == srv.FILTER_OPTIONS
        assert greeting["filter_options"]["temporal_persistency"] == 3.0
        assert greeting["filter_options"]["depth_max_m"] == 1.5
    finally:
        monkeypatch.undo()
        importlib.reload(srv)
```

In `tests/test_depth_geometry.py`, append:

```python
def test_a_greeting_with_the_full_runtime_filter_options_still_parses():
    """2026-09-01: the server reports EVERY safe-tier knob in filter_options, not
    just spatial_smooth_delta. from_greeting lifts the delta as before and the
    rest ride `raw` into the archive manifest untouched (spec test 8)."""
    base = gf.offset(color_K=K_C, color_size=SIZE_C).raw
    full = {"spatial_smooth_delta": 8.0, "spatial_magnitude": 2.0,
            "spatial_smooth_alpha": 0.5, "spatial_holes_fill": 0.0,
            "temporal_smooth_alpha": 0.4, "temporal_smooth_delta": 20.0,
            "temporal_persistency": 3.0, "depth_min_m": 0.15, "depth_max_m": 1.5,
            "hole_filling": None, "decimation": 0.0}
    g = dg.CameraGeometry.from_greeting({**base, "filter_options": full})
    assert g.spatial_smooth_delta == 8.0
    assert g.to_dict()["filter_options"] == full
```

- [ ] **Step 2: Run tests to verify the new/updated ones fail**

Run: `py -3.10 -m pytest tests/test_rs_geometry.py tests/test_server_env.py tests/test_depth_geometry.py -v`
Expected: the updated `test_rs_geometry` callers FAIL with `unexpected keyword argument 'filter_options'`; the depth_geometry test PASSES already (host is tolerant by design — that is fine, it is a regression guard, note it and move on).

- [ ] **Step 3: Implement**

In `server/rs_geometry.py`, change the signature and the one dict entry:

```python
def build_greeting(static: StaticGeometry, *, depth_unit_mm: float, filters: list,
                   temps: dict, global_time_enabled, achieved: dict, device: dict,
                   filter_options: dict) -> dict:
    """``filters`` names the chain that ran; ``filter_options`` says what it ran AT.

    ``filter_options`` holds the ACHIEVED value of every safe-tier knob, read
    back off the objects that actually run; ``None`` = that filter is absent
    from the chain (or the SDK would not report the option). It is REQUIRED
    rather than defaulted: it is the only record of which arm of an A/B a take
    came from, and a greeting path that forgot it would archive indistinguishable
    arms with no error (docs/inspection-roll-probe-handoff.md 3.1; the
    runtime-parameters spec 3.3 extends the rule to every settable knob).
    """
```

and in the returned dict replace the `"filter_options": {...}` entry with:

```python
        "filter_options": {k: (None if v is None else float(v))
                           for k, v in dict(filter_options).items()},
```

In `server/server_unicast_syncronous.py` `make_greeting`, replace `spatial_smooth_delta=FILTER_OPTIONS.get("spatial_smooth_delta")` with `filter_options=dict(FILTER_OPTIONS)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.10 -m pytest tests/test_rs_geometry.py tests/test_server_env.py tests/test_depth_geometry.py -v`
Expected: ALL PASS. (If the real extrusion archive exists under `runs/`, `test_every_greeting_already_on_disk_still_parses` runs against it — it must stay green.)

- [ ] **Step 5: Commit + push**

```bash
git add server/rs_geometry.py server/server_unicast_syncronous.py tests/test_rs_geometry.py tests/test_server_env.py tests/test_depth_geometry.py
git commit -m "feat(server): greeting archives the full achieved filter_options"
git push
```

---

### Task 4: `apply_filter_settings` + `_handle_set` — validate, swap, retire, reply

The runtime mutation core (spec §3.1, §3.4): all-or-nothing validation, chain rebuild under `_camera_lock`, generation bump LAST, one JSON reply line. No socket wiring yet — that is Task 5.

**Files:**
- Modify: `server/server_unicast_syncronous.py` (new code directly below `setup_depth_filters`)
- Test: `tests/test_server_env.py`

**Interfaces:**
- Consumes: `FILTER_SETTINGS`, `setup_depth_filters()`, `FILTER_OPTIONS`, `DEPTH_FILTER_NAMES` (Tasks 1-2), existing `_camera_lock`, `_camera_generation`, `_greeting_is_stale`.
- Produces: `SettingError(ValueError)`; `apply_filter_settings(updates: dict) -> dict` (returns the new achieved `FILTER_OPTIONS` copy; raises `SettingError`; empty dict = pure read, no bump); `_handle_set(line: bytes) -> bytes` (one JSON reply line, never raises: `{"ok":true,"filters":[...],"filter_options":{...}}` or `{"ok":false,"error":"..."}`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server_env.py` (add `import json` at the top of the file):

```python
# ------------------------------------------------- runtime-parameters, Task 4
# spec 3.1/3.4 and tests 1,2,4,5,6,7: the SET core, without the socket.

def _fresh(monkeypatch, **env):
    """Reloaded module with the fake SDK installed and a chain built (the state
    stream_burst runs under)."""
    chain = _chain(monkeypatch, **env)
    srv.depth_filters = chain
    srv._reset_camera_state()
    return chain


def test_bare_set_reads_without_changing_or_retiring(monkeypatch):
    """spec test 1: SET with no arguments is a read -- achieved values back,
    nothing rebuilt, nobody's session ends."""
    try:
        _fresh(monkeypatch)
        gen = srv._camera_generation
        reply = json.loads(srv._handle_set(b"SET"))
        assert reply["ok"] is True
        assert reply["filter_options"]["spatial_smooth_delta"] == 20.0
        assert reply["filters"] == ["threshold", "disparity", "spatial",
                                    "temporal", "disparity_inv"]
        assert srv._camera_generation == gen
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_set_applies_and_the_new_chain_serves_the_next_frame(monkeypatch):
    """spec test 3 shape: set -> achieved read-back -> and the module global the
    serving loops read (depth_filters) IS the new chain."""
    try:
        _fresh(monkeypatch)
        reply = json.loads(srv._handle_set(
            b"SET spatial_smooth_delta=8 temporal_persistency=5"))
        assert reply["ok"] is True
        assert reply["filter_options"]["spatial_smooth_delta"] == 8.0
        assert reply["filter_options"]["temporal_persistency"] == 5.0
        spatial = next(f for f in srv.depth_filters if f.kind == "spatial")
        assert spatial.options["filter_smooth_delta"] == 8.0
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_set_spatial_0_reaches_both_greeting_fields(monkeypatch):
    """spec test 2: the control arm must be visible in BOTH `filters` and
    `filter_options` of the next greeting."""
    try:
        _fresh(monkeypatch)
        reply = json.loads(srv._handle_set(b"SET spatial=0"))
        assert "spatial" not in reply["filters"]
        assert reply["filter_options"]["spatial_smooth_delta"] is None
        assert srv.DEPTH_FILTER_NAMES == ["threshold", "disparity", "temporal",
                                          "disparity_inv"]
        assert srv.FILTER_OPTIONS["spatial_smooth_delta"] is None
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_a_write_retires_sessions_greeted_before_it(monkeypatch):
    """spec test 6: the generation the old sessions were greeted under is stale
    the moment a write lands -- _stale_greeting_close then ends them. (The reply
    goes out before the issuing session's own close: Task 5 sends it in the SET
    branch, and the loop only checks staleness at the NEXT iteration.)"""
    try:
        _fresh(monkeypatch)
        greeted = srv._camera_generation
        json.loads(srv._handle_set(b"SET spatial_smooth_delta=8"))
        assert srv._greeting_is_stale(greeted)
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_unknown_setting_is_an_error_and_nothing_is_applied(monkeypatch):
    """spec test 4: unknown COMMANDS stay forgiving, but an unknown SETTING means
    the caller believes it changed something it did not. All-or-nothing: the
    valid key in the same line must NOT land either."""
    try:
        _fresh(monkeypatch)
        gen = srv._camera_generation
        reply = json.loads(srv._handle_set(b"SET spatial=0 laser_power=300"))
        assert reply["ok"] is False and "laser_power" in reply["error"]
        assert "spatial" in srv.DEPTH_FILTER_NAMES          # untouched
        assert srv._camera_generation == gen                 # nobody retired
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_malformed_and_non_numeric_tokens_are_errors(monkeypatch):
    try:
        _fresh(monkeypatch)
        assert json.loads(srv._handle_set(b"SET spatial"))["ok"] is False
        assert json.loads(srv._handle_set(b"SET spatial_smooth_delta=lots"))["ok"] is False
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_out_of_range_set_reports_the_clamped_value(monkeypatch):
    """spec test 5, end to end through the SET path."""
    try:
        _fresh(monkeypatch)
        reply = json.loads(srv._handle_set(b"SET spatial_smooth_delta=500"))
        assert reply["ok"] is True
        assert reply["filter_options"]["spatial_smooth_delta"] == 50.0
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_decimation_stays_refused_at_runtime(monkeypatch):
    """spec 2.2/2.5 (amended): enabling decimation would change the depth
    geometry the greeting already declared. Refused, loudly."""
    try:
        _fresh(monkeypatch)
        reply = json.loads(srv._handle_set(b"SET decimation=2"))
        assert reply["ok"] is False and "geometry" in reply["error"]
        assert json.loads(srv._handle_set(b"SET decimation=0"))["ok"] is True
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_hole_filling_round_trips_and_minus_one_removes_it(monkeypatch):
    try:
        _fresh(monkeypatch)
        reply = json.loads(srv._handle_set(b"SET hole_filling=1"))
        assert reply["filters"][-1] == "hole_filling"
        assert reply["filter_options"]["hole_filling"] == 1.0
        reply = json.loads(srv._handle_set(b"SET hole_filling=-1"))
        assert "hole_filling" not in reply["filters"]
        assert reply["filter_options"]["hole_filling"] is None
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_restart_returns_to_the_unit_files_values(monkeypatch):
    """spec test 7 / 4.2 -- THE central property: an override cannot survive a
    restart. A restart re-imports the module, which re-reads env."""
    try:
        _fresh(monkeypatch)
        json.loads(srv._handle_set(b"SET spatial_smooth_delta=8 depth_max_m=0.9"))
        assert srv.FILTER_SETTINGS["spatial_smooth_delta"] == 8.0
    finally:
        monkeypatch.undo()
        importlib.reload(srv)
    assert srv.FILTER_SETTINGS["spatial_smooth_delta"] == -1.0
    assert srv.FILTER_SETTINGS["depth_max_m"] == 1.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.10 -m pytest tests/test_server_env.py -v`
Expected: every new Task-4 test FAILS with `AttributeError: ... has no attribute '_handle_set'`; everything else PASSES.

- [ ] **Step 3: Implement**

In `server/server_unicast_syncronous.py`, directly below `setup_depth_filters`:

```python
class SettingError(ValueError):
    """A SET the caller must hear about: unknown key, bad number, refused knob."""


def apply_filter_settings(updates):
    """Validate + apply runtime filter settings; return the new achieved values.

    All-or-nothing: any unknown key or refused value raises SettingError BEFORE
    anything is touched -- a caller that thinks it changed something must never
    have half-changed it. An empty ``updates`` is a pure read: current achieved
    values back, nothing rebuilt, no generation bump, nobody's session ends.

    A write rebuilds the chain from FILTER_SETTINGS under _camera_lock and bumps
    _camera_generation LAST -- the same ordering invariant _rebuild_pipeline
    documents -- so _stale_greeting_close retires every session greeted under
    the old chain and no frame is ever archived under a greeting that describes
    a chain it did not run through (spec 3.4: a filter swap does not change
    geometry, so the fusion guard cannot catch it; this is what does).

    Known, accepted caveat (spec 3.4): _rebuild_pipeline treats a moved
    generation as "another thread already rebuilt the wedged pipeline" and skips
    one recovery attempt, so a SET landing during a genuine camera wedge delays
    recovery by one timeout cycle. It self-heals; do not "fix" it here.

    A runtime override lives only in FILTER_SETTINGS (RAM): a restart re-imports
    the module and re-reads env, which is the whole safety argument (spec 4.2).
    There is deliberately no persist path."""
    global depth_filters, _camera_generation
    unknown = sorted(k for k in updates if k not in FILTER_SETTINGS)
    if unknown:
        raise SettingError("unknown setting(s): {} (settable: {})".format(
            ", ".join(unknown), ", ".join(sorted(FILTER_SETTINGS))))
    if float(updates.get("decimation", 0.0)) != 0.0:
        raise SettingError("decimation changes the depth geometry the greeting "
                           "declares; it stays restart-path only (spec 2.5)")
    if not updates:
        return dict(FILTER_OPTIONS)
    clean = dict((k, float(v)) for k, v in updates.items())
    if "hole_filling" in clean:                    # constructor arg, not an rs.option:
        clean["hole_filling"] = min(max(clean["hole_filling"], -1.0), 2.0)
    with _camera_lock:
        FILTER_SETTINGS.update(clean)
        depth_filters = setup_depth_filters()
        _camera_generation += 1        # LAST: retires every session greeted before it
    print("Runtime SET applied: {} -> generation {}".format(
        clean, _camera_generation), flush=True)
    return dict(FILTER_OPTIONS)


def _handle_set(line):
    """One ``SET [key=value ...]`` line -> one JSON reply line. Never raises.

    Unknown COMMANDS in the burst loop stay forgiving (a newer client against an
    older server should degrade, not die), but an unknown SETTING inside a SET is
    an ERROR: the caller believes it changed something it did not (spec test 4)."""
    try:
        updates = {}
        for token in line.decode("ascii", "replace").split()[1:]:
            key, eq, raw = token.partition("=")
            if not eq:
                raise SettingError(
                    "malformed token {!r}: expected key=value".format(token))
            try:
                updates[key] = float(raw)
            except ValueError:
                raise SettingError("{}={!r} is not a number".format(key, raw))
        achieved = apply_filter_settings(updates)
        reply = {"ok": True, "filters": list(DEPTH_FILTER_NAMES),
                 "filter_options": achieved}
    except SettingError as exc:
        reply = {"ok": False, "error": str(exc)}
    except Exception as exc:      # a bug must degrade to an error line, not kill the thread
        reply = {"ok": False, "error": "internal: {!r}".format(exc)}
    return json.dumps(reply, separators=(",", ":")).encode("utf-8") + b"\n"
```

(`json` is already imported at the top of the file.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.10 -m pytest tests/test_server_env.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit + push**

```bash
git add server/server_unicast_syncronous.py tests/test_server_env.py
git commit -m "feat(server): apply_filter_settings + _handle_set -- validated runtime SET core"
git push
```

---

### Task 5: Burst-loop wiring + protocol docs + spec caveat

Five lines of socket wiring (spec §3.1 ordering: reply first, the loop's next staleness check ends the issuing session), plus the documentation that makes the behaviour a contract instead of a surprise (spec §4.4).

**Files:**
- Modify: `server/server_unicast_syncronous.py:1614` (`stream_burst` command read + dispatch; docstring ~1573-1592)
- Modify: `server/README.md` (~line 156, the `MODE BURST V2` command table)
- Modify: `docs/superpowers/specs/2026-09-01-realsense-runtime-parameters-design.md` (§2.2 decimation row caveat)
- Test: none new (the logic all lives in Task 4's tested functions; this wiring is verified live in Task 7 — record that explicitly in the commit message)

**Interfaces:**
- Consumes: `_handle_set` (Task 4), existing `_recv_line(conn, maxlen)`, existing `_stale_greeting_close` top-of-loop check in `stream_burst`.
- Produces: the wire behaviour Task 7 verifies: `SET ...\n` on a `MODE BURST V2` connection → one JSON line back; a successful write ends that session at its next loop iteration.

- [ ] **Step 1: Implement the dispatch**

In `stream_burst`, replace the single line

```python
            cmd = _recv_line(conn).strip().upper()
```

with

```python
            # maxlen 256: a multi-key SET line (spec 3.1) outgrows the default 64.
            line = _recv_line(conn, maxlen=256)
            cmd = line.strip().upper()
```

and insert, between the `if not cmd: break` and the `if cmd == b'CAP':` branch:

```python
            if cmd == b'SET' or cmd.startswith(b'SET '):
                # Runtime filter settings (runtime-parameters spec 3.1). Reply
                # FIRST; a successful WRITE bumped the generation, so the
                # top-of-loop staleness check ends this session on the next
                # iteration -- after the reply is out. The client reconnects
                # into a fresh greeting that records the new values. A bare SET
                # (a read) bumps nothing and the session continues. NOTE the
                # case split: keys are lowercase, so parse from `line`, not
                # the uppercased `cmd`.
                conn.sendall(_handle_set(line.strip()))
                continue
```

(The `if cmd == b'CAP':` that follows becomes `elif`-compatible automatically since the SET branch `continue`s.)

- [ ] **Step 2: Update the two protocol docs**

2a. `stream_burst` docstring: after the `CLEAR` line in the command table, add:

```
        SET    -> ``SET [key=value ...]``: change safe-tier depth-filter settings
                  at runtime (runtime-parameters spec). Reply is ONE JSON line:
                  {"ok":true,"filters":[...],"filter_options":{...}} with the
                  ACHIEVED values, or {"ok":false,"error":"..."}. A successful
                  write retires the camera generation: every session greeted
                  before it (THIS one included, after the reply) is closed and
                  reconnects into a fresh greeting. Bare SET = read-only.
                  Overrides die on restart -- the unit file stays the boot truth.
```

2b. `server/README.md`, in the `MODE BURST V2` command table (~line 156 area), add a row:

```
| `SET [k=v ...]` | one JSON line: `{"ok":true,"filters":[...],"filter_options":{...}}` (achieved values) or `{"ok":false,"error":"..."}`. A successful write retires the generation — sessions greeted before it end (the issuing one too, after the reply) and reconnect into a fresh greeting. A bare `SET` is a read. Overrides never survive a service restart. |
```

2c. In the spec, §2.2 table, change the decimation row's scope cell from `in, default off` to `in as a recorded constant 0 — enabling it at runtime is refused (it changes the depth geometry the greeting declared; restart path, §2.5)`.

- [ ] **Step 3: Compile-check and run the affected suites**

Run: `py -3.10 -m py_compile server/server_unicast_syncronous.py`
Run: `py -3.10 -m pytest tests/test_server_env.py tests/test_rs_geometry.py tests/test_depth_geometry.py tests/test_handshake.py -v`
Expected: ALL PASS (no behaviour change host-side; test_handshake guards that the burst handshake itself is untouched).

- [ ] **Step 4: Commit + push**

```bash
git add server/server_unicast_syncronous.py server/README.md docs/superpowers/specs/2026-09-01-realsense-runtime-parameters-design.md
git commit -m "feat(server): SET command wired into the burst loop (live-verified in the deploy task)"
git push
```

---

### Task 6: Capability-audit refresh (spec §2.6)

The spec names `docs/realsense-capability-audit-2026-08-29.md` stale in four places and says to fix it as part of this work so nobody re-chases closed items.

**Files:**
- Modify: `docs/realsense-capability-audit-2026-08-29.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Verify each claim in code, then mark the four items fixed**

Read the audit's R2, R3, R5 and R7 entries. For each, confirm the fix actually exists in code before writing (spec §2.6 asserts they are fixed; verify, don't trust):
- **R2** (depth units): `rs_config.py` sets `depth_units` to 0.1 mm (grep `depth_units` in `server/rs_config.py`).
- **R3** (Jetson-side `align` discarding half the depth field): the server ships NATIVE depth, `aligned: False` in the greeting (`rs_geometry.build_greeting`), host back-projects.
- **R5** (hole filling): now an explicit runtime knob defaulting OFF (`FILTER_SETTINGS["hole_filling"] = -1.0`, this work).
- **R7** (colour resolution): `COLOR_SIZE = (1920, 1080)` at `server/server_unicast_syncronous.py:881`.

Add to each entry a status line in the audit's own voice, e.g. `**FIXED** (2026-08-30 sensor-layer batch, merged in 54ab1da / this runtime-parameters work for R5) — see <file:line>.` Use the actual commit hashes visible in `git log --oneline -- server/` if they differ. Do NOT rewrite the entries themselves — the audit is a dated record; the status lines are the update.

Also add one line to the audit's intro noting that the safe-tier filter chain is runtime-settable via `SET` since this work, with achieved values in the greeting.

- [ ] **Step 2: Commit + push**

```bash
git add docs/realsense-capability-audit-2026-08-29.md
git commit -m "docs(audit): mark R2/R3/R5/R7 fixed; note the runtime SET tier"
git push
```

---### Task 7: Deploy to the Jetson + live verification

The wiring Task 5 could not unit-test gets proven against the real server, then the camera is restored to its boot state.

**Files:**
- Create: `<scratchpad>/verify_set.py` (throwaway; the scratchpad dir is in the session prompt — NOT committed, spec §6.3: no sweep tool yet)

**Interfaces:**
- Consumes: the deployed server on `10.12.171.70:1024`; `py -3.10 tools/jetson_deploy.py deploy|status`.

- [ ] **Step 1: Deploy**

Run: `py -3.10 tools/jetson_deploy.py deploy`
Expected: pulls `main` on the Jetson, restarts `realsense-camera`. Then `py -3.10 tools/jetson_deploy.py status` — service active, listening on 1024.

- [ ] **Step 2: Live round-trip**

Write `<scratchpad>/verify_set.py`:

```python
"""Live verification of the runtime SET tier (runtime-parameters spec, tests 1-7 live)."""
import json
import socket

HOST, PORT = "10.12.171.70", 1024


def recv_line(sock):
    buf = bytearray()
    while not buf.endswith(b"\n"):
        ch = sock.recv(1)
        if not ch:
            return bytes(buf)
        buf += ch
    return bytes(buf)


def burst_session():
    s = socket.create_connection((HOST, PORT), timeout=20)
    s.sendall(b"MODE BURST V2\n")
    assert recv_line(s) == b"BURST READY\n"
    greeting = json.loads(recv_line(s))
    return s, greeting


def one_shot(command):
    s, greeting = burst_session()
    s.sendall(command + b"\n")
    reply = json.loads(recv_line(s))
    s.close()
    return greeting, reply


# 1. Read: bare SET returns achieved values matching the greeting.
g, read0 = one_shot(b"SET")
assert read0["ok"], read0
baseline = g["filter_options"]["spatial_smooth_delta"]
assert read0["filter_options"] == g["filter_options"], (read0, g["filter_options"])
print("baseline smooth_delta:", baseline, "| full options:", g["filter_options"])

# 2. Write: achieved value comes back changed...
_, w = one_shot(b"SET spatial_smooth_delta=8")
assert w["ok"] and w["filter_options"]["spatial_smooth_delta"] == 8.0, w

# 3. ...and a FRESH connection's greeting records it (provenance, spec 3.3).
g2, _ = one_shot(b"SET")
assert g2["filter_options"]["spatial_smooth_delta"] == 8.0, g2["filter_options"]
print("override visible in fresh greeting: OK")

# 4. A session greeted BEFORE a write is retired (spec 3.4) -- open A, write on B,
#    then A's next command must meet a closed connection.
a, _ = burst_session()
one_shot(b"SET spatial_smooth_delta=12")
a.sendall(b"CAP\n")
tail = a.recv(64)          # server ends the session after the staleness check
assert tail == b"", "stale session was NOT closed: %r" % tail
a.close()
print("stale session retired on write: OK")

# 5. Unknown setting is an error (spec test 4). laser_power is device-tier: pinned.
_, bad = one_shot(b"SET laser_power=300")
assert not bad["ok"] and "laser_power" in bad["error"], bad
print("device-tier knob refused: OK")

# 6. Decimation refused (amended spec 2.2).
_, dec = one_shot(b"SET decimation=2")
assert not dec["ok"], dec
print("decimation refused: OK")

# 7. Over-long SET is refused LOUDLY and the session ends. This distinguishes the
#    fixed behaviour from the original bug: a line past SET_LINE_MAXLEN used to be
#    read truncated, leaving its tail in the socket to be replayed as a command --
#    and if the cut fell on a token boundary it applied a SUBSET of the keys under
#    "ok":true. Expect ok:false naming the length, then a CLOSED connection.
s, _ = burst_session()
s.sendall(b"SET " + b"spatial_smooth_delta=20 " * 40 + b"\n")   # ~960 bytes
long_reply = json.loads(recv_line(s))
assert not long_reply["ok"] and "byte" in long_reply["error"], long_reply
assert s.recv(64) == b"", "over-long SET did not end the session"
s.close()
print("over-long SET refused and session ended: OK")

# 8. Inverted thresholds. This probes the REAL rs.threshold_filter, which the fake
#    SDK in the unit tests cannot answer for: either its constructor raises (and
#    apply_filter_settings' rollback path catches it -> ok:false, FILTER_SETTINGS
#    restored) or it accepts the pair silently and clips every pixel away (-> ok:true
#    with achieved min>max, and the next CAP returns an empty depth frame). Both are
#    informative; record which one the camera does. Restore afterwards either way.
_, inv = one_shot(b"SET depth_min_m=1.0 depth_max_m=0.5")
print("inverted thresholds ->", "REFUSED by the SDK" if not inv["ok"]
      else "ACCEPTED, achieved %s" % {k: inv["filter_options"][k]
                                      for k in ("depth_min_m", "depth_max_m")})
_, restored = one_shot(b"SET depth_min_m=0.15 depth_max_m=1.5")
assert restored["ok"], restored
print("thresholds restored: OK")

print("\nALL LIVE CHECKS PASSED. Now restart the service and re-run step 3's check")
print("expecting the BASELINE value back (spec test 7):", baseline)
```

Run: `py -3.10 <scratchpad>/verify_set.py`
Expected: every assert passes. If a scan/inspection client is using the camera, WAIT — do not retire a live session under the operator.

- [ ] **Step 3: Restart-reverts check (spec test 7, live)**

Run: `py -3.10 tools/jetson_deploy.py deploy` (pull is a no-op; the restart is the point). Then re-run `py -3.10 <scratchpad>/verify_set.py` and confirm the printed `baseline smooth_delta` equals the original baseline from Step 2 (the unit file's value, NOT 12.0). The script's own asserts passing a second time also re-proves the whole tier post-restart.

- [ ] **Step 4: Report**

Summarize to the user: commit hashes pushed, Jetson deploy status, the live baseline value, and that the roll-probe §3.1 spatial A/B is now unblocked (a bare `SET spatial=0` / restart pair per arm, no agent on the Jetson).

---

## Self-review notes (already applied)

- **Spec coverage:** §2.2 knobs → Tasks 1-2; §3.1 command+ordering → Tasks 4-5; §3.3 provenance → Tasks 2-3; §3.4 generation reuse + caveat → Task 4 docstring; §3.5 layering → Task 1 (env) + Task 4 (no persist); §4.4 documented multi-client behaviour → Task 5 docs; §4.5 settling → no code by design (a sweep-procedure concern; the spec's own §5 list has no test for it); §5 tests 1-9 → mapped inline in the test names; §6 decisions honored (env vars for thresholds, end-don't-refuse, no sweep tool); §2.6 audit → Task 6; §7 out-of-scope respected (no device options, no auth, unit file untouched).
- **Deviation from the spec, made explicit:** the spec's §2.2 listed decimation "in, default off"; enabling it at runtime would break the greeting's declared depth geometry, so it ships as a recorded constant 0 with an explicit refusal — Task 5 amends the spec row to match.
- **Type consistency:** `FILTER_SETTINGS` values are all `float` (`spatial` stored as float, truthiness-tested); `FILTER_OPTIONS` values are `float | None`; `apply_filter_settings(dict) -> dict`; `_handle_set(bytes) -> bytes`; wire keys identical across `FILTER_SETTINGS`, `_OPTION_MAP`, greeting `filter_options`, and every test.
- **Python 3.6 discipline:** all new `server/` code uses `.format()` prints and no modern syntax; test files (3.10) use f-strings freely.
