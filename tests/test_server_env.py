"""Import-time env parsing on the Jetson server must never crash the service."""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
sys.modules.setdefault("pyrealsense2", SimpleNamespace())
sys.modules.setdefault("turbojpeg", SimpleNamespace())

import server.server_unicast_syncronous as srv  # noqa: E402


def test_env_number_falls_back_on_garbage():
    os.environ["X_TEST_NUM"] = "high_density"
    try:
        assert srv._env_number("X_TEST_NUM", -1.0) == -1.0
    finally:
        del os.environ["X_TEST_NUM"]
    assert srv._env_number("X_TEST_ABSENT", 7.5) == 7.5


def test_typoed_preset_name_does_not_kill_the_import(monkeypatch):
    """The unit is Restart=always with no start limit — an import-time
    ValueError becomes an infinite crash-loop with the camera dark for every
    module (scan, calibration, extrusion inspection)."""
    monkeypatch.setenv("RS_VISUAL_PRESET", "high_accuracy")
    monkeypatch.setenv("RS_LASER_POWER", "please")
    try:
        importlib.reload(srv)
        assert srv.RS_VISUAL_PRESET == -1
        assert srv.RS_LASER_POWER == -1.0
    finally:
        monkeypatch.undo()
        importlib.reload(srv)   # restore module state for other test files


def test_laser_power_defaults_to_leave_alone():
    """Review finding: a default of 300 doubled projector power vs the
    configuration the 2026-08-13 depth characterization was measured under."""
    assert srv.RS_LASER_POWER == -1.0


# --------------------------------------------------- spatial filter A/B lever
# The spatial filter runs in the DISPARITY domain with stock defaults
# (magnitude 2, alpha 0.5, smooth_delta 20). smooth_delta is the edge-preservation
# threshold: steps larger than it survive, smaller ones are smoothed across. At
# 300 mm with fx 637 and the D435i's ~50 mm baseline, 1 mm of relief is ~0.35
# disparity px -- so a 4-11 mm ring spans 1.4-3.9, comfortably under 20. The
# filter cannot tell the rope from the board and smooths across it twice.
#
# Measured 2026-08-31 against the operator's ruler at five clock positions: the
# camera tracks the ring's shape (r = 0.97) but reads ~1.5 mm low on every one.
# That eats most of a 1.78 mm floor budget, which is why a continuous ring reads
# as open. These levers exist to A/B that hypothesis on the cell; both DEFAULT TO
# CURRENT BEHAVIOUR, so nothing changes until someone sets them deliberately --
# the same discipline RS_LASER_POWER and RS_VISUAL_PRESET already follow.


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


class _FakeFilter:
    def __init__(self, kind, options=None):
        self.kind = kind
        self.options = dict(options or {})

    def set_option(self, option, value):
        self.options[option] = value

    def get_option(self, option):
        """What the filter is ACTUALLY set to -- an explicit set, else the SDK default."""
        if option in self.options:
            return float(self.options[option])
        try:
            return float(SDK_DEFAULTS[self.kind][option])
        except KeyError:
            raise RuntimeError(f"{self.kind} has no option {option!r}")

    def get_option_range(self, option):
        lo, hi = SDK_RANGES[self.kind][option]   # KeyError -> caller's except path
        return SimpleNamespace(min=lo, max=hi)


class _FakeRs:
    class option:
        filter_smooth_delta = "filter_smooth_delta"
        filter_magnitude = "filter_magnitude"
        filter_smooth_alpha = "filter_smooth_alpha"
        holes_fill = "holes_fill"
        min_distance = "min_distance"
        max_distance = "max_distance"

    @staticmethod
    def threshold_filter(lo, hi):
        return _FakeFilter("threshold", {"min_distance": lo, "max_distance": hi})

    @staticmethod
    def disparity_transform(forward):
        return _FakeFilter("disparity" if forward else "disparity_inv")

    @staticmethod
    def spatial_filter():
        return _FakeFilter("spatial")

    @staticmethod
    def temporal_filter():
        return _FakeFilter("temporal")

    @staticmethod
    def hole_filling_filter(mode):
        return _FakeFilter("hole_filling", {"holes_fill": float(mode)})


def _chain(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    importlib.reload(srv)
    monkeypatch.setattr(srv, "rs", _FakeRs)
    return srv.setup_depth_filters()


def test_the_spatial_filter_is_left_alone_by_default(monkeypatch):
    """No env set must reproduce today's chain EXACTLY, options untouched --
    every number in the archive was measured under it."""
    try:
        chain = _chain(monkeypatch)
        assert [f.kind for f in chain] == [
            "threshold", "disparity", "spatial", "temporal", "disparity_inv"]
        spatial = next(f for f in chain if f.kind == "spatial")
        assert spatial.options == {}
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_smooth_delta_can_be_lowered_so_the_ring_reads_as_an_edge(monkeypatch):
    try:
        chain = _chain(monkeypatch, RS_SPATIAL_SMOOTH_DELTA="4")
        spatial = next(f for f in chain if f.kind == "spatial")
        assert spatial.options == {_FakeRs.option.filter_smooth_delta: 4.0}
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_the_spatial_filter_can_be_dropped_for_the_control_arm(monkeypatch):
    try:
        chain = _chain(monkeypatch, RS_SPATIAL="0")
        assert [f.kind for f in chain] == [
            "threshold", "disparity", "temporal", "disparity_inv"]
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_the_greeting_names_the_filters_that_actually_ran(monkeypatch):
    """The greeting's filter list is archived as provenance on every take. If it
    keeps claiming 'spatial' while the control arm ran without it, the A/B this
    lever exists for is unreadable afterwards -- and nothing else records which
    arm a take came from."""
    try:
        _chain(monkeypatch, RS_SPATIAL="0")
        assert srv.DEPTH_FILTER_NAMES == [
            "threshold", "disparity", "temporal", "disparity_inv"]
    finally:
        monkeypatch.undo()
        importlib.reload(srv)
    chain = _chain(monkeypatch)
    try:
        assert srv.DEPTH_FILTER_NAMES == [f.kind for f in chain]
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_garbage_spatial_env_does_not_kill_the_service(monkeypatch):
    try:
        chain = _chain(monkeypatch, RS_SPATIAL="off", RS_SPATIAL_SMOOTH_DELTA="lots")
        assert [f.kind for f in chain] == [
            "threshold", "disparity", "spatial", "temporal", "disparity_inv"]
        assert next(f for f in chain if f.kind == "spatial").options == {}
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


# ------------------------------------- the ACHIEVED smooth_delta, in the greeting
# The filter-name list already tells an archived take whether `spatial` ran, so the
# RS_SPATIAL arm is readable afterwards. The DELTA is not: with the filter present,
# a take run at delta 20 and a take run at delta 4 archive byte-identical
# provenance, and the two arms of a delta sweep become indistinguishable the moment
# the operator forgets which was which. docs/inspection-roll-probe-handoff.md 3.1
# blocks the sweep on exactly this.
#
# It must be the ACHIEVED value, read back off the filter object that actually
# processes the frames -- not an echo of the env var. The env var's own default
# (-1 = "don't touch") names no number at all, and the number it silently inherits
# is librealsense's, which a future SDK is free to change under us.

def _snapshot(generation=0):
    """The CameraSnapshot a connecting client is greeted from, filled from the
    module globals exactly the way ``_camera_snapshot()`` fills it (filter
    description included -- that is what makes the greeting coherent)."""
    from server import rs_geometry

    static = rs_geometry.StaticGeometry(
        depth={"width": 1280, "height": 720, "fx": 640.0, "fy": 640.0, "ppx": 640.0,
               "ppy": 360.0, "model": "none", "coeffs": [0.0] * 5},
        color={"width": 1920, "height": 1080, "fx": 960.0, "fy": 960.0, "ppx": 960.0,
               "ppy": 540.0, "model": "none", "coeffs": [0.0] * 5},
        R_dc=np.eye(3), t_dc_mm=np.zeros(3),
        depth_size=(1280, 720), color_size=(1920, 1080))
    sensor = SimpleNamespace()
    pipeline = SimpleNamespace(get_active_profile=lambda: SimpleNamespace(
        get_device=lambda: SimpleNamespace(first_depth_sensor=lambda: sensor)))
    return srv.CameraSnapshot(
        pipeline=pipeline, depth_unit_mm=0.1, geometry=static,
        achieved={"visual_preset": 0.0, "laser_power": 150.0},
        device={"serial": "S1", "fw": "5.16.0.1", "librealsense": "2.55.1"},
        generation=generation,
        filter_names=list(srv.DEPTH_FILTER_NAMES),
        filter_options=dict(srv.FILTER_OPTIONS))


def _greeting(monkeypatch, **env):
    """A full protocol-2 greeting, built the way a connecting client gets one, with
    the filter chain that ``env`` produces actually installed."""
    chain = _chain(monkeypatch, **env)
    srv.depth_filters = chain
    snap = _snapshot()
    monkeypatch.setattr(srv, "_camera_snapshot", lambda: snap)
    return srv.make_greeting()


def test_the_greeting_records_the_smooth_delta_the_filter_actually_ran_with(monkeypatch):
    """Unset env still archives a NUMBER -- the SDK default the filter is really at."""
    try:
        greeting = _greeting(monkeypatch)
        assert "spatial" in greeting["filters"]
        assert greeting["filter_options"]["spatial_smooth_delta"] == 20.0
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_the_greeting_carries_the_full_achieved_options(monkeypatch):
    try:
        greeting = _greeting(monkeypatch)
        assert greeting["filter_options"] == srv.FILTER_OPTIONS
        assert greeting["filter_options"]["temporal_persistency"] == 3.0
        assert greeting["filter_options"]["depth_max_m"] == 1.5
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_a_lowered_smooth_delta_reaches_the_greeting(monkeypatch):
    """The delta-sweep arm. Two takes captured under different deltas must not
    archive the same provenance."""
    try:
        greeting = _greeting(monkeypatch, RS_SPATIAL_SMOOTH_DELTA="4")
        assert greeting["filter_options"]["spatial_smooth_delta"] == 4.0
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_no_spatial_filter_is_recorded_distinctly_from_the_sdk_default(monkeypatch):
    """RS_SPATIAL=0 means there is no spatial filter at all -- not "spatial at 20".
    Recording the env var instead of the achieved value would blur the two."""
    try:
        greeting = _greeting(monkeypatch, RS_SPATIAL="0")
        assert "spatial" not in greeting["filters"]
        assert greeting["filter_options"]["spatial_smooth_delta"] is None
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_a_smooth_delta_that_cannot_be_read_back_does_not_kill_the_service(monkeypatch):
    """Provenance is never worth the camera. An SDK that refuses the read-back must
    leave the chain intact and the service serving -- with the value recorded as
    unknown rather than as a number nobody measured."""
    class _NoReadback(_FakeRs):
        @staticmethod
        def spatial_filter():
            f = _FakeFilter("spatial")
            f.get_option = lambda option: (_ for _ in ()).throw(
                RuntimeError("option not supported by this build"))
            return f

    try:
        _chain(monkeypatch)                       # reload under a clean env first
        monkeypatch.setattr(srv, "rs", _NoReadback)
        chain = srv.setup_depth_filters()
        assert [f.kind for f in chain] == [
            "threshold", "disparity", "spatial", "temporal", "disparity_inv"]
        assert srv.FILTER_OPTIONS["spatial_smooth_delta"] is None
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


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


def test_apply_option_degrades_when_the_sdk_lacks_the_option_name(monkeypatch):
    """A pyrealsense2 build that lacks one of _OPTION_MAP's option names must not
    crash setup_depth_filters() -- under Restart=always with no start limit that
    is exactly the infinite crash-loop with the camera dark the read-back guard
    exists to prevent. The affected knob records None (unknown), not a
    fabricated number, and the chain still comes back complete and correctly
    ordered -- the service keeps serving."""
    class _MissingMagnitude(_FakeRs):
        class option:
            filter_smooth_delta = "filter_smooth_delta"
            filter_smooth_alpha = "filter_smooth_alpha"
            holes_fill = "holes_fill"
            min_distance = "min_distance"
            max_distance = "max_distance"
            # filter_magnitude intentionally absent

    try:
        _chain(monkeypatch, RS_SPATIAL="1")   # reload under a clean env first
        monkeypatch.setattr(srv, "rs", _MissingMagnitude)
        srv.FILTER_SETTINGS["spatial_magnitude"] = 3.0
        chain = srv.setup_depth_filters()
        assert [f.kind for f in chain] == [
            "threshold", "disparity", "spatial", "temporal", "disparity_inv"]
        assert srv.FILTER_OPTIONS["spatial_magnitude"] is None
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


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
        chain = _fresh(monkeypatch)
        gen = srv._camera_generation
        reply = json.loads(srv._handle_set(b"SET"))
        assert reply["ok"] is True
        assert reply["filter_options"]["spatial_smooth_delta"] == 20.0
        assert reply["filters"] == ["threshold", "disparity", "spatial",
                                    "temporal", "disparity_inv"]
        assert srv._camera_generation == gen
        # a read must not touch the chain object either -- only a write rebuilds.
        assert srv.depth_filters is chain
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


def test_decimation_refusal_is_all_or_nothing(monkeypatch):
    """review rider 3: the unknown-key gate already has all-or-nothing coverage
    (test_unknown_setting_is_an_error_and_nothing_is_applied); the decimation
    gate did not -- a valid key riding along in the same line must not land."""
    try:
        _fresh(monkeypatch)
        gen = srv._camera_generation
        reply = json.loads(
            srv._handle_set(b"SET spatial_smooth_delta=8 decimation=2"))
        assert reply["ok"] is False
        assert srv.FILTER_SETTINGS["spatial_smooth_delta"] == -1.0   # untouched
        assert srv._camera_generation == gen                          # nobody retired
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


def test_a_rejected_rebuild_rolls_back_filter_settings_and_stays_unwedged(monkeypatch):
    """review Important-1: apply_filter_settings used to update FILTER_SETTINGS
    BEFORE calling setup_depth_filters(). If the rebuild then raised (e.g. the
    SDK refusing an inverted depth_min_m/depth_max_m), FILTER_SETTINGS was left
    holding values no chain ever ran with -- and because every later SET
    rebuilds from that same dict, EVERY SUBSEQUENT SET failed identically until
    a restart. This is the only branch that was completely untested before this
    fix: pin that a rejected rebuild rolls FILTER_SETTINGS back exactly, moves
    nobody's generation, and leaves the next ordinary SET able to succeed."""
    try:
        _fresh(monkeypatch)
        gen = srv._camera_generation
        previous = dict(srv.FILTER_SETTINGS)
        real_setup = srv.setup_depth_filters

        def boom():
            raise RuntimeError("the SDK refused this configuration")

        monkeypatch.setattr(srv, "setup_depth_filters", boom)
        # Any setting that reaches the rebuild will do; this one is deliberately
        # NOT a depth threshold, since an inverted pair is now refused before the
        # rebuild is ever attempted and would exercise a different branch.
        reply = json.loads(srv._handle_set(b"SET spatial_smooth_delta=8"))
        assert reply["ok"] is False
        assert "rejected" in reply["error"]
        assert srv.FILTER_SETTINGS == previous            # rolled back exactly
        assert srv._camera_generation == gen               # nobody retired

        # not wedged: restore the real rebuild and confirm a following ordinary
        # SET still succeeds -- FILTER_SETTINGS was not left poisoned.
        monkeypatch.setattr(srv, "setup_depth_filters", real_setup)
        assert json.loads(
            srv._handle_set(b"SET spatial_smooth_delta=8"))["ok"] is True
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_generation_bump_happens_strictly_after_the_rebuild(monkeypatch):
    """review Important-2: test_a_write_retires_sessions_greeted_before_it only
    checks FINAL state, so it would still pass if `_camera_generation += 1` were
    moved above `depth_filters = setup_depth_filters()` -- exactly the ordering
    bug the whole design exists to prevent (a client greeted between the bump
    and the rebuild would get old names/options stamped with the new
    generation, never be retired, and read frames through a chain its greeting
    does not describe). Spy on setup_depth_filters to pin the order directly."""
    try:
        _fresh(monkeypatch)
        gen = srv._camera_generation
        real = srv.setup_depth_filters

        def spy():
            assert srv._camera_generation == gen, \
                "generation bumped before the rebuild"
            return real()

        monkeypatch.setattr(srv, "setup_depth_filters", spy)
        reply = json.loads(srv._handle_set(b"SET spatial_smooth_delta=8"))
        assert reply["ok"] is True
        assert srv._camera_generation == gen + 1
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_non_finite_values_are_rejected(monkeypatch):
    """review rider 2: float("nan") and float("inf") both parse as valid
    floats. Unchecked, SET spatial_smooth_delta=nan would land in
    FILTER_SETTINGS while setup_depth_filters' own `s[key] >= 0` guard is False
    for nan, so the filter silently keeps its SDK default -- FILTER_SETTINGS
    then claims a value that never reached anything."""
    try:
        _fresh(monkeypatch)
        gen = srv._camera_generation
        assert json.loads(
            srv._handle_set(b"SET spatial_smooth_delta=nan"))["ok"] is False
        assert json.loads(
            srv._handle_set(b"SET spatial_smooth_delta=inf"))["ok"] is False
        assert json.loads(
            srv._handle_set(b"SET spatial_smooth_delta=-inf"))["ok"] is False
        assert srv._camera_generation == gen                  # nothing landed
        assert srv.FILTER_SETTINGS["spatial_smooth_delta"] == -1.0
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


# ------------------------------------------- whole-branch review, Important 1
# The SET line's read cap, over the socket. These are the only tests that drive
# stream_burst's command loop, so they use a scripted fake connection: the loop
# touches nothing on a socket but recv / sendall / settimeout.


class _FakeConn:
    """A socket the burst loop can talk to: scripted bytes in, sends recorded.

    ``recv`` returning b'' models the peer closing, which is what ends the loop
    once the script is exhausted. ``remaining`` is the point of the whole fake --
    it shows what the server left UNREAD in the stream, which is precisely the
    hazard an over-long line creates."""

    def __init__(self, script):
        self._inbox = bytearray(script)
        self.sent = []
        self.timeouts = []

    def recv(self, n):
        if not self._inbox:
            return b""                      # peer closed
        chunk = bytes(self._inbox[:n])
        del self._inbox[:n]
        return chunk

    def sendall(self, data):
        self.sent.append(bytes(data))

    def settimeout(self, value):
        self.timeouts.append(value)

    def setsockopt(self, *args):
        pass

    @property
    def remaining(self):
        return bytes(self._inbox)


def _burst_session(monkeypatch, script):
    """Run ONE stream_burst session over ``script`` and hand back the fake socket.

    ``greet`` is stubbed to report the live generation without sending: these
    tests are about the command loop's line handling, and a real greeting would
    drag in device I/O (temperatures, global time) that has nothing to do with
    it. The generation it returns is real, so the staleness check still behaves."""
    monkeypatch.setattr(srv, "turbojpeg",
                        SimpleNamespace(TurboJPEG=lambda path: None))
    monkeypatch.setattr(srv, "greet", lambda conn: srv._camera_generation)
    conn = _FakeConn(script)
    srv.stream_burst(conn, ("test-client", 0))
    return conn


def test_a_successful_set_replies_then_ends_the_session(monkeypatch):
    """spec test 6, the half only the socket can show. The unit tests above prove
    the generation MOVES; this proves the move actually reaches
    ``_stale_greeting_close`` in the burst loop -- and in the right order.

    Both halves matter and neither implies the other. Reply-then-close: the SET
    branch sends before it `continue`s, so the issuing client gets its achieved
    values rather than a bare disconnect. Close-at-all: a session that survived
    its own SET would keep CAPping through the NEW chain while its greeting still
    describes the OLD one -- fusing frames from two chains into one median, which
    `_same_reconstruction_geometry` cannot catch because a filter swap does not
    change geometry (spec 3.4). The trailing CLEARs must therefore go UNSERVED."""
    try:
        _fresh(monkeypatch)
        gen = srv._camera_generation
        conn = _burst_session(monkeypatch, b"SET spatial_smooth_delta=8\nCLEAR\nCLEAR\n")

        assert srv._camera_generation == gen + 1              # the write retired it
        assert conn.sent[0] == b"BURST READY\n"
        reply = json.loads(conn.sent[-1])
        assert reply["ok"] is True                            # reply went out FIRST
        assert reply["filter_options"]["spatial_smooth_delta"] == 8.0
        # ...and then the loop's top-of-iteration staleness check ended the session:
        # neither CLEAR was served (each would have appended a b'\x00\x00\x00\x00').
        assert len(conn.sent) == 2, conn.sent
        assert conn.remaining == b"CLEAR\nCLEAR\n"
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_an_over_long_set_line_is_refused_and_ends_the_session(monkeypatch):
    """review Important-1: _recv_line stops AT its cap and leaves the rest of the
    line -- terminating newline included -- in the socket, where the loop reads it
    as the next command. Two failures follow. The loud one: the cut lands
    mid-token, _handle_set refuses, and the residual tail is then swallowed by the
    loop's forgiving unknown-command rule. The silent one, tested here: the cut
    lands on a token BOUNDARY, so the truncated line is VALID -- it applies a
    SUBSET of the requested keys under "ok":true and archives that subset in the
    greeting as the whole SET. That defeats apply_filter_settings' all-or-nothing
    rule from the socket layer. Reachable in ordinary use, not theoretical: spec
    4.1 tells a sweep to send an explicit restore line between arms, and a 12-key
    restore is 260 bytes against the 256 this shipped with. There is nothing to
    resync to, so hitting the cap must end the session."""
    try:
        _fresh(monkeypatch)
        gen = srv._camera_generation
        settings = dict(srv.FILTER_SETTINGS)

        # A line whose cut falls EXACTLY on a token boundary (the silent case),
        # with a DIFFERENT key past the cut so the truncated prefix is a strict
        # subset of what the caller asked for.
        head, token = b"SET ", b"spatial_smooth_delta=8 "
        n = (srv.SET_LINE_MAXLEN - len(head)) // len(token)
        pad = b" " * (srv.SET_LINE_MAXLEN - len(head) - n * len(token))
        over_long = head + pad + token * n + b"spatial=0"
        assert len(over_long) > srv.SET_LINE_MAXLEN
        assert over_long[:srv.SET_LINE_MAXLEN].endswith(token)

        conn = _burst_session(monkeypatch, over_long + b"\nCLEAR\n")

        reply = json.loads(conn.sent[-1])
        assert reply["ok"] is False, reply
        assert "byte" in reply["error"], reply          # names the length problem
        assert srv.FILTER_SETTINGS == settings           # nothing applied
        assert srv._camera_generation == gen             # nobody retired
        # The session ended: the trailing CLEAR was never served, and its bytes
        # (behind the unread tail) are still in the stream -- which is exactly why
        # the server must not try to resync past them.
        assert conn.sent[0] == b"BURST READY\n"
        assert len(conn.sent) == 2
        assert conn.remaining.startswith(b"spatial=0")
        assert b"CLEAR" in conn.remaining

        # ...and this is what accepting the truncated line would have meant: the
        # prefix _recv_line hands back is itself a perfectly valid SET that drops
        # `spatial=0` and answers ok:true.
        would_have = json.loads(
            srv._handle_set(over_long[:srv.SET_LINE_MAXLEN].strip()))
        assert would_have["ok"] is True
        assert srv.FILTER_SETTINGS["spatial_smooth_delta"] == 8.0   # applied
        assert srv.FILTER_SETTINGS["spatial"] == 1.0                # silently dropped
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_a_full_explicit_restore_line_is_read_whole(monkeypatch):
    """The same finding from the other side, and the regression test the original
    cap would have failed: the line spec 4.1 tells a sweep to send between arms --
    every FILTER_SETTINGS key with its value -- must be read WHOLE and applied.
    Built from the LIVE key set rather than a hardcoded string, so it keeps
    tracking the real knob list as that list grows."""
    try:
        _fresh(monkeypatch)
        restore = ("SET " + " ".join(
            "{}={}".format(key, float(value))
            for key, value in sorted(srv.FILTER_SETTINGS.items()))).encode("ascii")
        # 260 bytes today. If this ever drops under 256 the regression is no longer
        # being exercised and the sizing argument needs revisiting.
        assert len(restore) > 256, len(restore)
        assert len(restore) < srv.SET_LINE_MAXLEN, len(restore)

        json.loads(srv._handle_set(b"SET spatial_smooth_delta=8"))      # arm A
        assert srv.FILTER_SETTINGS["spatial_smooth_delta"] == 8.0

        conn = _burst_session(monkeypatch, restore + b"\n")

        reply = json.loads(conn.sent[-1])
        assert reply["ok"] is True, reply
        assert srv.FILTER_SETTINGS["spatial_smooth_delta"] == -1.0      # restored
        assert reply["filter_options"]["spatial_smooth_delta"] == 20.0  # SDK default
        # The whole line was consumed. Under the old 256-byte cap the tail was left
        # sitting in the socket, to be read as a bogus command of its own.
        assert conn.remaining == b""
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


# ------------------------------------------- whole-branch review, Important 2
# The greeting's filter description must come from the SNAPSHOT, structurally.


def test_the_greeting_describes_the_snapshots_chain_not_the_live_globals(monkeypatch):
    """review Important-2: make_greeting used to read DEPTH_FILTER_NAMES and
    FILTER_OPTIONS as unlocked module globals while taking the generation from the
    locked snapshot -- and a runtime SET rebinds exactly those two, from another
    client's thread. It was correct only because both call paths happened to take
    the snapshot BEFORE those reads. Reorder them above it and the tear inverts to
    an OLD chain description under a NEW generation: a greeting
    _stale_greeting_close never retires, describing a chain its frames do not run
    through -- the silent provenance corruption this branch exists to prevent.
    Pin it structurally: mutate the globals after the snapshot and the greeting
    must still report the snapshot."""
    try:
        _chain(monkeypatch)
        # The coherence now lives in _camera_snapshot, so it must carry them.
        live = srv._camera_snapshot()
        assert live.filter_names == srv.DEPTH_FILTER_NAMES
        assert live.filter_options == srv.FILTER_OPTIONS

        snap = _snapshot()
        # A concurrent SET lands between the snapshot and the greeting build.
        monkeypatch.setattr(srv, "DEPTH_FILTER_NAMES",
                            ["threshold", "disparity", "temporal", "disparity_inv"])
        monkeypatch.setattr(srv, "FILTER_OPTIONS",
                            dict(srv.FILTER_OPTIONS, spatial_smooth_delta=None))

        greeting = srv.make_greeting(snap)
        assert greeting["filters"] == snap.filter_names
        assert "spatial" in greeting["filters"]
        assert greeting["filter_options"]["spatial_smooth_delta"] == 20.0
    finally:
        monkeypatch.undo()
        importlib.reload(srv)


def test_an_inverted_depth_threshold_pair_is_refused(monkeypatch):
    """MEASURED ON THE CELL 2026-09-01: `SET depth_min_m=1.0 depth_max_m=0.5`
    came back ok:true with both values achieved.

    depth_min_m/depth_max_m reach librealsense through the threshold filter's
    CONSTRUCTOR, not through set_option, so `_apply_option`'s range clamp never
    sees them -- and the real SDK does not object to min > max. The chain then
    passes nothing at all: every depth frame comes back empty while the reply,
    the greeting and the archived provenance all say the settings applied. That
    is the worst shape a failure can take here, because a take captured under it
    looks exactly like a take of an empty scene.

    Both orderings are checked: the pair can be inverted by writing either half
    against whatever the other one already is.
    """
    try:
        _fresh(monkeypatch)
        gen = srv._camera_generation

        reply = json.loads(srv._handle_set(b"SET depth_min_m=1.0 depth_max_m=0.5"))
        assert reply["ok"] is False, reply
        assert "depth_min_m" in reply["error"] and "depth_max_m" in reply["error"]
        assert srv._camera_generation == gen            # nobody retired
        assert srv.FILTER_SETTINGS["depth_min_m"] == srv.RS_DEPTH_MIN_M
        assert srv.FILTER_SETTINGS["depth_max_m"] == srv.RS_DEPTH_MAX_M

        # ...and against the value already in force, not just within one command
        assert json.loads(srv._handle_set(b"SET depth_min_m=9.0"))["ok"] is False
        assert json.loads(srv._handle_set(b"SET depth_max_m=0.05"))["ok"] is False

        # a sane pair still applies
        ok = json.loads(srv._handle_set(b"SET depth_min_m=0.2 depth_max_m=1.2"))
        assert ok["ok"] is True, ok
    finally:
        monkeypatch.undo()
        importlib.reload(srv)
