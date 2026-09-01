"""Import-time env parsing on the Jetson server must never crash the service."""
from __future__ import annotations

import importlib
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

def _greeting(monkeypatch, **env):
    """A full protocol-2 greeting, built the way a connecting client gets one, with
    the filter chain that ``env`` produces actually installed."""
    chain = _chain(monkeypatch, **env)
    from server import rs_geometry

    srv.depth_filters = chain
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
    snap = srv.CameraSnapshot(
        pipeline=pipeline, depth_unit_mm=0.1, geometry=static,
        achieved={"visual_preset": 0.0, "laser_power": 150.0},
        device={"serial": "S1", "fw": "5.16.0.1", "librealsense": "2.55.1"},
        generation=0)
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
