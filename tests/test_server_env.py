"""Import-time env parsing on the Jetson server must never crash the service."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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


class _FakeFilter:
    def __init__(self, kind):
        self.kind = kind
        self.options = {}

    def set_option(self, option, value):
        self.options[option] = value


class _FakeRs:
    class option:
        filter_smooth_delta = "filter_smooth_delta"

    @staticmethod
    def threshold_filter(lo, hi):
        return _FakeFilter("threshold")

    @staticmethod
    def disparity_transform(forward):
        return _FakeFilter("disparity" if forward else "disparity_inv")

    @staticmethod
    def spatial_filter():
        return _FakeFilter("spatial")

    @staticmethod
    def temporal_filter():
        return _FakeFilter("temporal")


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
