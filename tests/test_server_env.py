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
