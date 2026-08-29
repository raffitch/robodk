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
