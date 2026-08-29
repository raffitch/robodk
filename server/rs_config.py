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
