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
    """Set emitter on, optional laser/preset and depth_units; return the ACHIEVED
    values (read back) keyed by option name, plus ``depth_unit_mm``.

    AE priority is NOT a depth option on this family -- see configure_color_sensor."""
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
    # NOTE: auto_exposure_priority is NOT set here. On the D400 series it is
    # registered on the COLOUR endpoint (librealsense src/ds5/ds5-color.cpp:161,
    # color_ep.register_pu(RS2_OPTION_AUTO_EXPOSURE_PRIORITY)), so asking the depth
    # sensor for it only ever logged "unsupported on this device/build - skipped" --
    # a misleading line, and the 30 fps guard was never actually in place. See
    # configure_color_sensor() below.
    try:
        achieved["depth_unit_mm"] = float(sensor.get_depth_scale()) * 1000.0
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: could not read depth scale: {e}")
        achieved["depth_unit_mm"] = None
    log(f"RealSense: depth_unit_mm = {achieved['depth_unit_mm']}")
    return achieved


def configure_color_sensor(sensor, rs, *, log=print) -> dict:
    """Pin the COLOUR stream's frame rate ahead of its exposure; return the ACHIEVED
    values (read back) keyed by option name.

    Frame rate is a contract for every client of the shared pipeline. With
    ``auto_exposure_priority`` on, AE is free to stretch exposure past the frame
    period in dim light and the sensor drops below 30 fps, which stalls
    ``wait_for_frames`` and costs the pipeline a recovery rebuild -- the colour
    stream being the one running at 1920x1080, so the one closest to its deadline.

    This option lives on the colour endpoint on the D400 series, not the depth one
    (librealsense src/ds5/ds5-color.cpp:161). It is a UVC passthrough option, so
    librealsense does not force a default and the endpoint inherits whatever the
    device had -- hence the PREVIOUS value is logged before it is changed: this is a
    live behavioural change to colour exposure and the operator has to be able to
    see what it was.

    Same contract as configure_depth_sensor: never raise over one option (the unit
    is Restart=always with no start limit, so an exception on the startup path is an
    infinite crash-loop with the camera dark).
    """
    achieved: dict = {}
    # Labelled "colour ..." in the journal on purpose: the old line ("auto_exposure_
    # priority unsupported on this device/build - skipped", emitted while asking the
    # DEPTH sensor) read as "this device cannot do it" when the truth was "we asked
    # the wrong endpoint". Which endpoint was asked is now in the log itself.
    label = "colour auto_exposure_priority"
    option = _opt(rs, "auto_exposure_priority")
    previous = _read(sensor, label, option, log)
    if previous is None:
        log(f"RealSense: {label} was not readable before the change")
    else:
        log(f"RealSense: {label} was {previous:g} before this change")
    achieved["auto_exposure_priority"] = _set_with_readback(sensor, label, option, 0.0, log)
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
