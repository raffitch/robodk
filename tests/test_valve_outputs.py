"""Valve outputs must reach the KUKA driver as NUMBERS, not names.

Cell 2026-08-28: the layer program aborted on the controller with
``KSS014444 Array index inadmissible $OUT[0]`` in module robodksync570 -- the
RoboDK KUKA *driver* module, used when running on the robot. setDO() was being
handed the string "IO_508"; a name the driver cannot resolve to an index becomes
$OUT[0], and KUKA's $OUT[] is 1-based, so index 0 is always invalid.

That is also why a dispatched layer "finished" in 0.5 s having never moved: it
faulted on the valve instruction and aborted before any motion.

    py -3.10 -m pytest tests/test_valve_outputs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.modules.extrusion.valve import (instructions_match,  # noqa: E402
                                           valve_output_index)


def test_a_named_output_yields_its_numeric_index():
    assert valve_output_index("IO_508") == 508
    assert valve_output_index("IO_601") == 601


def test_a_bare_number_is_already_an_index():
    assert valve_output_index("508") == 508
    assert valve_output_index(601) == 601


def test_other_prefixes_still_resolve():
    assert valve_output_index("DO_12") == 12
    assert valve_output_index("$OUT[7]") == 7


def test_index_zero_is_rejected_because_kuka_out_is_one_based():
    with pytest.raises(ValueError):
        valve_output_index("IO_0")


def test_an_unresolvable_name_is_an_error_not_a_silent_zero():
    """Silently sending 0 is exactly the bug: fail loudly instead."""
    with pytest.raises(ValueError):
        valve_output_index("AIR_VALVE")


def test_instructions_match_regardless_of_robodk_rendering():
    """RoboDK's instruction text is not a stable contract, so verify the numbers
    it contains rather than an exact string we would have to guess."""
    for rendered in (["Set 508=1", "Set 601=1"],
                     ["Set IO_508=1", "Set IO_601=1"],
                     ["setDO(508,1)", "setDO(601,1)"]):
        assert instructions_match(rendered, ["IO_508", "IO_601"], 1)


def test_instructions_do_not_match_when_the_value_is_wrong():
    assert not instructions_match(["Set 508=0", "Set 601=0"], ["IO_508", "IO_601"], 1)


def test_instructions_do_not_match_when_an_output_is_missing():
    assert not instructions_match(["Set 508=1"], ["IO_508", "IO_601"], 1)


def test_extra_instructions_are_rejected():
    """An AirOff that also moves the robot is not a fail-safe valve program."""
    assert not instructions_match(["Set 508=0", "Set 601=0", "MoveJ Home"],
                                  ["IO_508", "IO_601"], 0)


def test_station_requirements_always_reports_the_actual_valve_instructions():
    """A named and a numeric output render almost alike, so verification cannot
    tell a fixed station from a broken one. The operator can -- but only if the
    actual instruction text is always reported, not just when the check fails.
    """
    from tasni.core.config import ExtrusionConfig
    from tasni.modules.extrusion.service import station_requirements
    from tasni.modules.extrusion.models import CylinderRecipe, CylinderSetup
    from tasni.modules.extrusion.toolpath import generate_cylinder_plan

    class Rdk:
        def item_exists_as(self, name, kind): return True
        def program_instructions(self, name):
            return (["Set IO_508=1", "Set IO_601=1"] if name == "AirOn"
                    else ["Set IO_508=0", "Set IO_601=0"])

    plan = generate_cylinder_plan(
        CylinderRecipe(radius_mm=40, layer_count=1, layer_height_mm=5,
                       bead_diameter_mm=6, robot_speed_mm_s=75,
                       extrusion_rate_pct=0, points_per_circle=72),
        CylinderSetup(print_tool="T", work_frame="F", inspection_tool="C",
                      inspection_target="I", inspection_auto=False))
    report = station_requirements(Rdk(), plan, ExtrusionConfig())
    valve = next(i for i in report["items"]
                 if i["role"] == "valve_instruction_mapping")
    assert valve["actual"]["on"] == ["Set IO_508=1", "Set IO_601=1"]
    assert valve["actual"]["off"] == ["Set IO_508=0", "Set IO_601=0"]
