"""tools/inspection_roll.py -- arming a forced camera roll.

Both behaviours pinned here are silent-failure modes: a run that looks ordinary
and answers nothing. They cost a whole stack and a cell session to discover
afterwards, and nothing downstream flags them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.inspection_roll as ir  # noqa: E402


@pytest.fixture
def config(tmp_path, monkeypatch):
    path = tmp_path / "tasni.config.json"
    path.write_text(json.dumps({"extrusion": {"hardware_io_test_approved": True}}),
                    encoding="utf-8")
    monkeypatch.setattr(ir, "CONFIG", path)
    return path


def armed(path) -> list | None:
    return json.loads(path.read_text(encoding="utf-8"))["extrusion"].get(ir.KEY)


def test_arming_writes_exactly_one_candidate(config, monkeypatch):
    """A leftover fallback would silently capture at roll 0 and look ordinary."""
    monkeypatch.setattr(sys, "argv", ["inspection_roll", "90"])
    ir.main()
    assert armed(config) == [90.0], "a forced roll must be the ONLY candidate"


def test_arming_does_not_leave_the_ladder_behind(config, monkeypatch):
    """The default ladder makes roll 0 win, so 90 in a list is never reached."""
    monkeypatch.setattr(sys, "argv", ["inspection_roll", "90"])
    ir.main()
    assert 0.0 not in armed(config)
    assert len(armed(config)) == 1


def test_disarm_removes_the_key_rather_than_writing_the_ladder(config, monkeypatch):
    """Restoring the DEFAULT means absence: writing [0,180,90,270] explicitly would
    freeze today's ladder into the config and silently ignore a future change."""
    monkeypatch.setattr(sys, "argv", ["inspection_roll", "90"])
    ir.main()
    monkeypatch.setattr(sys, "argv", ["inspection_roll", "--disarm"])
    ir.main()
    assert armed(config) is None


def test_arming_preserves_the_rest_of_the_config(config, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["inspection_roll", "60"])
    ir.main()
    data = json.loads(config.read_text(encoding="utf-8"))
    assert data["extrusion"]["hardware_io_test_approved"] is True


def test_a_roll_on_the_wrist_limit_is_warned_about(config, monkeypatch, capsys):
    """90 deg lands exactly on max_tool_axis_spin_deg's default and was refused
    on this cell before; arming it silently would repeat that."""
    monkeypatch.setattr(sys, "argv", ["inspection_roll", "90"])
    ir.main()
    out = capsys.readouterr().out
    assert "WARNING" in out and "wrist limit" in out
    assert "maximum_tool_axis_spin_deg" in out, "must say WHERE to add headroom"


def test_a_roll_well_inside_the_limit_is_not_warned_about(config, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["inspection_roll", "60"])
    ir.main()
    assert "WARNING" not in capsys.readouterr().out


def test_every_arming_path_says_to_restart(config, monkeypatch, capsys):
    """The config is read at startup; arming without a restart changes nothing."""
    for argv in (["inspection_roll", "90"], ["inspection_roll", "--disarm"]):
        monkeypatch.setattr(sys, "argv", argv)
        ir.main()
        assert "RESTART" in capsys.readouterr().out
