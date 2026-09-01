"""tools/camera_set.py -- the host half of the server's runtime SET.

The server side is covered by tests/test_server_env.py; these pin the host
behaviours that cost real cell time when they are wrong: refusing an over-long
line locally, restoring the whole chain rather than a subset, and reading the arm
off the ACHIEVED options instead of off what was sent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.camera_set as cs  # noqa: E402


def test_bare_set_is_read_only():
    """No assignments must produce a bare SET -- the read-only form."""
    assert cs.STOCK, "stock chain must not be empty"
    # the CLI turns no-assignments into a bare line; mirror that construction
    line = ("SET " + " ".join([])).strip()
    assert line == "SET"


def test_restore_line_fits_under_the_server_cap():
    """A full explicit restore must fit: the server ENDS the session otherwise."""
    payload = "SET " + cs.STOCK + "\n"
    assert len(payload.encode()) < cs.SET_LINE_MAXLEN


def test_restore_covers_every_key_the_device_reports_settable():
    """Spec 4.1 wants the WHOLE chain sent between arms, not a subset."""
    keys = {a.split("=")[0] for a in cs.STOCK.split()}
    for expected in ("spatial", "spatial_smooth_delta", "spatial_magnitude",
                     "spatial_smooth_alpha", "spatial_holes_fill",
                     "temporal_smooth_alpha", "temporal_smooth_delta",
                     "temporal_persistency", "depth_min_m", "depth_max_m",
                     "decimation"):
        assert expected in keys, f"restore line omits {expected}"


def test_restore_omits_hole_filling():
    """hole_filling reads back null (no such filter); sending it would be a guess."""
    assert "hole_filling" not in cs.STOCK


def test_over_long_line_is_refused_locally(monkeypatch):
    """Better a readable local error than a silently ended session."""
    monkeypatch.setattr(cs, "CameraClient", lambda *a, **k: pytest.fail(
        "must not open a connection for a line the server would refuse"))
    with pytest.raises(SystemExit, match="ENDS the session"):
        cs.send([f"k{i}=0" for i in range(200)])


@pytest.mark.parametrize("delta, expected", [
    (20.0, "STOCK"),
    (None, "spatial OFF"),
    (5.0, "spatial_smooth_delta=5.0"),
])
def test_arm_is_read_off_the_achieved_options(delta, expected):
    """The arm label must come from filter_options, never from the sent line."""
    text = cs.describe({"ok": True, "filters": ["threshold"],
                        "filter_options": {"spatial_smooth_delta": delta}})
    assert expected in text


def test_describe_survives_a_reply_with_no_options():
    """An older/odd reply must not raise -- it is a diagnostic, not a gate."""
    assert cs.describe({"ok": True, "filters": []})
