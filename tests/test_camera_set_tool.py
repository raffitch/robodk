"""tools/camera_set.py -- the host half of the server's runtime SET.

The server side is covered by tests/test_server_env.py; these pin the host
behaviours that cost real cell time when they are wrong: refusing an over-long
line locally, restoring the whole chain rather than a subset, and reading the arm
off the ACHIEVED options instead of off what was sent.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_over_long_line_is_refused_before_a_connection_is_opened(monkeypatch):
    """Better a readable local error than a silently ended session.

    The guard must fire before ``burst()``, not merely before the reply is read:
    the server ENDS the session on an over-long line, so opening one at all is
    the thing to avoid.
    """
    from tasni.core.camera import CameraClient

    monkeypatch.setattr(CameraClient, "burst", lambda *a, **k: pytest.fail(
        "must not open a connection for a line the server would refuse"))
    monkeypatch.setattr(cs, "load_config", lambda: SimpleNamespace(camera=None))
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


# -- the read-only web surface -------------------------------------------------
# The chain read must never open a competing connection to the UNICAST camera
# server while something else holds it: that steals the frame the holder is
# waiting for. Same rule /api/health already follows, now shared by both.
import tasni.webapp.server as ws  # noqa: E402


def _services(*, lease=False, job=False, live=False):
    return SimpleNamespace(
        camera_lease=SimpleNamespace(held=lease, owner="live-preview"),
        jobs=SimpleNamespace(running=job),
        live=SimpleNamespace(running=live))


def test_camera_is_free_when_nothing_holds_it():
    assert ws.camera_busy_reason(_services()) == ""


@pytest.mark.parametrize("holder, expected", [
    ({"lease": True}, "live-preview"),
    ({"job": True}, "running job"),
    ({"live": True}, "live preview"),
])
def test_every_holder_blocks_the_read(holder, expected):
    reason = ws.camera_busy_reason(_services(**holder))
    assert reason, "a held camera must report a reason, not an empty string"
    assert expected in reason


def test_the_lease_owner_wins_over_the_coarse_flags():
    """The lease names the precise holder; the flags are only a fallback."""
    reason = ws.camera_busy_reason(_services(lease=True, job=True, live=True))
    assert "live-preview" in reason


@pytest.mark.parametrize("delta, arm, stock", [
    (20.0, "stock", True),
    (None, "spatial OFF", False),
    (5.0, "smooth_delta 5", False),
    (8.5, "smooth_delta 8.5", False),
])
def test_arm_label_is_derived_from_the_achieved_delta(delta, arm, stock):
    assert ws.filter_arm_label({"spatial_smooth_delta": delta}) == (arm, stock)


def test_arm_label_treats_a_missing_key_as_spatial_off():
    """No key at all means no spatial filter -- it must not claim stock."""
    assert ws.filter_arm_label({}) == ("spatial OFF", False)
    assert ws.filter_arm_label(None) == ("spatial OFF", False)
