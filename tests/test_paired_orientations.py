"""Paired vertical/horizontal capture: one trip out, two camera orientations.

Every behaviour pinned here is one where the wrong choice still produces a run
that looks completely normal in the archive -- which is the only kind of bug
that matters for a measurement this expensive to repeat.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core.config import ExtrusionConfig  # noqa: E402
from tasni.modules.extrusion.inspection import pose_candidates  # noqa: E402
from tasni.modules.extrusion.measure import _roll_label  # noqa: E402
from tasni.modules.extrusion.service import wrist_allowance_deg  # noqa: E402

AIM = np.array([200.0, 150.0, 0.0])


def rolls_of(candidates):
    return [c["roll_deg"] for c in candidates]


# -- forcing a roll ------------------------------------------------------------

def test_config_ladder_is_used_when_no_roll_is_forced():
    cfg = ExtrusionConfig()
    assert rolls_of(pose_candidates(AIM, 300.0, cfg))[:4] == \
        cfg.inspection_roll_candidates_deg


def test_the_ladders_first_entry_is_zero_so_a_listed_roll_never_runs():
    """THE trap: 90 is already in the default ladder, and never reached.

    The generator accepts the first candidate that solves and roll 0 always
    does, so 'add 90 to the list' silently keeps capturing at 0.
    """
    cfg = ExtrusionConfig()
    assert cfg.inspection_roll_candidates_deg[0] == 0.0
    assert 90.0 in cfg.inspection_roll_candidates_deg


def test_a_forced_roll_replaces_the_ladder_entirely():
    """An unreachable roll must fail, never quietly capture at 0."""
    got = rolls_of(pose_candidates(AIM, 300.0, ExtrusionConfig(), None, rolls=[90.0]))
    assert set(got) == {90.0}, "no other roll may remain as a silent fallback"
    assert 0.0 not in got


def test_a_forced_roll_gets_NO_tilt_ladder():
    """A pair must differ by roll ALONE.

    Cell 2026-09-01: roll 90 at tilt 0 had no IK, the walk fell through to
    tilt 10 / azimuth 270, and that moved the camera 52 mm sideways and tilted
    it 10° -- collapsing the substrate separation and invalidating the
    characterization. A tilted "pair" is not a pair: incidence costs ~4x what
    distance costs.
    """
    cfg = ExtrusionConfig()
    got = pose_candidates(AIM, 300.0, cfg, None, rolls=[90.0],
                          tilts=[0.0], azimuths=[0.0])
    assert {c["tilt_deg"] for c in got} == {0.0}
    assert {c["azimuth_deg"] for c in got} == {0.0}


def test_opposite_roll_puts_the_baseline_on_the_same_axis():
    """Why θ-180 is a legitimate fallback and a tilt is not.

    Rolling by θ and θ-180 puts the camera X axis on the same LINE, reversed.
    The stereo baseline is an axis, not a direction, so the two are the same
    measurement -- which is what makes trying both safe when only one wrist
    direction solves.
    """
    a = pose_candidates(AIM, 300.0, ExtrusionConfig(), None, rolls=[90.0],
                        tilts=[0.0], azimuths=[0.0])[0]["T"]
    b = pose_candidates(AIM, 300.0, ExtrusionConfig(), None, rolls=[-90.0],
                        tilts=[0.0], azimuths=[0.0])[0]["T"]
    assert np.allclose(a[:3, 0], -b[:3, 0], atol=1e-9), "same line, reversed"
    assert np.allclose(a[:3, 2], b[:3, 2], atol=1e-9), "same optical axis"
    assert np.allclose(a[:3, 3], b[:3, 3], atol=1e-9), "same position"


def test_a_forced_roll_actually_rotates_the_pose():
    """Guards against the roll being recorded but not applied to the matrix."""
    flat = pose_candidates(AIM, 300.0, ExtrusionConfig(), None, rolls=[0.0])[0]["T"]
    rolled = pose_candidates(AIM, 300.0, ExtrusionConfig(), None, rolls=[90.0])[0]["T"]
    # same optical axis and same position -- only the roll about it differs
    assert np.allclose(flat[:3, 2], rolled[:3, 2], atol=1e-9)
    assert np.allclose(flat[:3, 3], rolled[:3, 3], atol=1e-9)
    cos = float(np.dot(flat[:3, 0], rolled[:3, 0]))
    assert abs(cos) < 1e-6, "a 90 deg roll must put the X axes at right angles"


# -- the wrist limit -----------------------------------------------------------

def _setup(limit=90.0):
    return SimpleNamespace(maximum_tool_axis_spin_deg=limit)


def test_no_forced_roll_keeps_todays_guard_exactly():
    cfg = ExtrusionConfig()
    assert wrist_allowance_deg(_setup(90.0), None, cfg) == 90.0


def test_a_commanded_90_gets_headroom_above_the_90_limit():
    """90 deg of roll costs ~90 deg of A6 and the gate is `> limit`, so a bare
    90 vs 90 is decided by floating point -- it was refused on this cell."""
    cfg = ExtrusionConfig()
    assert wrist_allowance_deg(_setup(90.0), 90.0, cfg) > 90.0


def test_headroom_is_relative_to_the_commanded_roll():
    cfg = ExtrusionConfig()
    margin = cfg.inspection_roll_wrist_margin_deg
    assert wrist_allowance_deg(_setup(90.0), 120.0, cfg) == pytest.approx(120.0 + margin)


def test_a_negative_roll_gets_the_same_headroom():
    cfg = ExtrusionConfig()
    assert wrist_allowance_deg(_setup(90.0), -90.0, cfg) == \
        wrist_allowance_deg(_setup(90.0), 90.0, cfg)


def test_headroom_never_tightens_a_looser_configured_limit():
    cfg = ExtrusionConfig()
    assert wrist_allowance_deg(_setup(150.0), 90.0, cfg) == 150.0


# -- operator-facing labels ----------------------------------------------------

@pytest.mark.parametrize("roll, label", [
    (None, "vertical"), (0.0, "vertical"),
    (90.0, "horizontal"), (-90.0, "horizontal"),
    (45.0, "roll 45°"),
])
def test_orientation_labels(roll, label):
    assert _roll_label(roll) == label


# -- which characterization seeds the recipe -----------------------------------
# A characterization sets the plan's radius, centre and layer height, so every
# later take is scored against it. Applying the ROLLED view would derive the
# whole geometry from the very orientation under test, and nothing downstream
# would notice -- the plan would simply be wrong for the rest of the trial.

def pick(characterizations):
    """The selection rule from measure/apply-characterization."""
    upright = [c for c in characterizations
               if c.get("orientation") in (None, "vertical")]
    return (upright or characterizations)[-1]


def test_apply_takes_the_vertical_view_not_the_last_one():
    chosen = pick([{"index": 1, "orientation": "vertical"},
                   {"index": 2, "orientation": "horizontal"}])
    assert chosen["index"] == 1


def test_apply_still_takes_the_last_when_none_are_labelled():
    """Sessions predating paired capture carry no orientation at all."""
    assert pick([{"index": 1}, {"index": 2}])["index"] == 2


def test_apply_takes_the_newest_vertical_when_several_exist():
    chosen = pick([{"index": 1, "orientation": "vertical"},
                   {"index": 2, "orientation": "horizontal"},
                   {"index": 3, "orientation": "vertical"}])
    assert chosen["index"] == 3


def test_apply_falls_back_rather_than_crashing_on_only_rolled_views():
    """If the vertical view failed and only a rolled one landed, applying a
    wrong-but-present recipe beats an exception -- but it must be possible to
    see which it was, which is why orientation is carried on the summary."""
    chosen = pick([{"index": 2, "orientation": "horizontal"}])
    assert chosen["index"] == 2 and chosen["orientation"] == "horizontal"
