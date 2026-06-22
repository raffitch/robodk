"""Dry-tour (SimTourJob) unit test with a fake RoboDK — no station, no camera.

Drives the soft-gate path: visit each TasniCalib_* target in SIMULATE mode, record
per-pose reachable / collision, return to start, and — critically — restore the run
mode that was active *before* the tour (a dry run must never leave the cell silently
in RUN_ROBOT). Also checks an unreachable pose is flagged, a collision fails its pose,
and collisions degrade to "not checked" when the build can't report them.

    py -3.10 tests/test_sim_tour.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core.config import AppConfig  # noqa: E402
from tasni.core.jobrunner import JobContext  # noqa: E402
from tasni.modules.calibration import service as service_mod  # noqa: E402

TARGETS = [f"TasniCalib_{i:02d}" for i in range(1, 9)]
RUN_ROBOT, SIMULATE = 6, 1


class _Ctx(JobContext):
    def __init__(self): self.logs = []
    def progress(self, *a, **k): pass
    def log(self, m): self.logs.append(m)
    def frame(self, *a, **k): pass
    def check_cancel(self): pass


class FakeRdk:
    """Minimal RoboDK fake. ``target_pose_T`` returns the name as the 'pose' so
    ``is_reachable``/``collisions`` can key off the pose being visited."""

    def __init__(self, *, unreachable=(), collisions=None, collisions_on=True,
                 prior_mode=RUN_ROBOT, joint_targets=None, transit=None):
        self._unreachable = set(unreachable)
        self._collisions = collisions          # name -> resting colliding-pair count
        self._collisions_on = collisions_on
        self._joint_targets = joint_targets or {}   # name -> stored joint vector
        self._transit = transit or {}          # dest-joints -> swept colliding-pair count
        self.run_mode = prior_mode             # the cell was left in this mode
        self.mode_calls: list = []
        self.moved: list[str] = []
        self.returned = False
        self.collision_toggles: list[bool] = []
        self._current = None

    def item_exists(self, name): return True
    def list_targets(self, prefix=""):
        return sorted(n for n in TARGETS if n.startswith(prefix))
    def current_run_mode(self): return self.run_mode
    def set_run_mode_raw(self, v): self.run_mode = int(v)
    def apply_run_mode(self, mode=None):
        self.run_mode = SIMULATE if mode == "simulate" else RUN_ROBOT
        self.mode_calls.append(mode)
        return mode
    def use_camera_tool(self, tool): return np.eye(4)
    def set_collision_checking(self, active):
        self.collision_toggles.append(active)
        return self._collisions_on
    def ensure_mounted_tool_collision_pairs(self, skip_trailing=2):
        self.guard_skip = skip_trailing
        return {"tools": ["Realsense"], "links": [0, 1, 2, 3, 4],
                "pairs_enabled": 5, "pairs_failed": 0, "dof": 6}
    def collisions(self):
        if not self._collisions_on or self._collisions is None:
            return None
        return self._collisions.get(self._current, 0)
    def current_joints(self): return "START"
    def is_reachable(self, pose): return pose not in self._unreachable
    def target_pose_T(self, name):
        self._current = name
        return name
    def move_j(self, name):
        self._current = name
        self.moved.append(name)
    def move_j_joints(self, joints): self.returned = True
    def target_joints(self, name): return self._joint_targets.get(name)
    def move_j_test(self, j1, j2, step_deg=None):
        return self._transit.get(j2, 0)        # j2 is the destination joint vector


def _run(rdk) -> dict:
    services = SimpleNamespace(config=AppConfig(), rdk=rdk)
    return service_mod.SimTourJob(services)(_Ctx())


def test_all_reachable_passes():
    rdk = FakeRdk()
    out = _run(rdk)
    assert out["kind"] == "sim_tour"
    assert out["total"] == 8 and out["passed"] == 8
    assert out["unreachable"] == 0 and out["collisions"] == 0
    assert out["collisions_checked"] is True
    assert out["returned_to_start"] is True and out["all_ok"] is True
    assert rdk.moved == TARGETS            # visited every target in order
    assert rdk.returned is True
    # SIMULATE during the tour, but the prior RUN_ROBOT mode is restored after.
    assert "simulate" in rdk.mode_calls
    assert rdk.run_mode == RUN_ROBOT
    assert rdk.collision_toggles[-1] is False   # checking turned back off
    print("[all reachable] 8/8 OK, mode restored to RUN_ROBOT")


def test_unreachable_pose_flagged_but_mode_restored():
    rdk = FakeRdk(unreachable={"TasniCalib_04"})
    out = _run(rdk)
    assert out["passed"] == 7 and out["unreachable"] == 1
    assert out["all_ok"] is False
    bad = [p for p in out["poses"] if not p["ok"]]
    assert len(bad) == 1 and bad[0]["name"] == "TasniCalib_04"
    assert bad[0]["reachable"] is False
    assert "TasniCalib_04" not in rdk.moved        # never moved to an unreachable pose
    assert rdk.run_mode == RUN_ROBOT               # still restored on a failed tour
    print("[unreachable] flagged TasniCalib_04, mode restored")


def test_collision_fails_its_pose():
    rdk = FakeRdk(collisions={"TasniCalib_05": 1})
    out = _run(rdk)
    assert out["collisions"] == 1 and out["passed"] == 7
    bad = [p for p in out["poses"] if not p["ok"]]
    assert len(bad) == 1 and bad[0]["name"] == "TasniCalib_05"
    assert bad[0]["reachable"] is True and bad[0]["collision"] is True
    print("[collision] TasniCalib_05 reachable but colliding -> fail")


def test_transit_collision_flags_pose():
    """A pose whose APPROACH sweep collides (clears the endpoints but the tool
    passes through an arm link mid-move) is flagged even if its resting pose is
    clear — the inter-target path the real run actually drives."""
    rdk = FakeRdk(joint_targets={n: n for n in TARGETS},   # each target -> its own config
                  transit={"TasniCalib_03": 1})            # sweep INTO _03 collides
    out = _run(rdk)
    assert out["transit_collisions"] == 1
    assert out["all_ok"] is False
    bad = [p for p in out["poses"] if not p["ok"]]
    assert len(bad) == 1 and bad[0]["name"] == "TasniCalib_03"
    assert bad[0]["transit"] is True and bad[0]["reachable"] is True
    print("[transit] TasniCalib_03 approach collides -> flagged")


def test_collisions_not_supported_degrades():
    rdk = FakeRdk(collisions_on=False)
    out = _run(rdk)
    assert out["collisions_checked"] is False
    assert out["passed"] == 8 and out["all_ok"] is True
    assert all(p["collision"] is None for p in out["poses"])
    print("[no collision support] passes on reachability, collisions reported None")


def test_no_targets_raises():
    global TARGETS
    saved, TARGETS = TARGETS, []
    try:
        _run(FakeRdk())
        raise AssertionError("expected a refusal with no targets")
    except RuntimeError as e:
        assert "Create targets" in str(e) or "no TasniCalib" in str(e)
        print("[no targets]", str(e).splitlines()[0])
    finally:
        TARGETS = saved


if __name__ == "__main__":
    test_all_reachable_passes()
    test_unreachable_pose_flagged_but_mode_restored()
    test_collision_fails_its_pose()
    test_transit_collision_flags_pose()
    test_collisions_not_supported_degrades()
    test_no_targets_raises()
    print("\nSim-tour dry-run tests passed.")
