"""Job-level test for the new gate-gated flow (service.py) with fake core
services — no RoboDK, no camera. Drives the real path:

    generate_calibration_targets (gate check -> seed=current pose -> reachable
    TasniCalib_* targets) -> CalibrationJob visits them -> solve -> apply

and checks the solved tool pose matches ground truth, that targets persist after
the run (they are user-created now, not temp), and that generation refuses unless
the board is in the ideal band.

    py -3.10 tests/test_calibration_job.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasni.core.camera_lease import CameraLease  # noqa: E402
from tasni.core.config import AppConfig  # noqa: E402
from tasni.core.geometry import Rt_to_T, compose, invert_T  # noqa: E402
from tasni.core.jobrunner import JobContext  # noqa: E402
from tasni.modules.calibration import service as service_mod  # noqa: E402
from tasni.modules.calibration.charuco import ViewDetection  # noqa: E402
import test_calibration_synthetic as syn  # noqa: E402

K, DIST = syn.K, syn.DIST
BOARD_CENTER = np.array([800.0, 0.0, 200.0])


class _Ctx(JobContext):
    def __init__(self): self.frames = 0
    def progress(self, *a, **k): pass
    def log(self, *a, **k): pass
    def frame(self, *a, **k): self.frames += 1
    def check_cancel(self): pass


def _build_fakes():
    # Seed 450 mm from the board, fronto-parallel -> inside the ideal gate band.
    seed_pos = np.array([350.0, 0.0, 200.0])
    seed_T = syn._look_at(seed_pos, BOARD_CENTER, 0.0)
    T_base_target = syn._look_at(BOARD_CENTER, seed_pos, 0.0)   # board faces the camera
    X_true = Rt_to_T(syn._rot([0.3, 0.2, 1.0], 25), [40.0, -15.0, 55.0])  # cam2flange (= tool)
    obj = syn._make_board_points()
    state = {"cam": seed_T, "targets": {}, "joints": {}}

    class FakeRdk:
        deleted: list = []
        applied = None
        def item_exists(self, name): return True
        def apply_run_mode(self, mode=None): return "run_robot"
        def connect_robot(self, ip="", *, timeout_s=10.0, poll_s=0.4):
            return True, "ROBOTCOM_READY"        # real robot link is up in the fake
        def robot_connection_params(self): return {"ip": "10.0.0.5", "port": 7000}
        def use_camera_tool(self, tool): return X_true       # tool mount = truth
        def tcp_pose_T(self): return state["cam"]
        def camera_pose_T(self): return state["cam"]         # camera (TCP) in base
        def flange_pose_T(self):                             # base->flange = cam @ inv(mount)
            return compose(state["cam"], invert_T(X_true))
        def current_joints(self): return "HOME"
        def move_j_joints(self, j): state["cam"] = seed_T    # return-to-start
        def is_reachable(self, T): return True
        def screen_collisions(self, poses, *, guard_skip=None, **kw):
            # No station collision map in the fake -> "not checked", drop nothing,
            # no locked joints from the sweep -> generation must back-fill joints
            # via solve_joints_for_pose so every target is still joint-locked.
            # **kw absorbs obstacle_pairs/baseline_relative/path_samples.
            return [True] * len(poses), False, [None] * len(poses)
        def solve_joints_for_pose(self, T, seed=None):
            return ("joints", float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
        def ensure_mounted_tool_collision_pairs(self, skip_trailing=2):
            self.guard_skip = skip_trailing          # record on the INSTANCE
            return {"tools": ["Realsense", "Spindle"], "links": [0, 1, 2, 3, 4],
                    "pairs_enabled": 10, "pairs_failed": 0, "dof": 6}
        def list_targets(self, prefix=""):
            return sorted(n for n in state["targets"] if n.startswith(prefix))
        def add_target(self, name, T, joints=None):
            state["targets"][name] = T
            state["joints"][name] = joints
        def delete_items(self, names):
            FakeRdk.deleted = list(names)
            for n in list(names):
                state["targets"].pop(n, None)
        def move_j(self, name): state["cam"] = state["targets"][name]
        def set_tool_pose(self, tool, T): FakeRdk.applied = (tool, T)
        def add_keepout_box(self, name, board_pts_base, *, margin_mm, above_mm,
                            depth_mm, parent=None, color=None):
            FakeRdk.keepout = (name, np.asarray(board_pts_base).shape)
            return SimpleNamespace(Valid=lambda: True)
    rdk = FakeRdk()

    class FakeCamera:
        def grab(self, with_depth=False, timeout=None, color_only=False):
            return SimpleNamespace(color=np.zeros((720, 1280, 3), np.uint8),
                                   depth=None, timestamp=0.0)

    class FakeBoard:
        board_center = np.zeros(3)        # synthetic board points are centred at origin
        all_obj_points = obj              # board-frame corner cloud (visibility filter)
        def __init__(self, cfg): pass
        def detect(self, image, K, dist, *, min_corners=6):
            from tasni.modules.calibration.handeye import reproject
            T_cam_target = compose(invert_T(state["cam"]), T_base_target)
            corners = reproject(obj, T_cam_target, K, dist).reshape(-1, 1, 2)
            rvec, _ = cv2.Rodrigues(T_cam_target[:3, :3])
            return ViewDetection(corners.astype(np.float32),
                                 np.arange(obj.shape[0]).reshape(-1, 1).astype(np.int32),
                                 obj.astype(np.float32), rvec,
                                 T_cam_target[:3, 3].reshape(3, 1))
        def detect_median(self, images, K, dist, *, min_corners=6, min_frac=0.5):
            return self.detect(images[0], K, dist, min_corners=min_corners)
        def annotate(self, img, det, K, dist, label=""): return img
    service_mod.CharucoTarget = FakeBoard

    cfg = AppConfig()
    cfg.calibration.pose_count = 15
    cfg.calibration.holdout_count = 3
    # Isolate the hand-eye tests from the auto-intrinsics step (which writes config
    # + a marker); test_auto_intrinsics_* covers that path explicitly.
    cfg.calibration.auto_intrinsics = False
    services = SimpleNamespace(config=cfg, rdk=rdk, camera=FakeCamera(),
                               camera_lease=CameraLease(),
                               bus=SimpleNamespace(publish=lambda *a, **k: None),
                               live=SimpleNamespace(running=False, stop=lambda: None))
    return services, rdk, X_true, state


def test_generate_then_run_recovers_truth():
    services, rdk, X_true, state = _build_fakes()

    gen = service_mod.generate_calibration_targets(services)
    assert gen["created"] == 15
    assert gen["gate"]["ok"] is True
    assert all(n.startswith("TasniCalib_") for n in gen["targets"])
    assert abs(gen["look_distance_mm"] - 450) < 5

    # Every target is joint-locked to the camera TCP (back-filled here since the
    # fake reports no collision-map joints) — so selecting one drives the CAMERA,
    # not the flange, to the viewpoint regardless of the GUI's active tool.
    assert gen["targets_joint_locked"] == 15 and gen["targets_cartesian"] == 0
    assert all(state["joints"][n] is not None for n in gen["targets"])

    with tempfile.TemporaryDirectory() as tmp:
        service_mod.new_run_dir = lambda mid, stamp: Path(tmp)
        job = service_mod.CalibrationJob(services, service_mod.CalibrationParams())
        result = job(_Ctx())

        assert result["can_apply"] is True
        assert result["tool_name"] == "Realsense"
        assert result["n_captured"] == 15
        assert result["report"]["validation"]["n_views"] == 3
        assert result["report"]["train"]["rms_px"] < 1.0
        assert (Path(tmp) / "report.json").exists()

        # Phase-1 report additions.
        rep = result["report"]
        assert rep["method"] in {"TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"}
        assert rep["cross_val_rms_px"] is not None and rep["cross_val_rms_px"] < 1.0
        assert rep["intrinsics_check"] is not None
        assert rep["intrinsics_check"]["warn"] is False
        md = rep["motion_diversity"]
        assert md["n_pairs"] > 0 and md["max_pair_deg"] > 0
        assert md["well_conditioned"] is True

        rot = float(np.rad2deg(np.linalg.norm(
            cv2.Rodrigues(job.solved_X[:3, :3].T @ X_true[:3, :3])[0])))
        assert rot < 0.5, f"solved tool rotation off by {rot:.3f} deg"
        assert np.linalg.norm(job.solved_X[:3, 3] - X_true[:3, 3]) < 0.5

        # targets are user-created now -> the run does NOT delete them
        assert len(rdk.list_targets("TasniCalib_")) == 15

        tool = job.apply_to_tool()
        assert tool == "Realsense" and rdk.applied[0] == "Realsense"
        print("[gate->generate->run]\n" + result["summary"])


def test_auto_intrinsics_runs_when_missing_and_skips_when_present():
    """With no intrinsics marker on file, the hand-eye run auto-calibrates K +
    distortion from its own captured views, applies them (marker written), and the
    report carries an ``intrinsics_auto`` block. With the marker present it skips."""
    from tasni.core import config as cfgmod

    services, rdk, _X, _state = _build_fakes()
    services.config.calibration.auto_intrinsics = True

    # Mock persistence so the test never touches the real config / runs tree.
    marker: dict = {}
    fake_runs = SimpleNamespace(
        read_active=lambda m: marker.get(m),
        write_active=lambda m, payload: marker.__setitem__(m, payload),
        load_report=lambda *a, **k: (_ for _ in ()).throw(KeyError()))
    orig_runs, orig_save = service_mod.runs, cfgmod.save_overrides
    service_mod.runs = fake_runs
    cfgmod.save_overrides = lambda updates: None
    try:
        service_mod.generate_calibration_targets(services)
        with tempfile.TemporaryDirectory() as tmp:
            service_mod.new_run_dir = lambda mid, stamp: Path(tmp)

            res = service_mod.CalibrationJob(services, service_mod.CalibrationParams())(_Ctx())
            ia = res["report"].get("intrinsics_auto")
            assert ia is not None and ia["source"] == "auto" and ia["n_views"] >= 6
            assert "intrinsics" in marker          # marker written -> now "present"
            assert service_mod.intrinsics_present(services) is True

            # Second run: marker present -> auto path skipped (no intrinsics_auto block).
            res2 = service_mod.CalibrationJob(services, service_mod.CalibrationParams())(_Ctx())
            assert res2["report"].get("intrinsics_auto") is None
    finally:
        service_mod.runs, cfgmod.save_overrides = orig_runs, orig_save
    print(f"[auto intrinsics] applied from {ia['n_views']} views, "
          f"fit RMS {ia['rms_px']:.3f}px; second run skipped it")


def test_generate_refuses_when_not_aimed():
    services, rdk, _X, state = _build_fakes()
    # Move the camera too far (700 mm) -> distance gate fails -> must refuse.
    state["cam"] = syn._look_at(np.array([100.0, 0.0, 200.0]), BOARD_CENTER, 0.0)
    msg = ""
    try:
        service_mod.generate_calibration_targets(services)
        raise AssertionError("expected generation to refuse an un-aimed board")
    except RuntimeError as e:
        msg = str(e)
        assert "distance" in msg
    assert rdk.list_targets("TasniCalib_") == []   # nothing created
    print("[gate refusal]", msg.splitlines()[0])


def test_run_without_targets_errors():
    services, _rdk, _X, _state = _build_fakes()
    job = service_mod.CalibrationJob(services, service_mod.CalibrationParams())
    msg = ""
    try:
        job(_Ctx())
        raise AssertionError("expected run to require targets")
    except RuntimeError as e:
        msg = str(e)
        assert "Create targets" in msg or "targets" in msg
    print("[run needs targets]", msg.splitlines()[0])


def test_generate_drops_colliding_poses():
    """Collision-filtered generation excludes poses where the mounted tooling
    collides (checked=True), and reports how many were dropped."""
    services, rdk, _X, _state = _build_fakes()

    def mask(poses, *, guard_skip=None, **kw):   # mark ~1/4 of candidates as colliding
        m = [True] * len(poses)
        for i in range(0, len(poses), 4):
            m[i] = False
        return m, True, [None] * len(poses)
    rdk.screen_collisions = mask

    gen = service_mod.generate_calibration_targets(services)
    assert gen["collisions_checked"] is True
    assert gen["candidates_collided"] > 0
    assert gen["created"] == 15      # enough collision-free poses survive
    print("[collision filter] dropped", gen["candidates_collided"],
          "of", gen["candidates_reachable"], "reachable")


def test_generate_reports_collision_guard():
    """Generation force-enables the mounted-tool<->arm collision pairs RoboDK omits
    by default, reports which tools were guarded, and forwards the CONFIGURED
    wrist-skip (use a non-default value so a hardcoded/default can't pass)."""
    services, rdk, _X, _state = _build_fakes()
    services.config.calibration.collision_skip_wrist_links = 3   # non-default

    gen = service_mod.generate_calibration_targets(services)
    g = gen["collision_guard"]
    assert g is not None and g["pairs_enabled"] == 10
    assert "Spindle" in g["tools"]
    assert rdk.guard_skip == 3          # the configured value reached the guard
    print("[collision guard] guarded", g["tools"], "skip=", rdk.guard_skip)


def test_generate_skips_guard_when_disabled():
    """With collision_self_pairs off, generation does not touch the collision map
    and reports no guard."""
    services, rdk, _X, _state = _build_fakes()
    services.config.calibration.collision_self_pairs = False
    rdk.guard_skip = "NOT-CALLED"       # sentinel the guard would overwrite if run

    gen = service_mod.generate_calibration_targets(services)
    assert gen["collision_guard"] is None
    assert rdk.guard_skip == "NOT-CALLED"   # guard genuinely not invoked
    print("[collision guard off] no map changes")


def test_generate_refuses_when_tooling_collides():
    """If too few collision-free poses survive the (baseline-relative) screen,
    generation refuses with guidance rather than creating unsafe targets — it never
    ships a pose that introduces a new collision."""
    services, rdk, _X, _state = _build_fakes()

    def mask(poses, *, guard_skip=None, **kw):   # only 2 free -> below MIN_TRAIN_VIEWS
        m = [False] * len(poses)
        m[0] = m[1] = True
        return m, True, [None] * len(poses)
    rdk.screen_collisions = mask

    msg = ""
    try:
        service_mod.generate_calibration_targets(services)
        raise AssertionError("expected refusal due to tooling collisions")
    except RuntimeError as e:
        msg = str(e)
        assert "collision-free" in msg and "Re-seed" in msg
    assert rdk.list_targets("TasniCalib_") == []   # nothing created
    print("[collision refusal]", msg.splitlines()[0])


def test_generate_refuses_when_all_poses_collide():
    """When every reachable candidate introduces a new collision, generation REFUSES
    (the silent ship-everything bypass is gone) — a colliding target is never created.
    Replaces the old 'bypass keeps it usable' behaviour."""
    services, rdk, _X, _state = _build_fakes()

    def mask(poses, *, guard_skip=None, **kw):
        return [False] * len(poses), True, [None] * len(poses)
    rdk.screen_collisions = mask

    msg = ""
    try:
        service_mod.generate_calibration_targets(services)
        raise AssertionError("expected refusal when all candidates collide")
    except RuntimeError as e:
        msg = str(e)
        assert "collision-free" in msg
    assert rdk.list_targets("TasniCalib_") == []   # nothing shipped
    print("[all collide -> refuse]", msg.splitlines()[0])


if __name__ == "__main__":
    test_generate_then_run_recovers_truth()
    test_auto_intrinsics_runs_when_missing_and_skips_when_present()
    test_generate_refuses_when_not_aimed()
    test_run_without_targets_errors()
    test_generate_drops_colliding_poses()
    test_generate_reports_collision_guard()
    test_generate_skips_guard_when_disabled()
    test_generate_refuses_when_tooling_collides()
    test_generate_refuses_when_all_poses_collide()
    print("\nGate-gated calibration job tests passed.")
