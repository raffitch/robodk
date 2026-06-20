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
    state = {"cam": seed_T, "targets": {}}

    class FakeRdk:
        deleted: list = []
        applied = None
        def item_exists(self, name): return True
        def apply_run_mode(self, mode=None): return "run_robot"
        def use_camera_tool(self, tool): return X_true       # tool mount = truth
        def tcp_pose_T(self): return state["cam"]
        def current_joints(self): return "HOME"
        def move_j_joints(self, j): state["cam"] = seed_T    # return-to-start
        def is_reachable(self, T): return True
        def list_targets(self, prefix=""):
            return sorted(n for n in state["targets"] if n.startswith(prefix))
        def add_target(self, name, T): state["targets"][name] = T
        def delete_items(self, names):
            FakeRdk.deleted = list(names)
            for n in list(names):
                state["targets"].pop(n, None)
        def move_j(self, name): state["cam"] = state["targets"][name]
        def set_tool_pose(self, tool, T): FakeRdk.applied = (tool, T)
    rdk = FakeRdk()

    class FakeCamera:
        def grab(self, with_depth=False, timeout=None, color_only=False):
            return SimpleNamespace(color=np.zeros((720, 1280, 3), np.uint8),
                                   depth=None, timestamp=0.0)

    class FakeBoard:
        board_center = np.zeros(3)        # synthetic board points are centred at origin
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
    services = SimpleNamespace(config=cfg, rdk=rdk, camera=FakeCamera(),
                               bus=SimpleNamespace(publish=lambda *a, **k: None),
                               live=SimpleNamespace(running=False, stop=lambda: None))
    return services, rdk, X_true, state


def test_generate_then_run_recovers_truth():
    services, rdk, X_true, _state = _build_fakes()

    gen = service_mod.generate_calibration_targets(services)
    assert gen["created"] == 15
    assert gen["gate"]["ok"] is True
    assert all(n.startswith("TasniCalib_") for n in gen["targets"])
    assert abs(gen["look_distance_mm"] - 450) < 5

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


if __name__ == "__main__":
    test_generate_then_run_recovers_truth()
    test_generate_refuses_when_not_aimed()
    test_run_without_targets_errors()
    print("\nGate-gated calibration job tests passed.")
