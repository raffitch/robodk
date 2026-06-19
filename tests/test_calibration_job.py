"""Job-level test for the calibration orchestration (service.py) with fake core
services — no RoboDK, no camera. Replays the synthetic dome through the real
capture/split/solve/metrics/artifact path and checks the outcome + apply step.

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

from tasni.core.config import AppConfig  # noqa: E402
from tasni.core.jobrunner import JobContext  # noqa: E402
from tasni.modules.calibration import service as service_mod  # noqa: E402
from tasni.modules.calibration.charuco import ViewDetection  # noqa: E402
import test_calibration_synthetic as syn  # noqa: E402


class _FakeCtx(JobContext):
    def __init__(self):
        self.frames = 0
        self.logs = []

    def progress(self, *a, **k): pass
    def log(self, message): self.logs.append(message)
    def frame(self, jpeg_bytes): self.frames += 1
    def check_cancel(self): pass


class _FakeBus:
    def publish(self, *a, **k): pass


def _build_fakes():
    _, _, views = syn._build_views(n=12)
    by_name = {v.name: v for v in views}
    state = {"current": None}

    class FakeRdk:
        applied_tool = None
        def apply_run_mode(self, mode=None): return mode or "simulate"
        def list_targets(self): return [v.name for v in views]
        def move_j(self, name): state["current"] = name
        def target_pose_T(self, name): return by_name[name].T_base_gripper
        def set_tool_pose(self, tool, T): FakeRdk.applied_tool = (tool, T)
    rdk = FakeRdk()

    class FakeCamera:
        def grab(self, with_depth=False):
            return SimpleNamespace(
                color=np.zeros((1080, 1920, 3), np.uint8), depth=None, timestamp=0.0)
    cam = FakeCamera()

    # Replace ChArUco detection with a replay keyed off the moved-to target.
    class FakeBoard:
        def __init__(self, cfg): pass
        def detect(self, image, K, dist, *, min_corners=6):
            v = by_name[state["current"]]
            n = v.obj_points.shape[0]
            rvec, _ = cv2.Rodrigues(v.R_target2cam)
            return ViewDetection(
                corners=v.corners.astype(np.float32),
                ids=np.arange(n).reshape(-1, 1).astype(np.int32),
                obj_points=v.obj_points.astype(np.float32),
                rvec=rvec, tvec=v.t_target2cam.reshape(3, 1))
        def annotate(self, img, det, K, dist, label=""):
            return img
    service_mod.CharucoTarget = FakeBoard

    cfg = AppConfig()
    cfg.calibration.holdout_count = 3
    services = SimpleNamespace(config=cfg, rdk=rdk, camera=cam, bus=_FakeBus())
    return services, rdk


def test_job_runs_and_reports():
    services, rdk = _build_fakes()
    with tempfile.TemporaryDirectory() as tmp:
        # send artifacts to a temp dir
        service_mod.new_run_dir = lambda mid, stamp: Path(tmp)
        params = service_mod.CalibrationParams(tool_name="Eye", save_frames=True)
        job = service_mod.CalibrationJob(services, params)
        result = job(_FakeCtx())

        assert result["can_apply"] is True
        assert result["n_captured"] == 12
        assert result["report"]["validation"]["n_views"] == 3
        assert result["report"]["train"]["n_views"] == 9
        # Clean synthetic data -> small reprojection error after the full path.
        assert result["report"]["train"]["rms_px"] < 1.0
        assert result["report"]["validation"]["rms_px"] < 2.0
        assert (Path(tmp) / "report.json").exists()
        assert (Path(tmp) / "summary.txt").exists()

        # apply step writes into the tool
        tool = job.apply_to_tool()
        assert tool == "Eye"
        assert rdk.applied_tool[0] == "Eye"
        assert np.allclose(rdk.applied_tool[1], job.solved_X)
        print("[job]\n" + result["summary"])


if __name__ == "__main__":
    test_job_runs_and_reports()
    print("\nCalibration job test passed.")
