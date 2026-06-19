"""Job-level test for the auto-generate calibration orchestration (service.py)
with fake core services — no RoboDK, no camera. Drives the real generate ->
reachability -> capture -> solve -> cleanup -> apply path and checks the result,
that the solved tool pose matches ground truth, and that temp targets are deleted.

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


class _Ctx(JobContext):
    def __init__(self): self.frames = 0
    def progress(self, *a, **k): pass
    def log(self, *a, **k): pass
    def frame(self, *a, **k): self.frames += 1
    def check_cancel(self): pass


def _build_fakes():
    board_center = np.array([800.0, 0.0, 200.0])
    seed_pos = np.array([300.0, 0.0, 560.0])
    seed_T = syn._look_at(seed_pos, board_center, 0.0)
    T_base_target = syn._look_at(board_center, seed_pos, 0.0)
    X_true = Rt_to_T(syn._rot([0.3, 0.2, 1.0], 25), [40.0, -15.0, 55.0])  # cam2flange (= tool)
    obj = syn._make_board_points()
    state = {"cam": seed_T, "targets": {}}

    class FakeRdk:
        deleted: list = []
        applied = None
        def item_exists(self, name): return True
        def apply_run_mode(self, mode=None): return "run_robot"
        def use_tool_and_frame(self, tool, frame_of=None): return X_true  # tool mount = truth
        def move_j(self, name):
            state["cam"] = seed_T if name == "NEUTRAL" else state["targets"][name]
        def tcp_pose_T(self): return state["cam"]
        def is_reachable(self, T): return True
        def list_targets(self, prefix=""): return [n for n in state["targets"] if n.startswith(prefix)]
        def add_target(self, name, T): state["targets"][name] = T
        def delete_items(self, names):
            FakeRdk.deleted = list(names)
            for n in names:
                state["targets"].pop(n, None)
        def set_tool_pose(self, tool, T): FakeRdk.applied = (tool, T)
    rdk = FakeRdk()

    class FakeCamera:
        def grab(self, with_depth=False):
            return SimpleNamespace(color=np.zeros((1080, 1920, 3), np.uint8),
                                   depth=None, timestamp=0.0)

    class FakeBoard:
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
        def annotate(self, img, det, K, dist, label=""): return img
    service_mod.CharucoTarget = FakeBoard

    cfg = AppConfig()
    cfg.calibration.pose_count = 15
    cfg.calibration.holdout_count = 3
    services = SimpleNamespace(config=cfg, rdk=rdk, camera=FakeCamera(),
                               bus=SimpleNamespace(publish=lambda *a, **k: None))
    return services, rdk, X_true


def test_autogenerate_job():
    services, rdk, X_true = _build_fakes()
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

        # solved tool pose recovers ground-truth cam2flange
        rot = float(np.rad2deg(np.linalg.norm(
            cv2.Rodrigues(job.solved_X[:3, :3].T @ X_true[:3, :3])[0])))
        assert rot < 0.5, f"solved tool rotation off by {rot:.3f} deg"
        assert np.linalg.norm(job.solved_X[:3, 3] - X_true[:3, 3]) < 0.5

        # temp targets were created then deleted
        assert len(rdk.deleted) == 15
        assert all(n.startswith("TasniCalib_") for n in rdk.deleted)

        tool = job.apply_to_tool()
        assert tool == "Realsense" and rdk.applied[0] == "Realsense"
        print("[auto job]\n" + result["summary"])


def test_preview_mode_generates_but_does_not_solve():
    services, rdk, _ = _build_fakes()
    job = service_mod.CalibrationJob(services, service_mod.CalibrationParams(mode="preview"))
    result = job(_Ctx())
    assert result["mode"] == "preview"
    assert result["n_poses"] == 15
    assert result["can_apply"] is False
    assert job.solved_X is None          # no solve in preview
    assert rdk.deleted == []             # preview LEAVES the temp targets
    print("[preview]", result["summary"])


if __name__ == "__main__":
    test_autogenerate_job()
    test_preview_mode_generates_but_does_not_solve()
    print("\nAuto-generate + preview calibration job tests passed.")
