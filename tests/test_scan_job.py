"""Job-level scan test with fake core services — no RoboDK, real Open3D fusion.

Drives the real path:
  generate_scan_targets (depth gate -> seed=current pose -> reachable TasniScan_*)
  -> ScanCaptureJob (visit, grab synthetic depth, TSDF fuse, fit work plane)
  -> insert_scan (create frame + rectangle + mesh)

The fake camera renders depth of a flat 300x300 mm "table" at base z=0 from the
robot's current pose, so the fused surface + work frame are checked end to end.

    py -3.10 tests/test_scan_job.py
"""
from __future__ import annotations

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core import runs  # noqa: E402
from tasni.core.camera_lease import CameraLease  # noqa: E402
from tasni.core.config import AppConfig  # noqa: E402
from tasni.core.geometry import Rt_to_T  # noqa: E402
from tasni.core.jobrunner import JobContext  # noqa: E402
from tasni.modules.scan import service as scan_service  # noqa: E402

W, H = 320, 240
K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
TABLE_HALF_MM = 150.0
_ORIG_ROOT = runs.REPO_ROOT


class _Ctx(JobContext):
    def __init__(self): self.frames = 0
    def progress(self, *a, **k): pass
    def log(self, *a, **k): pass
    def frame(self, *a, **k): self.frames += 1
    def check_cancel(self): pass


def _look_at(cam_pos, target):
    cam_pos = np.asarray(cam_pos, float)
    z = np.asarray(target, float) - cam_pos
    z /= np.linalg.norm(z)
    a = np.array([1.0, 0, 0]) if abs(z[2]) > 0.9 else np.array([0, 0, 1.0])
    x = np.cross(a, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return Rt_to_T(np.column_stack([x, y, z]), cam_pos)


def _render(T_base_cam):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    dirs_cam = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, float)], -1)
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    dirs_base = dirs_cam @ R.T
    dz = dirs_base[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (0.0 - t[2]) / dz
    P = t + s[..., None] * dirs_base
    valid = ((np.abs(P[..., 0]) <= TABLE_HALF_MM) & (np.abs(P[..., 1]) <= TABLE_HALF_MM)
             & (s > 0) & np.isfinite(s))
    depth = np.where(valid, s, 0).astype(np.uint16)
    color = np.full((H, W, 3), 128, np.uint8)
    return SimpleNamespace(color=color, depth=depth, timestamp=0.0)


def _build_fakes(mount_mm=(40.0, -15.0, 55.0)):
    seed_T = _look_at((0, 0, 500), (0, 0, 0))           # straight down, 500 mm
    state = {"cam": seed_T, "targets": {}, "joints": {}}
    mount = Rt_to_T(np.eye(3), np.asarray(mount_mm, float))

    class FakeRdk:
        def __init__(self): self.inserted = {}
        def item_exists(self, name): return True
        def apply_run_mode(self, mode=None): return "run_robot"
        def connect_robot(self, ip="", *, timeout_s=10.0, poll_s=0.4):
            return True, "ROBOTCOM_READY"
        def robot_connection_params(self): return {"ip": "10.0.0.5", "port": 7000}
        def use_camera_tool(self, tool): return mount
        def camera_pose_T(self): return state["cam"]
        def current_joints(self): return "HOME"
        def move_j_joints(self, j): state["cam"] = seed_T
        def is_reachable(self, T): return True
        def screen_collisions(self, poses, *, guard_skip=None):
            return [True] * len(poses), False, [None] * len(poses)
        def solve_joints_for_pose(self, T, seed=None):
            return ("joints", float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
        def ensure_mounted_tool_collision_pairs(self, skip_trailing=2):
            return {"tools": ["Realsense"], "links": [0, 1, 2, 3, 4],
                    "pairs_enabled": 10, "pairs_failed": 0, "dof": 6}
        def list_targets(self, prefix=""):
            return sorted(n for n in state["targets"] if n.startswith(prefix))
        def add_target(self, name, T, joints=None):
            state["targets"][name] = T; state["joints"][name] = joints
        def delete_items(self, names):
            for n in list(names): state["targets"].pop(n, None)
        def move_j(self, name): state["cam"] = state["targets"][name]
        def add_frame(self, name, T, parent=None):
            self.inserted["frame"] = np.asarray(T, float)
            return SimpleNamespace(Valid=lambda: True)
        def add_rectangle(self, name, corners, parent=None, color=None):
            self.inserted["rect"] = np.asarray(corners, float)
            return SimpleNamespace(Valid=lambda: True)
        def add_mesh_file(self, name, path, parent=None, color=None):
            self.inserted["mesh"] = path
            return SimpleNamespace(Valid=lambda: True)

    class FakeBurst:
        """Mimics _BurstSession: CAP renders+buffers the frame at the current pose,
        GET returns them all in order, CLEAR records that the Jetson buffer dropped."""
        def __init__(self): self._buf = []; self.cleared = False
        def capture(self):
            self._buf.append(_render(state["cam"]))
            return b"thumb"                       # non-empty -> shows in the strip
        def fetch_all(self): return list(self._buf)
        def clear(self): self.cleared = True

    class FakeCamera:
        def __init__(self): self.last_burst = None
        def grab(self, with_depth=False, timeout=None, color_only=False):
            return _render(state["cam"])
        @contextmanager
        def burst(self, timeout=None):
            b = FakeBurst()
            self.last_burst = b
            yield b

    cfg = AppConfig()
    cfg.camera.intrinsics = {"320x240": K.tolist()}
    cfg.camera.resolution = "320x240"
    cfg.scan.pose_count = 8
    cfg.scan.cone_half_angle_deg = 30.0
    cfg.scan.voxel_size_m = 0.005
    services = SimpleNamespace(config=cfg, rdk=FakeRdk(), camera=FakeCamera(),
                               camera_lease=CameraLease(),
                               bus=SimpleNamespace(publish=lambda *a, **k: None),
                               live=SimpleNamespace(running=False, stop=lambda: None),
                               calib_dry_tour_required=False)
    return services, state


def test_generate_run_insert():
    try:
        import open3d  # noqa: F401
    except Exception:
        print("[skip] open3d not installed — `pip install -e .[scan]`")
        return
    services, state = _build_fakes()
    rdk = services.rdk

    gen = scan_service.generate_scan_targets(services)
    assert gen["created"] == 8, gen
    assert all(n.startswith("TasniScan_") for n in gen["targets"])
    assert gen["gate"]["ok"] is True
    assert gen["calibration_on_file"] is True
    # The full-frame survey now drives the actual scan geometry: this 300 mm square
    # fits best at ~487.5 mm with the configured FOV/margin, not the fixed 500 mm gate.
    assert abs(gen["look_distance_mm"] - 487.5) < 10
    assert gen["planned_cone_deg"] == services.config.scan.flat_cone_deg
    assert gen["planned_views"] == services.config.scan.flat_views

    with tempfile.TemporaryDirectory() as t:
        runs.REPO_ROOT = Path(t)

        def fake_new_run_dir(mid, stamp):
            d = Path(t) / "runs" / mid / stamp
            d.mkdir(parents=True, exist_ok=True)
            return d
        orig = scan_service.new_run_dir
        scan_service.new_run_dir = fake_new_run_dir
        try:
            job = scan_service.ScanCaptureJob(services, scan_service.ScanParams())
            res = job(_Ctx())
            assert res["kind"] == "scan" and res["can_insert"] is True
            assert res["n_views"] == 8
            assert res["mesh_vertices"] > 0
            sz = res["plane"]["size_mm"]
            assert 240 < sz[0] < 360 and 240 < sz[1] < 360, sz   # ~300 x 300 mm table
            assert res["plane"]["inlier_frac"] > 0.8
            # frame Z (col 2) points up out of the table
            fT = np.asarray(res["plane"]["frame_T_mm"], float)
            assert float(fT[:3, 2] @ [0, 0, 1]) > 0.99, fT[:3, 2]
            # targets persist (user-created)
            assert len(rdk.list_targets("TasniScan_")) == 8

            # Insert from the in-memory result -> frame + rectangle + mesh created.
            out = scan_service.insert_scan(services, job=job)
            assert out["status"] == "inserted"
            assert "frame" in rdk.inserted and "rect" in rdk.inserted and "mesh" in rdk.inserted
            assert rdk.inserted["rect"].shape == (4, 3)
            active = runs.read_active("scan")
            assert active["frame"] == scan_service.FRAME_NAME

            # Insert by run_id (from disk) also works.
            rdk.inserted.clear()
            stamp = res["stamp"]
            out2 = scan_service.insert_scan(services, run_id=stamp)
            assert out2["source"] == "run_id" and "frame" in rdk.inserted
        finally:
            scan_service.new_run_dir = orig
            runs.REPO_ROOT = _ORIG_ROOT
    print("[scan] gen 8 ->", res["n_views"], "views fused;",
          res["mesh_vertices"], "verts; surface",
          tuple(round(s) for s in res["plane"]["size_mm"]), "mm; inserted")


def test_lock_then_create_targets_reuses_frozen_surface():
    services, state = _build_fakes()
    locked = scan_service.lock_scan_surface(services)
    assert locked.gate_payload["ok"] is True
    assert locked.survey.detected is True
    gen = scan_service.generate_scan_targets(services, locked)
    assert gen["created"] == 8

    # A moved robot invalidates the frozen measurement instead of generating a
    # trajectory around stale geometry.
    locked2 = scan_service.lock_scan_surface(services)
    state["cam"] = locked2.seed_T.copy()
    state["cam"][0, 3] += 10.0
    try:
        scan_service.generate_scan_targets(services, locked2)
        raise AssertionError("expected moved robot to invalidate the lock")
    except RuntimeError as e:
        assert "moved after surface lock" in str(e), e
    print("[surface lock] frozen RGBD reused; 10 mm post-lock motion refused")


def test_targets_report_surface_coverage_from_footprint():
    """A fully-framed surface now drives COVERAGE-aware view selection (mirroring
    calibration), so target creation reports the predicted surface coverage and the
    survey carries the rectangle corners the footprint grid is built from."""
    services, _state = _build_fakes()
    locked = scan_service.lock_scan_surface(services)
    # The survey exposes the oriented-rectangle corners (camera frame) the scan
    # transforms to base + densifies into the coverage footprint.
    corners = np.asarray(locked.survey.corners_cam_mm, float)
    assert corners.shape == (4, 3), corners.shape

    gen = scan_service.generate_scan_targets(services, locked)
    assert gen["created"] == 8, gen
    # Small table, fully framed in every view -> the kept views tile (essentially)
    # the whole footprint, so coverage is reported and high (no missed region).
    assert gen["surface_coverage"] is not None, gen
    assert gen["surface_coverage"] >= 0.85, gen["surface_coverage"]
    print("[surface coverage] reported", f"{gen['surface_coverage']:.0%}",
          "from the measured rectangle footprint")


def test_burst_capture_path():
    """With scan.burst_capture on, the job captures via the burst session, fuses the
    same table, and CLEARs the Jetson buffer (no data left on the device)."""
    try:
        import open3d  # noqa: F401
    except Exception:
        print("[skip] open3d not installed — `pip install -e .[scan]`")
        return
    services, state = _build_fakes()
    services.config.scan.burst_capture = True

    gen = scan_service.generate_scan_targets(services)
    assert gen["created"] == 8, gen

    with tempfile.TemporaryDirectory() as t:
        runs.REPO_ROOT = Path(t)

        def fake_new_run_dir(mid, stamp):
            d = Path(t) / "runs" / mid / stamp
            d.mkdir(parents=True, exist_ok=True)
            return d
        orig = scan_service.new_run_dir
        scan_service.new_run_dir = fake_new_run_dir
        try:
            job = scan_service.ScanCaptureJob(services, scan_service.ScanParams())
            res = job(_Ctx())
            assert res["kind"] == "scan" and res["n_views"] == 8, res
            sz = res["plane"]["size_mm"]
            assert 240 < sz[0] < 360 and 240 < sz[1] < 360, sz   # ~300 x 300 mm table
            assert res["plane"]["inlier_frac"] > 0.8
            assert services.camera.last_burst is not None
            assert services.camera.last_burst.cleared, "burst buffer must be cleared on the Jetson"
        finally:
            scan_service.new_run_dir = orig
            runs.REPO_ROOT = _ORIG_ROOT
    print("[scan burst] gen 8 ->", res["n_views"], "views fused via burst; buffer cleared")


def test_save_views_persists_per_pose_frames():
    """scan.save_views writes each pose's color+depth+pose under <run>/views/ for a
    later camera-perspective coverage overlay (off by default, opt-in diagnostic)."""
    try:
        import open3d  # noqa: F401
    except Exception:
        print("[skip] open3d not installed — `pip install -e .[scan]`")
        return
    import json as _json

    services, _state = _build_fakes()
    services.config.scan.save_views = True
    scan_service.generate_scan_targets(services)

    with tempfile.TemporaryDirectory() as t:
        runs.REPO_ROOT = Path(t)

        def fake_new_run_dir(mid, stamp):
            d = Path(t) / "runs" / mid / stamp
            d.mkdir(parents=True, exist_ok=True)
            return d
        orig = scan_service.new_run_dir
        scan_service.new_run_dir = fake_new_run_dir
        try:
            job = scan_service.ScanCaptureJob(services, scan_service.ScanParams())
            res = job(_Ctx())
            vdir = Path(res["run_dir"]) / "views"
            assert vdir.is_dir(), "views/ dir not created"
            assert len(list(vdir.glob("view_*.jpg"))) == res["n_views"]
            assert len(list(vdir.glob("depth_*.png"))) == res["n_views"]
            meta = _json.loads((vdir / "views.json").read_text())
            assert len(meta["views"]) == res["n_views"]
            assert meta["size"] == [W, H] and len(meta["K"]) == 3
            assert len(meta["views"][0]["pose_T_mm"]) == 4   # 4x4 pose persisted
        finally:
            scan_service.new_run_dir = orig
            runs.REPO_ROOT = _ORIG_ROOT
    print("[save_views] persisted", res["n_views"], "color+depth frames + poses")


def test_generate_targets_when_survey_touches_border():
    """A full-frame survey can mark FRAMED red while the old centre gate is valid.

    Target creation should still use the current-pose cone, not fail just because the
    measured surface reaches the image border.
    """
    global TABLE_HALF_MM
    saved = TABLE_HALF_MM
    TABLE_HALF_MM = 1000.0
    try:
        services, _state = _build_fakes()
        gen = scan_service.generate_scan_targets(services)
        assert gen["created"] == 8, gen
        assert gen["gate"]["ok"] is True, gen["gate"]
        assert gen["gate"]["gates"].get("framed") is False, gen["gate"]
        assert abs(gen["look_distance_mm"] - 500) < 10, gen["look_distance_mm"]
        assert gen["crop_size_mm"] is not None
        assert gen["crop_size_mm"][0] > gen["crop_size_mm"][1] > 0
    finally:
        TABLE_HALF_MM = saved
    print("[survey border] framed red but centre gate OK -> created", gen["created"])


def test_scan_collision_filter_bypasses_noisy_wall_map_by_default():
    """Scan should match calibration's soft default for noisy collision maps.

    If RoboDK reports every reachable candidate as colliding (for example an oversized
    wall collision mesh), target creation still leaves reachable targets for operator
    inspection unless scan.collision_filter_hard_fail is enabled.
    """
    services, _state = _build_fakes()

    def all_collide(poses, *, guard_skip=None):
        return [False] * len(poses), True, [None] * len(poses)

    services.rdk.screen_collisions = all_collide
    gen = scan_service.generate_scan_targets(services)
    assert gen["created"] == 8, gen
    assert gen["collision_filter_bypassed"] is True, gen
    assert gen["candidates_collided"] == gen["candidates_reachable"], gen
    assert len(services.rdk.list_targets("TasniScan_")) == 8
    print("[scan collision bypass] wall/noisy map reported all colliding -> created", gen["created"])


def test_scan_collision_filter_hard_fail_can_still_refuse():
    services, _state = _build_fakes()
    services.config.scan.collision_filter_hard_fail = True

    def all_collide(poses, *, guard_skip=None):
        return [False] * len(poses), True, [None] * len(poses)

    services.rdk.screen_collisions = all_collide
    try:
        scan_service.generate_scan_targets(services)
        raise AssertionError("expected hard-fail scan collision filter to refuse")
    except RuntimeError as e:
        assert "collision-free poses" in str(e), e
    assert services.rdk.list_targets("TasniScan_") == []
    print("[scan collision hard fail] strict mode refused noisy collision map")


def test_generate_refuses_when_too_far():
    services, state = _build_fakes()
    state["cam"] = _look_at((0, 0, 900), (0, 0, 0))      # 900 mm > 500 +/- 150
    try:
        scan_service.generate_scan_targets(services)
        raise AssertionError("expected refusal — surface out of the standoff band")
    except RuntimeError as e:
        assert "distance" in str(e)
    assert services.rdk.list_targets("TasniScan_") == []
    print("[gate refusal] too-far standoff refused, nothing created")


def test_generate_accepts_dynamic_near_quality_distance():
    services, state = _build_fakes()
    state["cam"] = _look_at((0, 0, 310), (0, 0, 0))
    gen = scan_service.generate_scan_targets(services)
    assert gen["created"] == 8, gen
    assert 300 <= gen["look_distance_mm"] <= 800, gen["look_distance_mm"]
    print("[dynamic distance] near quality-band surface accepted at",
          round(gen["look_distance_mm"]), "mm")


def test_warns_but_proceeds_without_calibration():
    """Decoupling: a near-identity tool offset (no calibration) must NOT block —
    it warns and still creates targets (calibration_on_file=False)."""
    services, state = _build_fakes(mount_mm=(0.0, 0.0, 2.0))   # ~no offset
    gen = scan_service.generate_scan_targets(services)
    assert gen["created"] == 8
    assert gen["calibration_on_file"] is False
    print("[decoupled] no calibration on file -> warned, still created", gen["created"])


def test_run_without_targets_errors():
    services, _state = _build_fakes()
    try:
        scan_service.ScanCaptureJob(services, scan_service.ScanParams())(_Ctx())
        raise AssertionError("expected run to require targets")
    except RuntimeError as e:
        assert "targets" in str(e)
    print("[run needs targets] refused")


if __name__ == "__main__":
    test_generate_run_insert()
    test_lock_then_create_targets_reuses_frozen_surface()
    test_targets_report_surface_coverage_from_footprint()
    test_save_views_persists_per_pose_frames()
    test_burst_capture_path()
    test_generate_targets_when_survey_touches_border()
    test_scan_collision_filter_bypasses_noisy_wall_map_by_default()
    test_scan_collision_filter_hard_fail_can_still_refuse()
    test_generate_refuses_when_too_far()
    test_generate_accepts_dynamic_near_quality_distance()
    test_warns_but_proceeds_without_calibration()
    test_run_without_targets_errors()
    print("\nScan job (gate -> generate -> run -> insert) tests passed.")
