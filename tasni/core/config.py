"""Layered configuration for the platform.

Defaults live in the dataclasses below (so the app runs with zero config).
A user JSON file (``tasni.config.json`` in the repo root, or a path passed to
:func:`load_config`) overrides any subset of fields. Secrets (Jetson password
etc.) are NOT stored here — they stay in ``secrets/jetson.env`` (git-ignored).

Python 3.10 has no ``tomllib``, so we use JSON to avoid an extra dependency.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

# D435i color intrinsics per stream resolution, copied from the original
# AutoCalibrate macro (factory values read off this specific camera).
_DEFAULT_INTRINSICS: dict[str, list[list[float]]] = {
    "640x480": [[605.400024414062, 0, 326.824310302734],
                [0, 605.427429199219, 244.43229675293],
                [0, 0, 1]],
    "1280x720": [[908.100036621094, 0, 650.236450195312],
                 [0, 908.14111328125, 366.6484375],
                 [0, 0, 1]],
    "1920x1080": [[1362.15002441406, 0, 975.354675292969],
                  [0, 1362.21166992188, 549.97265625],
                  [0, 0, 1]],
}


@dataclass
class CameraConfig:
    """RealSense-over-TCP client settings (the Jetson camera server)."""

    # The Jetson camera host. The server binds 0.0.0.0:1024 (all interfaces), so
    # this is just whichever Jetson IP the workstation can reach — confirmed
    # 10.12.171.70 (the old 10.5.5.19 subnet is gone). Override per-machine in
    # tasni.config.json if the IP changes.
    ip: str = "10.12.171.70"
    port: int = 1024
    # The server streams color at 1280x720 (server_unicast_syncronous.py), so the
    # intrinsics must be the 720p K — a 1080p setting would skew distance/tilt.
    resolution: str = "1280x720"
    timeout_s: float = 10.0
    # resolution -> 3x3 color intrinsics K
    intrinsics: dict[str, list[list[float]]] = field(
        default_factory=lambda: {k: [row[:] for row in v]
                                 for k, v in _DEFAULT_INTRINSICS.items()})
    dist_coeffs: list[float] = field(default_factory=lambda: [0, 0, 0, 0, 0])

    @property
    def K(self) -> np.ndarray:
        """Camera matrix for the configured resolution."""
        return np.array(self.intrinsics[self.resolution], dtype=np.float64)

    @property
    def dist(self) -> np.ndarray:
        return np.array(self.dist_coeffs, dtype=np.float32).reshape(-1, 1)

    @property
    def size(self) -> tuple[int, int]:
        w, h = self.resolution.split("x")
        return int(w), int(h)


@dataclass
class BoardConfig:
    """ChArUco board geometry (eye-in-hand calibration target).

    This is the single source of truth: the printable PDF renders THESE exact
    dimensions at true physical size, so "what we print" always equals "what
    detection expects" — no matching step. The default fits A4 (landscape) 1:1.
    """

    dictionary: str = "DICT_6X6_250"
    squares_x: int = 8
    squares_y: int = 6
    square_size_mm: float = 30.0        # 8x30 = 240 mm wide -> fits A4 landscape
    marker_size_mm: float = 22.0


@dataclass
class RoboDKConfig:
    """How tasni talks to RoboDK and which cell items it drives."""

    robot_name: str = "KUKA KR150 R2700"
    # "attach": use the running RoboDK GUI instance (default). If it has no
    # station with this robot loaded, the app opens `station_path` into it (so
    # you don't end up driving an empty RoboDK). "isolated": private headless
    # instance that loads station_path — used by tests / when no GUI is open.
    connection: str = "attach"
    # The cell's RoboDK station; relative paths resolve against the repo root.
    station_path: str | None = "Tasni.rdk"
    station_name: str = "Tasni"          # station display name set after loading
    target_prefix: str = "Target"
    # The RealSense camera is mounted on the flange as a tool named "Realsense"
    # (with its 3D model) in Tasni.rdk. Calibration solves THIS tool's pose; it is
    # fixed, not user-selectable.
    camera_tool: str = "Realsense"
    # No taught home pose: the operator jogs the robot until the live aiming gate
    # is green, and the robot's *current* pose becomes the seed the calibration
    # poses orbit around (see CalibrationConfig gate knobs below).
    # "simulate" keeps the robot in RoboDK only; "run_robot" drives the real arm.
    # Calibration only makes sense on the real arm (the camera rides on it).
    run_mode: str = "run_robot"


@dataclass
class CalibrationConfig:
    """Calibration-module knobs (live gate + pose generation + capture + solve)."""

    settle_s: float = 0.4               # pause after MoveJ before grabbing a frame
    holdout_count: int = 3              # poses held out of the solve for validation
    refine: bool = True                 # post-TSAI reprojection-minimizing refinement
    min_charuco_corners: int = 6        # reject a view with fewer detected corners

    # Live aiming gate: before any targets are created, the operator jogs the
    # robot until the board sits at the ideal distance and angle. These bands
    # define when each HUD lamp goes green; all must be green to create targets.
    ideal_distance_mm: float = 450.0    # target working distance (board <-> camera)
    distance_tol_mm: float = 80.0       # +/- band around ideal_distance_mm
    max_tilt_deg: float = 25.0          # board may be off fronto-parallel by this much
    center_tol_mm: float = 40.0         # |x|,|y| under this counts as centred (advisory)
    preview_fps: float = 6.0            # max live-gate publish rate
    preview_timeout_s: float = 4.0      # per-frame camera timeout while streaming
    # HUD X/Y/Z jog hints are in the camera optical frame (X right, Y down, Z
    # forward). Flip an axis here if the pendant's TOOL axis runs the other way.
    jog_invert_x: bool = False
    jog_invert_y: bool = False
    jog_invert_z: bool = False

    # Auto pose generation: orbit the (gated) seed view in a cone (not a full dome)
    # so the board stays visible, with roll + distance variation for hand-eye
    # conditioning. The TasniCalib_* targets are left in the station to inspect.
    pose_count: int = 15                # reachable poses to capture
    cone_half_angle_deg: float = 32.0   # max view-angle change from the seed view
    roll_max_deg: float = 75.0          # roll spread about the optical axis
    distance_jitter: float = 0.12       # +/- fraction of working distance
    look_distance_mm: float = 500.0     # fallback if the seed board distance unknown


@dataclass
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    board: BoardConfig = field(default_factory=BoardConfig)
    robodk: RoboDKConfig = field(default_factory=RoboDKConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    web: WebConfig = field(default_factory=WebConfig)


def _merge(obj: Any, data: dict[str, Any]) -> None:
    """Recursively overlay ``data`` onto a dataclass instance in place."""
    valid = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in valid:
            raise KeyError(f"Unknown config key: {key!r}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(obj, key, value)


def config_file_path() -> Path:
    """Path to the user override file (``tasni.config.json`` at the repo root)."""
    return Path(__file__).resolve().parents[2] / "tasni.config.json"


def save_overrides(updates: dict[str, Any]) -> Path:
    """Deep-merge ``updates`` into ``tasni.config.json`` (created if absent).

    Used to persist UI-driven changes — e.g. syncing the printed board's
    dimensions into the config so detection matches what was printed.
    """
    path = config_file_path()
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def merge(dst: dict, src: dict) -> None:
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                merge(dst[key], value)
            else:
                dst[key] = value

    merge(data, updates)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_config(path: str | Path | None = None) -> AppConfig:
    """Build an :class:`AppConfig`, overlaying an optional JSON file.

    With no ``path``, looks for ``tasni.config.json`` next to the repo root and
    uses it if present; otherwise returns pure defaults.
    """
    cfg = AppConfig()
    if path is None:
        candidate = Path(__file__).resolve().parents[2] / "tasni.config.json"
        path = candidate if candidate.exists() else None
    if path is not None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        _merge(cfg, data)
    return cfg
