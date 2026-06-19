"""The live aiming gate — turn one ChArUco detection into the HUD readiness state.

Before any calibration targets are created the operator jogs the robot until the
board sits at the ideal working distance and roughly fronto-parallel. This module
turns a single :class:`~tasni.modules.calibration.charuco.ViewDetection` into the
numbers the HUD draws and the three lamps it lights:

    detected   board found with enough corners
    distance   board distance within ``ideal_distance_mm ± distance_tol_mm``
    angle      board tilt off fronto-parallel within ``max_tilt_deg``

Pure numpy (no RoboDK, no live camera) so it is unit-testable on any machine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .charuco import ViewDetection


@dataclass
class GateThresholds:
    """The bands that decide when each HUD lamp goes green."""

    min_corners: int = 6
    ideal_distance_mm: float = 450.0
    distance_tol_mm: float = 80.0
    max_tilt_deg: float = 25.0
    # The HUD's X/Y/Z jog hints are in the camera optical frame (X right, Y down,
    # Z forward). If a pendant TOOL axis is mirrored vs this convention, flip it
    # here so "X+" really points the way the operator must jog.
    invert_x: bool = False
    invert_y: bool = False
    invert_z: bool = False
    center_tol_mm: float = 40.0     # |x|,|y| under this counts as centred (advisory)


@dataclass
class GateReading:
    """One frame's aiming state, ready to ship to the HUD as JSON."""

    detected: bool
    n_corners: int
    distance_mm: float | None      # board <-> camera, None if no board
    tilt_deg: float | None         # 0 = fronto-parallel, 90 = edge-on
    offset: tuple[float, float] | None  # board centre vs frame centre, each in [-1, 1]
    gates: dict                    # {"detected": bool, "distance": bool, "angle": bool}
    ok: bool                       # all gates green -> targets may be created
    ideal_distance_mm: float
    distance_tol_mm: float
    max_tilt_deg: float
    # Camera-frame translation (mm) the camera must move to put the board centred
    # at the ideal distance: [+X right, +Y down, +Z forward]. Drives the X/Y/Z
    # jog hints. None if no board.
    move_cam: tuple[float, float, float] | None = None
    center_tol_mm: float = 40.0

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "n_corners": self.n_corners,
            "distance_mm": self.distance_mm,
            "tilt_deg": self.tilt_deg,
            "offset": list(self.offset) if self.offset is not None else None,
            "gates": self.gates,
            "ok": self.ok,
            "ideal_distance_mm": self.ideal_distance_mm,
            "distance_tol_mm": self.distance_tol_mm,
            "max_tilt_deg": self.max_tilt_deg,
            "move_cam": list(self.move_cam) if self.move_cam is not None else None,
            "center_tol_mm": self.center_tol_mm,
        }


def board_tilt_deg(R_target2cam: np.ndarray) -> float:
    """Angle (deg) of the board off fronto-parallel.

    The board normal is its +Z axis expressed in the camera frame (column 2 of
    ``R_target2cam``). Fronto-parallel means that normal is parallel to the camera
    optical axis (+Z_cam), regardless of which way it points — so we take the
    absolute alignment, giving 0° fronto-parallel ... 90° edge-on.
    """
    n = np.asarray(R_target2cam, dtype=float)[:, 2]
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
        return 0.0
    align = abs(float(n[2]) / norm)               # |n · [0,0,1]|
    return float(np.degrees(np.arccos(np.clip(align, 0.0, 1.0))))


def board_offset(t_target2cam: np.ndarray, K: np.ndarray,
                 image_shape: tuple) -> tuple[float, float]:
    """Board-centre pixel projected and expressed as a fraction of the frame,
    where (0, 0) is the image centre and ±1 is an edge. Drives the HUD lock-box."""
    t = np.asarray(t_target2cam, dtype=float).reshape(3)
    if abs(t[2]) < 1e-6:
        return (0.0, 0.0)
    K = np.asarray(K, dtype=float)
    u = K[0, 0] * (t[0] / t[2]) + K[0, 2]
    v = K[1, 1] * (t[1] / t[2]) + K[1, 2]
    h, w = image_shape[0], image_shape[1]
    return ((u - w / 2.0) / (w / 2.0), (v - h / 2.0) / (h / 2.0))


def evaluate_gate(det: ViewDetection | None, K: np.ndarray, image_shape: tuple,
                  th: GateThresholds) -> GateReading:
    """Build the :class:`GateReading` for one frame (``det`` is ``None`` if the
    board was not found)."""
    if det is None:
        gates = {"detected": False, "distance": False, "angle": False}
        return GateReading(False, 0, None, None, None, gates, False,
                           th.ideal_distance_mm, th.distance_tol_mm, th.max_tilt_deg,
                           None, th.center_tol_mm)

    t = np.asarray(det.t_target2cam, dtype=float).reshape(3)
    distance = float(np.linalg.norm(t))
    tilt = board_tilt_deg(det.R_target2cam)
    offset = board_offset(t, K, image_shape)

    # Translation to bring the board to (0, 0, ideal) in the camera frame, with
    # optional per-axis sign flips to match the pendant's TOOL convention.
    sx, sy, sz = (-1 if th.invert_x else 1), (-1 if th.invert_y else 1), (-1 if th.invert_z else 1)
    move_cam = (sx * float(t[0]), sy * float(t[1]), sz * float(t[2] - th.ideal_distance_mm))

    gates = {
        "detected": det.n_corners >= th.min_corners,
        "distance": abs(distance - th.ideal_distance_mm) <= th.distance_tol_mm,
        "angle": tilt <= th.max_tilt_deg,
    }
    return GateReading(True, det.n_corners, distance, tilt, offset, gates,
                       all(gates.values()), th.ideal_distance_mm,
                       th.distance_tol_mm, th.max_tilt_deg, move_cam, th.center_tol_mm)
