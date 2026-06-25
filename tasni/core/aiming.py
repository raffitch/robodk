"""The live aiming gate — turn one ChArUco detection into the HUD readiness state.

Before any calibration targets are created the operator jogs the robot until the
board sits at the ideal working distance and roughly fronto-parallel. This module
turns a single :class:`~tasni.core.charuco_types.ViewDetection` into the numbers
the HUD draws and the three lamps it lights:

    detected   board found with enough corners
    distance   board distance within ``ideal_distance_mm ± distance_tol_mm``
    angle      board tilt off fronto-parallel within ``max_tilt_deg``

Pure numpy (no RoboDK, no live camera, no ``modules.*``) so it is a core service
any workflow that aims a camera at a ChArUco board can reuse, and unit-testable on
any machine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .charuco_types import ViewDetection


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
    # Seed-view quality. This is intentionally a projected board-area fraction,
    # not corner count: enough board must fill the image for a precise seed PnP,
    # while leaving room for the generated tilted/offset calibration views.
    min_board_area_frac: float = 0.10
    max_board_area_frac: float = 0.40


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
    # How to correct the tilt, as TOOL-frame rotations (KUKA A/B/C convention:
    # A = rotate about Z, B = about Y, C = about X). A board tilt is fixed by B + C
    # (a rotation about Z / A doesn't change tilt). Signed degrees the operator
    # should rotate the TOOL to make the board fronto-parallel; None if no board.
    # Same decomposition as the scan gate (modules/scan/depth_gate.py) so the HUD's
    # ROTATE-TOOL panel reads identically for board aim and surface aim.
    tilt_b_deg: float | None = None      # rotate about camera/TOOL Y (KUKA B): left/right
    tilt_c_deg: float | None = None      # rotate about camera/TOOL X (KUKA C): fwd/back
    board_area_frac: float | None = None

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
            "tilt_b_deg": self.tilt_b_deg,
            "tilt_c_deg": self.tilt_c_deg,
            "board_area_frac": self.board_area_frac,
            "min_board_area_frac": self.min_board_area_frac,
            "max_board_area_frac": self.max_board_area_frac,
        }

    min_board_area_frac: float = 0.10
    max_board_area_frac: float = 0.40


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


def board_tilt_bc_deg(R_target2cam: np.ndarray) -> tuple[float, float]:
    """Signed TOOL rotations (KUKA B about Y, C about X) that bring the board
    fronto-parallel — i.e. which way to rotate the tool to level the board.

    The board normal is its +Z axis in the camera frame (column 2 of
    ``R_target2cam``). We orient it to face the camera (negative optical Z, like the
    scan gate) then decompose into a rotation about the camera/TOOL Y axis (KUKA B,
    left/right) and X axis (KUKA C, fwd/back). A rotation about Z (KUKA A) doesn't
    change tilt, so only B and C are returned. Signs match the scan gate's so the
    HUD's ◀▶ / ▲▼ arrows mean the same thing for both."""
    n = np.asarray(R_target2cam, dtype=float)[:, 2]
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
        return (0.0, 0.0)
    n = n / norm
    if n[2] > 0:                     # face the camera (toward -Z optical axis)
        n = -n
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
    denom = max(-nz, 1e-9)
    tilt_b_deg = float(np.degrees(np.arctan2(nx, denom)))   # about Y -> KUKA B
    tilt_c_deg = float(np.degrees(np.arctan2(ny, denom)))   # about X -> KUKA C
    return (tilt_b_deg, tilt_c_deg)


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
                  th: GateThresholds, board_center_mm: np.ndarray | None = None,
                  board_obj_points: np.ndarray | None = None) -> GateReading:
    """Build the :class:`GateReading` for one frame (``det`` is ``None`` if the
    board was not found).

    ``board_center_mm`` is the board centre in the board frame; when given,
    distance / centring / jog all reference the board CENTRE instead of the corner
    origin (so aiming targets the middle of the board)."""
    if det is None:
        gates = {"detected": False, "distance": False, "angle": False,
                 "center": False, "coverage": False}
        return GateReading(False, 0, None, None, None, gates, False,
                           th.ideal_distance_mm, th.distance_tol_mm, th.max_tilt_deg,
                           None, th.center_tol_mm,
                           min_board_area_frac=th.min_board_area_frac,
                           max_board_area_frac=th.max_board_area_frac)

    R = np.asarray(det.R_target2cam, dtype=float)
    t = np.asarray(det.t_target2cam, dtype=float).reshape(3)
    # Reference the board centre, not the corner origin, in the camera frame.
    center = (R @ np.asarray(board_center_mm, dtype=float).reshape(3) + t
              if board_center_mm is not None else t)
    distance = float(np.linalg.norm(center))
    tilt = board_tilt_deg(R)
    tilt_b, tilt_c = board_tilt_bc_deg(R)
    offset = board_offset(center, K, image_shape)
    board_area_frac = None
    if board_obj_points is not None:
        obj = np.asarray(board_obj_points, dtype=float).reshape(-1, 3)
        cam = obj @ R.T + t
        valid = cam[:, 2] > 1e-6
        if int(valid.sum()) >= 3:
            uv = np.column_stack([
                K[0, 0] * cam[valid, 0] / cam[valid, 2] + K[0, 2],
                K[1, 1] * cam[valid, 1] / cam[valid, 2] + K[1, 2],
            ])
            h, w = image_shape[0], image_shape[1]
            span = np.ptp(uv, axis=0)
            board_area_frac = float(
                max(0.0, span[0]) * max(0.0, span[1]) / max(float(w * h), 1.0))

    # Translation to bring the board centre to (0, 0, ideal) in the camera frame,
    # with optional per-axis sign flips to match the pendant's TOOL convention.
    sx, sy, sz = (-1 if th.invert_x else 1), (-1 if th.invert_y else 1), (-1 if th.invert_z else 1)
    move_cam = (sx * float(center[0]), sy * float(center[1]),
                sz * float(center[2] - th.ideal_distance_mm))

    gates = {
        "detected": det.n_corners >= th.min_corners,
        "distance": abs(distance - th.ideal_distance_mm) <= th.distance_tol_mm,
        "angle": tilt <= th.max_tilt_deg,
        "center": abs(float(center[0])) <= th.center_tol_mm
                  and abs(float(center[1])) <= th.center_tol_mm,
        "coverage": (board_area_frac is None
                     or th.min_board_area_frac <= board_area_frac
                     <= th.max_board_area_frac),
    }
    return GateReading(True, det.n_corners, distance, tilt, offset, gates,
                       all(gates.values()), th.ideal_distance_mm,
                       th.distance_tol_mm, th.max_tilt_deg, move_cam, th.center_tol_mm,
                       tilt_b_deg=tilt_b, tilt_c_deg=tilt_c,
                       board_area_frac=board_area_frac,
                       min_board_area_frac=th.min_board_area_frac,
                       max_board_area_frac=th.max_board_area_frac)
