"""ChArUco detection value types shared across the core.

``ViewDetection`` lives here (not in the calibration module) so core services —
the aiming gate (:mod:`tasni.core.aiming`) today, scan/aruco later — can speak the
same per-view detection shape without importing ``modules.*`` (the boundary that
keeps the core module-agnostic). The calibration module re-exports it from
``charuco.py`` for backward compatibility, so existing imports keep working.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ViewDetection:
    """One image's ChArUco detection and the board pose recovered from it."""

    corners: np.ndarray        # (N,1,2) float32 charuco corner pixels
    ids: np.ndarray            # (N,1) int charuco corner ids
    obj_points: np.ndarray     # (N,3) float32 board-frame coords (mm)
    rvec: np.ndarray           # (3,1) board->cam rotation (Rodrigues)
    tvec: np.ndarray           # (3,1) board->cam translation (mm)

    @property
    def n_corners(self) -> int:
        return int(self.ids.shape[0])

    @property
    def R_target2cam(self) -> np.ndarray:
        R, _ = cv2.Rodrigues(self.rvec)
        return R

    @property
    def t_target2cam(self) -> np.ndarray:
        return self.tvec.reshape(3)
