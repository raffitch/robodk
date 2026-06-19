"""ChArUco board definition, detection and per-view board-pose estimation.

This is the same detection chain the original macro used (detectMarkers ->
interpolateCornersCharuco -> estimatePoseCharucoBoard), packaged so each view
also carries the detected corner pixels + their 3D board coordinates, which the
quality metrics need for reprojection error.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ...core.config import BoardConfig


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


class CharucoTarget:
    """A ChArUco calibration board built from a :class:`BoardConfig`."""

    def __init__(self, config: BoardConfig):
        self.config = config
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, config.dictionary))
        self.board = cv2.aruco.CharucoBoard(
            (config.squares_x, config.squares_y),
            config.square_size_mm,
            config.marker_size_mm,
            self.dictionary,
        )
        # All inner chessboard corners in board frame (mm), indexed by charuco id.
        self._all_obj = np.asarray(self.board.getChessboardCorners(), dtype=np.float32)

    def detect(self, image: np.ndarray, K: np.ndarray, dist: np.ndarray,
               *, min_corners: int = 6) -> ViewDetection | None:
        """Detect the board in ``image`` and estimate its pose in the camera.

        Returns ``None`` if the board is absent or too few corners are found.
        """
        gray = (image if image.ndim == 2
                else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
        if marker_ids is None or len(marker_ids) < 1:
            return None
        retval, corners, ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, self.board)
        if retval is None or retval < min_corners or ids is None:
            return None

        rvec = np.zeros((3, 1))
        tvec = np.zeros((3, 1))
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            corners, ids, self.board, K, dist, rvec, tvec)
        if not ok:
            return None

        obj_points = self._all_obj[ids.flatten()]
        return ViewDetection(corners=corners, ids=ids, obj_points=obj_points,
                             rvec=rvec, tvec=tvec)

    def annotate(self, image_bgr: np.ndarray, det: ViewDetection,
                 K: np.ndarray, dist: np.ndarray, label: str = "") -> np.ndarray:
        """Draw detected corners + board axes for the live preview."""
        out = image_bgr.copy()
        cv2.aruco.drawDetectedCornersCharuco(out, det.corners, det.ids)
        axis_len = max(1.5 * self.config.square_size_mm, 5.0)
        cv2.drawFrameAxes(out, K, dist, det.rvec, det.tvec, axis_len)
        if label:
            cv2.putText(out, label, (40, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 2, cv2.LINE_AA)
        return out
