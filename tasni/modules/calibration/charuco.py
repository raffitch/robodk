"""ChArUco board definition, detection and per-view board-pose estimation.

This is the same detection chain the original macro used (detectMarkers ->
interpolateCornersCharuco -> estimatePoseCharucoBoard), packaged so each view
also carries the detected corner pixels + their 3D board coordinates, which the
quality metrics need for reprojection error.
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from ...core.charuco_types import ViewDetection
from ...core.config import BoardConfig

# ``ViewDetection`` now lives in core (so core services can use it without
# importing modules.*); re-exported here so existing `from .charuco import
# ViewDetection` imports keep working.
__all__ = ["ViewDetection", "CharucoTarget"]


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
        # Geometric centre of the board (board frame, mm). The board origin is a
        # corner, so distance/aiming should reference this instead. Using the mean
        # of all corners is robust to the origin convention.
        self.board_center = self._all_obj.mean(axis=0).astype(np.float64)

    @property
    def all_obj_points(self) -> np.ndarray:
        """All inner ChArUco corners in the board frame (mm) — the detectable
        feature cloud. Used by the visibility pre-filter (project these into a
        candidate camera to predict whether the board stays in frame)."""
        return self._all_obj

    def _detect_corners(self, image: np.ndarray, *, min_corners: int = 1):
        """Detect markers + interpolate ChArUco corners (no pose estimation).

        Returns ``(corners (N,1,2), ids (N,1))`` or ``None``. Split out so
        :meth:`detect_median` can pool corners across frames before posing once.
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
        return corners, ids

    def _pose(self, corners: np.ndarray, ids: np.ndarray, K: np.ndarray,
              dist: np.ndarray) -> ViewDetection | None:
        """Estimate the board pose from charuco corners and pack a ViewDetection."""
        rvec = np.zeros((3, 1))
        tvec = np.zeros((3, 1))
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            corners, ids, self.board, K, dist, rvec, tvec)
        if not ok:
            return None
        return ViewDetection(corners=corners, ids=ids,
                             obj_points=self._all_obj[ids.flatten()],
                             rvec=rvec, tvec=tvec)

    def detect_points(self, image: np.ndarray, *, min_corners: int = 6
                      ) -> "tuple[np.ndarray, np.ndarray, np.ndarray] | None":
        """Detect ChArUco corners and their board-frame coords, with **no** pose
        estimation (so it needs no K/dist). Returns ``(corners (N,1,2), ids (N,1),
        obj_points (N,3))`` or ``None``. Used by intrinsic calibration, which solves
        K/distortion from the raw 2D-3D correspondences."""
        found = self._detect_corners(image, min_corners=min_corners)
        if found is None:
            return None
        corners, ids = found
        return corners, ids, self._all_obj[ids.flatten()]

    def detect(self, image: np.ndarray, K: np.ndarray, dist: np.ndarray,
               *, min_corners: int = 6) -> ViewDetection | None:
        """Detect the board in ``image`` and estimate its pose in the camera.

        Returns ``None`` if the board is absent or too few corners are found.
        """
        found = self._detect_corners(image, min_corners=min_corners)
        if found is None:
            return None
        return self._pose(found[0], found[1], K, dist)

    def detect_median(self, images: list[np.ndarray], K: np.ndarray, dist: np.ndarray,
                      *, min_corners: int = 6, min_frac: float = 0.5
                      ) -> ViewDetection | None:
        """Detect across several frames of the *same* static view, median each
        corner's pixel location, then estimate the board pose once from the
        medianed corners. Averages out per-frame blur, glare and sensor noise —
        a cheap, standard robustness win over a single grab. One image reduces to
        ordinary :meth:`detect`.

        A corner is kept only if it was seen in at least ``min_frac`` of the
        frames (rejects flicker); ``min_corners`` survivors are required.
        """
        per = [self._detect_corners(img, min_corners=1) for img in images]
        per = [p for p in per if p is not None]
        if not per:
            return None
        acc: dict[int, list[np.ndarray]] = defaultdict(list)
        for corners, ids in per:
            for cid, px in zip(ids.flatten(), corners.reshape(-1, 2)):
                acc[int(cid)].append(px)
        keep_thresh = max(1, int(np.ceil(min_frac * len(per))))
        kept = sorted(cid for cid, pxs in acc.items() if len(pxs) >= keep_thresh)
        if len(kept) < min_corners:
            return None
        corners = np.array([np.median(np.stack(acc[c]), axis=0) for c in kept],
                           dtype=np.float32).reshape(-1, 1, 2)
        ids = np.array(kept, dtype=np.int32).reshape(-1, 1)
        return self._pose(corners, ids, K, dist)

    def annotate(self, image_bgr: np.ndarray, det: ViewDetection,
                 K: np.ndarray, dist: np.ndarray, label: str = "") -> np.ndarray:
        """Draw detected corners + board axes for the live preview."""
        out = image_bgr.copy()
        cv2.aruco.drawDetectedCornersCharuco(out, det.corners, det.ids)
        axis_len = max(1.5 * self.config.square_size_mm, 5.0)
        # Draw the axes at the board CENTRE (not the corner origin) so the visual
        # reference matches what the aiming gate measures.
        R, _ = cv2.Rodrigues(det.rvec)
        center_tvec = (R @ self.board_center + det.tvec.reshape(3)).reshape(3, 1)
        cv2.drawFrameAxes(out, K, dist, det.rvec, center_tvec, axis_len)
        if label:
            cv2.putText(out, label, (40, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 2, cv2.LINE_AA)
        return out
