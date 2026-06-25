"""Dedicated RGB intrinsic calibration — the missing camera-model step.

The hand-eye calibration consumes the camera intrinsics (K + lens distortion);
its accuracy is capped by how well those model the real lens. The D435i ships its
*color* stream with **zero** distortion coefficients (Intel calibrates depth/IR,
not the RGB lens), so without this step the hand-eye solve silently absorbs the
real distortion — exactly the "borderline / verify intrinsics" verdict we saw.

This collects many ChArUco views spread across the whole frame (auto-captured as
the board reaches new image regions) and solves K + distortion with
``cv2.calibrateCamera``. It is **camera-only** (no robot motion): the operator
waves the board around in front of the camera. ``k3`` is fixed to 0 by default —
the D4xx RGB lens is low-distortion, and a free ``k3`` overfits the limited image
region (the 4.3 we saw from the hand-eye captures). Applying writes the result
into the camera config (live, no restart, + persisted).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

# Coverage grid: the frame is split into GRID_X x GRID_Y cells so the operator can
# see (and the auto-capture can spread views across) the whole image — corners
# near the edges are what constrain the distortion terms.
GRID_X, GRID_Y = 4, 3
MIN_CORNERS = 15          # a view must see at least this many charuco corners
STABLE_PX = 2.0           # max median corner motion vs the previous frame to count as "still"
NOVEL_PX = 45.0           # a kept view must differ from every other by this much (median px)
MIN_VIEWS = 6             # cv2.calibrateCamera floor; below this the solve is meaningless
MAX_VIEWS = 40            # safety cap so a held board can't accumulate forever


@dataclass
class _View:
    corners: np.ndarray   # (N,1,2) f32 charuco corner pixels
    ids: np.ndarray       # (N,1) i32 charuco ids
    obj: np.ndarray       # (N,3) f32 board-frame coords (mm)
    cell: tuple[int, int]


class IntrinsicCalibSession:
    """Accumulates auto-captured ChArUco views and solves the intrinsics.

    Thread-safe: :meth:`offer` runs on the live-preview thread while
    :meth:`solve` / :meth:`reset` / :meth:`status` are called from request threads.
    """

    def __init__(self, image_size: tuple[int, int]):
        self.image_size = (int(image_size[0]), int(image_size[1]))   # (w, h)
        self._views: list[_View] = []
        self._cells = np.zeros((GRID_Y, GRID_X), dtype=int)
        self._prev: dict[int, np.ndarray] | None = None   # last frame's id->px (stability)
        self._lock = threading.Lock()
        self.last_report: dict | None = None

    # -- read-only views ----------------------------------------------------
    @property
    def count(self) -> int:
        return len(self._views)

    @property
    def coverage_pct(self) -> float:
        return float((self._cells > 0).sum()) / (GRID_X * GRID_Y)

    def _cells_grid(self) -> list[list[int]]:
        return [[int(self._cells[y, x]) for x in range(GRID_X)] for y in range(GRID_Y)]

    def status(self, *, detected: bool = False, n_corners: int = 0,
               stable: bool = False, captured: bool = False,
               cell: tuple[int, int] | None = None) -> dict:
        return {
            "mode": "intrinsics",
            "detected": bool(detected), "n_corners": int(n_corners),
            "stable": bool(stable), "captured": bool(captured),
            "cell": list(cell) if cell is not None else None,
            "count": len(self._views), "coverage_pct": self.coverage_pct,
            "cells": self._cells_grid(), "grid": [GRID_X, GRID_Y],
            "max_views": MAX_VIEWS, "min_views": MIN_VIEWS,
            "have_solve": self.last_report is not None,
        }

    # -- mutation -----------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            self._views.clear()
            self._cells[:] = 0
            self._prev = None
            self.last_report = None

    def _cell_of(self, centroid: np.ndarray) -> tuple[int, int]:
        w, h = self.image_size
        cx = min(GRID_X - 1, max(0, int(centroid[0] / w * GRID_X)))
        cy = min(GRID_Y - 1, max(0, int(centroid[1] / h * GRID_Y)))
        return cx, cy

    def _is_stable(self, ids: np.ndarray, pts: np.ndarray) -> bool:
        if self._prev is None:
            return False
        ds = [float(np.linalg.norm(pts[i] - self._prev[int(c)]))
              for i, c in enumerate(ids) if int(c) in self._prev]
        return bool(ds) and float(np.median(ds)) < STABLE_PX

    def _is_novel(self, ids: np.ndarray, pts: np.ndarray) -> bool:
        """Novel iff it differs from EVERY kept view (median shared-corner px) — this
        rejects near-duplicates of a held pose and naturally rewards moving/tilting."""
        for v in self._views:
            vp = {int(c): p for c, p in zip(v.ids.flatten(), v.corners.reshape(-1, 2))}
            ds = [float(np.linalg.norm(pts[i] - vp[int(c)]))
                  for i, c in enumerate(ids) if int(c) in vp]
            if ds and float(np.median(ds)) < NOVEL_PX:
                return False
        return True

    def offer(self, corners: np.ndarray, ids: np.ndarray, obj: np.ndarray) -> dict:
        """Per-frame auto-capture decision. ``corners`` is (N,1,2), ``ids`` (N,1),
        ``obj`` (N,3). Captures when the board is still, sees enough corners and is
        novel vs everything kept. Returns the HUD status dict."""
        pts = corners.reshape(-1, 2)
        idf = ids.flatten()
        cell = self._cell_of(pts.mean(axis=0))
        n = len(idf)
        captured = False
        with self._lock:
            stable = self._is_stable(idf, pts)
            if (n >= MIN_CORNERS and stable and len(self._views) < MAX_VIEWS
                    and self._is_novel(idf, pts)):
                self._views.append(_View(corners.copy(), ids.copy(), obj.copy(), cell))
                self._cells[cell[1], cell[0]] += 1
                captured = True
            self._prev = {int(c): p for c, p in zip(idf, pts)}
            return self.status(detected=True, n_corners=n, stable=stable,
                               captured=captured, cell=cell)

    def offer_none(self) -> dict:
        """No board this frame — drop the stability anchor so the next detection
        can't be falsely judged 'still' against a stale frame."""
        with self._lock:
            self._prev = None
            return self.status(detected=False)

    # -- solve --------------------------------------------------------------
    def solve(self, K0: np.ndarray, *, fix_k3: bool = True) -> dict:
        with self._lock:
            views = list(self._views)
        obj = [v.obj.astype(np.float32) for v in views]
        img = [v.corners.reshape(-1, 1, 2).astype(np.float32) for v in views]
        report = solve_intrinsics(obj, img, self.image_size, K0, fix_k3=fix_k3)
        self.last_report = report
        return report


def coverage_from_corners(img_list: list[np.ndarray],
                          image_size: tuple[int, int]) -> tuple[float, list[list[int]]]:
    """Bin every detected corner into the GRID and report the filled-cell fraction +
    the per-cell counts — the honest coverage of the calibration data (corners near
    the edges are what constrain distortion)."""
    w, h = int(image_size[0]), int(image_size[1])
    cells = np.zeros((GRID_Y, GRID_X), dtype=int)
    for corners in img_list:
        for x, y in np.asarray(corners, dtype=float).reshape(-1, 2):
            cx = min(GRID_X - 1, max(0, int(x / w * GRID_X)))
            cy = min(GRID_Y - 1, max(0, int(y / h * GRID_Y)))
            cells[cy, cx] += 1
    pct = float((cells > 0).sum()) / (GRID_X * GRID_Y)
    grid = [[int(cells[y, x]) for x in range(GRID_X)] for y in range(GRID_Y)]
    return pct, grid


def solve_intrinsics(obj_list: list[np.ndarray], img_list: list[np.ndarray],
                     image_size: tuple[int, int], K0: np.ndarray, *,
                     fix_k3: bool = True) -> dict:
    """Solve K + distortion from 2D-3D ChArUco correspondences (cv2.calibrateCamera).

    Shared by the interactive session and the under-the-hood auto path in the
    hand-eye run, so both produce the identical report shape. ``fix_k3`` holds the
    high-order radial term at 0 (the D4xx RGB lens is low-distortion; a free k3
    overfits a limited image region).
    """
    if len(obj_list) < MIN_VIEWS:
        raise RuntimeError(
            f"only {len(obj_list)} views; need >= {MIN_VIEWS}. Capture more (cover "
            f"the frame, especially the corners).")
    image_size = (int(image_size[0]), int(image_size[1]))
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | (cv2.CALIB_FIX_K3 if fix_k3 else 0)
    K0 = np.asarray(K0, dtype=np.float64).copy()
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_list, img_list, image_size, K0, np.zeros((5, 1)), flags=flags)

    per_view = []
    for i in range(len(obj_list)):
        proj, _ = cv2.projectPoints(obj_list[i], rvecs[i], tvecs[i], K, dist)
        d = np.linalg.norm(proj.reshape(-1, 2) - np.asarray(img_list[i]).reshape(-1, 2),
                           axis=1)
        per_view.append({"rms_px": float(np.sqrt(np.mean(d ** 2))),
                         "max_px": float(d.max()), "n_corners": int(len(d))})

    d5 = np.concatenate([np.asarray(dist, dtype=float).reshape(-1), np.zeros(5)])[:5]
    fx0, fy0, cx0, cy0 = K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]
    cov_pct, cov_cells = coverage_from_corners(img_list, image_size)
    return {
        "rms_px": float(rms),
        "n_views": len(obj_list),
        "coverage_pct": cov_pct,
        "fix_k3": bool(fix_k3),
        "image_size": list(image_size),
        "K": np.asarray(K, dtype=float).tolist(),
        "dist": [float(x) for x in d5],
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "delta_fx": float(K[0, 0] - fx0), "delta_fy": float(K[1, 1] - fy0),
        "delta_cx": float(K[0, 2] - cx0), "delta_cy": float(K[1, 2] - cy0),
        "per_view": per_view,
        "cells": cov_cells, "grid": [GRID_X, GRID_Y],
    }


# -- live overlay ----------------------------------------------------------
def draw_overlay(image: np.ndarray, corners: np.ndarray | None,
                 ids: np.ndarray | None, status: dict) -> np.ndarray:
    """Draw the coverage grid (filled cells highlighted), the detected corners and a
    capture flash onto a copy of ``image`` — the operator's auto-capture feedback."""
    out = image.copy()
    h, w = out.shape[:2]
    gx, gy = status.get("grid", [GRID_X, GRID_Y])
    cells = status.get("cells") or [[0] * gx for _ in range(gy)]
    overlay = out.copy()
    for y in range(gy):
        for x in range(gx):
            x0, y0 = int(x * w / gx), int(y * h / gy)
            x1, y1 = int((x + 1) * w / gx), int((y + 1) * h / gy)
            filled = cells[y][x] > 0
            if filled:
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 140, 0), -1)
            cv2.rectangle(out, (x0, y0), (x1, y1), (90, 90, 90), 1)
    cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)

    # Highlight the cell the board is currently in.
    cell = status.get("cell")
    if cell is not None:
        x, y = cell
        x0, y0 = int(x * w / gx), int(y * h / gy)
        x1, y1 = int((x + 1) * w / gx), int((y + 1) * h / gy)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 255), 3)

    if corners is not None and ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(out, corners, ids)

    count = status.get("count", 0)
    cov = int(round(status.get("coverage_pct", 0.0) * 100))
    cv2.putText(out, f"views {count}  coverage {cov}%", (20, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
    if status.get("captured"):
        cv2.putText(out, "CAPTURED", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (0, 255, 255), 3, cv2.LINE_AA)
    elif status.get("detected") and not status.get("stable"):
        cv2.putText(out, "hold still...", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (200, 200, 200), 2, cv2.LINE_AA)
    return out
