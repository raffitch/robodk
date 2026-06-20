"""ChArUco detection + median-of-frames, on a rendered board (no camera/robot).

Verifies the detection chain finds the board's own rendering and that
``detect_median`` (the per-corner pixel median introduced for capture robustness)
stays sub-pixel-faithful to a clean single-frame detection under per-frame noise.

    py -3.10 tests/test_detection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core.config import BoardConfig  # noqa: E402
from tasni.modules.calibration.charuco import CharucoTarget  # noqa: E402

K = np.array([[1200.0, 0, 640.0], [0, 1200.0, 480.0], [0, 0, 1]])
DIST = np.zeros((5, 1))


def _render(board: CharucoTarget) -> np.ndarray:
    img = board.board.generateImage((1280, 960))
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def test_detect_finds_all_inner_corners():
    board = CharucoTarget(BoardConfig())          # 8x6 squares -> 7x5 = 35 corners
    det = board.detect(_render(board), K, DIST, min_corners=6)
    assert det is not None, "board not detected in its own rendering"
    assert det.n_corners == 35, f"expected 35 corners, got {det.n_corners}"
    print(f"[detect] {det.n_corners} corners")


def test_median_is_robust_to_per_frame_noise():
    board = CharucoTarget(BoardConfig())
    clean = _render(board)
    base = board.detect(clean, K, DIST, min_corners=6)
    assert base is not None

    rng = np.random.default_rng(0)
    noisy = [np.clip(clean.astype(np.int16) + rng.normal(0, 6, clean.shape),
                     0, 255).astype(np.uint8) for _ in range(7)]
    med = board.detect_median(noisy, K, DIST, min_corners=6)
    assert med is not None and med.n_corners >= 6

    base_map = {int(i): c for i, c in zip(base.ids.flatten(), base.corners.reshape(-1, 2))}
    med_map = {int(i): c for i, c in zip(med.ids.flatten(), med.corners.reshape(-1, 2))}
    shared = set(base_map) & set(med_map)
    assert len(shared) >= 6
    err = float(np.mean([np.linalg.norm(base_map[i] - med_map[i]) for i in shared]))
    assert err < 1.5, f"median corners drifted {err:.3f}px from clean"
    print(f"[median] {med.n_corners} corners, mean drift {err:.3f}px over {len(shared)} ids")


if __name__ == "__main__":
    test_detect_finds_all_inner_corners()
    test_median_is_robust_to_per_frame_noise()
    print("\nDetection tests passed.")
