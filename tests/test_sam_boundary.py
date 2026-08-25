"""sam_boundary.py — the shared mask->boundary tail + the SAM ONNX segmenter.

The ``mask_to_boundary`` tests are pure (synthetic binary masks) and always run — they pin
the abstain-safe geometry both the colour and SAM producers share. The ``sam_work_boundary``
test needs onnxruntime + downloaded weights and **skips** otherwise (mirrors the open3d skip
in test_scan_job.py), so CI without the model stays green.

    py -3.10 tests/test_sam_boundary.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasni.modules.scan.color_boundary import mask_to_boundary  # noqa: E402


def _bbox_uv(outline_uv):
    a = np.asarray(outline_uv, float)
    return a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()


# ---- mask_to_boundary (pure, always run) --------------------------------------

def test_mask_to_boundary_hugs_square():
    m = np.zeros((480, 640), np.uint8)
    m[150:350, 240:440] = 1
    r = mask_to_boundary(m)
    assert r is not None
    umin, vmin, umax, vmax = _bbox_uv(r["outline_uv"])
    assert abs(umin - 240 / 640) < 0.02 and abs(umax - 440 / 640) < 0.02, (umin, umax)
    assert abs(vmin - 150 / 480) < 0.02 and abs(vmax - 350 / 480) < 0.02, (vmin, vmax)
    assert not r["overruns"] and r["border_touch"] == 0.0, r
    print("[mask_to_boundary] hugs a square, no overrun")


def test_mask_to_boundary_prefers_reticle_component():
    # Two separate blobs; the reticle sits on the LEFT (smaller) one -> pick it, not the
    # larger. This is what keeps the boundary on the object the operator is aiming at.
    m = np.zeros((400, 800), np.uint8)
    m[150:250, 80:180] = 1          # small left blob (reticle here)
    m[100:300, 500:760] = 1         # big right blob
    r = mask_to_boundary(m, (130, 200))
    assert r is not None
    umin, _, umax, _ = _bbox_uv(r["outline_uv"])
    assert umax < 0.4, (umin, umax)  # stayed on the left blob
    # No prompt -> the largest component (right blob).
    r2 = mask_to_boundary(m, None)
    u2min, _, u2max, _ = _bbox_uv(r2["outline_uv"])
    assert u2min > 0.5, (u2min, u2max)
    print("[mask_to_boundary] reticle selects its component; else largest")


def test_mask_to_boundary_abstains_when_tiny():
    m = np.zeros((480, 640), np.uint8)
    m[10:16, 10:16] = 1              # ~0.0001 fill, below min_fill_frac
    assert mask_to_boundary(m) is None
    print("[mask_to_boundary] tiny blob -> abstains")


def test_mask_to_boundary_abstains_when_filling_frame():
    m = np.ones((480, 640), np.uint8)  # whole frame -> untrustworthy
    assert mask_to_boundary(m) is None
    print("[mask_to_boundary] frame-filling blob -> abstains")


def test_mask_to_boundary_empty_is_none():
    assert mask_to_boundary(np.zeros((480, 640), np.uint8)) is None
    assert mask_to_boundary(None) is None
    print("[mask_to_boundary] empty mask -> None")


# ---- sam_work_boundary (needs onnxruntime + weights; skips otherwise) ----------

def _sam_ready():
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    enc = ROOT / "models" / "edge_sam_encoder.onnx"
    dec = ROOT / "models" / "edge_sam_decoder.onnx"
    return enc.exists() and dec.exists()


def test_sam_segments_synthetic_square():
    if not _sam_ready():
        print("[skip] SAM weights/onnxruntime absent — `pip install -e .[sam]` + "
              "`py -3.10 tools/download_sam.py`")
        return
    from tasni.modules.scan.sam_boundary import sam_work_boundary
    img = np.full((480, 640, 3), 30, np.uint8)
    img[150:350, 240:440] = 235                     # bright square on dark bg
    r = sam_work_boundary(img, point_uv=(0.53, 0.52),
                          model_dir=str(ROOT / "models"))
    assert r is not None, "SAM should find the square"
    umin, vmin, umax, vmax = _bbox_uv(r["outline_uv"])
    assert abs(umin - 240 / 640) < 0.05 and abs(umax - 440 / 640) < 0.05, (umin, umax)
    assert abs(vmin - 150 / 480) < 0.05 and abs(vmax - 350 / 480) < 0.05, (vmin, vmax)
    assert r["contrast"] > 0.8 and not r["overruns"], r
    print("[sam] segmented a synthetic square (score %.3f)" % r["contrast"])


def test_sam_abstains_on_featureless_frame():
    if not _sam_ready():
        print("[skip] SAM weights/onnxruntime absent")
        return
    from tasni.modules.scan.sam_boundary import sam_work_boundary
    flat = np.full((480, 640, 3), 120, np.uint8)
    assert sam_work_boundary(flat, model_dir=str(ROOT / "models")) is None
    print("[sam] featureless frame -> abstains (None)")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all sam_boundary tests passed")
