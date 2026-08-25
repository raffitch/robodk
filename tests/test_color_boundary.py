"""color_boundary.py — classical color segmentation of the object under the reticle.

Renders synthetic color frames (a block on a table) and asserts the returned boundary
hugs the block, keys the object side off the reticle (works light-on-dark AND
dark-on-light), and abstains when there is no contrast. No RealSense / RoboDK.

    py -3.10 tests/test_color_boundary.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.modules.scan.color_boundary import color_work_boundary  # noqa: E402


def _frame(w, h, block, fill, bg):
    """A (h,w,3) BGR frame: background ``bg``, a filled ``block`` rect (x0,y0,x1,y1)."""
    img = np.full((h, w, 3), bg, np.uint8)
    x0, y0, x1, y1 = block
    img[y0:y1, x0:x1] = fill
    return img


def _bbox_uv(outline_uv):
    a = np.asarray(outline_uv, float)
    return a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()


def test_light_block_on_dark_table():
    w, h = 640, 480
    block = (200, 150, 460, 350)          # centered-ish bright block
    img = _frame(w, h, block, fill=235, bg=30)
    r = color_work_boundary(img)
    assert r is not None, "should find the block"
    umin, vmin, umax, vmax = _bbox_uv(r["outline_uv"])
    # rectangle hugs the block edges (within a few percent of the true normalized bounds)
    assert abs(umin - 200 / w) < 0.04 and abs(umax - 460 / w) < 0.04, (umin, umax)
    assert abs(vmin - 150 / h) < 0.04 and abs(vmax - 350 / h) < 0.04, (vmin, vmax)
    assert r["contrast"] > 100 and not r["overruns"], r
    print("[color boundary] light block on dark table -> tight rectangle")


def test_dark_block_on_light_table_object_agnostic():
    # The object is DARKER than the background: the reticle sample must still pick the
    # block (not the table) as the object.
    w, h = 640, 480
    block = (220, 170, 430, 330)
    img = _frame(w, h, block, fill=25, bg=210)
    r = color_work_boundary(img)
    assert r is not None
    umin, vmin, umax, vmax = _bbox_uv(r["outline_uv"])
    assert abs(umin - 220 / w) < 0.05 and abs(umax - 430 / w) < 0.05, (umin, umax)
    assert abs(vmin - 170 / h) < 0.05 and abs(vmax - 330 / h) < 0.05, (vmin, vmax)
    print("[color boundary] dark block on light table -> reticle keys the object side")


def test_green_object_on_gray_table_uses_colour_not_luminance():
    # THE REAL CELL: a green cutting mat on a gray table has ~equal LUMINANCE but very
    # different COLOUR. A grayscale threshold misses it; Lab colour distance finds it.
    import cv2
    w, h = 640, 480
    img = np.full((h, w, 3), (83, 83, 83), np.uint8)       # gray table (BGR)
    img[150:350, 200:460] = (30, 120, 30)                   # green block (BGR), ~same luma
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)               # grayscale: nearly no contrast
    assert abs(int(np.median(g[150:350, 200:460])) - int(np.median(g[:8]))) < 20
    r = color_work_boundary(img)
    assert r is not None, "colour distance should find the green block a gray threshold misses"
    umin, vmin, umax, vmax = _bbox_uv(r["outline_uv"])
    assert abs(umin - 200 / w) < 0.05 and abs(umax - 460 / w) < 0.05, (umin, umax)
    print("[color boundary] green-on-gray segmented by colour, not luminance")


def test_low_contrast_abstains():
    # Object looks like the background (same colour AND luminance) -> abstains.
    img = _frame(640, 480, (200, 150, 460, 350), fill=(126, 126, 126), bg=(122, 122, 122))
    assert color_work_boundary(img) is None
    print("[color boundary] no colour separation -> abstains (None)")


def test_object_filling_view_abstains():
    # A blob that fills most of the frame is untrustworthy (genuine overrun OR a failed
    # low-contrast segmentation) -> abstain (None) so the HUD falls back to depth, never
    # drawing a whole-frame rectangle. This is the guard that caught the green-mat scene.
    img = _frame(640, 480, (5, 5, 635, 475), fill=235, bg=30)
    assert color_work_boundary(img) is None
    print("[color boundary] object fills the view -> abstains (falls back to depth)")


def test_tracks_block_position():
    # As the operator aims, the object stays under the reticle but its edges shift; the
    # returned rectangle must follow (the whole point: a live boundary at video rate).
    # Both blocks still cover the reticle center (x=320) — the app's aiming model.
    w, h = 640, 480
    left = color_work_boundary(_frame(w, h, (220, 160, 420, 340), 235, 30))
    right = color_work_boundary(_frame(w, h, (300, 160, 500, 340), 235, 30))
    assert left is not None and right is not None
    cl = np.asarray(left["outline_uv"], float)[:, 0].mean()
    cr = np.asarray(right["outline_uv"], float)[:, 0].mean()
    assert cr > cl + 0.05, (cl, cr)
    print("[color boundary] rectangle follows the object across the frame")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all color_boundary tests passed")
