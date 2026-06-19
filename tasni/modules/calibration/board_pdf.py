"""Render the calibration board — the SAME board detection uses.

The board geometry lives in one place (:class:`~tasni.core.config.BoardConfig`).
We render exactly that board at its true physical size, so a 100%-scale print
matches the detection config by construction — there is no "match" step. (A
print/scale mismatch would silently scale the hand-eye translation, and
reprojection error can't catch it, which is why this must be exact.) The page
just provides paper to print on; the board dimensions never change with it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...core.config import BoardConfig

MM_PER_IN = 25.4
PAGES_MM: dict[str, tuple[float, float]] = {     # portrait width x height
    "A4": (210.0, 297.0), "A3": (297.0, 420.0), "Letter": (215.9, 279.4),
}


@dataclass
class BoardSpec:
    dictionary: str
    squares_x: int
    squares_y: int
    square_size_mm: float
    marker_size_mm: float
    board_w_mm: float
    board_h_mm: float
    page: str
    landscape: bool
    fits: bool                          # does the true-size board fit this page?
    pages: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _board_size_mm(b: BoardConfig) -> tuple[float, float]:
    return round(b.squares_x * b.square_size_mm, 1), round(b.squares_y * b.square_size_mm, 1)


def board_spec(board: BoardConfig, page: str = "A4", margin_mm: float = 10.0) -> BoardSpec:
    page = page if page in PAGES_MM else "A4"
    bw, bh = _board_size_mm(board)
    pw, ph = PAGES_MM[page]
    # Pick the page orientation that fits the (fixed-size) board best.
    fit_portrait = bw <= pw - 2 * margin_mm and bh <= ph - 2 * margin_mm
    fit_landscape = bw <= ph - 2 * margin_mm and bh <= pw - 2 * margin_mm
    landscape = fit_landscape and (not fit_portrait or bw > bh)
    return BoardSpec(board.dictionary, board.squares_x, board.squares_y,
                     board.square_size_mm, board.marker_size_mm, bw, bh,
                     page, landscape, bool(fit_portrait or fit_landscape),
                     list(PAGES_MM))


def _charuco_image(board: BoardConfig, long_px: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board.dictionary))
    cb = cv2.aruco.CharucoBoard((board.squares_x, board.squares_y),
                                board.square_size_mm, board.marker_size_mm, dictionary)
    bw, bh = _board_size_mm(board)
    if bw >= bh:
        size = (long_px, max(1, round(long_px * bh / bw)))
    else:
        size = (max(1, round(long_px * bw / bh)), long_px)
    return cb.generateImage(size, marginSize=0, borderBits=1)


def render_png(board: BoardConfig, long_px: int = 700) -> bytes:
    """A plain PNG of the board pattern for an in-app visual reference."""
    ok, buf = cv2.imencode(".png", _charuco_image(board, long_px))
    if not ok:
        raise RuntimeError("failed to encode board PNG")
    return buf.tobytes()


def _font(px: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except Exception:
            continue
    return ImageFont.load_default()


def render_pdf(board: BoardConfig, page: str = "A4", margin_mm: float = 10.0,
               dpi: int = 300) -> tuple[bytes, BoardSpec]:
    """Render the board at TRUE physical size, centered on the page."""
    spec = board_spec(board, page, margin_mm)
    pw_mm, ph_mm = PAGES_MM[spec.page]
    if spec.landscape:
        pw_mm, ph_mm = ph_mm, pw_mm

    def mm2px(mm: float) -> int:
        return int(round(mm / MM_PER_IN * dpi))

    page_px = (mm2px(pw_mm), mm2px(ph_mm))
    board_px = (mm2px(spec.board_w_mm), mm2px(spec.board_h_mm))
    board_img = Image.fromarray(_charuco_image(board, max(board_px))).convert("RGB")
    board_img = board_img.resize(board_px)

    canvas = Image.new("RGB", page_px, "white")
    ox = (page_px[0] - board_px[0]) // 2
    canvas.paste(board_img, (ox, mm2px(margin_mm)))

    d = ImageDraw.Draw(canvas)
    font = _font(mm2px(3.2))
    cy = mm2px(margin_mm) + board_px[1] + mm2px(5)
    d.text((mm2px(margin_mm), cy),
           f"tasni ChArUco  {spec.squares_x}x{spec.squares_y}   "
           f"square = {spec.square_size_mm} mm   marker = {spec.marker_size_mm} mm   "
           f"{spec.dictionary}", fill="black", font=font)
    d.text((mm2px(margin_mm), cy + mm2px(5)),
           "Print at 100% (Actual size) — no 'fit to page'. Then check the ruler below.",
           fill="black", font=font)

    ry, rx0 = cy + mm2px(14), mm2px(margin_mm)
    rx1 = rx0 + mm2px(100)
    d.line([(rx0, ry), (rx1, ry)], fill="black", width=max(1, mm2px(0.3)))
    for x in (rx0, rx1):
        d.line([(x, ry - mm2px(2)), (x, ry + mm2px(2))], fill="black", width=max(1, mm2px(0.3)))
    d.text((rx0, ry + mm2px(3)), "100 mm — measure to confirm print scale",
           fill="black", font=font)

    out = BytesIO()
    canvas.save(out, format="PDF", resolution=dpi)
    return out.getvalue(), spec
