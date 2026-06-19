"""Generate a print-ready ChArUco calibration board as a PDF.

The board is rendered at EXACT physical size for the chosen page (A4/A3/Letter):
printed at 100% ("actual size"), a square measures exactly ``square_size_mm``.
That matters because detection uses ``square_size_mm`` in mm, and a uniform
print/scale mismatch yields a wrong-scale hand-eye translation that reprojection
error does NOT catch (it's scale-invariant in the board plane). So the page also
carries a 100 mm ruler to verify the print scale, and the UI offers to sync the
computed dimensions into the calibration config.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...core.config import BoardConfig

MM_PER_IN = 25.4

# Page sizes in mm, portrait (width x height).
PAGES_MM: dict[str, tuple[float, float]] = {
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
    "Letter": (215.9, 279.4),
}


@dataclass
class BoardSpec:
    dictionary: str
    squares_x: int
    squares_y: int
    square_size_mm: float
    marker_size_mm: float
    page: str
    landscape: bool
    board_w_mm: float
    board_h_mm: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_spec(board: BoardConfig, page: str = "A4", margin_mm: float = 12.0) -> BoardSpec:
    """Largest board of ``board.squares_x x squares_y`` that fits ``page`` with
    margins, keeping the marker/square ratio from the config. Orientation is
    auto-picked to maximize the square size."""
    page = page if page in PAGES_MM else "A4"
    pw, ph = PAGES_MM[page]
    sx, sy = board.squares_x, board.squares_y
    ratio = board.marker_size_mm / board.square_size_mm

    def fit(w: float, h: float) -> float:
        return min((w - 2 * margin_mm) / sx, (h - 2 * margin_mm) / sy)

    sq_portrait, sq_landscape = fit(pw, ph), fit(ph, pw)
    landscape = sq_landscape > sq_portrait
    square = max(sq_portrait, sq_landscape)
    square = np.floor(square * 10) / 10.0          # clean to 0.1 mm
    marker = round(square * ratio, 1)
    return BoardSpec(board.dictionary, sx, sy, float(square), float(marker),
                     page, landscape, round(sx * square, 1), round(sy * square, 1))


def _font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except Exception:
            continue
    return ImageFont.load_default()


def render_pdf(board: BoardConfig, page: str = "A4", margin_mm: float = 12.0,
               dpi: int = 300) -> tuple[bytes, BoardSpec]:
    """Render the fitted board onto a full page and return ``(pdf_bytes, spec)``."""
    spec = compute_spec(board, page, margin_mm)
    pw_mm, ph_mm = PAGES_MM[spec.page]
    if spec.landscape:
        pw_mm, ph_mm = ph_mm, pw_mm

    def mm2px(mm: float) -> int:
        return int(round(mm / MM_PER_IN * dpi))

    page_px = (mm2px(pw_mm), mm2px(ph_mm))
    board_px = (mm2px(spec.board_w_mm), mm2px(spec.board_h_mm))

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    cb = cv2.aruco.CharucoBoard((spec.squares_x, spec.squares_y),
                                spec.square_size_mm, spec.marker_size_mm, dictionary)
    board_img = cb.generateImage(board_px, marginSize=0, borderBits=1)

    canvas = Image.new("RGB", page_px, "white")
    canvas.paste(Image.fromarray(board_img).convert("RGB"),
                 ((page_px[0] - board_px[0]) // 2, mm2px(margin_mm)))

    draw = ImageDraw.Draw(canvas)
    cap_font = _font(mm2px(3.2))
    cap_y = mm2px(margin_mm) + board_px[1] + mm2px(5)
    draw.text((mm2px(margin_mm), cap_y),
              f"tasni ChArUco  {spec.squares_x}x{spec.squares_y}   "
              f"square = {spec.square_size_mm} mm   marker = {spec.marker_size_mm} mm   "
              f"{spec.dictionary}", fill="black", font=cap_font)
    draw.text((mm2px(margin_mm), cap_y + mm2px(5)),
              "Print at 100% (Actual size) — no 'fit to page'. Then check the ruler below.",
              fill="black", font=cap_font)

    # 100 mm scale-check ruler with end ticks.
    ry = cap_y + mm2px(14)
    rx0 = mm2px(margin_mm)
    rx1 = rx0 + mm2px(100)
    draw.line([(rx0, ry), (rx1, ry)], fill="black", width=max(1, mm2px(0.3)))
    for x in (rx0, rx1):
        draw.line([(x, ry - mm2px(2)), (x, ry + mm2px(2))], fill="black", width=max(1, mm2px(0.3)))
    draw.text((rx0, ry + mm2px(3)), "100 mm — measure to confirm print scale",
              fill="black", font=cap_font)

    buf = BytesIO()
    canvas.save(buf, format="PDF", resolution=dpi)
    return buf.getvalue(), spec
