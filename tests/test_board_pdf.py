"""Calibration-board paper profiles stay synchronized with detector geometry."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasni.core.config import BoardConfig  # noqa: E402
from tasni.modules.calibration.board_pdf import board_for_page, board_spec, render_pdf  # noqa: E402


def test_a3_uses_largest_clean_8x6_profile():
    board = BoardConfig()
    a3 = board_for_page(board, "A3")
    spec = board_spec(board, "A3")

    assert a3.paper_size == "A3"
    assert a3.square_size_mm == 40.0
    assert a3.marker_size_mm == 29.3
    assert (spec.board_w_mm, spec.board_h_mm) == (320.0, 240.0)
    assert spec.landscape is True
    assert spec.fits is True


def test_switching_back_restores_standard_geometry():
    a3 = board_for_page(BoardConfig(), "A3")
    a4 = board_for_page(a3, "A4")

    assert a4.paper_size == "A4"
    assert a4.square_size_mm == 30.0
    assert a4.marker_size_mm == 22.0


def test_a3_pdf_reports_the_same_geometry_it_renders():
    pdf, spec = render_pdf(BoardConfig(), "A3", dpi=72)

    assert pdf.startswith(b"%PDF")
    assert (spec.board_w_mm, spec.board_h_mm) == (320.0, 240.0)


if __name__ == "__main__":
    test_a3_uses_largest_clean_8x6_profile()
    test_switching_back_restores_standard_geometry()
    test_a3_pdf_reports_the_same_geometry_it_renders()
    print("Board-profile tests passed.")
