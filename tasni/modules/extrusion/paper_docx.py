"""A Word draft of the ring-stack results, rebuilt from the archive on demand.

The point is that nobody transcribes numbers. Every value here comes from
``paper_summary`` -- the same object the app shows -- so the document, the app
and the archive cannot disagree, and the tables arrive as real Word tables that
paste into the manuscript with their structure intact.

It is written to be read DURING collection: what is still missing is listed
explicitly, so a half-finished run is obviously half-finished rather than
quietly citable.

Needs the ``docx`` extra (``pip install -e .[docx]``); imported lazily so every
measurement still runs without it.
"""
from __future__ import annotations

import time
from pathlib import Path

from .measure import paper_summary

# The protocol's own targets, and requirement 3 of the paper's outstanding list.
TAKES_PER_CONDITION = 3
MEASUREMENTS_FOR_TIMING = 12

# Stated once, where the claim is made. Hand-eye board consistency and
# work-plane RMS for this cell; nothing below this floor may be reported.
ERROR_FLOOR_MM = 1.26
WORK_PLANE_RMS_MM = 1.39
STANDOFF_MM = 300

METHOD = (
    "Rings of dried, previously extruded material were placed by hand on the scanned work "
    "surface, one on top of another; no material was extruded and no valve was actuated "
    "during the measurements. Each measurement moved only the camera to a viewpoint "
    f"{STANDOFF_MM} mm above the top of the layer being measured, normal to the work plane, "
    "captured one RGB-D frame from an Intel RealSense D435i at 1280x720, and reconstructed "
    "the deposited ring's centreline in the work frame. Layer N was segmented above the "
    "measured top surface of layer N-1, so a displaced ring is compared against the ring it "
    "actually sits on. What follows is therefore a controlled validation of the "
    "sensing-and-comparison chain against a known introduced offset, and not the deposition "
    "deviation of a printed cylinder. Hand-eye calibration for this cell has a "
    f"board-consistency residual of {ERROR_FLOOR_MM:.2f} mm and a work-plane RMS of "
    f"{WORK_PLANE_RMS_MM:.2f} mm; no accuracy is claimed below that floor."
)

CONDITION_HEADERS = ["Condition", "n", "Centre offset (mm)", "Detection error (mm)",
                     "Mean |dev| (mm)", "RMS (mm)", "Max (mm)", "Shape RMS (mm)"]
TAKE_HEADERS = ["Layer", "Take", "Phase", "Introduced (mm)", "Measured offset (mm)",
                "Detection error (mm)", "RMS (mm)", "Acq to path (ms)",
                "Layer cost (s)", "Valid"]


def _pm(text: str) -> str:
    """Markdown writes +/- so it stays ASCII; Word can have the real sign."""
    return text.replace("+/-", "±")


def _stat_cell(stat: dict, digits: int = 2) -> str:
    if not stat or stat.get("mean") is None:
        return "-"
    text = f"{stat['mean']:.{digits}f}"
    if stat.get("sd") is not None:
        text += f" ± {stat['sd']:.{digits}f}"
    return text


def _offset_cell(offset) -> str:
    if not offset or not any(float(v) for v in offset):
        return "-"
    return f"({offset[0]:g}, {offset[1]:g})"


def _detection_error(manifest: dict) -> "float | None":
    from .measure import detection_error_mm
    return detection_error_mm(manifest)


def _gaps(summary: dict) -> list[str]:
    """What this run still owes the paper, in the operator's terms."""
    notes: list[str] = []
    for condition in summary["conditions"]:
        short = TAKES_PER_CONDITION - condition["takes"]
        if short > 0:
            notes.append(f"{condition['condition']}: {condition['takes']} take(s) recorded, "
                         f"{short} more for a mean and standard deviation.")
        invalid = condition["takes"] - condition["valid"]
        if invalid:
            notes.append(f"{condition['condition']}: {invalid} take(s) invalid and excluded "
                         "from every average. Reprocess them or repeat them.")
    recorded = summary["takes"]
    if recorded < MEASUREMENTS_FOR_TIMING:
        notes.append(f"{recorded} measurement(s) recorded; the acquisition-to-path mean and "
                     f"standard deviation needs {MEASUREMENTS_FOR_TIMING} -- "
                     f"{MEASUREMENTS_FOR_TIMING - recorded} more.")
    if not any(c["introduced_norm_mm"] > 0 for c in summary["conditions"]):
        notes.append("No condition with an introduced offset yet. The paper's claim is that a "
                     "known displacement is recovered, so at least one is required.")
    return notes


def _add_table(document, headers: list[str], rows: list[list[str]]):
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for cell, text in zip(table.rows[0].cells, headers):
        cell.text = text
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
    for values in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, values):
            cell.text = text
    return table


def _figure_paths(trial_dir: Path, summary: dict) -> list[tuple[Path, str]]:
    """The stack, plus the plan view of the last take of each condition."""
    from .figures import ensure_figure

    wanted: list[tuple[Path, str]] = []
    # The method figure leads: it is the paper's account of how a depth frame
    # becomes the number in Table 1.
    first = next((m for m in summary["manifests"]
                  if (m.get("metrics") or {}).get("valid")), None)
    method_dir = None if first is None else _layer_dir_for(trial_dir, first)
    if method_dir is not None:
        try:
            wanted.append((ensure_figure(method_dir, "pipeline.png"),
                           "One RGB-D frame becoming a measured centreline: captured depth, "
                           "the work region of interest, the deposit cluster, its top surface, "
                           "and the extracted centreline against nominal."))
        except Exception:
            pass
    for stem, caption in (
            ("stack.png", "Every layer's latest measured centreline against nominal."),
            ("tube.png", "The commanded bead drawn at its real diameter, each layer at its "
                         "own height, with the measured centreline running through it.")):
        try:
            wanted.append((ensure_figure(trial_dir, stem), caption))
        except Exception:
            continue
    from .measure import _condition_name
    for condition in summary["conditions"]:
        # The latest VALID take of this condition: the one a reader would check
        # the table against.
        chosen = None
        for manifest in summary["manifests"]:
            if not (manifest.get("metrics") or {}).get("valid"):
                continue
            if _condition_name(manifest) == condition["condition"]:
                chosen = manifest
        layer_dir = None if chosen is None else _layer_dir_for(trial_dir, chosen)
        if layer_dir is None:
            continue
        try:
            wanted.append((ensure_figure(layer_dir, "plan.png"),
                           f"{condition['condition']}: extracted centreline against nominal."))
        except Exception:
            continue
    return wanted


def _layer_dir_for(trial_dir: Path, manifest: dict) -> "Path | None":
    layer = int(manifest.get("layer_index") or 0)
    take = int(manifest.get("take") or 1)
    name = f"layer-{layer:03d}" + ("" if take == 1 else f"-take{take:02d}")
    path = trial_dir / name
    return path if path.is_dir() else None


def build_paper_docx(root, trial_id: str, out_path=None, *,
                     embed_figures: bool = True) -> Path:
    """Write the results draft for ``trial_id`` and return the file path."""
    from docx import Document                       # the `docx` extra
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Mm, Pt

    root = Path(root)
    trial_dir = root / trial_id
    summary = paper_summary(root, trial_id)
    out_path = Path(out_path) if out_path else (trial_dir / "paper-draft.docx")

    document = Document()
    document.add_heading("Ring-stack validation - results draft", level=1)
    stamp = document.add_paragraph(
        f"Trial {trial_id} - generated {time.strftime('%Y-%m-%d %H:%M')} - "
        f"{summary['takes']} measurement(s), {summary['valid']} valid. "
        "Rebuilt from the archive; every number here is the one the application reports.")
    stamp.runs[0].italic = True
    stamp.runs[0].font.size = Pt(9)

    document.add_heading("Method (paste into the method section)", level=2)
    document.add_paragraph(METHOD)

    document.add_heading("Results (paste into the results section)", level=2)
    document.add_paragraph(_pm(summary["headline"]))

    document.add_paragraph("Table 1. Geometric deviation by condition. Deviation is measured "
                           "from the nominal centre; detection error is the difference between "
                           "the measured displacement and the displacement introduced by hand.")
    _add_table(document, CONDITION_HEADERS,
               [[c["condition"], str(c["takes"]),
                 _stat_cell(c["center_offset_norm_mm"]), _stat_cell(c["detection_error_mm"]),
                 _stat_cell(c["mean_absolute_mm"]), _stat_cell(c["rms_mm"]),
                 _stat_cell(c["maximum_mm"]), _stat_cell(c["shape_rms_mm"])]
                for c in summary["conditions"]])
    document.add_paragraph()

    for paragraph in summary["prose"]:
        added = document.add_paragraph(_pm(paragraph))
        if paragraph.startswith("WARNING"):
            for run in added.runs:
                run.bold = True

    gaps = _gaps(summary)
    if gaps:
        document.add_heading("Still missing (delete before submitting)", level=2)
        warning = document.add_paragraph(
            "Not ready to cite as it stands. This run still owes:")
        warning.runs[0].bold = True
        for note in gaps:
            document.add_paragraph(note, style="List Bullet")

    document.add_heading("Every take (working record)", level=2)
    ordered = sorted(summary["manifests"],
                     key=lambda m: (int(m.get("layer_index") or 0), int(m.get("take") or 1)))
    rows = []
    for manifest in ordered:
        metrics = manifest.get("metrics") or {}
        timings = (manifest.get("processing") or {}).get("timings_ms") or {}
        error = _detection_error(manifest)
        acq = timings.get("acquisition_to_path_ms")
        rows.append([
            str(manifest.get("layer_index", "")), str(manifest.get("take", 1)),
            str((manifest.get("annotation") or {}).get("phase") or "-"),
            _offset_cell((manifest.get("annotation") or {}).get("introduced_offset_mm")),
            f"{metrics['center_offset_norm_mm']:.2f}" if metrics.get("center_offset_norm_mm") is not None else "-",
            f"{error:.2f}" if error is not None else "-",
            f"{metrics['rms_mm']:.2f}" if metrics.get("rms_mm") is not None else "-",
            f"{acq:.0f}" if acq else "-",
            f"{timings['inspection_cycle_ms'] / 1000:.1f}" if timings.get("inspection_cycle_ms") else "-",
            "yes" if metrics.get("valid") else "no",
        ])
    _add_table(document, TAKE_HEADERS, rows)

    if embed_figures:
        document.add_heading("Figures", level=2)
        try:
            figures = _figure_paths(trial_dir, summary)
        except ImportError:
            figures = []
            document.add_paragraph(
                "Figures were not rendered here (matplotlib is absent: pip install -e "
                ".[figures]). They live beside each take in the archive.")
        for index, (path, caption) in enumerate(figures, start=1):
            document.add_picture(str(path), width=Mm(150))
            document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            label = document.add_paragraph(f"Figure {index}. {caption} "
                                           f"Vector original: {path.with_suffix('.pdf')}")
            label.runs[0].italic = True
            label.runs[0].font.size = Pt(9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(out_path))
    return out_path
