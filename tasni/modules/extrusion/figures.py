"""Publication and in-app figures rendered from an archived take.

Everything here reads ONLY the archive (``manifest.json`` + the arrays written
next to it), so a figure can be produced long after the cell run, on a machine
with no robot and no camera, and re-produced identically. Nothing in this module
touches the robot, RoboDK or the job runner.

Seven figures per take:

``plan``       top view: deposit cloud, extracted centreline, nominal circle
``heightmap``  bird's-eye height map of the re-projected depth frame, z colourbar
``mesh``       the frame SURFACED: scene and deposit, each straight down and rotated
``iso``        the 3-D scene at a controllable azimuth/elevation, default oblique
``birdseye``   the SAME 3-D view pinned top-down, orthographic, framed to fit
               the whole ring with margin -- ``render_view(..., birdseye=True)``
``profile``    unrolled: height z(theta) and radial deviation dr(theta) over 360 deg
``pipeline``   the method figure: the six arrays the chain held, in order

and two per trial: ``stack`` (every layer's latest take, plan + oblique) and
``tube`` (the commanded bead against the measured footprint).

A take is either an archived layer (``layer-*/manifest.json``) or a ring
characterization (``characterize-*/report.json``, written before any recipe
exists -- see ``ExtrusionArchive.write_characterization``). Both load through
the same ``load_take``/``TakeData`` path so every figure in this module can be
produced from either kind of archived take.

Matplotlib is imported lazily behind the Agg backend: importing this module must
not require a display, and ``tasni`` must still import when matplotlib is absent.
"""
from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...core.depth_geometry import CameraGeometry
from .processing import depth_to_work_points

log = logging.getLogger(__name__)

LAYER_FIGURES = ("plan", "heightmap", "mesh", "iso", "birdseye", "profile", "pipeline")
TRIAL_FIGURES = ("stack", "tube")
FORMATS = ("png", "pdf")
DPI = 300

# The clouds ``process_observation`` hands back through its ``stages`` collector,
# in the order it holds them. Used to tell "the chain produced nothing" from
# "the chain stopped part-way" -- see ``_incomplete``.
STAGE_KEYS = ("backprojected", "work_roi", "above_floor", "deposit_cluster",
              "radial_trimmed", "top_surface")

# A 3-D box flatter than this reads as a pancake rather than a surface. The
# extra relief goes into the exaggeration FACTOR -- which multiplies the plotted
# Z, is printed on the Z axis and is stated in the caption -- and NEVER into the
# box aspect on its own: a floor on the box is exaggeration no reader can see.
MIN_RELIEF_RATIO = .12

# Heights that can plausibly belong to the work surface, in mm about the build
# plane. A raw D435i frame reaches the rest of the room: the failed take
# 20260828-124136 back-projects 12 m wide with NOTHING inside this band. Points
# outside it must never set a colour scale or the extent of a panel.
WORK_BAND_MM = (-15.0, 60.0)

# One palette for every figure, chosen to survive greyscale printing: the
# measured path is the darkest line, the nominal is dashed, the cloud is light.
CLOUD = "#98a2b3"
MEASURED = "#c1121f"
NOMINAL = "#2b6cb0"
ACCENT = "#b45309"
GROUND_TRUTH = "#0f766e"   # where a DISPLACED ring should be; not the nominal ring
CMAP = "viridis"           # perceptually uniform, colour-blind safe, prints well
# The honesty caption ("hand-placed bead, not printed") is the sentence that
# stops a figure being read as a printed-cylinder result, and with an introduced
# offset recorded it outruns the figure width. Wrap it rather than let the ends
# be cut off: 92 characters at 7.5 pt fits a 6 in figure with margin to spare.
CAPTION_WRAP = 92


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _colormap(name: str):
    """Empty height cells read as 'no return', not as the lowest height."""
    import matplotlib
    cmap = matplotlib.colormaps[name].copy()
    cmap.set_bad("#f3f4f6")
    return cmap


def wrap_caption(caption: str, width: int = CAPTION_WRAP) -> str:
    """Break a caption onto as many lines as it needs. Short ones stay on one."""
    # break_on_hyphens would split "hand-placed bead" across two lines, which is
    # exactly the phrase a reader must not have to reassemble.
    lines = textwrap.wrap(caption, width=width, break_on_hyphens=False,
                          break_long_words=False)
    return "\n".join(lines) or caption


def _points(path: Path) -> np.ndarray | None:
    """Read an archived path file (``{frame, units, points}``)."""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = payload.get("points") if isinstance(payload, dict) else payload
    array = np.asarray(points, dtype=float)
    return array.reshape(-1, 3) if array.size else None


@dataclass
class TakeData:
    """Everything a figure can draw, loaded from one layer directory."""

    layer_dir: Path
    manifest: dict
    nominal: np.ndarray | None
    measured: np.ndarray | None
    cloud: np.ndarray | None
    depth: np.ndarray | None
    K: np.ndarray | None
    T_work_camera: np.ndarray | None
    geometry: CameraGeometry | None = None
    dist_coeffs: np.ndarray | None = None

    @property
    def chroma_dist(self) -> "np.ndarray | None":
        """The colour lens model the MEASUREMENT sampled the chroma gate through.

        The chain projects registered depth points into the colour image and
        reads the bead/board gate at those pixels, so this choice decides which
        pixels are read -- a method figure that re-runs with a different model
        can show a DIFFERENT segmentation from the number it is captioned with
        (on the cell's protocol-2 characterization take 2.7% of registered
        points flip class, moving the fitted centre 0.27 mm and the RMS 0.12 mm).

        The rule is the measurement's, not the figure's, and both the live
        capture and offline reprocessing already follow it (see
        ``service.reprocess_layer``): a protocol-2 take registers native,
        unaligned depth into colour through the CALIBRATED model, so the gate is
        read at distorted pixels; a legacy (pre-protocol-2) take arrived already
        aligned, its registration is the identity (depth K == colour K, zero
        extrinsic), and pushing distortion through it would sample the gate
        off-pixel. ``None`` for a legacy take is therefore the model it used,
        not an omission.
        """
        if self.geometry is None or self.geometry.legacy:
            return None
        return self.dist_coeffs

    @property
    def _nominal_circle(self) -> tuple[tuple[float, float], float] | None:
        """Fit the archived nominal ring rather than averaging it.

        The archive writes a CLOSED path, so its first point is repeated; the
        arithmetic mean is then biased by radius/N (0.33 mm on the cell's
        181-point 40 mm ring) and every plotted deviation inherits that bias.
        """
        if self.nominal is None or len(self.nominal) < 3:
            return None
        from .comparison import fit_circle_xy
        center, radius = fit_circle_xy(self.nominal)
        return (float(center[0]), float(center[1])), float(radius)

    @property
    def center(self) -> tuple[float, float]:
        """The NOMINAL centre -- deviation is measured from it, not from the fit."""
        circle = self._nominal_circle
        if circle is not None:
            return circle[0]
        metrics = self.manifest.get("metrics") or {}
        found = metrics.get("measured_center_mm")
        return (float(found[0]), float(found[1])) if found else (0.0, 0.0)

    @property
    def radius(self) -> float:
        recipe = self.manifest.get("recipe") or {}
        if recipe.get("radius_mm"):
            return float(recipe["radius_mm"])
        circle = self._nominal_circle
        return circle[1] if circle is not None else 50.0

    @property
    def label(self) -> str:
        take = self.manifest.get("take") or 1
        if self.manifest.get("kind") == "characterization":
            label = f"{self.manifest.get('trial_id', '?')} · characterization {take}"
        else:
            label = (f"{self.manifest.get('trial_id', '?')} · "
                    f"layer {self.manifest.get('layer_index', '?')} take {take}")
        # The archive's one read-side fallback (figures.geometry_for_take):
        # a take with no recorded greeting is rendered as it was captured,
        # aligned depth at 1 mm -- flag it so a reader does not mistake it
        # for a protocol-2 (native, unaligned, 0.1 mm) capture.
        if self.geometry is not None and self.geometry.legacy:
            label += " · depth: legacy aligned 1 mm"
        return label

    @property
    def caption(self) -> str:
        """The one line that keeps a figure honest when it is pasted into a paper."""
        metrics = self.manifest.get("metrics") or {}
        annotation = self.manifest.get("annotation") or {}
        parts = []
        if self.manifest.get("kind") == "characterization":
            # No recipe existed yet, so there is no "introduced offset" to
            # report -- a characterization measures a ring as it was found.
            parts.append("ring characterization: coarse-fit ROI, no recipe assumed")
        else:
            offset = annotation.get("introduced_offset_mm")
            if offset and any(float(v) for v in offset):
                parts.append(f"introduced offset ({offset[0]:g}, {offset[1]:g}) mm")
            else:
                parts.append("no introduced offset")
        if metrics.get("center_offset_norm_mm") is not None:
            parts.append(f"measured centre offset {metrics['center_offset_norm_mm']:.2f} mm")
        if metrics.get("rms_mm") is not None:
            parts.append(f"RMS {metrics['rms_mm']:.2f} mm")
        if self.manifest.get("mode") == "MEASURE_ONLY":
            parts.append("hand-placed bead, not printed")
        return " · ".join(parts)


def expected_ring(take: TakeData) -> np.ndarray | None:
    """Where this take's ring SHOULD sit: the nominal circle moved by the offset
    the operator typed in before pressing Measure.

    This is the ground truth the controlled validation is scored against, so the
    figure can show the extracted centreline landing ON it rather than merely
    near the nominal ring it was deliberately moved away from.

    ``None`` unless a non-zero offset was actually introduced: a take with no
    displacement must not get a second line identical to the nominal circle,
    which a reader would take for evidence of something.
    """
    offset = (take.manifest.get("annotation") or {}).get("introduced_offset_mm")
    if take.nominal is None or not offset or len(offset) != 2:
        return None
    dx, dy = float(offset[0]), float(offset[1])
    if dx == 0.0 and dy == 0.0:
        return None
    moved = np.array(take.nominal, dtype=float, copy=True)
    moved[:, 0] += dx
    moved[:, 1] += dy
    return moved


def _camera_model(manifest: dict, layer_dir: Path
                  ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The archived colour model (K, dist_coeffs) for this take.

    The take's own ``camera_intrinsics`` block first, the trial's as the
    fallback -- the same order (and the same block, never K from one and the
    distortion from the other) that ``service.reprocess_layer`` reads them in.
    """
    blocks = [(manifest.get("provenance") or {}).get("camera_intrinsics") or {}]
    trial_file = layer_dir.parent / "trial.json"
    if trial_file.is_file():
        trial = json.loads(trial_file.read_text(encoding="utf-8"))
        blocks.append((trial.get("provenance") or {}).get("camera_intrinsics") or {})
    block = next((b for b in blocks if b.get("K") is not None), {})
    K = block.get("K")
    dist = block.get("dist_coeffs")
    return (None if K is None else np.asarray(K, dtype=float),
            None if dist is None else np.asarray(dist, dtype=float))


def geometry_for_take(manifest: dict, K: np.ndarray | None,
                      depth: np.ndarray | None) -> "CameraGeometry | None":
    """The take's depth geometry: protocol-2 greeting from provenance, else the
    legacy aligned model. This is the ONE place the archive read path may still
    build a legacy geometry -- live code never does (see ``depth_geometry.py``);
    it is what lets ring 1 and every pre-protocol-2 paper fixture keep rendering.
    """
    raw = (manifest.get("provenance") or {}).get("camera_geometry")
    if raw and not raw.get("legacy_aligned"):
        return CameraGeometry.from_greeting(raw)
    if K is None or depth is None:
        return None
    d = np.asarray(depth)
    return CameraGeometry.legacy_aligned(np.asarray(K, float), (d.shape[1], d.shape[0]))


def _characterization_manifest(char_dir: Path, report: dict) -> dict:
    """Adapt a ``characterize-NN/report.json`` into the manifest shape every
    figure builder in this module already reads.

    A characterization measures a ring with NO recipe assumption (see
    ``processing.characterize_ring``), so it is archived without a
    ``manifest.json``, a ``recipe`` or a ``nominal_path.json`` -- what IS on
    disk is the refined fit in ``report['summary']`` and the throwaway coarse
    recipe the pipeline actually ran with, in ``report['coarse']`` (see
    ``_compute_characterization_stages``, which rebuilds that recipe to
    re-run the method figure). Filenames (``color.png``/``depth.npy``/
    ``measured_path.json``) already match the layer defaults ``load_take``
    assumes, so only the manifest-shaped metadata needs synthesizing.
    """
    summary = report.get("summary") or {}
    trial_file = char_dir.parent / "trial.json"
    trial = (json.loads(trial_file.read_text(encoding="utf-8"))
             if trial_file.is_file() else {})
    provenance = dict(report.get("provenance") or {})
    provenance.setdefault("work_frame", report.get("coordinate_frame"))
    return {
        "trial_id": trial.get("trial_id", char_dir.parent.name),
        "layer_index": "characterization", "take": summary.get("index", 1),
        "mode": "CHARACTERIZE", "kind": "characterization",
        "recipe": {"radius_mm": summary.get("radius_mm"),
                   "bead_diameter_mm": summary.get("bead_width_mm")},
        "metrics": {"measured_center_mm": summary.get("center_mm")},
        "geometry": report.get("geometry"),
        "provenance": provenance,
        "color_file": "color.png", "depth_file": "depth.npy",
        "pointcloud_file": None,
        "warnings": report.get("warnings", []),
        "_report": report,      # kept for _compute_characterization_stages
    }


def load_take(layer_dir: Path) -> TakeData:
    layer_dir = Path(layer_dir)
    manifest_file = layer_dir / "manifest.json"
    if manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    else:
        report_file = layer_dir / "report.json"
        if not report_file.is_file():
            raise FileNotFoundError(f"not an archived take: {layer_dir}")
        manifest = _characterization_manifest(
            layer_dir, json.loads(report_file.read_text(encoding="utf-8")))
    cloud_file = layer_dir / (manifest.get("pointcloud_file") or "height-or-pointcloud.npy")
    depth_file = layer_dir / (manifest.get("depth_file") or "depth.npy")
    cloud = np.load(cloud_file) if cloud_file.is_file() else None
    if cloud is not None and (cloud.ndim != 2 or cloud.shape[1] != 3):
        cloud = None                                   # a height map, not a cloud
    transform = (manifest.get("provenance") or {}).get("T_work_camera")
    depth = np.load(depth_file) if depth_file.is_file() else None
    K, dist = _camera_model(manifest, layer_dir)
    return TakeData(
        layer_dir=layer_dir, manifest=manifest,
        nominal=_points(layer_dir / "nominal_path.json"),
        measured=_points(layer_dir / "measured_path.json"),
        cloud=cloud, depth=depth, K=K,
        T_work_camera=None if transform is None else np.asarray(transform, dtype=float),
        geometry=geometry_for_take(manifest, K, depth), dist_coeffs=dist)


# -- shared drawing helpers --------------------------------------------------

def _finish_plan_axes(ax, take: TakeData, *, title: str) -> None:
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"X in {take.manifest.get('provenance', {}).get('work_frame', 'work frame')} (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, color="#e5e7eb", linewidth=.6)
    ax.set_axisbelow(True)


def _scale_bar(ax, length_mm: float = 20.0) -> None:
    """A real scale bar: a figure that gets resized in InDesign still reads true."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + (x1 - x0) * .06
    y = y0 + (y1 - y0) * .07
    ax.plot([x, x + length_mm], [y, y], color="#111827", linewidth=2.5,
            solid_capstyle="butt", zorder=6)
    ax.text(x + length_mm / 2, y + (y1 - y0) * .02, f"{length_mm:g} mm",
            ha="center", va="bottom", fontsize=8, color="#111827", zorder=6)


def _grid_heights(points: np.ndarray, center, half_mm: float, cell_mm: float):
    """Mean z per XY cell -- a height map, with empty cells left as NaN."""
    cx, cy = center
    lo = np.array([cx - half_mm, cy - half_mm])
    size = int(np.ceil(2 * half_mm / cell_mm))
    index = np.floor((points[:, :2] - lo) / cell_mm).astype(int)
    inside = np.all((index >= 0) & (index < size), axis=1)
    index, z = index[inside], points[inside, 2]
    total = np.zeros((size, size))
    count = np.zeros((size, size))
    np.add.at(total, (index[:, 1], index[:, 0]), z)
    np.add.at(count, (index[:, 1], index[:, 0]), 1.0)
    with np.errstate(invalid="ignore"):
        heights = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    extent = (lo[0], lo[0] + size * cell_mm, lo[1], lo[1] + size * cell_mm)
    return heights, extent


def _scene_points(take: TakeData) -> np.ndarray | None:
    """The densest cloud available: the whole re-projected frame, else the archived cloud."""
    if (take.depth is not None and take.geometry is not None
            and take.T_work_camera is not None):
        points, _ = depth_to_work_points(take.depth, take.geometry, take.T_work_camera)
        return points if len(points) else take.cloud
    return take.cloud


def _z_exaggeration(radius: float, z_span: float) -> float:
    """Make a few mm of height readable next to tens of mm of radius, and say so."""
    if z_span <= 0:
        return 1.0
    return max(1.0, round((radius * .5) / z_span, 1))


def _legible_factor(factor: float, z_span: float, scale: float) -> float:
    """Raise a stated exaggeration until the 3-D box is legible -- never the box.

    ``set_box_aspect`` used to be floored at ``MIN_RELIEF_RATIO`` while the
    caption kept quoting the unfloored factor, so the drawing exaggerated Z past
    what it claimed. It bites whenever the window is wide for the ring: a 10 mm
    ring lands in the 60 mm minimum window of ``_work_window`` at ~0.05, i.e.
    2.3x more relief than the caption states (and, when the factor is 1.0, the
    caption states nothing at all).

    Putting the floor on the FACTOR instead keeps one number: it multiplies the
    plotted Z, labels the Z axis and is printed in the caption, and the box then
    carries exactly the data it is given.
    """
    if z_span <= 0 or scale <= 0:
        return factor
    return max(factor, round(MIN_RELIEF_RATIO * scale / z_span, 1))


def _view_extent(cloud, window) -> tuple[float, float]:
    """The X/Y extent a 3-D view will actually be framed to, in mm."""
    if window is not None:
        return float(window[1] - window[0]), float(window[3] - window[2])
    if cloud is not None and len(cloud):
        return float(np.ptp(cloud[:, 0])), float(np.ptp(cloud[:, 1]))
    return 1.0, 1.0


def _true_box(ax, *, dx: float, dy: float, z_lo: float, z_hi: float,
              factor: float = 1.0) -> None:
    """Give a 3-D axes a box and Z limits that carry EXACTLY ``factor``.

    Both have to be set together, and X/Y have to be pinned by the caller:

    * with no box aspect at all, matplotlib's default box is what exaggerates --
      the method figure's oblique panel stated x1.7 and drew x14.9;
    * with a box sized from the raw span but Z left to autoscale, matplotlib's
      ~5% margin a side shrinks the relief BELOW what is stated (the cell's
      take 5 stated x1.4 and drew x1.27);
    * with a FLOOR on the box (the old ``max(..., .12)``) the drawing gains
      exaggeration nobody can read off the figure. The floor belongs on the
      factor -- see ``_legible_factor`` -- where the caption carries it.

    A scene with no relief at all gets a merely drawable box: there is nothing
    to state and nothing to overstate.
    """
    scale = max(dx, dy, 1e-6)
    span = (float(z_hi) - float(z_lo)) * float(factor)
    try:
        if span > 0:
            pad = .05 * span
            low, high = z_lo * factor - pad, z_hi * factor + pad
            ax.set_zlim(low, high)
            ax.set_box_aspect((dx / scale, dy / scale, (high - low) / scale))
        else:
            ax.set_box_aspect((dx / scale, dy / scale, MIN_RELIEF_RATIO))
    except Exception:
        pass


def _note(stages: dict, message: str) -> None:
    """Something a reader of the method figure has to be told, on the figure."""
    stages.setdefault("notes", []).append(message)


def _incomplete(stages: dict, take: "TakeData", exc: BaseException) -> "dict | None":
    """A re-run that failed must never render as a finished method figure.

    The chain is re-run to draw what it actually held; when it raises part-way
    the earlier stages are still real and worth showing, but the figure is then
    evidence of a PARTIAL reconstruction and has to say so. Logged either way --
    a swallowed exception behind a paper figure is the failure mode that matters.
    """
    log.warning("method figure: the chain could not be re-run to the end for %s: %s",
                take.layer_dir, exc, exc_info=True)
    if not any(key in stages for key in STAGE_KEYS):
        return None
    stages["error"] = f"{type(exc).__name__}: {exc}"
    return stages


def _chroma_dist(take: "TakeData", stages: dict) -> "np.ndarray | None":
    """The lens model to re-run this take's chroma gate through -- see
    ``TakeData.chroma_dist``. A protocol-2 take whose archive never recorded the
    coefficients cannot be reproduced exactly; it falls back to the undistorted
    model (what ``service.reprocess_layer`` also does with a missing
    ``dist_coeffs``, so the figure and the reprocess button still agree), and
    says so on the figure rather than passing the guess off as the measurement.
    """
    dist = take.chroma_dist
    if dist is None and take.geometry is not None and not take.geometry.legacy:
        message = ("colour distortion not recorded for this take: the chroma gate "
                   "was re-sampled through an undistorted lens model, so this "
                   "segmentation may differ from the archived measurement")
        _note(stages, message)
        log.warning("method figure: %s: %s", take.layer_dir, message)
    return dist


# -- the four layer figures --------------------------------------------------

def _figure_plan(plt, take: TakeData):
    # Taller than it is wide: the ring is framed to fill the axes, so the legend
    # has to live BELOW them (an in-axes legend covers the measured centreline,
    # which is the evidence the figure exists to show).
    fig, ax = plt.subplots(figsize=(6.0, 7.0))
    if take.cloud is not None and len(take.cloud):
        ax.scatter(take.cloud[:, 0], take.cloud[:, 1], s=3, c=CLOUD, linewidths=0,
                   label=f"deposit surface ({len(take.cloud)} pts)", zorder=2)
    if take.nominal is not None:
        ax.plot(take.nominal[:, 0], take.nominal[:, 1], color=NOMINAL, linewidth=1.6,
                linestyle="--", label="nominal circle", zorder=3)
    truth = expected_ring(take)
    if truth is not None:
        ax.plot(truth[:, 0], truth[:, 1], color=GROUND_TRUTH, linewidth=1.7,
                linestyle="-.", label="ground truth (nominal + introduced offset)",
                zorder=3)
    if take.measured is not None:
        ax.plot(take.measured[:, 0], take.measured[:, 1], color=MEASURED, linewidth=2.0,
                label="extracted centreline", zorder=4)
    metrics = take.manifest.get("metrics") or {}
    found = metrics.get("measured_center_mm")
    if found:
        ax.plot(*found, marker="+", markersize=11, markeredgewidth=1.8, color=MEASURED,
                linestyle="none", label="fitted centre", zorder=5)
    ax.plot(*take.center, marker="+", markersize=11, markeredgewidth=1.8, color=NOMINAL,
            linestyle="none", label="nominal centre", zorder=5)
    _finish_plan_axes(ax, take, title=f"Plan view — {take.label}")
    _scale_bar(ax)
    fig.tight_layout(rect=(0, .155, 1, 1))
    # Far enough below the axes to clear the x-axis label, high enough to leave
    # the caption its own band at the foot of the figure.
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.11), ncol=2, fontsize=8,
              framealpha=.92, borderaxespad=0.)
    fig.text(.5, .015, wrap_caption(take.caption), ha="center", va="bottom",
             fontsize=7.5, color="#4b5563")
    return fig


def _deposit_band(z) -> tuple[float, float]:
    """Colour limits for a raw cloud, clipped to a plausible deposit band.

    A D435i frame carries dropouts hundreds of millimetres below the work plane.
    Left in, they own the colour scale and the ring -- the whole subject --
    flattens to a single colour.
    """
    z = np.asarray(z, dtype=float)
    band = z[(z > WORK_BAND_MM[0]) & (z < WORK_BAND_MM[1])]
    if not band.size:
        band = z
    lo, hi = (float(v) for v in np.percentile(band, [2, 99.5]))
    return lo, max(hi, lo + .1)


def heightmap_data(take: TakeData, *, cell_mm: float = .5) -> dict | None:
    """Gridded height map plus the colour range the relief should be read against.

    The scene is clipped to a plausible deposit band first. A D435i frame carries
    dropouts and edge points hundreds of millimetres below the work plane; left in,
    they own the colour scale and the ring -- the whole subject -- flattens to one
    colour. The band is generous enough that a genuinely tall or sunken deposit
    still shows; only physically impossible returns are cut.
    """
    points = _scene_points(take)
    if points is None or not len(points):
        return None
    tops = [a[:, 2].max() for a in (take.measured, take.cloud)
            if a is not None and len(a)]
    ceiling = (max(tops) if tops else take.radius) + 15.0
    band = points[(points[:, 2] > -15.0) & (points[:, 2] < ceiling)]
    if not len(band):
        return None
    heights, extent = _grid_heights(band, take.center, take.radius * 1.6, cell_mm)
    finite = heights[np.isfinite(heights)]
    if not finite.size:
        return None
    lo, hi = (float(v) for v in np.percentile(finite, [1, 99.5]))
    return {"heights": heights, "extent": extent, "vmin": lo, "vmax": max(hi, lo + .1)}


def _figure_heightmap(plt, take: TakeData):
    data = heightmap_data(take)
    if data is None:
        return None
    heights, extent = data["heights"], data["extent"]
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    cmap = _colormap(CMAP)
    image = ax.imshow(np.ma.masked_invalid(heights), origin="lower", extent=extent,
                      cmap=cmap, vmin=data["vmin"], vmax=data["vmax"],
                      interpolation="nearest")
    if take.nominal is not None:
        ax.plot(take.nominal[:, 0], take.nominal[:, 1], color="w", linewidth=2.4, zorder=3)
        ax.plot(take.nominal[:, 0], take.nominal[:, 1], color=NOMINAL, linewidth=1.3,
                linestyle="--", label="nominal", zorder=4)
    if take.measured is not None:
        ax.plot(take.measured[:, 0], take.measured[:, 1], color="w", linewidth=3.0, zorder=3)
        ax.plot(take.measured[:, 0], take.measured[:, 1], color=MEASURED, linewidth=1.8,
                label="centreline", zorder=4)
    bar = fig.colorbar(image, ax=ax, shrink=.86, pad=.03)
    bar.set_label("height above the work plane (mm)", fontsize=9)
    _finish_plan_axes(ax, take, title=f"Height map — {take.label}")
    ax.grid(False)
    if take.measured is not None or take.nominal is not None:
        ax.legend(loc="upper right", fontsize=8, framealpha=.92)
    fig.text(.5, .015, take.caption, ha="center", fontsize=7.5, color="#4b5563")
    fig.tight_layout(rect=(0, .035, 1, 1))
    return fig


def _view_cloud(take: TakeData) -> tuple[np.ndarray | None, tuple | None]:
    """The densest cloud a 3-D view can honestly draw, windowed to the whole
    ring with margin.

    Prefers the full re-projected frame (what the camera actually saw) over
    the archived post-processed cloud, which is the CREST only -- a top-layer
    selection made after the deposit is found (see ``_deposit_points``); a
    scene view exists to show what was captured, not just the feature the
    pipeline kept. Falls back to the archived cloud when there is no frame to
    re-project (no depth, or a legacy take with no greeting).
    """
    anchor = take.nominal if take.nominal is not None else (
        take.measured if take.measured is not None else take.cloud)
    window = (_work_window(anchor, take.radius)
              if anchor is not None and len(anchor) else None)
    scene = _scene_points(take)                    # frame, else the archived cloud
    cloud = None
    if scene is not None and len(scene):
        band = ((scene[:, 2] > WORK_BAND_MM[0]) & (scene[:, 2] < WORK_BAND_MM[1]))
        near = _within(scene[band], window) if window is not None else scene[band]
        if len(near) > 50:
            cloud = near
    if cloud is None and take.cloud is not None and len(take.cloud):
        # The window/band emptied the frame (or there was none) -- the
        # archived (post-processed) cloud is the fallback, not a blank axes.
        cloud = take.cloud
    return cloud, window


def render_view(take: TakeData, *, azim: float = -58.0, elev: float = 26.0,
                birdseye: bool = False, title: str | None = None, plt=None):
    """One 3-D view of a take's scene, at whatever azimuth/elevation is asked for.

    This is the controllable entry point behind both the ``iso`` figure
    (default oblique) and the ``birdseye`` figure (``birdseye=True``): the
    same rendering, aimed differently, so the two can never show different
    data by accident.

    ``birdseye=True`` looks straight down (elevation pinned to 90°, an
    orthographic projection so it reads as a true plan rather than a
    perspective sketch) and widens the window so the WHOLE ring sits inside
    the axes with margin -- not a crop of it.
    """
    plt = plt or _pyplot()
    cloud, window = _view_cloud(take)
    if cloud is None and take.measured is None and take.nominal is None:
        return None
    if birdseye:
        elev, azim = 90.0, -90.0
        if window is not None:
            cx, cy = (window[0] + window[1]) / 2.0, (window[2] + window[3]) / 2.0
            half = max(window[1] - window[0], window[3] - window[2]) / 2.0 * 1.15
            window = (cx - half, cx + half, cy - half, cy + half)
    zs = [a[:, 2] for a in (cloud, take.measured, take.nominal) if a is not None and len(a)]
    z_lo = min(float(z.min()) for z in zs) if zs else 0.0
    z_hi = max(float(z.max()) for z in zs) if zs else 0.0
    span = z_hi - z_lo
    # The frame the view is drawn in, needed BEFORE the factor: the legibility
    # floor that used to be applied to the box aspect is applied to the factor
    # instead, and the factor has to be settled before anything is plotted with it.
    dx, dy = _view_extent(cloud, window)
    scale = max(dx, dy, 1e-6)
    # A top-down view has no perspective for height to read through, so there
    # is nothing honest to exaggerate -- Z is drawn true and said so.
    factor = (1.0 if birdseye
              else _legible_factor(_z_exaggeration(take.radius, span), span, scale))

    fig = plt.figure(figsize=(7.2, 6.6) if birdseye else (6.6, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    if cloud is not None and len(cloud):
        dots = ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2] * factor, s=2,
                          c=cloud[:, 2], cmap=CMAP, linewidths=0,
                          label=f"scene ({len(cloud)} pts)")
        fig.colorbar(dots, ax=ax, shrink=.7, pad=.09).set_label("Z (mm)", fontsize=8)
    if take.nominal is not None:
        ax.plot(take.nominal[:, 0], take.nominal[:, 1], take.nominal[:, 2] * factor,
                color=NOMINAL, linewidth=1.6, linestyle="--", label="nominal circle")
    truth = expected_ring(take)
    if truth is not None:
        ax.plot(truth[:, 0], truth[:, 1], truth[:, 2] * factor, color=GROUND_TRUTH,
                linewidth=1.7, linestyle="-.", label="ground truth")
    if take.measured is not None:
        ax.plot(take.measured[:, 0], take.measured[:, 1], take.measured[:, 2] * factor,
                color=MEASURED, linewidth=2.4, label="extracted centreline")
    ax.set_xlabel("X (mm)", fontsize=9)
    ax.set_ylabel("Y (mm)", fontsize=9)
    if birdseye:
        # Looking straight down, the Z axis is edge-on: its ticks/label land
        # on top of the title rather than reading as a third dimension.
        # Height is still carried honestly -- by the colour, via the colourbar.
        ax.set_zticks([])
        ax.set_zlabel("")
    else:
        ax.set_zlabel("Z (mm)" if factor == 1.0 else f"Z × {factor:g} (mm)", fontsize=9)
    kind = "Bird's-eye (top-down)" if birdseye else "Oblique view"
    # A 3-D axes' own ``set_title`` centres on the AXES, not the figure -- for
    # a long label that can push the text left of the canvas edge and clip it.
    # ``suptitle`` centres on the whole figure instead.
    fig.suptitle(title or f"{kind} — {take.label}", fontsize=10,
                y=.97 if birdseye else .99)
    ax.view_init(elev=elev, azim=azim)
    if birdseye:
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass
    if window is not None:
        ax.set_xlim(window[0], window[1])
        ax.set_ylim(window[2], window[3])
    if cloud is not None and len(cloud):
        ax.legend(loc="upper left", fontsize=8)
    # The box has to carry the SAME proportions as the data, or the drawing
    # exaggerates by whatever the box happens to be, on top of (or instead of)
    # the stated factor. ``_legible_factor`` has already raised ``factor`` far
    # enough for the box to read, so no floor is needed here.
    _true_box(ax, dx=dx, dy=dy, z_lo=z_lo, z_hi=z_hi, factor=factor)
    caption = take.caption
    if factor != 1.0:
        caption += f" · vertical exaggeration ×{factor:g}"
    if birdseye:
        caption += " · top-down, framed to fit the whole ring with margin"
    fig.text(.5, .015, wrap_caption(caption), ha="center", fontsize=7.5, color="#4b5563")
    fig.tight_layout(rect=(0, .035, 1, .95 if birdseye else 1))
    return fig


def _figure_iso(plt, take: TakeData):
    return render_view(take, azim=-58.0, elev=26.0, birdseye=False, plt=plt)


def _figure_birdseye(plt, take: TakeData):
    return render_view(take, birdseye=True, plt=plt)


def unrolled_profile(measured: np.ndarray, center, radius: float) -> dict:
    """Height and radial deviation against angle, the way ``compare_circle`` measures.

    Deviation is radial distance from the NOMINAL centre minus the nominal
    radius, so these curves are the per-sample series behind the manifest's
    mean/RMS/max -- a reader can check the table against the plot.
    """
    measured = np.asarray(measured, dtype=float).reshape(-1, 3)
    rel = measured[:, :2] - np.asarray(center, dtype=float)
    angle = np.degrees(np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2 * np.pi))
    order = np.argsort(angle)
    deviation = np.linalg.norm(rel, axis=1) - float(radius)
    return {"angle_deg": angle[order], "deviation_mm": deviation[order],
            "z_mm": measured[order, 2],
            "mean_absolute_mm": float(np.abs(deviation).mean()),
            "rms_mm": float(np.sqrt(np.mean(deviation ** 2))),
            "maximum_mm": float(np.abs(deviation).max())}


# Two figures re-run the chain (the method figure and the surfaced view), and a
# gallery load asks for them one request apart. Re-running costs ~2 s and ~25 MB
# of intermediate cloud, so the LAST take is remembered and nothing else: keyed
# on the manifest's mtime, so a reprocessed take is never served from the cache.
_STAGE_CACHE: dict[tuple[str, int], "dict | None"] = {}


def take_stages(take: TakeData) -> "dict | None":
    """Re-run the archived frame through the real chain, keeping every stage.

    Nothing is re-implemented here: this calls ``measure_take`` with a
    collector, so the panels show the arrays the pipeline actually held. Needs
    the scan extra (Open3D); without it the method figure is skipped like any
    other figure that cannot be drawn.
    """
    if (take.depth is None or take.K is None or take.T_work_camera is None
            or take.geometry is None):
        return None
    manifest_file = take.layer_dir / "manifest.json"
    key = (str(take.layer_dir.resolve()),
           manifest_file.stat().st_mtime_ns if manifest_file.is_file() else 0)
    if key in _STAGE_CACHE:
        return _STAGE_CACHE[key]
    stages = (_compute_characterization_stages(take)
              if take.manifest.get("kind") == "characterization"
              else _compute_stages(take))
    _STAGE_CACHE.clear()
    _STAGE_CACHE[key] = stages
    return stages


def reconstruct_take_inputs(take: TakeData, stages: "dict | None" = None) -> "dict | None":
    """The inputs ``measure_take`` needs to reprocess ``take`` exactly as the
    archived pass did: the plan and layer ``plan_for_archived_take`` rebuilds
    from the manifest/trial, the config resolved from manifest-then-trial
    provenance, and the lens model ``_chroma_dist`` chooses for the chroma
    gate.

    Returns ``None`` where the archive lacks what reprocessing needs: no
    ``trial.json`` next to the layer, no ``processing_config`` in either
    provenance, or a ``layer_index`` past the end of the rebuilt plan -- the
    same conditions ``_compute_stages`` used to check inline before this was
    pulled out, so it and the golden tests can never drift apart.

    ``stages`` is optional and, when given, is MUTATED to collect any note
    ``_chroma_dist`` has for the method figure caption. Callers with no
    figure to annotate (the golden tests) can omit it.
    """
    from .processing import plan_for_archived_take

    trial_file = take.layer_dir.parent / "trial.json"
    if not trial_file.is_file():
        return None
    trial = json.loads(trial_file.read_text(encoding="utf-8"))
    config_payload = ((take.manifest.get("provenance") or {}).get("processing_config")
                      or (trial.get("provenance") or {}).get("processing_config"))
    if not config_payload:
        return None
    from ...core.config import ExtrusionConfig
    plan = plan_for_archived_take(take.manifest, trial, nominal_xyz=take.nominal)
    index = int(take.manifest.get("layer_index") or 1)
    if index > len(plan.layers):
        return None
    return {
        "plan": plan,
        "layer": plan.layers[index - 1],
        "config": ExtrusionConfig.model_validate(config_payload),
        "dist": _chroma_dist(take, stages if stages is not None else {}),
    }


def _compute_stages(take: TakeData) -> "dict | None":
    from .processing import measure_take

    stages: dict = {}
    inputs = reconstruct_take_inputs(take, stages)
    if inputs is None:
        return None
    colour = take.layer_dir / (take.manifest.get("color_file") or "color.png")
    image = None
    if colour.is_file():
        import cv2
        image = cv2.imread(str(colour), cv2.IMREAD_COLOR)
    if image is None:
        image = np.zeros((*np.asarray(take.depth).shape[:2], 3), np.uint8)
    try:
        result = measure_take(
            color=image, depth=take.depth, geometry=take.geometry,
            T_work_camera=take.T_work_camera, K=take.K,
            # The lens model the MEASUREMENT gated with, not a second choice --
            # see TakeData.chroma_dist.
            dist=inputs["dist"], plan=inputs["plan"], layer=inputs["layer"],
            config=inputs["config"], stages=stages)
    except Exception as exc:
        # A take that cannot be reconstructed still gets its other figures --
        # but the method figure is then partial, and has to be marked and logged.
        return _incomplete(stages, take, exc)
    stages["result"] = result
    return stages


def _compute_characterization_stages(take: TakeData) -> "dict | None":
    """Re-run a characterization's own (unarchived) throwaway plan.

    ``characterize_ring`` fits a coarse circle from the frame itself, builds a
    one-layer plan from it (radius/height/bead clipped exactly the way it
    does), and re-runs ``measure_take`` on that plan -- but the coarse
    plan is never written to disk, only the numbers it was built from
    (``report['coarse']``, stashed on the manifest by
    ``_characterization_manifest``). Rebuilding it here, the same way, lets
    the method figure show what THIS pass actually saw instead of drawing a
    second, possibly-drifted implementation of the same fit.
    """
    report = take.manifest.get("_report") or {}
    coarse = report.get("coarse")
    config_payload = (take.manifest.get("provenance") or {}).get("processing_config")
    if (not coarse or not config_payload or take.depth is None or take.K is None
            or take.T_work_camera is None or take.geometry is None):
        return None
    trial_file = take.layer_dir.parent / "trial.json"
    if not trial_file.is_file():
        return None
    trial = json.loads(trial_file.read_text(encoding="utf-8"))
    from .models import CylinderRecipe, CylinderSetup
    from .toolpath import generate_cylinder_plan
    from ...core.config import ExtrusionConfig
    trial_setup = trial.get("setup") or {}
    center = coarse.get("center_mm") or [0.0, 0.0]
    try:
        config = ExtrusionConfig.model_validate(config_payload)
        recipe = CylinderRecipe(
            radius_mm=float(np.clip(coarse["radius_mm"], 5.0, 500.0)), layer_count=1,
            layer_height_mm=float(np.clip(coarse["height_mm"], 0.5, 50.0)),
            bead_diameter_mm=float(np.clip(coarse["bead_width_mm"], 0.5, 50.0)),
            robot_speed_mm_s=75.0, extrusion_rate_pct=0.0,
            points_per_circle=config.measured_spline_points)
        setup = CylinderSetup(
            print_tool=trial_setup.get("print_tool") or "unknown",
            work_frame=trial_setup.get("work_frame") or "work frame",
            inspection_tool=trial_setup.get("inspection_tool") or "unknown",
            inspection_auto=True, center_x_mm=float(center[0]), center_y_mm=float(center[1]))
        plan = generate_cylinder_plan(recipe, setup)
    except Exception as exc:
        # No plan means no method figure at all (rather than a partial one), but
        # it is still a failure and must not disappear.
        log.warning("method figure: %s's coarse plan could not be rebuilt: %s",
                    take.layer_dir, exc, exc_info=True)
        return None
    stages: dict = {}
    colour = take.layer_dir / (take.manifest.get("color_file") or "color.png")
    image = None
    if colour.is_file():
        import cv2
        image = cv2.imread(str(colour), cv2.IMREAD_COLOR)
    if image is None:
        image = np.zeros((*np.asarray(take.depth).shape[:2], 3), np.uint8)
    from .processing import measure_take
    try:
        result = measure_take(
            color=image, depth=take.depth, geometry=take.geometry,
            T_work_camera=take.T_work_camera, K=take.K,
            # ``characterize_ring`` gated this take through the CALIBRATED
            # colour model; re-running the figure with a different one samples
            # the gate at different pixels and can segment the ring differently
            # from the number the figure is captioned with -- see
            # TakeData.chroma_dist.
            dist=_chroma_dist(take, stages), plan=plan,
            layer=plan.layers[0], config=config, stages=stages)
    except Exception as exc:
        return _incomplete(stages, take, exc)
    stages["result"] = result
    return stages


def _panel_cloud(ax, cloud, *, colour, size=1.2, label=None):
    if cloud is None or not len(cloud):
        return
    ax.scatter(cloud[:, 0], cloud[:, 1], s=size, c=colour, linewidths=0, label=label)


def _work_window(cloud, radius: float):
    """The neighbourhood worth looking at, in mm.

    A depth frame reaches well past the table -- the first real capture carried
    returns a metre away on either side. Framed on those, the board and the ring
    shrink to a smudge in the middle and the panel shows cell furniture instead
    of the subject. The window is the deposit's own extent, generously padded.
    """
    if cloud is None or not len(cloud):
        return None
    cx, cy = float(np.median(cloud[:, 0])), float(np.median(cloud[:, 1]))
    reach = max(float(radius or 0) * 2.6, 60.0,
                float(np.ptp(cloud[:, 0])), float(np.ptp(cloud[:, 1]))) * .8
    return (cx - reach, cx + reach, cy - reach, cy + reach)


def _within(cloud, window):
    if cloud is None or not len(cloud) or window is None:
        return cloud
    x0, x1, y0, y1 = window
    keep = ((cloud[:, 0] >= x0) & (cloud[:, 0] <= x1)
            & (cloud[:, 1] >= y0) & (cloud[:, 1] <= y1))
    return cloud[keep]


def _figure_pipeline(plt, take: TakeData):
    """How one depth frame becomes a measured centreline, stage by stage.

    This is the method figure: the same six arrays the pipeline held, in the
    order it held them -- captured depth, the same cloud obliquely, the work
    ROI, the deposit cluster, the top surface, and the extracted centreline
    against nominal. Panels 1-3 share one window so the reader can watch points
    being removed rather than re-reading three different scales.

    A re-run that stopped part-way still draws the stages it reached, but is
    banner-marked INCOMPLETE: a partial method figure that reads as a finished
    one is the way a paper ends up illustrating a chain that never ran.
    """
    stages = take_stages(take)
    if not stages:
        return None
    raw = stages.get("backprojected")
    if raw is None or not len(raw):
        return None
    result = stages.get("result")
    radius = take._nominal_circle[1] if take._nominal_circle is not None else take.radius
    roi = stages.get("above_floor")
    if roi is None or not len(roi):
        roi = stages.get("work_roi")
    window = _work_window(roi if roi is not None and len(roi) else raw, radius)
    near = _within(raw, window)
    band = _deposit_band(near[:, 2] if len(near) else raw[:, 2])

    fig = plt.figure(figsize=(11.4, 7.4))
    grid = fig.add_gridspec(2, 3, hspace=.33, wspace=.30,
                            left=.055, right=.975, top=.90, bottom=.075)

    # 1 -- everything the camera saw over the work, from above.
    ax = fig.add_subplot(grid[0, 0])
    dots = ax.scatter(near[:, 0], near[:, 1], s=.5, c=near[:, 2], cmap="viridis",
                      vmin=band[0], vmax=band[1], linewidths=0)
    fig.colorbar(dots, ax=ax, fraction=.046, pad=.03).set_label("Z (mm)", fontsize=8)
    ax.set_title("1 · depth as captured", fontsize=10)
    _square(ax, None, window)

    # 2 -- the same points obliquely: the bead stands proud of the board.
    ax = fig.add_subplot(grid[0, 1], projection="3d")
    tall = near[(near[:, 2] > band[0]) & (near[:, 2] < band[1] + 4.0)]
    thin = tall[:: max(1, len(tall) // 9000)] if len(tall) else tall
    z_lo = float(thin[:, 2].min()) if len(thin) else 0.0
    z_hi = float(thin[:, 2].max()) if len(thin) else 0.0
    panel_dx, panel_dy = _view_extent(thin if len(thin) else None, window)
    # Settled BEFORE anything is plotted with it -- the panel states this number
    # in its title, so it is the number the drawing has to carry. The rule is
    # ``_relief_factor``'s, not ``_z_exaggeration``'s: this panel shows a whole
    # frame standing proud of a board, so the relief is judged against the
    # panel's own footprint (the ring's radius would leave it a flat smear now
    # that the box is honest about it).
    factor = _legible_factor(_relief_factor(panel_dx, panel_dy, z_hi - z_lo),
                             z_hi - z_lo, max(panel_dx, panel_dy, 1e-6))
    if len(thin):
        ax.scatter(thin[:, 0], thin[:, 1], thin[:, 2] * factor, s=.7, c=thin[:, 2],
                   cmap="viridis", vmin=band[0], vmax=band[1], linewidths=0)
    ax.set_title(f"2 · the same points obliquely (Z × {factor:g})", fontsize=10)
    ax.view_init(elev=24, azim=-62)
    if window:
        ax.set_xlim(window[0], window[1])
        ax.set_ylim(window[2], window[3])
    _true_box(ax, dx=panel_dx, dy=panel_dy, z_lo=z_lo, z_hi=z_hi, factor=factor)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_tick_params(labelsize=6)
    ax.set_xlabel("X (mm)", fontsize=7)
    ax.set_ylabel("Y (mm)", fontsize=7)

    # 3 -- the work ROI: height band, radial band, previous layer's top.
    ax = fig.add_subplot(grid[0, 2])
    _panel_cloud(ax, near, colour="#dbe1e8", size=.5)
    _panel_cloud(ax, roi, colour=CLOUD, size=1.2)
    ax.set_title("3 · kept by the work ROI", fontsize=10)
    _square(ax, None, window)

    # 4 -- the deposit itself: largest cluster, then trimmed to the fitted ring.
    ax = fig.add_subplot(grid[1, 0])
    cluster, trimmed = stages.get("deposit_cluster"), stages.get("radial_trimmed")
    _panel_cloud(ax, cluster, colour="#e0857b", size=1.4, label="trimmed away")
    _panel_cloud(ax, trimmed, colour=CLOUD, size=1.4, label="kept")
    if cluster is not None and trimmed is not None and len(cluster) > len(trimmed):
        ax.legend(loc="upper right", fontsize=7, markerscale=4)
    ax.set_title("4 · deposit cluster, radially trimmed", fontsize=10)
    _square(ax, cluster)

    # 5 -- the crest only, and the centreline thinned from it.
    ax = fig.add_subplot(grid[1, 1])
    top = stages.get("top_surface")
    _panel_cloud(ax, trimmed, colour="#dbe1e8", size=1.0)
    _panel_cloud(ax, top, colour=MEASURED, size=1.4)
    if result is not None and result.measured_xyz is not None:
        ax.plot(result.measured_xyz[:, 0], result.measured_xyz[:, 1],
                color="#0b1017", linewidth=1.1, alpha=.7)
    ax.set_title("5 · top surface, thinned to a centreline", fontsize=10)
    _square(ax, top if top is not None and len(top) else trimmed)

    # 6 -- what the paper measures.
    ax = fig.add_subplot(grid[1, 2])
    error = stages.get("error")
    measured = result.measured_xyz if result is not None else take.measured
    if take.nominal is not None:
        ax.plot(take.nominal[:, 0], take.nominal[:, 1], color=NOMINAL, linewidth=1.3,
                linestyle="--", label="nominal")
    if measured is not None and len(measured):
        # With no result the panel is the ARCHIVED path, not this re-run's
        # output; label it as such so the last panel cannot be read as evidence
        # the chain reached it.
        ax.plot(measured[:, 0], measured[:, 1], color=MEASURED, linewidth=2.0,
                label="extracted" if result is not None else "archived path")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title("6 · extracted vs nominal" if result is not None
                 else "6 · archived path (not re-run)", fontsize=10)
    _square(ax, measured if measured is not None else take.nominal)

    title = f"From one RGB-D frame to a measured centreline — {take.label}"
    fig.suptitle(f"{title} — INCOMPLETE" if error else title, fontsize=12)
    if error:
        # A chain failure carries its whole diagnostic payload; unwrapped it runs
        # off both edges of the figure and reads as nothing at all.
        detail = " ".join(str(error).split())
        if len(detail) > 200:
            detail = detail[:197] + "..."
        banner = "\n".join(textwrap.wrap(
            f"INCOMPLETE: the chain could not be re-run to the end ({detail}). "
            "Stages after the failure are missing.", width=150)) or detail
        fig.text(.5, .965, banner, ha="center", va="top", fontsize=8,
                 color=MEASURED, weight="bold", linespacing=1.25)
    caption = take.caption
    notes = stages.get("notes") or []
    if notes:
        caption = " · ".join([caption, *notes]) if caption else " · ".join(notes)
    fig.text(.5, .012, wrap_caption(caption, width=150), ha="center", va="bottom",
             fontsize=7.5, color="#4b5563")
    return fig


def _square(ax, cloud, window=None):
    """Equal aspect, mm axes, and a frame that fits what is being shown."""
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)
    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("Y (mm)", fontsize=8)
    if window is not None:
        ax.set_xlim(window[0], window[1])
        ax.set_ylim(window[2], window[3])
        return
    if cloud is None or not len(cloud):
        return
    pad = max(4.0, .06 * float(max(np.ptp(cloud[:, 0]), np.ptp(cloud[:, 1]))))
    ax.set_xlim(cloud[:, 0].min() - pad, cloud[:, 0].max() + pad)
    ax.set_ylim(cloud[:, 1].min() - pad, cloud[:, 1].max() + pad)


# -- the surfaced (meshed) view ----------------------------------------------
#
# The legacy scan macro built a Poisson mesh and handed it to
# ``o3d.visualization.draw_geometries``: an interactive window that had to be
# rotated and screenshotted by hand, and wrote no file. Everything else in this
# module renders from the archive with no display, and the surfaced view is no
# exception -- it is a 2.5-D mesh, because a single top-down RGB-D frame
# measures a height field and nothing else. Surfacing it as a closed solid
# would invent the underside the camera never saw.

@dataclass
class MeshSurface:
    """A triangle mesh over a top-down cloud: vertices in mm, triangles as indices."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    triangles: np.ndarray
    cell_mm: float

    def triangulation(self):
        from matplotlib.tri import Triangulation
        return Triangulation(self.x, self.y, self.triangles)


def _cell_counts(points, window, cell: float):
    """Points per grid cell, plus the grid shape they were binned into."""
    x0, x1, y0, y1 = window
    nx = max(int(np.ceil((x1 - x0) / cell)), 2)
    ny = max(int(np.ceil((y1 - y0) / cell)), 2)
    ix = np.clip(((points[:, 0] - x0) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((points[:, 1] - y0) / cell).astype(int), 0, ny - 1)
    flat = iy * nx + ix
    return np.bincount(flat, minlength=nx * ny), nx, ny


def _auto_cell(points, window, *, cells_across: int, min_cell_mm: float,
               per_cell: float = 2.5) -> float:
    """Coarsen until the cells actually hold points.

    A pitch finer than the cloud's own spacing produces a grid of isolated
    cells with no shared edges, and a mesh of isolated cells is a scatter of
    specks (26 triangles from the cell's 1517-point bead). The rule is the
    cloud's, not the figure's: grow the cell until it averages a few returns.
    """
    x0, x1, y0, y1 = window
    span = max(x1 - x0, y1 - y0)
    ceiling = max(span / 12.0, min_cell_mm)
    cell = min(max(min_cell_mm, span / max(int(cells_across), 8)), ceiling)
    for _ in range(8):
        counts, _, _ = _cell_counts(points, window, cell)
        filled = int((counts > 0).sum())
        if not filled or len(points) / filled >= per_cell or cell >= ceiling:
            break
        cell = min(cell * 1.35, ceiling)
    return cell


def surface_mesh(points, *, window=None, cells_across: int = 120,
                 min_cell_mm: float = .35, max_step_mm: float = 25.0
                 ) -> "MeshSurface | None":
    """Turn a cloud into a surface: mean height per cell, triangulated in place.

    Gaps stay gaps. A cell the camera returned nothing for has no vertex, so no
    triangle can cover it -- the hole in a ring stays a hole, which a convex
    triangulation would roof over with long thin faces. ``max_step_mm`` drops
    the triangles that would otherwise bridge a dropout cliff (the D435i leaves
    returns hundreds of mm below the plane) and draw a wall that was never there.

    Gridding rather than Delaunay is deliberate: it is deterministic, so the
    same archived take re-renders to the same mesh, and the cell pitch bounds
    the triangle count no matter how dense the frame is.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return None
    if window is None:
        window = (float(points[:, 0].min()), float(points[:, 0].max()),
                  float(points[:, 1].min()), float(points[:, 1].max()))
    window = tuple(float(v) for v in window)
    x0, x1, y0, y1 = window
    if max(x1 - x0, y1 - y0) <= 0:
        return None
    inside = ((points[:, 0] >= x0) & (points[:, 0] <= x1)
              & (points[:, 1] >= y0) & (points[:, 1] <= y1))
    points = points[inside]
    if len(points) < 3:
        return None

    cell = _auto_cell(points, window, cells_across=cells_across,
                      min_cell_mm=min_cell_mm)
    counts, nx, ny = _cell_counts(points, window, cell)
    ix = np.clip(((points[:, 0] - x0) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((points[:, 1] - y0) / cell).astype(int), 0, ny - 1)
    totals = np.bincount(iy * nx + ix, weights=points[:, 2], minlength=nx * ny)
    filled = counts > 0
    if filled.sum() < 3:
        return None
    height = np.where(filled, totals / np.maximum(counts, 1), np.nan)

    # Vertex ids for the cells that hold a measurement; -1 for the empty ones.
    ids = np.full(height.size, -1, dtype=np.int64)
    ids[filled] = np.arange(int(filled.sum()))
    ids = ids.reshape(ny, nx)
    a, b = ids[:-1, :-1], ids[:-1, 1:]
    c, d = ids[1:, :-1], ids[1:, 1:]
    quads = np.concatenate((np.stack((a, b, c), axis=-1).reshape(-1, 3),
                            np.stack((b, d, c), axis=-1).reshape(-1, 3)))
    triangles = quads[(quads >= 0).all(axis=1)]
    if not len(triangles):
        return None

    xs = x0 + (np.arange(nx) + .5) * cell
    ys = y0 + (np.arange(ny) + .5) * cell
    grid_x, grid_y = np.meshgrid(xs, ys)
    vx, vy = grid_x.ravel()[filled], grid_y.ravel()[filled]
    vz = height[filled]

    triangles = triangles[np.ptp(vz[triangles], axis=1) <= float(max_step_mm)]
    if not len(triangles):
        return None
    used, flat_index = np.unique(triangles, return_inverse=True)
    return MeshSurface(x=vx[used], y=vy[used], z=vz[used],
                       triangles=flat_index.reshape(-1, 3), cell_mm=cell)


@dataclass
class MeshPanel:
    """One surface worth drawing, and the neighbourhood to draw it in."""

    key: str
    title: str
    points: np.ndarray
    window: tuple | None = None
    cells_across: int = 120


def _deposit_points(take: TakeData) -> "np.ndarray | None":
    """The bead itself, as densely as the archive can show it.

    The archived cloud is the CREST the centreline was thinned from -- 578
    points on the cell's first ring, a bead's worth of dots with its flanks
    already discarded. Meshing that draws specks. The chain's own deposit
    cluster is the same bead with its sides still on it (1517 points on that
    take), which is what closes into a surface with a width and a height.
    """
    stages = take_stages(take) or {}
    for key in ("radial_trimmed", "deposit_cluster"):
        points = stages.get(key)
        if points is not None and len(points) > 200:
            return np.asarray(points, dtype=float)
    return take.cloud if take.cloud is not None and len(take.cloud) > 50 else None


def mesh_panels(take: TakeData) -> list[MeshPanel]:
    """The surfaces this take can honestly show: the scene, and the deposit.

    The scene panel needs a re-projectable frame. Without one the archived cloud
    IS the deposit, so a second panel labelled 'scene' would show the same points
    twice and claim a view that was never captured; the figure then draws the
    deposit alone.
    """
    deposit = _deposit_points(take)
    panels: list[MeshPanel] = []
    if take.depth is not None and take.K is not None and take.T_work_camera is not None:
        frame = _scene_points(take)
        if frame is not None and frame is take.cloud:
            frame = None      # the re-projection came back empty; this is its fallback
        # The window is anchored on something the RECIPE knows about -- the
        # deposit, else the ring that was commanded, else (a characterization
        # has no commanded ring) the centreline it measured. Never on the
        # frame itself: a raw frame's extent is the room, and the failed take
        # 20260828-124136 sized a 32 m panel from one (32 triangles at a
        # 222 mm pitch).
        anchor = deposit if deposit is not None else (
            take.nominal if take.nominal is not None else take.measured)
        if frame is not None and len(frame) and anchor is not None and len(anchor):
            window = _work_window(anchor, take.radius)
            near = _within(frame, window)
            # The work band, not the room. A frame with nothing in it never had
            # the work surface in view, and gets no panel at all.
            near = near[(near[:, 2] > WORK_BAND_MM[0]) & (near[:, 2] < WORK_BAND_MM[1])]
            if len(near) > 50:
                # A dropout 400 mm below the plane is not part of the surface,
                # it is the absence of one.
                lo, hi = _deposit_band(near[:, 2])
                near = near[(near[:, 2] > lo - 5.0) & (near[:, 2] < hi + 15.0)]
            if len(near) > 50:
                panels.append(MeshPanel("scene", "work surface as captured", near,
                                        window, 120))
    if deposit is not None:
        panels.append(MeshPanel("deposit", "deposit only", deposit, None, 110))
    return panels


def _relief_factor(dx: float, dy: float, dz: float, target: float = .30) -> float:
    """Vertical exaggeration that makes millimetres of bead read over a table.

    ``_z_exaggeration`` sets the scale from the ring's radius, which is right
    for a centreline drawn in open space; a surface is judged against its own
    footprint, so this one asks for a relief about a third as tall as the panel
    is wide, whatever the frame happens to contain.
    """
    if dz <= 0:
        return 1.0
    return max(1.0, round(target * max(dx, dy) / dz, 1))


def _draw_mesh_pair(fig, grid, row: int, panel: MeshPanel, mesh: MeshSurface) -> str:
    """One surface as two panels: straight down, then rotated into 3-D."""
    tri = mesh.triangulation()
    lo, hi = _deposit_band(mesh.z)
    # The wireframe is what separates a mesh from a heat map, so it is drawn
    # wherever a cell is still a few pixels wide at 300 dpi. On the shaded 3-D
    # surface the same edges read far heavier, and past a few thousand triangles
    # they close over the surface entirely.
    edged, edged_3d = len(mesh.triangles) <= 45000, len(mesh.triangles) <= 8000

    flat = fig.add_subplot(grid[row, 0])
    shaded = flat.tripcolor(tri, mesh.z, shading="gouraud", cmap=CMAP,
                            vmin=lo, vmax=hi)
    if edged:
        flat.triplot(tri, color="#ffffff", linewidth=.12, alpha=.28)
    fig.colorbar(shaded, ax=flat, fraction=.046, pad=.03).set_label("Z (mm)", fontsize=8)
    flat.set_title(f"{panel.title} — from above", fontsize=10)
    _square(flat, panel.points, panel.window)

    window = panel.window or (float(mesh.x.min()), float(mesh.x.max()),
                              float(mesh.y.min()), float(mesh.y.max()))
    dx, dy = window[1] - window[0], window[3] - window[2]
    dz = float(mesh.z.max() - mesh.z.min())
    scale = max(dx, dy, 1e-6)
    factor = _legible_factor(_relief_factor(dx, dy, dz), dz, scale)
    turned = fig.add_subplot(grid[row, 1], projection="3d")
    surface = turned.plot_trisurf(tri, mesh.z * factor, cmap=CMAP, antialiased=False,
                                  linewidth=.08 if edged_3d else 0,
                                  edgecolor="#33415555" if edged_3d else "none",
                                  shade=True)
    surface.set_clim(lo * factor, hi * factor)
    turned.set_title(f"{panel.title} — rotated (Z × {factor:g})", fontsize=10)
    turned.view_init(elev=26, azim=-62)
    for axis in (turned.xaxis, turned.yaxis, turned.zaxis):
        axis.set_tick_params(labelsize=6)
    turned.set_xlabel("X (mm)", fontsize=7)
    turned.set_ylabel("Y (mm)", fontsize=7)
    turned.set_zlabel(f"Z × {factor:g} (mm)", fontsize=7)
    turned.set_xlim(window[0], window[1])
    turned.set_ylim(window[2], window[3])
    # The box has to carry the SAME proportions as the data, or the drawing
    # exaggerates by whatever the box happens to be (a flat (1, 1, .55) box
    # stretched this bead ~6x while the axis claimed x2). X and Y keep the ring
    # round; Z is the stated factor and nothing more -- the legibility floor
    # lives in ``_legible_factor``, which raises the factor the title and the Z
    # axis both quote rather than the box alone.
    _true_box(turned, dx=dx, dy=dy, z_lo=float(mesh.z.min()), z_hi=float(mesh.z.max()),
              factor=factor)
    return f"{panel.title}: {len(mesh.triangles)} triangles at {mesh.cell_mm:.2f} mm"


def _figure_mesh(plt, take: TakeData):
    """The captured frame as a SURFACE — straight down, and rotated into 3-D.

    The other figures draw points and lines; this one closes them into a mesh,
    which is what makes a bead read as a bead rather than a smear of dots. Both
    rows are the same mesh seen twice, so a reader can carry a feature from the
    plan view into the oblique one.
    """
    drawn = [(panel, surface_mesh(panel.points, window=panel.window,
                                  cells_across=panel.cells_across))
             for panel in mesh_panels(take)]
    drawn = [(panel, mesh) for panel, mesh in drawn if mesh is not None]
    if not drawn:
        return None
    rows = len(drawn)
    fig = plt.figure(figsize=(10.4, 4.7 * rows))
    grid = fig.add_gridspec(rows, 2, hspace=.30, wspace=.22,
                            left=.06, right=.97,
                            top=.90 if rows > 1 else .84,
                            bottom=.11 if rows > 1 else .14)
    notes = [_draw_mesh_pair(fig, grid, row, panel, mesh)
             for row, (panel, mesh) in enumerate(drawn)]
    fig.suptitle(f"Surfaced view — {take.label}", fontsize=12)
    fig.text(.5, .055 if rows > 1 else .07,
             "Mesh built from the measured points; unmeasured cells are left open. "
             + " · ".join(notes), ha="center", fontsize=7.5, color="#4b5563")
    # This figure is 10.4 in wide, not the 6 in CAPTION_WRAP was tuned for.
    fig.text(.5, .015, wrap_caption(take.caption, width=150), ha="center", va="bottom",
             fontsize=7.5, color="#4b5563")
    return fig


def _tube(centreline: np.ndarray, bead_mm: float, *, sides: int = 20):
    """A pipe of diameter ``bead_mm`` swept along a closed centreline.

    The toolpath is a curve, but what is deposited is a bead with a width and a
    height; drawing the curve alone hides exactly the quantity the experiment
    measures. Returns X, Y, Z surface arrays for plot_surface.
    """
    path = np.asarray(centreline, dtype=float).reshape(-1, 3)
    if len(path) < 3:
        return None
    if not np.allclose(path[0], path[-1]):
        path = np.vstack([path, path[:1]])
    tangent = np.gradient(path, axis=0)
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.where(norms == 0, 1.0, norms)
    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(tangent, up)
    side_norm = np.linalg.norm(side, axis=1, keepdims=True)
    side = side / np.where(side_norm == 0, 1.0, side_norm)
    normal = np.cross(side, tangent)
    angles = np.linspace(0, 2 * np.pi, sides)
    radius = float(bead_mm) / 2.0
    offsets = (np.cos(angles)[None, :, None] * side[:, None, :]
               + np.sin(angles)[None, :, None] * normal[:, None, :])
    surface = path[:, None, :] + radius * offsets
    return surface[:, :, 0], surface[:, :, 1], surface[:, :, 2]


def measured_bead_mm(take: TakeData) -> "float | None":
    """The bead footprint this take actually measured, if it measured one.

    The commanded bead comes from the recipe; the deposited bead is a
    measurement (10.8 mm against a commanded 12.8 mm on the first real
    capture). Drawing the outcome at the commanded width would show a
    comparison that was never made.
    """
    geometry = take.manifest.get("geometry") or {}
    width = geometry.get("bead_width_mean_mm")
    try:
        width = float(width)
    except (TypeError, ValueError):
        return None
    return width if width > 0 else None


def _ribbon(centreline: np.ndarray, width_mm: float):
    """A flat band of ``width_mm`` laid along a centreline, in the XY plane.

    The measured bead width is a RADIAL FOOTPRINT -- the extent of the deposit
    cloud per angular bin -- not a cross-section diameter. Sweeping it as a pipe
    would draw a vertical extent that was never measured (and, on a bead whose
    crest wanders by 8 mm, tangles into spikes). A ribbon at the measured height
    is exactly what the measurement says.
    """
    path = np.asarray(centreline, dtype=float).reshape(-1, 3)
    if len(path) < 3:
        return None
    if not np.allclose(path[0], path[-1]):
        path = np.vstack([path, path[:1]])
    tangent = np.gradient(path[:, :2], axis=0)
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.where(norms == 0, 1.0, norms)
    side = np.column_stack((-tangent[:, 1], tangent[:, 0])) * (float(width_mm) / 2.0)
    inner, outer = path[:, :2] - side, path[:, :2] + side
    x = np.column_stack((inner[:, 0], outer[:, 0]))
    y = np.column_stack((inner[:, 1], outer[:, 1]))
    z = np.column_stack((path[:, 2], path[:, 2]))
    return x, y, z


def _figure_tube(plt, takes: list[TakeData], trial_id: str):
    """The stack as it is actually deposited: a bead with thickness, per layer.

    Nominal is drawn as the pipe the recipe asks for -- at each layer's own
    height, so the stack climbs by the layer height -- and the measured
    centreline runs through it. Where the measured line leaves the pipe, the
    deposit is off by more than half a bead.
    """
    drawable = [t for t in takes if t.measured is not None and len(t.measured)]
    if not drawable:
        return None
    bead = 0.0
    for take in drawable:
        bead = max(bead, float((take.manifest.get("recipe") or {}).get("bead_diameter_mm") or 0))
    if bead <= 0:
        bead = 8.0
    fig = plt.figure(figsize=(7.4, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    deposited: list[float] = []
    for take in drawable:
        if take.nominal is not None and len(take.nominal) >= 3:
            pipe = _tube(take.nominal, bead)
            if pipe is not None:
                ax.plot_surface(*pipe, color=NOMINAL, alpha=.16, linewidth=0,
                                shade=True, rstride=2, cstride=1)
        # The deposited bead at the width it was measured at, not the commanded
        # one: the two footprints are the comparison.
        width = measured_bead_mm(take)
        if width:
            deposited.append(width)
            laid = _ribbon(take.measured, width)
            if laid is not None:
                ax.plot_surface(*laid, color=MEASURED, alpha=.55, linewidth=0,
                                shade=False, rstride=1, cstride=1)
        ax.plot(take.measured[:, 0], take.measured[:, 1], take.measured[:, 2],
                color=MEASURED, linewidth=1.6)
    ax.plot([], [], color=NOMINAL, linewidth=6, alpha=.35,
            label=f"commanded bead (Ø {bead:g} mm)")
    ax.plot([], [], color=MEASURED, linewidth=6, alpha=.55,
            label=("measured footprint (%.1f mm wide)" % (sum(deposited) / len(deposited))
                   if deposited else "measured centreline"))
    ax.set_xlabel("X (mm)", fontsize=9)
    ax.set_ylabel("Y (mm)", fontsize=9)
    ax.set_zlabel("Z (mm)", fontsize=9)
    ax.set_title(f"Commanded bead against what was measured — {trial_id}", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=18, azim=-62)
    # The caption says TRUE SCALE, so the box has to be true: a flat (1, 1, .55)
    # box over autoscaled axes drew this ring's few mm of height ~50x its width
    # ratio while the caption promised none. X and Y also keep the ring round.
    paths = [a for take in drawable for a in (take.measured, take.nominal)
             if a is not None and len(a)]
    if paths:
        points = np.vstack(paths)
        edge = bead / 2.0                  # the pipe/ribbon reaches half a bead out
        x_lo, x_hi = float(points[:, 0].min()) - edge, float(points[:, 0].max()) + edge
        y_lo, y_hi = float(points[:, 1].min()) - edge, float(points[:, 1].max()) + edge
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        _true_box(ax, dx=x_hi - x_lo, dy=y_hi - y_lo,
                  z_lo=float(points[:, 2].min()) - edge,
                  z_hi=float(points[:, 2].max()) + edge)
    caption = "True scale. Each layer sits at its own height."
    if deposited:
        caption += (" Commanded bead Ø %.1f mm; measured footprint %.1f mm wide."
                    % (bead, sum(deposited) / len(deposited)))
    fig.text(.5, .02, wrap_caption(caption), ha="center", va="bottom",
             fontsize=7.5, color="#4b5563")
    fig.tight_layout(rect=(0, .035, 1, 1))
    return fig


def _figure_profile(plt, take: TakeData):
    if take.measured is None or len(take.measured) < 3:
        return None
    profile = unrolled_profile(take.measured, take.center, take.radius)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    top.plot(profile["angle_deg"], profile["z_mm"], color=ACCENT, linewidth=1.8)
    geometry = take.manifest.get("geometry") or {}
    if geometry.get("top_z_mean_mm") is not None:
        top.axhline(geometry["top_z_mean_mm"], color="#6b7280", linewidth=.9,
                    linestyle=":", label=f"mean {geometry['top_z_mean_mm']:.2f} mm")
        top.legend(loc="upper right", fontsize=8)
    reference = geometry.get("height_reference", "build_plane").replace("_", " ")
    top.set_ylabel(f"height (mm)\nover {reference}", fontsize=9)
    top.set_title(f"Unrolled profile — {take.label}", fontsize=10)
    top.grid(True, color="#e5e7eb", linewidth=.6)
    top.set_axisbelow(True)

    bottom.axhline(0, color=NOMINAL, linewidth=1.2, linestyle="--", label="nominal")
    bottom.plot(profile["angle_deg"], profile["deviation_mm"], color=MEASURED, linewidth=1.8,
                label="radial deviation")
    for value, style, name in ((profile["rms_mm"], ":", "RMS"),
                               (profile["maximum_mm"], "-.", "max")):
        bottom.axhline(value, color="#6b7280", linewidth=.9, linestyle=style,
                       label=f"{name} {value:.2f} mm")
        bottom.axhline(-value, color="#6b7280", linewidth=.9, linestyle=style)
    bottom.set_xlabel("angle about the nominal centre (deg)", fontsize=9)
    bottom.set_ylabel("radial deviation (mm)", fontsize=9)
    bottom.set_xlim(0, 360)
    bottom.set_xticks(range(0, 361, 45))
    bottom.grid(True, color="#e5e7eb", linewidth=.6)
    bottom.set_axisbelow(True)
    bottom.legend(loc="upper right", fontsize=8, ncol=2)
    fig.text(.5, .015, take.caption, ha="center", fontsize=7.5, color="#4b5563")
    fig.tight_layout(rect=(0, .035, 1, 1))
    return fig


_LAYER_BUILDERS = {"plan": _figure_plan, "heightmap": _figure_heightmap,
                   "mesh": _figure_mesh, "iso": _figure_iso,
                   "birdseye": _figure_birdseye,
                   "profile": _figure_profile, "pipeline": _figure_pipeline}


# -- the trial figure --------------------------------------------------------

def latest_takes(trial_dir: Path) -> list[TakeData]:
    """One TakeData per layer: the newest take of each, in layer order."""
    newest: dict[int, tuple[int, TakeData]] = {}
    for manifest_file in sorted(Path(trial_dir).glob("layer-*/manifest.json")):
        try:
            take = load_take(manifest_file.parent)
        except Exception:
            continue
        index = int(take.manifest.get("layer_index") or 0)
        number = int(take.manifest.get("take") or 1)
        if index not in newest or number >= newest[index][0]:
            newest[index] = (number, take)
    return [take for _, (_, take) in sorted(newest.items())]


def _figure_stack(plt, takes: list[TakeData], trial_id: str):
    if not takes:
        return None
    fig = plt.figure(figsize=(11.0, 5.4))
    flat = fig.add_subplot(1, 2, 1)
    oblique = fig.add_subplot(1, 2, 2, projection="3d")
    colours = plt.get_cmap("plasma")(np.linspace(.1, .8, max(len(takes), 1)))
    drawn = [t.measured for t in takes if t.measured is not None and len(t.measured)]
    zs = [a[:, 2] for a in drawn]
    span = float(np.ptp(np.concatenate(zs))) if zs else 0.0
    radius = takes[0].radius
    # The oblique panel quotes this factor in its title AND its Z label, so it
    # is settled here -- before anything is plotted with it -- against the frame
    # the panel will actually be drawn in.
    if drawn:
        points = np.vstack(drawn)
        pad = .05 * max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0)
        box = (float(points[:, 0].min()) - pad, float(points[:, 0].max()) + pad,
               float(points[:, 1].min()) - pad, float(points[:, 1].max()) + pad)
        z_lo, z_hi = float(points[:, 2].min()), float(points[:, 2].max())
    else:
        box, z_lo, z_hi = (0.0, 1.0, 0.0, 1.0), 0.0, 0.0
    factor = _legible_factor(_z_exaggeration(radius, span), span,
                             max(box[1] - box[0], box[3] - box[2], 1e-6))
    labelled_truth = False
    for colour, take in zip(colours, takes):
        index = take.manifest.get("layer_index")
        if take.nominal is not None:
            flat.plot(take.nominal[:, 0], take.nominal[:, 1], color=NOMINAL, linewidth=1.0,
                      linestyle="--", zorder=2)
        truth = expected_ring(take)
        if truth is not None:
            # Labelled once: one legend entry, however many layers were displaced.
            flat.plot(truth[:, 0], truth[:, 1], color=GROUND_TRUTH, linewidth=1.2,
                      linestyle="-.", zorder=2,
                      label=None if labelled_truth
                      else "ground truth (nominal + introduced offset)")
            labelled_truth = True
        if take.measured is None:
            continue
        flat.plot(take.measured[:, 0], take.measured[:, 1], color=colour, linewidth=1.9,
                  label=f"layer {index}", zorder=3)
        oblique.plot(take.measured[:, 0], take.measured[:, 1], take.measured[:, 2] * factor,
                     color=colour, linewidth=1.9, label=f"layer {index}")
    flat.set_aspect("equal", adjustable="datalim")
    flat.set_xlabel("X (mm)")
    flat.set_ylabel("Y (mm)")
    flat.set_title("Measured centrelines (dashed = nominal"
                   + ("; dash-dot = ground truth)" if labelled_truth else ")"),
                   fontsize=10)
    flat.grid(True, color="#e5e7eb", linewidth=.6)
    flat.set_axisbelow(True)
    _scale_bar(flat)
    flat.legend(loc="upper right", fontsize=8)
    oblique.set_xlabel("X (mm)", fontsize=9)
    oblique.set_ylabel("Y (mm)", fontsize=9)
    oblique.set_zlabel(f"Z × {factor:g} (mm)", fontsize=9)
    oblique.set_title(f"Stack, vertical exaggeration ×{factor:g}", fontsize=10)
    oblique.view_init(elev=24, azim=-58)
    oblique.set_xlim(box[0], box[1])
    oblique.set_ylim(box[2], box[3])
    _true_box(oblique, dx=box[1] - box[0], dy=box[3] - box[2],
              z_lo=z_lo, z_hi=z_hi, factor=factor)
    fig.suptitle(f"Ring stack — {trial_id}", fontsize=11)
    fig.tight_layout(rect=(0, .02, 1, .97))
    return fig


# -- entry points ------------------------------------------------------------

def _save(fig, directory: Path, stem: str, formats, dpi: int) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in formats:
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, facecolor="white",
                    metadata=None if suffix == "png" else {"Creator": "Tasni"})
        written.append(path)
    return written


def render_layer_figures(layer_dir, *, formats=FORMATS, dpi: int = DPI,
                         only: str | None = None) -> list[Path]:
    """Render (and overwrite) this take's figures. Returns what was written."""
    plt = _pyplot()
    take = load_take(Path(layer_dir))
    out = Path(layer_dir) / "figures"
    written: list[Path] = []
    for stem in (LAYER_FIGURES if only is None else (only,)):
        builder = _LAYER_BUILDERS[stem]
        fig = builder(plt, take)
        if fig is None:
            continue
        try:
            written.extend(_save(fig, out, stem, formats, dpi))
        finally:
            plt.close(fig)
    return written


def render_trial_figures(trial_dir, *, formats=FORMATS, dpi: int = DPI) -> list[Path]:
    plt = _pyplot()
    trial_dir = Path(trial_dir)
    takes = latest_takes(trial_dir)
    written: list[Path] = []
    for stem, builder in (("stack", _figure_stack), ("tube", _figure_tube)):
        fig = builder(plt, takes, trial_dir.name)
        if fig is None:
            continue
        try:
            written.extend(_save(fig, trial_dir / "figures", stem, formats, dpi))
        finally:
            plt.close(fig)
    return written


def ensure_figure(directory, filename: str) -> Path:
    """Serve-time helper: render the figure if it is missing, then return its path.

    Lets the API expose figures for takes archived before this module existed
    (and for live-print layers, whose job does not render eagerly) without ever
    re-running the robot.
    """
    directory = Path(directory)
    stem, _, suffix = filename.rpartition(".")
    if suffix not in FORMATS or not stem:
        raise ValueError(f"unsupported figure: {filename!r}")
    path = directory / "figures" / filename
    if path.is_file():
        return path
    if stem in LAYER_FIGURES:
        render_layer_figures(directory, formats=(suffix,), only=stem)
    elif stem in TRIAL_FIGURES:
        render_trial_figures(directory, formats=(suffix,))
    else:
        raise ValueError(f"unknown figure: {stem!r}")
    if not path.is_file():
        raise FileNotFoundError(f"{stem} cannot be drawn from {directory}")
    return path
