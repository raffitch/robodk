"""Publication and in-app figures rendered from an archived take.

Everything here reads ONLY the archive (``manifest.json`` + the arrays written
next to it), so a figure can be produced long after the cell run, on a machine
with no robot and no camera, and re-produced identically. Nothing in this module
touches the robot, RoboDK or the job runner.

Four figures per take:

``plan``       top view: deposit cloud, extracted centreline, nominal circle
``heightmap``  bird's-eye height map of the re-projected depth frame, z colourbar
``iso``        oblique 3-D view of cloud + centreline (z exaggerated, and said so)
``profile``    unrolled: height z(theta) and radial deviation dr(theta) over 360 deg

and one per trial, ``stack``: every layer's latest take, plan + oblique.

Matplotlib is imported lazily behind the Agg backend: importing this module must
not require a display, and ``tasni`` must still import when matplotlib is absent.
"""
from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .processing import depth_to_work_points

LAYER_FIGURES = ("plan", "heightmap", "iso", "profile", "pipeline")
TRIAL_FIGURES = ("stack", "tube")
FORMATS = ("png", "pdf")
DPI = 300

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
        return (f"{self.manifest.get('trial_id', '?')} · "
                f"layer {self.manifest.get('layer_index', '?')} take {take}")

    @property
    def caption(self) -> str:
        """The one line that keeps a figure honest when it is pasted into a paper."""
        metrics = self.manifest.get("metrics") or {}
        annotation = self.manifest.get("annotation") or {}
        offset = annotation.get("introduced_offset_mm")
        parts = []
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


def _intrinsics(manifest: dict, layer_dir: Path) -> np.ndarray | None:
    """K from the layer's provenance, falling back to the trial's."""
    found = ((manifest.get("provenance") or {}).get("camera_intrinsics") or {}).get("K")
    if found is None:
        trial_file = layer_dir.parent / "trial.json"
        if trial_file.is_file():
            trial = json.loads(trial_file.read_text(encoding="utf-8"))
            found = (((trial.get("provenance") or {}).get("camera_intrinsics") or {})
                     .get("K"))
    return None if found is None else np.asarray(found, dtype=float)


def load_take(layer_dir: Path) -> TakeData:
    layer_dir = Path(layer_dir)
    manifest_file = layer_dir / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"not an archived take: {layer_dir}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    cloud_file = layer_dir / (manifest.get("pointcloud_file") or "height-or-pointcloud.npy")
    depth_file = layer_dir / (manifest.get("depth_file") or "depth.npy")
    cloud = np.load(cloud_file) if cloud_file.is_file() else None
    if cloud is not None and (cloud.ndim != 2 or cloud.shape[1] != 3):
        cloud = None                                   # a height map, not a cloud
    transform = (manifest.get("provenance") or {}).get("T_work_camera")
    return TakeData(
        layer_dir=layer_dir, manifest=manifest,
        nominal=_points(layer_dir / "nominal_path.json"),
        measured=_points(layer_dir / "measured_path.json"),
        cloud=cloud,
        depth=np.load(depth_file) if depth_file.is_file() else None,
        K=_intrinsics(manifest, layer_dir),
        T_work_camera=None if transform is None else np.asarray(transform, dtype=float))


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
    if take.depth is not None and take.K is not None and take.T_work_camera is not None:
        points, _ = depth_to_work_points(take.depth, take.K, take.T_work_camera)
        return points if len(points) else take.cloud
    return take.cloud


def _z_exaggeration(radius: float, z_span: float) -> float:
    """Make a few mm of height readable next to tens of mm of radius, and say so."""
    if z_span <= 0:
        return 1.0
    return max(1.0, round((radius * .5) / z_span, 1))


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
    band = z[(z > -15.0) & (z < 60.0)]
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


def _figure_iso(plt, take: TakeData):
    if take.cloud is None and take.measured is None:
        return None
    fig = plt.figure(figsize=(6.6, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    zs = [a[:, 2] for a in (take.cloud, take.measured) if a is not None and len(a)]
    span = float(np.ptp(np.concatenate(zs))) if zs else 0.0
    factor = _z_exaggeration(take.radius, span)
    if take.cloud is not None and len(take.cloud):
        ax.scatter(take.cloud[:, 0], take.cloud[:, 1], take.cloud[:, 2] * factor,
                   s=2, c=CLOUD, linewidths=0, label="deposit surface")
    if take.nominal is not None:
        ax.plot(take.nominal[:, 0], take.nominal[:, 1], take.nominal[:, 2] * factor,
                color=NOMINAL, linewidth=1.4, linestyle="--", label="nominal circle")
    if take.measured is not None:
        ax.plot(take.measured[:, 0], take.measured[:, 1], take.measured[:, 2] * factor,
                color=MEASURED, linewidth=2.2, label="extracted centreline")
    ax.set_xlabel("X (mm)", fontsize=9)
    ax.set_ylabel("Y (mm)", fontsize=9)
    ax.set_zlabel(f"Z × {factor:g} (mm)", fontsize=9)
    ax.set_title(f"Oblique view — {take.label}", fontsize=10)
    ax.view_init(elev=26, azim=-58)
    ax.legend(loc="upper left", fontsize=8)
    fig.text(.5, .015, f"{take.caption} · vertical exaggeration ×{factor:g}",
             ha="center", fontsize=7.5, color="#4b5563")
    fig.tight_layout(rect=(0, .035, 1, 1))
    return fig


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


def take_stages(take: TakeData) -> "dict | None":
    """Re-run the archived frame through the real chain, keeping every stage.

    Nothing is re-implemented here: this calls ``process_observation`` with a
    collector, so the panels show the arrays the pipeline actually held. Needs
    the scan extra (Open3D); without it the method figure is skipped like any
    other figure that cannot be drawn.
    """
    if take.depth is None or take.K is None or take.T_work_camera is None:
        return None
    from .processing import plan_for_archived_take, process_observation

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
    stages: dict = {}
    colour = take.layer_dir / (take.manifest.get("color_file") or "color.png")
    image = None
    if colour.is_file():
        import cv2
        image = cv2.imread(str(colour), cv2.IMREAD_COLOR)
    if image is None:
        image = np.zeros((*np.asarray(take.depth).shape[:2], 3), np.uint8)
    try:
        result = process_observation(
            color=image, depth=take.depth,
            T_work_camera=take.T_work_camera, K=take.K, plan=plan,
            layer=plan.layers[index - 1],
            config=ExtrusionConfig.model_validate(config_payload), stages=stages)
    except Exception:
        # A take that cannot be reconstructed still gets its other figures.
        return stages or None
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
    """
    stages = take_stages(take)
    if not stages:
        return None
    raw = stages.get("backprojected")
    if raw is None or not len(raw):
        return None
    result = stages.get("result")
    radius = (take._nominal_circle or ((0.0, 0.0), 40.0))[1]
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
    factor = _z_exaggeration(radius or 40.0,
                             float(np.ptp(thin[:, 2])) if len(thin) else 0.0)
    if len(thin):
        ax.scatter(thin[:, 0], thin[:, 1], thin[:, 2] * factor, s=.7, c=thin[:, 2],
                   cmap="viridis", vmin=band[0], vmax=band[1], linewidths=0)
    ax.set_title(f"2 · the same points obliquely (Z × {factor:g})", fontsize=10)
    ax.view_init(elev=24, azim=-62)
    if window:
        ax.set_xlim(window[0], window[1])
        ax.set_ylim(window[2], window[3])
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
    measured = result.measured_xyz if result is not None else take.measured
    if take.nominal is not None:
        ax.plot(take.nominal[:, 0], take.nominal[:, 1], color=NOMINAL, linewidth=1.3,
                linestyle="--", label="nominal")
    if measured is not None and len(measured):
        ax.plot(measured[:, 0], measured[:, 1], color=MEASURED, linewidth=2.0,
                label="extracted")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title("6 · extracted vs nominal", fontsize=10)
    _square(ax, measured if measured is not None else take.nominal)

    fig.suptitle(f"From one RGB-D frame to a measured centreline — {take.label}",
                 fontsize=12)
    fig.text(.5, .012, take.caption, ha="center", fontsize=7.5, color="#4b5563")
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
    try:                                   # keep the ring round, not an ellipse
        ax.set_box_aspect((1, 1, .55))
    except Exception:
        pass
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
                   "iso": _figure_iso, "profile": _figure_profile,
                   "pipeline": _figure_pipeline}


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
    zs = [t.measured[:, 2] for t in takes if t.measured is not None and len(t.measured)]
    span = float(np.ptp(np.concatenate(zs))) if zs else 0.0
    radius = takes[0].radius
    factor = _z_exaggeration(radius, span)
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
