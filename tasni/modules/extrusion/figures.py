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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .processing import depth_to_work_points

LAYER_FIGURES = ("plan", "heightmap", "iso", "profile")
TRIAL_FIGURES = ("stack",)
FORMATS = ("png", "pdf")
DPI = 300

# One palette for every figure, chosen to survive greyscale printing: the
# measured path is the darkest line, the nominal is dashed, the cloud is light.
CLOUD = "#98a2b3"
MEASURED = "#c1121f"
NOMINAL = "#2b6cb0"
ACCENT = "#b45309"
CMAP = "viridis"           # perceptually uniform, colour-blind safe, prints well


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
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    if take.cloud is not None and len(take.cloud):
        ax.scatter(take.cloud[:, 0], take.cloud[:, 1], s=3, c=CLOUD, linewidths=0,
                   label=f"deposit surface ({len(take.cloud)} pts)", zorder=2)
    if take.nominal is not None:
        ax.plot(take.nominal[:, 0], take.nominal[:, 1], color=NOMINAL, linewidth=1.6,
                linestyle="--", label="nominal circle", zorder=3)
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
    ax.legend(loc="upper right", fontsize=8, framealpha=.92)
    fig.text(.5, .015, take.caption, ha="center", fontsize=7.5, color="#4b5563")
    fig.tight_layout(rect=(0, .035, 1, 1))
    return fig


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
                   "iso": _figure_iso, "profile": _figure_profile}


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
    for colour, take in zip(colours, takes):
        index = take.manifest.get("layer_index")
        if take.nominal is not None:
            flat.plot(take.nominal[:, 0], take.nominal[:, 1], color=NOMINAL, linewidth=1.0,
                      linestyle="--", zorder=2)
        if take.measured is None:
            continue
        flat.plot(take.measured[:, 0], take.measured[:, 1], color=colour, linewidth=1.9,
                  label=f"layer {index}", zorder=3)
        oblique.plot(take.measured[:, 0], take.measured[:, 1], take.measured[:, 2] * factor,
                     color=colour, linewidth=1.9, label=f"layer {index}")
    flat.set_aspect("equal", adjustable="datalim")
    flat.set_xlabel("X (mm)")
    flat.set_ylabel("Y (mm)")
    flat.set_title("Measured centrelines (dashed = nominal)", fontsize=10)
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
    fig = _figure_stack(plt, latest_takes(trial_dir), trial_dir.name)
    if fig is None:
        return []
    try:
        return _save(fig, trial_dir / "figures", "stack", formats, dpi)
    finally:
        plt.close(fig)


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
