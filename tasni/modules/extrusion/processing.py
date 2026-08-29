"""Single-RGB-D deposited-centreline reconstruction for cylinder layers.

All points are converted to the selected work frame in millimetres before any
geometry decision. Open3D is imported lazily; saved observations can therefore
be archived and the rest of Tasni imported even when the scan extra is absent.
"""
from __future__ import annotations

import math
import json
import time
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

from ...core.depth_geometry import CameraGeometry, ColorRegistered, backproject
from ...core.geometry import transform_points
from .comparison import compare_circle, corrected_circle, fit_circle_xy
from .models import (CylinderPlan, CylinderRecipe, CylinderSetup, DeviationMetrics,
                     LayerPath, RingGeometry)
from .toolpath import generate_cylinder_plan


@dataclass
class ProcessingResult:
    measured_xyz: np.ndarray
    corrected_xyz: np.ndarray | None
    metrics: DeviationMetrics
    segmentation: np.ndarray
    skeleton: np.ndarray
    comparison: np.ndarray
    report: dict
    filtered_xyz: np.ndarray | None = None
    geometry: RingGeometry | None = None


def depth_to_work_points(depth: np.ndarray, geometry: CameraGeometry,
                         T_work_camera: np.ndarray) -> tuple[np.ndarray, int]:
    """Native depth -> work-frame mm. ``T_work_camera`` is the COLOUR camera's pose
    (the hand-eye); ``backproject`` already returns colour-frame points."""
    camera, _uv = backproject(depth, geometry)
    return transform_points(T_work_camera, camera), int(len(camera))


def deposit_floor_mm(config, chroma_gated: bool) -> float:
    """Lowest height above the work plane a point may have and still be deposit.

    Coupled to the colour gate on purpose: the low floor is EARNED by it. With
    the board removed by chroma, 1.5 mm stops amputating the bead; without it,
    1.5 mm lets the board flood in and every take exhausts the branch guard
    (measured on all four cell frames of 2026-08-29). So an abstaining gate --
    an RGB dropout, a depth-only fixture -- also restores the conservative floor
    rather than handing the chain a cloud it cannot survive.
    """
    floor = max(config.deposit_min_height_mm, config.plane_distance_threshold_m * 1000.0)
    if chroma_gated:
        return float(floor)
    return float(max(floor, getattr(config, "deposit_min_height_no_chroma_mm", 0.0)))


def chroma_gate_mask(color: np.ndarray | None, registered: ColorRegistered, config,
                     counts: dict | None = None) -> tuple[np.ndarray, bool]:
    """Per REGISTERED POINT: True where the colour frame says "bead", not "board".

    Depth is native and not aligned to colour any more (camera protocol 2), so
    the gate cannot blank depth pixels in place; each depth point is projected
    into the calibrated colour model (``registered.uv``) and the saturation mask
    is read there. Everything else is the 2026-08-29 gate unchanged (`041ad1b`):
    height cannot tell bead from board (the bare ChArUco board reads 1-3 LSB
    above the work plane, so every floor a real bead clears passes board with
    it -- cell 2026-08-29 take 4: a 22-point patch of black checker 12 mm
    outside the ring survived the radial trim, dilated into a 17-22 px arm and
    exhausted the branch guard). Saturation separates the two ~20:1 (the clay is
    chromatic, the printed board is not): saturation > threshold, a
    chroma-fraction abstention for RGB dropouts and depth-only fixtures, and a
    closing so speckle inside the bead does not punch holes. Points that project
    OUTSIDE the colour image (the depth field is wider than colour) have no
    colour evidence and are dropped while the gate applies -- they are far
    outside any ring ROI anyway. Abstains as ``(all True, False)``, which
    ``deposit_floor_mm`` turns into the 2.5 mm floor.
    """
    def note(key: str, value: int) -> None:
        if counts is not None:
            counts[key] = value

    n = len(registered)
    threshold = int(getattr(config, "deposit_min_saturation", 0) or 0)
    if threshold <= 0 or color is None:
        note("chroma_gate_applied", 0)
        return np.ones(n, bool), False
    image = np.asarray(color)
    w, h = registered.color_size
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != (h, w):
        note("chroma_gate_applied", 0)
        return np.ones(n, bool), False
    saturation = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV)[:, :, 1]
    keep = (saturation > threshold).astype(np.uint8)
    note("chroma_gate_kept_pixels", int(keep.sum()))
    if float(keep.mean()) < float(getattr(config, "deposit_min_chroma_fraction", 0.0)):
        note("chroma_gate_applied", 0)
        return np.ones(n, bool), False
    # Speckle inside the bead would otherwise punch holes through it. The close
    # kernel was tuned at 720p; scale it with the colour frame's own width so a
    # 1920x1080 colour image (protocol 2) closes the same physical gap.
    k = max(3, int(round(5 * w / 1280.0)))
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    u = np.rint(registered.uv[:, 0]).astype(int)
    v = np.rint(registered.uv[:, 1]).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    mask = np.zeros(n, bool)
    mask[inside] = keep[v[inside], u[inside]] > 0
    note("chroma_gate_applied", 1)
    note("chroma_gate_outside_colour", int((~inside).sum()))
    return mask, True


def _largest_label(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    good = labels >= 0
    if not np.any(good):
        return points[:0]
    ids, counts = np.unique(labels[good], return_counts=True)
    return points[labels == ids[int(np.argmax(counts))]]


def _thin(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is not None and hasattr(ximgproc, "thinning"):
        return (ximgproc.thinning(binary) > 0).astype(np.uint8)
    # Morphological fallback for OpenCV builds without ximgproc.
    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = binary.copy()
    while cv2.countNonZero(work):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = cv2.erode(work, element)
    return (skeleton > 0).astype(np.uint8)


_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
               (0, 1), (1, -1), (1, 0), (1, 1)]


def _graph(skeleton: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    pixels = {tuple(p) for p in np.argwhere(skeleton > 0)}
    graph: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for p in pixels:
        links = []
        for dr, dc in _NEIGHBOURS:
            q = (p[0] + dr, p[1] + dc)
            if q not in pixels:
                continue
            if dr and dc:
                # A thinned raster stair-step often contains an orthogonal corner
                # plus the diagonal shortcut. Keeping all three makes a perfect
                # circle look branched. Suppress only that redundant diagonal;
                # a genuine diagonal segment (no orthogonal bridge) is retained.
                if (p[0] + dr, p[1]) in pixels or (p[0], p[1] + dc) in pixels:
                    continue
            links.append(q)
        graph[p] = sorted(links)
    return graph


def _ordered_skeleton(skeleton: np.ndarray) -> tuple[np.ndarray, int, int]:
    graph = _graph(skeleton)
    if not graph:
        raise RuntimeError("skeleton is empty")
    branches = sum(1 for links in graph.values() if len(links) > 2)
    endpoints = sorted(p for p, links in graph.items() if len(links) == 1)
    if branches:
        raise RuntimeError(f"skeleton has {branches} branch pixel(s)")
    start = endpoints[0] if endpoints else min(graph)
    ordered = [start]
    visited = {start}
    previous = None
    current = start
    while True:
        choices = [p for p in graph[current] if p not in visited]
        if not choices:
            break
        if previous is None:
            nxt = choices[0]
        else:
            incoming = np.array(current, dtype=float) - np.array(previous, dtype=float)
            # Deterministic straightest continuation prevents pixel-order zig-zag.
            scored = []
            for candidate in choices:
                outgoing = np.array(candidate, dtype=float) - np.array(current, dtype=float)
                denom = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
                score = float(np.dot(incoming, outgoing) / denom) if denom else -1.0
                scored.append((-score, candidate))
            nxt = min(scored)[1]
        previous, current = current, nxt
        ordered.append(current)
        visited.add(current)
    completeness = len(visited) / max(1, len(graph))
    if completeness < 0.95:
        raise RuntimeError(f"ordered skeleton covers only {completeness:.1%} of pixels")
    return np.asarray(ordered, dtype=int), branches, len(endpoints)


def _prune_short_spurs(skeleton: np.ndarray, max_length: int) -> tuple[np.ndarray, int]:
    """Remove only endpoint-to-junction twigs shorter than ``max_length``.

    Raster thinning commonly leaves one- or two-pixel cardinal ticks on an
    otherwise closed annulus.  A real process branch has a longer arm and is
    still rejected by the guarded graph ordering below.
    """
    result = (skeleton > 0).astype(np.uint8).copy()
    removed = 0
    while True:
        graph = _graph(result)
        candidates: list[list[tuple[int, int]]] = []
        for endpoint in sorted(p for p, links in graph.items() if len(links) == 1):
            path = [endpoint]
            previous = None
            current = endpoint
            # ``path`` includes the junction; the removable arm does not.
            while len(path) <= max_length + 1:
                links = [p for p in graph[current] if p != previous]
                if len(graph[current]) > 2:
                    candidates.append(path[:-1])
                    break
                if not links:
                    break
                previous, current = current, links[0]
                path.append(current)
        pixels = {p for path in candidates for p in path}
        if not pixels:
            return result, removed
        for row, col in pixels:
            result[row, col] = 0
        removed += len(pixels)


def _rasterize(points: np.ndarray, mm_per_pixel: float, bead_mm: float,
               attempt: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = points[:, :2]
    margin = max(4.0, bead_mm)
    lo = xy.min(axis=0) - margin
    hi = xy.max(axis=0) + margin
    size = np.ceil((hi - lo) / mm_per_pixel).astype(int) + 1
    if np.any(size > 4096):
        raise RuntimeError(f"processing raster too large: {size[0]}x{size[1]}")
    pixels = np.rint((xy - lo) / mm_per_pixel).astype(int)
    mask = np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
    mask[pixels[:, 1], pixels[:, 0]] = 255
    radius_px = max(1, int(round(bead_mm / (2 * mm_per_pixel))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * radius_px + 1, 2 * radius_px + 1))
    mask = cv2.dilate(mask, kernel)
    close_px = max(1, radius_px + attempt)
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * close_px + 1, 2 * close_px + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    return mask, lo, pixels


def _fit_spline(points: np.ndarray, count: int, closed: bool) -> np.ndarray:
    if len(points) < 4:
        raise RuntimeError("too few ordered centreline points for spline fitting")
    fit = points[:-1] if closed and np.allclose(points[0], points[-1]) else points
    k = min(3, len(fit) - 1)
    tck, _ = splprep(fit.T, s=0.0, per=closed, k=k)
    u = np.linspace(0, 1, count, endpoint=not closed)
    result = np.column_stack(splev(u, tck))
    return np.vstack((result, result[0])) if closed else result


def _comparison_image(mask: np.ndarray, lo: np.ndarray, mm_per_pixel: float,
                      nominal: np.ndarray, measured: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    def polyline(points, color, thickness):
        uv = np.rint((points[:, :2] - lo) / mm_per_pixel).astype(np.int32)
        cv2.polylines(image, [uv.reshape(-1, 1, 2)], True, color, thickness, cv2.LINE_AA)

    polyline(nominal, (255, 160, 40), 2)   # commanded/nominal blue
    polyline(measured, (40, 40, 240), 2)   # measured red
    return image


def bead_width_profile(cluster_xyz, center_xy, *, bins: int = 36,
                       low_pct: float = 2.5, high_pct: float = 97.5) -> dict:
    """Radial extent of the deposit per angular bin, about ``center_xy``.

    The XY FOOTPRINT width of the bead, not a cross-section measured normal to
    the surface. Percentiles rather than min/max so a stray point cannot widen a
    bin; bins with too little deposit report ``None`` rather than a guess.
    """
    pts = np.asarray(cluster_xyz, dtype=float)
    rel = pts[:, :2] - np.asarray(center_xy, dtype=float)
    radii = np.linalg.norm(rel, axis=1)
    angle = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2 * math.pi)
    edges = np.linspace(0.0, 2 * math.pi, bins + 1)
    which = np.clip(np.digitize(angle, edges) - 1, 0, bins - 1)
    widths = np.full(bins, np.nan)
    for index in range(bins):
        r = radii[which == index]
        if len(r) >= 8:
            widths[index] = np.percentile(r, high_pct) - np.percentile(r, low_pct)
    valid = widths[np.isfinite(widths)]
    if not len(valid):
        raise RuntimeError("bead width: no angular bin had enough deposit points")
    return {"bins": bins, "bins_with_data": int(len(valid)),
            "per_bin_mm": [None if not np.isfinite(w) else float(w) for w in widths],
            "mean_mm": float(valid.mean()), "min_mm": float(valid.min()),
            "max_mm": float(valid.max())}


def ring_geometry(measured_xyz, cluster_xyz, center_xy, *, floor_profile=None,
                  build_plane_z_mm: float = 0.0, bins: int = 36) -> RingGeometry:
    """Height profile along the measured centreline plus the bead's footprint.

    Height is measured against the previous ring's own measured top where one is
    given, so a stacked ring reports ITS layer height rather than its absolute
    elevation above the table.
    """
    measured = np.asarray(measured_xyz, dtype=float)
    top = measured[:, 2]
    if floor_profile is None:
        reference = np.full(len(top), float(build_plane_z_mm))
        reference_name = "build_plane"
    else:
        profile = np.asarray(floor_profile, dtype=float).reshape(-1, 3)
        _, nearest = cKDTree(profile[:, :2]).query(measured[:, :2])
        reference = profile[nearest, 2]
        reference_name = "previous_layer_measured"
    height = top - reference
    width = bead_width_profile(cluster_xyz, center_xy, bins=bins)
    return RingGeometry(
        top_z_mean_mm=float(top.mean()), top_z_min_mm=float(top.min()),
        top_z_max_mm=float(top.max()), top_z_std_mm=float(top.std()),
        height_mean_mm=float(height.mean()), height_min_mm=float(height.min()),
        height_max_mm=float(height.max()), height_reference=reference_name,
        bead_width_mean_mm=width["mean_mm"], bead_width_min_mm=width["min_mm"],
        bead_width_max_mm=width["max_mm"], bead_width_bins=bins)


def _deposit_clusters(points: np.ndarray, config, counts: dict) -> list[np.ndarray]:
    """Voxel/outlier filter, then return every non-noise DBSCAN cluster by size."""
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required for extrusion processing; install tasni[scan]") from exc
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(config.voxel_size_m * 1000.0)
    counts["after_voxel"] = len(cloud.points)
    cloud, _ = cloud.remove_statistical_outlier(
        nb_neighbors=config.statistical_neighbors, std_ratio=config.statistical_std_ratio)
    counts["after_statistical"] = len(cloud.points)
    cloud, _ = cloud.remove_radius_outlier(
        nb_points=config.radius_neighbors, radius=config.radius_m * 1000.0)
    counts["after_radius"] = len(cloud.points)
    if len(cloud.points) < config.cluster_min_points:
        raise RuntimeError("deposited cloud was removed by outlier filtering")
    labels = np.asarray(cloud.cluster_dbscan(
        eps=config.cluster_eps_m * 1000.0, min_points=config.cluster_min_points,
        print_progress=False))
    filtered = np.asarray(cloud.points)
    clusters = [filtered[labels == label] for label in np.unique(labels) if label >= 0]
    clusters.sort(key=len, reverse=True)
    counts["dbscan_cluster_count"] = len(clusters)
    if not clusters:
        raise RuntimeError("no deposited cluster survived DBSCAN filtering")
    return clusters


def _filter_deposit(points: np.ndarray, config, counts: dict, *,
                    assemble_arcs: bool = False,
                    search_center_xy: np.ndarray | None = None) -> np.ndarray:
    """Voxel -> statistical -> radius outliers -> largest DBSCAN cluster (Open3D).

    Everything the deposit IS, flanks included -- the bead's full footprint. The
    upward-facing crest is a further selection made by :func:`_top_surface`; bead
    width has to be measured before that selection throws the flanks away.

    ``assemble_arcs`` rejoins clusters that are arcs of one ring, for the single
    isolated ring a characterization looks at. It stays OFF for layer
    measurement: there the ROI deliberately spans the ring beneath, and fusing a
    displaced ring to the crescent of its neighbour would destroy the very
    displacement the measurement exists to report.
    """
    clusters = _deposit_clusters(points, config, counts)
    if assemble_arcs and len(clusters) > 1 and search_center_xy is not None:
        try:
            selected, selector = _select_ring_cluster(
                clusters, np.asarray(search_center_xy, dtype=float), counts)
        except RuntimeError as exc:
            # Refinement must not invent a failure the coarse pass did not have:
            # fall back to the largest arc and let the completeness metric report
            # how much of the ring it actually covers.
            counts["assembly_skipped"] = str(exc)[:120]
        else:
            counts["assembled_clusters"] = next(
                (c["cluster_count"] for c in selector["candidates"]
                 if c.get("selected")), 1)
            return selected
    points = clusters[0]
    counts["after_largest_cluster"] = len(points)
    return points


def _radial_trim(points: np.ndarray, schedule_mm, counts: dict, *,
                 minimum: int = 8) -> np.ndarray:
    """Keep only points within a tightening band of the circle fitted to them.

    One pass per band in ``schedule_mm``: fit on what is kept, select from
    EVERYTHING, so a point cut by a first fit biased toward the contamination can
    return once the refit has moved onto the ring. The bead is a narrow annulus;
    board-plane noise fused to it is not -- and it shares the bead's height band
    and its upward normals, so this is the only filter in the chain that can see
    the difference.

    Never empties the set: a band that would leave fewer than ``minimum`` points
    is skipped and the previous set stands. ``after_radial_trim`` records what
    survived.
    """
    kept = points
    for band in schedule_mm or ():
        if len(kept) < minimum or float(band) <= 0:
            break
        center, radius = fit_circle_xy(kept)
        distance = np.abs(np.linalg.norm(points[:, :2] - center, axis=1) - radius)
        candidate = points[distance <= float(band)]
        if len(candidate) < minimum:
            break
        kept = candidate
    counts["after_radial_trim"] = len(kept)
    return kept


_ASSEMBLY_MAX_CLUSTERS = 12


def _ring_shape(points: np.ndarray, angular_bins: int) -> dict:
    """The circle fit and the two shape statistics the ring gate is built on."""
    center, radius = fit_circle_xy(points)
    radii = np.linalg.norm(points[:, :2] - center, axis=1)
    theta = np.mod(np.arctan2(points[:, 1] - center[1],
                              points[:, 0] - center[0]), 2 * math.pi)
    occupied = len(np.unique(np.floor(theta / (2 * math.pi) * angular_bins).astype(int)))
    span = float(np.percentile(radii, 97.5) - np.percentile(radii, 2.5))
    return {"center": center, "radius": float(radius), "occupied": int(occupied),
            "coverage": occupied / angular_bins, "span": span,
            "span_ratio": span / max(float(radius), 1e-9)}


def _assemble_ring_arcs(clusters: list[np.ndarray], shapes: list[dict | None], *,
                        angular_bins: int, min_coverage: float,
                        max_span_ratio: float, min_radius_mm: float) -> list[tuple[int, ...]]:
    """Group clusters that are arcs of ONE ring.

    A bead that dips under the ROI height floor -- or is occluded, or glares --
    reaches DBSCAN as two or more arcs of the same circle. Graded separately none
    of them need span 70% of the circumference, so a fully captured ring could be
    rejected outright: the 2026-08-29 low-relief capture arrived as a 48/72 arc
    and a 25/72 arc that together cover 71/72.

    A merge is accepted only when the union stays as tight radially as the gate
    already demands AND spans more of the circle than before, so unrelated blobs
    -- which blow the radial span apart -- can never join. Assembly is attempted
    only from an incomplete seed, and any group containing a cluster that is
    already a complete ring on its own is discarded: a frame that works today
    must keep selecting exactly what it selects today.
    """
    groups: list[tuple[int, ...]] = []
    complete = {i for i, s in enumerate(shapes)
                if s is not None and s["coverage"] >= min_coverage
                and s["radius"] >= min_radius_mm and s["span_ratio"] <= max_span_ratio}
    # The search is cubic in the cluster count, and a ring only ever arrives in a
    # handful of arcs; clusters are size-ordered, so the tail is speckle.
    considered = min(len(clusters), _ASSEMBLY_MAX_CLUSTERS)
    for seed in range(considered):
        shape = shapes[seed]
        if shape is None or seed in complete or shape["radius"] < min_radius_mm:
            continue
        members = [seed]
        while True:
            best: tuple[int, dict] | None = None
            for other in range(considered):
                if other in members or shapes[other] is None:
                    continue
                try:
                    trial = _ring_shape(
                        np.vstack([clusters[i] for i in (*members, other)]), angular_bins)
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    continue
                if (trial["radius"] < min_radius_mm
                        or trial["span_ratio"] > max_span_ratio
                        or trial["occupied"] <= shape["occupied"]):
                    continue
                if best is None or trial["occupied"] > best[1]["occupied"]:
                    best = (other, trial)
            if best is None:
                break
            members.append(best[0])
            shape = best[1]
        key = tuple(sorted(members))
        if len(members) > 1 and not complete.intersection(members) and key not in groups:
            groups.append(key)
    return groups


def _select_ring_cluster(clusters: list[np.ndarray], search_center_xy: np.ndarray,
                         counts: dict) -> tuple[np.ndarray, dict]:
    """Choose a complete annulus, not simply the largest above-plane residual.

    The real checkerboard has millimetre-scale depth bias over broad patches. Such
    a patch can contain more points than the deposited ring, but it has a much
    larger radial spread after a circle fit. A ring must cover most angular bins
    and keep its central 95% radial span below 80% of its fitted radius.

    Completeness is judged on the assembled ring, never on one connected
    component: see :func:`_assemble_ring_arcs` for why a ring can arrive in
    pieces and why joining them cannot admit anything that is not ring-shaped.
    """
    angular_bins = 72
    min_coverage = 0.70
    max_span_ratio = 0.80
    min_radius_mm = 5.0
    diagnostics: list[dict] = []
    eligible: list[tuple[float, int, np.ndarray]] = []

    shapes: list[dict | None] = []
    for cluster in clusters:
        try:
            shapes.append(_ring_shape(cluster, angular_bins))
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            shapes.append(None)
    groups = _assemble_ring_arcs(clusters, shapes, angular_bins=angular_bins,
                                 min_coverage=min_coverage,
                                 max_span_ratio=max_span_ratio,
                                 min_radius_mm=min_radius_mm)
    members: list[tuple[int, ...]] = [(i,) for i in range(len(clusters))] + groups

    for index, member in enumerate(members):
        points = (clusters[member[0]] if len(member) == 1
                  else np.vstack([clusters[i] for i in member]))
        record: dict = {"candidate": index + 1, "points": int(len(points)),
                        "cluster_count": len(member),
                        "cluster_indices": [i + 1 for i in member]}
        try:
            shape = (shapes[member[0]] if len(member) == 1
                     else _ring_shape(points, angular_bins))
            if shape is None:
                raise RuntimeError("circle fit failed")
            center, radius = shape["center"], shape["radius"]
            coverage, span_ratio = shape["coverage"], shape["span_ratio"]
            center_offset = float(np.linalg.norm(center - search_center_xy))
            is_eligible = (radius >= min_radius_mm and coverage >= min_coverage
                           and span_ratio <= max_span_ratio)
            score = math.sqrt(len(points)) * coverage / max(span_ratio, 0.05)
            record.update({
                "center_mm": [float(center[0]), float(center[1])],
                "center_offset_mm": center_offset,
                "radius_mm": float(radius),
                "angular_bins_occupied": shape["occupied"],
                "angular_coverage": float(coverage),
                "radial_span_95_mm": shape["span"],
                "radial_span_ratio": float(span_ratio),
                "eligible": bool(is_eligible),
                "score": float(score),
            })
            if is_eligible:
                eligible.append((score, index, points))
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            record.update({"eligible": False, "rejection": str(exc)})
        diagnostics.append(record)

    selector = {
        "angular_bins": angular_bins,
        "minimum_angular_coverage": min_coverage,
        "maximum_radial_span_ratio": max_span_ratio,
        "minimum_radius_mm": min_radius_mm,
        "assembled_candidates": [[i + 1 for i in group] for group in groups],
        "candidates": diagnostics,
    }
    if not eligible:
        raise RuntimeError(
            "deposited geometry was found, but no complete ring-like cluster passed "
            f"the characterization shape gate: {json.dumps(selector)}")
    _, selected_index, selected = max(eligible, key=lambda item: item[0])
    selector["selected_candidate"] = selected_index + 1
    diagnostics[selected_index]["selected"] = True
    counts["after_ring_selection"] = len(selected)
    return selected, selector


def _top_surface(points: np.ndarray, config, counts: dict) -> np.ndarray:
    """Upward-facing points of the deposit, then the largest cluster of those.

    This is the surface the centreline is read from: the crest, not the flanks.
    """
    import open3d as o3d
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=100.0, max_nn=30))
    cloud.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
    normals = np.asarray(cloud.normals)
    points = points[normals[:, 2] > config.upwards_normal_z]
    counts["after_upward_normals"] = len(points)
    if len(points) < config.cluster_min_points:
        raise RuntimeError("too few upward-facing deposited points")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    labels = np.asarray(cloud.cluster_dbscan(
        eps=config.normal_cluster_eps_m * 1000.0,
        min_points=config.cluster_min_points, print_progress=False))
    points = _largest_label(points, labels)
    counts["after_normal_cluster"] = len(points)
    return points


def plan_for_archived_take(manifest: dict, trial: dict, *,
                           nominal_xyz: np.ndarray | None = None) -> CylinderPlan:
    """The plan an archived take was MEASURED against, not the trial's first plan.

    A measure-only session is created before Characterize -> Apply, so
    ``trial.json`` carries the pre-Apply recipe and centre. The take's own
    manifest holds the recipe it was measured with, and its archived nominal
    ring holds the centre -- fitted, never averaged, because the archived path
    is closed and its first point repeats. Shared by offline reprocessing and by
    the method figure so the two cannot disagree about what a take meant.
    """
    recipe = (CylinderRecipe.model_validate(manifest["recipe"]) if manifest.get("recipe")
              else CylinderRecipe.model_validate(trial["recipe"]))
    setup = CylinderSetup.model_validate(trial["setup"])
    if nominal_xyz is not None:
        nominal = np.asarray(nominal_xyz, dtype=float).reshape(-1, 3)
        if len(nominal) >= 3:
            center, _ = fit_circle_xy(nominal)
            setup = setup.model_copy(update={"center_x_mm": float(center[0]),
                                             "center_y_mm": float(center[1])})
    return generate_cylinder_plan(recipe, setup)


def process_observation(*, color: np.ndarray, depth: np.ndarray,
                        geometry: CameraGeometry, T_work_camera: np.ndarray,
                        K: np.ndarray, dist: np.ndarray | None,
                        plan: CylinderPlan, layer: LayerPath, config,
                        floor_profile: np.ndarray | None = None,
                        stages: dict | None = None,
                        assemble_arcs: bool = False) -> ProcessingResult:
    """Reconstruct one layer from exactly one saved synchronized RGB-D frame.

    ``geometry`` is the frame's own greeting (native depth intrinsics + the
    depth->colour extrinsic; Task 7's ``CameraGeometry``); ``T_work_camera`` is
    the COLOUR camera's hand-eye pose, which is what its colour-frame points
    need. ``K``/``dist`` are the CALIBRATED colour model, used for exactly one
    thing: projecting registered depth points into the colour image for the
    chroma gate (see :func:`chroma_gate_mask`).

    ``floor_profile`` is the previous layer's measured centreline (Nx3, work
    frame). Given it, the ROI floor becomes that surface's local height at the
    nearest XY sample rather than a single build-plane number -- which is what
    lets a DISPLACED ring be measured without the exposed crescent of the ring
    beneath it being dragged into the same skeleton. Omitted, behaviour is
    exactly as before.

    ``assemble_arcs`` rejoins ring arcs that the height floor or an occlusion
    split apart; see :func:`_filter_deposit` for why only a characterization
    turns it on.

    ``stages``, when a dict is passed, is filled with a copy of the cloud at
    each step of the chain (backprojected, work_roi, above_floor,
    deposit_cluster, radial_trimmed, top_surface). It exists so the method
    figure can draw what this function actually did instead of a second
    implementation of it that could drift. Collecting costs a few copies and
    changes nothing else.
    """
    def keep(name: str, cloud: np.ndarray) -> None:
        if stages is not None:
            stages[name] = np.asarray(cloud, dtype=float).copy()
    started = time.perf_counter()
    timings: dict[str, float] = {}
    counts: dict[str, int] = {}

    mark = time.perf_counter()
    reg = ColorRegistered.build(depth, geometry, K, dist)
    keep_mask, chroma_gated = chroma_gate_mask(color, reg, config, counts)
    points = transform_points(T_work_camera, reg.pts_mm[keep_mask])
    counts["raw_depth_pixels"] = int(keep_mask.sum())
    timings["backproject_ms"] = (time.perf_counter() - mark) * 1000
    keep("backprojected", points)
    setup, recipe = plan.setup, plan.recipe
    radius = np.linalg.norm(points[:, :2] - np.array([setup.center_x_mm, setup.center_y_mm]), axis=1)
    max_z = layer.nominal_z_mm + recipe.bead_diameter_mm / 2 + config.deposit_height_margin_mm
    # The selected work frame defines the build plane at Z=0, so deterministic
    # height subtraction is more reproducible than fitting a new plane per frame.
    min_z = deposit_floor_mm(config, chroma_gated)
    in_height = (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
    r_lo = recipe.radius_mm - config.radial_roi_margin_mm
    r_hi = recipe.radius_mm + config.radial_roi_margin_mm
    in_radial = (radius >= r_lo) & (radius <= r_hi)
    roi = in_height & in_radial
    # Keep the per-band tallies: when the ROI comes back empty the operator needs
    # to know WHICH band rejected the geometry (a wrong build plane and a wrong
    # centre look identical from the point count alone).
    roi_diag = {
        "height_band_mm": [round(float(min_z), 2), round(float(max_z), 2)],
        "radial_band_mm": [round(float(r_lo), 2), round(float(r_hi), 2)],
        "center_xy_mm": [round(float(setup.center_x_mm), 2),
                         round(float(setup.center_y_mm), 2)],
        "backprojected": int(len(points)),
        "in_height_band": int(in_height.sum()),
        "in_radial_band": int(in_radial.sum()),
        "in_both": int(roi.sum()),
    }
    if len(points):
        pct = lambda a, q: round(float(np.percentile(a, q)), 1)  # noqa: E731
        roi_diag["observed_z_mm"] = [pct(points[:, 2], 1), pct(points[:, 2], 50),
                                     pct(points[:, 2], 99)]
        roi_diag["observed_radius_mm"] = [pct(radius, 1), pct(radius, 50),
                                          pct(radius, 99)]
        if in_radial.any():     # what heights show up where the ring should be
            zr = points[in_radial][:, 2]
            roi_diag["z_within_radial_band_mm"] = [pct(zr, 1), pct(zr, 50), pct(zr, 99)]
    counts.update({k: v for k, v in roi_diag.items() if isinstance(v, int)})
    points = points[roi]
    keep("work_roi", points)
    floor = {"source": "build_plane", "margin_mm": 0.0, "mean_mm": float(min_z)}
    if floor_profile is not None and len(points):
        profile = np.asarray(floor_profile, dtype=float).reshape(-1, 3)
        _, nearest = cKDTree(profile[:, :2]).query(points[:, :2])
        local = profile[nearest, 2] + config.layer_floor_margin_mm
        points = points[points[:, 2] >= local]
        floor = {"source": "previous_layer_measured",
                 "margin_mm": float(config.layer_floor_margin_mm),
                 "mean_mm": float(local.mean())}
        keep("above_floor", points)
    counts["after_work_roi"] = len(points)
    if len(points) < config.cluster_min_points:
        raise RuntimeError(
            "not enough deposited-geometry points inside the configured work ROI "
            f"(need {config.cluster_min_points}); {json.dumps(roi_diag)}")

    mark = time.perf_counter()
    deposit = _filter_deposit(points, config, counts, assemble_arcs=assemble_arcs,
                              search_center_xy=np.array([setup.center_x_mm,
                                                         setup.center_y_mm]))
    keep("deposit_cluster", deposit)
    # Before the crest is picked and before the bead width is read from the
    # flanks: both must see the bead alone, not the board fused to it.
    deposit = _radial_trim(deposit, getattr(config, "radial_trim_schedule_mm", ()), counts)
    keep("radial_trimmed", deposit)
    points = _top_surface(deposit, config, counts)
    keep("top_surface", points)
    timings["filter_ms"] = (time.perf_counter() - mark) * 1000

    attempts: list[dict] = []
    ordered_pixels = None
    final_mask = final_skeleton = None
    lo = None
    for attempt in range(config.branch_guard_max_attempts):
        mark = time.perf_counter()
        mask, candidate_lo, _ = _rasterize(
            points, config.raster_mm_per_pixel, recipe.bead_diameter_mm, attempt)
        skeleton = _thin(mask)
        # The surface cloud already spans the bead width and raster dilation
        # adds roughly another half-width.  Prune only twigs no longer than that
        # combined footprint; longer arms remain safety-significant branches.
        spur_limit = max(2, int(math.ceil(
            1.5 * recipe.bead_diameter_mm / config.raster_mm_per_pixel)))
        skeleton, pruned = _prune_short_spurs(skeleton, spur_limit)
        graph = _graph(skeleton)
        branches = sum(1 for links in graph.values() if len(links) > 2)
        record = {"attempt": attempt + 1, "mask_pixels": int(np.count_nonzero(mask)),
                  "skeleton_pixels": int(np.count_nonzero(skeleton)),
                  "short_spur_pixels_pruned": pruned,
                  "branch_pixels": branches,
                  "time_ms": (time.perf_counter() - mark) * 1000}
        attempts.append(record)
        if not branches:
            try:
                ordered_pixels, _, endpoints = _ordered_skeleton(skeleton)
                record["endpoints"] = endpoints
                final_mask, final_skeleton, lo = mask, skeleton, candidate_lo
                break
            except RuntimeError as exc:
                record["warning"] = str(exc)
    if ordered_pixels is None or final_mask is None or lo is None:
        raise RuntimeError(
            f"branch guard exhausted after {config.branch_guard_max_attempts} attempt(s): "
            f"{attempts}")

    mark = time.perf_counter()
    ordered_xy = np.column_stack((ordered_pixels[:, 1], ordered_pixels[:, 0]))
    ordered_xy = ordered_xy * config.raster_mm_per_pixel + lo
    _, nearest = cKDTree(points[:, :2]).query(ordered_xy)
    ordered_xyz = np.column_stack((ordered_xy, points[nearest, 2]))
    # A skeleton with endpoints is an ARC, not a loop. Splining it periodically
    # would bridge the two ends and report the invented span as measured
    # geometry -- and because a periodic curve is 100% complete by construction,
    # compare_circle's completeness guard could never fire. Measure what was
    # seen; let the gap reach the metrics.
    closed = len(_graph(final_skeleton)) > 2 and not any(
        len(v) == 1 for v in _graph(final_skeleton).values())
    measured = _fit_spline(ordered_xyz, config.measured_spline_points, closed=closed)
    nominal_center = (setup.center_x_mm, setup.center_y_mm)
    metrics = compare_circle(measured, recipe.radius_mm,
                             nominal_center_mm=nominal_center)
    geometry = ring_geometry(measured, deposit, metrics.measured_center_mm,
                             floor_profile=floor_profile,
                             build_plane_z_mm=setup.build_plane_z_mm,
                             bins=config.bead_width_bins)
    corrected = None
    if recipe.correction_enabled and metrics.valid and closed:
        corrected = corrected_circle(
            measured, recipe.radius_mm, layer.nominal_z_mm,
            nominal_center_mm=nominal_center,
            point_count=recipe.points_per_circle,
            gain=config.correction_gain,
            smoothing_points=config.correction_smoothing_points,
            max_correction_mm=config.correction_max_mm)
    timings["centreline_ms"] = (time.perf_counter() - mark) * 1000
    theta = np.linspace(0, 2 * math.pi, recipe.points_per_circle + 1)
    nominal = np.column_stack((setup.center_x_mm + recipe.radius_mm * np.cos(theta),
                               setup.center_y_mm + recipe.radius_mm * np.sin(theta),
                               np.full_like(theta, layer.nominal_z_mm)))
    overlay = _comparison_image(final_mask, lo, config.raster_mm_per_pixel,
                                nominal, measured)
    timings["total_ms"] = (time.perf_counter() - started) * 1000
    report = {
        "counts": counts, "timings_ms": timings, "branch_guard_attempts": attempts,
        "floor": floor, "geometry": geometry.model_dump(mode="json"),
        "coordinate_frame": plan.setup.work_frame, "units": "mm",
        "closed": bool(closed),
        "measured_path_completeness": float(metrics.path_completeness),
        "valid": metrics.valid, "warnings": metrics.warnings,
    }
    return ProcessingResult(measured, corrected, metrics, final_mask,
                            final_skeleton * 255, overlay, report,
                            filtered_xyz=points.copy(), geometry=geometry)


@dataclass
class CharacterizationResult:
    """What a physical ring actually IS, measured with no recipe assumption."""

    radius_mm: float
    center_mm: tuple[float, float]
    bead_width_mm: float
    bead_width_min_mm: float
    bead_width_max_mm: float
    top_z_mean_mm: float
    top_z_min_mm: float
    top_z_max_mm: float
    measured_xyz: np.ndarray
    segmentation: np.ndarray
    skeleton: np.ndarray
    comparison: np.ndarray
    report: dict

    def summary(self) -> dict:
        return {k: getattr(self, k) for k in (
            "radius_mm", "center_mm", "bead_width_mm", "bead_width_min_mm",
            "bead_width_max_mm", "top_z_mean_mm", "top_z_min_mm", "top_z_max_mm")}


def characterize_ring(*, color: np.ndarray, depth: np.ndarray, geometry: CameraGeometry,
                      T_work_camera: np.ndarray, K: np.ndarray, dist: np.ndarray | None,
                      search_center_mm, work_frame: str, config,
                      inspection_tool: str = "Realsense",
                      print_tool: str = "LongCalibTool") -> CharacterizationResult:
    """Measure a ring with NO recipe assumption: coarse fit, then the normal pipeline.

    Pass 1 takes everything above the build plane inside a search cylinder around
    ``search_center_mm``, filters it like a deposit, and fits a circle to get a
    coarse centre/radius/bead. Pass 2 hands those to ``process_observation`` as a
    throwaway recipe so the refined centreline, radius and height profile come out
    of the same code the layer measurements use -- one pipeline, one set of
    numbers, no second implementation to keep honest.

    ``geometry`` is the frame's own greeting; ``K``/``dist`` are the CALIBRATED
    colour model used only to register depth points into the colour image for
    the chroma gate -- see :func:`process_observation`.
    """
    started = time.perf_counter()
    counts: dict[str, int] = {}
    # The same gate as the layer measurements: characterization defines the
    # recipe they are then judged against, so the two must see the same cloud.
    reg = ColorRegistered.build(depth, geometry, K, dist)
    keep_mask, chroma_gated = chroma_gate_mask(color, reg, config, counts)
    points = transform_points(T_work_camera, reg.pts_mm[keep_mask])
    counts["raw_depth_pixels"] = int(keep_mask.sum())
    center = np.asarray(search_center_mm, dtype=float)
    min_z = deposit_floor_mm(config, chroma_gated)
    radial = np.linalg.norm(points[:, :2] - center, axis=1)
    roi = ((points[:, 2] >= min_z) & (points[:, 2] <= config.characterize_max_height_mm)
           & (radial <= config.characterize_search_radius_mm))
    points = points[roi]
    counts["after_search_roi"] = len(points)
    if len(points) < config.cluster_min_points:
        raise RuntimeError("no deposited geometry inside the characterization search region")
    clusters = _deposit_clusters(points, config, counts)
    deposit, selector = _select_ring_cluster(clusters, center, counts)
    coarse_center, coarse_radius = fit_circle_xy(deposit)
    width = bead_width_profile(deposit, coarse_center, bins=config.bead_width_bins)
    top = _top_surface(deposit, config, counts)
    coarse_height = float(np.percentile(top[:, 2], 90))
    coarse = {"center_mm": [float(coarse_center[0]), float(coarse_center[1])],
              "radius_mm": float(coarse_radius), "bead_width_mm": width["mean_mm"],
              "height_mm": coarse_height, "time_ms": (time.perf_counter() - started) * 1000}

    recipe = CylinderRecipe(
        radius_mm=float(np.clip(coarse_radius, 5.0, 500.0)), layer_count=1,
        layer_height_mm=float(np.clip(coarse_height, 0.5, 50.0)),
        bead_diameter_mm=float(np.clip(width["mean_mm"], 0.5, 50.0)),
        robot_speed_mm_s=75.0, extrusion_rate_pct=0.0,
        points_per_circle=config.measured_spline_points)
    setup = CylinderSetup(
        print_tool=print_tool, work_frame=work_frame, inspection_tool=inspection_tool,
        inspection_auto=True, center_x_mm=float(coarse_center[0]),
        center_y_mm=float(coarse_center[1]))
    plan = generate_cylinder_plan(recipe, setup)
    refined = process_observation(color=color, depth=depth, geometry=geometry,
                                  T_work_camera=T_work_camera, K=K, dist=dist,
                                  plan=plan, layer=plan.layers[0], config=config,
                                  assemble_arcs=True)
    geometry = refined.geometry
    report = {**refined.report, "coarse": coarse, "counts_coarse": counts,
              "ring_selector": selector,
              "kind": "characterization",
              "total_ms": (time.perf_counter() - started) * 1000}
    return CharacterizationResult(
        radius_mm=refined.metrics.measured_radius_mm,
        center_mm=refined.metrics.measured_center_mm,
        bead_width_mm=geometry.bead_width_mean_mm,
        bead_width_min_mm=geometry.bead_width_min_mm,
        bead_width_max_mm=geometry.bead_width_max_mm,
        top_z_mean_mm=geometry.top_z_mean_mm, top_z_min_mm=geometry.top_z_min_mm,
        top_z_max_mm=geometry.top_z_max_mm, measured_xyz=refined.measured_xyz,
        segmentation=refined.segmentation, skeleton=refined.skeleton,
        comparison=refined.comparison, report=report)
