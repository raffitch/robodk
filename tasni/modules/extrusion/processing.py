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


def depth_to_work_points(depth: np.ndarray, K: np.ndarray,
                         T_work_camera: np.ndarray, *, depth_scale: float = 1000.0
                         ) -> tuple[np.ndarray, int]:
    depth = np.asarray(depth)
    valid = np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(float) / float(depth_scale) * 1000.0
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    camera = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    return transform_points(T_work_camera, camera), int(valid.sum())


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


def _filter_deposit(points: np.ndarray, config, counts: dict) -> np.ndarray:
    """Voxel -> statistical -> radius outliers -> largest DBSCAN cluster (Open3D).

    Everything the deposit IS, flanks included -- the bead's full footprint. The
    upward-facing crest is a further selection made by :func:`_top_surface`; bead
    width has to be measured before that selection throws the flanks away.
    """
    points = _deposit_clusters(points, config, counts)[0]
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


def _select_ring_cluster(clusters: list[np.ndarray], search_center_xy: np.ndarray,
                         counts: dict) -> tuple[np.ndarray, dict]:
    """Choose a complete annulus, not simply the largest above-plane residual.

    The real checkerboard has millimetre-scale depth bias over broad patches. Such
    a patch can contain more points than the deposited ring, but it has a much
    larger radial spread after a circle fit. A ring must cover most angular bins
    and keep its central 95% radial span below 80% of its fitted radius.
    """
    angular_bins = 72
    min_coverage = 0.70
    max_span_ratio = 0.80
    min_radius_mm = 5.0
    diagnostics: list[dict] = []
    eligible: list[tuple[float, int, np.ndarray]] = []
    for index, cluster in enumerate(clusters):
        record: dict = {"candidate": index + 1, "points": int(len(cluster))}
        try:
            center, radius = fit_circle_xy(cluster)
            radii = np.linalg.norm(cluster[:, :2] - center, axis=1)
            theta = np.mod(np.arctan2(cluster[:, 1] - center[1],
                                      cluster[:, 0] - center[0]), 2 * math.pi)
            occupied = len(np.unique(np.floor(theta / (2 * math.pi) * angular_bins).astype(int)))
            coverage = occupied / angular_bins
            radial_span = float(np.percentile(radii, 97.5) - np.percentile(radii, 2.5))
            span_ratio = radial_span / max(float(radius), 1e-9)
            center_offset = float(np.linalg.norm(center - search_center_xy))
            is_eligible = (radius >= min_radius_mm and coverage >= min_coverage
                           and span_ratio <= max_span_ratio)
            score = math.sqrt(len(cluster)) * coverage / max(span_ratio, 0.05)
            record.update({
                "center_mm": [float(center[0]), float(center[1])],
                "center_offset_mm": center_offset,
                "radius_mm": float(radius),
                "angular_bins_occupied": int(occupied),
                "angular_coverage": float(coverage),
                "radial_span_95_mm": radial_span,
                "radial_span_ratio": float(span_ratio),
                "eligible": bool(is_eligible),
                "score": float(score),
            })
            if is_eligible:
                eligible.append((score, index, cluster))
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            record.update({"eligible": False, "rejection": str(exc)})
        diagnostics.append(record)

    selector = {
        "angular_bins": angular_bins,
        "minimum_angular_coverage": min_coverage,
        "maximum_radial_span_ratio": max_span_ratio,
        "minimum_radius_mm": min_radius_mm,
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


def process_observation(*, color: np.ndarray, depth: np.ndarray,
                        T_work_camera: np.ndarray, K: np.ndarray,
                        plan: CylinderPlan, layer: LayerPath, config,
                        floor_profile: np.ndarray | None = None) -> ProcessingResult:
    """Reconstruct one layer from exactly one saved synchronized RGB-D frame.

    ``floor_profile`` is the previous layer's measured centreline (Nx3, work
    frame). Given it, the ROI floor becomes that surface's local height at the
    nearest XY sample rather than a single build-plane number -- which is what
    lets a DISPLACED ring be measured without the exposed crescent of the ring
    beneath it being dragged into the same skeleton. Omitted, behaviour is
    exactly as before.
    """
    started = time.perf_counter()
    timings: dict[str, float] = {}
    counts: dict[str, int] = {}

    mark = time.perf_counter()
    points, counts["raw_depth_pixels"] = depth_to_work_points(
        depth, K, T_work_camera, depth_scale=1000.0)
    timings["backproject_ms"] = (time.perf_counter() - mark) * 1000
    setup, recipe = plan.setup, plan.recipe
    radius = np.linalg.norm(points[:, :2] - np.array([setup.center_x_mm, setup.center_y_mm]), axis=1)
    max_z = layer.nominal_z_mm + recipe.bead_diameter_mm / 2 + config.deposit_height_margin_mm
    # The selected work frame defines the build plane at Z=0, so deterministic
    # height subtraction is more reproducible than fitting a new plane per frame.
    min_z = max(config.deposit_min_height_mm,
                config.plane_distance_threshold_m * 1000.0)
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
    floor = {"source": "build_plane", "margin_mm": 0.0, "mean_mm": float(min_z)}
    if floor_profile is not None and len(points):
        profile = np.asarray(floor_profile, dtype=float).reshape(-1, 3)
        _, nearest = cKDTree(profile[:, :2]).query(points[:, :2])
        local = profile[nearest, 2] + config.layer_floor_margin_mm
        points = points[points[:, 2] >= local]
        floor = {"source": "previous_layer_measured",
                 "margin_mm": float(config.layer_floor_margin_mm),
                 "mean_mm": float(local.mean())}
    counts["after_work_roi"] = len(points)
    if len(points) < config.cluster_min_points:
        raise RuntimeError(
            "not enough deposited-geometry points inside the configured work ROI "
            f"(need {config.cluster_min_points}); {json.dumps(roi_diag)}")

    mark = time.perf_counter()
    deposit = _filter_deposit(points, config, counts)
    # Before the crest is picked and before the bead width is read from the
    # flanks: both must see the bead alone, not the board fused to it.
    deposit = _radial_trim(deposit, getattr(config, "radial_trim_schedule_mm", ()), counts)
    points = _top_surface(deposit, config, counts)
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
    closed = len(_graph(final_skeleton)) > 2 and not any(
        len(v) == 1 for v in _graph(final_skeleton).values())
    measured = _fit_spline(ordered_xyz, config.measured_spline_points, closed=True)
    nominal_center = (setup.center_x_mm, setup.center_y_mm)
    metrics = compare_circle(measured, recipe.radius_mm,
                             nominal_center_mm=nominal_center)
    geometry = ring_geometry(measured, deposit, metrics.measured_center_mm,
                             floor_profile=floor_profile,
                             build_plane_z_mm=setup.build_plane_z_mm,
                             bins=config.bead_width_bins)
    corrected = None
    if recipe.correction_enabled and metrics.valid:
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


def characterize_ring(*, color: np.ndarray, depth: np.ndarray, T_work_camera: np.ndarray,
                      K: np.ndarray, search_center_mm, work_frame: str, config,
                      inspection_tool: str = "Realsense",
                      print_tool: str = "LongCalibTool") -> CharacterizationResult:
    """Measure a ring with NO recipe assumption: coarse fit, then the normal pipeline.

    Pass 1 takes everything above the build plane inside a search cylinder around
    ``search_center_mm``, filters it like a deposit, and fits a circle to get a
    coarse centre/radius/bead. Pass 2 hands those to ``process_observation`` as a
    throwaway recipe so the refined centreline, radius and height profile come out
    of the same code the layer measurements use -- one pipeline, one set of
    numbers, no second implementation to keep honest.
    """
    started = time.perf_counter()
    counts: dict[str, int] = {}
    points, counts["raw_depth_pixels"] = depth_to_work_points(depth, K, T_work_camera)
    center = np.asarray(search_center_mm, dtype=float)
    min_z = max(config.deposit_min_height_mm, config.plane_distance_threshold_m * 1000.0)
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
    refined = process_observation(color=color, depth=depth, T_work_camera=T_work_camera,
                                  K=K, plan=plan, layer=plan.layers[0], config=config)
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
