import socket
import struct
import subprocess
import threading
import traceback
import json
import math
import os
import time
from collections import deque, namedtuple
import pyrealsense2 as rs
import numpy as np
import lz4.frame as lz4f
import io
import turbojpeg
import scan_overlay  # pure-numpy live-rectangle trim + colour edge cross-check
from handshake import parse_handshake
import rs_config
import rs_geometry

ASFOUND_DIR = os.environ.get("RS_ASFOUND_DIR", "/home/jetson/robodk-characterization")

port = 1024
SCAN_COLOR_JPEG_QUALITY = 100
SCAN_TELEMETRY_PERIOD_S = 1.0

_telemetry_cond = threading.Condition()
_telemetry_seq = 0
_telemetry_payload = None


def publish_scan_telemetry(payload):
    global _telemetry_seq, _telemetry_payload
    with _telemetry_cond:
        _telemetry_payload = payload
        _telemetry_seq += 1
        _telemetry_cond.notify_all()


def center_connected_mask(mask):
    """Keep the 8-connected plane component crossing the image-center reticle."""
    src = np.asarray(mask, dtype=bool)
    h, w = src.shape
    # Bridge isolated invalid-depth pinholes without allowing broad expansion.
    padded = np.pad(src, 1, constant_values=False)
    neighbours = sum(
        padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    bridged = src | (neighbours >= 5)

    cy, cx = h // 2, w // 2
    ry, rx = max(2, h // 10), max(2, w // 10)
    seeds = np.argwhere(
        bridged[max(0, cy - ry):min(h, cy + ry + 1),
                max(0, cx - rx):min(w, cx + rx + 1)])
    if len(seeds):
        seeds[:, 0] += max(0, cy - ry)
        seeds[:, 1] += max(0, cx - rx)
    else:
        all_pts = np.argwhere(bridged)
        if not len(all_pts):
            return np.zeros_like(src)
        nearest = all_pts[np.argmin(
            (all_pts[:, 0] - cy) ** 2 + (all_pts[:, 1] - cx) ** 2)]
        seeds = nearest.reshape(1, 2)

    out = np.zeros_like(src)
    q = deque()
    for y, x in seeds:
        if not out[y, x]:
            out[y, x] = True
            q.append((int(y), int(x)))
    while q:
        y, x = q.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dx or dy):
                    continue
                yy, xx = y + dy, x + dx
                if (0 <= yy < h and 0 <= xx < w
                        and bridged[yy, xx] and not out[yy, xx]):
                    out[yy, xx] = True
                    q.append((yy, xx))
    return out


def convex_hull_2d(points):
    """Monotone-chain hull for visible plane pixels in color-image coordinates."""
    pts = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                cross = ((b[0] - a[0]) * (p[1] - a[1])
                         - (b[1] - a[1]) * (p[0] - a[0]))
                if cross <= 0:
                    out.pop()
                else:
                    break
            out.append(p)
        return out

    return np.asarray(half(pts)[:-1] + half(pts[::-1])[:-1])


def min_area_rectangle_2d(points, preferred_axis=None, area_tolerance=0.02):
    """Return a stable minimum-area rectangle around 2-D plane coordinates.

    The visible depth silhouette can contain notches where IR depth is missing.
    Rotating calipers keeps those notches from becoming bends in the work-region
    overlay. Near-equal solutions (common for square tops) are resolved toward the
    camera image X axis so the rectangle does not jump by 45/90 degrees.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    hull = convex_hull_2d(pts)
    if len(hull) < 3:
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return (np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                float(hi[0] - lo[0]), float(hi[1] - lo[1]),
                float(lo[0]), float(lo[1]))

    candidates = []
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        length = float(np.linalg.norm(edge))
        if length < 1e-9:
            continue
        ux = edge / length
        uy = np.array([-ux[1], ux[0]])
        px, py = pts @ ux, pts @ uy
        width = float(px.max() - px.min())
        height = float(py.max() - py.min())
        candidates.append((
            width * height, ux, uy, width, height,
            float(px.min()), float(py.min())))

    min_area = min(c[0] for c in candidates)
    near_best = [
        c for c in candidates
        if c[0] <= min_area * (1.0 + max(0.0, float(area_tolerance)))
    ]
    pref = None if preferred_axis is None else np.asarray(
        preferred_axis, dtype=float).reshape(2)
    if pref is not None and float(np.linalg.norm(pref)) > 1e-9:
        pref /= np.linalg.norm(pref)
        best = max(near_best, key=lambda c: max(
            abs(float(c[1] @ pref)), abs(float(c[2] @ pref))))
    else:
        best = min(candidates, key=lambda c: c[0])
    _, ux, uy, width, height, lo_x, lo_y = best
    return ux, uy, width, height, lo_x, lo_y


def footprint_edge_angle_deg(hull):
    """Orientation of the longest visible footprint edge relative to image X."""
    h = np.asarray(hull, float).reshape(-1, 2)
    if len(h) < 2:
        return None
    edges = np.roll(h, -1, axis=0) - h
    lengths = np.linalg.norm(edges, axis=1)
    edge = edges[int(np.argmax(lengths))]
    angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
    # Rectangle edges are unoriented: normalize to the smallest correction that
    # makes the dominant edge horizontal.
    return ((angle + 45.0) % 90.0) - 45.0


def deproject_pixels(pixels, depths_mm, intrinsics, exact_deproject=None):
    """Back-project pixels into the depth-camera frame."""
    uv = np.asarray(pixels, dtype=float).reshape(-1, 2)
    z = np.asarray(depths_mm, dtype=float).reshape(-1)
    if exact_deproject is not None:
        return np.asarray(exact_deproject(uv, z), dtype=float).reshape(-1, 3)
    fx, fy = float(intrinsics.fx), float(intrinsics.fy)
    cx, cy = float(intrinsics.ppx), float(intrinsics.ppy)
    return np.column_stack([
        (uv[:, 0] - cx) / fx * z,
        (uv[:, 1] - cy) / fy * z,
        z,
    ])


def fit_nearest_plane(points, *, distance_mm=5.0, min_inlier_frac=0.12,
                      iterations=160, seed=7):
    """Return the nearest coherent plane instead of mixing stacked surfaces."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n = len(pts)
    minimum = max(24, int(np.ceil(n * float(min_inlier_frac))))
    if n < minimum:
        raise ValueError("not enough depth points for a coherent plane")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(int(iterations)):
        a, b, c = pts[rng.choice(n, 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal /= norm
        residual = np.abs((pts - a) @ normal)
        mask = residual <= float(distance_mm)
        count = int(mask.sum())
        if count < minimum:
            continue
        candidate = (float(np.median(pts[mask, 2])), -count, mask)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise ValueError("no coherent depth plane found")
    mask = best[2]
    centroid = pts[mask].mean(axis=0)
    _, _, vt = np.linalg.svd(pts[mask] - centroid, full_matrices=False)
    normal = vt[2] / max(float(np.linalg.norm(vt[2])), 1e-9)
    if normal[2] > 0:
        normal = -normal
    residual = np.abs((pts - centroid) @ normal)
    mask = residual <= float(distance_mm)
    if int(mask.sum()) < minimum:
        raise ValueError("nearest depth plane was unstable after refinement")
    centroid = pts[mask].mean(axis=0)
    return normal, centroid, mask


# Generic work-square size (mm) drawn on the plane, centred on the reticle, when the
# surface overruns the view (its real edges are off-frame, so a fitted board rectangle
# would over-run the table). Matches the workstation's scan.work_crop_mm default so the
# live overlay and the locked/inserted box agree.
WORK_CROP_MM = 1000.0


def scan_plane_telemetry(depth, intrinsics, depth_unit_mm=1.0,
                         patch_frac=0.25, min_valid_frac=0.25,
                         overlay_project=None, overlay_project_points=None,
                         overlay_transform_points=None, overlay_size=None,
                         depth_deproject_points=None, color_image=None):
    """Fit the central depth patch locally and return compact live-guidance data."""
    d = np.asarray(depth)
    h, w = d.shape[:2]
    cw, ch = max(4, int(w * patch_frac)), max(4, int(h * patch_frac))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    patch = d[y0:y0 + ch, x0:x0 + cw].astype(np.float64)
    valid = patch > 0
    valid_frac = float(valid.mean())
    if valid_frac < min_valid_frac:
        return {"detected": False, "valid_frac": valid_frac,
                "timestamp": time.time()}

    ys, xs = np.nonzero(valid)
    z_mm = patch[valid] * float(depth_unit_mm)
    if len(z_mm) > 3000:
        step = int(np.ceil(len(z_mm) / 3000.0))
        ys, xs, z_mm = ys[::step], xs[::step], z_mm[::step]
    u, v = xs + x0, ys + y0
    fx, fy = float(intrinsics.fx), float(intrinsics.fy)
    cx, cy = float(intrinsics.ppx), float(intrinsics.ppy)
    pts = deproject_pixels(
        np.column_stack([u, v]), z_mm, intrinsics, depth_deproject_points)
    try:
        normal, centroid, center_inliers = fit_nearest_plane(pts)
    except ValueError:
        return {"detected": False, "valid_frac": valid_frac,
                "timestamp": time.time()}
    selected_pts = pts[center_inliers]
    selected_z = selected_pts[:, 2]
    nx, ny, nz = [float(x) for x in normal]
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(nz), 0.0, 1.0))))
    denom = max(-nz, 1e-9)
    payload = {
        "detected": True,
        "distance_mm": float(np.median(selected_z)),
        "tilt_deg": tilt_deg,
        "tilt_b_deg": float(np.degrees(np.arctan2(nx, denom))),
        "tilt_c_deg": float(np.degrees(np.arctan2(ny, denom))),
        "normal_cam": [nx, ny, nz],
        "centroid_cam_mm": [float(x) for x in centroid],
        "valid_frac": valid_frac,
        "timestamp": time.time(),
    }
    center_residual = np.abs((selected_pts - centroid) @ normal)
    # Derive a tight expansion band from the selected top plane. The low cap keeps
    # a lower lip or support layer from joining the upper footprint.
    # center patch instead of clipping the same plane at a fixed ±8 mm.
    plane_tolerance_mm = float(np.clip(
        np.quantile(center_residual, 0.98) + 1.5, 3.0, 7.0))
    payload["plane_tolerance_mm"] = plane_tolerance_mm

    # Expand the center-selected plane across a sparse full-frame sample. This is
    # intentionally anchored to the reticle plane rather than "largest plane wins":
    # floor/walls and objects cannot steal the live work-surface measurement.
    stride = max(2, int(round(max(h, w) / 220.0)))
    sampled = d[::stride, ::stride].astype(np.float64)
    sy, sx = np.nonzero(sampled > 0)
    if len(sy) >= 50:
        au, av = sx * stride, sy * stride
        az = sampled[sy, sx] * float(depth_unit_mm)
        all_pts = deproject_pixels(
            np.column_stack([au, av]), az, intrinsics, depth_deproject_points)
        inlier = np.abs((all_pts - centroid) @ normal) < plane_tolerance_mm
        plane_mask = np.zeros(sampled.shape, dtype=bool)
        plane_mask[sy[inlier], sx[inlier]] = True
        connected = center_connected_mask(plane_mask)
        keep = connected[sy, sx] & inlier
        if int(keep.sum()) >= 50:
            ip = all_pts[keep]
            iu, iv = au[keep], av[keep]
            pc = ip.mean(axis=0)
            rel = ip - pc
            _, _, plane_axes = np.linalg.svd(rel, full_matrices=False)
            basis_u = plane_axes[0] - float(plane_axes[0] @ normal) * normal
            basis_u /= max(float(np.linalg.norm(basis_u)), 1e-9)
            basis_v = np.cross(normal, basis_u)
            plane_coords = np.column_stack([rel @ basis_u, rel @ basis_v])
            # Remove only isolated fringe points before fitting the rectangle.
            # Missing depth inside the footprint then cannot bend its four edges.
            if len(plane_coords) >= 100:
                trim_lo = np.quantile(plane_coords, 0.002, axis=0)
                trim_hi = np.quantile(plane_coords, 0.998, axis=0)
                trimmed = np.all(
                    (plane_coords >= trim_lo) & (plane_coords <= trim_hi), axis=1)
                if int(trimmed.sum()) >= max(50, int(0.9 * len(plane_coords))):
                    fit_coords = plane_coords[trimmed]
                else:
                    fit_coords = plane_coords
            else:
                fit_coords = plane_coords
            camera_x_in_plane = np.array([
                float(np.array([1.0, 0.0, 0.0]) @ basis_u),
                float(np.array([1.0, 0.0, 0.0]) @ basis_v),
            ])
            rect_u, rect_v, len1, len2, lo1, lo2 = min_area_rectangle_2d(
                fit_coords, preferred_axis=camera_x_in_plane)
            ax1 = basis_u * rect_u[0] + basis_v * rect_u[1]
            ax2 = basis_u * rect_v[0] + basis_v * rect_v[1]
            margin = max(2 * stride, 10)
            depth_fully_framed = not (
                bool(np.any(iu < margin)) or bool(np.any(iu > w - 1 - margin)) or
                bool(np.any(iv < margin)) or bool(np.any(iv > h - 1 - margin)))
            fully_framed = depth_fully_framed
            corners = [
                pc + lo1 * ax1 + lo2 * ax2,
                pc + (lo1 + len1) * ax1 + lo2 * ax2,
                pc + (lo1 + len1) * ax1 + (lo2 + len2) * ax2,
                pc + lo1 * ax1 + (lo2 + len2) * ax2,
            ]
            corners_array = np.asarray(corners, dtype=float)
            corners_color = (
                np.asarray(overlay_transform_points(corners_array), dtype=float)
                if overlay_transform_points is not None else None)
            overlay_w, overlay_h = overlay_size or (w, h)
            if overlay_project_points is not None:
                projected = np.asarray(overlay_project_points(ip), float)
                # The sparse point hull is diagnostic only, so its vectorized
                # pinhole projection is acceptable. The four work-rectangle
                # corners drive the solid blue operator overlay and must use the
                # RealSense distortion model when that projector is available.
                if overlay_project is not None:
                    projected_corners = np.asarray(
                        [overlay_project(p) for p in corners], float)
                else:
                    projected_corners = np.asarray(
                        overlay_project_points(np.asarray(corners)), float)
            elif overlay_project is not None:
                projected = np.asarray([overlay_project(p) for p in ip], float)
                projected_corners = np.asarray(
                    [overlay_project(p) for p in corners], float)
            else:
                projected = np.column_stack([
                    ip[:, 0] * fx / ip[:, 2] + cx,
                    ip[:, 1] * fy / ip[:, 2] + cy,
                ])
                corner_array = np.asarray(corners)
                projected_corners = np.column_stack([
                    corner_array[:, 0] * fx / corner_array[:, 2] + cx,
                    corner_array[:, 1] * fy / corner_array[:, 2] + cy,
                ])
            projected_uv = projected / np.array([overlay_w, overlay_h])
            rectangle_uv = projected_corners / np.array([overlay_w, overlay_h])
            finite = np.all(np.isfinite(projected_uv), axis=1)
            visible_hull = convex_hull_2d(projected_uv[finite])
            raw_outline = rectangle_uv.tolist()
            # The solid guidance polygon is the plane-space work rectangle. The
            # raw pixel hull is sent separately so missing depth remains visible
            # diagnostically without deforming the selected region.
            outline = np.clip(rectangle_uv, 0.0, 1.0).tolist()
            visible_outline = np.clip(visible_hull, 0.0, 1.0).tolist()
            edge_angle_deg = footprint_edge_angle_deg(rectangle_uv)
            if len(raw_outline) >= 3:
                # "Framed" must describe what the operator sees in the COLOR
                # preview, not merely what fits in the wider raw depth FOV.
                color_margin = 0.015
                fully_framed = fully_framed and all(
                    color_margin <= uv[0] <= 1.0 - color_margin
                    and color_margin <= uv[1] <= 1.0 - color_margin
                    for uv in raw_outline)
                # Required standoff is continuous across the framed boundary:
                # projected half-span × current distance. The workstation applies
                # its configured margin and clamps to the RealSense quality band.
                max_center_span = max(
                    max(abs(float(uv[0]) - 0.5), abs(float(uv[1]) - 0.5))
                    for uv in raw_outline)
                color_fit_standoff_per_margin_mm = (
                    float(np.median(selected_z)) * 2.0 * max_center_span)
            else:
                color_fit_standoff_per_margin_mm = None

            # --- DISPLAY rectangle: density-cliff trim + colour edge confirmation ---
            # The raw rectangle above still drives EVERY gate (framing, standoff,
            # corners) unchanged. Only the operator's overlay box is tightened: pull
            # each edge in to the dense body so it hugs the real surface instead of a
            # sparse coplanar halo just past the edge, then VETO any edge whose trim
            # the colour image does not corroborate (uniform colour across the cliff =
            # a depth hole on a continuing surface, not a real edge — so do not cut it).
            pa = plane_coords @ rect_u
            pb = plane_coords @ rect_v
            raw_hi1, raw_hi2 = lo1 + len1, lo2 + len2
            tlo1, thi1 = scan_overlay.density_extent_1d(pa, lo1, raw_hi1)
            tlo2, thi2 = scan_overlay.density_extent_1d(pb, lo2, raw_hi2)

            def _edge_intensity(points3d):
                """Mean colour-image intensity at the projection of plane points
                (empty on any failure -> the colour veto simply abstains)."""
                if color_image is None or overlay_project_points is None:
                    return np.empty(0)
                try:
                    px = np.rint(np.asarray(overlay_project_points(points3d), float))
                    hc, wc = color_image.shape[:2]
                    ok = (np.all(np.isfinite(px), axis=1)
                          & (px[:, 0] >= 0) & (px[:, 0] < wc)
                          & (px[:, 1] >= 0) & (px[:, 1] < hc))
                    if not bool(ok.any()):
                        return np.empty(0)
                    idx = px[ok].astype(int)
                    s = color_image[idx[:, 1], idx[:, 0]]
                    return s.mean(axis=1) if s.ndim == 2 else s.astype(float)
                except Exception:
                    return np.empty(0)

            def _confirm(raw_edge, cand, ax_a, ax_c, c_lo, c_hi, sign):
                if abs(cand - raw_edge) < 1.0:
                    return cand                       # nothing trimmed on this side
                inset = 10.0
                inside = _edge_intensity(scan_overlay.side_sample_points(
                    pc, ax_a, ax_c, cand - sign * inset, c_lo, c_hi))
                outside = _edge_intensity(scan_overlay.side_sample_points(
                    pc, ax_a, ax_c, 0.5 * (cand + raw_edge), c_lo, c_hi))
                return raw_edge if scan_overlay.edge_continues(inside, outside) else cand

            thi1 = _confirm(raw_hi1, thi1, ax1, ax2, tlo2, thi2, +1.0)
            tlo1 = _confirm(lo1, tlo1, ax1, ax2, tlo2, thi2, -1.0)
            thi2 = _confirm(raw_hi2, thi2, ax2, ax1, tlo1, thi1, +1.0)
            tlo2 = _confirm(lo2, tlo2, ax2, ax1, tlo1, thi1, -1.0)
            tlen1, tlen2 = thi1 - tlo1, thi2 - tlo2

            trimmed_corners = [
                pc + tlo1 * ax1 + tlo2 * ax2,
                pc + thi1 * ax1 + tlo2 * ax2,
                pc + thi1 * ax1 + thi2 * ax2,
                pc + tlo1 * ax1 + thi2 * ax2,
            ]
            # The trimmed corners in the COLOR-camera frame (mm), so the workstation
            # can re-project them through its RealSense calibration and draw the LIVE
            # overlay as the same density/colour-trimmed box the lock/insert uses.
            # (rectangle_corners_color_mm carries the RAW corners for the framing test.)
            trimmed_corners_color = (
                np.asarray(overlay_transform_points(np.asarray(trimmed_corners, float)),
                           dtype=float)
                if overlay_transform_points is not None else None)
            if overlay_project is not None:
                tpc = np.asarray([overlay_project(p) for p in trimmed_corners], float)
            elif overlay_project_points is not None:
                tpc = np.asarray(
                    overlay_project_points(np.asarray(trimmed_corners)), float)
            else:
                tca = np.asarray(trimmed_corners)
                tpc = np.column_stack([tca[:, 0] * fx / tca[:, 2] + cx,
                                       tca[:, 1] * fy / tca[:, 2] + cy])
            trimmed_outline = np.clip(
                tpc / np.array([overlay_w, overlay_h]), 0.0, 1.0).tolist()

            # When the surface overruns the view (not depth_fully_framed), its real
            # edges are off-frame, so the board rectangle above would over-run the
            # table. Draw a GENERIC fixed work square on the plane, centred on the
            # reticle (the aim point), projected to colour — matching the host lock/run
            # crop. Fully-framed surfaces keep the measured (trimmed) board rectangle.
            crop_outline = None
            if not depth_fully_framed:
                sq = scan_overlay.reticle_plane_square(
                    normal, pc, (WORK_CROP_MM, WORK_CROP_MM))[0]
                if overlay_project is not None:
                    sq_px = np.asarray([overlay_project(p) for p in sq], float)
                elif overlay_project_points is not None:
                    sq_px = np.asarray(overlay_project_points(sq), float)
                else:
                    sq_px = np.column_stack([sq[:, 0] * fx / sq[:, 2] + cx,
                                             sq[:, 1] * fy / sq[:, 2] + cy])
                sq_uv = sq_px / np.array([overlay_w, overlay_h])
                if np.all(np.isfinite(sq_uv)):
                    # Sent UN-clipped (unlike the in-frame board outline): the 1 m
                    # square overruns the view, so the browser SVG clips it for display
                    # while the corners keep true edge directions (a per-corner clamp
                    # would bend the box). Matches the host survey's unclipped square.
                    crop_outline = sq_uv.tolist()

            xlo, xhi = np.quantile(ip[:, 0], [0.005, 0.995])
            ylo, yhi = np.quantile(ip[:, 1], [0.005, 0.995])
            # Detected-surface DOTS for the HUD: the ACTUAL measured surface points
            # (where depth truly landed), snapped to a FIXED image grid and sent as
            # the occupied cells. `projected_uv[finite]` is exactly the inlier cloud
            # `ip` in screen position, so each dot marks a real measurement — not an
            # idealized cell center derived from a per-frame surface estimate (which
            # drifts with depth noise, so its dots slide apart between frames and an
            # accumulated union never overlaps). The grid is fixed in the IMAGE, so a
            # steady camera yields steady dots; a cell appears only if a real point
            # fell in it, so an empty cell is a genuine coverage hole. Each frame
            # carries that frame's own dropouts, which is what lets the frontend's
            # multi-frame union fill stochastic stereo gaps. `np.unique` bounds the
            # count; a coarse stride caps telemetry for very large surfaces.
            real_uv = projected_uv[finite]
            if len(real_uv):
                in_frame = np.all((real_uv >= 0.0) & (real_uv <= 1.0), axis=1)
                real_uv = real_uv[in_frame]
            if len(real_uv):
                GRID = 180  # matches the frontend's coverage-dedupe resolution
                cells = np.unique(np.floor(real_uv * GRID).astype(int), axis=0)
                if len(cells) > 4000:
                    cells = cells[:: int(np.ceil(len(cells) / 4000.0))]
                dot_uv = (cells + 0.5) / float(GRID)
                surface_points_uv = np.round(dot_uv, 4).tolist()
            else:
                surface_points_uv = None
            payload.update({
                "points_uv": surface_points_uv,
                "fully_framed": fully_framed,
                "depth_fully_framed": depth_fully_framed,
                "surface_mode": "full" if depth_fully_framed else "crop",
                "extent_mm": [max(len1, len2), min(len1, len2)],
                # Physical lengths of the DISPLAY rectangle (edges 0->1, 1->2): the
                # generic crop square when the surface overruns the view, else the
                # trimmed real board. extent_mm stays raw for framing/planning.
                "rectangle_size_mm": ([WORK_CROP_MM, WORK_CROP_MM]
                                      if not depth_fully_framed
                                      else [float(tlen1), float(tlen2)]),
                "rectangle_corners_color_mm": (
                    corners_color.tolist() if corners_color is not None else None),
                "trimmed_corners_color_mm": (
                    trimmed_corners_color.tolist()
                    if trimmed_corners_color is not None else None),
                # Operator overlay: the generic reticle-centred work square when the
                # surface overruns the view (edges off-frame), else the density/colour-
                # trimmed board box (hugs the surface). Falls back to the raw outline
                # if trimming/projection degenerated.
                "outline_uv": (crop_outline
                               if crop_outline is not None and len(crop_outline) >= 3
                               else trimmed_outline if len(trimmed_outline) >= 3
                               else (outline if len(outline) >= 3 else None)),
                "visible_outline_uv": (
                    visible_outline if len(visible_outline) >= 3 else None),
                "surface_center_cam_mm": [float(x) for x in pc],
                "surface_center_uv": (
                    [float(np.mean(projected_uv[finite, 0])),
                     float(np.mean(projected_uv[finite, 1]))]
                    if int(finite.sum()) else None),
                "edge_angle_deg": edge_angle_deg,
                # Multiply this by the workstation's configured frame margin to
                # obtain the standoff needed to fit this plane in the depth FOV.
                "fit_standoff_per_margin_mm": float(max(
                    (xhi - xlo) * fx / w,
                    (yhi - ylo) * fy / h)),
                "color_fit_standoff_per_margin_mm":
                    color_fit_standoff_per_margin_mm,
            })
    return payload


def stream_telemetry(conn, addr):
    """Send length-prefixed JSON snapshots produced by the active scan video loop."""
    seq = -1
    try:
        while True:
            with _telemetry_cond:
                _telemetry_cond.wait_for(
                    lambda: _telemetry_seq != seq and _telemetry_payload is not None,
                    timeout=2.0)
                payload = _telemetry_payload
                seq = _telemetry_seq
            if payload is None:
                continue
            data = json.dumps(payload, separators=(',', ':')).encode('utf-8')
            conn.sendall(struct.pack('<I', len(data)) + data)
    except (ConnectionResetError, BrokenPipeError, socket.timeout, OSError) as e:
        print(f"Telemetry connection to {addr} ended: {e}")

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


# --- Camera supervision -----------------------------------------------------
#
# ``pipeline`` is built once at startup and shared by every client thread, and
# until now nothing ever rebuilt it. On 2026-08-29 the D435i
# streamed for two healthy hours and then stopped delivering at 14:44:04; from
# that moment EVERY acquisition raised "Frame didn't arrive within 5000" -- 57 in
# a row -- while the service stayed ``active``, kept LISTENING on 1024 and kept
# accepting clients. The operator saw an app that connected normally and then
# showed nothing, with no error anywhere near the UI.
#
# All acquisition now funnels through ``read_frames()``, so one stall detector
# sees the timeouts from all client threads. It rides out a brief stall, rebuilds
# the pipeline once for a persistent one, and if it cannot, stops serving instead
# of feeding clients nothing forever.

class CameraUnavailable(RuntimeError):
    """The camera is not delivering frames and could not be recovered."""


# Below this, a stall is just a dropped frameset -- librealsense recovers from
# those on its own, and a rebuild costs seconds and re-opens the USB device.
FRAME_WEDGE_THRESHOLD = 3

# Reopen attempts, and the pause before each one. Bounded and increasing on
# purpose: re-opening a USB device in a tight loop is how a Tegra host controller
# ends up dead ("tegra-xusb: HC died; cleaning up", same cell, same day).
RECOVERY_BACKOFF_S = (1.0, 5.0, 15.0)

# A device can enumerate cleanly and still deliver nothing. Treating the reopen
# itself as success would loop -- stall, rebuild, stall, rebuild -- re-opening the
# USB device every ~20 s for as long as the service runs, which is the hammering
# the backoff above exists to prevent. So a rebuild is only credited once frames
# actually flow again, and only so many uncredited rebuilds are allowed.
MAX_REBUILDS_WITHOUT_PROGRESS = 3
HEALTHY_FRAMES_AFTER_REBUILD = 30      # ~1 s of streaming at 30 fps

pipeline = None          # replaced at startup, and on every successful rebuild
depth_unit_mm = None
depth_filters = None

_camera_lock = threading.RLock()   # one camera behind many threads
_camera_generation = 0             # bumped on every successful rebuild
_consecutive_timeouts = 0
_rebuilds_without_progress = 0
_frames_since_rebuild = 0
_camera_unavailable_reason = None


def _recovery_sleep(seconds):
    """The backoff pause, as a seam so tests do not actually wait."""
    time.sleep(seconds)


def _reset_camera_state():
    """Return the supervisor to its healthy startup state (test seam)."""
    global _camera_generation, _consecutive_timeouts, _camera_unavailable_reason
    global _rebuilds_without_progress, _frames_since_rebuild
    with _camera_lock:
        _camera_generation = 0
        _consecutive_timeouts = 0
        _rebuilds_without_progress = 0
        _frames_since_rebuild = 0
        _camera_unavailable_reason = None


def _is_frame_timeout(exc):
    """True only for librealsense's "Frame didn't arrive within N".

    Anything else -- "No device connected", a genuine bug -- must propagate.
    Rebuilding on those would spin until the process is killed.
    """
    return "didn't arrive" in str(exc)


def _give_up(reason):
    """Stop pretending to be a camera.

    Exiting is deliberate. The process is holding a listening port and client
    sockets it can no longer serve, and ``Restart=always``/``RestartSec=3`` gives
    a clean re-open attempt from scratch. If the camera really is gone the unit
    then loops on "No device connected" -- a visible, diagnosable state, unlike
    silently accepting clients forever.
    """
    print(f"CAMERA UNAVAILABLE: {reason}", flush=True)
    print("Exiting so systemd restarts the service and re-opens the device.",
          flush=True)
    os._exit(1)


def _release_pipeline():
    """Stop the wedged pipeline so the device can be re-opened.

    Guarded because the teardown is itself the dangerous part: on 2026-08-29
    stopping a wedged D435i took the whole USB host controller down with it and
    the process died on SIGSEGV. There is no way to re-open without releasing
    first, so the best available behaviour is to try and to carry on if it throws.
    """
    old = pipeline
    if old is None:
        return
    try:
        old.stop()
    except Exception as exc:
        print(f"  (releasing the wedged pipeline raised {exc!r}; continuing)",
              flush=True)


def _reread_depth_unit_mm(new_pipeline, previous):
    """The REOPENED device's millimetres-per-count. Never raises, never silent.

    Protocol 2 ships raw depth counts, so this one number is the scale factor every
    host measurement is multiplied by: get it wrong and every scan, gate reading and
    extrusion measurement is off by a constant factor, with nothing anywhere in the
    chain able to notice. ``configure_depth_sensor`` SETS ``depth_units`` on every
    open and reads the achieved value back, so it is a per-open fact -- it cannot
    simply be assumed to carry over from the open that just died.

    Three sources, in falling order of directness:

    1. the reopened device, asked here;
    2. ``ACHIEVED_OPTIONS["depth_unit_mm"]`` -- which ``openPipeline`` read off that
       SAME reopened device moments ago (and which the rebuild has already rebound,
       being a side effect of ``openPipeline``). This is the fallback that used to
       be missing: the old code swallowed the failure with "same device; the startup
       value still holds" and kept the module global, which is the PREVIOUS open's
       number, not this one's;
    3. the previous value, only when neither read worked -- and said out loud, so a
       possibly-stale scale factor appears in the journal instead of nowhere.

    Never raising is deliberate: the unit is ``Restart=always`` with no start limit,
    so an exception on the recovery path is an infinite crash-loop with the camera
    dark for every module.
    """
    try:
        return float(new_pipeline.get_active_profile().get_device()
                     .first_depth_sensor().get_depth_scale() * 1000.0)
    except Exception as exc:            # noqa: BLE001 - see the docstring
        print(f"WARNING: could not re-read the depth scale off the reopened camera "
              f"({exc!r}); depth_unit_mm scales EVERY measurement the host makes.",
              flush=True)
    try:
        fallback = float(ACHIEVED_OPTIONS.get("depth_unit_mm"))
    except (AttributeError, TypeError, ValueError):
        # openPipeline could not read it either (it stores None then), or left
        # something unexpected behind.
        print(f"  the reopened device reported no depth scale either; KEEPING "
              f"{previous} mm/count from the previous open -- verify it before "
              f"trusting a measurement from this camera.", flush=True)
        return previous
    print(f"  using the value openPipeline read back off the reopened device: "
          f"depth_unit_mm = {fallback}", flush=True)
    return fallback


def _rebuild_pipeline(observed_generation):
    """Re-open the camera once, on behalf of every waiting client thread."""
    global pipeline, depth_unit_mm
    global _camera_generation, _consecutive_timeouts, _camera_unavailable_reason
    global _rebuilds_without_progress, _frames_since_rebuild

    with _camera_lock:
        if _camera_unavailable_reason is not None:
            raise CameraUnavailable(_camera_unavailable_reason)
        if _camera_generation != observed_generation:
            return          # another thread already rebuilt it; just retry

        print("Camera stalled; rebuilding the RealSense pipeline...", flush=True)
        last_error = None
        for attempt, pause in enumerate(RECOVERY_BACKOFF_S, start=1):
            _recovery_sleep(pause)
            try:
                _release_pipeline()
                new_pipeline = openPipeline()
            except Exception as exc:
                last_error = exc
                print(f"  reopen attempt {attempt}/{len(RECOVERY_BACKOFF_S)} "
                      f"failed: {exc}", flush=True)
                continue

            pipeline = new_pipeline
            was = depth_unit_mm
            depth_unit_mm = _reread_depth_unit_mm(new_pipeline, depth_unit_mm)
            if depth_unit_mm != was:
                # A silent change of the scale factor across a rebuild is exactly
                # the failure this whole function exists to make visible. (Clients
                # greeted before the rebuild are dropped by _greeting_is_stale, so
                # none of them keeps quoting the old number.)
                print(f"  depth_unit_mm changed across the rebuild: {was} -> "
                      f"{depth_unit_mm} mm/count", flush=True)
            _camera_generation += 1
            _consecutive_timeouts = 0
            _rebuilds_without_progress += 1
            _frames_since_rebuild = 0
            print(f"Camera re-opened (generation {_camera_generation}); "
                  f"waiting for frames to confirm recovery.", flush=True)

            if _rebuilds_without_progress > MAX_REBUILDS_WITHOUT_PROGRESS:
                _camera_unavailable_reason = (
                    f"camera re-opens but never streams "
                    f"({_rebuilds_without_progress} rebuilds, no frames between)")
                _give_up(_camera_unavailable_reason)
                raise CameraUnavailable(_camera_unavailable_reason)
            return

        _camera_unavailable_reason = str(last_error)
        _give_up(f"pipeline could not be re-opened after "
                 f"{len(RECOVERY_BACKOFF_S)} attempts: {last_error}")
        raise CameraUnavailable(_camera_unavailable_reason)


def read_frames():
    """Wait for a frameset, recovering the pipeline if the camera has wedged.

    Every acquisition path in the server goes through here, so the stall counter
    reflects the camera rather than any one client. Raises ``CameraUnavailable``
    once the camera is known to be gone, so callers fail immediately instead of
    each paying a fresh round of 5-second timeouts.
    """
    global _consecutive_timeouts, _rebuilds_without_progress, _frames_since_rebuild

    while True:
        with _camera_lock:
            if _camera_unavailable_reason is not None:
                raise CameraUnavailable(_camera_unavailable_reason)
            generation = _camera_generation
            current = pipeline

        try:
            frames = current.wait_for_frames()
        except RuntimeError as exc:
            if not _is_frame_timeout(exc):
                raise
            with _camera_lock:
                _consecutive_timeouts += 1
                wedged = _consecutive_timeouts >= FRAME_WEDGE_THRESHOLD
            if wedged:
                _rebuild_pipeline(generation)
            continue

        with _camera_lock:
            _consecutive_timeouts = 0
            # Credit a rebuild only once the camera has actually streamed for a
            # while, so a flapping device cannot earn a fresh set of attempts by
            # delivering a single frame between stalls.
            if _rebuilds_without_progress:
                _frames_since_rebuild += 1
                if _frames_since_rebuild >= HEALTHY_FRAMES_AFTER_REBUILD:
                    print(f"Camera streaming again after "
                          f"{_rebuilds_without_progress} rebuild(s).", flush=True)
                    _rebuilds_without_progress = 0
                    _frames_since_rebuild = 0
        return frames


def getFrames(depth_filters):
    """One native depth frame (filtered) + the colour frame. NOT aligned: the host
    back-projects with the depth intrinsics it received in the greeting."""
    frames = read_frames()
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    if not depth or not color:
        return None, None, None
    for f in depth_filters:
        depth = f.process(depth)
    return (np.asanyarray(depth.get_data()), np.asanyarray(color.get_data()),
            frames.get_timestamp())

DEPTH_SIZE = (1280, 720)      # the D435i's top depth mode
COLOR_SIZE = (1920, 1080)     # audit R7: ChArUco corner precision bounds every downstream number


# Depth acquisition tuning. Overridable per-cell via the service environment so a
# change can be trialled without editing code: RS_VISUAL_PRESET (rs400 enum ordinal
# -- 3 high_accuracy, 4 high_density, 5 medium_density) and RS_LASER_POWER (mW-ish,
# 0..360 on a D435i).
#
# Measured on the cell 2026-08-13, against the printed ChArUco ruler on an A3 panel
# 630 mm above the floor: valid depth stopped ~20 mm short of the panel's top and
# bottom edges (left/right were within 0.3-2.7 mm). The stereo matcher needs
# texture, and the IR projector is what supplies it on blank surfaces -- it was
# running at the 150 default, only 42% of this device's 360 maximum.
# Both knobs are deliberately LEFT ALONE by default (-1 = don't touch). Today's
# distance characterization was measured under the preset the device is actually
# running, so silently changing it would invalidate that dated record; and High
# Accuracy in particular is the wrong direction here (it raises the confidence
# threshold, returning fewer but surer points, when the measured defect is missing
# coverage). Set RS_VISUAL_PRESET=4 to trial high_density as a SEPARATE experiment,
# with its own before/after measurement.
def _env_number(name: str, default: float) -> float:
    """A numeric env override, or ``default`` when unset OR unparsable. A
    typo'd value must never take the service down: the unit is Restart=always
    with no start limit, so an import-time ValueError becomes an infinite
    crash-loop with the camera dark for every module."""
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        print(f"WARNING: {name}={raw!r} is not a number — using {default:g}",
              flush=True)
        return float(default)


RS_VISUAL_PRESET = int(_env_number('RS_VISUAL_PRESET', -1))
# -1 = leave the device's current laser power alone. The 2026-08-13 depth
# characterization was measured at the device's 150; a silent default of 300
# doubled projector power on every restart and invalidated that dated envelope
# (the same invariant that keeps RS_VISUAL_PRESET at leave-alone). Trial higher
# power as an explicit experiment with its own before/after measurement.
RS_LASER_POWER = _env_number('RS_LASER_POWER', -1.0)

# -- spatial filter, for the crest-height A/B ---------------------------------
# The chain runs the spatial filter in the DISPARITY domain with stock defaults
# (magnitude 2, smooth_alpha 0.5, smooth_delta 20). smooth_delta is the
# edge-preservation threshold -- steps larger than it survive, smaller ones are
# smoothed across. At 300 mm with depth fx 637 and the D435i's ~50 mm baseline,
# 1 mm of relief is only ~0.35 disparity px, so a 4-11 mm extrusion ring spans
# 1.4-3.9: far under 20. The filter cannot tell the deposit from the board and
# smooths across it, twice, at alpha 0.5.
#
# Measured 2026-08-31 against the operator's ruler at five clock positions on one
# ring: the camera reproduces the ring's shape well (r = 0.97 on absolute height)
# but reads ~1.5 mm LOW at every position, even taking each wedge's single
# highest point. That shortfall eats most of a 1.78 mm deposit floor, which is
# how a continuous ring came to be reported open.
#
# Both levers DEFAULT TO CURRENT BEHAVIOUR. Every number in the archive was
# measured with the stock filter, and a silent change would invalidate all of
# them -- the same invariant RS_LASER_POWER and RS_VISUAL_PRESET are kept under.
# RS_SPATIAL=0 drops the filter entirely (the control arm);
# RS_SPATIAL_SMOOTH_DELTA lowers the threshold so a millimetre-scale bead reads
# as an edge instead of as texture.
RS_SPATIAL = int(_env_number('RS_SPATIAL', 1))
RS_SPATIAL_SMOOTH_DELTA = _env_number('RS_SPATIAL_SMOOTH_DELTA', -1.0)

# Work-volume clip ahead of the spatial filter (audit R5): background depth must
# not be smoothed into surface edges, and nothing may fabricate depth. Env vars
# since 2026-09-01 (runtime-parameters spec 6.4) so the knob has all three
# precedence layers; the defaults are the constants every archived take was
# clipped under.
RS_DEPTH_MIN_M = _env_number('RS_DEPTH_MIN_M', 0.15)
RS_DEPTH_MAX_M = _env_number('RS_DEPTH_MAX_M', 1.5)

# The single source the depth-filter chain is built from. Env feeds it once at
# import (the unit file's boot defaults); a runtime SET (stream_burst) may
# rewrite entries later, which cannot survive a restart -- that impermanence is
# the safety argument of the runtime-parameters spec (4.2). -1.0 = leave the
# SDK's own default alone, the same sentinel RS_SPATIAL_SMOOTH_DELTA uses.
# `decimation` is recorded but pinned at 0: enabling it would change the depth
# geometry the greeting already declared (spec 2.2/2.5), so a SET refuses it.
FILTER_SETTINGS = {
    "spatial":                float(RS_SPATIAL),
    "spatial_smooth_delta":   RS_SPATIAL_SMOOTH_DELTA,
    "spatial_magnitude":      -1.0,
    "spatial_smooth_alpha":   -1.0,
    "spatial_holes_fill":     -1.0,
    "temporal_smooth_alpha":  -1.0,
    "temporal_smooth_delta":  -1.0,
    "temporal_persistency":   -1.0,
    "depth_min_m":            RS_DEPTH_MIN_M,
    "depth_max_m":            RS_DEPTH_MAX_M,
    "hole_filling":           -1.0,
    "decimation":             0.0,
}

# Derived from the SAME constants the chain is built from, at module level, so
# the greeting cannot describe a chain that did not run and no call ordering can
# leave it stale. This list is archived as provenance on every take, and it is
# the ONLY record of which arm of the A/B a take came from.
DEPTH_FILTER_NAMES = (["threshold", "disparity"]
                      + (["spatial"] if FILTER_SETTINGS["spatial"] else [])
                      + ["temporal", "disparity_inv"])

# The ACHIEVED value of every safe-tier knob, read back by setup_depth_filters()
# off the objects that actually process frames; None = that filter is absent from
# the chain, or the SDK would not report the option. Archived in the greeting's
# `filter_options` -- it is the ONLY record of which arm of an A/B a take came
# from (docs/inspection-roll-probe-handoff.md 3.1). Read-back rather than an echo
# of FILTER_SETTINGS because the -1 sentinel names no number at all: the filter
# then runs at librealsense's default, which a future SDK is free to change.
# Rebound (with DEPTH_FILTER_NAMES) by every setup_depth_filters() call -- at
# boot, and on every runtime SET -- always under _camera_lock or before any
# client thread exists, and always BEFORE _camera_generation moves, so a greeting
# can never pair old names with new options.
FILTER_OPTIONS = {}

# wire key -> (chain filter kind, rs.option attribute name). Temporal persistency
# and both hole-fill knobs ride the SDK's one `holes_fill` option, disambiguated
# by which filter object they are set on. threshold and hole_filling take their
# values through their CONSTRUCTORS (handled in setup_depth_filters), so this map
# only drives set_option for spatial/temporal -- but it drives READ-BACK for all.
_OPTION_MAP = {
    "spatial_smooth_delta":  ("spatial",      "filter_smooth_delta"),
    "spatial_magnitude":     ("spatial",      "filter_magnitude"),
    "spatial_smooth_alpha":  ("spatial",      "filter_smooth_alpha"),
    "spatial_holes_fill":    ("spatial",      "holes_fill"),
    "temporal_smooth_alpha": ("temporal",     "filter_smooth_alpha"),
    "temporal_smooth_delta": ("temporal",     "filter_smooth_delta"),
    "temporal_persistency":  ("temporal",     "holes_fill"),
    "depth_min_m":           ("threshold",    "min_distance"),
    "depth_max_m":           ("threshold",    "max_distance"),
    "hole_filling":          ("hole_filling", "holes_fill"),
}


STATIC_GEOMETRY = None      # rs_geometry.StaticGeometry, set by openPipeline
ACHIEVED_OPTIONS = {}       # read-back values, set by openPipeline
DEVICE_INFO = {}


# Everything one client is served from, captured together. A rebuild rebinds the
# five globals below ONE AT A TIME (openPipeline sets the three geometry/option/
# device ones as a side effect of restarting the streams; _rebuild_pipeline then
# sets `pipeline` and re-reads `depth_unit_mm`), so a reader that samples them
# separately can catch a MIX -- e.g. the reopened device's DEVICE_INFO with the
# pre-stall STATIC_GEOMETRY. That mix does not raise and is not logged: the client
# just back-projects every depth frame of that connection through the wrong
# numbers. `generation` is stamped in the same breath so a connection can later
# tell whether the camera it was greeted with is still the one streaming.
CameraSnapshot = namedtuple(
    "CameraSnapshot",
    "pipeline depth_unit_mm geometry achieved device generation")


def _camera_snapshot() -> CameraSnapshot:
    """One COHERENT view of the camera state, for anything that serves a client.

    Every write to these globals happens under ``_camera_lock`` (the whole rebuild
    runs inside it; startup binds them before any client thread exists), so reading
    them under the same lock is what makes the set atomic.

    The lock is held only for the six reads -- no device I/O inside it, so a slow or
    hung control transfer can never keep the supervisor from rebuilding. A reader
    that arrives mid-rebuild does WAIT for it (``_rebuild_pipeline`` holds the lock
    across the backoff and the multi-second ``openPipeline``), but it would have
    waited anyway: ``read_frames`` takes the same lock on the very next statement,
    so a greeted client pays the same total delay either way -- it simply gets the
    greeting after the rebuild instead of a torn one before it.
    """
    with _camera_lock:
        return CameraSnapshot(pipeline, depth_unit_mm, STATIC_GEOMETRY,
                              ACHIEVED_OPTIONS, DEVICE_INFO, _camera_generation)


def _greeting_is_stale(generation) -> bool:
    """True once a rebuild has invalidated the greeting this connection was sent.

    The greeting goes out ONCE per connection and the frame stream carries no
    generation marker, so a client that was already connected when the camera was
    rebuilt keeps describing the new device's frames with the old open's numbers --
    ``depth_unit_mm`` above all, which ``configure_depth_sensor`` re-applies on
    every open and is therefore not guaranteed to survive unchanged. Observed live
    2026-08-30: client 40437 connected 01:03:42, the camera rebuilt at 01:03:57,
    and that connection lived on until 01:08:35 -- 4.5 minutes on a stale greeting.

    Re-sending the greeting is not available: the wire format is one greeting line
    per connection and the host parses exactly one. So the connection is dropped
    instead, and the client's existing reconnect path fetches a fresh one through
    the unchanged handshake.

    Deliberately an unlocked read: rebinding/reading a module-level int is atomic
    under the GIL, ``_camera_generation`` is the LAST thing a rebuild rebinds, and
    taking ``_camera_lock`` here would make every frame of every client wait out a
    rebuild before being allowed to notice one.
    """
    return generation is not None and generation != _camera_generation


def _stale_greeting_close(addr, greeted_generation, detail) -> bool:
    """True -- after saying so in the journal -- when this connection must end.

    Every serving loop checks staleness TWICE, and this is why:

    * once at the top, so a connection idling through a rebuild ends without
      spending a second of the Nano's filter chain on a frame it must throw away;
    * once again after acquisition and immediately before the bytes go out, because
      ``getFrames`` runs that whole chain -- roughly a second -- and a rebuild that
      lands *inside* it has already passed the top check. Without the second check
      the frame is compressed and sent under the pre-rebuild greeting and only the
      NEXT iteration closes. One frame is the whole defect: ``CameraClient.grab()``
      reads exactly one frame per connection, so the leaked frame IS the per-pose
      scan capture / gate grab / extrusion measurement, scaled by the wrong
      ``depth_unit_mm`` with no error and no log line.

    Ordering, and why the second check is sufficient. ``_rebuild_pipeline`` holds
    ``_camera_lock`` across the ENTIRE rebuild and bumps ``_camera_generation``
    after it has rebound ``pipeline`` (and the geometry/options/device globals
    ``openPipeline`` sets); ``read_frames`` samples ``pipeline`` under that same
    lock. So an acquisition cannot straddle a rebuild: it sees generation G with the
    pipeline that belongs to G. Generation only ever increases, the top check read
    ``g`` before the sample (so G >= g) and this check reads ``g`` after it
    (so G <= g) -- hence G == g, and the frame in hand provably came from the open
    the client was greeted with.

    A rebuild completing between this check and ``sendall`` is therefore HARMLESS
    rather than impossible: the frame was already acquired from the greeted open, so
    it is honest data honestly described, and the next top-of-loop check ends the
    connection before anything from the new open can follow it.
    """
    if not _greeting_is_stale(greeted_generation):
        return False
    print(f"Closing {addr}: camera rebuilt (greeting describes generation "
          f"{greeted_generation}, camera is now {_camera_generation}); {detail}",
          flush=True)
    return True


def _first_color_sensor(device):
    """The RGB endpoint, however this librealsense build exposes it, or None.

    ``device.first_color_sensor()`` is not present in every pyrealsense2 build, so
    fall back to scanning the sensors by name ("RGB Camera" on a D435i). Returns
    None rather than raising: a colour endpoint we cannot find must degrade to a log
    line, never to an exception on the startup path -- the unit is Restart=always
    with no start limit, so that would be an infinite crash-loop with the camera
    dark for every module."""
    try:
        found = device.first_color_sensor()
        if found is not None:
            return found
    except Exception:
        pass                        # older/newer binding without that convenience
    try:
        sensors = list(device.query_sensors())
    except Exception as e:
        _log(f"WARNING: could not enumerate sensors to find the colour endpoint: {e}")
        return None
    for s in sensors:
        try:
            name = str(s.get_info(rs.camera_info.name)).lower()
        except Exception:
            continue
        if "rgb" in name or "color" in name or "colour" in name:
            return s
    return None


def openPipeline():
    """Start depth 720p + colour 1080p (no infrared: nobody reads it), configure the
    depth and colour sensors with read-back, record the as-found ASIC JSON, and
    extract the static geometry the greeting carries. Returns the pipeline."""
    global STATIC_GEOMETRY, ACHIEVED_OPTIONS, DEVICE_INFO
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, DEPTH_SIZE[0], DEPTH_SIZE[1], rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, COLOR_SIZE[0], COLOR_SIZE[1], rs.format.bgr8, 30)
    pipeline = rs.pipeline()
    profile = pipeline.start(cfg)
    try:
        device = profile.get_device()
        try:
            # As-found FIRST: the depth table (incl. depthUnits) is part of this
            # record. dump_advanced_mode_json guards its device calls internally,
            # but its os.makedirs/open are not guarded (rs_config.py is frozen) --
            # a full/unwritable ASFOUND_DIR must never stop the server serving
            # depth, so the call itself is guarded here instead.
            rs_config.dump_advanced_mode_json(device, rs, ASFOUND_DIR, log=_log)
        except Exception as e:
            _log(f"WARNING: as-found dump failed: {e}")
        sensor = device.first_depth_sensor()
        ACHIEVED_OPTIONS = rs_config.configure_depth_sensor(
            sensor, rs, laser_power=RS_LASER_POWER, visual_preset=RS_VISUAL_PRESET, log=_log)
        # auto_exposure_priority is registered on the COLOUR endpoint on the D400
        # series (librealsense src/ds5/ds5-color.cpp:161), so it has to be set on the
        # colour sensor; asking the depth one only ever logged "unsupported". Without
        # it AE can stretch colour exposure past the frame period in dim light, the
        # sensor drops below 30 fps, wait_for_frames stalls and the supervisor
        # rebuilds the pipeline. Guarded end to end: a colour sensor we cannot find,
        # or an option that will not take, is a log line and the server carries on.
        color_achieved = {}
        try:
            color_sensor = _first_color_sensor(device)
            if color_sensor is None:
                _log("RealSense: no colour sensor found - auto_exposure_priority not set")
            else:
                color_achieved = rs_config.configure_color_sensor(color_sensor, rs, log=_log)
        except Exception as e:
            _log(f"WARNING: colour sensor configuration failed: {e}")
        DEVICE_INFO = {
            "serial": device.get_info(rs.camera_info.serial_number),
            "fw": device.get_info(rs.camera_info.firmware_version),
            "librealsense": rs_config.librealsense_version(rs),
            # Carried in the greeting's `device` block (rs_geometry.build_greeting
            # splats DEVICE_INFO into it, so this needs no greeting schema change and
            # the host's CameraGeometry.from_greeting keeps it verbatim: it does
            # `device=dict(d.get("device") or {})`). None = not readable/not set.
            "color_auto_exposure_priority": color_achieved.get("auto_exposure_priority"),
        }
        STATIC_GEOMETRY = rs_geometry.static_geometry(profile, rs)     # raises on a bad extrinsic
        gte = rs_config.read_global_time_enabled(sensor, rs, log=_log)
        _log(f"RealSense: global_time_enabled = {gte}; temps = "
             f"{rs_config.read_temperatures(sensor, rs, log=_log)}; "
             f"depth {STATIC_GEOMETRY.depth_size} colour {STATIC_GEOMETRY.color_size}; "
             f"depth->colour t = {STATIC_GEOMETRY.t_dc_mm.round(2).tolist()} mm")
    except Exception:
        # A started pipeline still holds the USB device. Leaving it running while
        # this exception propagates (the extrinsic assert is DESIGNED to raise)
        # means the next rebuild attempt's pipeline.start() can fail with a
        # device-busy error caused by THIS attempt -- burning the supervisor's
        # 3-attempt recovery budget on a self-inflicted wound. Stop it
        # (best-effort; must never mask the real error) before re-raising.
        try:
            pipeline.stop()
        except Exception:
            pass
        raise
    return pipeline


def _log(msg):
    print(msg, flush=True)


def make_greeting(snapshot: CameraSnapshot = None) -> dict:
    """The per-connection greeting: static geometry + LIVE temps + achieved options.

    Built from ONE ``_camera_snapshot()`` rather than from the globals directly, so
    a rebuild running concurrently cannot hand a connecting client a mixture of two
    opens. The temperature/global-time reads are live device I/O and happen after
    the snapshot, outside the lock.
    """
    snap = _camera_snapshot() if snapshot is None else snapshot
    sensor = snap.pipeline.get_active_profile().get_device().first_depth_sensor()
    return rs_geometry.build_greeting(
        snap.geometry, depth_unit_mm=snap.depth_unit_mm,
        filters=list(DEPTH_FILTER_NAMES),
        filter_options=dict(FILTER_OPTIONS),
        temps=rs_config.read_temperatures(sensor, rs, log=_log),
        global_time_enabled=rs_config.read_global_time_enabled(sensor, rs, log=_log),
        achieved=snap.achieved, device=snap.device)


def greet(conn) -> int:
    """Send the protocol-2 greeting and return the camera generation it describes.

    Returning the generation (rather than sending and forgetting) is what lets the
    serving loop notice later that a rebuild has made this connection's one and only
    greeting a lie -- see ``_greeting_is_stale``.
    """
    snap = _camera_snapshot()
    conn.sendall(rs_geometry.greeting_line(make_greeting(snap)))
    return snap.generation


def handle_client(conn, addr):
    jpeg = turbojpeg.TurboJPEG('/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0')

    # Latency: disable Nagle so each frame's bytes go out immediately instead of
    # being coalesced/delayed (Nagle + delayed-ACK adds tens of ms per frame on a
    # request/stream protocol like this). Costs nothing for our large sends.
    try:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass

    # Handshake. Right after connecting a client sends ONE line declaring the
    # stream it wants (parsed by handshake.parse_handshake — shared with the host
    # test suite so the two sides can't silently drift):
    #     "MODE FULL V2"      -> depth+colour, protocol 2 (JSON greeting, raw depth)
    #     "MODE BURST V2"     -> burst capture of protocol-2 frames (see stream_burst)
    #     "MODE COLOR [Q<n>] [H264 [B<kbps>]] [SCAN]"  -> colour-only paths, unchanged
    #     "MODE TELEMETRY"    -> scan telemetry side-channel, unchanged
    # Anything else that asks for depth — no line (timeout), a bare "MODE FULL", or
    # garbage — is a pre-V2 depth request and is REFUSED below: a host that did not
    # restart after the protocol change must fail loudly at the handshake, not
    # misread the JSON greeting as a frame length and hang. A bare 'C' is still
    # accepted for MODE COLOR. Color-only is used by the live aiming preview +
    # calibration (which never use depth); it cuts ~75% of the per-frame bytes AND
    # the Nano's filter CPU — the difference between a ~0.5 fps preview and a
    # realtime one. The optional Q<n> lets the *preview* trade a little image
    # quality for speed while full-res captures keep the default (high) quality.
    # H264 goes further — the Nano's dedicated hardware encoder cuts preview
    # bandwidth ~10-20x and offloads the CPU. It is lossy + inter-frame (can soften
    # ChArUco corners) so it's the live-preview path only; one-shot captures keep
    # the JPEG/lossless path (the client never requests H264 for those). MODE BURST
    # buffers frames in RAM and ships them all in one transfer at the end — keeps
    # the robot tour from stalling on a per-pose depth transfer over Wi-Fi.
    #
    # Send timeout: a client that dies without a clean RST leaves sendall blocked
    # forever. When this server was single-threaded with listen(1) that alone
    # stopped it accepting anyone (the "NO SIGNAL" wedge); `main()` now runs a
    # thread per client against listen(5), so the wedge is no longer immediate --
    # but each stuck client still pins a thread and a socket for good, and enough
    # of those reproduce the same symptom (seen on the cell as sockets piling up
    # in CLOSE_WAIT: 10 on 2026-08-28, 8 on 2026-08-29 -- see _serve_client).
    # With a timeout, a stuck/dead client is dropped and its thread exits.
    # Generous enough that a slow-but-alive link (a full depth+color frame over
    # slow Wi-Fi can take a few seconds) is not killed.
    req = b""
    try:
        conn.settimeout(0.5)
        req = conn.recv(64)
    except (socket.timeout, OSError):
        pass
    finally:
        conn.settimeout(10.0)     # see the comment above about the send timeout
    hs = parse_handshake(req)
    color_only, codec, quality = hs["mode"] == "color", hs["codec"], hs["quality"]
    h264_bitrate, burst = hs["bitrate"], hs["mode"] == "burst"
    telemetry_only, scan_telemetry = hs["mode"] == "telemetry", hs["scan_telemetry"]
    print(f"Connection from {addr} ({hs})", flush=True)

    if hs["depth_requested"] and not hs["v2"]:
        # Big-bang protocol change: a host that did not restart must fail HERE,
        # loudly, not misread the JSON greeting as a frame length and hang.
        print(f"client {addr[0]} did not request V2; this server speaks protocol 2 only "
              f"(got {req!r})", flush=True)
        try:
            conn.sendall(b"ERR protocol 2 required; send MODE FULL V2\n")
        except OSError:
            pass
        conn.close()
        return

    if telemetry_only:
        stream_telemetry(conn, addr)
        conn.close()
        return

    if burst:
        # Burst capture: interactive CAP/GET/CLEAR loop on this connection; returns
        # to accept() when the client disconnects (or the buffer is done).
        stream_burst(conn, addr)
        conn.close()
        return

    if codec == 'h264':
        # Hardware H.264 path: relay the NVENC byte-stream over this connection and
        # return to accept() when the client disconnects (or the encoder dies).
        stream_h264(conn, addr, h264_bitrate, scan_telemetry=scan_telemetry)
        conn.close()
        return

    # Colour-only carries no geometry (no greeting is sent and the host reads
    # none), so a depth rebuild means nothing to it: greeted_generation stays None
    # and the staleness check below never fires for it.
    greeted_generation = None if color_only else greet(conn)

    while True:
        # This connection's geometry/depth scale describe a camera open that no
        # longer exists. There is no way to correct it in-band -- one greeting
        # per connection is the protocol -- so drop the connection and let the
        # client reconnect into a fresh greeting. Losing the connection is a
        # condition every host path already handles (the live preview
        # reconnects after a 1 s backoff, a burst tour falls back to per-pose
        # grabs, and a per-pose grab opens its own connection anyway); silently
        # wrong millimetres is not.
        if _stale_greeting_close(addr, greeted_generation,
                                 "the client must reconnect for a fresh greeting"):
            break
        if color_only:
            # Fast path: skip the depth filters entirely — they cost the Nano ~a
            # second per frame and we're throwing depth away anyway. Just grab the
            # raw color frame.
            frames = read_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            timestamp = frames.get_timestamp()
            depth_compressed = b''
        else:
            depth, color, timestamp = getFrames(depth_filters)
            if depth is None or color is None:
                continue
            depth_buffer = io.BytesIO()
            np.save(depth_buffer, depth)
            depth_buffer.seek(0)
            depth_compressed = lz4f.compress(depth_buffer.read())

        length_depth = struct.pack('<I', len(depth_compressed))
        data_color = (jpeg.encode(color, quality=quality) if quality is not None
                      else jpeg.encode(color, quality=SCAN_COLOR_JPEG_QUALITY))
        length_color = struct.pack('<I', len(data_color))
        ts = struct.pack('<d', timestamp)

        frame_data = length_depth + length_color + ts + depth_compressed + data_color
        # Second check, with the frame in hand and nothing but the send left to do:
        # getFrames() is ~1 s of filter chain, so a rebuild can land entirely inside
        # it and the top-of-loop check above would have waved this frame through.
        # See _stale_greeting_close for why this placement is sufficient. (No-op for
        # colour-only clients: their greeted_generation is None.)
        if _stale_greeting_close(
                addr, greeted_generation,
                "DISCARDING the frame captured across the rebuild rather than "
                "sending it under the old open's numbers; the client must "
                "reconnect for a fresh greeting"):
            break
        try:
            conn.sendall(frame_data)
        except (ConnectionResetError, BrokenPipeError, socket.timeout) as e:
            print(f"Lost connection to {addr}: {e}")
            break

    conn.close()


def _write_all(stream, data):
    """Write every byte of ``data`` to an unbuffered (bufsize=0) pipe, which can
    accept a partial write and return a short count."""
    mv = memoryview(data)
    while mv:
        n = stream.write(mv)
        if n is None:        # only on a non-blocking fd; ours is blocking
            continue
        mv = mv[n:]


def stream_h264(conn, addr, bitrate_kbps, scan_telemetry=False):
    """Encode the live color stream with the Nano's hardware H.264 encoder (NVENC)
    and relay the resulting Annex-B byte-stream over ``conn``.

    Rather than depend on GStreamer Python bindings (``gi`` is only available under
    the system Python 3.6, not this server's 3.10 venv), we drive ``gst-launch-1.0``
    as a subprocess: raw BGR frames go in on its stdin (``fdsrc``), encoded H.264
    comes out on its stdout (``fdsink``). A feeder thread keeps the encoder fed with
    the newest camera frames while this thread shovels encoder output to the socket;
    decoupling the two means a slow link can't stall the capture/encode.

    Wire format here is just the raw H.264 byte-stream (no per-frame header) — the
    client feeds it straight to a streaming decoder, which finds the access-unit
    boundaries itself. Baseline profile (no B-frames) keeps latency low and the SPS/
    PPS are inlined (``insert-sps-pps``/``config-interval=-1``) so a client that
    connects mid-stream can start decoding at the next IDR.
    """
    width, height = COLOR_SIZE
    cmd = [
        'gst-launch-1.0', '-q',
        'fdsrc', 'fd=0', '!',
        'rawvideoparse', 'use-sink-caps=false',
        f'width={width}', f'height={height}', 'format=bgr', 'framerate=30/1', '!',
        'videoconvert', '!', 'video/x-raw,format=BGRx', '!',
        'nvvidconv', '!', 'video/x-raw(memory:NVMM),format=NV12', '!',
        'nvv4l2h264enc', 'control-rate=1', f'bitrate={int(bitrate_kbps) * 1000}',
        'iframeinterval=30', 'idrinterval=30', 'insert-sps-pps=1',
        'maxperf-enable=1', 'preset-level=1', 'profile=0', '!',
        'h264parse', 'config-interval=-1', '!',
        'video/x-h264,stream-format=byte-stream,alignment=au', '!',
        'fdsink', 'fd=1',
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, bufsize=0)
    except (OSError, ValueError) as e:
        print(f"H264: cannot start gst-launch ({e}); dropping {addr}")
        return

    frame_bytes = width * height * 3  # BGR8
    stop = threading.Event()

    def feeder():
        last_telemetry = 0.0
        # ONE pointcloud processing block for the life of this feeder thread, instead
        # of a fresh one per telemetry tick. rs.pointcloud() is a librealsense
        # processing block with internal allocation/frame-queue caching that a
        # per-tick rebuild threw away. This is the right scope on three counts:
        #   - not module scope: the host test suite imports this file with a stubbed
        #     `rs` namespace, so constructing one at import time would raise there;
        #   - it survives a recovery rebuild: the block is handed a depth frame
        #     explicitly by calculate() and holds no reference to the pipeline, so
        #     _rebuild_pipeline() swapping the device out underneath it is a no-op
        #     for it (it re-reads the stream profile off each frame);
        #   - per-thread, not shared: two concurrent telemetry clients get their own
        #     block rather than racing inside one processing block.
        # Its existence IS the telemetry enable flag in the loop below, so a
        # construction failure costs telemetry rather than the whole H.264 stream
        # this thread feeds (the try: below catches CameraUnavailable/OSError from
        # streaming, not a construction failure).
        pointcloud = None
        if scan_telemetry:
            try:
                pointcloud = rs.pointcloud()
            except Exception as e:                       # noqa: BLE001
                print(f"H264: no pointcloud block ({e}); scan telemetry disabled",
                      flush=True)
        try:
            while not stop.is_set():
                frames = read_frames()
                color = frames.get_color_frame()
                if not color:
                    continue
                if pointcloud is not None and (
                        time.monotonic() - last_telemetry >= SCAN_TELEMETRY_PERIOD_S):
                    depth = frames.get_depth_frame()
                    if depth:
                        try:
                            depth_profile = depth.profile.as_video_stream_profile()
                            color_profile = color.profile.as_video_stream_profile()
                            intr = depth_profile.intrinsics
                            color_intr = color_profile.intrinsics
                            depth_to_color = depth_profile.get_extrinsics_to(color_profile)
                            # One coherent read per tick: the extrinsic and the
                            # depth scale come from the SAME open, so a rebuild
                            # in flight cannot pair the new device's geometry
                            # with the old open's millimetres-per-count.
                            snap = _camera_snapshot()
                            R_vec = snap.geometry.R_dc           # row-major, asserted at start
                            t_dc_mm = snap.geometry.t_dc_mm
                            depth_points = pointcloud.calculate(depth)
                            depth_vertices_mm = (
                                np.asanyarray(depth_points.get_vertices())
                                .view(np.float32)
                                .reshape(depth.get_height(), depth.get_width(), 3)
                                * 1000.0)

                            def overlay_project(p):
                                color_point = rs.rs2_transform_point_to_point(
                                    depth_to_color, [float(x) for x in p])
                                return rs.rs2_project_point_to_pixel(color_intr, color_point)

                            def project_color_points(points_color):
                                cp = np.asarray(points_color, float)
                                zc = cp[:, 2]
                                return np.column_stack([
                                    cp[:, 0] * float(color_intr.fx) / zc + float(color_intr.ppx),
                                    cp[:, 1] * float(color_intr.fy) / zc + float(color_intr.ppy)])

                            def overlay_project_points(points):
                                return project_color_points(np.asarray(points, float) @ R_vec.T + t_dc_mm)

                            def overlay_transform_points(points):
                                return np.asarray(points, float) @ R_vec.T + t_dc_mm

                            def depth_deproject_points(pixels, depths_mm):
                                uv = np.rint(np.asarray(pixels, float)).astype(int)
                                uv[:, 0] = np.clip(uv[:, 0], 0, depth.get_width() - 1)
                                uv[:, 1] = np.clip(uv[:, 1], 0, depth.get_height() - 1)
                                return depth_vertices_mm[uv[:, 1], uv[:, 0]]

                            payload = scan_plane_telemetry(
                                np.asanyarray(depth.get_data()), intr, snap.depth_unit_mm,
                                overlay_project=overlay_project,
                                overlay_project_points=overlay_project_points,
                                overlay_transform_points=overlay_transform_points,
                                depth_deproject_points=depth_deproject_points,
                                overlay_size=(color.get_width(), color.get_height()),
                                color_image=np.asanyarray(color.get_data()))
                            publish_scan_telemetry(payload)
                        except Exception as e:
                            publish_scan_telemetry({
                                "detected": False, "valid_frac": 0.0,
                                "error": str(e), "timestamp": time.time()})
                    last_telemetry = time.monotonic()
                buf = np.asanyarray(color.get_data())
                if buf.nbytes != frame_bytes:        # unexpected size -> skip
                    continue
                _write_all(proc.stdin, buf.tobytes())
        except CameraUnavailable as e:
            # Say why in one line; the supervisor has already logged the detail.
            print(f"H264 feeder stopping: {e}", flush=True)
        except (BrokenPipeError, OSError, ValueError):
            pass  # encoder gone / pipe closed -> sender loop will end too
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    feeder_thread = threading.Thread(target=feeder, name='h264-feeder', daemon=True)
    feeder_thread.start()
    print(f"H264 stream to {addr} @ {bitrate_kbps} kbps")
    try:
        while True:
            # bufsize=0 -> stdout is raw, so read() returns as soon as any bytes are
            # available (one syscall) rather than blocking to fill the buffer.
            chunk = proc.stdout.read(65536)
            if not chunk:
                break                                # encoder exited / EOF
            conn.sendall(chunk)
    except (ConnectionResetError, BrokenPipeError, socket.timeout, OSError) as e:
        print(f"Lost H264 connection to {addr}: {e}")
    finally:
        stop.set()
        try:
            proc.terminate()
        except OSError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
        feeder_thread.join(timeout=2)


def _recv_line(conn, maxlen=64):
    """Read one newline-terminated command line from ``conn`` (CAP/GET/CLEAR).

    Commands are tiny and the client sends one at a time (it waits for each reply
    before sending the next), so a byte-at-a-time read can't over-read into the next
    command or into frame data. Returns b'' if the peer closed."""
    buf = bytearray()
    while len(buf) < maxlen:
        ch = conn.recv(1)
        if not ch:
            break                 # peer closed
        if ch == b'\n':
            break
        buf.extend(ch)
    return bytes(buf)


def stream_burst(conn, addr, max_frames=64):
    """Burst capture: buffer native depth+color frames on the Jetson, then transfer
    them all in one burst — so the robot tour isn't stalled by a per-pose depth
    transfer over the cell's Wi-Fi (a full depth+color frame can take 6-11 s).

    Protocol on this connection (newline-terminated commands; length-prefixed replies):

        (server first sends BURST READY\\n then the protocol-2 greeting line)
        CAP    -> grab + filter + compress ONE frame into a RAM buffer; reply
                  ``<I idx><I thumb_len><thumb JPEG>`` (a small color thumbnail for the
                  client's live per-pose strip). thumb_len=0 signals a skip.
        GET    -> reply ``<I count>`` then each buffered frame as
                  ``<I depth_len><I color_len><d ts> + depth(lz4 .npy) + color(JPEG)``
                  (identical per-frame framing to the normal stream, so the client
                  reuses its decoder).
        CLEAR  -> drop the buffer; reply ``<I 0>``.
        SET    -> ``SET [key=value ...]``: change safe-tier depth-filter settings
                  at runtime (runtime-parameters spec). Reply is ONE JSON line:
                  {"ok":true,"filters":[...],"filter_options":{...}} with the
                  ACHIEVED values, or {"ok":false,"error":"..."}. A successful
                  write retires the camera generation: every session greeted
                  before it (THIS one included, after the reply) is closed and
                  reconnects into a fresh greeting. Bare SET = read-only.
                  Overrides die on restart -- the unit file stays the boot truth.

    The buffer is RAM-only and is ALSO dropped when the connection ends (finally), so
    an abandoned/dropped burst leaves NO data on the Jetson — no disk garbage. Between
    CAP commands the robot is moving, so the command read uses a generous timeout."""
    jpeg = turbojpeg.TurboJPEG('/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0')
    buffer = []   # list of (depth_compressed: bytes, color_jpeg: bytes, ts: float)
    try:
        conn.sendall(b'BURST READY\n')
        greeted_generation = greet(conn)
    except OSError:
        return
    print(f"Burst session opened with {addr}")
    try:
        while True:
            # Same contract as the streaming path: the buffered frames and the
            # ones still to come no longer share the geometry this session was
            # greeted with. End the session (the buffer is dropped in `finally`,
            # as always) so the client re-opens and is greeted afresh -- the host
            # falls back to the per-pose grab path on a burst error.
            if _stale_greeting_close(addr, greeted_generation,
                                     "ending the burst session"):
                break
            # The next command may be many seconds away (the robot is moving between
            # poses); wait generously rather than dropping the connection.
            conn.settimeout(180.0)
            # maxlen 256: a multi-key SET line (spec 3.1) outgrows the default 64.
            line = _recv_line(conn, maxlen=256)
            cmd = line.strip().upper()
            if not cmd:
                break                      # peer closed

            if cmd == b'SET' or cmd.startswith(b'SET '):
                # Runtime filter settings (runtime-parameters spec 3.1). Reply
                # FIRST; a successful WRITE bumped the generation, so the
                # top-of-loop staleness check ends this session on the next
                # iteration -- after the reply is out. The client reconnects
                # into a fresh greeting that records the new values. A bare SET
                # (a read) bumps nothing and the session continues. NOTE the
                # case split: keys are lowercase, so parse from `line`, not
                # the uppercased `cmd`.
                conn.sendall(_handle_set(line.strip()))
                continue

            if cmd == b'CAP':
                depth = color = None
                tries = 0
                while (depth is None or color is None) and tries < 10:
                    depth, color, _ts = getFrames(depth_filters)
                    tries += 1
                # Same second check as the streaming path, in the same place: after
                # acquisition (which here can be up to ten filter-chain passes) and
                # before this frame reaches EITHER the socket or the RAM buffer that
                # a later GET ships wholesale. Ending the session drops the frames
                # buffered BEFORE the rebuild too, even though the greeting still
                # describes those correctly -- deliberately: the tour has a hole in
                # it either way (this pose was not captured and no further pose can
                # be), the protocol is client-driven so the server cannot push the
                # partial buffer out before closing, and a short buffer delivered as
                # if complete is a worse failure than none. `_capture` catches the
                # burst error and falls back to per-pose grabs, each of which opens
                # its own connection and is greeted afresh.
                if _stale_greeting_close(
                        addr, greeted_generation,
                        "DISCARDING the frame captured across the rebuild and "
                        "ending the burst session"):
                    break
                if depth is None or color is None or len(buffer) >= max_frames:
                    # signal a skip (no frame / buffer full) — index still advances
                    conn.sendall(struct.pack('<I', len(buffer)) + struct.pack('<I', 0))
                    continue
                depth_buffer = io.BytesIO()
                np.save(depth_buffer, depth)
                depth_buffer.seek(0)
                depth_compressed = lz4f.compress(depth_buffer.read())
                color_jpeg = jpeg.encode(color, quality=SCAN_COLOR_JPEG_QUALITY)
                idx = len(buffer)
                buffer.append((depth_compressed, color_jpeg, _ts))
                # Small thumbnail (~4x downscale) for the live per-pose strip.
                thumb_src = np.ascontiguousarray(color[::4, ::4])
                thumb = jpeg.encode(thumb_src, quality=60)
                conn.sendall(struct.pack('<I', idx) + struct.pack('<I', len(thumb)) + thumb)

            elif cmd == b'GET':
                # One bulk transfer of all buffered frames. Send frame-by-frame (each
                # bounded like the normal path) under a generous timeout.
                conn.settimeout(120.0)
                conn.sendall(struct.pack('<I', len(buffer)))
                for depth_compressed, color_jpeg, ts in buffer:
                    frame_data = (struct.pack('<I', len(depth_compressed))
                                  + struct.pack('<I', len(color_jpeg))
                                  + struct.pack('<d', ts)
                                  + depth_compressed + color_jpeg)
                    conn.sendall(frame_data)

            elif cmd == b'CLEAR':
                buffer = []
                conn.sendall(struct.pack('<I', 0))

            # unknown commands are ignored (keeps the protocol forgiving)
    except (ConnectionResetError, BrokenPipeError, socket.timeout, OSError) as e:
        print(f"Burst connection to {addr} ended: {e}")
    finally:
        n = len(buffer)
        buffer = []                        # RAM buffer dropped — nothing left on the Jetson
        print(f"Burst session with {addr} closed; cleared {n} buffered frame(s)")


def _apply_option(filt, option_name, value):
    """Set one filter option, clamped to the SDK's advertised range when it
    exposes one (spec test 5; mirrors rs_config._set_with_readback's discipline:
    a bad value must never take the service down -- the achieved read-back in
    FILTER_OPTIONS is what gets archived, so a clamp is visible, not silent)."""
    try:
        option = getattr(rs.option, option_name)
    except Exception as exc:
        print("WARNING: this pyrealsense2 build has no option {!r} ({}); {} left "
              "at its previous value".format(option_name, exc, option_name), flush=True)
        return
    try:
        rng = filt.get_option_range(option)
        value = min(max(float(value), float(rng.min)), float(rng.max))
    except Exception:
        pass                       # no range API on this build: try the raw value
    try:
        filt.set_option(option, value)
    except Exception as exc:
        print("WARNING: could not set {}={:g} ({}); the filter keeps its "
              "previous value".format(option_name, value, exc), flush=True)


def _achieved_filter_options(by_kind):
    """One achieved value per wire key, read off the filters that will run.
    Guarded per option: provenance is never worth the camera, so a refused
    read-back records None (unknown) and the service keeps serving.
    `decimation` is a constant 0.0 -- this tier cannot enable it (it would
    change the depth geometry the greeting already declared, spec 2.2/2.5), and
    recording the 0 makes "we do not decimate, on purpose" a value instead of an
    absence someone has to infer."""
    def read(kind, option_name):
        filt = by_kind.get(kind)
        if filt is None:
            return None
        try:
            return float(filt.get_option(getattr(rs.option, option_name)))
        except Exception as exc:
            print("WARNING: could not read back {}.{} ({}) -- takes captured now "
                  "will not record it".format(kind, option_name, exc), flush=True)
            return None
    achieved = {}
    for key in _OPTION_MAP:
        kind, option_name = _OPTION_MAP[key]
        achieved[key] = read(kind, option_name)
    achieved["decimation"] = 0.0
    return achieved


def setup_depth_filters():
    """threshold -> disparity -> [spatial] -> temporal -> disparity_inv ->
    [hole_filling], on NATIVE depth, built ENTIRELY from FILTER_SETTINGS.

    No decimation ever (it changes the depth geometry the greeting declares --
    restart path only, spec 2.5) and hole filling only by explicit request: a
    filled pixel is fabricated depth, fabricated exactly where the metrology
    cares (surface edges). Threshold first so background is never smoothed into
    an edge; hole filling last, in the depth domain, per Intel's recommended
    order. -1.0 anywhere = leave that SDK default alone.

    Also rebinds DEPTH_FILTER_NAMES and FILTER_OPTIONS (the achieved values,
    read back off these very objects) so the greeting always describes the chain
    that is actually installed -- see the FILTER_OPTIONS comment block for the
    ordering contract with _camera_generation."""
    global DEPTH_FILTER_NAMES, FILTER_OPTIONS
    s = FILTER_SETTINGS
    by_kind = {"threshold": rs.threshold_filter(float(s["depth_min_m"]),
                                                float(s["depth_max_m"]))}
    chain = [by_kind["threshold"], rs.disparity_transform(True)]
    names = ["threshold", "disparity"]
    if s["spatial"]:
        by_kind["spatial"] = rs.spatial_filter()
        chain.append(by_kind["spatial"])
        names.append("spatial")
    by_kind["temporal"] = rs.temporal_filter()
    chain += [by_kind["temporal"], rs.disparity_transform(False)]
    names += ["temporal", "disparity_inv"]
    if s["hole_filling"] >= 0:
        by_kind["hole_filling"] = rs.hole_filling_filter(int(s["hole_filling"]))
        chain.append(by_kind["hole_filling"])
        names.append("hole_filling")
    for key in _OPTION_MAP:
        kind, option_name = _OPTION_MAP[key]
        if kind in ("threshold", "hole_filling"):
            continue               # constructed with their values above
        if kind in by_kind and s[key] >= 0:
            _apply_option(by_kind[kind], option_name, s[key])
    DEPTH_FILTER_NAMES = names
    FILTER_OPTIONS = _achieved_filter_options(by_kind)
    print("RealSense: depth filters {}, options {}".format(
        DEPTH_FILTER_NAMES, FILTER_OPTIONS), flush=True)
    return chain


class SettingError(ValueError):
    """A SET the caller must hear about: unknown key, bad number, refused knob."""


def apply_filter_settings(updates):
    """Validate + apply runtime filter settings; return (filter_options, filters).

    All-or-nothing: any unknown key or refused value raises SettingError BEFORE
    anything is touched -- a caller that thinks it changed something must never
    have half-changed it. An empty ``updates`` is a pure read: current achieved
    values back, nothing rebuilt, no generation bump, nobody's session ends.

    A write rebuilds the chain from FILTER_SETTINGS under _camera_lock and bumps
    _camera_generation LAST -- the same ordering invariant _rebuild_pipeline
    documents -- so _stale_greeting_close retires every session greeted under
    the old chain and no frame is ever archived under a greeting that describes
    a chain it did not run through (spec 3.4: a filter swap does not change
    geometry, so the fusion guard cannot catch it; this is what does).

    ``FILTER_OPTIONS`` and ``DEPTH_FILTER_NAMES`` are read back TOGETHER, inside
    the same locked section that does the write (or, for a read, the same locked
    section that touches nothing) -- never one before the lock and one after.
    One thread per accepted connection can call this concurrently, and reading
    the two globals separately would let a second SET land in the gap, handing a
    caller a reply pairing one chain's names with a DIFFERENT chain's options.

    If ``setup_depth_filters()`` rejects the new values (e.g. an SDK-level
    constructor refusing an inverted ``depth_min_m``/``depth_max_m``),
    ``FILTER_SETTINGS`` is rolled back to what it held before this call, so the
    rejection surfaces as one failed SET rather than poisoning the dict every
    later SET rebuilds from -- without this, a single bad value would wedge
    runtime parameters until a restart, which is exactly what this whole feature
    exists to avoid needing. ``depth_filters``/``DEPTH_FILTER_NAMES``/
    ``FILTER_OPTIONS`` need no matching rollback: ``setup_depth_filters`` only
    rebinds them at its very end, after every filter object has already been
    built without raising, so a raise there never touches them in the first
    place.

    Known, accepted caveat (spec 3.4): _rebuild_pipeline treats a moved
    generation as "another thread already rebuilt the wedged pipeline" and skips
    one recovery attempt, so a SET landing during a genuine camera wedge delays
    recovery by one timeout cycle. It self-heals; do not "fix" it here.

    A runtime override lives only in FILTER_SETTINGS (RAM): a restart re-imports
    the module and re-reads env, which is the whole safety argument (spec 4.2).
    There is deliberately no persist path."""
    global depth_filters, _camera_generation
    unknown = sorted(k for k in updates if k not in FILTER_SETTINGS)
    if unknown:
        raise SettingError("unknown setting(s): {} (settable: {})".format(
            ", ".join(unknown), ", ".join(sorted(FILTER_SETTINGS))))
    if float(updates.get("decimation", 0.0)) != 0.0:
        raise SettingError("decimation changes the depth geometry the greeting "
                           "declares; it stays restart-path only (spec 2.5)")
    clean = dict((k, float(v)) for k, v in updates.items())
    if "hole_filling" in clean:                    # constructor arg, not an rs.option:
        clean["hole_filling"] = min(max(clean["hole_filling"], -1.0), 2.0)
    with _camera_lock:
        if updates:
            previous = dict(FILTER_SETTINGS)
            FILTER_SETTINGS.update(clean)
            try:
                depth_filters = setup_depth_filters()
            except Exception as exc:
                FILTER_SETTINGS.clear()
                FILTER_SETTINGS.update(previous)
                raise SettingError("rejected by the filter chain: {}".format(exc))
            _camera_generation += 1    # LAST: retires every session greeted before it
        achieved = dict(FILTER_OPTIONS)
        names = list(DEPTH_FILTER_NAMES)
    if updates:
        print("Runtime SET applied: {} -> generation {}".format(
            clean, _camera_generation), flush=True)
    return achieved, names


def _handle_set(line):
    """One ``SET [key=value ...]`` line -> one JSON reply line. Never raises.

    Unknown COMMANDS in the burst loop stay forgiving (a newer client against an
    older server should degrade, not die), but an unknown SETTING inside a SET is
    an ERROR: the caller believes it changed something it did not (spec test 4)."""
    try:
        updates = {}
        for token in line.decode("ascii", "replace").split()[1:]:
            key, eq, raw = token.partition("=")
            if not eq:
                raise SettingError(
                    "malformed token {!r}: expected key=value".format(token))
            try:
                value = float(raw)
            except ValueError:
                raise SettingError("{}={!r} is not a number".format(key, raw))
            if not math.isfinite(value):
                # nan/inf both parse as valid floats. Unchecked, nan would land
                # in FILTER_SETTINGS while `s[key] >= 0` (setup_depth_filters'
                # own guard) is False for nan, so the filter silently keeps its
                # SDK default -- FILTER_SETTINGS then claims a value that never
                # reached anything.
                raise SettingError(
                    "{}={!r} must be finite".format(key, raw))
            updates[key] = value
        achieved, names = apply_filter_settings(updates)
        reply = {"ok": True, "filters": names, "filter_options": achieved}
    except SettingError as exc:
        reply = {"ok": False, "error": str(exc)}
    except Exception as exc:      # a bug must degrade to an error line, not kill the thread
        reply = {"ok": False, "error": "internal: {!r}".format(exc)}
    return json.dumps(reply, separators=(",", ":")).encode("utf-8") + b"\n"


def _serve_client(conn, addr):
    """Run handle_client for one client and ALWAYS close its socket.

    handle_client's own conn.close() calls sit on its normal-return paths. When
    acquisition fails it raises straight out of the handler, the thread dies and
    the accepted socket is never closed -- it sits in CLOSE_WAIT forever. Against
    listen(5) a handful of those stop new clients connecting, so a *recoverable*
    camera stall turns into a server that must be restarted by hand (seen on the
    cell 2026-08-28: 10 leaked sockets, and again 2026-08-29: 8). The traceback is
    still printed, because it is what identifies the failure in the journal.

    Since the camera supervisor landed, a wedge surfaces here as CameraUnavailable
    rather than a raw "Frame didn't arrive within 5000" -- read_frames() rides out
    a brief stall and rebuilds the pipeline for a persistent one, so reaching this
    point means recovery was tried and failed.
    """
    try:
        handle_client(conn, addr)
    except Exception:
        print(f"camera-client {addr[0]}:{addr[1]} failed:")
        traceback.print_exc()
    finally:
        try:
            conn.close()
        except OSError:
            pass


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen(5)
    print(f'Server ip-address: {get_ip_address()}:{port}')

    while True:
        conn, addr = server_socket.accept()
        threading.Thread(target=_serve_client, args=(conn, addr),
                         name=f"camera-client-{addr[0]}:{addr[1]}",
                         daemon=True).start()

if __name__ == '__main__':
    print(f"Initiating Jetson-Realsense Wi-Fi Server: depth {DEPTH_SIZE}, colour {COLOR_SIZE}, protocol 2")
    try:
        pipeline = openPipeline()
        depth_unit_mm = (pipeline.get_active_profile().get_device()
                         .first_depth_sensor().get_depth_scale() * 1000.0)
        depth_filters = setup_depth_filters()
        main()
    except Exception as e:
        print(f"Unexpected error: {e}")

