import socket
import struct
import subprocess
import threading
import json
import time
from collections import deque
import pyrealsense2 as rs
import numpy as np
import lz4.frame as lz4f
import io
import turbojpeg

port = 1024

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


def scan_plane_telemetry(depth, intrinsics, depth_unit_mm=1.0,
                         patch_frac=0.25, min_valid_frac=0.25,
                         overlay_project=None, overlay_project_points=None,
                         overlay_transform_points=None, overlay_size=None,
                         depth_deproject_points=None):
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
                # Physical lengths corresponding to outline edges 0->1 and 1->2.
                # Unlike extent_mm, this deliberately preserves corner order.
                "rectangle_size_mm": [float(len1), float(len2)],
                "rectangle_corners_color_mm": (
                    corners_color.tolist() if corners_color is not None else None),
                "outline_uv": outline if len(outline) >= 3 else None,
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


def getFrames(pipeline, align, depth_filters):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth = aligned_frames.get_depth_frame()
    color = aligned_frames.get_color_frame()

    if not depth or not color:
        return None, None, None

    for filter in depth_filters:
        depth = filter.process(depth)

    depth_data = np.asanyarray(depth.get_data())
    color_data = np.asanyarray(color.get_data())

    ts = frames.get_timestamp()

    return depth_data, color_data, ts

width = 1280;
height = 720;
def openPipeline():
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.infrared, 1)
    pipeline = rs.pipeline()
    pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    return pipeline, align

def handle_client(conn, addr):
    jpeg = turbojpeg.TurboJPEG('/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0')

    # Latency: disable Nagle so each frame's bytes go out immediately instead of
    # being coalesced/delayed (Nagle + delayed-ACK adds tens of ms per frame on a
    # request/stream protocol like this). Costs nothing for our large sends.
    try:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass

    # Optional, backward-compatible handshake. Right after connecting a client may
    # send ONE line declaring the stream it wants:
    #     "MODE COLOR"        -> lightweight COLOR-ONLY (depth_len=0, no align/filter/lz4)
    #     "MODE COLOR Q60"    -> color-only AND encode JPEG at quality 60 (smaller =
    #                            fewer bytes over Wi-Fi = higher preview fps)
    #     "MODE COLOR H264"   -> color-only, hardware-NVENC H.264 byte-stream (see
    #                            stream_h264) instead of per-frame JPEG. "B<kbps>"
    #                            sets the encoder bitrate (e.g. "MODE COLOR H264 B4000").
    #     "MODE FULL"         -> the full depth+color stream
    # No line (timeout) / anything unrecognized => FULL at default quality, so
    # existing depth/scan clients are unaffected (they just connect and read). A
    # bare 'C' is also accepted. Color-only is used by the live aiming preview +
    # calibration (which never use depth); it cuts ~75% of the per-frame bytes AND
    # the Nano's align+filter CPU — the difference between a ~0.5 fps preview and a
    # realtime one. The optional Q<n> lets the *preview* trade a little image
    # quality for speed while full-res captures keep the default (high) quality.
    # H264 goes further — the Nano's dedicated hardware encoder cuts preview
    # bandwidth ~10-20x and offloads the CPU. It is lossy + inter-frame (can soften
    # ChArUco corners) so it's the live-preview path only; one-shot captures keep
    # the JPEG/lossless path (the client never requests H264 for those).
    #     "MODE BURST"        -> burst capture: buffer aligned depth+color frames in
    #                            RAM and ship them all in one transfer at the end (see
    #                            stream_burst). Keeps the robot tour from stalling on a
    #                            per-pose depth transfer over Wi-Fi.
    color_only = False
    quality = None
    codec = 'jpeg'
    burst = False
    telemetry_only = False
    scan_telemetry = False
    h264_bitrate = 4000  # kbps; overridden by a "B<kbps>" handshake token
    try:
        conn.settimeout(0.5)
        req = conn.recv(64).strip().upper()
        burst = req.startswith(b'MODE BURST')
        telemetry_only = req.startswith(b'MODE TELEMETRY')
        color_only = req.startswith(b'MODE COLOR') or req == b'C'
        scan_telemetry = b'SCAN' in req.split()
        for tok in req.split():
            if tok == b'H264':
                codec = 'h264'
            elif tok.startswith(b'Q') and tok[1:].isdigit():
                quality = max(10, min(100, int(tok[1:])))
            elif tok.startswith(b'B') and tok[1:].isdigit():
                h264_bitrate = max(500, min(20000, int(tok[1:])))
    except (socket.timeout, OSError):
        pass
    finally:
        # Send timeout: this server is single-threaded with listen(1), so a client
        # that dies without a clean RST would otherwise leave sendall blocked
        # forever and the server would stop accepting anyone (the "NO SIGNAL"
        # wedge). With a timeout, a stuck/dead client just gets dropped and we go
        # back to accept(). Generous enough that a slow-but-alive link (a full
        # depth+color frame over slow Wi-Fi can take a few seconds) is not killed.
        conn.settimeout(10.0)
    print(f"Connection from {addr} (color_only={color_only}, codec={codec}, "
          f"quality={quality}, bitrate={h264_bitrate}, burst={burst}, "
          f"telemetry_only={telemetry_only}, scan_telemetry={scan_telemetry})")

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
        stream_h264(conn, addr, width, height, h264_bitrate,
                    scan_telemetry=scan_telemetry)
        conn.close()
        return

    while True:
        if color_only:
            # Fast path: skip align (depth->color) AND the spatial depth filter
            # entirely — they cost the Nano ~a second per frame and we're throwing
            # depth away anyway. Just grab the raw color frame. (align leaves color
            # unchanged, so intrinsics/detection are identical to the full path.)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            timestamp = frames.get_timestamp()
            depth_compressed = b''
        else:
            depth, color, timestamp = getFrames(pipeline, align, depth_filters)
            if depth is None or color is None:
                continue
            depth_buffer = io.BytesIO()
            np.save(depth_buffer, depth)
            depth_buffer.seek(0)
            depth_compressed = lz4f.compress(depth_buffer.read())

        length_depth = struct.pack('<I', len(depth_compressed))
        data_color = (jpeg.encode(color, quality=quality) if quality is not None
                      else jpeg.encode(color))
        length_color = struct.pack('<I', len(data_color))
        ts = struct.pack('<d', timestamp)

        frame_data = length_depth + length_color + ts + depth_compressed + data_color
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


def stream_h264(conn, addr, width, height, bitrate_kbps, scan_telemetry=False):
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
        try:
            while not stop.is_set():
                frames = pipeline.wait_for_frames()
                color = frames.get_color_frame()
                if not color:
                    continue
                if scan_telemetry and time.monotonic() - last_telemetry >= 0.40:
                    depth = frames.get_depth_frame()
                    if depth:
                        try:
                            depth_profile = depth.profile.as_video_stream_profile()
                            color_profile = color.profile.as_video_stream_profile()
                            intr = depth_profile.intrinsics
                            color_intr = color_profile.intrinsics
                            depth_to_color = depth_profile.get_extrinsics_to(color_profile)
                            R_dc = np.asarray(depth_to_color.rotation, dtype=float).reshape(3, 3)
                            t_dc_mm = np.asarray(depth_to_color.translation, dtype=float) * 1000.0
                            pointcloud = rs.pointcloud()
                            depth_points = pointcloud.calculate(depth)
                            depth_vertices_mm = (
                                np.asanyarray(depth_points.get_vertices())
                                .view(np.float32)
                                .reshape(depth.get_height(), depth.get_width(), 3)
                                * 1000.0)

                            def overlay_project(p):
                                color_point = rs.rs2_transform_point_to_point(
                                    depth_to_color, [float(x) for x in p])
                                return rs.rs2_project_point_to_pixel(
                                    color_intr, color_point)

                            def overlay_project_points(points):
                                cp = np.asarray(points, float) @ R_dc.T + t_dc_mm
                                zc = cp[:, 2]
                                return np.column_stack([
                                    cp[:, 0] * float(color_intr.fx) / zc
                                    + float(color_intr.ppx),
                                    cp[:, 1] * float(color_intr.fy) / zc
                                    + float(color_intr.ppy),
                                ])

                            def overlay_transform_points(points):
                                return np.asarray(points, float) @ R_dc.T + t_dc_mm

                            def depth_deproject_points(pixels, depths_mm):
                                uv = np.rint(np.asarray(pixels, float)).astype(int)
                                uv[:, 0] = np.clip(uv[:, 0], 0, depth.get_width() - 1)
                                uv[:, 1] = np.clip(uv[:, 1], 0, depth.get_height() - 1)
                                return depth_vertices_mm[uv[:, 1], uv[:, 0]]

                            payload = scan_plane_telemetry(
                                np.asanyarray(depth.get_data()), intr, depth_unit_mm,
                                overlay_project=overlay_project,
                                overlay_project_points=overlay_project_points,
                                overlay_transform_points=overlay_transform_points,
                                depth_deproject_points=depth_deproject_points,
                                overlay_size=(color.get_width(), color.get_height()))
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
    """Burst capture: buffer aligned depth+color frames on the Jetson, then transfer
    them all in one burst — so the robot tour isn't stalled by a per-pose depth
    transfer over the cell's Wi-Fi (a full depth+color frame can take 6-11 s).

    Protocol on this connection (newline-terminated commands; length-prefixed replies):

        (server first sends ``BURST READY\\n`` so the client can confirm support)
        CAP    -> grab + align + filter + compress ONE frame into a RAM buffer; reply
                  ``<I idx><I thumb_len><thumb JPEG>`` (a small color thumbnail for the
                  client's live per-pose strip). thumb_len=0 signals a skip.
        GET    -> reply ``<I count>`` then each buffered frame as
                  ``<I depth_len><I color_len><d ts> + depth(lz4 .npy) + color(JPEG)``
                  (identical per-frame framing to the normal stream, so the client
                  reuses its decoder).
        CLEAR  -> drop the buffer; reply ``<I 0>``.

    The buffer is RAM-only and is ALSO dropped when the connection ends (finally), so
    an abandoned/dropped burst leaves NO data on the Jetson — no disk garbage. Between
    CAP commands the robot is moving, so the command read uses a generous timeout."""
    jpeg = turbojpeg.TurboJPEG('/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0')
    buffer = []   # list of (depth_compressed: bytes, color_jpeg: bytes, ts: float)
    try:
        conn.sendall(b'BURST READY\n')
    except OSError:
        return
    print(f"Burst session opened with {addr}")
    try:
        while True:
            # The next command may be many seconds away (the robot is moving between
            # poses); wait generously rather than dropping the connection.
            conn.settimeout(180.0)
            cmd = _recv_line(conn).strip().upper()
            if not cmd:
                break                      # peer closed

            if cmd == b'CAP':
                depth = color = None
                tries = 0
                while (depth is None or color is None) and tries < 10:
                    depth, color, _ts = getFrames(pipeline, align, depth_filters)
                    tries += 1
                if depth is None or color is None or len(buffer) >= max_frames:
                    # signal a skip (no frame / buffer full) — index still advances
                    conn.sendall(struct.pack('<I', len(buffer)) + struct.pack('<I', 0))
                    continue
                depth_buffer = io.BytesIO()
                np.save(depth_buffer, depth)
                depth_buffer.seek(0)
                depth_compressed = lz4f.compress(depth_buffer.read())
                color_jpeg = jpeg.encode(color)
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


def setup_depth_filters():
    # Decimation Filter
    # decimation = rs.decimation_filter()
    # decimation.set_option(rs.option.filter_magnitude, 2)

    # Spatial Filter
    spatial = rs.spatial_filter()


    # Temporal Filter with the desired settings
    # smooth_alpha = 0.4
    # smooth_delta = 20
    # persistence_control = 7  # Valid in 1 / last 8
    #
    # temporal = rs.temporal_filter(smooth_alpha, smooth_delta, persistence_control)


    # Hole filling
    # hole_filling = rs.hole_filling_filter()

    return [spatial]

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen(5)
    print(f'Server ip-address: {get_ip_address()}:{port}')

    while True:
        conn, addr = server_socket.accept()
        threading.Thread(target=handle_client, args=(conn, addr),
                         name=f"camera-client-{addr[0]}:{addr[1]}",
                         daemon=True).start()

if __name__ == '__main__':
    print(f"Initiating Jetson-Realsense Wi-Fi Server with resolution {width}x{height}")
    try:
        pipeline, align = openPipeline()
        depth_unit_mm = (
            pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
            * 1000.0)
        depth_filters = setup_depth_filters()

        main()
    except Exception as e:
        print(f"Unexpected error: {e}")

