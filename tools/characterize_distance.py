"""characterize_distance.py — sweep candidate camera standoffs over a known
ChArUco board and pick the closest passing distance (d*), spec §5 (Phase 0).

Why this exists (spec §10): "characterization that isn't a button will not be
re-run." Without a tool, the measurement chain's error budget (RealSense +
hand-eye calibration + robot pose + processing) is validated once, by hand, and
then silently trusted forever. This script makes it a repeatable command that
stores a DATED, IDENTIFIABLE result (characterization/characterization-
YYYYMMDD.json, git-ignored — machine-specific measurement data, not source),
which modules/scan/service.py's lock_scan_surface reads back on every surface
lock so a missing or stale characterization is visible (or refused), not silent.

Interactive, operator-driven step-and-measure (spec §5): for each configured
distance, jog the real robot until the camera sits at that standoff over the
board, press Enter, and the tool captures N frames and folds them into a
DistanceTrial (tasni.core.characterize). After the normal-incidence sweep it
also prompts ONE oblique-incidence capture at the worst planned tilt — spec
§5's own tolerances must not be validated only at normal incidence — and
reports it as a separate sanity check (not part of the d* selection, since
choose_dstar's "closest distance" logic assumes one trial per distance at a
comparable geometry).

Run (RoboDK attached to the real cell; the operator jogs the pendant by hand):

    py -3.10 tools/characterize_distance.py --distances 300,400,500,600,800

HEADLESS IMPORTABILITY (required for tests/test_characterize.py to run without
any camera/robot/RoboDK): everything above `main()` is pure Python — no
CameraClient, no RdkIO/RdkSession, no cv2 window. Those are imported only
inside main() and its capture helpers, which are never called at import time.
`latest_characterization()` in particular is what the lock-side gate imports.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Force UTF-8 console so non-ASCII output (em dashes, °, §) doesn't mis-render
# or crash on Windows's default codepage — same fix as tools/jetson_probe.py.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tasni.core.characterize import (  # noqa: E402
    CHARACTERIZATION_DIR, choose_dstar, latest_characterization, summarize_distance_trial)
from tasni.core.config import load_config  # noqa: E402
from tasni.modules.calibration.charuco import CharucoTarget  # noqa: E402
from tasni.modules.scan.survey_contract import camera_calibration_id  # noqa: E402

# `latest_characterization` now lives in tasni.core.characterize (Task 16
# review, Finding 2 — see that module's docstring for why), re-exported here
# for backward compatibility: existing callers (and tests) that do
# `from tools.characterize_distance import latest_characterization` keep
# working unchanged. OUTPUT_DIR is likewise an alias for the same reason.
OUTPUT_DIR = CHARACTERIZATION_DIR

# Default error budget for choose_dstar (spec §5). These are conservative
# placeholders for a D435i eye-in-hand rig at a sub-metre working distance;
# tune per cell with the --max-*/--min-coverage flags once real sweep data is
# in hand. plane_max_mm/length_spread_mm are the two gates the review round
# added after the brief this tool was originally specced against was frozen —
# see the module-level DEFAULT_BUDGET dict below for the authoritative list.
DEFAULT_BUDGET = dict(
    max_rms_mm=1.0,
    max_plane_max_mm=3.0,
    max_height_repeat_mm=1.0,
    max_normal_repeat_deg=1.0,
    max_length_err_mm=1.0,
    max_length_spread_mm=1.0,
    min_coverage_frac=0.5,
)


# --------------------------------------------------------------------------
# Capture helpers. Pure numpy/cv2 (no hardware access themselves) — they take
# already-grabbed Frame objects, so they are unit-tested directly in
# tests/test_characterize.py (Task 16 review, Finding 3).
# --------------------------------------------------------------------------

# A full-res D435i frame backprojects to ~920k points, and plane_metrics RANSACs
# each capture 1000 times over ALL of them -- ~49 s per capture, ~4 min per stop,
# ~37 min of compute across a 9-stop sweep, with the operator standing at the cell
# the whole time. A dominant plane filling the frame is estimated just as well from
# a uniform subsample: RANSAC's inlier ratio is unchanged (it is a proportion, not
# a count), and the SVD refine that sets the final normal still sees tens of
# thousands of points. Deterministic stride, so a re-run reproduces exactly.
PLANE_FIT_MAX_POINTS = 60_000


def _subsample_for_plane_fit(pts: np.ndarray,
                             max_points: int = PLANE_FIT_MAX_POINTS) -> np.ndarray:
    """Uniformly thin ``pts`` to at most ``max_points`` (deterministic stride)."""
    n = len(pts)
    if n <= max_points:
        return pts
    return pts[::int(np.ceil(n / max_points))]


def _board_region_mask(shape, corners_px, *, margin_px: float = 6.0) -> np.ndarray:
    """Boolean mask of the detected ChArUco region, slightly dilated.

    THE critical step for a distance sweep. Every metric here is meant to describe
    the BOARD, but a frame also contains the table, the floor and the far wall, and
    the board's share of it shrinks as the square of the standoff: ~33% of frame
    area at 300 mm, ~1% at 1.5 m. Fit a plane to the unmasked frame and at range you
    are measuring the ROOM -- the residual becomes the scene's depth spread (hundreds
    of mm) and the median depth becomes the background, not the board. Both then
    degrade smoothly with distance, which looks exactly like a real physical result.
    """
    import cv2

    pts = np.asarray(corners_px, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts)
    if margin_px:
        c = hull.reshape(-1, 2).mean(axis=0)
        v = hull.reshape(-1, 2) - c
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        hull = (c + v * (1.0 + margin_px / np.maximum(norms, 1e-6))).reshape(-1, 1, 2)
    mask = np.zeros(tuple(shape[:2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 1)
    return mask.astype(bool)


def _reject_off_board_points(pts: np.ndarray, *, band_mm: float = 25.0, k_sigma: float = 6.0):
    """Drop points too far from the board's own plane to be board. Returns
    ``(kept, n_rejected)``.

    Even inside the detected ChArUco hull the RealSense returns a small fraction of
    spurious depths (stereo mismatches, alignment edge artifacts). They are rare,
    but ``plane_metrics`` computes its residual over EVERY point it is handed, so a
    couple of percent of pixels sitting ~0.5 m off the board dominates the result:
    the live cell reported 40 mm RMS on a sheet of paper that is flat to well under
    a millimetre.

    The band is deliberately loose (25 mm, versus real board flatness under ~5 mm):
    the job here is to exclude what physically cannot be the board, not to flatter
    the noise measurement. It is tilt-safe because the reference is the fitted
    plane, which follows the board's tilt, not a fixed depth. Non-finite points
    fail the comparison and are therefore rejected AND COUNTED, so contamination is
    reported rather than silently laundered.
    """
    from tasni.modules.scan.plane import fit_plane

    p = np.asarray(pts, dtype=float).reshape(-1, 3)
    if len(p) < 3:
        return p, 0, None
    n, c, _mask = fit_plane(p, distance=6.0)
    plane = (np.asarray(n, float), np.asarray(c, float))
    res = np.abs((p - c) @ n)
    # A FIXED band silently truncates the residual distribution once the real noise
    # approaches it, which is exactly what happens at grazing incidence: the live
    # cell's 20/30 deg captures pinned plane_max at 24.97/25.29 against a 25 mm band,
    # so their RMS was reported LOWER than the truth. Widen the band to the data's own
    # robust spread (MAD-based sigma) whenever that exceeds the floor, so genuinely
    # noisy-but-real board points survive and only wild outliers are cut.
    finite = res[np.isfinite(res)]
    if len(finite):
        sigma = 1.4826 * float(np.median(np.abs(finite - np.median(finite))))
        band_mm = max(float(band_mm), k_sigma * sigma)
    keep = res <= float(band_mm)                      # NaN -> False -> rejected
    return p[keep], int((~keep).sum()), plane


def _backproject_valid_mm(depth, K, depth_scale: float = 1000.0,
                          mask: "np.ndarray | None" = None, dist=None) -> np.ndarray:
    """Every valid (>0) depth pixel, backprojected to camera-frame millimetres.

    Mirrors modules/scan/service.py's own ``_backproject_depth`` convention:
    the raw uint16 depth array is already in millimetres, and dividing by
    ``depth_scale`` (default 1000.0, RealSense's metres-per-unit) then
    multiplying back by 1000.0 is an identity unless the server ever reports a
    different scale — kept explicit so this stays correct if that changes.

    ``mask``, when given, restricts the result to those pixels — pass the board
    region (see :func:`_board_region_mask`) so the metrics describe the board
    rather than whatever else is in frame.
    """
    d = np.asarray(depth, dtype=float)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    valid = d > 0
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(valid)
    if dist is not None and len(ys):
        # Matters for a TILTED capture: with the plane fronto-parallel a lateral
        # x/y error leaves z unchanged, so the residual is blind to it, but once the
        # plane is inclined that same error maps straight into the residual.
        z_mm = d[ys, xs] / float(depth_scale) * 1000.0
        xy = _undistort_uv(np.column_stack([xs, ys]).astype(np.float64), K, dist)
        return np.column_stack([xy[:, 0] * z_mm, xy[:, 1] * z_mm, z_mm])
    if len(ys) == 0:
        return np.zeros((0, 3), dtype=float)
    z_mm = d[ys, xs] / float(depth_scale) * 1000.0
    return np.column_stack([(xs - cx) / fx * z_mm, (ys - cy) / fy * z_mm, z_mm])


def _undistort_uv(uv, K, dist) -> np.ndarray:
    """Pixel(s) -> NORMALISED, undistorted camera coordinates (x/z, y/z).

    Backprojecting with the plain pinhole formula ``(u - cx) / fx`` assumes a
    distortion-free lens. This D435i's measured RGB coefficients are k1=0.115,
    k2=-0.239, which displaces a point at r/f = 0.46 (a board corner when the board
    fills the frame) by ~1.4% — 3.9 mm across a 288 mm board diagonal. That is
    almost exactly the length error the first cell sweep reported at close range,
    and it shrank with distance precisely because the board moved toward the
    principal point where distortion vanishes.
    """
    import cv2

    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, np.asarray(K, dtype=np.float64),
                              np.asarray(dist, dtype=np.float64).ravel())
    return out.reshape(-1, 2)


def _corner_on_plane_mm(u: float, v: float, K, dist, plane) -> "np.ndarray | None":
    """Locate a board corner by intersecting its ray with the FITTED BOARD PLANE.

    Strictly better than reading depth at the corner pixel, for two reasons the cell
    made obvious:

    1. **Quantization.** Depth arrives as uint16 MILLIMETRES, so a corner's z is an
       integer. ``_corner_point_mm`` takes the median of a 5x5 window of integers,
       which is itself an integer, so the same corner yields the identical z every
       frame — which is why ``length_spread`` was EXACTLY 0.000 at every stop. The
       metric had no resolution; it was reporting the quantizer, not the camera.
    2. **Corners are the worst place to sample depth** — they sit on black/white
       edges where stereo matching and depth-to-color alignment are least reliable.

    The plane is fitted from tens of thousands of board pixels, so it is accurate to
    well under the 1 mm quantum, and the corner's PIXEL position comes from ChArUco's
    sub-pixel detector. Intersecting the two uses the best of both and returns a
    continuous position. Returns ``None`` if the ray is parallel to the plane.
    """
    n, c = plane
    ray = np.array([*_undistort_uv([[float(u), float(v)]], K, dist)[0], 1.0])
    denom = float(ray @ n)
    if abs(denom) < 1e-9:
        return None
    return ray * (float(c @ n) / denom)


def _corner_point_mm(depth, K, u: float, v: float, depth_scale: float = 1000.0,
                     window: int = 2, dist=None) -> "np.ndarray | None":
    """Backproject one ChArUco corner pixel to camera-frame mm, using the
    median valid depth in a small window around it (robust to a single noisy
    pixel landing exactly on a corner)."""
    d = np.asarray(depth, dtype=float)
    h, w = d.shape
    y0, y1 = max(0, int(round(v)) - window), min(h, int(round(v)) + window + 1)
    x0, x1 = max(0, int(round(u)) - window), min(w, int(round(u)) + window + 1)
    patch = d[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    z_mm = float(np.median(valid)) / float(depth_scale) * 1000.0
    if dist is not None:
        x, y = _undistort_uv([[float(u), float(v)]], K, dist)[0]
        return np.array([x * z_mm, y * z_mm, z_mm])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return np.array([(float(u) - cx) / fx * z_mm, (float(v) - cy) / fy * z_mm, z_mm])


def _reference_corner_pair(board: CharucoTarget):
    """Two ChArUco inner-corner ids at opposite corners of the board's inner
    grid, plus their known board-frame separation (mm).

    No separate ruled/printed reference is needed: BoardConfig is already the
    single source of truth for the board's physical dimensions (the printable
    PDF renders at true size), so the board's OWN geometry is a valid known
    length. ``all_obj_points`` is ordered by charuco id (0..N-1, row-major),
    so id 0 and id N-1 sit at opposite corners of the inner grid — close to
    the board's full diagonal, a nicely large (low-relative-noise) length.
    """
    obj = board.all_obj_points
    id_a, id_b = 0, len(obj) - 1
    true_mm = float(np.linalg.norm(obj[id_b] - obj[id_a]))
    return id_a, id_b, true_mm


def capture_distance_trial(camera, board: CharucoTarget, cfg, distance_mm: float,
                           n_frames: int, *, timeout: float,
                           frame_interval_s: float = 0.0, sleep=time.sleep) -> "object":
    """Grab ``n_frames`` RGB-D pairs at the current (operator-jogged) standoff
    and fold them into one :class:`~tasni.core.characterize.DistanceTrial`.

    Plane points come from EVERY valid depth pixel per frame (the operator
    aims the whole frame at the board for this bench sweep); the trial's own
    ``summarize_distance_trial`` -> ``plane_metrics`` -> ``fit_plane`` already
    RANSACs its own inliers out of that, so no separate inlier pre-filter is
    needed here. Length samples come from the two ChArUco corners
    :func:`_reference_corner_pair` selects, when both are detected in a frame.
    ``coverage_frac`` is the mean fraction of the board's inner corners
    detected across the ``n_frames`` captures — how much of the board this
    distance actually let the detector see.

    ``distance_mm`` is the operator's TARGET standoff and is used only for error
    messages: the returned trial carries the standoff actually MEASURED from the
    captured depth, so you do not have to hit the nominal value precisely.

    Returns ``(trial, measured_tilt_deg)`` — the incidence angle is measured for the
    same reason, and is ``None`` only if no capture yielded a fittable plane.
    """
    K, depth_scale = cfg.camera.K, cfg.scan.depth_scale
    id_a, id_b, true_mm = _reference_corner_pair(board)
    total_corners = len(board.all_obj_points)

    plane_sets: list[np.ndarray] = []
    length_samples: list[tuple] = []
    coverage_samples: list[float] = []
    standoff_samples: list[float] = []
    rejected_fracs: list[float] = []
    pair_a: list[np.ndarray] = []
    pair_b: list[np.ndarray] = []

    detected_any = False
    window_t0 = time.monotonic()
    for i in range(max(1, int(n_frames))):
        # The Jetson builds its depth filter chain ONCE at startup
        # (server_unicast_syncronous.py:1092) and shares it across every client for
        # the life of the process, so the rs.temporal_filter() state never resets.
        # Back-to-back grabs of a static scene therefore return a converged, almost
        # identical stream -- which is why every repeatability metric read ~0.03 mm
        # and length_spread was EXACTLY 0.000. Spacing the frames does not
        # manufacture independence (a converged filter on a static scene stays
        # converged), but it does let slower variation -- thermal drift, exposure,
        # filter re-convergence, lighting -- into the sample instead of measuring
        # one instant five times.
        if i and frame_interval_s > 0:
            sleep(float(frame_interval_s))
        frame = camera.grab(with_depth=True, timeout=timeout)
        if frame.depth is None:
            continue

        # Detect FIRST: the detection defines which pixels are board, and every
        # plane metric below must be restricted to those. Without that restriction
        # the fit describes the whole scene (see _board_region_mask).
        found = board.detect_points(frame.color, min_corners=1)
        if found is None:
            coverage_samples.append(0.0)
            continue
        detected_any = True
        corners, ids, _obj = found
        ids_flat = [int(v) for v in ids.flatten().tolist()]
        coverage_samples.append(len(ids_flat) / float(total_corners))

        mask = _board_region_mask(np.asarray(frame.depth).shape, corners.reshape(-1, 2))
        pts = _subsample_for_plane_fit(
            _backproject_valid_mm(frame.depth, K, depth_scale, mask=mask,
                                  dist=cfg.camera.dist))
        pts, n_rejected, board_plane = _reject_off_board_points(pts)
        rejected_fracs.append(n_rejected / max(n_rejected + len(pts), 1))
        if len(pts) >= 3:
            plane_sets.append(pts)
            # What standoff this capture ACTUALLY happened at, measured rather than
            # assumed. The operator jogs by hand, so the nominal --distances value is
            # a target, not an achievement; recording the target would silently
            # mislabel the whole distance-vs-quality curve that d* is chosen from.
            standoff_samples.append(float(np.median(pts[:, 2])))

        px_by_id = {cid: corners[i, 0] for i, cid in enumerate(ids_flat)}
        if id_a in px_by_id and id_b in px_by_id:
            if board_plane is not None:
                pa = _corner_on_plane_mm(*px_by_id[id_a], K, cfg.camera.dist, board_plane)
                pb = _corner_on_plane_mm(*px_by_id[id_b], K, cfg.camera.dist, board_plane)
            else:   # no usable plane this frame — fall back to raw corner depth
                pa = _corner_point_mm(frame.depth, K, *px_by_id[id_a], depth_scale,
                                      dist=cfg.camera.dist)
                pb = _corner_point_mm(frame.depth, K, *px_by_id[id_b], depth_scale,
                                      dist=cfg.camera.dist)
            if pa is not None and pb is not None:
                # ACCUMULATE across frames into one sample. Emitting one sample per
                # frame made length_spread_mm structurally zero: it is the std of the
                # pair distances WITHIN a sample, and a sample holding a single pair
                # has no spread by definition. That is why the metric read exactly
                # 0.000 at every stop of every sweep -- it was never measuring the
                # camera, and no amount of frame spacing could have changed it.
                pair_a.append(pa)
                pair_b.append(pb)

    if not plane_sets:
        why = ("the board was never detected" if not detected_any else
               "the board was detected but carried no valid depth")
        raise RuntimeError(
            f"distance {distance_mm:.0f} mm: {why} across {n_frames} frame(s) — the "
            "metrics must be measured on the board, so this distance cannot be "
            "characterized. Move closer, improve lighting, or drop this distance.")

    if pair_a:
        # One sample holding EVERY frame's pair: length_err_mm is then the bias of
        # their mean against the true length, and length_spread_mm their genuine
        # frame-to-frame repeatability — which is what both metrics always claimed.
        length_samples = [(np.asarray(pair_a).reshape(-1, 3),
                           np.asarray(pair_b).reshape(-1, 3), true_mm)]
    if not length_samples:
        # summarize_distance_trial folds an empty list into length_err_mm = NaN, and
        # choose_dstar rejects any non-finite metric -- so this trial is already
        # doomed. Say why now, at the cell, instead of letting it read as a quality
        # failure in the verdict later.
        print("   !! no known-length sample: the two OPPOSITE inner corners of the "
              "board must BOTH be visible. This capture can never pass — re-aim so "
              "the whole board is in frame and repeat this stop.")
    coverage_frac = float(np.mean(coverage_samples)) if coverage_samples else 0.0
    measured_mm = float(np.median(standoff_samples)) if standoff_samples else float(distance_mm)
    bad = float(np.mean(rejected_fracs)) if rejected_fracs else 0.0
    if bad > 0.10:
        # Not fatal, but it IS the headline fact about this capture: a tenth of the
        # board returning nonsense depth says more about the distance than any
        # residual computed from what is left.
        print(f"   !! {bad:.0%} of board pixels were off-plane garbage (excluded). "
              "Depth quality is poor here — treat this distance with suspicion.")
    trial = summarize_distance_trial(measured_mm, plane_sets, length_samples, coverage_frac)
    window_s = time.monotonic() - window_t0
    if len(pair_a) > 1 and trial.length_spread_mm == 0.0:
        # Degenerate repeatability is not a good result, it is an ABSENT one. Say so
        # here rather than let a 0.000 mm spread be read as astonishing precision and
        # copied into a budget nothing can ever fail.
        print(f"   !! {len(pair_a)} length samples over a {window_s:.1f} s window all "
              "measured IDENTICALLY (spread 0.000 mm). With independent frames that is "
              "not plausible — suspect the depth stream is repeating (a converged "
              "temporal filter, or a stalled feed). Treat repeatability as UNMEASURED.")
    return trial, _measured_tilt_deg(plane_sets)


def _aim_overlay(img, board, id_a: int, id_b: int, total_corners: int):
    """Draw the aiming HUD on a preview frame. Returns ``(img, ready)``.

    Pure drawing over an already-grabbed colour frame, so the capture logic and the
    preview agree by construction: it reports the SAME things the capture will
    measure — how much of the board is detected, and whether the two opposite
    reference corners are both visible (without them the known-length sample is
    absent and the whole stop is void).
    """
    import cv2

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.drawMarker(img, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 28, 1)

    found = board.detect_points(img, min_corners=1)
    coverage, have_ref, centre = 0.0, False, None
    if found is not None:
        corners, ids, _obj = found
        ids_flat = [int(v) for v in ids.flatten().tolist()]
        coverage = len(ids_flat) / float(total_corners)
        have_ref = id_a in ids_flat and id_b in ids_flat
        pts = corners.reshape(-1, 2).astype(np.int32)
        hull = cv2.convexHull(pts)
        cv2.polylines(img, [hull], True, (0, 255, 0) if have_ref else (0, 165, 255), 2)
        for p in pts:
            cv2.circle(img, tuple(p), 2, (0, 255, 0), -1)
        centre = tuple(int(v) for v in pts.mean(axis=0))
        cv2.drawMarker(img, centre, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
        cv2.line(img, (cx, cy), centre, (255, 0, 255), 1)

    ready = bool(found is not None and have_ref and coverage >= 0.99)
    off = None if centre is None else np.hypot(centre[0] - cx, centre[1] - cy)
    lines = [
        f"corners {coverage:.0%}" + ("" if found is not None else "  (BOARD NOT DETECTED)"),
        ("reference corners VISIBLE" if have_ref else
         "reference corners MISSING - this capture would be void"),
        "centred" if off is None else f"off-centre {off:.0f} px",
        "SPACE/ENTER = capture    q = abort",
    ]
    for i, text in enumerate(lines):
        colour = (0, 255, 0) if ready or i == 3 else (0, 165, 255)
        cv2.putText(img, text, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    colour, 1, cv2.LINE_AA)
    return img, ready


def _aim_prompt(camera, board, cfg, message: str, id_a: int, id_b: int,
                total_corners: int) -> None:
    """Live colour preview with the aiming HUD; blocks until SPACE/ENTER.

    Exists because the sweep needs sole use of a UNICAST camera, so the operator
    cannot keep the Tasni preview open to aim with — without this they are jogging
    blind. Colour-only streaming (no depth payload) keeps it realtime: a full
    depth+colour grab is ~5 s over this Wi-Fi link, which would be useless for aiming.

    Falls back to the plain text prompt if no GUI/stream is available, so nothing
    here can block a headless run.
    """
    import cv2

    print(message)
    print("   (aiming window open — SPACE/ENTER to capture, q to abort)")
    # h264 first: the Nano's HARDWARE encoder. Measured on this cell's Wi-Fi:
    # 24.9 fps / 31 ms p50, versus 8.3 fps / 156 ms for jpeg q60 — the q60 limiter is
    # the Nano's CPU JPEG encoder, not the link (q60 uses only ~4 Mbit/s and arrival
    # keeps pace with production). jpeg q35 is the fallback for a host without PyAV
    # or a server predating h264; the text prompt remains the fallback of last resort.
    for codec_kwargs in ({"codec": "h264", "bitrate": 4000},
                         {"color_only": True, "quality": 35}):
        try:
            with camera.stream(timeout=cfg.camera.timeout_s, **codec_kwargs) as stream:
                win = "characterize — aim the board, SPACE to capture"
                while True:
                    frame = stream.read(drain=True)
                    img, _ready = _aim_overlay(np.ascontiguousarray(frame.color), board,
                                               id_a, id_b, total_corners)
                    cv2.imshow(win, img)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (32, 13, 10):               # SPACE / ENTER
                        break
                    if key in (ord("q"), 27):             # q / ESC
                        cv2.destroyWindow(win)
                        raise SystemExit("aborted at the aiming step")
                cv2.destroyWindow(win)
                cv2.waitKey(1)
            return
        except SystemExit:
            raise
        except Exception as e:
            print(f"   ({codec_kwargs} preview failed: {e} — trying the next option)")
    print("   (no aiming preview available — falling back to the text prompt)")
    _prompt("   Press Enter when steady...")


def _prompt(message: str) -> None:
    """``input()`` with an actionable message when there is no interactive stdin.

    This sweep is operator-driven step-and-measure: it MUST be able to block while
    you jog the robot. Run through a wrapper that closes stdin (Claude Code's ``!``
    prefix, a CI step, a piped/redirected shell) and the first prompt raises a bare
    EOFError traceback that says nothing about the real problem.
    """
    try:
        input(message)
    except EOFError:
        print("\n\nerror: no interactive terminal (stdin is closed), so this sweep "
              "cannot prompt you to jog between captures.\n"
              "Run it directly in a PowerShell/terminal window rather than through a "
              "wrapper that redirects stdin.")
        raise SystemExit(2)


def _measured_tilt_deg(plane_sets) -> "float | None":
    """Incidence angle actually achieved, from the captured depth (deg).

    0 = fronto-parallel, 90 = edge-on: the angle between the camera's optical axis
    (+Z in the camera frame these points are in) and the fitted plane normal — the
    same definition depth_gate's HUD tilt lamp uses.

    Exists for the same reason the standoff is measured rather than assumed: the
    operator tilts a board by hand, so the ``--tilts`` value is a target. Recording
    the target would report a permitted incidence band that was never tested.
    Median over a few captures, so one bad fit cannot move it.
    """
    from tasni.modules.scan.plane import fit_plane

    tilts = []
    for pts in list(plane_sets)[:3]:
        p = np.asarray(pts, dtype=float).reshape(-1, 3)
        if len(p) < 3:
            continue
        n, _centroid, _mask = fit_plane(p, distance=6.0)
        nz = float(n[2]) / max(float(np.linalg.norm(n)), 1e-9)
        tilts.append(float(np.degrees(np.arccos(min(1.0, abs(nz))))))
    return float(np.median(tilts)) if tilts else None


def _format_trial(label: str, trial) -> str:
    # Ambiguity resolution #5 (inherited from characterize.py's own docstring
    # caveat): never print height_repeat_mm without normal_repeat_deg next to
    # it — two exactly-opposite capture normals make height_repeat_mm read a
    # misleadingly confident 0.0 while normal_repeat_deg correctly reports 90.
    return (f"{label}: rms={trial.plane_rms_mm:.3f}mm plane_max={trial.plane_max_mm:.3f}mm "
           f"height_repeat={trial.height_repeat_mm:.3f}mm "
           f"normal_repeat={trial.normal_repeat_deg:.2f}deg "
           f"length_err={trial.length_err_mm:.3f}mm length_spread={trial.length_spread_mm:.3f}mm "
           f"coverage={trial.coverage_frac:.0%}")


def _parse_floats(raw: str, flag: str, example: str) -> "list[float]":
    """Parse a comma-separated ``--flag`` value into floats, rejecting empties.

    Pure (no hardware) so this validation is directly unit-testable on its own,
    independent of main()'s hardware imports. Raises ``ValueError`` with an
    operator-facing message for an empty/degenerate list (``""``, ``","``,
    ``",,,"``) instead of letting the caller silently sweep zero values and crash
    later with an unhandled ``IndexError`` (Task 16 review, Finding 4).
    """
    values = [float(x) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError(
            f"{flag} {raw!r} contains no usable values — pass a comma-separated "
            f"list, e.g. {flag} {example}")
    return values


def _parse_distances(raw: str) -> "list[float]":
    """Parse ``--distances`` into a list of mm floats (see :func:`_parse_floats`)."""
    return _parse_floats(raw, "--distances", "300,400,500")


# Budget that rejects nothing, for a FIRST (discovery) sweep on a new cell: the
# shipped DEFAULT_BUDGET is a placeholder, so gating against it on run one tells
# you only that the guess was wrong, not what the cell actually achieves. With
# this, every trial is recorded and the achieved envelope is printed, and you set
# a real budget from that. Runs made this way are tagged discovery=True in the
# JSON so the lock-side gate refuses to accept them as a validated envelope.
DISCOVERY_BUDGET = dict(
    max_rms_mm=float("inf"), max_plane_max_mm=float("inf"),
    max_height_repeat_mm=float("inf"), max_normal_repeat_deg=float("inf"),
    max_length_err_mm=float("inf"), max_length_spread_mm=float("inf"),
    min_coverage_frac=0.0,
)


def achieved_envelope(trials) -> dict:
    """Worst observed value of each metric across ``trials`` (best coverage).

    This is what a discovery sweep is FOR: the numbers a real budget should be
    derived from. Empty input gives an empty dict rather than raising, so a sweep
    that captured nothing still reports cleanly.
    """
    trials = list(trials)
    if not trials:
        return {}
    return {
        "max_rms_mm": max(t.plane_rms_mm for t in trials),
        "max_plane_max_mm": max(t.plane_max_mm for t in trials),
        "max_height_repeat_mm": max(t.height_repeat_mm for t in trials),
        "max_normal_repeat_deg": max(t.normal_repeat_deg for t in trials),
        "max_length_err_mm": max(t.length_err_mm for t in trials),
        "max_length_spread_mm": max(t.length_spread_mm for t in trials),
        "min_coverage_frac": min(t.coverage_frac for t in trials),
    }


def passing_incidence_range_deg(incidence_trials) -> "tuple[float, float] | None":
    """The contiguous tilt band, starting at the smallest tilt, that passes.

    ``incidence_trials`` is ``[(tilt_deg, passed), ...]``. Incidence quality
    degrades monotonically with tilt, so the useful answer is "up to N degrees",
    not a scattered set: this walks tilts in increasing order and stops at the
    first failure. Returns ``None`` when even the smallest tilt fails.
    """
    ordered = sorted(incidence_trials, key=lambda p: float(p[0]))
    band = []
    for tilt, passed in ordered:
        if not passed:
            break
        band.append(float(tilt))
    return (min(band), max(band)) if band else None


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep candidate camera distances over the ChArUco board "
                    "and pick d* (spec §5, Phase 0).")
    p.add_argument("--distances", default="300,400,500,600,800",
                   help="comma-separated standoffs (mm) to sweep")
    p.add_argument("--frames", type=int, default=5,
                   help="RGB-D frames captured per distance")
    p.add_argument("--frame-interval", type=float, default=0.4,
                   help="seconds between frames within one capture. The Jetson's "
                        "temporal filter is shared and never resets, so back-to-back "
                        "frames of a static scene are near-identical and the "
                        "repeatability metrics read as ~0. Spacing them samples slower "
                        "drift. 0 restores the old back-to-back behaviour")
    p.add_argument("--tilt", type=float, default=25.0,
                   help="worst planned tilt (deg) for the oblique-incidence sanity check "
                        "(ignored when --tilts is given)")
    p.add_argument("--tilts", default=None,
                   help="comma-separated incidence angles (deg) to sweep at d*, e.g. "
                        "0,10,20,30. Yields the permitted incidence-angle RANGE; without "
                        "it only the single --tilt sanity check is taken")
    p.add_argument("--no-preview", action="store_true",
                   help="skip the live aiming window and just prompt on the console")
    p.add_argument("--discovery", action="store_true",
                   help="first run on a new cell: accept every trial (no budget gating) "
                        "and report the achieved envelope to derive a real budget from. "
                        "Tagged in the JSON; will NOT satisfy the lock-side gate")
    p.add_argument("--max-rms", type=float, default=DEFAULT_BUDGET["max_rms_mm"])
    p.add_argument("--max-plane-max", type=float, default=DEFAULT_BUDGET["max_plane_max_mm"])
    p.add_argument("--max-height-repeat", type=float,
                   default=DEFAULT_BUDGET["max_height_repeat_mm"])
    p.add_argument("--max-normal-repeat", type=float,
                   default=DEFAULT_BUDGET["max_normal_repeat_deg"])
    p.add_argument("--max-length-err", type=float, default=DEFAULT_BUDGET["max_length_err_mm"])
    p.add_argument("--max-length-spread", type=float,
                   default=DEFAULT_BUDGET["max_length_spread_mm"])
    p.add_argument("--min-coverage", type=float, default=DEFAULT_BUDGET["min_coverage_frac"])
    p.add_argument("--config", default=None, help="path to a tasni.config.json override")
    return p.parse_args(argv)


def main(argv=None) -> None:
    # Hardware/UI imports live HERE, not at module scope, so this file stays
    # importable with no camera, no robot, and no RoboDK running (tests only
    # need latest_characterization above).
    from tasni.core.camera import CameraClient
    from tasni.core.config import RoboDKConfig
    from tasni.core.rdk_io import RdkIO
    from tasni.core.session import RdkSession

    args = _parse_args(argv)
    try:
        distances = _parse_distances(args.distances)
    except ValueError as e:
        print(f"error: {e}")
        raise SystemExit(2)
    try:
        tilts = (_parse_floats(args.tilts, "--tilts", "0,10,20,30")
                 if args.tilts else [float(args.tilt)])
    except ValueError as e:
        print(f"error: {e}")
        raise SystemExit(2)
    cfg = load_config(args.config)
    board = CharucoTarget(cfg.board)
    camera = CameraClient(cfg.camera)
    if args.discovery:
        budget = dict(DISCOVERY_BUDGET)
    else:
        budget = dict(
            max_rms_mm=args.max_rms, max_plane_max_mm=args.max_plane_max,
            max_height_repeat_mm=args.max_height_repeat,
            max_normal_repeat_deg=args.max_normal_repeat,
            max_length_err_mm=args.max_length_err,
            max_length_spread_mm=args.max_length_spread,
            min_coverage_frac=args.min_coverage)

    _id_a, _id_b, _true_mm = _reference_corner_pair(board)
    _total_corners = len(board.all_obj_points)

    def aim(message: str) -> None:
        """Prompt, with a live aiming window unless --no-preview."""
        if args.no_preview:
            _prompt(message + " Press Enter when steady...")
        else:
            _aim_prompt(camera, board, cfg, message, _id_a, _id_b, _total_corners)

    print(f"=== Distance characterization: {distances} mm, {args.frames} frames each ===")
    print(f"Board: {cfg.board.dictionary} {cfg.board.squares_x}x{cfg.board.squares_y} "
         f"@ {cfg.board.square_size_mm:.1f} mm squares")
    print("!! The board's OWN geometry is the known-length reference, so the PHYSICAL "
         "board must match those numbers exactly, or every length error is wrong.")
    if args.discovery:
        print("DISCOVERY MODE: no budget gating; the achieved envelope is reported at the "
              "end. This run will NOT satisfy the lock-side characterization gate.")

    session = None
    rdk_io = None
    try:
        session = RdkSession(RoboDKConfig(connection="attach"))
        rdk_io = RdkIO(session)
        rdk_io.rdk.Item("").Name()  # cheap round trip: force+verify the connection now
        print("RoboDK connected (attach) — logging the camera pose per distance.")
    except Exception as e:
        print(f"(no RoboDK connection — continuing without pose logging: {e})")
        session, rdk_io = None, None

    # "Also consider" (Task 16 review): everything from here on can raise
    # mid-sweep (a bad capture, Ctrl-C between prompts, a camera timeout) — a
    # try/finally ensures the RoboDK session is always released rather than
    # only on the happy path. Low-impact (it attaches to an operator-
    # controlled RoboDK instance, so nothing else is blocked by an open
    # handle), but cheap and clearly correct, so fixed rather than left.
    try:
        trials = []
        camera_T_per_distance: list = []
        for d in distances:
            aim(f"\n>> Jog the camera to a {d:.0f} mm standoff, fronto-parallel over the "
                f"ChArUco board.")
            trial, tilt_now = capture_distance_trial(
                camera, board, cfg, d, args.frames, timeout=cfg.camera.timeout_s,
                frame_interval_s=args.frame_interval)
            trials.append(trial)
            pose = None
            if rdk_io is not None:
                try:
                    pose = rdk_io.camera_pose_T().tolist()
                except Exception:
                    pose = None
            camera_T_per_distance.append(pose)
            drift = trial.distance_mm - d
            print("   " + _format_trial(
                f"target {d:.0f}mm -> MEASURED {trial.distance_mm:.0f}mm "
                f"({drift:+.0f}mm)", trial))
            if abs(drift) > 60.0:
                print(f"   note: {abs(drift):.0f} mm off the target. That is fine — the "
                      "MEASURED value is what gets recorded — but keep the sweep spread "
                      "out so the distances don't bunch together.")
            # These stops are supposed to be fronto-parallel; say so out loud, since a
            # tilted "distance" capture silently confounds distance with incidence.
            if tilt_now is not None:
                print(f"   incidence here: {tilt_now:.1f} deg" + (
                    "  <-- NOT fronto-parallel; re-level the board/camera and redo this "
                    "distance, or it confounds the distance curve" if tilt_now > 8.0 else ""))

        best = choose_dstar(trials, **budget)
        print("\n=== verdict ===")
        if best is None:
            print("NO distance in the sweep passed every budget criterion — widen the "
                 "sweep or loosen the budget flags and re-run.")
        else:
            print("d* = " + _format_trial(f"{best.distance_mm:.0f} mm", best))

        oblique_distance = best.distance_mm if best is not None else distances[-1]
        # Incidence sweep (plan Task 6): one oblique sample proves a point but not a
        # RANGE, and the planner needs a permitted incidence band to filter poses by.
        # Taken at d* so distance is held constant and tilt is the only variable.
        incidence: list = []
        for tilt in tilts:
            aim(f"\n>> Incidence check: tilt the board/camera to ~{tilt:.0f} deg at "
                f"~{oblique_distance:.0f} mm standoff — tolerances must not be validated "
                "only at normal incidence.")
            trial, measured_tilt = capture_distance_trial(
                camera, board, cfg, oblique_distance, args.frames,
                timeout=cfg.camera.timeout_s, frame_interval_s=args.frame_interval)
            passed = choose_dstar([trial], **budget) is not None
            # The MEASURED angle is authoritative, for the same reason the standoff is:
            # reporting a permitted band at angles that were never actually tested
            # would be a fabricated envelope. Fall back to the target only if no plane
            # could be fitted at all.
            actual = measured_tilt if measured_tilt is not None else float(tilt)
            # Under --discovery the budget rejects nothing, so "PASS" is vacuous —
            # printing it would be the same false green the discovery tag exists to
            # prevent.
            verdict = ("(no gating — discovery)" if args.discovery
                       else f"-> {'PASS' if passed else 'FAIL'} against the same budget")
            print("   " + _format_trial(
                f"target {tilt:.0f}deg -> MEASURED {actual:.1f}deg", trial) + " " + verdict)
            incidence.append({"tilt_deg": float(actual),
                              "target_tilt_deg": float(tilt),
                              "measured_tilt_deg": measured_tilt,
                              "trial": trial.to_dict(), "passed": bool(passed)})

        # Only meaningful when a real budget did the gating (see the verdict note
        # above): with DISCOVERY_BUDGET every tilt "passes", so a range computed from
        # that would claim an envelope nothing was ever tested against.
        band = (None if args.discovery else passing_incidence_range_deg(
            [(e["tilt_deg"], e["passed"]) for e in incidence]))
        if args.discovery:
            print("\nincidence: no permitted range — discovery mode gates nothing. "
                  "The rms-vs-tilt numbers above are the measurement; re-run with a "
                  "real budget to establish the range.")
        elif band is None:
            print("\nincidence: NO tilt in the sweep passed — the permitted range is unknown.")
        else:
            print(f"\nincidence: permitted range {band[0]:.0f}-{band[1]:.0f} deg "
                 f"(contiguous from the smallest tilt tested)")
        # Preserve the pre-existing single-sample key so older readers keep working.
        oblique_trial_dict = incidence[-1]["trial"] if incidence else None
        oblique_passed = bool(incidence[-1]["passed"]) if incidence else False

        envelope = achieved_envelope(trials)
        if args.discovery:
            print("\n=== achieved envelope (worst across the distance sweep) ===")
            for k, v in envelope.items():
                print(f"   {k} = {v:.3f}")
            print("Set a real budget from these + your process tolerance, then re-run "
                  "WITHOUT --discovery to record the official result.")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d")
        out_path = OUTPUT_DIR / f"characterization-{stamp}.json"
        payload = {
            "calibration_id": camera_calibration_id(cfg.camera),
            "date": datetime.now().isoformat(timespec="seconds"),
            "trials": [t.to_dict() for t in trials],
            "dstar_mm": best.distance_mm if best is not None else None,
            "budget": budget,
            # A discovery sweep gated nothing, so its dstar proves nothing. Tagged so
            # the lock-side gate can refuse to treat it as a validated envelope.
            "discovery": bool(args.discovery),
            # Targets vs what was achieved: every trial's distance_mm is measured, so
            # keep the requested list alongside it for traceability.
            "nominal_distances_mm": [float(d) for d in distances],
            # Repeatability is only interpretable alongside the window it was
            # sampled over -- see capture_distance_trial on the shared temporal filter.
            "frames_per_capture": int(args.frames),
            "frame_interval_s": float(args.frame_interval),
            "achieved_envelope": envelope,
            "incidence_sweep": incidence,
            "incidence_range_deg": list(band) if band else None,
            "oblique_check": {
                "tilt_deg": float(tilts[-1]), "distance_mm": oblique_distance,
                "trial": oblique_trial_dict, "passed": oblique_passed,
            },
            "robot_camera_T_mm": camera_T_per_distance,
            "board": {
                "dictionary": cfg.board.dictionary, "squares_x": cfg.board.squares_x,
                "squares_y": cfg.board.squares_y, "square_size_mm": cfg.board.square_size_mm,
                "marker_size_mm": cfg.board.marker_size_mm,
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    main()
