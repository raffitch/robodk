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

from tasni.core.characterize import choose_dstar, summarize_distance_trial  # noqa: E402
from tasni.core.config import load_config  # noqa: E402
from tasni.modules.calibration.charuco import CharucoTarget  # noqa: E402
from tasni.modules.scan.survey_contract import camera_calibration_id  # noqa: E402

OUTPUT_DIR = _REPO / "characterization"
FILENAME_GLOB = "characterization-*.json"

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


def latest_characterization(root) -> "dict | None":
    """The most recently DATED characterization JSON under ``root``, or
    ``None`` if ``root`` doesn't exist or holds no characterization files.

    "Most recent" is by FILENAME (``characterization-YYYYMMDD.json`` sorts
    lexicographically by date), not filesystem mtime — the date that matters
    is the one the operator measured on, which is encoded in the name, not
    whatever the OS happened to touch the file at (a copy/restore would lie).

    A malformed or unreadable JSON file is SKIPPED, never raised: this is
    called from modules/scan/service.py's lock_scan_surface on every surface
    lock (real hardware, mid-operation), so one corrupted file on disk must
    never turn a routine lock into a crash. It is treated exactly as if that
    dated measurement doesn't exist — the search falls through to the
    next-newest file, and returns None if none remain.
    """
    root = Path(root)
    if not root.is_dir():
        return None
    for path in sorted(root.glob(FILENAME_GLOB), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return data
    return None


# --------------------------------------------------------------------------
# Capture helpers. Pure numpy/cv2 (no hardware access themselves) — they take
# already-grabbed Frame objects, so they stay unit-testable in principle even
# though nothing in this module currently exercises them without hardware.
# --------------------------------------------------------------------------

def _backproject_valid_mm(depth, K, depth_scale: float = 1000.0) -> np.ndarray:
    """Every valid (>0) depth pixel, backprojected to camera-frame millimetres.

    Mirrors modules/scan/service.py's own ``_backproject_depth`` convention:
    the raw uint16 depth array is already in millimetres, and dividing by
    ``depth_scale`` (default 1000.0, RealSense's metres-per-unit) then
    multiplying back by 1000.0 is an identity unless the server ever reports a
    different scale — kept explicit so this stays correct if that changes.
    """
    d = np.asarray(depth, dtype=float)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    ys, xs = np.nonzero(d > 0)
    if len(ys) == 0:
        return np.zeros((0, 3), dtype=float)
    z_mm = d[ys, xs] / float(depth_scale) * 1000.0
    return np.column_stack([(xs - cx) / fx * z_mm, (ys - cy) / fy * z_mm, z_mm])


def _corner_point_mm(depth, K, u: float, v: float, depth_scale: float = 1000.0,
                     window: int = 2) -> "np.ndarray | None":
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
                           n_frames: int, *, timeout: float) -> "object":
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
    """
    K, depth_scale = cfg.camera.K, cfg.scan.depth_scale
    id_a, id_b, true_mm = _reference_corner_pair(board)
    total_corners = len(board.all_obj_points)

    plane_sets: list[np.ndarray] = []
    length_samples: list[tuple] = []
    coverage_samples: list[float] = []

    for _ in range(max(1, int(n_frames))):
        frame = camera.grab(with_depth=True, timeout=timeout)
        if frame.depth is None:
            continue
        pts = _backproject_valid_mm(frame.depth, K, depth_scale)
        if len(pts):
            plane_sets.append(pts)

        found = board.detect_points(frame.color, min_corners=1)
        if found is None:
            coverage_samples.append(0.0)
            continue
        corners, ids, _obj = found
        ids_flat = [int(v) for v in ids.flatten().tolist()]
        coverage_samples.append(len(ids_flat) / float(total_corners))
        px_by_id = {cid: corners[i, 0] for i, cid in enumerate(ids_flat)}
        if id_a in px_by_id and id_b in px_by_id:
            pa = _corner_point_mm(frame.depth, K, *px_by_id[id_a], depth_scale)
            pb = _corner_point_mm(frame.depth, K, *px_by_id[id_b], depth_scale)
            if pa is not None and pb is not None:
                length_samples.append((pa.reshape(1, 3), pb.reshape(1, 3), true_mm))

    if not plane_sets:
        raise RuntimeError(
            f"distance {distance_mm:.0f} mm: no valid depth captured across "
            f"{n_frames} frame(s) — check the camera/board alignment and retry")

    coverage_frac = float(np.mean(coverage_samples)) if coverage_samples else 0.0
    return summarize_distance_trial(distance_mm, plane_sets, length_samples, coverage_frac)


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


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep candidate camera distances over the ChArUco board "
                    "and pick d* (spec §5, Phase 0).")
    p.add_argument("--distances", default="300,400,500,600,800",
                   help="comma-separated standoffs (mm) to sweep")
    p.add_argument("--frames", type=int, default=5,
                   help="RGB-D frames captured per distance")
    p.add_argument("--tilt", type=float, default=25.0,
                   help="worst planned tilt (deg) for the oblique-incidence sanity check")
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
    cfg = load_config(args.config)
    board = CharucoTarget(cfg.board)
    camera = CameraClient(cfg.camera)
    distances = [float(x) for x in args.distances.split(",") if x.strip()]
    budget = dict(
        max_rms_mm=args.max_rms, max_plane_max_mm=args.max_plane_max,
        max_height_repeat_mm=args.max_height_repeat,
        max_normal_repeat_deg=args.max_normal_repeat,
        max_length_err_mm=args.max_length_err, max_length_spread_mm=args.max_length_spread,
        min_coverage_frac=args.min_coverage)

    print(f"=== Distance characterization: {distances} mm, {args.frames} frames each ===")
    print(f"Board: {cfg.board.dictionary} {cfg.board.squares_x}x{cfg.board.squares_y} "
         f"@ {cfg.board.square_size_mm:.1f} mm squares")

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

    trials = []
    camera_T_per_distance: list = []
    for d in distances:
        input(f"\n>> Jog the camera to a {d:.0f} mm standoff, fronto-parallel over the "
             f"ChArUco board. Press Enter when steady...")
        trial = capture_distance_trial(camera, board, cfg, d, args.frames,
                                       timeout=cfg.camera.timeout_s)
        trials.append(trial)
        pose = None
        if rdk_io is not None:
            try:
                pose = rdk_io.camera_pose_T().tolist()
            except Exception:
                pose = None
        camera_T_per_distance.append(pose)
        print("   " + _format_trial(f"d={d:.0f}mm", trial))

    best = choose_dstar(trials, **budget)
    print("\n=== verdict ===")
    if best is None:
        print("NO distance in the sweep passed every budget criterion — widen the "
             "sweep or loosen the budget flags and re-run.")
    else:
        print("d* = " + _format_trial(f"{best.distance_mm:.0f} mm", best))

    oblique_distance = best.distance_mm if best is not None else distances[-1]
    input(f"\n>> Oblique-incidence check: tilt the board/camera to ~{args.tilt:.0f} deg "
         f"(the worst planned tilt) at ~{oblique_distance:.0f} mm standoff — the spec "
         "requires tolerances not be validated only at normal incidence. "
         "Press Enter when steady...")
    oblique_trial = capture_distance_trial(camera, board, cfg, oblique_distance, args.frames,
                                           timeout=cfg.camera.timeout_s)
    oblique_passed = choose_dstar([oblique_trial], **budget) is not None
    print("   " + _format_trial(f"oblique @ {args.tilt:.0f}deg", oblique_trial)
         + f" -> {'PASS' if oblique_passed else 'FAIL'} against the same budget")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"characterization-{stamp}.json"
    payload = {
        "calibration_id": camera_calibration_id(cfg.camera),
        "date": datetime.now().isoformat(timespec="seconds"),
        "trials": [t.to_dict() for t in trials],
        "dstar_mm": best.distance_mm if best is not None else None,
        "budget": budget,
        "oblique_check": {
            "tilt_deg": args.tilt, "distance_mm": oblique_distance,
            "trial": oblique_trial.to_dict(), "passed": oblique_passed,
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
    if session is not None:
        session.close()


if __name__ == "__main__":
    main()
