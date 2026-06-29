"""Layered configuration for the platform.

Defaults live in the Pydantic models below (so the app runs with zero config).
A user JSON file (``tasni.config.json`` in the repo root, or a path passed to
:func:`load_config`) overrides any subset of fields. Secrets (Jetson password
etc.) are NOT stored here — they stay in ``secrets/jetson.env`` (git-ignored).

Pydantic (already a FastAPI dependency) gives type validation on load + on
override (``validate_assignment``) and a JSON schema the UI can render, while
preserving the previous JSON-override semantics (deep-merge, "unknown key" error).
Python 3.10 has no ``tomllib``, so we use JSON to avoid an extra dependency.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

# D435i color intrinsics per stream resolution, copied from the original
# AutoCalibrate macro (factory values read off this specific camera).
_DEFAULT_INTRINSICS: dict[str, list[list[float]]] = {
    "640x480": [[605.400024414062, 0, 326.824310302734],
                [0, 605.427429199219, 244.43229675293],
                [0, 0, 1]],
    "1280x720": [[908.100036621094, 0, 650.236450195312],
                 [0, 908.14111328125, 366.6484375],
                 [0, 0, 1]],
    "1920x1080": [[1362.15002441406, 0, 975.354675292969],
                  [0, 1362.21166992188, 549.97265625],
                  [0, 0, 1]],
}


class _Model(BaseModel):
    """Shared base: validate on assignment (so JSON overrides are type-checked)
    and forbid unknown keys (a typo'd config key is an error, not a silent no-op)."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class CameraConfig(_Model):
    """RealSense-over-TCP client settings (the Jetson camera server)."""

    # The Jetson camera host. The server binds 0.0.0.0:1024 (all interfaces), so
    # this is just whichever Jetson IP the workstation can reach — confirmed
    # 10.12.171.70 (the old 10.5.5.19 subnet is gone). Override per-machine in
    # tasni.config.json if the IP changes.
    ip: str = "10.12.171.70"
    port: int = 1024
    # The server streams color at 1280x720 (server_unicast_syncronous.py), so the
    # intrinsics must be the 720p K — a 1080p setting would skew distance/tilt.
    resolution: str = "1280x720"
    timeout_s: float = 10.0
    # resolution -> 3x3 color intrinsics K
    intrinsics: dict[str, list[list[float]]] = Field(
        default_factory=lambda: {k: [row[:] for row in v]
                                 for k, v in _DEFAULT_INTRINSICS.items()})
    dist_coeffs: list[float] = Field(default_factory=lambda: [0, 0, 0, 0, 0])

    @property
    def K(self) -> np.ndarray:
        """Camera matrix for the configured resolution."""
        return np.array(self.intrinsics[self.resolution], dtype=np.float64)

    @property
    def dist(self) -> np.ndarray:
        return np.array(self.dist_coeffs, dtype=np.float32).reshape(-1, 1)

    @property
    def size(self) -> tuple[int, int]:
        w, h = self.resolution.split("x")
        return int(w), int(h)


class BoardConfig(_Model):
    """ChArUco board geometry (eye-in-hand calibration target).

    This is the single source of truth: the printable PDF renders THESE exact
    dimensions at true physical size, so "what we print" always equals "what
    detection expects" — no matching step. The default fits A4 (landscape) 1:1.
    """

    dictionary: str = "DICT_6X6_250"
    squares_x: int = 8
    squares_y: int = 6
    square_size_mm: float = 30.0        # 8x30 = 240 mm wide -> fits A4 landscape
    marker_size_mm: float = 22.0
    paper_size: str = "A4"              # active print/detection profile


class RoboDKConfig(_Model):
    """How tasni talks to RoboDK and which cell items it drives."""

    robot_name: str = "KUKA KR150 R2700"
    # "attach": use the running RoboDK GUI instance (default). If it has no
    # station with this robot loaded, the app opens `station_path` into it (so
    # you don't end up driving an empty RoboDK). "isolated": private headless
    # instance that loads station_path — used by tests / when no GUI is open.
    connection: str = "attach"
    # The cell's RoboDK station; relative paths resolve against the repo root.
    station_path: str | None = "Tasni.rdk"
    station_name: str = "Tasni"          # station display name set after loading
    target_prefix: str = "Target"
    # The RealSense camera is mounted on the flange as a tool named "Realsense"
    # (with its 3D model) in Tasni.rdk. Calibration solves THIS tool's pose; it is
    # fixed, not user-selectable.
    camera_tool: str = "Realsense"
    # No taught home pose: the operator jogs the robot until the live aiming gate
    # is green, and the robot's *current* pose becomes the seed the calibration
    # poses orbit around (see CalibrationConfig gate knobs below).
    # "simulate" keeps the robot in RoboDK only; "run_robot" drives the real arm.
    # Calibration only makes sense on the real arm (the camera rides on it).
    run_mode: str = "run_robot"
    # robolink's default socket timeout is only 10 s — too short for the first
    # access to the 117 MB station (RoboDK is busy loading / recomputing, so the
    # bare existence queries don't return in time and the FIRST Connect "fails"
    # while a second click — by now the station is loaded — succeeds instantly).
    # Give the connection a generous timeout so the first load completes in one
    # click. Applied right after the socket opens, before any heavy query.
    connect_timeout_s: float = 120.0
    # A station saved with collision checking ON makes RoboDK recompute the whole
    # collision map on EVERY API call against the 117 MB cell — the chronic
    # slowness behind the timing-out first connect. Turn checking OFF right after
    # the station is loaded (the collision MAP / pair config is preserved; the app
    # re-enables checking only transiently for the pose filter + the dry tour).
    disable_collisions_on_connect: bool = True
    # Real-robot driver link. In "run_robot" mode RoboDK must be connected to the
    # physical KUKA controller or it reports the robot "offline" and refuses to
    # move — previously the operator had to link it by hand in RoboDK's "Connect
    # robot" panel before every run. With this on, /connect links the controller
    # (best-effort; the controller may be off) so the simulated robot then tracks
    # the real one and the seed pose Create-targets reads is the arm's ACTUAL pose;
    # the real run re-ensures the link and fails clearly if it can't establish it.
    connect_robot_on_connect: bool = True
    robot_ip: str = ""                 # blank -> use the IP stored on the robot item
    robot_connect_timeout_s: float = 10.0


class CalibrationConfig(_Model):
    """Calibration-module knobs (live gate + pose generation + capture + solve)."""

    settle_s: float = 0.4               # pause after MoveJ before grabbing a frame
    holdout_count: int = 3              # poses held out of the solve for validation
    refine: bool = True                 # post-solve reprojection-minimizing refinement
    # Two corner floors, deliberately decoupled. ``min_charuco_corners`` is the
    # DETECTION floor — used by the live aiming gate's DETECT lamp and the
    # authoritative generate-grab: "can we see the board well enough to aim". Keep
    # it low so aiming isn't punishing. ``min_charuco_corners_solve`` is the higher
    # SOLVE-ACCEPTANCE floor — a captured pose only contributes to the hand-eye solve
    # if it has at least this many corners, so each per-view PnP board pose (the B in
    # AX=XB) is well-constrained. A weak 6-corner view detects fine for aiming but
    # gives a noisy pose that drags the linear solve; requiring more for the solve
    # raises input quality without making the operator aim harder.
    min_charuco_corners: int = 6        # detection floor (aiming gate + generate grab)
    min_charuco_corners_solve: int = 12  # solve-acceptance floor (a pose must beat this to be used)

    # Capture: median several frames per pose (per-corner pixel median) to beat
    # per-frame blur/glare/sensor noise. 1 = single grab (the old behaviour).
    frames_per_pose: int = 5
    # Solver: "best" runs every OpenCV linear hand-eye method and keeps the one
    # with the lowest training reprojection — robust to TSAI's ~180deg-mount
    # singularity. Or force one of TSAI/PARK/HORAUD/ANDREFF/DANIILIDIS.
    solver_method: str = "best"
    # Validation split: "shuffle" (seeded, unbiased) vs "tail" (the last N poses —
    # the old behaviour, but systematically the widest-angle views). Plus optional
    # k-fold cross-validation RMS (cheap, linear-only; 0 disables).
    holdout_strategy: str = "shuffle"
    split_seed: int = 0
    cross_val_folds: int = 5
    # Diagnostic intrinsic check: re-estimate K/distortion from the captured board
    # corners and WARN if they diverge from the configured camera matrix. Never
    # feeds the solve (review-then-apply); set False to skip the extra solve.
    verify_intrinsics: bool = True

    # Auto intrinsic calibration: on a hand-eye run, if the cell has no calibrated
    # intrinsics yet (no "intrinsics applied" marker — still on factory K / zero
    # distortion), derive K + distortion from the SAME captured board views, apply
    # them, and recompute each view's board pose with the better model before the
    # hand-eye solve. Runs on every sufficiently covered calibration capture so an
    # old low-coverage marker can never suppress a better lens fit. Target generation
    # deliberately spreads the board over the image; no separate operator step.
    auto_intrinsics: bool = True
    intrinsics_fix_k3: bool = True      # fix the overfit-prone high-order radial term
    intrinsics_edge_fraction: float = 0.30  # board centers near 30%/70% of frame
    min_intrinsics_coverage: float = 0.75   # refuse target sets below this 4x3 grid coverage
    max_intrinsics_rms_px: float = 1.0      # reject a noisy intrinsic fit above this

    # Robust outlier rejection: after the initial solve, drop training views whose
    # per-view reprojection RMS is an outlier (a mis-detected board / bad pose drags
    # the linear solve), then re-solve on the survivors. Conservative by design — a
    # view is dropped only if it exceeds BOTH ``outlier_px`` absolute AND
    # ``outlier_factor`` x the median, so a clean capture loses nothing. Held-out
    # validation views are never dropped (that would bias the metric optimistically).
    reject_outliers: bool = True
    outlier_px: float = 3.0             # absolute per-view reproj RMS (px) floor for an outlier
    outlier_factor: float = 3.0         # ...and must also exceed this x the median view error

    # Live aiming gate: before any targets are created, the operator jogs the
    # robot until the board sits at the ideal distance and angle. These bands
    # define when each HUD lamp goes green; all must be green to create targets.
    ideal_distance_mm: float = 450.0    # target working distance (board <-> camera)
    distance_tol_mm: float = 80.0       # +/- band around ideal_distance_mm
    max_tilt_deg: float = 10.0          # seed view should be nearly fronto-parallel
    center_tol_mm: float = 40.0         # |x|,|y| under this counts as centred
    seed_min_board_area_frac: float = 0.10  # projected board bbox / image area
    seed_max_board_area_frac: float = 0.40  # avoid clipping/noisy edge poses
    seed_stable_s: float = 1.0          # all seed gates must remain green this long
    preview_fps: float = 6.0            # max live-gate publish rate
    preview_timeout_s: float = 4.0      # per-frame camera timeout while streaming
    # JPEG quality the Jetson encodes the live preview at (10-100). Lower = fewer
    # bytes over Wi-Fi = higher fps; aiming tolerates softness. One-shot captures
    # ignore this and use the server's default (high) quality for crisp corners.
    preview_jpeg_quality: int = 60
    # Live-preview codec: "jpeg" (per-frame TurboJPEG, the default — no extra deps)
    # or "h264" (the Nano's hardware NVENC encoder — ~10-20x less bandwidth and far
    # lower CPU/latency, decoded client-side via PyAV). H.264 is lossy + inter-frame
    # so it can soften ChArUco corners; it is the *preview* path only — one-shot
    # captures always use the JPEG/lossless path regardless of this setting. Using
    # "h264" requires PyAV on the workstation (`pip install av`) and a Jetson with
    # the GStreamer NVENC stack (present on JetPack 4.6.x).
    preview_codec: str = "jpeg"
    preview_h264_bitrate_kbps: int = 4000   # NVENC target bitrate for the preview

    @field_validator("preview_codec")
    @classmethod
    def _check_codec(cls, v: str) -> str:
        if v not in ("jpeg", "h264"):
            raise ValueError("preview_codec must be 'jpeg' or 'h264'")
        return v
    # HUD X/Y/Z jog hints are in the camera optical frame (X right, Y down, Z
    # forward). Flip an axis here if the pendant's TOOL axis runs the other way.
    jog_invert_x: bool = False
    jog_invert_y: bool = False
    jog_invert_z: bool = False

    # Auto pose generation: orbit the (gated) seed view in a cone (not a full dome)
    # so the board stays visible, with roll + distance variation for hand-eye
    # conditioning. The TasniCalib_* targets are left in the station to inspect.
    pose_count: int = 15                # reachable poses to capture
    # A wider cone => more diverse view-direction (tilt) axes => a better-conditioned
    # hand-eye solve (measured on the generator: 32deg -> axis-spread 0.10, 45deg ->
    # 0.17). 45deg keeps the board within reliable ChArUco detection. The cone tilts
    # supply rotation axes in the camera X-Y plane; roll about the optical axis
    # supplies the third (Z) axis. select_diverse() picks the kept set by full-
    # rotation farthest-point sampling, so BOTH the tilt and the roll spread are
    # maximized (not just the view direction). The motion_diversity metric reports it.
    cone_half_angle_deg: float = 45.0   # max view-angle change from the seed view
    roll_max_deg: float = 75.0          # roll spread about the optical axis (the 3rd rot axis)
    distance_jitter: float = 0.12       # +/- fraction of working distance
    look_distance_mm: float = 500.0     # fallback if the seed board distance unknown
    # Large A3 board profile (320x240 mm). Its extra image occupancy needs a longer
    # target radius than the 240x180 mm A4 board. More importantly, wide cone poses
    # must preserve board-plane clearance: a 450 mm radial pose at 45 deg has only
    # 318 mm perpendicular clearance and over-fills/crops the A3 board.
    large_board_width_mm: float = 300.0
    large_board_target_distance_mm: float = 600.0
    large_board_min_perpendicular_mm: float = 425.0
    large_board_distance_jitter: float = 0.08
    large_board_intrinsics_edge_fraction: float = 0.34
    large_board_min_full_views: int = 10
    # Visibility pre-filter: a pose can be reachable AND collision-free yet aim so
    # the board clips the frame edge (or leaves view), which only shows up as a
    # skipped capture after the robot has already driven there. Before writing
    # targets we project the board (corners derived from the seed detection) into
    # each candidate's image and drop poses where too little of the board lands
    # inside the frame. Pure pinhole (no distortion — adequate for a margin gate on
    # the low-distortion D4xx lens). So the pre-run guarantee becomes reachable +
    # collision-free + board-in-frame, not just the first two.
    visibility_filter: bool = True
    min_board_visible_frac: float = 0.99   # outer board corners must all land in-frame
    board_visible_margin_frac: float = 0.04  # inset the frame by this fraction (keep off the edge)
    # Derived board keep-out: the physical platform the board sits on is often bigger
    # than (or absent from) the station's CAD, so a pose that grazes it isn't caught
    # by the modelled collision objects. From the seed board detection we add a box
    # spanning the board footprint + a lateral margin, from just above the board down
    # toward the floor — a conservative stand-in for the platform — as a collision
    # object (``TasniBoardKeepout``). The baseline-relative screen then drops any pose
    # whose tool/arm enters it (the seed view sits well clear, so it's not in the
    # baseline). Tune the margin to cover your real platform.
    board_keepout: bool = True
    # Lateral margin beyond the board footprint — set it to how far your physical
    # platform extends past the board. ~300 mm was needed to catch the observed 8->9
    # transit dip in the test cell (the long turret tools swing ~300 mm out from the
    # board); reduce for a small pedestal. Generous is safe: only a tool/arm that
    # actually dips below the box top within this footprint trips it.
    board_keepout_margin_mm: float = 300.0
    board_keepout_above_mm: float = 10.0    # box top this far above the board surface
    board_keepout_depth_mm: float = 600.0   # how far down toward the floor the box extends
    # Drop generated poses where the robot (incl. mounted tooling like a spindle)
    # collides, checked in RoboDK SIMULATE before any target is written — so a pose
    # that would drive the spindle into the arm never becomes a TasniCalib_* target.
    # Needs the station's collision map set up; degrades to "not checked" (drops
    # nothing) where the build/station can't evaluate collisions. The dry tour
    # remains the final pre-run gate.
    collision_filter: bool = True
    # Station objects excluded from calibration collision checking. WallAll is a
    # visual envelope in this cell, not a physical obstacle the robot can hit.
    collision_ignore_objects: list[str] = ["WallAll"]
    # Exact collision pairs the operator has marked as modelling artifacts from the
    # Tasni UI. Format matches RdkIO.collision_pairs(), e.g.
    # "WallAll ↔ KUKA KR150 R2700:L2". These are disabled after RoboDK rebuilds the
    # collision map, so Recheck/Create targets do not require closing the app.
    collision_ignore_pairs: list[str] = []
    # Baseline-relative screening: a real cell reports many CONSTANT collisions even
    # at the safe pose the operator aimed from — the robot base overlapping a
    # pedestal, each flange tool touching the wrist it is bolted to, a parked
    # external axis clipping a wall. Counting *total* collisions then marks every
    # pose as "colliding", which used to trip the fallback below and ship the whole
    # set (the bug behind a spindle-into-arm target). Instead we record the colliding
    # PAIR-SET at the safe seed and reject a pose/transit only if it introduces a
    # pair NOT in that baseline — so constant artifacts are ignored and only genuine
    # new contact (a tool entering the pedestal, the spindle swinging into a link) is
    # dropped. Leave on; off restores the old total-count behaviour.
    collision_baseline_relative: bool = True
    # Joint-path samples per move when screening the SWEPT approach (start->pose and,
    # in the dry tour, pose->pose): a pose can rest clear yet bump an obstacle mid-
    # move (the 8->9 transit the operator reported). Endpoints + interior points are
    # each pair-checked against the baseline. More = finer (slower on the big station).
    collision_path_samples: int = 6
    # Also force-enable collision pairs between every flange-mounted body and every
    # static OBJECT in the station (the board pedestal, walls, cabinet, floor).
    # RoboDK omits tool<->object pairs by default the same way it omits tool<->own-
    # robot, so a tool dipping into the board's pedestal goes unseen. Safe with
    # baseline-relative on (constant overlaps are subtracted out).
    collision_obstacle_pairs: bool = True
    # With baseline-relative screening, genuinely-colliding poses are always dropped
    # and never shipped. This flag now only governs the case where collisions cannot
    # be evaluated at all (no station collision map): False = proceed with the
    # reachable poses and warn (inspect + dry-run first); True = refuse instead.
    collision_filter_hard_fail: bool = False
    # RoboDK's default collision map EXCLUDES a tool from colliding with its own
    # robot (the tool is the robot's child), so a flange-mounted spindle swinging
    # into a forearm link is never flagged — and the pose survives collision_filter
    # even though it self-collides. When True, target creation + the dry tour
    # force-enable collision pairs between EVERY body mounted on the flange (the
    # camera, the spindle, any other tool/object) and the robot's arm links, so
    # those tool-vs-arm self-collisions are actually caught. Modifies the live
    # collision map only (it is not written back to the .rdk).
    collision_self_pairs: bool = True
    # Trailing robot links — the wrist + the mounting flange the tools naturally
    # sit against — skipped when enabling tool<->arm pairs, so a tool isn't forever
    # "colliding" with the flange it is bolted to. 2 skips A5+A6 on a 6-axis arm,
    # leaving A1..A4 checked (A4 is the link tools were observed to hit). Raise it
    # if a bulky tool false-triggers against the wrist.
    collision_skip_wrist_links: int = 2


class ScanConfig(_Model):
    """Scan-module knobs (module #2): standoff gate + pose generation + capture +
    TSDF fusion + plane/rectangle/frame extraction.

    The scan reuses the *stored* hand-eye result (the camera tool offset +
    intrinsics) to register each view — it never runs calibration. Defaults target
    a small table at ~0.5 m standoff with a D435i.
    """

    target_prefix: str = "TasniScan_"   # generated scan poses (never reuse calib targets)

    # -- standoff gate (depth-based; a table has no ChArUco board) ----------
    # The operator jogs the camera to look down at the surface from a neutral
    # standoff; these bands decide when the HUD lamps go green (all green ->
    # Create targets). Distance is the median depth of a central image patch.
    ideal_distance_mm: float = 500.0    # fallback camera<->surface standoff
    distance_tol_mm: float = 50.0       # +/- band around the live planned standoff
    max_tilt_deg: float = 6.0           # surface normal must be close to fronto-parallel
    center_patch_frac: float = 0.25     # central image fraction sampled for depth/normal
    min_valid_depth_frac: float = 0.5   # >= this fraction of the patch must have valid depth

    # -- live preview while aiming ------------------------------------------
    # The video streams COLOR-ONLY (fast, like calibration); the depth gate
    # (distance + tilt) is refreshed on an interleave every ``gate_period_s`` (the
    # unicast camera can't stream fast color AND depth at once). So the preview
    # stays smooth while the standoff/tilt readout updates ~once a second.
    preview_fps: float = 6.0
    preview_timeout_s: float = 4.0
    preview_jpeg_quality: int = 60      # color-only preview JPEG quality (Wi-Fi)
    # The preview is a smooth COLOR-ONLY stream (same as calibration); the standoff +
    # tilt panels (RANGE · TILT · LEVEL) are shown after the depth check on Create
    # targets (it refuses + shows the tilt-fix if out of band, so you adjust and retry).
    # Kept for compatibility with older config files. The scan preview no longer
    # interleaves depth under any setting: doing so interrupts the unicast color
    # stream and produces the FPS/no-signal/timeout cycle. Create targets performs
    # the authoritative depth check instead.
    live_depth_gate: bool = False
    gate_period_s: float = 1.2          # deprecated compatibility field
    # Depth grabs are slow + variable over Wi-Fi (measured ~6-11 s), so the socket
    # timeout for a depth-bearing grab must be generous or poses fail/skip. Color-only
    # grabs/streaming keep the shorter ``camera.timeout_s``.
    grab_timeout_s: float = 25.0
    # HUD jog hints are in the camera optical frame (X right, Y down, Z forward).
    jog_invert_x: bool = False
    jog_invert_y: bool = False
    jog_invert_z: bool = False

    # -- pose generation (reuses the calibration cone+roll generator) -------
    # Orbit the gated standoff seed in a cone so the surface stays in view, with
    # roll + distance variation for viewing-angle diversity (better fusion).
    pose_count: int = 12                # reachable poses to capture
    cone_half_angle_deg: float = 40.0   # max view-angle change from the seed (down) view
    roll_max_deg: float = 30.0          # roll spread about the optical axis
    distance_jitter: float = 0.15       # +/- fraction of standoff
    look_distance_mm: float = 500.0     # fallback standoff if the gate distance is unknown

    # -- surface-aware scan planning (survey → planner → execute) -----------
    # Replaces fixed standoff/cone/count with values derived from the measured surface.
    # Old knobs (cone_half_angle_deg, pose_count, voxel_size_m) remain as fallbacks.
    accurate_min_mm: float = 300.0       # near edge of D435i accurate depth band
    accurate_max_mm: float = 800.0       # far edge; beyond -> reference mode (no tour/mesh)
    frame_margin: float = 1.05           # keep just enough border; closer standoff = better depth resolution
    survey_max_tilt_deg: float = 6.0     # survey squareness gate (tighter than max_tilt_deg)
    center_tol_mm: float = 30.0          # finite-platform centroid offset allowed
    edge_align_tol_deg: float = 5.0      # finite-platform edge yaw allowed
    voxel_k: float = 0.003               # voxel_size_m = standoff_mm * voxel_k, clamped
    voxel_min_m: float = 0.001           # finest voxel (small / close surfaces)
    voxel_max_m: float = 0.002           # coarsest voxel
    surface_type: str = "flat"           # "flat" | "raised" → cone/count preset
    flat_cone_deg: float = 18.0          # cone half-angle for flat surfaces
    flat_views: int = 12                 # view count for flat surfaces
    raised_cone_deg: float = 38.0        # cone half-angle for raised objects
    raised_views: int = 13               # view count for raised objects
    min_surface_coverage: float = 0.85   # warn if the chosen views tile < this fraction
                                         # of the surface footprint grid (a missed region)
    grid_target_px: int = 64             # desired on-screen grid cell (px) for live overlay
    # When the surface overruns the view (edges not fully framed) its real edges are
    # untrustworthy, so we stop fitting the board and project a GENERIC fixed work
    # square on the plane, centred on the camera reticle (the aim point). This is its
    # size; the run crops to it around the aim (bounded_work_plane).
    work_crop_mm: tuple[float, float] = (1000.0, 1000.0)
    # Legacy adaptive-crop knobs (superseded by the fixed work_crop_mm above); kept so
    # an existing tasni.config.json that still sets them loads without error.
    large_surface_crop_fraction: float = 0.75  # fraction of visible FOV used as work crop
    large_surface_crop_max_mm: float = 800.0   # cap either crop dimension

    # -- capture ------------------------------------------------------------
    settle_s: float = 0.4               # pause after MoveJ before grabbing
    frames_per_pose: int = 3            # depth frames per pose, median-fused before TSDF
    # Diagnostics: persist each pose's color + 16-bit depth + camera pose under
    # <run>/views/ so a camera-perspective coverage overlay can be built later.
    # Off by default (~tens of MB/run); enable for a coverage-debugging scan.
    save_views: bool = False
    # Burst capture (opt-in; needs the burst-capable Jetson server). Default
    # per-pose capture grabs depth+color over Wi-Fi at EACH pose (~6-11 s each),
    # stalling the robot between poses. Burst mode instead has the Jetson buffer
    # each pose's frame in RAM and transfer them all in ONE burst after the tour —
    # the robot tour isn't blocked on transfer and the (same total) bytes move in a
    # single efficient stream. The Jetson buffer is RAM-only and is dropped after a
    # successful transfer AND on disconnect, so no data is left on the device.
    # OFF until the burst-capable server is deployed: a pre-burst server would
    # mishandle the handshake, so the client probes for support and FALLS BACK to
    # per-pose grab if the server doesn't advertise it.
    burst_capture: bool = False

    # -- TSDF fusion (Open3D ScalableTSDFVolume) ----------------------------
    # Per-view RGBD is integrated with the camera pose as extrinsic; the volume is
    # a 3D weighted average -> denoised mesh. voxel_size drives resolution/cost.
    voxel_size_m: float = 0.0015        # 1.5 mm TSDF voxel fallback
    sdf_trunc_m: float = 0.008          # truncation distance (~4-8 voxels)
    depth_scale: float = 1000.0         # RealSense depth units -> metres (uint16 mm)
    depth_min_m: float = 0.2            # ignore depth nearer than this
    depth_max_m: float = 1.5            # ignore depth farther than this (table standoff)
    preview_max_points: int = 300000    # decimate the cloud before streaming to the viewer
    surface_mesh_spacing_m: float = 0.001  # dense flat output mesh grid (1 mm)

    # -- region of interest: isolate the work surface (the "top layer") ----
    # Without this, fusing every view captures the whole room and RANSAC locks
    # onto the FLOOR (the biggest plane), not the table. We crop the fused cloud to
    # a box around where the camera was aimed (the look-point = each view's optical
    # axis at its central depth, averaged) BEFORE fitting: a Z band a little below
    # the surface to well above it drops the floor/ceiling, and the XY radius drops
    # far walls/clutter. Generous XY by default so a normal table isn't clipped;
    # raise it for a bigger surface, or disable to fuse everything.
    roi_enabled: bool = True
    roi_radius_m: float = 1.0           # XY half-extent around the aim (2 m box)
    roi_below_m: float = 0.10           # keep this far below the surface (floor dropped)
    roi_above_m: float = 0.40           # keep this far above (objects on the surface)

    # -- plane + rectangle extraction (RANSAC on the fused cloud) -----------
    ransac_distance_m: float = 0.006    # plane inlier band
    ransac_n: int = 3
    ransac_iterations: int = 1000
    plane_downsample_m: float = 0.005   # voxel-downsample before RANSAC (speed/robustness)
    min_inlier_frac: float = 0.25       # plane must claim >= this fraction of points to trust

    # -- collision guard + dry tour (same semantics as calibration) --------
    collision_filter: bool = True
    collision_ignore_pairs: list[str] = []
    # Same soft default as calibration: if RoboDK's collision map reports so many
    # collisions that target creation would fail (common with stale/oversized wall
    # or fixture geometry), keep the reachable targets and make the operator inspect
    # them / dry-run before moving the real robot. Set True for strict refusal.
    collision_filter_hard_fail: bool = False
    collision_self_pairs: bool = True
    collision_skip_wrist_links: int = 2


class WebConfig(_Model):
    host: str = "127.0.0.1"
    port: int = 8000


class AppConfig(_Model):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    board: BoardConfig = Field(default_factory=BoardConfig)
    robodk: RoboDKConfig = Field(default_factory=RoboDKConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    scan: ScanConfig = Field(default_factory=ScanConfig)
    web: WebConfig = Field(default_factory=WebConfig)


def _merge(obj: BaseModel, data: dict[str, Any]) -> None:
    """Recursively overlay ``data`` onto a Pydantic model in place. Leaf
    assignments are validated (``validate_assignment``); an unknown key raises a
    clear error rather than being silently dropped."""
    valid = set(type(obj).model_fields)
    for key, value in data.items():
        if key not in valid:
            raise KeyError(f"Unknown config key: {key!r}")
        current = getattr(obj, key)
        if isinstance(current, BaseModel) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(obj, key, value)


def config_file_path() -> Path:
    """Path to the user override file (``tasni.config.json`` at the repo root)."""
    return Path(__file__).resolve().parents[2] / "tasni.config.json"


def save_overrides(updates: dict[str, Any]) -> Path:
    """Deep-merge ``updates`` into ``tasni.config.json`` (created if absent).

    Used to persist UI-driven changes — e.g. syncing the printed board's
    dimensions into the config so detection matches what was printed.
    """
    path = config_file_path()
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def merge(dst: dict, src: dict) -> None:
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                merge(dst[key], value)
            else:
                dst[key] = value

    merge(data, updates)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_config(path: str | Path | None = None) -> AppConfig:
    """Build an :class:`AppConfig`, overlaying an optional JSON file.

    With no ``path``, looks for ``tasni.config.json`` next to the repo root and
    uses it if present; otherwise returns pure defaults.
    """
    cfg = AppConfig()
    if path is None:
        candidate = Path(__file__).resolve().parents[2] / "tasni.config.json"
        path = candidate if candidate.exists() else None
    if path is not None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        _merge(cfg, data)
    return cfg
