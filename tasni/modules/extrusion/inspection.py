"""Where to put the camera to inspect the layer just printed — pure numpy.

The operator used to hand-teach one RoboDK target and pick it from a dropdown, so
the inspection distance was whatever that taught pose happened to be and the
cylinder landed wherever it landed in the frame. This module derives the pose
instead, from the same placement the cylinder itself is built on:

  * **Centred by construction.** The aim point is the cylinder axis at the top of
    the layer just deposited, and every candidate pose puts that point on the
    camera's +Z axis at exactly the standoff — so it projects onto the principal
    point regardless of which candidate the reachability search ends up taking.
  * **Distance from the camera's own optics, not a magic number.** The standoff is
    the pinhole fit-to-frame distance (identical rule to
    :func:`tasni.modules.scan.planner.plan_scan`), then clamped into the D435i's
    accurate depth band. Anchor: an A3 sheet fills this camera's frame at ~380-400
    mm on the real cell, and 297 mm * fy / H = 375 mm reproduces that from the
    intrinsics alone (the short side binds — see the test of the same name).
  * **Fronto-parallel first.** The 2026-08-13 cell characterization measured
    incidence costing about 4x what distance costs (29 deg of tilt multiplies plane
    RMS by 11; tripling the distance only triples it), so the candidate order starts
    straight down and only tilts when the robot cannot reach that.

At cylinder scale the near limit, not framing, is what binds: a 86 mm wall wants
138 mm to fill the frame, which is inside the sensor's blind zone, so the standoff
clamps up to the accurate band's near edge and the object simply stays small in
frame. That is reported (``clamped_to``), never silently applied.

Pure numpy — no RoboDK, no camera — so it runs in preflight, in the API preview
before anything is connected, and in unit tests.
"""
from __future__ import annotations

import math

import numpy as np


def cylinder_diameter_mm(recipe) -> float:
    """Outside of the deposited wall — what actually has to fit in the frame."""
    return float(2.0 * recipe.radius_mm + recipe.bead_diameter_mm)


def framing_standoff(*, width_mm: float, height_mm: float, K: np.ndarray,
                     size_px: tuple[int, int], frame_margin: float,
                     near_mm: float, far_mm: float) -> dict:
    """Camera distance that frames a ``width_mm`` x ``height_mm`` object.

    ``d_fit_mm`` is the bare pinhole answer (before the depth band is applied);
    ``standoff_mm`` is that clamped into ``[near_mm, far_mm]``, with ``clamped_to``
    naming which edge moved it. ``fits`` is False only when the object cannot be
    framed *within* the band at all — that case is refused by callers rather than
    answered by backing the camera out past ``far_mm``, where depth quality is no
    longer characterized.
    """
    fx, fy = float(K[0][0]), float(K[1][1])
    width_px, height_px = int(size_px[0]), int(size_px[1])
    by_width = frame_margin * width_mm * fx / width_px
    by_height = frame_margin * height_mm * fy / height_px
    d_fit = max(by_width, by_height)
    standoff = float(np.clip(d_fit, near_mm, far_mm))
    clamped_to = ("near" if d_fit < near_mm else "far" if d_fit > far_mm else None)
    fill = {"width": float(width_mm * fx / (standoff * width_px)),
            "height": float(height_mm * fy / (standoff * height_px))}
    warnings: list[str] = []
    if clamped_to == "near":
        warnings.append(
            f"The object frames at {d_fit:.0f} mm, closer than the camera's accurate "
            f"near limit ({near_mm:.0f} mm), so the standoff is held there. It will "
            f"fill {fill['height'] * 100:.0f}% of the frame height.")
    elif clamped_to == "far":
        warnings.append(
            f"A {max(width_mm, height_mm):.0f} mm object needs {d_fit:.0f} mm to be "
            f"framed, past the {far_mm:.0f} mm end of the accurate depth band — "
            "depth quality is not characterized out there. Reduce the radius or "
            "inspect it in sections.")
    return {
        "d_fit_mm": float(d_fit), "standoff_mm": standoff, "clamped_to": clamped_to,
        "fits": bool(d_fit <= far_mm), "binding_axis": "width" if by_width >= by_height else "height",
        "fill_fraction": fill, "frame_margin": float(frame_margin),
        "near_mm": float(near_mm), "far_mm": float(far_mm), "warnings": warnings,
    }


def layer_top_z_mm(recipe, setup, layer_index: int) -> float:
    """Height of the *surface being measured* — the top of that layer's bead.

    ``generate_cylinder_plan`` puts a layer's centre line half a bead above the
    plane below it, so the deposited top is another half bead above that.
    """
    centre_z = (setup.build_plane_z_mm + recipe.bead_diameter_mm / 2.0
                + (layer_index - 1) * recipe.layer_height_mm)
    return float(centre_z + recipe.bead_diameter_mm / 2.0)


def aim_point_mm(recipe, setup, layer_index: int) -> np.ndarray:
    """The point the camera looks at: the cylinder axis at that layer's top."""
    return np.array([float(setup.center_x_mm), float(setup.center_y_mm),
                     layer_top_z_mm(recipe, setup, layer_index)], dtype=float)


FRAME_ROLL_REFERENCE = np.array([1.0, 0.0, 0.0])


def _roll_reference_axis(z_axis: np.ndarray, reference_x) -> np.ndarray:
    """Unit vector for roll zero, perpendicular to the optical axis.

    ``reference_x`` (work-frame coordinates) lets the caller measure roll from the
    ROBOT rather than from the work frame. That distinction is not cosmetic: on
    this cell the camera's own +X at the parked pose reads ``[-1, 0, 0]`` in the
    work frame, so the frame-fixed default put "roll 0" 179.7 deg from the robot's
    natural camera orientation, and RoboDK could only reach it through a wrist
    flip — every branch came back with |dA4| or |dA6| about 178 deg.

    Falls back to the frame axis when no reference is given, and when the one
    given projects to nothing because it lies along the optical axis.
    """
    if reference_x is not None:
        candidate = np.asarray(reference_x, dtype=float).reshape(3)
        projected = candidate - float(candidate @ z_axis) * z_axis
        if float(np.linalg.norm(projected)) > 1e-6:
            return projected / np.linalg.norm(projected)
    reference = FRAME_ROLL_REFERENCE
    if abs(float(reference @ z_axis)) > 0.9:         # degenerate: aim along +X
        reference = np.array([0.0, 1.0, 0.0])
    projected = reference - float(reference @ z_axis) * z_axis
    return projected / np.linalg.norm(projected)


def pose_from_aim(aim_mm, standoff_mm: float, *, tilt_deg: float = 0.0,
                  azimuth_deg: float = 0.0, roll_deg: float = 0.0,
                  reference_x=None) -> np.ndarray:
    """Camera pose (4x4, work frame) looking at ``aim_mm`` from ``standoff_mm``.

    OpenCV camera convention: +Z out of the lens, +X right in the image, +Y down.
    The camera sits on a cone of half-angle ``tilt_deg`` about the surface normal
    (``azimuth_deg`` picks which side), so the aim point stays exactly on the
    optical axis at exactly the standoff for every candidate — tilt trades
    incidence for reach, never centring or distance.

    ``reference_x`` decides what roll zero MEANS — see :func:`_roll_reference_axis`.
    """
    aim = np.asarray(aim_mm, dtype=float).reshape(3)
    tilt = math.radians(tilt_deg)
    azimuth = math.radians(azimuth_deg)
    # Unit vector from the aim point toward the camera (up and off to one side).
    away = np.array([math.sin(tilt) * math.cos(azimuth),
                     math.sin(tilt) * math.sin(azimuth),
                     math.cos(tilt)], dtype=float)
    z_axis = -away                                   # camera looks back down it
    x_axis = _roll_reference_axis(z_axis, reference_x)
    y_axis = np.cross(z_axis, x_axis)
    roll = math.radians(roll_deg)
    rolled_x = math.cos(roll) * x_axis + math.sin(roll) * y_axis
    rolled_y = np.cross(z_axis, rolled_x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = rolled_x, rolled_y, z_axis
    T[:3, 3] = aim + standoff_mm * away
    return T


def pose_candidates(aim_mm, standoff_mm: float, config,
                    reference_x=None) -> list[dict]:
    """Ordered poses to try: straight down first, tilted only as a fallback.

    Roll is tried before tilt because rotating about the optical axis costs the
    measurement nothing (the surface is still fronto-parallel) while giving the
    wrist a genuinely different configuration to reach — often the difference
    between an unreachable and a reachable pose on this KUKA.
    """
    rolls = [float(v) for v in config.inspection_roll_candidates_deg]
    tilts = [float(v) for v in config.inspection_tilt_candidates_deg]
    azimuths = [float(v) for v in config.inspection_azimuth_candidates_deg]
    ordered: list[tuple[float, float, float]] = [(0.0, 0.0, roll) for roll in rolls]
    for tilt in tilts:
        if tilt == 0.0:
            continue
        for azimuth in azimuths:
            ordered.extend((tilt, azimuth, roll) for roll in rolls)
    candidates = []
    for tilt, azimuth, roll in ordered:
        T = pose_from_aim(aim_mm, standoff_mm, tilt_deg=tilt,
                          azimuth_deg=azimuth, roll_deg=roll,
                          reference_x=reference_x)
        candidates.append({
            "tilt_deg": tilt, "azimuth_deg": azimuth, "roll_deg": roll,
            "xyz_mm": [float(v) for v in T[:3, 3]], "T": T,
        })
    return candidates


def star_view_angles(config) -> list[tuple[str, float, float]]:
    """The (name, tilt, azimuth) table for one multi-view star, top view first.

    A "Mercedes star": the top view as today plus one ring of the scan module's
    dome at a single cone angle. All views share the aim point and the standoff
    (pose_from_aim guarantees both), so the ring fills the frame identically
    from every side and no view is better framed than another.

    Azimuth is measured in the WORK frame from +X -- the axis the paired
    detection offset is expressed along -- so a star has the same orientation
    across takes and across sessions.
    """
    tilt = min(float(config.multiview_tilt_deg), float(config.multiview_max_tilt_deg))
    views = [("top", 0.0, 0.0)]
    for azimuth in config.multiview_azimuths_deg:
        views.append((f"star-{int(round(float(azimuth))):03d}", tilt, float(azimuth)))
    return views


def star_view_candidates(aim_mm, standoff_mm: float, config, *, tilt_deg: float,
                         azimuth_deg: float, reference_x=None) -> list[dict]:
    """Ordered candidate poses for ONE view of the star.

    Roll varies; tilt and azimuth never do. That asymmetry is the point:
    ``pose_candidates`` treats tilt and azimuth as FALLBACKS for an unreachable
    fronto-parallel pose, but here they are what the view IS. Substituting them
    would silently hand the merge two clouds of the same flank and none of the
    other, which no diagnostic downstream could detect.
    """
    candidates = []
    for roll in config.inspection_roll_candidates_deg:
        candidates.append({
            "T_work_camera": pose_from_aim(aim_mm, standoff_mm, tilt_deg=tilt_deg,
                                           azimuth_deg=azimuth_deg, roll_deg=roll,
                                           reference_x=reference_x),
            "tilt_deg": float(tilt_deg), "azimuth_deg": float(azimuth_deg),
            "roll_deg": float(roll)})
    return candidates


def multiview_plan(recipe, setup, *, K: np.ndarray, size_px: tuple[int, int],
                   config) -> dict:
    """Descriptors for the whole star: preview, the dry tour, and the job log."""
    aim = aim_point_mm(recipe, setup, 1)
    standoff = framing_standoff(width_mm=cylinder_diameter_mm(recipe),
                                height_mm=cylinder_diameter_mm(recipe),
                                K=K, size_px=size_px, config=config)
    return {"standoff_mm": float(standoff), "aim_mm": [float(v) for v in aim],
            "views": [{"name": name, "tilt_deg": tilt, "azimuth_deg": azimuth}
                      for name, tilt, azimuth in star_view_angles(config)]}


def inspection_plan(recipe, setup, *, K: np.ndarray, size_px: tuple[int, int],
                    config, reference_x=None) -> dict:
    """The complete derived inspection geometry for one cylinder — JSON-safe.

    Candidate *descriptors* only (no 4x4s), so this is what preflight and the API
    preview return; execution rebuilds the matrices with :func:`pose_candidates`
    from the same numbers.
    """
    diameter = cylinder_diameter_mm(recipe)
    framing = framing_standoff(
        width_mm=diameter, height_mm=diameter, K=K, size_px=size_px,
        frame_margin=config.inspection_frame_margin,
        near_mm=config.inspection_min_mm, far_mm=config.inspection_max_mm)
    standoff = framing["standoff_mm"]
    layers = []
    for index in range(1, recipe.layer_count + 1):
        aim = aim_point_mm(recipe, setup, index)
        layers.append({
            "layer_index": index,
            "top_z_mm": float(aim[2]),
            "aim_mm": [float(v) for v in aim],
            "camera_z_mm": float(aim[2] + standoff),
            "candidates": [{k: v for k, v in candidate.items() if k != "T"}
                           for candidate in pose_candidates(aim, standoff, config,
                                                            reference_x)],
        })
    # Roll zero is meaningless without saying what it is measured FROM, and the
    # two answers are 180 deg apart on this cell. Label it so a report cannot be
    # read backwards.
    used_reference = (FRAME_ROLL_REFERENCE if reference_x is None
                      else np.asarray(reference_x, dtype=float).reshape(3))
    return {
        "auto": bool(getattr(setup, "inspection_auto", False)),
        "roll_reference": "frame_x" if reference_x is None else "camera_at_start",
        "roll_reference_x": [float(v) for v in used_reference],
        "object_diameter_mm": diameter,
        "standoff_mm": standoff,
        "framing": framing,
        "work_frame": setup.work_frame,
        "inspection_tool": setup.inspection_tool,
        "layers": layers,
        "ok": bool(framing["fits"]),
        "warnings": list(framing["warnings"]),
    }


def order_candidates_seed_first(candidates: list[dict], seed: dict | None) -> list[dict]:
    """Move the candidate matching ``seed``'s (tilt, azimuth, roll) to the front.

    Rejections are usually constant across layers (straight-down collides with
    the same fixture at every height), so the previous layer's winner is by far
    the most likely first pass — trying it first collapses the per-layer search
    to one collision-validated attempt. Validation still gates every layer;
    only the search ORDER changes. Absent/unknown seeds return the list as-is.
    """
    if not seed:
        return candidates
    key = (float(seed.get("tilt_deg", 0.0)), float(seed.get("azimuth_deg", 0.0)),
           float(seed.get("roll_deg", 0.0)))
    for index, candidate in enumerate(candidates):
        if (float(candidate["tilt_deg"]), float(candidate["azimuth_deg"]),
                float(candidate["roll_deg"])) == key:
            return [candidate] + candidates[:index] + candidates[index + 1:]
    return candidates


def standoff_report(T_work_camera, aim_point_xyz_mm, depth, *,
                    patch_frac: float = 0.06, unit_mm: float = 1.0) -> dict:
    """Distance the pose says vs the distance the camera sees, to the aim point.

    The inspection pose is commanded to put the camera a known distance from the
    layer's aim point. If the arm is not actually there -- an unsettled read, a
    model that has drifted from the real controller -- then EVERY back-projected
    point is displaced by the difference, silently, because the depth frame itself
    is perfectly good. On the cell (2026-08-28) that displaced the ring 142 mm
    below the deposit height band and produced five identical "not enough
    deposited-geometry points" failures with no hint of the cause.

    ``measured_mm`` is the median of the valid depth in a central patch (the pose
    aims the optical axis at the aim point, so the frame centre is that point),
    scaled by ``unit_mm`` -- ``depth`` is raw camera WORDS, not millimetres
    (protocol 2's native depth is 0.1 mm/word; the caller passes
    ``frame.geometry.depth_unit_mm``). Zero/invalid depth pixels are excluded; a
    patch with none yields NaN, which :func:`standoff_fault` treats as a fault
    rather than a pass.
    """
    T = np.asarray(T_work_camera, dtype=float)
    camera = T[:3, 3]
    expected = float(np.linalg.norm(camera - np.asarray(aim_point_xyz_mm, dtype=float)))
    d = np.asarray(depth, dtype=float)
    h, w = d.shape
    ph, pw = max(1, int(h * patch_frac)), max(1, int(w * patch_frac))
    patch = d[h // 2 - ph:h // 2 + ph, w // 2 - pw:w // 2 + pw]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    measured = float(np.median(valid)) * float(unit_mm) if valid.size else float("nan")
    return {"expected_mm": round(expected, 1), "measured_mm": round(measured, 1),
            "delta_mm": round(measured - expected, 1), "samples": int(valid.size)}


def standoff_fault(report: dict, tolerance_mm: float) -> "str | None":
    """An operator-actionable message when pose and depth disagree, else None."""
    measured, delta = report["measured_mm"], report["delta_mm"]
    if not np.isfinite(measured):
        return ("inspection depth frame has no valid samples at the aim point — "
                "cannot confirm the camera reached the commanded standoff")
    if abs(delta) > float(tolerance_mm):
        return (f"camera is {delta:+.0f} mm from where it was commanded "
                f"(commanded {report['expected_mm']:.0f} mm to the aim point, "
                f"camera sees {measured:.0f} mm; tolerance {tolerance_mm:.0f} mm). "
                "The robot is not where RoboDK thinks it is — check that the arm "
                "settled at the inspection pose and that the work frame is correct. "
                "Measuring now would displace every point by this amount.")
    return None
