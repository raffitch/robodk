"""The scan → extrusion handoff: place the cylinder on the middle of the platform.

The Scan module inserts a work frame (``Tasni Work Frame``), a centre frame
(``Tasni Work Center``), and the oriented rectangle it measured, and records the
applied run in ``runs/scan/active.json``. This module turns those into a placement,
so a trial is centred on the *measured platform* instead of on whatever the selected
frame's origin happens to be. That distinction matters: the scan deliberately puts
its work-frame origin on the rectangle **corner** nearest the robot base, so the
default (0, 0) placement sits on a corner of the table, not in the middle of it.

:func:`resolve_platform_center` is the live path, and it reads the STATION first --
the centre frame, else the surface object's own mesh -- falling back to the run
pointer on disk only last. RoboDK saves that geometry with the station, so clearing
``runs/`` no longer orphans a placement whose surface is still in the tree.

Placement stays opt-in. A plan carries ``setup.scan_run_id`` only when it was
centred this way; manual/World placement is still legal and simply reports
``placement: "manual"``. When a plan *is* surface-placed, the checks here are
fail-closed: re-scanning the table changes the active run id and invalidates the
placement, because the surface the coordinates were derived from no longer exists.

Pure disk + numpy — no RoboDK — so it is unit-testable and can run before connect.
"""
from __future__ import annotations

import numpy as np

from ...core import runs
from ..scan.plane import rectangle_in_frame

SCAN_MODULE = "scan"

#: The frame the scan drops on the MIDDLE of the measured rectangle. Its origin IS
#: the platform centre, so centring on it needs no rectangle at all.
CENTER_FRAME_NAME = "Tasni Work Center"
#: The quad OBJECT the scan inserts. Its own mesh carries the four corners, which is
#: what lets a centre outlive a deleted ``runs/scan`` (see :func:`resolve_platform_center`).
SURFACE_OBJECT_NAME = "Tasni Work Surface"

#: Bounds are only honest while the rectangle is axis-aligned in the asked-for frame.
#: Compare the polygon's own area against its bounding box: they match for an aligned
#: rectangle and diverge as it rotates.
_ALIGNED_AREA_TOLERANCE = 0.01


def mesh_corners(points) -> np.ndarray | None:
    """Distinct corner XYZ from a RoboDK object mesh (``GetPoints`` XYZijk rows).

    ``add_rectangle`` writes the quad as four triangles with both windings, so the
    mesh is a 12-row triangle soup over 4 real corners. Returns ``None`` rather than
    a guess when the mesh is empty, non-finite, or too degenerate to be a surface.
    """
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] < 3:
        return None
    xyz = array[:, :3]
    if not np.isfinite(xyz).all():
        return None
    unique = np.unique(np.round(xyz, 6), axis=0)
    return unique if len(unique) >= 3 else None


def _polygon_area(xy: np.ndarray) -> float:
    """Shoelace area of the ring through ``xy``, ordered by angle about its mean."""
    order = np.argsort(np.arctan2(xy[:, 1] - xy[:, 1].mean(), xy[:, 0] - xy[:, 0].mean()))
    ring = xy[order]
    return float(abs(np.dot(ring[:, 0], np.roll(ring[:, 1], -1))
                     - np.dot(ring[:, 1], np.roll(ring[:, 0], -1))) / 2.0)


def platform_from_corners(corners) -> dict:
    """Centre (always) and axis-aligned extents (only when they are true) for ``corners``.

    The centre is the mean of the corners, exact under any rotation of the frame the
    corners are expressed in. The *bounds*, by contrast, only describe the surface
    while the rectangle is aligned with that frame's axes — in a rotated frame the
    bounding box is strictly larger than the platform, so reporting it would let a wall
    that overhangs the real table pass the fit check. Extents are withheld instead.
    """
    xyz = np.asarray(corners, dtype=float).reshape(-1, 3)
    center = xyz.mean(axis=0)
    result = {"center_mm": [round(float(center[0]), 6), round(float(center[1]), 6)],
              "center_z_mm": round(float(center[2]), 6),
              "bounds_mm": None, "size_mm": None, "extents_known": False}
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    box_area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
    if box_area <= 0.0:
        return result
    if abs(_polygon_area(xyz[:, :2]) - box_area) > _ALIGNED_AREA_TOLERANCE * box_area:
        return result
    result.update({
        "bounds_mm": {"x_min": float(lo[0]), "x_max": float(hi[0]),
                      "y_min": float(lo[1]), "y_max": float(hi[1])},
        "size_mm": [round(float(hi[0] - lo[0]), 6), round(float(hi[1] - lo[1]), 6)],
        "extents_known": True})
    return result


def resolve_platform_center(work_frame: str, *, center_frame_xyz=None,
                            surface_corners=None,
                            active: dict | None = None) -> dict | None:
    """The middle of the platform, expressed in ``work_frame`` — or ``None``.

    Sources, best first:

    1. ``center_frame_xyz`` — origin of :data:`CENTER_FRAME_NAME` read out of the
       station. It *is* the centre, so it needs no rectangle.
    2. ``surface_corners`` — corners of :data:`SURFACE_OBJECT_NAME`'s own mesh,
       already expressed in ``work_frame``.
    3. ``active`` — ``runs/scan/active.json``. Last resort, and only when it describes
       this very frame: its corners are recorded in the scan's frame and say nothing
       about any other one.

    Sources 1 and 2 live in the station, so a deleted ``runs/scan`` no longer orphans a
    placement whose geometry is still sitting in RoboDK. Only source 3 yields a
    ``run_id``: a station-derived centre has no run to claim and must not pretend
    otherwise, or the staleness check in :func:`surface_check` would compare against a
    scan this placement was never derived from.
    """
    extents = {"bounds_mm": None, "size_mm": None, "extents_known": False}
    surface: dict | None = None
    if surface_corners is not None:
        corners = np.asarray(surface_corners, dtype=float).reshape(-1, 3)
        if len(corners) >= 3 and np.isfinite(corners).all():
            surface = platform_from_corners(corners)
            extents = {key: surface[key] for key in extents}
    if center_frame_xyz is not None:
        origin = np.asarray(center_frame_xyz, dtype=float).reshape(-1)
        if origin.size >= 3 and np.isfinite(origin[:3]).all():
            # The centre frame fixes WHERE; a surface object alongside it still says
            # HOW BIG, so keep whatever extents it contributed.
            return {"center_mm": [float(origin[0]), float(origin[1])],
                    "center_z_mm": float(origin[2]), "source": "center_frame",
                    "frame": work_frame, "run_id": None, **extents}
    if surface is not None:
        return {"center_mm": surface["center_mm"], "center_z_mm": surface["center_z_mm"],
                "source": "surface_object", "frame": work_frame,
                "run_id": None, **extents}
    if active and active.get("frame") == work_frame:
        corners = _corners_from_payload(active)
        if corners is not None:
            box = platform_from_corners(corners)
            return {"center_mm": box["center_mm"], "center_z_mm": box["center_z_mm"],
                    "source": "scan_run", "frame": work_frame,
                    "run_id": active.get("run_id"),
                    **{key: box[key] for key in extents}}
    return None


def active_scan_payload(root=None) -> dict | None:
    """``runs/scan/active.json`` exactly as written, or ``None`` — the last-resort source."""
    try:
        return runs.read_active(SCAN_MODULE, root) or None
    except (OSError, ValueError):
        return None


def _corners_from_payload(payload: dict, root=None) -> np.ndarray | None:
    """Frame-local rectangle corners (4x3) for an ``active.json`` payload.

    Prefers the corners the insert recorded. Falls back to recomputing them from
    the run's ``report.json`` so a surface inserted before that field existed still
    works; returns ``None`` when neither source is available, which callers treat as
    "cannot centre — re-insert the scan" rather than guessing a centre.
    """
    recorded = payload.get("rectangle_corners_frame_mm")
    if recorded:
        corners = np.asarray(recorded, dtype=float)
        if corners.shape == (4, 3) and np.isfinite(corners).all():
            return corners
    run_id = payload.get("run_id")
    if not run_id:
        return None
    try:
        plane = runs.load_report(SCAN_MODULE, str(run_id), root).get("plane") or {}
        corners = rectangle_in_frame(np.asarray(plane["frame_T_mm"], dtype=float),
                                     np.asarray(plane["corners_mm"], dtype=float))
    except (runs.RunNotFound, OSError, ValueError, KeyError, TypeError):
        return None
    return corners if np.isfinite(corners).all() else None


def active_scan_surface(root=None) -> dict | None:
    """The scan-derived work surface recorded on disk, or ``None``.

    Disk only. :func:`resolve_platform_center` is what centring actually uses; this
    stays the plain reader of ``runs/scan/active.json`` and of the geometry recovery
    that resolver shares through :func:`_corners_from_payload`.

    ``available`` is the only field callers should gate centring on: a surface can
    be applied but un-centrable if its geometry is no longer recoverable.
    """
    try:
        payload = runs.read_active(SCAN_MODULE, root)
    except (OSError, ValueError):
        return None
    if not payload or not payload.get("frame"):
        return None
    corners = _corners_from_payload(payload, root)
    surface = {
        "frame": str(payload["frame"]),
        "rectangle": payload.get("rectangle"),
        "run_id": payload.get("run_id"),
        "applied_at": payload.get("applied_at"),
        "size_mm": payload.get("size_mm"),
        "available": corners is not None,
        "center_mm": None, "bounds_mm": None, "corners_frame_mm": None,
        "note": ("Re-insert the scan to record the rectangle geometry this build "
                 "needs (the applied run predates it or its report is missing)."),
    }
    if corners is None:
        return surface
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    center = corners.mean(axis=0)
    surface.update({
        "corners_frame_mm": corners.tolist(),
        "center_mm": [float(center[0]), float(center[1])],
        "bounds_mm": {"x_min": float(lo[0]), "x_max": float(hi[0]),
                      "y_min": float(lo[1]), "y_max": float(hi[1])},
        "note": "",
    })
    return surface


def surface_fit(surface: dict, *, center_x_mm: float, center_y_mm: float,
                outer_radius_mm: float) -> dict:
    """Does a circle of ``outer_radius_mm`` sit inside the measured rectangle?

    ``outer_radius_mm`` is the path radius plus half a bead, i.e. the outside of the
    deposited wall rather than the tool centre line. Margins are per edge and signed,
    so a negative one names exactly which side overhangs and by how much.
    """
    bounds = surface.get("bounds_mm")
    if not bounds:
        return {"checked": False, "inside": False,
                "reason": surface.get("note") or "surface geometry unavailable"}
    margins = {
        "x_min": (center_x_mm - outer_radius_mm) - bounds["x_min"],
        "x_max": bounds["x_max"] - (center_x_mm + outer_radius_mm),
        "y_min": (center_y_mm - outer_radius_mm) - bounds["y_min"],
        "y_max": bounds["y_max"] - (center_y_mm + outer_radius_mm),
    }
    minimum = min(margins.values())
    return {"checked": True, "inside": bool(minimum >= 0.0),
            "margins_mm": {k: float(v) for k, v in margins.items()},
            "minimum_margin_mm": float(minimum),
            "outer_radius_mm": float(outer_radius_mm)}


def surface_check(setup, recipe, surface: dict | None) -> dict:
    """Preflight section: is this plan's placement still true to the active scan?

    Manual placement passes with an advisory. Surface placement must still match the
    applied scan — same run, same frame — and the wall must fit inside the measured
    rectangle.
    """
    claimed = getattr(setup, "scan_run_id", None)
    outer_radius = recipe.radius_mm + recipe.bead_diameter_mm / 2.0
    if not claimed:
        # A centre read out of the STATION carries no run id, so it arrives here as
        # "manual". The staleness check is genuinely inapplicable then — but the fit
        # is not: the platform is the table, and a wall hanging off it is wrong
        # however the centre was obtained. So whenever this plan is placed in the
        # platform's own frame, still measure it against the platform.
        if surface is None or surface["frame"] != setup.work_frame:
            advisory = ""
            if surface:
                advisory = (f"A scanned surface is applied on {surface['frame']!r}, but "
                            f"this plan is placed manually in {setup.work_frame!r}.")
            return {"ok": True, "placement": "manual", "advisory": advisory,
                    "active_run_id": (surface or {}).get("run_id")}
        fit = surface_fit(surface, center_x_mm=setup.center_x_mm,
                          center_y_mm=setup.center_y_mm, outer_radius_mm=outer_radius)
        if not fit["checked"]:
            return {"ok": True, "placement": "manual", "fit": fit,
                    "active_run_id": surface.get("run_id"),
                    "advisory": ("The platform's extents are not known here, so the wall "
                                 f"could not be checked against them: {fit['reason']}")}
        result = {"ok": bool(fit["inside"]), "placement": "manual", "advisory": "",
                  "active_run_id": surface.get("run_id"), "frame": surface["frame"],
                  "size_mm": surface.get("size_mm"), "fit": fit}
        if not result["ok"]:
            result["problem"] = _overhang_problem(fit)
        return result
    if surface is None:
        return {"ok": False, "placement": "scan_surface", "claimed_run_id": claimed,
                "problem": "This cylinder was centred on a scanned surface, but no scan "
                           "is applied to the station now. Re-run/insert the scan, or "
                           "re-place the path manually."}
    if surface.get("run_id") != claimed:
        return {"ok": False, "placement": "scan_surface", "claimed_run_id": claimed,
                "active_run_id": surface.get("run_id"),
                "problem": f"The surface was re-scanned since this path was placed "
                           f"(applied run {surface.get('run_id')!r}, path built on "
                           f"{claimed!r}). Re-centre on the current surface."}
    if surface["frame"] != setup.work_frame:
        return {"ok": False, "placement": "scan_surface", "claimed_run_id": claimed,
                "problem": f"Work frame {setup.work_frame!r} is not the scanned surface "
                           f"frame {surface['frame']!r}. Re-centre on the surface."}
    fit = surface_fit(surface, center_x_mm=setup.center_x_mm,
                      center_y_mm=setup.center_y_mm, outer_radius_mm=outer_radius)
    result = {"ok": bool(fit.get("inside")), "placement": "scan_surface",
              "claimed_run_id": claimed, "active_run_id": surface.get("run_id"),
              "frame": surface["frame"], "size_mm": surface.get("size_mm"), "fit": fit}
    if not result["ok"]:
        result["problem"] = _overhang_problem(fit)
    return result


def _overhang_problem(fit: dict) -> str:
    """Name every edge the wall hangs over, and by how much."""
    overhang = ", ".join(f"{edge} by {-value:.1f} mm"
                         for edge, value in sorted(fit.get("margins_mm", {}).items())
                         if value < 0) or fit.get("reason", "unknown")
    return (f"A {fit.get('outer_radius_mm', 0.0):.1f} mm wall radius overhangs the "
            f"measured surface: {overhang}.")
