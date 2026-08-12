"""The scan → extrusion handoff: place the cylinder on the scanned work surface.

The Scan module inserts a work frame (``Tasni Work Frame``) plus the oriented
rectangle it measured, and records the applied run in ``runs/scan/active.json``.
This module is the reader of that pointer for extrusion, so a trial can be centred
on the *measured surface* instead of on whatever the selected frame's origin
happens to be. That distinction matters: the scan deliberately puts its frame
origin on the rectangle **corner** nearest the robot base, so the default (0, 0)
placement sits on a corner of the table, not in the middle of it.

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
    """The scan-derived work surface currently inserted in the station, or ``None``.

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
    if not claimed:
        advisory = ""
        if surface and surface["frame"] != setup.work_frame:
            advisory = (f"A scanned surface is applied on {surface['frame']!r}, but this "
                        f"plan is placed manually in {setup.work_frame!r}.")
        return {"ok": True, "placement": "manual", "advisory": advisory,
                "active_run_id": (surface or {}).get("run_id")}
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
                      center_y_mm=setup.center_y_mm,
                      outer_radius_mm=recipe.radius_mm + recipe.bead_diameter_mm / 2.0)
    result = {"ok": bool(fit.get("inside")), "placement": "scan_surface",
              "claimed_run_id": claimed, "active_run_id": surface.get("run_id"),
              "frame": surface["frame"], "size_mm": surface.get("size_mm"), "fit": fit}
    if not result["ok"]:
        overhang = ", ".join(f"{edge} by {-value:.1f} mm"
                             for edge, value in sorted(fit.get("margins_mm", {}).items())
                             if value < 0) or fit.get("reason", "unknown")
        result["problem"] = (f"A {fit.get('outer_radius_mm', 0.0):.1f} mm wall radius "
                             f"overhangs the measured surface: {overhang}.")
    return result
