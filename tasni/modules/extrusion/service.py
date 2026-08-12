"""Pure planning/preflight services for the cylinder workflow."""
from __future__ import annotations

import math

import numpy as np

from .models import CylinderPlan
from .toolpath import points_array


def geometry_preflight(plan: CylinderPlan) -> dict:
    """Validate the generated geometry without claiming a RoboDK dry-run pass."""
    layers: list[dict] = []
    all_ok = True
    for layer in plan.layers:
        pts = points_array(layer)
        gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        expected = 2.0 * math.pi * plan.recipe.radius_mm
        actual = float(gaps.sum())
        closed = bool(np.allclose(pts[0], pts[-1], atol=1e-9))
        finite = bool(np.isfinite(pts).all())
        length_ok = abs(actual - expected) <= max(0.1, expected * 0.001)
        ok = closed and finite and length_ok and len(pts) == plan.recipe.points_per_circle + 1
        all_ok &= ok
        layers.append({
            "layer_index": layer.layer_index, "point_count": len(pts),
            "closed": closed, "finite": finite, "length_mm": actual,
            "maximum_segment_mm": float(gaps.max()), "ok": ok,
        })
    return {
        "kind": "geometry_preflight", "fingerprint": plan.fingerprint,
        "all_ok": all_ok, "layers": layers,
        "dry_run_passed": False,
        "note": "Geometry is valid; RoboDK reachability/collision/program execution is still required.",
        "simulated_valve_events": [
            {"event": "AirOn", "physical_output_blocked": True},
            {"event": "AirOff", "physical_output_blocked": True},
        ],
    }
