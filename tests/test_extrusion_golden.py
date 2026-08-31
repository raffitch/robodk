"""Golden reprocess of the 2026-08-30 cell archive (spec §5).

Read-only on runs/. Skips on machines without the archive (runs/ is
git-ignored). Layer-2 takes are EXPECTED INVALID -- a change that makes them
valid is the false positive this file exists to catch (spec §2.4).
"""
import json
from pathlib import Path

import numpy as np
import pytest

ARCHIVE = (Path(__file__).resolve().parents[1]
           / "runs" / "extrusion" / "20260830-202416-293b208d")
pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_dir(),
    reason="golden archive not on this machine (runs/ is git-ignored)")

LAYER1 = ["layer-001"] + [f"layer-001-take{i:02d}" for i in range(2, 9)]
LAYER2 = ["layer-002", "layer-002-take02", "layer-002-take03"]


def _measure(name):
    import cv2
    from tasni.modules.extrusion import figures, processing
    take = figures.load_take(ARCHIVE / name)
    inputs = figures.reconstruct_take_inputs(take)
    assert inputs is not None, f"{name}: archive lacks reprocess provenance"
    color = cv2.imread(str(ARCHIVE / name / "color.png"), cv2.IMREAD_COLOR)
    return processing.measure_take(
        color=color, depth=take.depth, geometry=take.geometry,
        T_work_camera=take.T_work_camera, K=take.K, dist=inputs["dist"],
        plan=inputs["plan"], layer=inputs["layer"], config=inputs["config"])


def test_layer1_acceptance_holds():
    radii = []
    for name in LAYER1:
        result = _measure(name)
        assert result.metrics.valid, name
        assert result.metrics.path_completeness >= 0.990, (
            name, result.metrics.path_completeness)
        radii.append(result.metrics.measured_radius_mm)
    assert abs(float(np.mean(radii)) - 41.0) <= 0.10, radii
    assert float(np.std(radii, ddof=1)) <= 0.15, radii   # spec §2.1: σ stays measured


def test_layer2_stays_invalid_and_completeness_stays_honest():
    for name in LAYER2:
        archived = json.loads(
            (ARCHIVE / name / "report.json").read_text(encoding="utf-8"))
        result = _measure(name)
        assert not result.metrics.valid, (
            f"{name}: a 'fixed' layer-2 take is the false positive spec §2.4 pins")
        assert abs(result.metrics.path_completeness
                   - float(archived["metrics"]["path_completeness"])) <= 0.05, name
