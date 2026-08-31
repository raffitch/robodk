"""Golden reprocess of the 2026-08-30 cell archive (spec §5).

Read-only on runs/. Skips on machines without the archive (runs/ is
git-ignored). Layer-2 takes are EXPECTED INVALID -- a change that makes them
valid is the false positive this file exists to catch (spec §2.4).

Layer-1 baseline (old colour-gate chain, measured 2026-08-30, task 2 of
docs/superpowers/plans/2026-08-30-deposit-segmentation.md) -- the reference
every later front-end swap is judged against. Recorded here, in git, rather
than only in a workspace report, since ``runs/`` itself is git-ignored:

    take                 completeness    radius_mm
    layer-001            0.992708883967  41.039018767060
    layer-001-take02     0.992419975006  41.020663136472
    layer-001-take03     0.992766907434  41.094806889604
    layer-001-take04     0.992808549589  40.960166800983
    layer-001-take05     0.992804187879  41.053317029914
    layer-001-take06     0.992688831845  40.988700894443
    layer-001-take07     0.992836701368  40.948850735418
    layer-001-take08     0.992743916555  40.935166700058

    mean radius = 41.00508636924414 mm, std(ddof=1) = 0.056197953380802246 mm

Layer-2 completeness is frozen below as ``LAYER2_BASELINE_COMPLETENESS``
literals rather than read from the archive's ``report.json`` at test time:
``service.py``'s offline-reprocess path
(``archive.rewrite_processing(..., report=report)``) overwrites that exact
file, so a report.json read at test time would silently start comparing the
NEW chain's own output against itself the moment anyone presses reprocess on
one of these takes through the app while later tasks are in flight -- exactly
the false positive spec §2.4 exists to catch, and it would happen silently
because ``runs/`` is git-ignored. The frozen values were measured by this same
harness against the OLD (colour-gate) chain, 2026-08-30, and matched the
archive's report.json bit-for-bit at that time (``==``, not just within
tolerance, on all three takes). report.json is still read at test time below,
but only as a secondary, best-effort cross-check that warns if the archive
appears to have drifted since -- never as the value the measurement is judged
against.
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

# Frozen 2026-08-30 old-chain baseline -- see module docstring. Do NOT read
# these from report.json at test time; that file is overwritten by every
# offline reprocess (service.py's archive.rewrite_processing).
LAYER2_BASELINE_COMPLETENESS = {
    "layer-002": 0.6236813839365787,
    "layer-002-take02": 0.5962753533256153,
    "layer-002-take03": 0.5629560794477885,
}


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
        result = _measure(name)
        assert not result.metrics.valid, (
            f"{name}: a 'fixed' layer-2 take is the false positive spec §2.4 pins")
        baseline = LAYER2_BASELINE_COMPLETENESS[name]
        assert abs(result.metrics.path_completeness - baseline) <= 0.05, (
            name, result.metrics.path_completeness, baseline)

        # Secondary cross-check only -- report.json is NOT the source of
        # truth (it is overwritten by every offline reprocess; see module
        # docstring). A failure here with the assertion above still passing
        # means the archive was likely reprocessed since this baseline was
        # frozen, not that the frozen baseline is wrong.
        archived = json.loads(
            (ARCHIVE / name / "report.json").read_text(encoding="utf-8"))
        archived_completeness = float(archived["metrics"]["path_completeness"])
        assert abs(archived_completeness - baseline) <= 1e-6, (
            f"{name}: report.json's completeness ({archived_completeness!r}) no "
            f"longer matches the frozen baseline ({baseline!r}) -- the archive "
            "appears to have been reprocessed since this baseline was frozen "
            "(see module docstring); re-verify the frozen literal against a "
            "known-old-chain measurement rather than trusting what is now on disk")
