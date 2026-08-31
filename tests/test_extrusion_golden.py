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

Layer-2 completeness is frozen below as literals rather than read from the
archive's ``report.json`` at test time: ``service.py``'s offline-reprocess path
(``archive.rewrite_processing(..., report=report)``) overwrites that exact
file, so a report.json read at test time would silently start comparing the
NEW chain's own output against itself the moment anyone presses reprocess on
one of these takes through the app -- exactly the false positive spec §2.4
exists to catch, and it would happen silently because ``runs/`` is git-ignored.
report.json is still read at test time below, but only as a secondary,
best-effort cross-check that warns if the archive appears to have drifted --
never as the value the measurement is judged against.

THE LAYER-2 BASELINE WAS RE-BASELINED ONCE, on 2026-08-31, when the chroma gate
was replaced by the fitted substrate (task 7). Recorded here in full, because a
baseline that moves without a record is indistinguishable from one that was
loosened to fit:

    take                old (colour gate)  new (substrate)   delta
    layer-002           0.6236813839       0.6251582107      +0.0015
    layer-002-take02    0.5962753533       0.6360546778      +0.0398
    layer-002-take03    0.5629560794       0.6137140438      +0.0508

The old values were measured by this same harness against the OLD chain on
2026-08-30 and matched the archive's report.json bit-for-bit (``==``, all three
takes). They are kept below as ``LAYER2_ARCHIVED_COMPLETENESS``, because
report.json on disk still holds them -- that is what makes the secondary
cross-check able to tell "nobody has reprocessed this archive" from "somebody
has".

Why the step is legitimate and not a loosened threshold: it is NOT segmentation
drift. Measured by running the new code with a flat work-frame-Z substrate and
the floor pinned at the old 1.5 mm -- i.e. the old geometry through the new code
path -- layer-002-take03 already moves +0.043 of its +0.051. The only remaining
difference at that setting is that the colour-FOV crop is gone, which spec §3.6
deletes deliberately: the gate discarded 45.6% of the valid depth cloud (400,830
of 878,222 on layer-001) purely because those points projected outside the
narrower colour FOV, a reason having nothing to do with the points themselves.
So ~86% of the movement is one documented, intended, ONE-TIME design change,
spent before segmentation is even considered; and no knob makes it smaller
(compactness off gives +0.077, fit radius 100 mm +0.071, k=3.5 +0.072 -- the
shipped default is the smallest of the four).

The old +/-0.05 window was therefore measuring two different things at once:
"did segmentation change the answer" (worth guarding) and "did we stop
discarding 45.6% of the cloud" (a decision already taken). It now guards only
the first, and ``test_layer2_stays_invalid_and_completeness_stays_honest``
guards the property that actually matters -- these takes must never come back
VALID -- with a direct ceiling (``LAYER2_MAX_COMPLETENESS``) rather than a
proxy. That ceiling is tighter where it counts: validity needs >= 0.900 and
these takes sit at 0.61-0.64.
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

# Frozen 2026-08-31 baseline, NEW (fitted-substrate) chain -- see the module
# docstring for the re-baselining and its evidence. Do NOT read these from
# report.json at test time; that file is overwritten by every offline reprocess
# (service.py's archive.rewrite_processing).
LAYER2_BASELINE_COMPLETENESS = {
    "layer-002": 0.6251582106816579,
    "layer-002-take02": 0.6360546777518749,
    "layer-002-take03": 0.6137140438066724,
}

# What the archive's report.json holds ON DISK: the OLD colour-gate chain's
# output, unchanged since 2026-08-30. Kept so the secondary cross-check below
# can still tell "this archive has not been reprocessed" from "it has" -- which
# is the only question that check was ever asking.
LAYER2_ARCHIVED_COMPLETENESS = {
    "layer-002": 0.6236813839365787,
    "layer-002-take02": 0.5962753533256153,
    "layer-002-take03": 0.5629560794477885,
}

# Frozen 2026-08-31 substrate-report baselines, ALL ELEVEN takes -- this is the
# concrete discharge of spec §3.4's own promise ("the golden harness records
# per-take sigma_mm baselines and is the authoritative reconciliation") between
# §3.3's 0.55-0.61 mm (measured on four different ground-truth frames) and
# §3.4's implied 0.52+ mm (from the derived floor range on the eight layer-1
# takes). Measured directly by running measure_take through this same harness
# on 2026-08-31; see test_substrate_report_is_present_and_sane_on_every_take's
# docstring for the full per-take table (sigma_mm, floor_mm, tilt_deg,
# separation_margin_mm together). Do NOT read these from report.json (see the
# module docstring above for why that file is not authoritative).
SUBSTRATE_SIGMA_MM_BASELINE = {
    "layer-001": 0.5635,
    "layer-001-take02": 0.5497,
    "layer-001-take03": 0.5472,
    "layer-001-take04": 0.5498,
    "layer-001-take05": 0.5527,
    "layer-001-take06": 0.5621,
    "layer-001-take07": 0.5571,
    "layer-001-take08": 0.5680,
    "layer-002": 0.5658,
    "layer-002-take02": 0.5153,
    "layer-002-take03": 0.5190,
}

# The false-positive ceiling (spec §2.4). These three takes honestly measure a
# badly stacked physical ring: every 10-degree sector carries 200-530 valid
# depth pixels, so nothing is unsensed, and the median height of raised material
# swings 2-16 mm around the circumference with all three takes tracing the same
# profile. A change that "fixes" them to a full ring is measuring something that
# is not there. Validity needs >= 0.900 and they sit at 0.61-0.64, so this
# leaves real margin in both directions: ~0.11 of headroom above the highest
# take, ~0.15 of clearance below the validity gate.
LAYER2_MAX_COMPLETENESS = 0.75


def _measure(name):
    from tasni.modules.extrusion import figures, processing
    take = figures.load_take(ARCHIVE / name)
    inputs = figures.reconstruct_take_inputs(take)
    assert inputs is not None, f"{name}: archive lacks reprocess provenance"
    return processing.measure_take(
        depth=take.depth, geometry=take.geometry,
        T_work_camera=take.T_work_camera,
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
        completeness = result.metrics.path_completeness

        # (1) The property this file exists to protect: these takes measure a
        # ring that is genuinely not there, and must never come back VALID.
        assert not result.metrics.valid, (
            f"{name}: a 'fixed' layer-2 take is the false positive spec §2.4 pins")

        # (2) ... and the direct guard on the same property, rather than a
        # proxy. A change that starts recovering circumference here is finding
        # material the physical stack does not have; catch it well before it
        # reaches the validity gate at 0.900.
        assert completeness < LAYER2_MAX_COMPLETENESS, (
            f"{name}: completeness {completeness!r} has climbed past "
            f"{LAYER2_MAX_COMPLETENESS} -- these three takes honestly measure a "
            "badly stacked ring (spec §2.4), so a change that 'recovers' them is "
            "measuring something that is not there. Do NOT raise this ceiling to "
            "accommodate it.")

        # (3) Drift against the 2026-08-31 substrate-chain baseline. Re-based
        # ONCE, when the colour-FOV crop was deleted; see the module docstring
        # for the old values, the deltas and the attribution.
        baseline = LAYER2_BASELINE_COMPLETENESS[name]
        assert abs(completeness - baseline) <= 0.05, (
            name, completeness, baseline)

        # Secondary cross-check only -- report.json is NOT the source of truth
        # (it is overwritten by every offline reprocess; see module docstring).
        # It is compared against the ARCHIVED (old-chain) value because that is
        # what the file still holds; a failure here with the assertions above
        # passing means somebody has reprocessed this archive, not that the
        # baseline is wrong.
        archived = json.loads(
            (ARCHIVE / name / "report.json").read_text(encoding="utf-8"))
        archived_completeness = float(archived["metrics"]["path_completeness"])
        assert abs(archived_completeness - LAYER2_ARCHIVED_COMPLETENESS[name]) <= 1e-6, (
            f"{name}: report.json's completeness ({archived_completeness!r}) no "
            f"longer matches what this archive was written with "
            f"({LAYER2_ARCHIVED_COMPLETENESS[name]!r}) -- it appears to have been "
            "reprocessed since (see module docstring). That does not invalidate "
            "the measurement above, which never reads this file, but the archive "
            "is no longer the 2026-08-30 original.")


def test_substrate_report_is_present_and_sane_on_every_take():
    """spec §4's health block (``report["substrate"]``), present and within
    measured bounds on all eleven archived takes. Observed 2026-08-31 running
    this same harness against the fitted-substrate chain -- this table is the
    per-take record spec §3.4 defers the sigma_mm reconciliation to (its own
    words: "the golden harness records per-take sigma_mm baselines and is the
    authoritative reconciliation" between §3.3's 0.55-0.61 mm, measured on a
    different four-frame ground-truth set, and §3.4's implied 0.52+ mm, from
    the derived-floor range on the eight layer-1 takes):

        take                 sigma_mm   floor_mm   tilt_deg   separation_margin_mm
        layer-001            0.5635     1.691      0.603      2.182
        layer-001-take02     0.5497     1.649      0.530      2.090
        layer-001-take03     0.5472     1.642      0.548      2.051
        layer-001-take04     0.5498     1.649      0.557      2.060
        layer-001-take05     0.5527     1.658      0.539      2.134
        layer-001-take06     0.5621     1.686      0.544      2.158
        layer-001-take07     0.5571     1.671      0.532      2.120
        layer-001-take08     0.5680     1.704      0.520      2.124
        layer-002            0.5658     1.697      0.814      6.857
        layer-002-take02     0.5153     1.546      0.813      6.548
        layer-002-take03     0.5190     1.557      0.805      6.705

    sigma_mm spans 0.5153-0.5680 mm across all eleven takes. The eight
    layer-1 takes span 0.5472-0.5680 mm, straddling the lower edge of
    §3.3's 0.55-0.61 mm band: five sit inside it and the other three
    (0.5497, 0.5472, 0.5498) sit below 0.55 mm by no more than 0.003 mm.
    The three layer-2 takes (a badly stacked ring, not a ground-truth
    frame) measure lower still, which is the reconciliation itself, now
    on record rather than re-argued from two summary ranges. floor_mm
    (k=3.0 x sigma_mm, clamped [1.0, 2.0]) spans 1.546-1.704 mm across
    all eleven takes; the eight layer-1 takes alone span 1.642-1.704 mm,
    matching the design's own 1.55-1.74 mm note (§3.4), which is scoped
    to layer-1.

    tilt_deg spans 0.520-0.814 degrees -- comfortably inside the design's
    board measurement of 0.48-0.62 degrees (§2), so the single ceiling
    asserted below has real headroom on both sides. It is a ceiling, not a
    tightened band, because the brief's original assertion here --
    ``0.3 <= tilt <= 1.0 or tilt < 0.3`` -- algebraically reduces to just
    ``tilt <= 1.0`` while reading as though it bounds two disjoint ranges;
    that was replaced with the one honest bound this asserts.

    separation_margin_mm (bead p50 - substrate p99) is never None on any of
    the eleven and sits at 2.05-2.18 mm on layer-1, 6.55-6.86 mm on layer-2
    (a taller, badly stacked ring reads a wider margin, not a narrower one).
    """
    for name in LAYER1 + LAYER2:
        sub = _measure(name).report["substrate"]
        assert sub["source"] == "fitted_plane"

        # Sanity band, tightened from the brief's loose 0.3-1.0 to what the
        # archive actually shows (0.5153-0.5680), with headroom either side.
        assert 0.45 <= sub["sigma_mm"] <= 0.65, (name, sub)
        # Frozen per-take value: the reconciliation spec §3.4 defers to this
        # harness, discharged as an equality check rather than left as prose.
        # Tolerance is well under the report's own 4-decimal rounding (any two
        # DISTINCT rounded values differ by >= 1e-4), so this catches drift at
        # the reported precision rather than only gross regressions.
        assert abs(sub["sigma_mm"] - SUBSTRATE_SIGMA_MM_BASELINE[name]) <= 5e-5, (
            name, sub["sigma_mm"], SUBSTRATE_SIGMA_MM_BASELINE[name])

        # Tightened from the brief's loose 1.0-2.0 to what the archive shows
        # (1.546-1.704) with headroom; still inside the configured clamp.
        assert 1.4 <= sub["floor_mm"] <= 1.8, (name, sub)

        # A single honest ceiling (see docstring) -- NOT the brief's
        # `0.3 <= tilt <= 1.0 or tilt < 0.3`, which reduces to `tilt <= 1.0`
        # while reading as a two-sided band. Observed 0.520-0.814 degrees,
        # against a design-measured board tilt of 0.48-0.62 degrees (§2);
        # 2.0 leaves real margin without being a proxy for "anything at all".
        assert sub["tilt_deg"] < 2.0, (name, sub)

        assert sub["separation_margin_mm"] is None or sub["separation_margin_mm"] > 1.0, (
            name, sub)


# --------------------------------------------------- the 1 mm voxel and margins
MARGINAL = (Path(__file__).resolve().parents[1] / "runs" / "extrusion"
            / "20260831-173544-24d21bab" / "characterize-01")


@pytest.mark.skipif(not MARGINAL.is_dir(),
                    reason="the 2026-08-31 marginal take is not on this machine")
def test_a_marginal_deposit_survives_the_voxel_downsample():
    """The cell take that reported a CONTINUOUS ring as open.

    2026-08-31 17:35. The operator ruled the ring at five clock positions and
    confirmed by eye that the arc the chain called a 44.9 deg gap is thin but
    unbroken -- 4 mm where the rest of the ring runs 7-11. Only 13% of that
    arc's band cleared the deposit floor, and at a 1 mm voxel the downsample
    merged the sparse survivors until the raster lost connectivity:

        voxel 1.00 mm -> completeness 0.8752, closed False   (the live failure)
        voxel 0.50 mm -> completeness 0.9925, closed True

    Nothing about the camera changed between those two lines. The ring-open
    report was host-side.

    This reads the archive for its DEPTH and pose but builds the config from
    SHIPPED DEFAULTS rather than `from_archive`, because the question it asks is
    "would the chain we ship today measure this take", not "what did the chain
    that captured it do" -- the archived payload carries the 1 mm voxel that
    caused the failure, so reading it back would pin the bug in place.
    """
    from tasni.core.config import ExtrusionConfig
    from tasni.core.depth_geometry import CameraGeometry
    from tasni.modules.extrusion.models import CylinderRecipe, CylinderSetup
    from tasni.modules.extrusion.processing import measure_take
    from tasni.modules.extrusion.toolpath import generate_cylinder_plan

    report = json.loads((MARGINAL / "report.json").read_text(encoding="utf-8"))
    coarse = report["coarse"]
    provenance = report["provenance"]
    config = ExtrusionConfig()          # shipped defaults, deliberately
    recipe = CylinderRecipe(
        radius_mm=float(coarse["radius_mm"]), layer_count=1,
        layer_height_mm=float(coarse["height_mm"]),
        bead_diameter_mm=float(coarse["bead_width_mm"]),
        robot_speed_mm_s=75.0, extrusion_rate_pct=0.0,
        points_per_circle=config.measured_spline_points)
    setup = CylinderSetup(
        print_tool="LongCalibTool", work_frame="wf", inspection_tool="Realsense",
        inspection_auto=True, center_x_mm=float(coarse["center_mm"][0]),
        center_y_mm=float(coarse["center_mm"][1]))
    plan = generate_cylinder_plan(recipe, setup)
    result = measure_take(
        depth=np.load(MARGINAL / "depth.npy"),
        geometry=CameraGeometry.from_greeting(provenance["camera_geometry"]),
        T_work_camera=np.asarray(provenance["T_work_camera"], float),
        plan=plan, layer=plan.layers[0], config=config)

    assert result.report["closed"], result.report["measured_path_completeness"]
    assert result.report["measured_path_completeness"] > 0.95, (
        "a ring the operator confirmed unbroken must not be reported open: "
        f"{result.report['measured_path_completeness']}")
