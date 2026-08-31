"""Pure centreline-processing helpers (no RoboDK/camera/Open3D device)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry_fixtures as gf  # noqa: E402
from tasni.modules.extrusion.processing import (_graph, _ordered_skeleton,  # noqa: E402
                                                 _prune_short_spurs, _rasterize,
                                                 _thin, depth_to_work_points)


def test_depth_backprojection_uses_explicit_work_transform():
    depth = np.array([[0, 1000], [1000, 0]], dtype=np.uint16)
    K = np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]], dtype=float)
    T = np.eye(4); T[:3, 3] = [10, 20, 30]
    points, raw = depth_to_work_points(depth, gf.aligned(K, (2, 2)), T)
    assert raw == 2
    np.testing.assert_allclose(points, [[11, 20, 1030], [10, 21, 1030]])


def test_depth_backprojection_honours_units_and_the_depth_to_colour_extrinsic():
    K_c = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
    geom = gf.offset(color_K=K_c, color_size=(320, 240))          # 0.1 mm, non-identity extrinsic
    truth_color = np.array([[0.0, 0.0, 400.0], [40.0, -25.0, 405.0]])
    depth = gf.render_depth_in_depth_camera(truth_color, geom)
    T = np.eye(4); T[:3, 3] = [100, 200, 0]
    points, raw = depth_to_work_points(depth, geom, T)
    assert raw == 2
    for t in truth_color + [100, 200, 0]:
        assert np.linalg.norm(points - t, axis=1).min() < 3.5


def test_graph_ordering_is_deterministic_and_rejects_branches():
    line = np.zeros((7, 9), np.uint8); line[3, 1:8] = 1
    first, branches, endpoints = _ordered_skeleton(line)
    second, *_ = _ordered_skeleton(line)
    np.testing.assert_array_equal(first, second)
    assert branches == 0 and endpoints == 2 and len(first) == 7

    tee = line.copy(); tee[1:4, 4] = 1
    with pytest.raises(RuntimeError, match="branch"):
        _ordered_skeleton(tee)


def test_short_raster_spurs_are_pruned_but_real_branch_is_rejected():
    theta = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    ring = np.column_stack((40 * np.cos(theta), 40 * np.sin(theta),
                            np.full_like(theta, 5.0)))
    mask, _, _ = _rasterize(ring, 1.0, 6.0, 0)
    raw = _thin(mask)
    assert any(len(links) > 2 for links in _graph(raw).values())
    cleaned, removed = _prune_short_spurs(raw, 6)
    _, branches, endpoints = _ordered_skeleton(cleaned)
    assert removed > 0 and branches == 0 and endpoints == 0

    tee = np.zeros((20, 20), np.uint8)
    tee[10, 2:18] = 1
    tee[2:11, 10] = 1
    cleaned_tee, _ = _prune_short_spurs(tee, 3)
    with pytest.raises(RuntimeError, match="branch"):
        _ordered_skeleton(cleaned_tee)


# --------------------------------- segmentation reads no colour at all (spec §3.6)

def test_the_seam_takes_no_colour_input_at_all():
    """The colour frame is still captured and archived; it takes no part in any
    decision. A signature that still ACCEPTED colour would let a caller quietly
    reintroduce it, so the absence is pinned here rather than left to review.

    Spec §1: the saturation gate's premise inverted (bead median S 25, printed
    board 28), and colour auto-exposure runs free on the Jetson, so a fixed
    threshold on saturation was never a calibrated quantity.
    """
    import inspect
    from tasni.modules.extrusion import processing
    for name in ("process_observation", "measure_take", "characterize_ring"):
        params = set(inspect.signature(getattr(processing, name)).parameters)
        assert not params & {"color", "K", "dist"}, (name, sorted(params))
    assert not hasattr(processing, "chroma_gate_mask")
    assert not hasattr(processing, "deposit_floor_mm")


# ------------------------------------------------------- 1 mm voxel default (Task 9)

def test_the_voxel_stays_fine_enough_for_0_1_mm_words_and_for_thin_deposits():
    """Pinned at 0.5 mm, and the direction of travel is one-way: 2 mm -> 1 mm when
    protocol 2 brought 0.1 mm depth words (f662dae, so the finer words could reach
    the ring numbers), 1 mm -> 0.5 mm on 2026-08-31 because 1 mm was also DELETING
    marginal deposits -- a ring the operator confirmed unbroken came back open at
    completeness 0.875, and closed at 0.9925 with no camera change
    (test_extrusion_golden.test_a_marginal_deposit_survives_the_voxel_downsample
    holds that end).

    Both moves are the same argument, so this asserts the BOUND rather than the
    value: coarsening is what has ever been shown to hurt. If a future change
    wants a coarser grid it has to beat this test deliberately and say why, which
    is the point -- the equality pin this replaces would have blocked the 2026-08-31
    fix without ever stating what it was protecting.
    """
    from tasni.core.config import ExtrusionConfig
    assert ExtrusionConfig().voxel_size_m == 0.0005
    assert ExtrusionConfig().voxel_size_m <= 0.001


# ----------------------------------- the radial trim settles instead of running out

def _ring_with_a_slab(*, radius=60.0, half_bead=4.0):
    """A dense annulus plus a rectangular slab welded to its outer flank.

    The board-bias failure with no camera, no Open3D and no renderer in the way:
    the slab is 27% of the cloud, exactly the regime that drags the circle fit
    off the ring. Everything is coplanar on purpose -- height cannot separate
    these two populations, which is the premise the whole radial trim exists to
    answer (see `_radial_trim`).
    """
    theta = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
    offset = np.arange(-half_bead, half_bead + 0.01, 0.5)
    T, D = np.meshgrid(theta, offset, indexing="ij")
    R = radius + D
    ring = np.column_stack((R.ravel() * np.cos(T.ravel()),
                            R.ravel() * np.sin(T.ravel()), np.zeros(T.size)))
    xs = np.arange(radius + 2.0, radius + 30.01, 0.5)
    ys = np.arange(-14.0, 14.01, 0.5)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    slab = np.column_stack((X.ravel(), Y.ravel(), np.zeros(X.size)))
    return ring, np.vstack((ring, slab))


def test_radial_trim_settles_at_a_fixed_point_instead_of_where_the_schedule_ends():
    """The band schedule's premise is that the fit walks onto the ring as the
    bands tighten. Nothing used to check that it had ARRIVED before the list ran
    out, and on a solid contamination patch it had not -- so the tightest band
    selected against a circle still biased by the very thing it was removing.

    Measured here, with no sensor model in the way (numbers are this fixture's,
    2026-08-31): schedule-only keeps the raster out to r 73.7 mm on a 60 mm
    ring; settling the last band reaches a fixed point after 4 passes at
    r 71.7 mm. It is a genuine fixed point, not the cap -- 12 passes and 60
    give bit-identical output.
    """
    from tasni.modules.extrusion.processing import _radial_trim
    _ring, cloud = _ring_with_a_slab()
    schedule = [15.0, 12.0, 10.0]

    unsettled_counts, settled_counts, generous_counts = {}, {}, {}
    unsettled = _radial_trim(cloud, schedule, unsettled_counts, settle_passes=0)
    settled = _radial_trim(cloud, schedule, settled_counts)
    generous = _radial_trim(cloud, schedule, generous_counts, settle_passes=60)

    def reach(points):
        return float(np.linalg.norm(points[:, :2], axis=1).max())

    assert unsettled_counts["radial_trim_settle_passes"] == 0
    assert settled_counts["radial_trim_settle_passes"] > 0
    assert reach(settled) < reach(unsettled) - 1.0, (reach(settled), reach(unsettled))
    # A FIXED POINT, not an iteration budget: more passes change nothing.
    assert settled_counts["radial_trim_settle_passes"] == \
        generous_counts["radial_trim_settle_passes"]
    np.testing.assert_array_equal(settled, generous)


def test_radial_trim_settling_leaves_an_uncontaminated_ring_completely_alone():
    """The cost side of the same change: on a cloud whose fit had already
    arrived, the extra passes must select exactly what the schedule's last pass
    selected and stop immediately -- zero settle passes, not one point moved.
    That is why this cannot quietly erode a real, thin bead."""
    from tasni.modules.extrusion.processing import _radial_trim
    ring, _cloud = _ring_with_a_slab()
    counts = {}

    kept = _radial_trim(ring, [15.0, 12.0, 10.0], counts)

    assert counts["radial_trim_settle_passes"] == 0
    assert counts["after_radial_trim"] == len(ring)
    np.testing.assert_array_equal(kept, ring)


# --------------------------------------------------- one seam for every caller (§3.7)

def test_measure_take_and_not_its_caller_decides_arc_assembly(monkeypatch):
    """``assemble_arcs`` is decided HERE, never by the caller -- the
    live/reprocess/figure divergence was the defect this pins.

    It used to be derived from the layer (True on layer 1, False above it), on
    the reasoning that a higher layer's ROI spans the ring beneath and assembly
    could fuse the two. The value is now True for every layer and the protection
    moved to where it belongs: ``process_observation`` puts a deposit floor under
    layer N at the top of layer N-1, so the ring beneath is not in the population
    being assembled at all. Keeping assembly off was never a substitute -- it
    left "keep the largest arc and call it the ring" in place, which is how layer
    2 measured 0.294 complete off a 110 deg arc on 2026-08-31.
    """
    from tasni.modules.extrusion import processing
    seen = []
    monkeypatch.setattr(processing, "process_observation",
                        lambda **kw: seen.append(kw) or "sentinel")
    from tasni.core.config import ExtrusionConfig
    from tasni.modules.extrusion.models import CylinderRecipe, CylinderSetup
    from tasni.modules.extrusion.toolpath import generate_cylinder_plan
    plan = generate_cylinder_plan(
        CylinderRecipe(radius_mm=40.0, layer_count=2, layer_height_mm=5.0,
                       bead_diameter_mm=9.0, robot_speed_mm_s=75.0,
                       extrusion_rate_pct=0.0, points_per_circle=180),
        CylinderSetup(print_tool="LongCalibTool", work_frame="Tasni Work Frame",
                      inspection_tool="Realsense", inspection_auto=True,
                      center_x_mm=0.0, center_y_mm=0.0))
    # (construction idiom copied from scene_plan() in tests/test_extrusion_measure.py:73)
    common = dict(depth=None, geometry=None, T_work_camera=None,
                  plan=plan, config=ExtrusionConfig())
    assert processing.measure_take(layer=plan.layers[0], **common) == "sentinel"
    assert seen[-1]["assemble_arcs"] is True
    processing.measure_take(layer=plan.layers[1], **common)
    assert seen[-1]["assemble_arcs"] is True
    # The layer still reaches process_observation -- it is what the deposit floor
    # is derived from, so a take can never be scored without saying which layer
    # it is.
    assert seen[-1]["layer"] is plan.layers[1]


def test_ring_geometry_measures_height_against_the_substrate_it_is_given():
    """Height is measured against the surface FITTED IN THIS FRAME, and the
    substrate is required -- there is no build-plane fallback to slip back to.
    The work frame's Z=0 was measured to be the wrong datum (the board sits
    1.2 mm below it, tilted ~0.5 deg), so a reference that is not the substrate
    is a silent ~2 mm error, not a convenience.
    """
    import inspect
    from tasni.modules.extrusion.processing import ring_geometry
    from tasni.modules.extrusion.substrate import PlaneSubstrate

    signature = inspect.signature(ring_geometry)
    assert "build_plane_z_mm" not in signature.parameters
    assert signature.parameters["substrate"].default is inspect.Parameter.empty

    theta = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    ring = np.column_stack((40.0 * np.cos(theta), 40.0 * np.sin(theta),
                            np.full_like(theta, 8.0)))
    cluster = np.repeat(ring, 4, axis=0)
    cluster[:, :2] *= np.linspace(0.95, 1.05, len(cluster))[:, None]

    # A plane 2 mm above z=0, tilted so the reference is genuinely per-point:
    # a build-plane subtraction could not reproduce these heights.
    plane = PlaneSubstrate(a=0.01, b=0.0, c=2.0, sigma_mm=0.3, inlier_fraction=1.0,
                           clamp_mm=(1.0, 2.0), bias_correction_mm=0.0,
                           bias_correction_sigma=0.0)
    fitted = ring_geometry(ring, cluster, (0.0, 0.0), substrate=plane, bins=36)
    assert fitted.height_reference == "fitted_plane"     # substrate.source, not a literal
    assert fitted.top_z_mean_mm == pytest.approx(8.0)    # top is raw work-frame Z
    assert fitted.height_mean_mm == pytest.approx(6.0)                  # 8 - (0.01x + 2)
    assert fitted.height_max_mm - fitted.height_min_mm == pytest.approx(0.8, abs=1e-6)


def test_no_caller_bypasses_the_seam():
    """Grep guard: outside processing.py, nothing in the extrusion module may call
    process_observation directly (spec §3.7)."""
    from pathlib import Path
    import tasni.modules.extrusion as ext
    root = Path(ext.__file__).parent
    offenders = [p.name for p in root.glob("*.py")
                 if p.name != "processing.py"
                 and "process_observation(" in p.read_text(encoding="utf-8")]
    assert not offenders, f"route these through measure_take: {offenders}"


def test_archived_configs_with_retired_keys_still_validate():
    """extra='forbid' + archived processing_config payloads means a field can
    never be deleted without this shim (spec §3.6): retired keys are DROPPED,
    never reinterpreted, and unknown keys still fail loudly.

    The three assertions below are load-bearing TOGETHER. While
    RETIRED_EXTRUSION_CONFIG_KEYS was empty this test passed without exercising
    the shim at all -- `with_retired` was just `payload`. So it now pins that
    the set is non-empty, that every name in it is genuinely GONE from the
    model (a live field listed as retired would be silently dropped from every
    archive), and that plain validation actually REFUSES what from_archive
    accepts -- otherwise the shim could be a no-op and nothing would say so.
    """
    import pytest
    from tasni.core import config as cfg
    retired = cfg.RETIRED_EXTRUSION_CONFIG_KEYS
    assert retired, "nothing retired yet -- this test would not exercise the shim"
    live = set(cfg.ExtrusionConfig.model_fields)
    assert not (retired & live), f"retired but still a field: {sorted(retired & live)}"

    payload = cfg.ExtrusionConfig().model_dump()
    with_retired = dict(payload, **{k: 1 for k in retired})
    assert cfg.ExtrusionConfig.from_archive(with_retired) is not None
    with pytest.raises(Exception):                  # the shim is doing real work
        cfg.ExtrusionConfig.model_validate(with_retired)
    with pytest.raises(Exception):
        cfg.ExtrusionConfig.from_archive(dict(payload, not_a_field_ever=1))


def test_a_users_config_file_survives_a_retired_extrusion_key(tmp_path, capsys):
    """The ARCHIVE shim is not the whole story: a key an operator set in their
    own tasni.config.json travels load_config -> _merge, which RAISES on an
    unknown key. So retiring a field from RETIRED_EXTRUSION_CONFIG_KEYS alone
    leaves a STARTUP CRASH behind -- `KeyError: Unknown config key` naming a key
    that appears in no document the operator would reach for. (Live risk, not
    hypothetical: docs/pfh-paper-handoff.md still tells the operator to lower
    `extrusion.layer_floor_margin_mm`.) Every retired extrusion field must
    therefore also be in LEGACY_CONFIG_KEYS, and be DROPPED with a note giving
    its own reason -- not one borrowed from an unrelated removal.
    """
    import json
    from tasni.core import config as cfg
    legacy_extrusion = {key for section, key in cfg.LEGACY_CONFIG_KEYS
                        if section == "extrusion"}
    assert cfg.RETIRED_EXTRUSION_CONFIG_KEYS <= legacy_extrusion, (
        "retired from archives but not from user configs -- these still crash "
        f"startup: {sorted(cfg.RETIRED_EXTRUSION_CONFIG_KEYS - legacy_extrusion)}")

    path = tmp_path / "tasni.config.json"
    path.write_text(json.dumps({"extrusion": {
        **{k: 1.0 for k in cfg.RETIRED_EXTRUSION_CONFIG_KEYS},
        "radius_mm": 42.0,                      # a live key, set alongside
    }}), encoding="utf-8")

    loaded = cfg.load_config(path)              # must NOT raise

    assert loaded.extrusion.radius_mm == 42.0   # the live override still applies
    out = capsys.readouterr().out
    for key in cfg.RETIRED_EXTRUSION_CONFIG_KEYS:
        assert f"extrusion.{key}" in out        # the operator is told, not ignored
    # ...and told the RIGHT reason: the message used to be hardcoded to the one
    # removal it was written for, which would have blamed "camera protocol 2"
    # for a segmentation change.
    assert "camera protocol 2" not in out
    assert "unknown config key" not in out.lower()
