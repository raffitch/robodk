"""Pure centreline-processing helpers (no RoboDK/camera/Open3D device)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extrusion_synthetic as syn  # noqa: E402
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


# --------------------------------------- chroma gate on registered points (Task 9)

def test_chroma_gate_masks_points_by_their_colour_projection_not_depth_pixel():
    """Depth is not aligned to colour any more: a bead point must be kept because the
    colour pixel it PROJECTS TO is saturated, even though the depth pixel with the
    same (v, u) index looks at something else."""
    from tasni.core.config import ExtrusionConfig
    from tasni.core.depth_geometry import ColorRegistered
    from tasni.modules.extrusion.processing import chroma_gate_mask
    K_c = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
    geom = gf.offset(color_K=K_c, color_size=(320, 240), depth_size=(160, 120))
    # two colour-frame points at z=400: one lands on the saturated blob, one on grey
    pts = np.array([[0.0, 0.0, 400.0], [-80.0, 0.0, 400.0]])
    depth = gf.render_depth_in_depth_camera(pts, geom)
    color = np.full((240, 320, 3), 128, np.uint8)               # achromatic everywhere ...
    color[100:140, 140:180] = (20, 40, 220)                    # ... except a chromatic blob at the reticle
    reg = ColorRegistered.build(depth, geom, K_c, None)
    cfg = ExtrusionConfig(deposit_min_chroma_fraction=0.001)
    keep, applied = chroma_gate_mask(color, reg, cfg)
    assert applied and keep.shape == (len(reg),)
    on_blob = np.linalg.norm(reg.pts_mm - pts[0], axis=1) < 3.5
    assert keep[on_blob].all() and not keep[~on_blob].any()


def test_chroma_gate_at_production_resolution_drops_outside_points_and_scales_the_close_kernel():
    """1920x1080 colour / 1280x720 depth -- the shapes this port exists for.

    Task 9 review, Important 3: every other chroma-gate test here uses
    identity-aligned same-size frames or gf.offset at 320x240/160x120. Neither
    exercises a colour FOV narrower than depth's (most points project outside
    and are dropped -- the review measured 456,026 of 921,600 real-capture
    points, 49%, doing exactly this) nor the close kernel scaling to width
    (k=8 at 1920 wide, not the 720p-tuned 5).
    """
    from tasni.core.config import ExtrusionConfig
    from tasni.core.depth_geometry import ColorRegistered
    from tasni.modules.extrusion.processing import chroma_gate_mask
    depth_K = syn.K_720P
    # A much narrower FOV than the depth camera's, on purpose: most of a full
    # depth frame reprojects outside a 1920x1080 image at this focal length.
    color_K = np.array([[3000.0, 0, 960.0], [0, 3000.0, 540.0], [0, 0, 1.0]])
    geom = gf.offset(color_K=color_K, color_size=(1920, 1080),
                     depth_K=depth_K, depth_size=(1280, 720),
                     rot_deg=(0.0, 0.0, 0.0), t_mm=(0.0, 0.0, 0.0))
    # A flat plane filling the WHOLE depth frame: 1280 x 720 = 921,600 points,
    # the review's own real-capture point count.
    depth = np.full((720, 1280), 4000, np.uint16)          # 4000 * 0.1 mm = 400 mm
    color = np.full((1080, 1920, 3), 128, np.uint8)        # achromatic background
    # Two chromatic bars either side of the optical axis (colour pixel ~960,540,
    # where the depth camera's own principal point reprojects with this zero
    # rotation/translation), separated by a 6 px achromatic gap: wider than the
    # OLD 720p-tuned 5x5 close can bridge, narrower than the correct k=8 at
    # 1920 px wide -- only the right kernel keeps the axis point.
    color[520:560, 900:957] = (20, 40, 220)
    color[520:560, 963:1020] = (20, 40, 220)

    reg = ColorRegistered.build(depth, geom, color_K, None)
    counts: dict = {}
    keep, applied = chroma_gate_mask(
        color, reg, ExtrusionConfig(deposit_min_chroma_fraction=0.001), counts)

    assert applied and keep.shape == (len(reg),)
    u, v = reg.uv[:, 0], reg.uv[:, 1]
    inside = (u >= 0) & (u < 1920) & (v >= 0) & (v < 1080)
    assert counts["chroma_gate_outside_colour"] == int((~inside).sum())
    # A substantial, not edge-case, fraction drops -- this is the shape the
    # review measured on a real capture, not a corner case.
    assert (~inside).sum() > 0.5 * len(reg)

    def nearest(target_u, target_v, *, only=None):
        d = np.hypot(u - target_u, v - target_v)
        if only is not None:
            d = np.where(only, d, np.inf)
        return int(np.argmin(d))

    axis = nearest(960.0, 540.0)                      # sits in the achromatic gap
    assert keep[axis], "the axis point sits in the achromatic gap; only k=8 bridges it"
    on_bar = nearest(920.0, 540.0)                     # squarely on the left chromatic bar
    assert keep[on_bar]
    background = nearest(960.0, 900.0, only=inside)    # achromatic, inside the frame
    assert not keep[background]


def test_chroma_gate_abstains_on_an_achromatic_frame_or_size_mismatch():
    from tasni.core.config import ExtrusionConfig
    from tasni.core.depth_geometry import ColorRegistered
    from tasni.modules.extrusion.processing import chroma_gate_mask, deposit_floor_mm
    K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
    depth = np.full((240, 320), 4000, np.uint16)
    reg = ColorRegistered.build(depth, gf.aligned(K, (320, 240), depth_unit_mm=0.1), K, None)
    cfg = ExtrusionConfig()
    keep, applied = chroma_gate_mask(np.full((240, 320, 3), 128, np.uint8), reg, cfg)
    assert not applied and keep.all()
    assert deposit_floor_mm(cfg, applied) == 2.5
    keep, applied = chroma_gate_mask(np.zeros((100, 100, 3), np.uint8), reg, cfg)   # wrong size
    assert not applied and keep.all()


# ------------------------------------------------------- 1 mm voxel default (Task 9)

def test_default_voxel_is_1_mm_so_0_1_mm_depth_words_reach_the_ring_numbers():
    from tasni.core.config import ExtrusionConfig
    assert ExtrusionConfig().voxel_size_m == 0.001


# --------------------------------------------------- one seam for every caller (§3.7)

def test_measure_take_derives_arc_assembly_from_the_layer_itself(monkeypatch):
    """assemble_arcs is a property of the take (isolated first layer), not of the
    caller -- the live/reprocess/figure divergence was the defect."""
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
    common = dict(color=None, depth=None, geometry=None, T_work_camera=None,
                  K=None, dist=None, plan=plan, config=ExtrusionConfig())
    assert processing.measure_take(layer=plan.layers[0], **common) == "sentinel"
    assert seen[-1]["assemble_arcs"] is True
    processing.measure_take(layer=plan.layers[1], **common)
    assert seen[-1]["assemble_arcs"] is False


def test_ring_geometry_measures_height_against_the_substrate_when_given_one():
    """The `substrate=` slot that replaced `floor_profile` is DORMANT (every
    caller passes None until Task 7 wires the fit in) -- so pin it here, or it
    ships unexercised. Without a substrate the reference is the build plane;
    with one, height is the substrate's own and the report names its source.
    """
    from tasni.modules.extrusion.processing import ring_geometry
    from tasni.modules.extrusion.substrate import PlaneSubstrate

    theta = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    ring = np.column_stack((40.0 * np.cos(theta), 40.0 * np.sin(theta),
                            np.full_like(theta, 8.0)))
    cluster = np.repeat(ring, 4, axis=0)
    cluster[:, :2] *= np.linspace(0.95, 1.05, len(cluster))[:, None]

    plain = ring_geometry(ring, cluster, (0.0, 0.0), bins=36)
    assert plain.height_reference == "build_plane"
    assert plain.height_mean_mm == pytest.approx(8.0)

    # A plane 2 mm above z=0, tilted so the reference is genuinely per-point:
    # a build-plane subtraction could not reproduce these heights.
    plane = PlaneSubstrate(a=0.01, b=0.0, c=2.0, sigma_mm=0.3, inlier_fraction=1.0,
                           clamp_mm=(1.0, 2.0), bias_correction_mm=0.0,
                           bias_correction_sigma=0.0)
    fitted = ring_geometry(ring, cluster, (0.0, 0.0), substrate=plane, bins=36)
    assert fitted.height_reference == "fitted_plane"     # substrate.source, not a literal
    assert fitted.top_z_mean_mm == pytest.approx(plain.top_z_mean_mm)   # top is unchanged
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
    payload["deposit_min_saturation"] = 60          # will be retired by Task 7
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
