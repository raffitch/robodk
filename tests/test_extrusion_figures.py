"""Paper/app figures rendered from an archived take.

The renderer reads only what the archive holds, so these tests build a REAL
trial directory with the real writer and then render it -- the file format is
the contract under test, not the processing chain.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extrusion_synthetic as syn  # noqa: E402
import geometry_fixtures as gf  # noqa: E402
from tasni.core.config import ExtrusionConfig  # noqa: E402
from tasni.modules.extrusion.archive import ExtrusionArchive  # noqa: E402
from tasni.modules.extrusion.models import CylinderRecipe, CylinderSetup, LayerManifest  # noqa: E402
from tasni.modules.extrusion.toolpath import generate_cylinder_plan  # noqa: E402

CENTER = (200.0, 150.0)
RADIUS = 60.0


def _plan(*, layers: int = 1, layer_height: float = 6.0):
    recipe = CylinderRecipe(radius_mm=RADIUS, layer_count=layers,
                            layer_height_mm=layer_height, bead_diameter_mm=8.0,
                            robot_speed_mm_s=75, extrusion_rate_pct=0,
                            points_per_circle=180)
    setup = CylinderSetup(print_tool="LongCalibTool", work_frame="Tasni Work Frame",
                          inspection_tool="Realsense", inspection_auto=True,
                          center_x_mm=CENTER[0], center_y_mm=CENTER[1])
    return generate_cylinder_plan(recipe, setup)


def _ring_xyz(*, radius=RADIUS, center=CENTER, z=6.0, wave=0.0, count=181):
    theta = np.linspace(0, 2 * np.pi, count)
    return np.column_stack((center[0] + radius * np.cos(theta),
                            center[1] + radius * np.sin(theta),
                            np.full_like(theta, z) + wave * np.sin(2 * theta)))


def _cloud_xyz(*, radius=RADIUS, center=CENTER, z=6.0, bead=8.0, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, 900)
    r = radius + rng.uniform(-bead / 2, bead / 2, theta.size)
    return np.column_stack((center[0] + r * np.cos(theta),
                            center[1] + r * np.sin(theta),
                            z + rng.normal(0, 0.4, theta.size)))


def write_take(root, *, trial_id="t1", layer_index=1, take=1, measured=None,
               cloud=None, depth=True, mode="MEASURE_ONLY", annotation=None,
               geometry=None, metrics=None, legacy_depth=False):
    """One archived take, written exactly the way the measure job writes it.

    ``legacy_depth`` writes a PRE-protocol-2 take instead: 1 mm depth words and
    no ``camera_geometry`` in provenance, which is what an archive from before
    2026-08 looks like and what ``figures.geometry_for_take``'s legacy fallback
    exists for. Everything else records the 0.1 mm words the renderer (and the
    real cell) actually produce -- without that the archive would be read back
    through the 1 mm fallback and every reconstructed point would be 10x too
    far away.
    """
    plan = _plan(layers=max(layer_index, 1))
    archive = ExtrusionArchive(root)
    if not (root / trial_id / "trial.json").is_file():
        archive.create_trial(trial_id, plan, mode=mode,
                             provenance={"camera_intrinsics": {"K": syn.K_720P.tolist()}})
    layer = plan.layers[layer_index - 1]
    T = syn.inspection_camera_T((CENTER[0], CENTER[1], layer.nominal_z_mm), 300.0)
    nominal = _ring_xyz(z=layer.nominal_z_mm)
    measured = _ring_xyz(center=(CENTER[0] + 10.0, CENTER[1]), wave=2.0) \
        if measured is None else measured
    cloud = _cloud_xyz() if cloud is None else cloud
    frame = None
    if depth:
        frame = syn.render_scene([syn.RingSpec(RADIUS, 8.0, CENTER, height_fn=syn.flat(6.0))],
                                 T, plane_center_xy_mm=CENTER, seed=1,
                                 depth_unit_mm=1.0 if legacy_depth else syn.DEPTH_UNIT_MM)
    manifest = LayerManifest(
        trial_id=trial_id, layer_index=layer_index, take=take, mode=mode,
        recipe=plan.recipe, toolpath_fingerprint=plan.fingerprint,
        annotation=annotation or {}, color_file=None,
        depth_file="depth.npy" if depth else None,
        measured_path_file="measured_path.json",
        pointcloud_file="height-or-pointcloud.npy",
        metrics=metrics or {"mean_absolute_mm": 6.4, "rms_mm": 7.1, "maximum_mm": 10.0,
                            "measured_center_mm": [CENTER[0] + 10.0, CENTER[1]],
                            "measured_radius_mm": RADIUS, "path_completeness": 1.0,
                            "maximum_angular_gap_deg": 2.0, "valid": True,
                            "center_offset_mm": [10.0, 0.0], "center_offset_norm_mm": 10.0,
                            "shape_rms_mm": 0.4, "shape_max_mm": 0.9},
        geometry=geometry or {"top_z_mean_mm": 6.0, "top_z_min_mm": 4.0,
                              "top_z_max_mm": 8.0, "top_z_std_mm": 1.4,
                              "height_mean_mm": 6.0, "height_min_mm": 4.0,
                              "height_max_mm": 8.0, "height_reference": "build_plane",
                              "bead_width_mean_mm": 8.0, "bead_width_min_mm": 6.0,
                              "bead_width_max_mm": 10.0, "bead_width_bins": 36},
        processing={"valid": True, "timings_ms": {"total_ms": 900.0, "capture_ms": 2200.0,
                                                  "acquisition_to_path_ms": 3100.0}},
        provenance={"T_work_camera": np.asarray(T, dtype=float).tolist(),
                    "camera_intrinsics": {"K": syn.K_720P.tolist()},
                    # The method figure re-runs the chain from the frame, and
                    # the chain is reproducible only with the config it used.
                    "processing_config": ExtrusionConfig().model_dump(mode="json"),
                    "work_frame": "Tasni Work Frame",
                    **({} if legacy_depth
                       else {"camera_geometry": syn.geometry_dict()})})
    return archive.write_layer(manifest, nominal_xyz=nominal, commanded_xyz=nominal,
                               measured_xyz=measured, pointcloud_xyz=cloud, depth=frame)


def write_characterization(root, *, trial_id="c1", index=1, radius=RADIUS, center=CENTER,
                           bead=8.0, height=6.0, seed=5):
    """One archived ring characterization, produced by the REAL ``characterize_ring``
    pass (not a hand-built report.json) -- so ``report['coarse']`` is the actual
    throwaway recipe ``_compute_characterization_stages`` has to rebuild."""
    from tasni.modules.extrusion.processing import characterize_ring

    plan = _plan(layers=1)
    archive = ExtrusionArchive(root)
    if not (root / trial_id / "trial.json").is_file():
        archive.create_trial(trial_id, plan, mode="MEASURE_ONLY")
    T = syn.inspection_camera_T((center[0], center[1], height), 300.0)
    geometry = syn.geometry()
    depth = syn.render_scene([syn.RingSpec(radius, bead, center, height_fn=syn.flat(height))],
                             T, plane_center_xy_mm=center, seed=seed)
    color = np.zeros((syn.SIZE_720P[1], syn.SIZE_720P[0], 3), np.uint8)
    config = ExtrusionConfig()
    found = characterize_ring(
        depth=depth, geometry=geometry, T_work_camera=T,
        search_center_mm=center,
        work_frame="Tasni Work Frame", config=config,
        inspection_tool="Realsense", print_tool="LongCalibTool")
    report = {**found.report, "summary": {**found.summary(), "index": index},
             "provenance": {"T_work_camera": np.asarray(T, dtype=float).tolist(),
                            "camera_intrinsics": {"K": syn.K_720P.tolist()},
                            "camera_geometry": syn.geometry_dict(),
                            "processing_config": config.model_dump(mode="json")}}
    return archive.write_characterization(
        trial_id, index, color=color, depth=depth, measured_xyz=found.measured_xyz,
        derived_images={"segmentation.png": found.segmentation,
                        "skeleton.png": found.skeleton, "comparison.png": found.comparison},
        report=report)


def test_rendering_a_take_writes_every_figure_in_both_formats(tmp_path):
    from tasni.modules.extrusion import figures

    layer_dir = write_take(tmp_path)
    written = figures.render_layer_figures(layer_dir)

    names = {path.name for path in written}
    assert names == {f"{stem}.{ext}" for stem in figures.LAYER_FIGURES
                     for ext in ("png", "pdf")}
    for path in written:
        assert path.parent == layer_dir / "figures"
        assert path.stat().st_size > 1000, f"{path.name} is suspiciously small"
    assert (layer_dir / "figures" / "plan.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (layer_dir / "figures" / "plan.pdf").read_bytes()[:5] == b"%PDF-"


def test_profile_curves_are_the_series_behind_the_manifest_metrics(tmp_path):
    """A reader must be able to check the deviation table against the plot."""
    from tasni.modules.extrusion import figures
    from tasni.modules.extrusion.comparison import compare_circle

    measured = _ring_xyz(center=(CENTER[0] + 10.0, CENTER[1]), wave=2.0)
    profile = figures.unrolled_profile(measured, CENTER, RADIUS)
    expected = compare_circle(measured, RADIUS, nominal_center_mm=CENTER)

    assert profile["rms_mm"] == pytest.approx(expected.rms_mm, abs=1e-9)
    assert profile["maximum_mm"] == pytest.approx(expected.maximum_mm, abs=1e-9)
    assert profile["mean_absolute_mm"] == pytest.approx(expected.mean_absolute_mm, abs=1e-9)
    assert np.all(np.diff(profile["angle_deg"]) >= 0), "angle must sweep 0..360 in order"
    assert profile["deviation_mm"].max() == pytest.approx(10.0, abs=.2)


def test_heightmap_falls_back_to_the_archived_cloud_when_the_take_has_no_depth(tmp_path):
    from tasni.modules.extrusion import figures

    layer_dir = write_take(tmp_path, depth=False)
    written = figures.render_layer_figures(layer_dir, formats=("png",))

    # Everything except the method figure, which re-runs the chain from the raw
    # frame: without depth there is no pipeline to show, and an invented one
    # would be worse than none.
    assert {p.name for p in written} == {
        f"{s}.png" for s in figures.LAYER_FIGURES if s != "pipeline"}


def test_a_take_whose_processing_failed_still_renders_what_it_has(tmp_path):
    """The raw frame is all a failed measurement leaves; it must still be drawable."""
    from tasni.modules.extrusion import figures

    layer_dir = write_take(tmp_path)
    (layer_dir / "measured_path.json").unlink()
    (layer_dir / "height-or-pointcloud.npy").unlink()

    written = figures.render_layer_figures(layer_dir, formats=("png",))

    names = {p.name for p in written}
    assert "heightmap.png" in names, "the depth frame is still there"
    assert "profile.png" not in names, "no centreline means no profile, not an empty plot"


def test_ensure_figure_renders_on_demand_then_reuses_the_file(tmp_path):
    from tasni.modules.extrusion import figures

    layer_dir = write_take(tmp_path)
    assert not (layer_dir / "figures").exists()

    path = figures.ensure_figure(layer_dir, "plan.png")
    assert path.is_file()
    stamp = path.stat().st_mtime_ns

    assert figures.ensure_figure(layer_dir, "plan.png").stat().st_mtime_ns == stamp


def test_ensure_figure_refuses_a_name_outside_the_allowlist(tmp_path):
    """The name reaches this from a URL path segment."""
    from tasni.modules.extrusion import figures

    layer_dir = write_take(tmp_path)
    for name in ("../../trial.json", "plan.svg", "manifest.json", r"..\secrets.env"):
        with pytest.raises(ValueError):
            figures.ensure_figure(layer_dir, name)


def test_stack_figure_draws_the_latest_take_of_every_layer(tmp_path):
    from tasni.modules.extrusion import figures

    write_take(tmp_path, layer_index=1, take=1)
    write_take(tmp_path, layer_index=1, take=2,
               measured=_ring_xyz(center=(CENTER[0] + 15.0, CENTER[1])))
    write_take(tmp_path, layer_index=2, take=1, measured=_ring_xyz(z=12.0))

    takes = figures.latest_takes(tmp_path / "t1")
    assert [(t.manifest["layer_index"], t.manifest["take"]) for t in takes] == [(1, 2), (2, 1)]

    written = figures.render_trial_figures(tmp_path / "t1", formats=("png",))
    assert [p.name for p in written] == ["stack.png", "tube.png"]
    assert written[0].parent == tmp_path / "t1" / "figures"


# ------------------------------- the ring the operator MEANT to place (ground truth)

def _plan_line_labels(figures, take):
    plt = figures._pyplot()
    fig = figures._figure_plan(plt, take)
    try:
        return [str(line.get_label()) for line in fig.axes[0].get_lines()]
    finally:
        plt.close(fig)


def test_the_expected_ring_is_the_nominal_circle_moved_by_the_operators_offset(tmp_path):
    """Ground truth for a displaced take is the nominal ring plus the typed offset."""
    from tasni.modules.extrusion import figures

    take = figures.load_take(write_take(
        tmp_path, annotation={"introduced_offset_mm": [10.0, -4.0]}))
    expected = figures.expected_ring(take)

    assert expected is not None
    assert np.allclose(expected[:, 0], take.nominal[:, 0] + 10.0)
    assert np.allclose(expected[:, 1], take.nominal[:, 1] - 4.0)
    assert np.allclose(expected[:, 2], take.nominal[:, 2]), "a shift in XY moves no height"


def test_a_take_with_no_introduced_offset_has_no_expected_ring(tmp_path):
    """Nothing was introduced, so there is no second truth to draw -- not a copy of nominal."""
    from tasni.modules.extrusion import figures

    assert figures.expected_ring(figures.load_take(write_take(tmp_path))) is None
    zero = figures.load_take(write_take(tmp_path, trial_id="t-zero",
                                        annotation={"introduced_offset_mm": [0.0, 0.0]}))
    assert figures.expected_ring(zero) is None


def test_the_plan_view_draws_where_the_ring_was_moved_to(tmp_path):
    """The paper's figure has to show the measured centreline ON the ground truth."""
    from tasni.modules.extrusion import figures

    take = figures.load_take(write_take(
        tmp_path, annotation={"introduced_offset_mm": [10.0, 0.0]}))

    labels = _plan_line_labels(figures, take)

    assert any("ground truth" in label for label in labels), labels


def test_the_plan_view_draws_no_ground_truth_when_nothing_was_introduced(tmp_path):
    from tasni.modules.extrusion import figures

    labels = _plan_line_labels(figures, figures.load_take(write_take(tmp_path)))

    assert not any("ground truth" in label for label in labels), labels


def test_the_stack_figure_draws_the_ground_truth_of_a_displaced_layer(tmp_path):
    from tasni.modules.extrusion import figures

    write_take(tmp_path, layer_index=1, take=1)
    write_take(tmp_path, layer_index=2, take=1,
               annotation={"introduced_offset_mm": [10.0, 0.0]},
               measured=_ring_xyz(center=(CENTER[0] + 10.0, CENTER[1]), z=12.0))

    plt = figures._pyplot()
    fig = figures._figure_stack(plt, figures.latest_takes(tmp_path / "t1"), "t1")
    try:
        labels = [str(line.get_label()) for line in fig.axes[0].get_lines()]
    finally:
        plt.close(fig)

    assert any("ground truth" in label for label in labels), labels


# ------------------------------------------- the figure has to stay READABLE

def _drawn_plan(figures, take):
    plt = figures._pyplot()
    fig = figures._figure_plan(plt, take)
    fig.canvas.draw()
    return plt, fig


def test_the_plan_legend_never_sits_on_top_of_the_measured_ring(tmp_path):
    """A legend over the ring hides the very thing the figure is evidence for.

    The ring fills the axes by construction (the view is framed on it), so an
    in-axes legend has nowhere to go that is not data.
    """
    from tasni.modules.extrusion import figures

    take = figures.load_take(write_take(
        tmp_path, annotation={"introduced_offset_mm": [10.0, 0.0]}))
    plt, fig = _drawn_plan(figures, take)
    try:
        axes = fig.axes[0]
        legend = axes.get_legend().get_window_extent()
        data = axes.get_window_extent()
        assert legend.y1 <= data.y0 + 1.0, (
            f"legend {legend} overlaps the plotted data {data}")
    finally:
        plt.close(fig)


def test_the_plan_legend_clears_the_axis_label_it_sits_under(tmp_path):
    """Moving the legend out of the data must not park it on the x-axis label."""
    from tasni.modules.extrusion import figures

    take = figures.load_take(write_take(
        tmp_path, annotation={"introduced_offset_mm": [10.0, 0.0]}))
    plt, fig = _drawn_plan(figures, take)
    try:
        axes = fig.axes[0]
        legend = axes.get_legend().get_window_extent()
        label = axes.xaxis.label.get_window_extent()
        assert legend.y1 <= label.y0 + 1.0, (
            f"legend {legend} overlaps the axis label {label}")
        caption = fig.texts[-1].get_window_extent()
        assert caption.y1 <= legend.y0 + 1.0, "caption must clear the legend too"
    finally:
        plt.close(fig)


def test_a_long_honesty_caption_is_wrapped_rather_than_clipped(tmp_path):
    """The caption is what stops a figure being read as a printed-cylinder result.

    With an introduced offset recorded it runs past the width of the figure and
    was silently cut off at BOTH ends -- taking the wording constraint with it.
    """
    from tasni.modules.extrusion import figures

    take = figures.load_take(write_take(
        tmp_path, annotation={"introduced_offset_mm": [-15.13, 2.74]}))
    assert len(take.caption) > figures.CAPTION_WRAP, "fixture must produce a long caption"

    plt, fig = _drawn_plan(figures, take)
    try:
        caption = fig.texts[-1]
        drawn = caption.get_text()
        assert "hand-placed bead" in drawn, "the honesty clause must survive"
        assert max(len(line) for line in drawn.split("\n")) <= figures.CAPTION_WRAP
        assert caption.get_window_extent().x0 >= -1.0, "caption runs off the left edge"
        assert caption.get_window_extent().x1 <= fig.bbox.x1 + 1.0, "caption runs off the right"
    finally:
        plt.close(fig)


def test_a_short_caption_is_left_on_one_line(tmp_path):
    from tasni.modules.extrusion import figures

    assert "\n" not in figures.wrap_caption("no introduced offset - RMS 0.50 mm")


def test_height_colour_range_is_set_by_the_deposit_not_by_depth_dropouts(tmp_path):
    """Dropouts far below the plane flatten the ring to one colour if they count."""
    from tasni.modules.extrusion import figures

    ring = _cloud_xyz(z=6.0)
    dropouts = np.column_stack((np.full(40, CENTER[0]), np.full(40, CENTER[1]),
                                np.full(40, -250.0)))
    layer_dir = write_take(tmp_path, depth=False,
                           cloud=np.vstack((ring, dropouts)))

    data = figures.heightmap_data(figures.load_take(layer_dir))

    assert data["vmin"] > -15.0, "a dropout at -250 mm must not set the bottom of the scale"
    assert data["vmax"] - data["vmin"] < 30.0, "the scale must be tight enough to show relief"
    assert data["vmax"] > 4.0, "the ring must still set the top of the scale"


def test_profile_uses_the_true_nominal_centre_of_a_CLOSED_nominal_path(tmp_path):
    """The archive closes the nominal ring, so its first point is repeated.

    Averaging those points biases the centre by radius/N and the plotted RMS
    then disagrees with the manifest a reader is checking it against.
    """
    from tasni.modules.extrusion import figures
    from tasni.modules.extrusion.comparison import compare_circle

    layer_dir = write_take(tmp_path)
    take = figures.load_take(layer_dir)
    assert np.allclose(take.nominal[0], take.nominal[-1]), "fixture must close the ring"

    assert take.center == pytest.approx(CENTER, abs=1e-6)
    profile = figures.unrolled_profile(take.measured, take.center, take.radius)
    expected = compare_circle(take.measured, take.radius, nominal_center_mm=CENTER)
    assert profile["rms_mm"] == pytest.approx(expected.rms_mm, abs=1e-6)


def test_the_method_figure_draws_every_stage_of_the_pipeline(tmp_path):
    """The paper's method figure: one depth frame becoming a centreline.

    It re-runs the archived frame through the real chain with a stage
    collector, so what it draws is what the pipeline held -- not a second
    implementation that could drift from it.
    """
    pytest.importorskip("open3d")
    from tasni.modules.extrusion.figures import render_layer_figures, take_stages, load_take

    root = tmp_path / "runs" / "extrusion"
    layer_dir = write_take(root, layer_index=1)

    stages = take_stages(load_take(layer_dir))
    written = render_layer_figures(layer_dir, only="pipeline")

    assert {p.name for p in written} == {"pipeline.png", "pipeline.pdf"}
    assert (layer_dir / "figures" / "pipeline.png").stat().st_size > 20_000
    assert stages and len(stages["backprojected"]) > len(stages["work_roi"])


def test_the_tube_figure_draws_the_bead_as_a_pipe_at_each_layer_height(tmp_path):
    """A curve hides the bead thickness, which is the quantity being measured."""
    from tasni.modules.extrusion.figures import render_trial_figures, _tube

    root = tmp_path / "runs" / "extrusion"
    write_take(root, layer_index=1)
    write_take(root, layer_index=2, measured=_ring_xyz(z=12.0))

    written = render_trial_figures(root / "t1")

    assert {p.name for p in written} == {"stack.png", "stack.pdf", "tube.png", "tube.pdf"}
    # The pipe is a surface swept at the bead radius about the centreline.
    surface = _tube(_ring_xyz(z=6.0), 8.0)
    assert surface is not None and surface[0].shape[1] == 20
    radii = np.linalg.norm(np.stack(surface, axis=-1)[:, :, :2]
                           - np.array([CENTER[0], CENTER[1]]), axis=-1)
    assert radii.max() - radii.min() == pytest.approx(8.0, abs=0.5)


def test_the_measured_bead_width_is_read_from_the_take_that_measured_it(tmp_path):
    """Intended and outcome are different widths, and only one of them is a guess.

    The commanded bead comes from the recipe; the deposited bead was measured
    (10.77 mm against a commanded 12.8 mm on the first real capture). Drawing
    the measurement at the commanded width would show a comparison that was
    never made.
    """
    from tasni.modules.extrusion.figures import load_take, measured_bead_mm

    layer_dir = write_take(tmp_path, layer_index=1)
    take = load_take(layer_dir)

    assert measured_bead_mm(take) == pytest.approx(8.0)      # the helper's geometry

    bare = json.loads((layer_dir / "manifest.json").read_text(encoding="utf-8"))
    bare.pop("geometry", None)
    (layer_dir / "manifest.json").write_text(json.dumps(bare), encoding="utf-8")
    assert measured_bead_mm(load_take(layer_dir)) is None


# ------------------------------------------------- the surfaced (meshed) view

def _annulus_cloud(*, inner=50.0, outer=70.0, center=CENTER, z=6.0, count=6000, seed=3):
    """A ring-shaped cloud with a genuine hole in the middle."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, count)
    r = np.sqrt(rng.uniform(inner ** 2, outer ** 2, count))
    return np.column_stack((center[0] + r * np.cos(theta),
                            center[1] + r * np.sin(theta),
                            np.full(count, z) + rng.normal(0, .15, count)))


def test_a_surface_mesh_leaves_the_hole_the_camera_actually_saw():
    """Triangulating a ring must not roof over its middle.

    A convex triangulation spans the hole with long thin triangles, drawing a
    solid disc where the measurement found nothing -- the one thing a surface
    figure must never invent.
    """
    from tasni.modules.extrusion.figures import surface_mesh

    mesh = surface_mesh(_annulus_cloud())
    assert mesh is not None and len(mesh.triangles) > 100

    corners = np.stack((mesh.x, mesh.y), axis=1)[mesh.triangles]
    radii = np.linalg.norm(corners.mean(axis=1) - np.array(CENTER), axis=1)
    assert radii.min() > 45.0, "a triangle sits inside the ring's hole"
    assert radii.max() < 75.0, "a triangle reaches past the deposit"
    assert np.isfinite(mesh.z).all(), "every meshed vertex must carry a height"


def test_a_surface_mesh_does_not_bridge_a_dropout_cliff():
    """D435i dropouts sit hundreds of mm below the plane; a triangle joining one
    to the deposit would draw a wall that was never there."""
    from tasni.modules.extrusion.figures import surface_mesh

    cloud = _annulus_cloud()
    dropouts = cloud[:400].copy()
    dropouts[:, 2] = -350.0                          # the holes the D435i leaves
    mesh = surface_mesh(np.vstack((cloud, dropouts)))

    assert mesh is not None
    spans = np.ptp(mesh.z[mesh.triangles], axis=1)
    assert spans.max() < 25.0, "a triangle bridges the cliff to a dropout"


def test_the_mesh_figure_surfaces_both_the_scene_and_the_deposit(tmp_path):
    """The old paper's picture: the frame as a surface, top-down and rotated."""
    from tasni.modules.extrusion.figures import load_take, mesh_panels, render_layer_figures

    layer_dir = write_take(tmp_path, layer_index=1)
    panels = mesh_panels(load_take(layer_dir))

    assert [p.key for p in panels] == ["scene", "deposit"]
    assert all(len(p.points) for p in panels)

    written = render_layer_figures(layer_dir, only="mesh")
    assert {p.name for p in written} == {"mesh.png", "mesh.pdf"}
    assert (layer_dir / "figures" / "mesh.png").stat().st_size > 20_000


def test_the_mesh_figure_drops_the_scene_panel_when_there_is_no_frame(tmp_path):
    """Without depth the only cloud is the deposit; drawing it twice, once
    labelled 'scene', would claim a second view that was never captured."""
    from tasni.modules.extrusion.figures import load_take, mesh_panels, render_layer_figures

    layer_dir = write_take(tmp_path, layer_index=1, depth=False)
    panels = mesh_panels(load_take(layer_dir))

    assert [p.key for p in panels] == ["deposit"]
    written = render_layer_figures(layer_dir, only="mesh")
    assert {p.name for p in written} == {"mesh.png", "mesh.pdf"}


def test_a_frame_with_nothing_near_the_work_plane_gets_no_scene_panel(tmp_path):
    """A frame that never had the work surface in view must not be surfaced.

    The failed cell take 20260828-124136 back-projects 12 x 16 m with ZERO
    points inside the work band -- the pose was wrong. Sizing the panel from
    that frame drew a 32 m window holding 32 triangles at a 222 mm pitch and
    a -800..-200 mm colour scale: a picture of the room, captioned as the work
    surface. The window is anchored on the deposit or the commanded ring
    instead, and a frame with nothing in the band gets no panel.
    """
    from tasni.modules.extrusion.figures import load_take, mesh_panels

    layer_dir = write_take(tmp_path, layer_index=1)
    depth = np.load(layer_dir / "depth.npy")
    np.save(layer_dir / "depth.npy", np.full_like(depth, 20_000))   # all metres away

    assert [p.key for p in mesh_panels(load_take(layer_dir))] == ["deposit"]

    (layer_dir / "height-or-pointcloud.npy").unlink()
    assert mesh_panels(load_take(layer_dir)) == [], "nothing measurable, nothing drawn"


# --------------------------------------------------- camera geometry (Task 9)

def test_a_take_without_camera_geometry_renders_as_legacy_aligned(tmp_path):
    from tasni.modules.extrusion import figures
    layer_dir = write_take(tmp_path, legacy_depth=True)    # a pre-protocol-2 archive
    take = figures.load_take(layer_dir)
    assert take.geometry is not None and take.geometry.legacy
    assert take.geometry.depth_size == (1280, 720)
    assert "legacy aligned" in take.label
    assert figures._scene_points(take) is not None


def test_a_protocol_2_take_uses_its_recorded_geometry(tmp_path):
    import geometry_fixtures as gf
    from tasni.modules.extrusion import figures
    geom = gf.offset(color_K=syn.K_720P, color_size=syn.SIZE_720P,
                     depth_K=syn.K_720P, depth_size=syn.SIZE_720P)
    layer_dir = write_take(tmp_path, trial_id="t-v2")
    manifest_file = layer_dir / "manifest.json"
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    payload["provenance"]["camera_geometry"] = geom.to_dict()
    manifest_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    take = figures.load_take(layer_dir)
    assert take.geometry.legacy is False and take.geometry.depth_unit_mm == 0.1
    assert "legacy" not in take.label


# ------------------------------------------ the controllable 3-D view (iso/birdseye)

def test_render_view_azimuth_and_elevation_are_controllable(tmp_path):
    """The one thing the operator asked to be sure of: not stuck at one angle."""
    from tasni.modules.extrusion import figures

    take = figures.load_take(write_take(tmp_path))
    fig = figures.render_view(take, azim=12.0, elev=48.0)
    try:
        ax = fig.axes[0]
        assert ax.elev == pytest.approx(48.0)
        assert ax.azim == pytest.approx(12.0)
    finally:
        figures._pyplot().close(fig)


def test_birdseye_looks_straight_down_orthographic_with_no_exaggeration(tmp_path):
    from tasni.modules.extrusion import figures

    take = figures.load_take(write_take(tmp_path))
    fig = figures._figure_birdseye(figures._pyplot(), take)
    try:
        ax = fig.axes[0]
        assert ax.elev == pytest.approx(90.0)
        assert getattr(ax, "get_proj_type", lambda: "ortho")() == "ortho"
        caption = fig.texts[-1].get_text()
        assert "vertical exaggeration" not in caption, "nothing to exaggerate looking straight down"
        # Looking straight down, the Z axis is edge-on and its label would sit
        # on top of the title -- height is still carried honestly by colour.
        assert ax.get_zlabel() == ""
        assert ax.get_zticks().size == 0
    finally:
        figures._pyplot().close(fig)


def test_birdseye_frames_the_whole_ring_with_margin_not_a_crop(tmp_path):
    from tasni.modules.extrusion import figures

    take = figures.load_take(write_take(tmp_path))
    fig = figures._figure_birdseye(figures._pyplot(), take)
    try:
        ax = fig.axes[0]
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ring = take.measured
        assert x0 < ring[:, 0].min() and x1 > ring[:, 0].max(), "the ring is clipped in X"
        assert y0 < ring[:, 1].min() and y1 > ring[:, 1].max(), "the ring is clipped in Y"
    finally:
        figures._pyplot().close(fig)


def test_iso_states_and_actually_applies_its_vertical_exaggeration(tmp_path):
    """The recorded trap: a figure that CLAIMS a factor but draws a different one.

    The stated factor is read straight off the caption (black-box), then
    checked against the actual plotted data -- not recomputed from a helper
    the figure itself might disagree with.
    """
    import re
    from tasni.modules.extrusion import figures

    take = figures.load_take(write_take(tmp_path))
    fig = figures._figure_iso(figures._pyplot(), take)
    try:
        caption = fig.texts[-1].get_text()
        match = re.search(r"vertical exaggeration ×([\d.]+)", caption)
        assert match, caption
        factor = float(match.group(1))
        assert factor > 1.0, "fixture must produce a real exaggeration to check against"
        line = next(l for l in fig.axes[0].get_lines()
                   if l.get_label() == "extracted centreline")
        drawn_z = np.asarray(line.get_data_3d()[2])
        assert np.allclose(drawn_z, take.measured[:, 2] * factor)
    finally:
        figures._pyplot().close(fig)


def test_iso_and_birdseye_are_saved_for_every_take(tmp_path):
    from tasni.modules.extrusion import figures

    layer_dir = write_take(tmp_path)
    written = {p.name for p in figures.render_layer_figures(layer_dir, formats=("png",))}
    assert {"iso.png", "birdseye.png"} <= written


# --------------------------------------------------- ring characterizations (no recipe)

def test_load_take_reads_a_characterization_directory(tmp_path):
    """No manifest.json exists for a characterization -- report.json stands in."""
    from tasni.modules.extrusion import figures

    char_dir = write_characterization(tmp_path)
    assert not (char_dir / "manifest.json").is_file()
    assert (char_dir / "report.json").is_file()

    take = figures.load_take(char_dir)
    assert take.manifest["kind"] == "characterization"
    assert take.nominal is None, "no recipe existed yet -- there is no nominal ring"
    assert take.measured is not None and len(take.measured) > 10
    assert take.radius == pytest.approx(RADIUS, rel=.15)
    assert take.center == pytest.approx(CENTER, abs=5.0)
    assert "characterization" in take.label
    assert "no recipe assumed" in take.caption


def test_a_characterization_renders_the_full_figure_set(tmp_path):
    """The paper's whole stage sequence, reproducible from a characterization too."""
    pytest.importorskip("open3d")
    from tasni.modules.extrusion import figures

    char_dir = write_characterization(tmp_path)
    written = figures.render_layer_figures(char_dir, formats=("png",))
    names = {p.name for p in written}

    # Every figure this module knows how to draw comes out of a characterization,
    # not just the subset that happens to need no reconstructed plan.
    assert names == {f"{stem}.png" for stem in figures.LAYER_FIGURES}
    for path in written:
        assert path.stat().st_size > 1000, f"{path.name} is suspiciously small"


def test_a_characterizations_pipeline_figure_reruns_its_own_coarse_plan(tmp_path):
    """The coarse recipe characterize_ring used is never archived -- only the
    numbers it was built from (report['coarse']). This rebuilds it and re-runs
    the real chain, so the method figure shows what THIS pass actually saw."""
    pytest.importorskip("open3d")
    from tasni.modules.extrusion import figures

    take = figures.load_take(write_characterization(tmp_path))
    stages = figures.take_stages(take)

    assert stages and "result" in stages
    assert len(stages["backprojected"]) > len(stages["work_roi"]) > 0
    assert stages["result"].measured_xyz is not None


# ---------------------------------------------- a failed re-run is not a figure

def _break_the_chain(monkeypatch, message="synthetic chain failure"):
    """Fail ``process_observation`` after it has filled the early stages."""
    from tasni.modules.extrusion import processing

    def boom(**kwargs):
        stages = kwargs.get("stages")
        if stages is not None:
            stages["backprojected"] = np.column_stack(
                (np.full(40, CENTER[0]) + np.linspace(-30, 30, 40),
                 np.full(40, CENTER[1]) + np.linspace(-30, 30, 40),
                 np.linspace(0.0, 6.0, 40)))
            stages["work_roi"] = stages["backprojected"][:20].copy()
        raise RuntimeError(message)

    monkeypatch.setattr(processing, "process_observation", boom)


def test_a_characterization_whose_re_run_failed_is_marked_incomplete(
        tmp_path, monkeypatch, caplog):
    """A partial method figure that reads as a finished one is how a paper ends
    up illustrating a chain that never ran to the end."""
    pytest.importorskip("open3d")
    import logging
    from tasni.modules.extrusion import figures

    char_dir = write_characterization(tmp_path)
    _break_the_chain(monkeypatch)
    figures._STAGE_CACHE.clear()

    take = figures.load_take(char_dir)
    with caplog.at_level(logging.WARNING):
        stages = figures.take_stages(take)

    assert stages is not None, "the stages it did reach are still worth drawing"
    assert "result" not in stages
    assert "synthetic chain failure" in (stages.get("error") or "")
    assert "synthetic chain failure" in caplog.text, "the failure was swallowed"

    fig = figures._figure_pipeline(figures._pyplot(), take)
    try:
        text = " ".join(t.get_text() for t in fig.texts)
        assert "INCOMPLETE" in text
        assert "synthetic chain failure" in text
        titles = {ax.get_title() for ax in fig.axes}
        assert any("not re-run" in title for title in titles), \
            "the last panel is the archived path, not this run's output"
    finally:
        figures._pyplot().close(fig)


def test_a_layer_take_whose_re_run_failed_is_marked_incomplete_too(
        tmp_path, monkeypatch, caplog):
    import logging
    from tasni.modules.extrusion import figures

    layer_dir = write_take(tmp_path)
    _break_the_chain(monkeypatch, "layer chain failure")
    figures._STAGE_CACHE.clear()

    take = figures.load_take(layer_dir)
    with caplog.at_level(logging.WARNING):
        stages = figures.take_stages(take)

    assert stages and "layer chain failure" in (stages.get("error") or "")
    assert "layer chain failure" in caplog.text


# ------------------------------------- the box carries the stated exaggeration

def _bare_take(figures, *, radius: float, z: float = 6.0):
    """A take with no depth frame: enough for a 3-D view, nothing to re-run.

    A small ring is what exposes the defect -- ``_work_window`` never frames
    tighter than 60 mm, so a 10 mm ring sits in a window ~10x its own relief.
    """
    return figures.TakeData(
        layer_dir=Path("."),
        manifest={"trial_id": "box", "layer_index": 1, "take": 1, "mode": "MEASURE_ONLY",
                  "recipe": {"radius_mm": radius}, "metrics": {},
                  "provenance": {"work_frame": "Tasni Work Frame"}},
        nominal=_ring_xyz(radius=radius, z=z),
        measured=_ring_xyz(radius=radius, z=z, wave=.5),
        cloud=_cloud_xyz(radius=radius, z=z, bead=radius / 3.0),
        depth=None, K=None, T_work_camera=None)


def _drawn_z(ax) -> np.ndarray:
    values = [np.asarray(line.get_data_3d()[2], float) for line in ax.get_lines()
              if len(line.get_data_3d()[2])]
    for collection in ax.collections:
        offsets = getattr(collection, "_offsets3d", None)
        if offsets is not None:
            values.append(np.asarray(offsets[2], float))
    return np.concatenate([v for v in values if v.size])


def _drawn_exaggeration(ax, factor: float) -> float:
    """How many times taller a millimetre of Z is DRAWN than a millimetre of X.

    Black-box: the box proportions and the axis limits together, which is what a
    reader measures off the page -- not the number the code intended.
    """
    box = np.asarray(ax.get_box_aspect(), float)
    x0, x1 = ax.get_xlim()
    z0, z1 = ax.get_zlim()
    return (box[2] / box[0]) * ((x1 - x0) / (z1 - z0)) * factor


def test_the_oblique_view_draws_exactly_the_exaggeration_it_states(tmp_path):
    """Recorded defect: ``set_box_aspect`` was floored at .12, so the drawing
    exaggerated Z past the factor the caption quotes -- and for a factor of 1.0
    the caption quotes nothing at all. The floor now raises the FACTOR, which
    the plotted data, the Z axis and the caption all carry."""
    import re
    from tasni.modules.extrusion import figures

    take = _bare_take(figures, radius=10.0)
    fig = figures.render_view(take, plt=figures._pyplot())
    try:
        ax = fig.axes[0]
        caption = fig.texts[-1].get_text()
        match = re.search(r"vertical exaggeration ×([\d.]+)", caption)
        assert match, caption
        factor = float(match.group(1))
        assert np.allclose(
            np.asarray(ax.get_lines()[-1].get_data_3d()[2], float),
            take.measured[:, 2] * factor), "the data must carry the stated factor"

        box = np.asarray(ax.get_box_aspect(), float)
        x0, x1 = ax.get_xlim()
        drawn = np.ptp(_drawn_z(ax))          # == span * factor, by construction
        # The box's Z:X is the data's Z:X, give or take the padding both share.
        # A floor here would make the ratio larger than the data warrants --
        # exaggeration nobody can read off the figure.
        assert box[2] / box[0] == pytest.approx(drawn / (x1 - x0), rel=.11)
        # ...and end to end: a millimetre of height is drawn exactly ``factor``
        # times a millimetre of width, which is what the caption promises.
        assert _drawn_exaggeration(ax, factor) == pytest.approx(factor, rel=1e-6)
    finally:
        figures._pyplot().close(fig)


def test_a_normal_ring_is_drawn_at_the_exaggeration_its_caption_states(tmp_path):
    """The cell's own geometry, where the box floor never bit: the caption still
    has to match the drawing (it was 10% out -- Z was autoscaled with a margin
    while X was pinned to the window)."""
    import re
    from tasni.modules.extrusion import figures

    take = figures.load_take(write_take(tmp_path))
    fig = figures._figure_iso(figures._pyplot(), take)
    try:
        ax = fig.axes[0]
        match = re.search(r"vertical exaggeration ×([\d.]+)", fig.texts[-1].get_text())
        assert match
        factor = float(match.group(1))
        assert _drawn_exaggeration(ax, factor) == pytest.approx(factor, rel=1e-6)
    finally:
        figures._pyplot().close(fig)


def test_the_legibility_floor_is_stated_in_the_caption_not_hidden_in_the_box(tmp_path):
    """The floor is still applied -- a 3-D box has to read as one -- but through
    the number the figure prints, so a reader can undo it."""
    from tasni.modules.extrusion import figures

    take = _bare_take(figures, radius=10.0)
    cloud, window = figures._view_cloud(take)
    zs = [a[:, 2] for a in (cloud, take.measured, take.nominal) if a is not None]
    span = float(np.ptp(np.concatenate(zs)))
    dx, dy = figures._view_extent(cloud, window)
    plain = figures._z_exaggeration(take.radius, span)

    assert span * plain / max(dx, dy) < figures.MIN_RELIEF_RATIO, \
        "fixture must be one the old box floor would have silently stretched"
    raised = figures._legible_factor(plain, span, max(dx, dy))
    assert raised > plain
    assert span * raised / max(dx, dy) == pytest.approx(figures.MIN_RELIEF_RATIO, abs=.02)

    fig = figures.render_view(take, plt=figures._pyplot())
    try:
        assert f"vertical exaggeration ×{raised:g}" in fig.texts[-1].get_text()
    finally:
        figures._pyplot().close(fig)


def test_every_three_d_panel_draws_exactly_the_z_scale_it_states(tmp_path):
    """The same defect as the oblique view, in every other 3-D panel.

    Measured on the cell's own takes: the method figure's oblique panel stated
    ``Z × 1.7`` and drew ×14.9 (no box aspect at all, so matplotlib's default
    box was the exaggeration), and the tube figure captioned "True scale" over
    a flat ``(1, 1, .55)`` box. A panel that quotes a factor has to draw it.
    """
    pytest.importorskip("open3d")
    import re
    from tasni.modules.extrusion import figures

    plt = figures._pyplot()
    take = figures.load_take(write_take(tmp_path))
    checked = []
    for name, fig in (("pipeline", figures._figure_pipeline(plt, take)),
                      ("mesh", figures._figure_mesh(plt, take)),
                      ("stack", figures._figure_stack(plt, [take], "t1")),
                      ("tube", figures._figure_tube(plt, [take], "t1"))):
        assert fig is not None, f"{name} must be drawable from this fixture"
        try:
            for ax in fig.axes:
                if not hasattr(ax, "get_zlim"):
                    continue                       # a colourbar or a plan panel
                match = re.search(r"×\s?([\d.]+)", ax.get_title())
                factor = float(match.group(1)) if match else 1.0
                assert _drawn_exaggeration(ax, factor) == pytest.approx(factor, rel=1e-6), \
                    f"{name}: {ax.get_title()!r} states ×{factor:g}"
                checked.append(f"{name}:{factor:g}")
        finally:
            plt.close(fig)
    assert len(checked) >= 4, checked
