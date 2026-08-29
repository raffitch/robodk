"""Paper/app figures rendered from an archived take.

The renderer reads only what the archive holds, so these tests build a REAL
trial directory with the real writer and then render it -- the file format is
the contract under test, not the processing chain.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import extrusion_synthetic as syn
from tasni.core.config import ExtrusionConfig
from tasni.modules.extrusion.archive import ExtrusionArchive
from tasni.modules.extrusion.models import CylinderRecipe, CylinderSetup, LayerManifest
from tasni.modules.extrusion.toolpath import generate_cylinder_plan

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
               geometry=None, metrics=None):
    """One archived take, written exactly the way the measure job writes it."""
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
                                 T, plane_center_xy_mm=CENTER, seed=1)
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
                    "work_frame": "Tasni Work Frame"})
    return archive.write_layer(manifest, nominal_xyz=nominal, commanded_xyz=nominal,
                               measured_xyz=measured, pointcloud_xyz=cloud, depth=frame)


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
