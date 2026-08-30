import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { moduleApi } from "../api/client";
import { useEvents, type JobEvent } from "../api/events";

const api = moduleApi("extrusion");

interface Recipe {
  radius_mm: number; layer_count: number; layer_height_mm: number;
  bead_diameter_mm: number; robot_speed_mm_s: number;
  travel_speed_mm_s: number; path_rounding_mm: number;
  extrusion_rate_pct: number; points_per_circle: number;
  correction_enabled: boolean; material: unknown[];
}
interface Setup {
  print_tool: string; work_frame: string; inspection_tool: string;
  inspection_target: string; inspection_auto: boolean;
  center_x_mm: number; center_y_mm: number;
  build_plane_z_mm: number; scan_run_id: string | null;
  orientation_rpy_deg: [number, number, number];
  maximum_tool_axis_spin_deg: number;
  approach_clearance_mm: number; retreat_clearance_mm: number;
}
interface SurfaceFit {
  checked: boolean; inside: boolean; reason?: string;
  minimum_margin_mm?: number; outer_radius_mm?: number;
  margins_mm?: Record<string, number>;
}
interface ScanSurface {
  applied: boolean; available: boolean; note?: string;
  frame?: string; rectangle?: string | null; run_id?: string | null;
  applied_at?: string | null; size_mm?: [number, number] | null;
  center_mm?: [number, number] | null; center_z_mm?: number;
  // Where the middle came from. The station sources outlive a cleared runs/ tree;
  // "scan_run" is the disk pointer and is the only one that carries a run id.
  source?: "center_frame" | "surface_object" | "scan_run";
  extents_known?: boolean;
}
const PLATFORM_SOURCE: Record<string, string> = {
  center_frame: "centre frame in RoboDK",
  surface_object: "work-surface object in RoboDK",
  scan_run: "applied scan run",
};
interface InspectionPreview {
  auto: boolean; object_diameter_mm: number; standoff_mm: number; ok: boolean;
  work_frame: string; warnings: string[];
  framing: {
    d_fit_mm: number; clamped_to: string | null; fits: boolean;
    binding_axis: string; near_mm: number; far_mm: number;
    fill_fraction: { width: number; height: number };
  };
  layers: Array<{ layer_index: number; top_z_mm: number; camera_z_mm: number }>;
}
interface Point { x_mm: number; y_mm: number; z_mm: number; }
interface Layer { layer_index: number; nominal_z_mm: number; points: Point[]; }
interface Plan {
  fingerprint: string; recipe: Recipe; setup: Setup;
  total_path_length_mm: number; layers: Layer[];
  /** Set when the backend rebuilt this plan from a measurement session after a restart. */
  restored_from?: string | null;
}
interface Config {
  defaults: Recipe; setup_defaults: Setup;
  integration: {
    air_on_program: string; air_off_program: string; valve_outputs: string[];
    mapping_source: string; mapping_verified: boolean;
    hardware_io_test_approved: boolean; extrusion_rate_control: string;
  };
  measure_close_range_min_mm: number;
  live_print_enabled: boolean;
}
interface StationOptions { tools: string[]; frames: string[]; targets: string[]; programs: string[]; }
interface Status {
  status: string; running: boolean; result?: any; error?: string | null;
  fingerprint?: string | null; geometry_preflight_passed: boolean;
  quick_sim_passed: boolean; quick_sim_layers: number[];
  quick_sim_live_approved: boolean; dry_run_passed: boolean;
  hardware_io_test_approved: boolean;
  live_print_enabled: boolean;
  measure_session?: string | null;
}

// -- ring-stack measure-only experiment (no extrusion; camera move only) ----
interface MeasureTake {
  layer_index: number; take: number; layer_dir: string; valid: boolean; timestamp: string;
  // Absent on takes archived before figures existed; derived from index+take then.
  layer_name?: string;
  annotation: { introduced_offset_mm?: [number, number] | null; note?: string; phase?: string };
  // A take whose processing failed has no metrics: its raw RGB-D is archived and it
  // can be reprocessed offline, so it stays in the table with the reason.
  metrics: { mean_absolute_mm: number; rms_mm: number; maximum_mm: number;
    center_offset_mm: [number, number]; center_offset_norm_mm: number; shape_rms_mm: number;
    measured_radius_mm: number; path_completeness: number } | null;
  geometry: { height_mean_mm: number; height_min_mm: number; height_max_mm: number;
    bead_width_mean_mm: number } | null;
  timings_ms: { capture_ms?: number; total_ms?: number; acquisition_to_path_ms?: number;
    move_to_pose_ms?: number; settle_ms?: number; return_ms?: number;
    /** Out of the path, settle, capture, reconstruct, back: what one inspection costs. */
    inspection_cycle_ms?: number };
  /** One side-on RGB photo of the stack, taken after this layer's capture. */
  side_view?: { captured: boolean; image_file?: string | null; target?: string | null;
                approach_target?: string | null; excursion_ms?: number | null;
                error?: string | null } | null;
  error?: string | null;
  reprocessed?: boolean;
}
/** The plan a session applied from its characterization: every take is scored against it. */
interface AppliedPlan {
  fingerprint: string; applied_at?: string; characterization_index?: number;
  recipe: { radius_mm: number; bead_diameter_mm: number; layer_height_mm: number; layer_count: number };
  setup: { center_x_mm: number; center_y_mm: number };
}
interface Characterization {
  index: number; radius_mm: number; center_mm: [number, number]; bead_width_mm: number;
  top_z_mean_mm: number; top_z_min_mm: number; top_z_max_mm: number;
}
interface MeasureSession {
  trial_id: string; takes: Record<string, number>; records: MeasureTake[];
  characterizations: Characterization[];
  // layer index -> the measured centreline of its latest take, in work-frame mm.
  tops?: Record<string, number[][]>;
  applied?: AppliedPlan | null;
}

/** The protocol's phases. Grouped by the paper summary, so they must be recorded. */
const PHASES: Array<{ value: string; label: string; hint: string }> = [
  { value: "noise floor", label: "Noise floor",
    hint: "ring untouched between takes — this is sensing repeatability" },
  { value: "re-placed", label: "Re-placed",
    hint: "ring lifted and put back by hand — placement repeatability" },
  { value: "stacked true", label: "Stacked true",
    hint: "the next ring placed as accurately as you can, no deliberate offset" },
  { value: "top ring shifted", label: "Top ring shifted",
    hint: "the top ring placed off-centre by the offset you typed — the controlled validation" },
];

/** The protocol's shape. Constants, not state: they define the run, not its progress. */
const REPEATS = 3;          // frames per condition, taken with the arm PARKED
const NOISE_TRIPS = 5;      // whole excursions out and back, unattended, one press
// Three rings stacked as true as a hand can, then a fourth placed deliberately
// off-centre: the controlled error the paper's claim rests on.
const STACK_LAYERS = 4;

/** One step of the ring run: what to do, the single press that does it. */
interface RunStep {
  id: string; label: string; title: string; done: boolean;
  /** The layer this step measures, when it measures one. */
  layer?: number;
  hands?: string;
  records?: string;
  button?: string;
  onRun?: () => void;
  disabled?: boolean;
  blocked?: string | null;
  note?: string;
  progress?: { have: number; need: number };
  moves?: boolean;
  offsetInput?: boolean;
  axisAck?: boolean;
  /** This step's press calls /measure/layer, so the capture toggles apply to it.
   *  False for "ring1" (characterize — multiview is accepted there but not yet
   *  wired to the star capture, so the toggle must not appear on it). */
  capture?: boolean;
}

/** The archive's own naming: take 1 keeps the historical name, repeats get a suffix. */
function layerDirName(take: MeasureTake): string {
  if (take.layer_name) return take.layer_name;
  const layer = `layer-${String(take.layer_index).padStart(3, "0")}`;
  return take.take === 1 ? layer : `${layer}-take${String(take.take).padStart(2, "0")}`;
}

const FIGURES: Array<{ stem: string; label: string; hint: string }> = [
  { stem: "pipeline", label: "How it was measured",
    hint: "every stage: captured depth → ROI → deposit → crest → centreline" },
  { stem: "plan", label: "Plan view", hint: "deposit cloud, extracted centreline, nominal ring" },
  { stem: "heightmap", label: "Height map", hint: "bird's-eye relief of the depth frame" },
  { stem: "mesh", label: "Surfaced view",
    hint: "the frame meshed — work surface and deposit, from above and rotated" },
  { stem: "iso", label: "Oblique", hint: "3-D cloud + centreline, vertical exaggeration" },
  { stem: "profile", label: "Unrolled profile", hint: "height and radial deviation vs angle" },
];

/** Figures are rendered on first request, so the first load of a take is slow. */
function TakeFigures({ trialId, take }: { trialId: string; take: MeasureTake }) {
  const base = `/api/modules/extrusion/trials/${encodeURIComponent(trialId)}`
    + `/layers/${encodeURIComponent(layerDirName(take))}`;
  return <div className="figure-gallery">
    {FIGURES.map(({ stem, label, hint }) => <figure key={stem} className="figure-card">
      <a href={`${base}/figures/${stem}.png`} target="_blank" rel="noreferrer">
        <img src={`${base}/figures/${stem}.png`} alt={`${label} of layer ${take.layer_index} take ${take.take}`} loading="lazy" />
      </a>
      <figcaption>
        <strong>{label}</strong> <span className="hint">{hint}</span>
        <span className="figure-links">
          <a href={`${base}/figures/${stem}.png`} target="_blank" rel="noreferrer">PNG</a>
          <a href={`${base}/figures/${stem}.pdf`} target="_blank" rel="noreferrer">PDF</a>
        </span>
      </figcaption>
    </figure>)}
    <figure className="figure-card">
      <a href={`${base}/files/color.png`} target="_blank" rel="noreferrer">
        <img src={`${base}/files/color.png`} alt="Colour frame as captured" loading="lazy" />
      </a>
      <figcaption><strong>Colour frame</strong> <span className="hint">what the camera saw</span></figcaption>
    </figure>
    <figure className="figure-card">
      <a href={`${base}/files/comparison.png`} target="_blank" rel="noreferrer">
        <img src={`${base}/files/comparison.png`} alt="Segmentation with nominal and measured paths" loading="lazy" />
      </a>
      <figcaption><strong>Segmentation</strong> <span className="hint">raster the centreline came from</span></figcaption>
    </figure>
    <figure className="figure-card">
      <a href={`${base}/files/skeleton.png`} target="_blank" rel="noreferrer">
        <img src={`${base}/files/skeleton.png`} alt="Skeleton thinned from the segmentation" loading="lazy" />
      </a>
      <figcaption><strong>Skeleton</strong> <span className="hint">
        the segmentation thinned to one pixel — the centreline before it was
        mapped back to 3-D</span></figcaption>
    </figure>
    {/* Only the last take of a press carries the photo -- the ring does not move
        between the frames of one capture, so the others would repeat it. */}
    {take.side_view?.captured && <figure className="figure-card">
      <a href={`${base}/files/side.png`} target="_blank" rel="noreferrer">
        <img src={`${base}/files/side.png`} alt={`The stack seen from the side after layer ${take.layer_index}`} loading="lazy" />
      </a>
      <figcaption><strong>Side view</strong> <span className="hint">
        the stack from {take.side_view.target ?? "the taught side pose"} — a photo for the
        paper, measured from nothing</span></figcaption>
    </figure>}
    {take.side_view && !take.side_view.captured && <figure className="figure-card empty">
      <figcaption><strong>Side view</strong> <span className="hint warn-text">
        {take.side_view.error ?? "not captured"}</span></figcaption>
    </figure>}
  </div>;
}

const recipeFields: Array<{ key: keyof Recipe; label: string; min: number; max: number; step: number; unit: string }> = [
  { key: "radius_mm", label: "Radius", min: 5, max: 150, step: 1, unit: "mm" },
  { key: "layer_count", label: "Layers", min: 1, max: 30, step: 1, unit: "" },
  { key: "layer_height_mm", label: "Layer height", min: .5, max: 20, step: .5, unit: "mm" },
  { key: "bead_diameter_mm", label: "Bead diameter", min: .5, max: 30, step: .5, unit: "mm" },
  { key: "robot_speed_mm_s", label: "Process speed", min: 5, max: 500, step: 5, unit: "mm/s" },
  { key: "travel_speed_mm_s", label: "Travel speed", min: 5, max: 1000, step: 5, unit: "mm/s" },
  { key: "path_rounding_mm", label: "Path blending", min: 0, max: 25, step: .5, unit: "mm" },
  { key: "extrusion_rate_pct", label: "Extrusion rate", min: 0, max: 100, step: 1, unit: "%" },
  { key: "points_per_circle", label: "Curve samples", min: 24, max: 720, step: 12, unit: "pts" },
];

function BirdseyeStack({ plan, selectedLayer, onSelect, measured, showBead,
                        measuredBead }: {
  plan: Plan; selectedLayer: number; onSelect: (layer: number) => void;
  // Measured centrelines by layer index, in work-frame mm: what is actually
  // on the table, drawn over what was commanded.
  measured?: Record<string, number[][]>;
  // Draw each layer at the commanded bead width instead of as a hairline: what
  // is deposited is a bead with a footprint, and the curve alone hides the
  // quantity the inspection measures.
  showBead?: boolean;
  // Layer index -> the bead footprint that layer's latest take MEASURED, in mm.
  // The commanded width is a setting, this one is a result; drawing the result
  // at the commanded width would show a comparison that was never made.
  measuredBead?: Record<string, number>;
}) {
  const width = 640, height = 440, pad = 38;
  const { radius_mm: radius, layer_height_mm: layerHeight } = plan.recipe;
  const { center_x_mm: centerX, center_y_mm: centerY } = plan.setup;
  const baseZ = plan.layers[0].nominal_z_mm;
  const topZ = plan.layers[plan.layers.length - 1].nominal_z_mm;
  const actualHeight = Math.max(0, topZ - baseZ);
  // Thin real layers need a modest vertical exaggeration to remain individually
  // visible. XY stays exact; the displayed factor makes the visualization honest.
  const desiredHeight = plan.layers.length > 1
    ? Math.min(radius * 1.4, Math.max(actualHeight, (plan.layers.length - 1) * radius * .16))
    : 0;
  const zExaggeration = actualHeight > 0 ? desiredHeight / actualHeight : 1;
  const raw = (x: number, y: number, z: number) => ({
    x: (x - centerX - (y - centerY)) * .866,
    y: (x - centerX + (y - centerY)) * .36 - (z - baseZ) * zExaggeration,
  });
  const extent = Math.max(radius * 1.25, 1);
  const plane = [
    raw(centerX - extent, centerY - extent, baseZ),
    raw(centerX + extent, centerY - extent, baseZ),
    raw(centerX + extent, centerY + extent, baseZ),
    raw(centerX - extent, centerY + extent, baseZ),
  ];
  const projectedLayers = plan.layers.map((item) => ({
    item,
    rawPoints: item.points.map((p) => raw(p.x_mm, p.y_mm, p.z_mm)),
  }));
  const projectedMeasured = Object.entries(measured ?? {}).map(([index, points]) => ({
    layerIndex: Number(index),
    rawPoints: points.map((p) => raw(p[0], p[1], p[2])),
  })).filter((entry) => entry.rawPoints.length > 1);
  const bounds = [...plane, ...projectedLayers.flatMap((entry) => entry.rawPoints),
                  ...projectedMeasured.flatMap((entry) => entry.rawPoints)];
  const minX = Math.min(...bounds.map((p) => p.x));
  const maxX = Math.max(...bounds.map((p) => p.x));
  const minY = Math.min(...bounds.map((p) => p.y));
  const maxY = Math.max(...bounds.map((p) => p.y));
  const scale = Math.min((width - 2 * pad) / Math.max(1, maxX - minX),
                         (height - 2 * pad) / Math.max(1, maxY - minY));
  const project = (p: { x: number; y: number }) => ({
    x: pad + (p.x - minX) * scale,
    y: pad + (p.y - minY) * scale,
  });
  const path = (points: Array<{ x: number; y: number }>) => points.map((p, index) => {
    const q = project(p);
    return `${index ? "L" : "M"}${q.x.toFixed(2)},${q.y.toFixed(2)}`;
  }).join(" ");
  const floor = plane.map(project);
  const axisOrigin = project(raw(centerX, centerY, baseZ));
  const axes = [
    { label: "+X", end: project(raw(centerX + extent, centerY, baseZ)), color: "#f0a45d" },
    { label: "+Y", end: project(raw(centerX, centerY + extent, baseZ)), color: "#39d0bd" },
    { label: "+Z", end: project(raw(centerX, centerY, topZ || baseZ)), color: "#8ab4ff" },
  ];
  const guideIndices = [0, .25, .5, .75].map((fraction) =>
    Math.min(plan.recipe.points_per_circle - 1,
             Math.round(fraction * plan.recipe.points_per_circle)));
  // The oblique projection compresses X by .866, so that is the honest factor
  // for a width drawn in the plane of the layer.
  const beadPx = Math.max(2, plan.recipe.bead_diameter_mm * scale * .866);

  return <svg viewBox={`0 0 ${width} ${height}`} className="birdseye-map"
              aria-label="Oblique bird's-eye view of the complete layer stack">
    <rect width={width} height={height} fill="#090d14" />
    <polygon points={floor.map((p) => `${p.x},${p.y}`).join(" ")}
             fill="#0d1420" stroke="#263247" strokeWidth="1.2" />
    {[-.5, 0, .5].map((offset) => {
      const a = project(raw(centerX - extent, centerY + offset * extent, baseZ));
      const b = project(raw(centerX + extent, centerY + offset * extent, baseZ));
      const c = project(raw(centerX + offset * extent, centerY - extent, baseZ));
      const d = project(raw(centerX + offset * extent, centerY + extent, baseZ));
      return <g key={offset} stroke={offset === 0 ? "#344055" : "#1b2636"} strokeWidth="1">
        <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} />
        <line x1={c.x} y1={c.y} x2={d.x} y2={d.y} />
      </g>;
    })}
    {plan.layers.length > 1 && guideIndices.map((pointIndex) => {
      const bottom = project(projectedLayers[0].rawPoints[pointIndex]);
      const top = project(projectedLayers[projectedLayers.length - 1].rawPoints[pointIndex]);
      return <line key={pointIndex} x1={bottom.x} y1={bottom.y} x2={top.x} y2={top.y}
                   stroke="#31425b" strokeWidth="1" strokeDasharray="4 5" />;
    })}
    {projectedLayers.map(({ item, rawPoints }) => {
      const selected = item.layer_index === selectedLayer;
      const start = project(rawPoints[0]);
      const depth = .42 + item.layer_index / plan.layers.length * .34;
      return <g key={item.layer_index} className="stack-layer"
                onClick={() => onSelect(item.layer_index)}>
        {showBead && <path d={path(rawPoints)} fill="none"
              stroke={selected ? "#66a6ff" : "#39d0bd"}
              strokeOpacity={(selected ? .34 : depth * .38)}
              strokeWidth={beadPx} strokeLinecap="round" strokeLinejoin="round" />}
        <path d={path(rawPoints)} fill={selected && !showBead ? "rgba(76,154,255,.10)" : "none"}
              stroke={selected ? "#66a6ff" : "#39d0bd"}
              strokeOpacity={selected ? 1 : depth}
              strokeWidth={selected ? (showBead ? 2 : 4) : (showBead ? 1.2 : 2)}
              strokeLinecap="round" strokeLinejoin="round" />
        {selected && <circle cx={start.x} cy={start.y} r="5" fill="#f0a45d" />}
      </g>;
    })}
    {showBead && projectedMeasured.map(({ layerIndex, rawPoints }) => {
      const width = measuredBead?.[String(layerIndex)];
      if (!width) return null;
      return <path key={`measured-bead-${layerIndex}`} d={path(rawPoints)} fill="none"
        stroke="#f0616d" strokeOpacity={layerIndex === selectedLayer ? .38 : .26}
        strokeWidth={Math.max(2, width * scale * .866)}
        strokeLinecap="round" strokeLinejoin="round" />;
    })}
    {projectedMeasured.map(({ layerIndex, rawPoints }) => <path
      key={`measured-${layerIndex}`} d={path(rawPoints)} fill="none" stroke="#f0616d"
      strokeWidth={layerIndex === selectedLayer ? 3.4 : 2.2}
      strokeOpacity={layerIndex === selectedLayer ? 1 : .72}
      strokeLinecap="round" strokeLinejoin="round" />)}
    {axes.map((axis) => <g key={axis.label}>
      <line x1={axisOrigin.x} y1={axisOrigin.y} x2={axis.end.x} y2={axis.end.y}
            stroke={axis.color} strokeWidth="1.6" />
      <text x={axis.end.x + 6} y={axis.end.y - 5} className="preview-axis"
            style={{ fill: axis.color }}>{axis.label}</text>
    </g>)}
    <text x="14" y={height - 14} className="preview-note">
      OBLIQUE XYZ · Z ×{zExaggeration.toFixed(1)} · ΔZ {layerHeight.toFixed(2)} mm
      {projectedMeasured.length ? "  ·  teal = commanded, red = measured" : ""}
    </text>
  </svg>;
}

export default function Extrusion() {
  const { subscribe } = useEvents();
  const [config, setConfig] = useState<Config | null>(null);
  const [recipe, setRecipe] = useState<Recipe | null>(null);
  const [setup, setSetup] = useState<Setup | null>(null);
  const [options, setOptions] = useState<StationOptions | null>(null);
  const [connected, setConnected] = useState(false);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [selectedLayer, setSelectedLayer] = useState(1);
  const [preflight, setPreflight] = useState<any>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [result, setResult] = useState<any>(null);
  const [message, setMessage] = useState("Connect to RoboDK and select the exact station items.");
  const [logs, setLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState({ step: 0, total: 1, message: "" });
  const [busy, setBusy] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [confirmLive, setConfirmLive] = useState(false);
  const [quickLayers, setQuickLayers] = useState<number[]>([1]);
  const [approveRepresentativeLayers, setApproveRepresentativeLayers] = useState(false);
  // The operator's decision, not a derived one: the quick visual simulation is
  // where they see collisions, and per-layer re-validation is the slow step.
  const [liveCollisionCheck, setLiveCollisionCheck] = useState(false);
  // Generated programs/targets are the only record of what the robot was told
  // to do, so the operator decides whether the run cleans them up.
  const [keepArtifacts, setKeepArtifacts] = useState(false);
  const [surface, setSurface] = useState<ScanSurface | null>(null);
  const [inspection, setInspection] = useState<InspectionPreview | null>(null);
  const [measureSession, setMeasureSession] = useState<MeasureSession | null>(null);
  const [measureLayer, setMeasureLayer] = useState(1);
  const [offsetX, setOffsetX] = useState(0);
  const [offsetY, setOffsetY] = useState(0);
  const [measureNote, setMeasureNote] = useState("");
  const [phase, setPhase] = useState<string>("noise floor");
  const [guideOpen, setGuideOpen] = useState(false);
  // Two workflows share this page and need almost nothing from each other. The
  // measuring run is the one on the clock, so it opens by default and the print
  // controls stay out of the way until they are asked for.
  const [mode, setMode] = useState<"measure" | "print">(() => {
    try { return localStorage.getItem("extrusion.mode") === "print" ? "print" : "measure"; }
    catch { return "measure"; }
  });
  useEffect(() => {
    try { localStorage.setItem("extrusion.mode", mode); } catch { /* private mode */ }
  }, [mode]);
  const guideDialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = guideDialog.current;
    if (!dialog) return;
    if (guideOpen && !dialog.open) dialog.showModal();
    if (!guideOpen && dialog.open) dialog.close();
  }, [guideOpen]);
  const [confirmMotion, setConfirmMotion] = useState(false);
  // Independent of each other and of everything else on the run: merged-with-
  // no-photo, single-view-with-photo, both, neither — all reachable. Defaults
  // mirror the config the job falls back to when a request omits the field
  // (multiview off, side photo on), so leaving these untouched behaves exactly
  // like the pre-toggle backend default.
  const [multiview, setMultiview] = useState(false);
  const [sidePhoto, setSidePhoto] = useState(true);
  const [paper, setPaper] = useState<string | null>(null);
  const [openTake, setOpenTake] = useState<string | null>(null);
  const [showStack, setShowStack] = useState(false);
  // Eight columns answer "did that take work?"; the rest are for writing the
  // paper afterwards and only get in the way at the cell.
  const [allColumns, setAllColumns] = useState(false);
  // Which step the operator is looking at. Null = wherever the run actually is;
  // clicking a chip in the rail pins an earlier step until it is done again.
  const [stepPin, setStepPin] = useState<number | null>(null);
  const [offsetAxis, setOffsetAxis] = useState<"X" | "Y">("X");
  const [offsetMag, setOffsetMag] = useState(10);
  const [axisKnown, setAxisKnown] = useState(false);
  const [manualOpen, setManualOpen] = useState(false);
  // Off by default: the hairline is the toolpath, and the band is what that
  // toolpath actually lays down. Remembered per browser.
  const [showBead, setShowBead] = useState(() => {
    try { return localStorage.getItem("extrusion.showBead") === "1"; } catch { return false; }
  });
  useEffect(() => {
    try { localStorage.setItem("extrusion.showBead", showBead ? "1" : "0"); }
    catch { /* private mode */ }
  }, [showBead]);
  const logRef = useRef<HTMLDivElement>(null);

  const refreshStatus = useCallback(() => {
    api.get<Status>("/status").then((value) => {
      setStatus(value);
      if (value.result?.kind?.startsWith("cylinder_")) setResult(value.result);
    }).catch(() => {});
  }, []);
  // The platform is resolved IN a frame, so this has to be re-asked whenever the
  // work-frame selection changes -- the same table has different coordinates in each.
  const refreshSurface = useCallback((frame: string) => {
    if (!frame) { setSurface(null); return; }
    api.get<ScanSurface>(`/scan-surface?work_frame=${encodeURIComponent(frame)}`)
      .then(setSurface).catch(() => setSurface(null));
  }, []);
  useEffect(() => {
    api.get<Config>("/config").then((value) => {
      setConfig(value); setRecipe(value.defaults); setSetup(value.setup_defaults);
    }).catch((e) => setMessage(e.message));
    api.get<Plan>("/plan").then((value) => {
      setPlan(value); setRecipe(value.recipe); setSetup(value.setup);
      setSelectedLayer(1); setQuickLayers([1]);
    }).catch(() => {});
    refreshStatus();
  }, [refreshStatus]);
  useEffect(() => {
    refreshSurface(setup?.work_frame ?? "");
  }, [refreshSurface, setup?.work_frame, connected]);

  useEffect(() => subscribe((event: JobEvent) => {
    const name = event.payload?.name as string | undefined;
    if (event.type === "progress" && busy) {
      setProgress(event.payload); setMessage(event.payload.message || "Working…");
    } else if (event.type === "log" && busy) {
      setLogs((old) => [...old, event.payload.message]);
    } else if (event.type === "result" && name?.startsWith("extrusion-")) {
      setResult(event.payload.result); setBusy(false); setCancelling(false); setConfirmLive(false);
      setMessage(name === "extrusion-quick-sim"
        ? `Quick visual simulation passed for layer(s) ${event.payload.result.simulated_layer_indices.join(", ")}.`
        : name === "extrusion-dry-run"
          ? "Collision-validated dry run passed for this exact plan."
          : "Print and layer archive completed.");
      refreshStatus();
    } else if (event.type === "error" && name?.startsWith("extrusion-")) {
      setBusy(false); setCancelling(false); setMessage(event.payload.message); setLogs((old) => [...old, `ERROR: ${event.payload.message}`]);
      refreshStatus();
    } else if (event.type === "status" && name?.startsWith("extrusion-")) {
      if (event.payload.status === "cancelled") {
        setBusy(false); setCancelling(false); setConfirmLive(false);
        setMessage("Cancelled. RoboDK has finished the job exit sequence.");
        setLogs((old) => [...old, "Cancellation complete."]);
      }
      refreshStatus();
    }
  }), [subscribe, refreshStatus, busy]);

  // Poll while cancellation is pending in case the terminal WebSocket event
  // was emitted while this page was reconnecting.
  useEffect(() => {
    if (!cancelling) return;
    const timer = window.setInterval(refreshStatus, 500);
    return () => window.clearInterval(timer);
  }, [cancelling, refreshStatus]);

  useEffect(() => {
    if (!cancelling || !status || status.running) return;
    setBusy(false); setCancelling(false); setConfirmLive(false);
    setMessage(status.status === "cancelled"
      ? "Cancelled. RoboDK has finished the job exit sequence."
      : `The job ended with status: ${status.status}.`);
  }, [cancelling, status]);

  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [logs]);

  const invalidate = (note = "Inputs changed — generate again; prior checks were invalidated.") => {
    setPlan(null); setPreflight(null); setResult(null); setConfirmLive(false);
    setQuickLayers([1]); setApproveRepresentativeLayers(false);
    setInspection(null); setMessage(note);
    setStatus((old) => old ? { ...old, geometry_preflight_passed: false,
      quick_sim_passed: false, quick_sim_layers: [], quick_sim_live_approved: false,
      dry_run_passed: false, live_print_enabled: false } : old);
  };
  const updateRecipe = (key: keyof Recipe, value: number | boolean) => {
    setRecipe((old) => old ? { ...old, [key]: value } : old); invalidate();
  };
  const updateSetup = (key: keyof Setup, value: any) => {
    setSetup((old) => old ? { ...old, [key]: value } : old); invalidate();
  };
  const updateOrientation = (index: number, value: number) => {
    if (!setup) return;
    const next = [...setup.orientation_rpy_deg] as [number, number, number]; next[index] = value;
    updateSetup("orientation_rpy_deg", next);
  };

  const seedFromCurrentTcp = async () => {
    if (!setup || !recipe || !setup.print_tool || !setup.work_frame) return;
    setBusy(true); setMessage("Reading the selected TCP pose from RoboDK…");
    try {
      const pose = await api.post<{ xyz_mm: number[]; rpy_deg: number[] }>("/current-tcp", {
        print_tool: setup.print_tool, work_frame: setup.work_frame,
      });
      const next: Setup = {
        ...setup,
        // Circle angle zero is center + radius on X, so make it the current TCP.
        center_x_mm: pose.xyz_mm[0] - recipe.radius_mm,
        center_y_mm: pose.xyz_mm[1],
        build_plane_z_mm: pose.xyz_mm[2] - recipe.bead_diameter_mm / 2,
        orientation_rpy_deg: pose.rpy_deg as [number, number, number],
        // Jogged placement is manual by definition — it must not keep claiming the
        // scanned surface, or preflight would check a fit this centre never had.
        scan_run_id: null,
      };
      setSetup(next); invalidate("Path start and orientation captured from the current TCP — generate coordinates next.");
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };

  const centerOnPlatform = async () => {
    if (!setup || !recipe || !setup.work_frame) return;
    // Re-centring on the scanned surface throws away a placement measured from
    // the physical ring. Doing that mid-session and regenerating is exactly how
    // the 2026-08-28 stale-plan artifact was produced.
    if (measureSession?.applied && !window.confirm(
      `This session measures against the ring characterized at r ${measureSession.applied.recipe.radius_mm} mm, `
      + `centre (${measureSession.applied.setup.center_x_mm.toFixed(1)}, `
      + `${measureSession.applied.setup.center_y_mm.toFixed(1)}). Centring on the scanned surface `
      + "replaces that placement with the platform centre, and measuring against it would score the "
      + "ring against a plan it was never placed on.\n\nRe-centre anyway?")) return;
    setBusy(true); setMessage(`Locating the middle of the platform in ${setup.work_frame}…`);
    try {
      const response = await api.post<{ setup: Partial<Setup>; surface: ScanSurface; fit: SurfaceFit }>(
        "/center-on-surface", { radius_mm: recipe.radius_mm,
          bead_diameter_mm: recipe.bead_diameter_mm, work_frame: setup.work_frame });
      const frame = response.setup.work_frame || setup.work_frame;
      const pose = await api.post<{ xyz_mm: number[]; rpy_deg: number[] }>("/current-tcp", {
        print_tool: setup.print_tool, work_frame: frame,
      });
      setSurface({ ...response.surface, applied: true });
      setSetup({ ...setup, ...response.setup,
        orientation_rpy_deg: pose.rpy_deg as [number, number, number] });
      const size = response.surface.size_mm;
      const where = `on the middle of the platform${size ? ` (${size[0].toFixed(0)} × ${size[1].toFixed(0)} mm)` : ""} in ${frame}, from the ${PLATFORM_SOURCE[response.surface.source ?? ""] ?? "platform"}`;
      invalidate(response.fit.checked === false
        ? `Centred ${where}; its extents are unknown here, so the wall was NOT checked against them — ${response.fit.reason ?? "no bounds"}.`
        : response.fit.inside
        ? `Centred ${where}; current neutral TCP orientation captured as ${pose.rpy_deg.map((v) => v.toFixed(1)).join(", ")}° — generate coordinates next.`
        : `Centred, but the wall overhangs the platform by ${Math.abs(response.fit.minimum_margin_mm ?? 0).toFixed(1)} mm. Reduce the radius or re-scan a larger surface; preflight will reject it.`);
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };

  const connect = async () => {
    setBusy(true); setMessage("Loading RoboDK station…");
    try {
      await api.post("/connect");
      const discovered = await api.get<StationOptions>("/station-options");
      setOptions(discovered); setConnected(true);
      setMessage("Station loaded. Select print/inspection tools, work frame, and inspection target.");
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
  const generate = async (recipeOverride?: Recipe) => {
    const useRecipe = recipeOverride ?? recipe;
    if (!useRecipe || !setup) return;
    setBusy(true);
    try {
      if (recipeOverride) setRecipe(recipeOverride);
      const next = await api.post<Plan>("/generate", { recipe: useRecipe, setup });
      setPlan(next); setSelectedLayer(1); setQuickLayers([1]);
      setApproveRepresentativeLayers(false); setPreflight(null); setResult(null);
      // Pure geometry, no station: safe to show the derived viewpoint immediately.
      const derived = await api.post<InspectionPreview>(
        "/inspection-pose", { fingerprint: next.fingerprint }).catch(() => null);
      setInspection(derived);
      setMessage(`Generated ${next.layers.length} complete closed paths.`); refreshStatus();
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
  const runPreflight = async () => {
    if (!plan) return;
    setBusy(true);
    try {
      const value = await api.post<any>("/preflight", { fingerprint: plan.fingerprint });
      setPreflight(value);
      const unreachable = value.station?.reachability?.first_unreachable;
      setMessage(value.surface?.ok === false ? value.surface.problem
        : value.station?.ready ? value.note
        : unreachable
          ? `No IK solution at sampled ${value.station.reachability.frame} coordinate (${unreachable.xyz_mm.map((v: number) => v.toFixed(1)).join(", ")}) mm. Re-seed from the current TCP.`
          : value.station?.error || "Geometry passed, but station placement or selected items are not ready.");
      refreshStatus();
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
  const startJob = async (kind: "quick-sim" | "dry-run" | "print") => {
    if (!plan) return;
    const selectedCount = kind === "quick-sim" ? quickLayers.length : plan.layers.length;
    setBusy(true); setCancelling(false); setLogs([]); setProgress({ step: 0, total: selectedCount, message: "Starting…" });
    try {
      await api.post(`/${kind}`, kind === "print"
        ? { fingerprint: plan.fingerprint, confirm_live: confirmLive,
            collision_check_enabled: liveCollisionCheck, keep_artifacts: keepArtifacts }
        : kind === "quick-sim"
          ? { fingerprint: plan.fingerprint, layer_indices: quickLayers,
              approve_full_plan: approveRepresentativeLayers }
          : { fingerprint: plan.fingerprint });
      setMessage(kind === "quick-sim"
        ? `QUICK SIMULATION started for layer(s) ${quickLayers.join(", ")} — collision checks and physical outputs are blocked.`
        : kind === "dry-run"
          ? "VALIDATED DRY RUN started — collision checks on; physical outputs blocked."
          : `LIVE_PRINT started — collision checks ${liveCollisionCheck ? "ON" : "OFF by operator selection"}.`);
      refreshStatus();
    } catch (e: any) { setBusy(false); setMessage(e.message); }
  };
  // -- ring-stack measure-only experiment ------------------------------------
  const refreshMeasure = useCallback(async () => {
    try {
      const data = await api.get<{ session: MeasureSession | null }>("/measure/session");
      setMeasureSession(data.session);
    } catch { /* module unavailable */ }
  }, []);
  useEffect(() => { refreshMeasure(); }, [refreshMeasure]);
  // A take is only on disk once the job finishes, so re-read on every
  // running -> idle edge rather than polling.
  useEffect(() => { if (status && !status.running) refreshMeasure(); },
            [status?.running, refreshMeasure]);
  /** What a press will archive as ground truth, in the operator's words. */
  const annotationLabel = (layerIndex = measureLayer, takePhase = phase,
                           offset: [number, number] = [offsetX, offsetY]): string => {
    const parts = [`layer ${layerIndex}`];
    if (takePhase) parts.push(takePhase);
    parts.push(offset[0] || offset[1] ? `introduced offset (${offset[0]}, ${offset[1]}) mm`
                                      : "no introduced offset");
    return parts.join(" · ");
  };
  /** Tabula rasa: aim at the table, freeze a 3-layer plan there, open a session.
   *
   * The plan is only a SEARCH SEED -- characterize_ring makes no recipe
   * assumption, it fits a circle within 150 mm of the plan's centre -- so a
   * fresh ring needs a centre over the table and nothing else. Asking the
   * operator to "generate a plan" first implied they had to know the ring's
   * size before measuring it, which is backwards.
   */
  const freshStart = async () => {
    if (!recipe || !setup) return;
    const takenSoFar = measureSession?.records?.length ?? 0;
    if (takenSoFar && !window.confirm(
      `Session ${measureSession?.trial_id} has ${takenSoFar} take(s). They stay archived, `
      + "but measurements from now on go to a NEW session and the run starts again at "
      + "ring 1.\n\nStart a fresh ring?")) return;
    setBusy(true); setLogs([]); setMessage("Aiming at the table for a fresh ring…");
    try {
      let nextSetup: Setup = { ...setup };
      if (surface?.available) {
        const centred = await api.post<{ setup: Partial<Setup>; surface: ScanSurface }>(
          "/center-on-surface",
          { radius_mm: recipe.radius_mm, bead_diameter_mm: recipe.bead_diameter_mm,
            work_frame: setup.work_frame });
        nextSetup = { ...nextSetup, ...centred.setup };
        setSurface({ ...centred.surface, applied: true });
        if (connected && setup.print_tool) {
          const pose = await api.post<{ rpy_deg: number[] }>("/current-tcp", {
            print_tool: setup.print_tool,
            work_frame: centred.setup.work_frame || setup.work_frame,
          });
          nextSetup.orientation_rpy_deg = pose.rpy_deg as [number, number, number];
        }
      }
      const nextRecipe = { ...recipe, layer_count: Math.max(STACK_LAYERS, recipe.layer_count) };
      const generated = await api.post<Plan>("/generate",
        { recipe: nextRecipe, setup: nextSetup });
      setRecipe(nextRecipe); setSetup(nextSetup); setPlan(generated);
      setPreflight(null); setResult(null); setPaper(null); setOpenTake(null);
      const created = await api.post<{ session: MeasureSession }>(
        "/measure/session/new", { note: measureNote });
      setMeasureSession(created.session); setStepPin(null); setAxisKnown(false);
      setOffsetX(0); setOffsetY(0); setPhase("noise floor"); setMeasureLayer(1);
      setMessage(`Fresh session ${created.session.trial_id}${surface?.available
        ? ", aimed at the centre of the scanned surface" : ""}. Place ring 1 on the board `
        + "and press Characterize — its radius, bead and height are measured, not typed.");
      refreshStatus();
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
  const newMeasureSession = async () => {
    setBusy(true);
    try {
      const data = await api.post<{ session: MeasureSession }>("/measure/session/new", { note: measureNote });
      setMeasureSession(data.session); setPaper(null);
      setMessage(`New measure-only session ${data.session.trial_id}.`);
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
  const characterize = async () => {
    setBusy(true); setLogs([]);
    try {
      await api.post("/measure/characterize", {
        confirm_robot_motion: confirmMotion,
        collision_check_enabled: false,
      });
      setMessage("CHARACTERIZE started — collision validation OFF; the robot moves the camera over ring 1, measures it and returns.");
      refreshStatus();
    } catch (e: any) { setBusy(false); setMessage(e.message); }
  };
  const applyCharacterization = async () => {
    setBusy(true);
    try {
      const next = await api.post<Plan>("/measure/apply-characterization");
      setPlan(next); setRecipe(next.recipe); setSetup(next.setup);
      setPreflight(null); setResult(null); setSelectedLayer(1);
      // Applying binds the SESSION to this plan, and that binding is what marks
      // the step done and moves the run on. Without re-reading it the rail sat
      // on "Ring 1" for ever and the button just re-applied.
      await refreshMeasure();
      setMessage(`Recipe and placement set from the measured ring: r ${next.recipe.radius_mm} mm, `
        + `bead ${next.recipe.bead_diameter_mm} mm, layer ${next.recipe.layer_height_mm} mm. `
        + "Next: five noise-floor takes without touching the ring.");
      refreshStatus();
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
  const measure = async (take?: { layer: number; phase: string; offset: [number, number];
                                  /** Frames grabbed with the arm PARKED at the pose. */
                                  repeats?: number;
                                  /** Whole trips out and back, one frame each. */
                                  excursions?: number }) => {
    if (!plan) return;
    const layerIndex = take ? take.layer : measureLayer;
    const takePhase = take ? take.phase : phase;
    const [dx, dy] = take ? take.offset : [offsetX, offsetY];
    const repeats = Math.max(1, Math.min(10, take?.repeats ?? 1));
    const excursions = Math.max(1, Math.min(10, take?.excursions ?? 1));
    // One take's timeline per take: stale lines from the previous ring read as
    // this one's.
    setBusy(true); setLogs([]);
    try {
      const shifted = dx !== 0 || dy !== 0;
      await api.post("/measure/layer", {
        fingerprint: plan.fingerprint, layer_index: layerIndex,
        annotation: { introduced_offset_mm: shifted ? [dx, dy] : null,
                      phase: takePhase || undefined, note: measureNote },
        confirm_robot_motion: confirmMotion,
        collision_check_enabled: false, repeats, excursions,
        side_photo: sidePhoto, multiview,
      });
      // Echo the ground truth back: it is the number every detection-error
      // figure is measured against, and nothing else on screen repeats it.
      const batch = excursions > 1
        ? ` — ${excursions} trips out and back, unattended; do not touch the cell`
        : repeats > 1 ? ` — ${repeats} frames on one trip, arm parked` : "";
      setMessage(`MEASURE started — recording as ${annotationLabel(layerIndex, takePhase, [dx, dy])}`
        + `${batch}. Collision validation OFF; camera move only, no extrusion, no valve.`);
      refreshStatus();
    } catch (e: any) { setBusy(false); setMessage(e.message); }
  };
  const reprocessTake = async (take: MeasureTake) => {
    if (!measureSession) return;
    setBusy(true);
    setMessage(`Reprocessing layer ${take.layer_index} take ${take.take} from its archived `
      + "RGB-D frame — no robot motion.");
    try {
      const done = await api.post<{ metrics: { valid: boolean; rms_mm: number } }>(
        `/trials/${encodeURIComponent(measureSession.trial_id)}/layers/${take.layer_index}`
        + `/reprocess?take=${take.take}`);
      setMessage(done.metrics.valid
        ? `Reprocessed: VALID, RMS ${done.metrics.rms_mm.toFixed(2)} mm. The take is back in `
          + "the session and can serve as the floor for the layer above."
        : "Reprocessed, but the measurement is still invalid — see the take's figures.");
      refreshMeasure();
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
  /** Fetch the .docx and save it, refusing to write anything that is not one.
   *
   * An <a download> saves whatever the server returns, including an HTML error
   * page, under the name it asked for -- which is how a missing endpoint became
   * a 496-byte "Word file" with no error shown anywhere.
   */
  const downloadWordDraft = async () => {
    if (!measureSession) return;
    setBusy(true); setMessage("Building the Word draft from the archive…");
    try {
      const url = `/api/modules/extrusion/trials/`
        + `${encodeURIComponent(measureSession.trial_id)}/paper-draft.docx`;
      const response = await fetch(url);
      const kind = response.headers.get("content-type") || "";
      if (!response.ok || !kind.includes("wordprocessingml")) {
        let detail = `${response.status} ${response.statusText}`;
        try {
          const body = await response.json();
          if (body?.detail) detail = body.detail;
        } catch { /* not JSON: keep the status line */ }
        throw new Error(kind.includes("text/html")
          ? "The backend answered with the web app, which means it is running older "
            + "code without this endpoint. Restart Tasni and try again."
          : `The draft could not be built: ${detail}`);
      }
      const blob = await response.blob();
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${measureSession.trial_id}-paper-draft.docx`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(link.href);
      setMessage(`Word draft saved (${Math.round(blob.size / 1024)} KB) — method, condition `
        + "table, per-take table and figures, ready to paste into the manuscript.");
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
  const showPaper = async () => {
    if (!measureSession) return;
    try {
      const data = await api.get<{ markdown: string }>(`/trials/${measureSession.trial_id}/paper-summary`);
      setPaper(data.markdown);
    } catch (e: any) { setMessage(e.message); }
  };
  const cancel = async () => {
    setCancelling(true);
    try {
      await api.post("/cancel");
      setMessage("Cancellation requested. Waiting for the current RoboDK command to return, then completing the job exit sequence.");
      refreshStatus();
    } catch (e: any) {
      setCancelling(false); setMessage(e.message);
    }
  };

  const captureCurrentOrientation = async () => {
    if (!setup?.print_tool || !setup.work_frame) return;
    setBusy(true); setMessage("Capturing the current neutral TCP orientation…");
    try {
      const pose = await api.post<{ xyz_mm: number[]; rpy_deg: number[] }>("/current-tcp", {
        print_tool: setup.print_tool, work_frame: setup.work_frame,
      });
      setSetup({ ...setup, orientation_rpy_deg: pose.rpy_deg as [number, number, number] });
      invalidate(`Current neutral TCP orientation captured as ${pose.rpy_deg.map((v) => v.toFixed(1)).join(", ")}°. Cylinder center and build Z were not changed — generate coordinates next.`);
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
  const resetGenerated = async () => {
    setBusy(true);
    try {
      const response = await api.post<{ removed: string[] }>("/reset");
      setPlan(null); setPreflight(null); setResult(null); setConfirmLive(false);
      setQuickLayers([1]); setApproveRepresentativeLayers(false);
      setStatus((old) => old ? { ...old, fingerprint: null,
        geometry_preflight_passed: false, quick_sim_passed: false,
        quick_sim_layers: [], quick_sim_live_approved: false,
        dry_run_passed: false,
        live_print_enabled: false } : old);
      setMessage(`Generated plan reset; removed ${response.removed.length} temporary RoboDK artifact(s).`);
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
  const applyCorrection = async () => {
    if (!plan) return;
    setBusy(true);
    try {
      const corrected = await api.post<Plan>("/correction/apply", { fingerprint: plan.fingerprint });
      setPlan(corrected); setSelectedLayer(1); setQuickLayers([1]);
      setApproveRepresentativeLayers(false); setPreflight(null); setResult(null); setConfirmLive(false);
      setMessage("Corrected command generated. It has a new fingerprint and must pass preflight + visual simulation.");
      refreshStatus();
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };

  // Analytic browser-only draft for immediate visual feedback. This never enters
  // the robot workflow: only /generate can return the executable coordinates and
  // fingerprint that preflight/dry-run accept.
  const previewPlan = useMemo<Plan | null>(() => {
    if (!recipe || !setup) return null;
    const theta = Array.from({ length: recipe.points_per_circle + 1 }, (_, index) =>
      index * 2 * Math.PI / recipe.points_per_circle);
    const layers = Array.from({ length: recipe.layer_count }, (_, index): Layer => {
      const z = setup.build_plane_z_mm + recipe.bead_diameter_mm / 2 + index * recipe.layer_height_mm;
      return {
        layer_index: index + 1,
        nominal_z_mm: z,
        points: theta.map((angle) => ({
          x_mm: setup.center_x_mm + recipe.radius_mm * Math.cos(angle),
          y_mm: setup.center_y_mm + recipe.radius_mm * Math.sin(angle),
          z_mm: z,
        })),
      };
    });
    return {
      fingerprint: "DRAFT-NOT-GENERATED", recipe, setup, layers,
      total_path_length_mm: recipe.layer_count * 2 * Math.PI * recipe.radius_mm,
    };
  }, [recipe, setup]);
  const visualPlan = plan ?? previewPlan;
  const visualSelectedLayer = visualPlan
    ? Math.min(visualPlan.layers.length, Math.max(1, selectedLayer)) : 1;
  const layer = visualPlan?.layers[visualSelectedLayer - 1];
  const selectionsReady = Boolean(setup?.print_tool && setup?.work_frame && setup?.inspection_tool
    && (setup?.inspection_auto || setup?.inspection_target));
  const stationReady = Boolean(preflight?.station?.ready);
  const allQuickLayersSelected = Boolean(plan && quickLayers.length === plan.layers.length);
  const toggleQuickLayer = (layerIndex: number) => setQuickLayers((old) =>
    old.includes(layerIndex)
      ? old.filter((value) => value !== layerIndex)
      : [...old, layerIndex].sort((a, b) => a - b));
  const pct = progress.total ? Math.round(progress.step / progress.total * 100) : 0;
  const headlineMetrics = result?.kind === "cylinder_print" ? result.layers : [];

  // -- ring-stack protocol state ---------------------------------------------
  // Every count below is of VALID takes: an invalid one measured nothing, so it
  // cannot satisfy a step of the protocol.
  const takes = measureSession?.records ?? [];
  const validTakes = takes.filter((t) => t.valid);
  const offsetNorm = (t: MeasureTake) => {
    const o = t.annotation?.introduced_offset_mm;
    return o ? Math.hypot(o[0], o[1]) : 0;
  };
  const countPhase = (layerIndex: number, want: string) => validTakes.filter(
    (t) => t.layer_index === layerIndex && (t.annotation?.phase || "") === want
           && offsetNorm(t) === 0).length;
  // The bead each layer's newest VALID take measured. An invalid take measured
  // no footprint, so it must not contribute one.
  const measuredBeadByLayer = useMemo(() => {
    const widths: Record<string, number> = {};
    for (const take of measureSession?.records ?? []) {
      const width = take.geometry?.bead_width_mean_mm;
      if (!take.valid || !width) continue;
      widths[String(take.layer_index)] = width;
    }
    return widths;
  }, [measureSession]);
  const beadWidths = Object.values(measuredBeadByLayer);
  const beadSummary = beadWidths.length
    ? (beadWidths.reduce((sum, v) => sum + v, 0) / beadWidths.length).toFixed(1) : null;
  const applied = measureSession?.applied ?? null;
  const characterization = measureSession?.characterizations?.length
    ? measureSession.characterizations[measureSession.characterizations.length - 1] : null;
  const planIsApplied = !applied || plan?.fingerprint === applied.fingerprint;
  const floorReady = measureLayer <= 1
    || Boolean(measureSession?.tops?.[String(measureLayer - 1)]);

  // -- the run as a sequence -------------------------------------------------
  // One thing to do at a time, in the order the protocol needs it. Each step
  // owns the take it records, so the operator never sets layer/phase/offset by
  // hand and a mislabelled group cannot happen by forgetting a control.
  //
  // Two shapes of press, because repeatability has two sources that cost
  // differently to measure:
  //   * the NOISE FLOOR buys whole trips out and back (the arm's re-approach is
  //     the thing under test), five of them, unattended on one press;
  //   * every later condition buys frames with the arm PARKED -- the sensing
  //     spread alone, three frames on one trip, in seconds.
  // Read together they say how much of the deviation is the robot and how much
  // is the camera, which is the claim the noise floor exists to support.
  const noiseTakes = countPhase(1, "noise floor");
  const replacedTakes = countPhase(1, "re-placed");
  const topLayer = plan?.layers.length ?? 1;
  const offsetVector: [number, number] = offsetAxis === "X" ? [offsetMag, 0] : [0, offsetMag];
  // The two displacement conditions are "about 10" and "about 15" mm, not
  // exactly 10 and 15: the operator measures what they actually moved with a
  // steel rule and types that. So a take belongs to whichever condition it is
  // nearer -- 12 mm is the small one. Matching an exact millimetre (as this
  // did) left a step that instructed "12 mm scores as well as 10" unable to
  // ever count that take, and so unable to finish.
  const SHIFT_SPLIT_MM = 12.5;
  const countShift = (layerIndex: number, band: "small" | "large") => validTakes.filter(
    (t) => {
      const norm = offsetNorm(t);
      if (t.layer_index !== layerIndex || norm < 0.51) return false;
      return band === "small" ? norm < SHIFT_SPLIT_MM : norm >= SHIFT_SPLIT_MM;
    }).length;
  const motionBlocked = !plan ? "Generate the plan first."
    : !connected ? "Connect to RoboDK — the camera move is a real robot motion."
    : !planIsApplied ? "Press “Use this ring” so the session measures against the plan it applied."
    : !confirmMotion ? "Tick “Hands clear” — the robot moves the camera."
    : null;
  const stalled = Boolean(motionBlocked) || busy || Boolean(status?.running);
  const runStep = (over: Partial<RunStep> & { id: string; label: string; title: string;
                                              done: boolean }): RunStep => ({ ...over } as RunStep);
  /** A condition measured with the arm parked: one trip, the frames still owed. */
  const parkedStep = (over: {
    id: string; label: string; title: string; layer: number; phase: string;
    hands: string; note: string; offset?: [number, number];
  }): RunStep => {
    const offset = over.offset ?? ([0, 0] as [number, number]);
    const have = offset[0] || offset[1]
      ? countShift(over.layer, Math.hypot(...offset) < SHIFT_SPLIT_MM ? "small" : "large")
      : countPhase(over.layer, over.phase);
    const owed = Math.max(1, REPEATS - have);
    return runStep({
      id: over.id, label: over.label, title: over.title, layer: over.layer,
      hands: over.hands, note: over.note,
      done: have >= REPEATS, progress: { have, need: REPEATS },
      button: `Measure ${owed} frame${owed === 1 ? "" : "s"} — one trip, arm parked`,
      records: annotationLabel(over.layer, over.phase, offset),
      onRun: () => measure({ layer: over.layer, phase: over.phase, offset, repeats: owed }),
      moves: true, disabled: stalled, blocked: motionBlocked, capture: true,
    });
  };
  /** The top ring, PLACED off-centre: the controlled error the paper shows. */
  const displacedStep = (layer: number): RunStep => {
    const placed = countShift(layer, "small") + countShift(layer, "large");
    const owed = Math.max(1, REPEATS - placed);
    // Which way is +X has to be known before an offset can be typed with a sign.
    // One throwaway take answers it, and it is excluded from the pairing.
    const needsAxis = !axisKnown;
    return runStep({
      id: `displaced${layer}`, label: `Ring ${layer} offset`, layer,
      title: needsAxis ? "Find out which way is +X"
        : `Ring ${layer} placed deliberately off-centre`,
      hands: needsAxis
        ? `Place ring ${layer} on top, pushed roughly 10 mm off-centre along one board `
          + "edge, and measure once. The Offset column below then tells you which axis "
          + "moved and in which direction."
        : `Place ring ${layer} on top of the stack, deliberately off-centre along a board `
          + `edge. Measure how far it sits from the ring beneath it with a steel rule and `
          + `type that — anything from about 5 to 25 mm works, and 12 mm scores as well `
          + `as 10. Then keep clear.`,
      done: !needsAxis && placed >= REPEATS,
      progress: needsAxis ? undefined : { have: placed, need: REPEATS },
      button: needsAxis ? "Take one throwaway measurement"
        : `Measure ${owed} frame${owed === 1 ? "" : "s"} — one trip, arm parked`,
      records: needsAxis ? undefined
        : annotationLabel(layer, "top ring shifted", offsetVector),
      onRun: () => measure(needsAxis
        ? { layer, phase: "axis check", offset: [0, 0] }
        : { layer, phase: "top ring shifted", offset: offsetVector, repeats: owed }),
      moves: true, disabled: stalled, blocked: motionBlocked, capture: true,
      offsetInput: !needsAxis, axisAck: true,
      note: "This ring arrives already displaced, so it has no undisplaced take of its own "
        + `to be scored against. It is paired against the measured centre of ring ${layer - 1} `
        + "instead — which is what the rule measured anyway: how far this ring sits from the "
        + "one it was stacked on.",
    });
  };

  const RUN: RunStep[] = [
    runStep({
      id: "fresh", label: "Fresh ring", title: "Start a fresh ring",
      hands: "Nothing to place yet. Clear the board and have a scanned surface applied "
        + "(Setup above); this press only decides where the camera looks.",
      done: Boolean(plan) && (plan?.layers.length ?? 0) >= STACK_LAYERS && Boolean(measureSession),
      button: "Aim at the table & open a session",
      onRun: freshStart,
      disabled: !recipe || !selectionsReady || busy || Boolean(status?.running),
      blocked: !selectionsReady
        ? "Open Setup and choose the work frame, print tool and camera tool." : null,
      note: "The ring's radius, bead and height are MEASURED in the next step, never typed: "
        + `this only aims the search (within 150 mm of the centre) and fixes ${STACK_LAYERS} `
        + "layers, because Apply never changes the layer count later. Three go on true; the "
        + "fourth goes on deliberately off-centre.",
    }),
    runStep({
      id: "ring1", label: "Ring 1", title: characterization
        ? "Use this ring as the recipe" : "Characterize ring 1",
      hands: characterization
        ? "Leave the ring exactly where it is."
        : "Place ring 1 flat on the board, within about 50 mm of the table centre.",
      done: Boolean(applied) && planIsApplied,
      button: characterization ? "Use this ring" : "Characterize ring 1",
      onRun: characterization ? applyCharacterization : characterize,
      moves: !characterization,
      disabled: characterization
        ? (busy || Boolean(status?.running))
        : (!plan || !connected || !confirmMotion || busy || Boolean(status?.running)),
      blocked: characterization ? null
        : (!connected ? "Connect to RoboDK first." : !confirmMotion ? "Tick “Hands clear”." : null),
      note: characterization
        ? `Measured r ${characterization.radius_mm.toFixed(1)} mm · bead `
          + `${characterization.bead_width_mm.toFixed(1)} mm · height `
          + `${characterization.top_z_min_mm.toFixed(1)}–${characterization.top_z_max_mm.toFixed(1)} mm. `
          + "This sets the recipe and the cylinder centre from the physical ring."
        : "The robot moves the camera over the ring, takes one frame and returns.",
    }),
    runStep({
      id: "noise", layer: 1, label: "Noise floor",
      title: `Noise floor — ${NOISE_TRIPS} trips, unattended`,
      hands: "Hands off, and stay out of the cell until it stops. One press sends the arm out "
        + "and back five times on its own; nothing is touched between trips.",
      done: noiseTakes >= NOISE_TRIPS, progress: { have: noiseTakes, need: NOISE_TRIPS },
      button: `Run ${Math.max(1, NOISE_TRIPS - noiseTakes)} trip`
        + `${NOISE_TRIPS - noiseTakes === 1 ? "" : "s"} — unattended`,
      records: annotationLabel(1, "noise floor", [0, 0]),
      onRun: () => measure({ layer: 1, phase: "noise floor", offset: [0, 0],
                             excursions: Math.max(1, NOISE_TRIPS - noiseTakes) }),
      moves: true, disabled: stalled, blocked: motionBlocked, capture: true,
      note: "The only step that re-approaches the ring for every take, so it is the only one "
        + "whose spread contains the robot as well as the camera. Every later condition is "
        + "measured with the arm parked and read against this. A trip that fails keeps the "
        + "takes before it — press again to finish the set.",
    }),
    runStep({
      id: "replace", layer: 1, label: "Re-place", title: "Placement repeatability — three takes",
      hands: "Lift ring 1 off the board and set it back down as accurately as you can. Once "
        + "per press — the hand has to move between these, so they cannot be batched.",
      done: replacedTakes >= REPEATS, progress: { have: replacedTakes, need: REPEATS },
      button: `Measure take ${Math.min(replacedTakes + 1, REPEATS)} of ${REPEATS}`,
      records: annotationLabel(1, "re-placed", [0, 0]),
      onRun: () => measure({ layer: 1, phase: "re-placed", offset: [0, 0] }),
      moves: true, disabled: stalled, blocked: motionBlocked, capture: true,
      note: "How repeatably a hand places a ring — separate from how well the chain sees it.",
    }),
    parkedStep({
      id: "ring2", label: "Ring 2", layer: 2, phase: "stacked true",
      title: "Ring 2 on the stack — three frames",
      hands: "Place ring 2 on top of ring 1, as true as you can, then keep clear.",
      note: "Check the first take says VALID before continuing: where ring 2 sits low there is "
        + "under a millimetre of floor margin and its low stretches can be clipped.",
    }),
    parkedStep({
      id: "ring3", label: "Ring 3", layer: 3, phase: "stacked true",
      title: "Ring 3 on the stack — three frames",
      hands: "Place ring 3 on top, as true as you can, then keep clear.",
      note: "The camera climbs with the stack: every layer is measured from 300 mm above its "
        + "own top. Three rings placed as true as a hand can is the baseline the deliberate "
        + "offset is read against.",
    }),
    displacedStep(topLayer),
    runStep({
      id: "summary", label: "Summary", title: "Take the numbers",
      hands: "Nothing to touch in the cell.",
      done: Boolean(paper),
      button: "Paper summary",
      onRun: showPaper,
      disabled: !measureSession,
      note: "Grouped per layer and phase, with the detection error against what you typed.",
    }),
  ];
  const autoIndex = RUN.findIndex((step) => !step.done);
  // A pin is honoured whatever the step's state. Refusing it for a COMPLETED
  // step made every green chip in the rail unclickable -- and on a resumed
  // session the first steps are green, so there was no way back to "start a
  // fresh ring" at all.
  const resumeIndex = autoIndex < 0 ? RUN.length - 1 : autoIndex;
  const activeIndex = (stepPin !== null && RUN[stepPin]) ? stepPin : resumeIndex;
  const pinnedAway = stepPin !== null && stepPin !== resumeIndex;
  const active = RUN[activeIndex];
  // Finish a step and the run moves on by itself, pin or no pin.
  //
  // The pin exists so the operator can go back and look at a step; it should
  // not also mean "and stay here forever". Without this, working from a pinned
  // step (which is how you resume after a restart, or redo a condition) left
  // the panel sitting on a finished step showing "3 of 3 done" with a button
  // that would only over-measure it. Only a pin that was placed on UNFINISHED
  // work releases -- clicking back to an already-green step keeps you there,
  // which is the whole point of clicking it.
  const pinWasUnfinished = useRef(false);
  useEffect(() => {
    pinWasUnfinished.current = stepPin !== null && !RUN[stepPin]?.done;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stepPin]);
  const pinnedDone = stepPin !== null && Boolean(RUN[stepPin]?.done);
  useEffect(() => {
    if (pinnedDone && pinWasUnfinished.current) { pinWasUnfinished.current = false; setStepPin(null); }
  }, [pinnedDone]);
  // The floor is the layer BELOW the one this step measures -- not the one the
  // manual selector happens to be showing.
  const stepFloorReady = !active.layer || active.layer <= 1
    || Boolean(measureSession?.tops?.[String(active.layer - 1)]);

  return <div>
    <div className="page-head">
      <div>
        <h1 className="page-title">{mode === "measure" ? "Ring stack" : "Cylinder print"}</h1>
        <p className="page-sub">{mode === "measure"
          ? "Hand-placed rings, measured one layer at a time. The camera moves; nothing is extruded."
          : "Top-down toolpath review → complete RoboDK dry run → layer print → one RGB-D inspection → archive."}</p>
      </div>
      <div className="mode-switch" role="group" aria-label="Workflow">
        <button type="button" className={mode === "measure" ? "on" : ""}
                aria-pressed={mode === "measure"}
                onClick={() => setMode("measure")}>Measure rings</button>
        <button type="button" className={mode === "print" ? "on" : ""}
                aria-pressed={mode === "print"}
                onClick={() => setMode("print")}>Print cylinder</button>
      </div>
    </div>

    <div className={`card conn-banner ${connected ? "ready" : ""}`}>
      <div className="conn-row"><span className="conn-label">RoboDK station</span>
        <span className="status-line">{connected ? "connected · station items discovered" : "not connected"}</span>
        <button className="secondary" disabled={busy || status?.running} onClick={connect}>{connected ? "Refresh items" : "Connect"}</button>
      </div>
    </div>

    {/* One status line for the page: it used to live inside the print-only card,
        where a measuring run could not see its own errors. */}
    <div className="run-status">
      <p className="status-line">{message}</p>
      {(busy || status?.running) && <><div className="progress"><div style={{ width: `${pct}%` }} /></div>
        <div className="status-line">{progress.message}</div></>}
    </div>

    <details className="setup-fold" open={mode === "print"}>
      <summary>Setup — station, recipe and plan
        <span className="hint">{plan
          ? `r ${plan.recipe.radius_mm} · bead ${plan.recipe.bead_diameter_mm} · ${plan.layers.length} layer${plan.layers.length === 1 ? "" : "s"} · ${plan.setup.work_frame}`
          : "no plan generated yet"}</span></summary>
    <div className="extrusion-layout">
      <div className="card">
        <h2>Station & motion</h2>
        <div className="station-select-grid">
          <label>Print tool<select value={setup?.print_tool || ""} disabled={!options} onChange={(e) => updateSetup("print_tool", e.target.value)}>
            <option value="">Select tool…</option>{options?.tools.map((v) => <option key={v}>{v}</option>)}</select></label>
          <label>Work frame<select value={setup?.work_frame || ""} disabled={!options} onChange={(e) => updateSetup("work_frame", e.target.value)}>
            <option value="">Select frame…</option>{options?.frames.map((v) => <option key={v}>{v}</option>)}</select></label>
          <label>Inspection tool<select value={setup?.inspection_tool || ""} disabled={!options} onChange={(e) => updateSetup("inspection_tool", e.target.value)}>
            <option value="">Select camera tool…</option>{options?.tools.map((v) => <option key={v}>{v}</option>)}</select></label>
          <label>Inspection target<select value={setup?.inspection_target || ""}
            disabled={!options || Boolean(setup?.inspection_auto)}
            onChange={(e) => updateSetup("inspection_target", e.target.value)}>
            <option value="">{setup?.inspection_auto ? "derived per layer" : "Select target…"}</option>
            {options?.targets.map((v) => <option key={v}>{v}</option>)}</select></label>
        </div>
        <label className="live-confirm"><input type="checkbox" checked={Boolean(setup?.inspection_auto)}
          onChange={(e) => updateSetup("inspection_auto", e.target.checked)} />
          Derive the inspection pose from the cylinder (recommended) — square to the build
          plane, centred on the cylinder axis, at the distance this camera needs to frame it.
        </label>
        {setup && <>
          <div className="motion-number-grid">
            <label>Center X (mm)<input type="number" value={setup.center_x_mm} onChange={(e) => updateSetup("center_x_mm", Number(e.target.value))} /></label>
            <label>Center Y (mm)<input type="number" value={setup.center_y_mm} onChange={(e) => updateSetup("center_y_mm", Number(e.target.value))} /></label>
            <label>Build plane Z (mm)<input type="number" value={setup.build_plane_z_mm} onChange={(e) => updateSetup("build_plane_z_mm", Number(e.target.value))} /></label>
            {["A / roll", "B / pitch", "C / yaw"].map((label, i) => <label key={label}>{label} (°)<input type="number" value={setup.orientation_rpy_deg[i]} onChange={(e) => updateOrientation(i, Number(e.target.value))} /></label>)}
            <label>Maximum wrist rotation — axes 4 &amp; 6 (°)<input type="number" min="1" max="180"
              value={setup.maximum_tool_axis_spin_deg}
              onChange={(e) => updateSetup("maximum_tool_axis_spin_deg", Number(e.target.value))} /></label>
            <label>Approach (mm)<input type="number" min="1" value={setup.approach_clearance_mm} onChange={(e) => updateSetup("approach_clearance_mm", Number(e.target.value))} /></label>
            <label>Retreat (mm)<input type="number" min="1" value={setup.retreat_clearance_mm} onChange={(e) => updateSetup("retreat_clearance_mm", Number(e.target.value))} /></label>
          </div>
          <div className="btn-row">
            <button className="secondary" disabled={!surface?.available || !connected || !setup.print_tool || busy || status?.running}
              title={`Put the cylinder on the middle of the platform, expressed in ${setup.work_frame || "the selected work frame"}`}
              onClick={centerOnPlatform}>Center on platform</button>
            <button className="secondary" disabled={!connected || busy || !setup.print_tool || !setup.work_frame}
              onClick={captureCurrentOrientation}>Capture neutral orientation only</button>
            <button className="secondary" disabled={!connected || busy || !setup.print_tool || !setup.work_frame}
              onClick={seedFromCurrentTcp}>Seed path start from current TCP</button>
          </div>
          <div className={`surface-row ${surface?.available ? "ready" : "warn-text"}`}>
            <span className="conn-label">Platform</span>
            <span className="status-line">{surface?.available
              ? `middle at (${surface.center_mm?.[0].toFixed(1)}, ${surface.center_mm?.[1].toFixed(1)}) in ${surface.frame} · ${surface.size_mm ? `${surface.size_mm[0].toFixed(0)} × ${surface.size_mm[1].toFixed(0)} mm · ` : "size unknown · "}${PLATFORM_SOURCE[surface.source ?? ""] ?? "unknown source"}${surface.run_id ? ` ${surface.run_id}` : ""}`
              : surface?.note || `No platform is known in ${setup.work_frame || "the selected work frame"}.`}</span>
            <button className="secondary" disabled={busy}
              onClick={() => refreshSurface(setup.work_frame)}>Re-check</button>
          </div>
          {surface?.available && !surface.extents_known && <div className="hint warn-text">
            The middle is known but the platform&apos;s size is not, in this frame — the wall
            will NOT be checked for overhang. Select <code>Tasni Work Frame</code> for that check.</div>}
          <div className="hint">Preferred flow: scan the platform, insert it, then <b>Center on platform</b> — the cylinder lands in the middle of the measured rectangle and automatically captures the current neutral TCP orientation. The middle is read from RoboDK (the <code>Tasni Work Center</code> frame, else the <code>Tasni Work Surface</code> object) and only then from the applied scan run, so it survives clearing the runs folder; the work-frame dropdown chooses which coordinate system it is expressed in, not where the platform is. <b>Capture neutral orientation only</b> refreshes orientation without changing the cylinder center or Z. Center, build-plane Z, and RoboDK XYZRPW are expressed in the selected work frame. The generated IK keeps the neutral front/elbow/wrist configuration, limits both axes 4 and 6 to ±90°, and samples the interpolated joint path to block hidden wrist flips before playback. These exact values are fingerprinted and simulated.</div>
          {setup.scan_run_id
            ? <div className="hint">Placed on scan run <code>{setup.scan_run_id}</code>. Re-scanning the surface invalidates this placement, and preflight rejects a wall that overhangs the measured rectangle.</div>
            : surface?.available && surface.center_mm
              && (Math.hypot(setup.center_x_mm - surface.center_mm[0],
                             setup.center_y_mm - surface.center_mm[1]) > 1) &&
              <div className="hint warn-text">This path is placed at ({setup.center_x_mm.toFixed(1)}, {setup.center_y_mm.toFixed(1)}), but the middle of the platform in <code>{setup.work_frame}</code> is ({surface.center_mm[0].toFixed(1)}, {surface.center_mm[1].toFixed(1)}).</div>}
          {setup.work_frame === "World" && <div className="hint warn-text">World is allowed, but every X/Y/Z value is then a station-world coordinate. Use the current-TCP seed button unless world zero is deliberately your build plane.</div>}
        </>}
      </div>

      <div className="card">
        <h2>Recipe</h2>
        {recipe && recipeFields.map((field) => <label className="recipe-field" key={field.key}>
          <span>{field.label}</span><input type="range" min={field.min} max={field.max} step={field.step}
            value={recipe[field.key] as number} onChange={(e) => updateRecipe(field.key, Number(e.target.value))} />
          <input className="recipe-number" type="number" min={field.min} max={field.max} step={field.step}
            value={recipe[field.key] as number} onChange={(e) => updateRecipe(field.key, Number(e.target.value))} /><small>{field.unit}</small>
        </label>)}
        {recipe && <label className="correction-toggle"><input type="checkbox" checked={recipe.correction_enabled}
          onChange={(e) => updateRecipe("correction_enabled", e.target.checked)} />Calculate bounded compensation after valid measurements</label>}
        <div className="hint warn-text">Extrusion rate is recorded only; it is not mapped to the valve outputs or an unverified analog controller command.</div>
        <div className="hint">RoboDK receives one disposable XYZ+IJK curve per layer. Curve samples control geometric resolution; process/travel speeds and blending are native Curve Follow settings.</div>
      </div>
    </div>

    <div className="card birdseye-card">
      <div className="birdseye-head"><div><h2>Live bird’s-eye draft</h2><p>Sliders update this analytic preview immediately. It is not an executable robot plan until coordinates are generated below.</p></div>
        <div className="birdseye-tools">
          <label className="bead-toggle"><input type="checkbox" checked={showBead}
            onChange={(e) => setShowBead(e.target.checked)} />
            Bead{recipe ? ` · commanded Ø ${recipe.bead_diameter_mm} mm` : ""}
            {beadSummary && <span className="measured-bead"> · measured Ø {beadSummary} mm</span>}
          </label>
          {layer && <div className="layer-readout">LAYER {layer.layer_index}<b>Z {layer.nominal_z_mm.toFixed(2)} mm</b></div>}
        </div></div>
      {visualPlan && layer ? <div className="birdseye-layout">
        <BirdseyeStack plan={visualPlan} selectedLayer={visualSelectedLayer} onSelect={setSelectedLayer}
                       measured={measureSession?.tops} showBead={showBead}
                       measuredBead={measuredBeadByLayer} />
        <div className="layer-rail">{visualPlan.layers.map((item) => <button key={item.layer_index}
          className={`layer-tile ${item.layer_index === visualSelectedLayer ? "selected" : ""}`}
          onClick={() => setSelectedLayer(item.layer_index)}>
          <i className="layer-swatch" style={{ opacity: .42 + item.layer_index / visualPlan.layers.length * .58 }} />
          <span>Layer {item.layer_index}</span><small>Z {item.nominal_z_mm.toFixed(2)} mm</small>
        </button>)}</div>
      </div> : <div className="empty cylinder-empty">Loading recipe preview…</div>}
      {visualPlan && <div className="kv preview-kv">
        <span className="k">Visualization state</span><span className={`v preview-state ${plan ? "generated" : "draft"}`}>{plan ? "GENERATED · FINGERPRINTED" : "LIVE DRAFT · NOT ROBOT-READY"}</span>
        <span className="k">Estimated circular path</span><span className="v">{visualPlan.total_path_length_mm.toFixed(1)} mm</span>
        <span className="k">Preview samples / layer</span><span className="v">{visualPlan.layers[0].points.length}</span>
        {plan && <><span className="k">Plan fingerprint</span><span className="v">{plan.fingerprint.slice(0, 16)}</span></>}
      </div>}
      <div className="preview-generation">
        <div><b>Generate robot coordinates</b><span>This freezes the current recipe and station selections into an exact fingerprint. Any later input change invalidates it.</span></div>
        <div className="btn-row"><button disabled={!recipe || !selectionsReady || busy || status?.running} onClick={() => generate()}>Generate coordinates & fingerprint</button>
          <button className="secondary" disabled={!plan || busy || status?.running} onClick={runPreflight}>Geometry & station preflight</button>
          <button className="secondary" disabled={!connected || busy || status?.running} onClick={resetGenerated}>Reset / clean RoboDK path</button></div>
      </div>
    </div>
    </details>

    {mode === "print" && <div className={`card extrusion-safety ${status?.live_print_enabled ? "ready" : "locked"}`}>
      <h2>Safety workflow</h2>
      <div className="workflow-steps"><span className={plan ? "done" : ""}>1 Generate</span>
        <span className={preflight?.all_ok && stationReady ? "done" : ""}>2 Preflight</span>
        <span className={status?.quick_sim_passed ? "done" : ""}>3a Visual simulation</span>
        <span className={status?.dry_run_passed ? "done" : ""}>3b Collision-validated dry run</span>
        <span className={result?.kind === "cylinder_print" ? "done" : ""}>4 Print & record</span></div>
      {config && <div className="io-note">Valve: <code>{config.integration.valve_outputs.join(" + ")}</code> via {config.integration.air_on_program}/{config.integration.air_off_program}. Hardware approval: <b>{config.integration.hardware_io_test_approved ? "APPROVED" : "NOT APPROVED"}</b>.</div>}
      {preflight?.surface && <div className={`io-note ${preflight.surface.ok ? "" : "warn-text"}`}>
        Placement: <b>{preflight.surface.placement === "scan_surface" ? "scanned surface" : "manual"}</b>
        {preflight.surface.fit?.checked && ` · clearance to surface edge ${preflight.surface.fit.minimum_margin_mm.toFixed(1)} mm`}
        {preflight.surface.problem && ` — ${preflight.surface.problem}`}
        {preflight.surface.advisory && ` — ${preflight.surface.advisory}`}
      </div>}
      {inspection?.auto && <div className={`io-note ${inspection.ok ? "" : "warn-text"}`}>
        Inspection pose: <b>{inspection.standoff_mm.toFixed(0)} mm</b> above each layer top,
        square to the build plane and centred on the cylinder axis in <code>{inspection.work_frame}</code>.
        A {inspection.object_diameter_mm.toFixed(0)} mm wall then fills{" "}
        <b>{(inspection.framing.fill_fraction.height * 100).toFixed(0)}%</b> of the frame height
        {inspection.framing.clamped_to === "near"
          ? ` (it would frame at ${inspection.framing.d_fit_mm.toFixed(0)} mm, inside the camera's ${inspection.framing.near_mm.toFixed(0)} mm near limit, so the standoff is held there)`
          : inspection.framing.clamped_to === "far" ? " — too large to frame inside the accurate depth band" : ""}.
        Camera Z rises {inspection.layers[0]?.camera_z_mm.toFixed(0)} → {inspection.layers[inspection.layers.length - 1]?.camera_z_mm.toFixed(0)} mm over {inspection.layers.length} layer(s).
        {inspection.warnings.map((w) => ` ${w}`)}
      </div>}
      {preflight?.station?.reachability && <div className="io-note">Path IK sample: <b>{preflight.station.reachability.reachable_count}/{preflight.station.reachability.sample_count}</b> reachable in <code>{preflight.station.reachability.frame}</code>. Full curve generation and collision validation still run in the dry run.</div>}
      {plan && <div className="io-note">
        <b>Quick-simulation layers:</b>
        <div className="btn-row">
          <button className="secondary" disabled={busy || status?.running}
            onClick={() => setQuickLayers([visualSelectedLayer])}>Current layer {visualSelectedLayer}</button>
          <button className="secondary" disabled={busy || status?.running}
            onClick={() => setQuickLayers(plan.layers.map((item) => item.layer_index))}>All layers</button>
          {plan.layers.map((item) => <button key={item.layer_index}
            className={quickLayers.includes(item.layer_index) ? "" : "secondary"}
            disabled={busy || status?.running} onClick={() => toggleQuickLayer(item.layer_index)}>
            L{item.layer_index}</button>)}
        </div>
        <label className="live-confirm"><input type="checkbox"
          checked={approveRepresentativeLayers} disabled={allQuickLayersSelected || busy || status?.running}
          onChange={(e) => setApproveRepresentativeLayers(e.target.checked)} />
          Treat the selected layer(s) as representative and approve all remaining layers for the live run.
        </label>
        <div className="hint">Without that approval, Live Print unlocks only after every layer has been visually simulated. Layer coverage accumulates for this exact fingerprint.</div>
      </div>}
      {status?.quick_sim_passed && <div className={`io-note ${status.quick_sim_live_approved ? "" : "warn-text"}`}>
        Visual simulation passed for layer(s) <b>{status.quick_sim_layers.join(", ")}</b> with collision checking disabled. {status.quick_sim_live_approved
          ? "The complete live plan is approved for operator confirmation."
          : "Simulate the remaining layers or approve the selected layers as representative."}
      </div>}
      <div className="btn-row"><button className="secondary" disabled={!plan || quickLayers.length === 0 || !preflight?.all_ok || !stationReady || busy || status?.running} onClick={() => startJob("quick-sim")}>Quick visual simulation — collisions OFF</button>
        <button disabled={!plan || !preflight?.all_ok || !stationReady || busy || status?.running} onClick={() => startJob("dry-run")}>Complete validated dry run — collisions ON</button>
        {(busy || status?.running) && <button className="secondary" onClick={cancel}
          disabled={cancelling}>{cancelling ? "Cancellation requested…" : "Cancel safely"}</button>}</div>
      <label className="live-confirm"><input type="checkbox" checked={liveCollisionCheck}
        onChange={(e) => setLiveCollisionCheck(e.target.checked)}
        disabled={!status?.quick_sim_live_approved || status?.running} />
        Validate collisions before each physical layer (slow: RoboDK re-checks the whole station per layer).
      </label>
      {!liveCollisionCheck && status?.quick_sim_live_approved && <div className="io-note warn-text">
        Live collision validation is OFF. The physical run relies on what you saw in the quick visual
        simulation, preflight IK sampling, and your cell-clearance confirmation.
        {status?.dry_run_passed ? " The dry run has also passed for this toolpath." : ""}
      </div>}
      <label className="live-confirm"><input type="checkbox" checked={keepArtifacts}
        onChange={(e) => setKeepArtifacts(e.target.checked)} disabled={status?.running} />
        Keep the generated RoboDK programs and targets after the run (for inspecting what was commanded).
      </label>
      {keepArtifacts && <div className="io-note">
        The curve, machining project, layer programs and inspection targets will be left in the station.
        Reset / clean RoboDK path removes them.
      </div>}
      <label className="live-confirm"><input type="checkbox" checked={confirmLive} onChange={(e) => setConfirmLive(e.target.checked)} disabled={!status?.live_print_enabled || status?.running} />
        I confirm the selected tool/frame/orientation, cell clearance, material system, and live extrusion run.</label>
      <button className="live-print-btn" disabled={!plan || !status?.live_print_enabled || !confirmLive || busy || status?.running} onClick={() => startJob("print")}>Print & record — LIVE ROBOT · collisions {liveCollisionCheck ? "ON" : "OFF"}</button>
      <div className="log" ref={logRef}>{logs.length ? logs.map((line, i) => <div key={i} className={line.startsWith("ERROR") ? "err" : ""}>{line}</div>) : "Job timeline will appear here."}</div>
    </div>}

    {mode === "measure" && <div className="card ring-card">
      <h2 className="visually-hidden">Ring stack — measure only</h2>
      <div className="ring-head">
        <div className="step-rail" role="list">
          {RUN.map((step, index) => <button key={step.id} type="button" role="listitem"
            className={`step-chip${step.done ? " done" : ""}${index === activeIndex ? " current" : ""}`}
            aria-current={index === activeIndex ? "step" : undefined}
            onClick={() => setStepPin(index === activeIndex ? null : index)}>
            <span className="dot">{step.done ? "✓" : index + 1}</span>{step.label}
          </button>)}
        </div>
        <button type="button" className="secondary guide-btn"
                onClick={() => setGuideOpen(true)}>Run guide</button>
      </div>
      {pinnedAway && <p className="hint">Viewing a step you chose. The run itself is at
        <b> {RUN[resumeIndex].label}</b>.{" "}
        <button type="button" className="linkish"
                onClick={() => setStepPin(null)}>Back to the run</button></p>}

      {applied && !planIsApplied && <div className="io-note warn-text">
        <b>This is not the plan the session applied</b> (ring characterized at r {applied.recipe.radius_mm} mm,
        centre {applied.setup.center_x_mm.toFixed(1)}, {applied.setup.center_y_mm.toFixed(1)}).
        Press <b>Use this ring</b> again before measuring.
      </div>}
      {plan?.restored_from && <p className="hint">Resumed session
        <code> {plan.restored_from} </code>with its plan — the first steps show as done
        because this ring was already characterized and applied. To measure a different ring,
        open <b>Fresh ring</b> in the rail (or <b>Start over with a new ring</b> below).</p>}

      {/* Exactly one thing to do, and the press that does it. Everything the step
          records is set by the step itself, so no control can be left on a stale
          value from the take before. */}
      <div className={`step-panel${active.moves ? " moves" : ""}`}>
        <div className="step-what">
          <h3>{active.title}</h3>
          {active.hands && <p className="hands">{active.hands}</p>}
        </div>

        {active.offsetInput && <div className="step-inputs">
          <label>Moved along <select value={offsetAxis}
            onChange={(e) => setOffsetAxis(e.target.value === "Y" ? "Y" : "X")}>
            <option value="X">X</option><option value="Y">Y</option>
          </select></label>
          <label>by <input type="number" step={1} value={offsetMag}
            onChange={(e) => setOffsetMag(Number(e.target.value))} /> mm</label>
          <span className="hint">negative if it moved the other way</span>
        </div>}

        {active.records && <p className="records"><span className="k">Records</span>
          <b>{active.records}</b></p>}

        {/* Independent of each other: neither disables nor implies the other, so
            merged-with-no-photo, single-view-with-photo, both and neither all stay
            reachable. Only shown on steps that actually call /measure/layer —
            "Characterize ring 1" accepts multiview for API symmetry but does not
            yet wire it to a star capture, so the toggle would promise nothing there. */}
        {active.capture && <div className="step-go capture-toggles">
          <label className="go-confirm">
            <input type="checkbox" checked={multiview}
                   onChange={(e) => setMultiview(e.target.checked)} />
            Multi-view capture — 4 trips instead of 1 (~15 s of arm time)</label>
          <label className="go-confirm">
            <input type="checkbox" checked={sidePhoto}
                   onChange={(e) => setSidePhoto(e.target.checked)} />
            Side photo — one extra excursion after the capture</label>
        </div>}

        <div className="step-go">
          {active.moves && <label className="go-confirm">
            <input type="checkbox" checked={confirmMotion}
                   onChange={(e) => setConfirmMotion(e.target.checked)} />
            Hands clear — the robot may move the camera</label>}
          <button className="go-btn" disabled={active.disabled || !stepFloorReady}
                  onClick={active.onRun}>{active.button}</button>
          {(busy || status?.running) && <button className="secondary" onClick={cancel}>Cancel</button>}
          {active.progress && <span className="step-progress">
            {active.progress.have} of {active.progress.need} done</span>}
        </div>

        {active.blocked && <p className="hint warn-text">{active.blocked}</p>}
        {!stepFloorReady && <p className="hint warn-text">
          Measure layer {(active.layer ?? 1) - 1} first — it is the measurement floor for
          layer {active.layer}.</p>}
        {active.note && <p className="hint">{active.note}</p>}
        {active.axisAck && <label className="axis-ack">
          <input type="checkbox" checked={axisKnown}
                 onChange={(e) => setAxisKnown(e.target.checked)} />
          I know which axis and sign the ring moves along</label>}
      </div>

      <div className="ring-foot">
        <span className="hint">{measureSession
          ? `session ${measureSession.trial_id}` : "no session yet"}</span>
        {plan && <span className="hint">r {plan.recipe.radius_mm} · bead {plan.recipe.bead_diameter_mm}
          {" "}· {plan.layers.length} layer{plan.layers.length === 1 ? "" : "s"}</span>}
        <span className="spacer" />
        <button type="button" className="linkish" disabled={busy || status?.running}
                onClick={freshStart}>Start over with a new ring</button>
        <button type="button" className="linkish"
                onClick={() => setManualOpen(!manualOpen)}>{manualOpen ? "Hide" : "Manual"} controls</button>
      </div>

      {manualOpen && <div className="manual-panel">
        <p className="hint">Off-script takes: choose the layer, phase and offset yourself. The run
          above keeps counting whatever matches its steps.</p>
        <div className="ring-row">
          <input className="note-input" placeholder="note" value={measureNote}
                 onChange={(e) => setMeasureNote(e.target.value)} />
          <button className="secondary" disabled={!plan || busy || status?.running}
                  onClick={newMeasureSession}>New session</button>
          <span className="spacer" />
          <label>Layer <select value={measureLayer}
            onChange={(e) => setMeasureLayer(Number(e.target.value))}>
            {(plan?.layers ?? []).map((l) => <option key={l.layer_index} value={l.layer_index}>{l.layer_index}</option>)}
          </select></label>
          <label>Phase <select value={phase} onChange={(e) => setPhase(e.target.value)}>
            <option value="">(none)</option>
            {PHASES.map((p) => <option key={p.value} value={p.value}>{p.label}</option>)}
          </select></label>
          <label>X <input type="number" step={1} value={offsetX}
            onChange={(e) => setOffsetX(Number(e.target.value))} /></label>
          <label>Y <input type="number" step={1} value={offsetY}
            onChange={(e) => setOffsetY(Number(e.target.value))} /></label>
          <button className="secondary" disabled={!plan || !connected || !confirmMotion || busy
                    || status?.running || !planIsApplied || !floorReady}
                  onClick={() => measure()}>Measure — {annotationLabel()}</button>
        </div>
      </div>}

      {logs.length > 0 && <div className="log measure-log">{logs.slice(-6).map((line, i) =>
        <div key={i} className={line.startsWith("ERROR") ? "err" : ""}>{line}</div>)}</div>}
      {takes.length ? <div className="take-head">
        <span className="hint">{takes.length} take{takes.length === 1 ? "" : "s"} ·
          {" "}{validTakes.length} valid · click a row for its figures</span>
        <button type="button" className="secondary" onClick={() => setAllColumns(!allColumns)}>
          {allColumns ? "Fewer columns" : "All columns"}</button>
      </div> : null}
      {takes.length ? <table className="metrics">
        <thead><tr><th>Layer</th><th>Take</th><th>Phase</th><th>Introduced</th>
          <th>Measured |d|</th><th>Detection error</th>
          {allColumns && <><th>dx / dy</th><th>Mean |dev|</th><th>Max</th><th>Shape RMS</th>
            <th>Height min/mean/max</th><th>Bead</th><th>Acq→path</th><th>Layer cost</th></>}
          <th>RMS</th><th>Valid</th></tr></thead>
        <tbody>{takes.map((r) => {
          const truth = r.annotation?.introduced_offset_mm;
          // How far the measured displacement lands from the one that was typed:
          // the claim the paper actually makes, per take.
          const detection = r.metrics
            ? Math.hypot(r.metrics.center_offset_mm[0] - (truth?.[0] ?? 0),
                         r.metrics.center_offset_mm[1] - (truth?.[1] ?? 0))
            : null;
          return <tr key={`${r.layer_index}-${r.take}`}
            className={`clickable${openTake === layerDirName(r) ? " selected" : ""}${r.valid ? "" : " invalid"}`}
            title="Show the figures for this take"
            onClick={() => setOpenTake(openTake === layerDirName(r) ? null : layerDirName(r))}>
            <td>{r.layer_index}</td><td>{r.take}{r.reprocessed && <sup title="reprocessed offline from the archived frame">R</sup>}</td>
            <td>{r.annotation?.phase || "—"}</td>
            <td>{truth ? `(${truth.join(", ")}) mm` : "—"}</td>
            <td className="num">{r.metrics ? r.metrics.center_offset_norm_mm.toFixed(2) : "—"}</td>
            <td className="num">{detection === null ? "—" : detection.toFixed(2)}</td>
            {allColumns && <>
              <td className="num">{r.metrics ? `${r.metrics.center_offset_mm[0].toFixed(1)} / ${r.metrics.center_offset_mm[1].toFixed(1)}` : "—"}</td>
              <td className="num">{r.metrics ? r.metrics.mean_absolute_mm.toFixed(2) : "—"}</td>
              <td className="num">{r.metrics ? r.metrics.maximum_mm.toFixed(2) : "—"}</td>
              <td className="num">{r.metrics ? r.metrics.shape_rms_mm.toFixed(2) : "—"}</td>
              <td className="num">{r.geometry ? `${r.geometry.height_min_mm.toFixed(1)} / ${r.geometry.height_mean_mm.toFixed(1)} / ${r.geometry.height_max_mm.toFixed(1)}` : "—"}</td>
              <td className="num">{r.geometry ? r.geometry.bead_width_mean_mm.toFixed(1) : "—"}</td>
              <td className="num">{r.timings_ms?.acquisition_to_path_ms
                ? `${Math.round(r.timings_ms.acquisition_to_path_ms)} ms` : "—"}</td>
              <td className="num">{r.timings_ms?.inspection_cycle_ms
                ? `${(r.timings_ms.inspection_cycle_ms / 1000).toFixed(1)} s` : "—"}</td>
            </>}
            <td className="num">{r.metrics ? r.metrics.rms_mm.toFixed(2) : "—"}</td>
            <td><span className={`badge ${r.valid ? "good" : "bad"}`}>{r.valid ? "VALID" : "INVALID"}</span>
              {!r.valid && <button className="secondary reprocess-btn" disabled={busy || status?.running}
                onClick={(e) => { e.stopPropagation(); reprocessTake(r); }}>Reprocess</button>}</td>
          </tr>;
        })}</tbody></table> : null}
      {takes.some((t) => !t.valid) ? <p className="hint warn-text">
        An invalid take measured nothing and is left out of every statistic — but its raw
        RGB-D frame is archived, so <b>Reprocess</b> re-runs the current processing on it
        with no robot motion. {takes.filter((t) => !t.valid).map((t) =>
          `L${t.layer_index} take ${t.take}: ${t.error || "invalid"}`).join(" · ")}
      </p> : null}
      {takes.some((t) => t.reprocessed) ? <p className="hint">
        <sup>R</sup> = reprocessed offline from the archived frame: its capture time is real,
        its processing time is not, and the summary keeps it out of the cycle statistic.
      </p> : null}
      {openTake && measureSession ? (() => {
        const take = measureSession.records.find((r) => layerDirName(r) === openTake);
        return take ? <TakeFigures trialId={measureSession.trial_id} take={take} /> : null;
      })() : null}
      <div className="btn-row">
        <button className="secondary" disabled={!measureSession} onClick={showPaper}>Paper summary</button>
        <button className="secondary" disabled={!measureSession || busy}
                onClick={downloadWordDraft}>Word draft (.docx)</button>
        <button className="secondary" disabled={!measureSession}
                onClick={() => setShowStack(!showStack)}>
          {showStack ? "Hide" : "Show"} stack figure
        </button>
      </div>
      {showStack && measureSession ? <div className="figure-gallery" style={{ marginTop: 10 }}>
        {[{ stem: "stack", label: "Ring stack",
            hint: "every layer's latest take, measured against nominal" },
          { stem: "tube", label: "Commanded bead vs measured",
            hint: "the bead drawn at its real diameter, each layer at its own height" }
         ].map(({ stem, label, hint }) => {
          const base = `/api/modules/extrusion/trials/${encodeURIComponent(measureSession.trial_id)}/figures/${stem}`;
          return <figure className="figure-card" key={stem}>
            <a href={`${base}.png`} target="_blank" rel="noreferrer">
              <img src={`${base}.png`} alt={label} loading="lazy" />
            </a>
            <figcaption><strong>{label}</strong> <span className="hint">{hint}</span>
              <span className="figure-links">
                <a href={`${base}.png`} target="_blank" rel="noreferrer">PNG</a>
                <a href={`${base}.pdf`} target="_blank" rel="noreferrer">PDF</a>
              </span>
            </figcaption>
          </figure>;
        })}
      </div> : null}
      {paper && <pre className="log" style={{ whiteSpace: "pre-wrap" }}>{paper}</pre>}
    </div>}

    {mode === "print" && headlineMetrics.length > 0 && <div className="card"><h2>Measured layers</h2>
      <table className="metrics"><thead><tr><th>Layer</th><th>Mean |dev|</th><th>RMS</th><th>Maximum</th><th>Completeness</th><th>Valid</th></tr></thead>
        <tbody>{headlineMetrics.map((item: any) => <tr key={item.layer_index}><td>{item.layer_index}</td>
          <td className="num">{item.metrics.mean_absolute_mm.toFixed(2)} mm</td><td className="num">{item.metrics.rms_mm.toFixed(2)} mm</td>
          <td className="num">{item.metrics.maximum_mm.toFixed(2)} mm</td><td className="num">{(item.metrics.path_completeness * 100).toFixed(1)}%</td>
          <td><span className={`badge ${item.metrics.valid ? "good" : "bad"}`}>{item.metrics.valid ? "VALID" : "INVALID"}</span></td></tr>)}</tbody></table>
      {result?.correction_available && <div className="btn-row"><button onClick={applyCorrection} disabled={busy || status?.running}>Create corrected cylinder plan</button>
        <span className="hint">Creates a new fingerprint; correction is not executed until it passes another dry run.</span></div>}
    </div>}

    {/* The protocol, out of the way until it is wanted. Clicking the backdrop or
        pressing Escape closes it; the page behind keeps its state. */}
    <dialog className="guide-modal" ref={guideDialog} onClose={() => setGuideOpen(false)}
            onClick={(e) => { if (e.target === guideDialog.current) setGuideOpen(false); }}>
      <div className="guide-body">
        <div className="guide-top">
          <h2>Run guide</h2>
          <button type="button" className="secondary" onClick={() => setGuideOpen(false)}>Close</button>
        </div>
        <ol className="guide-steps">
          {RUN.map((step, index) => <li key={step.id}
            className={step.done ? "done" : index === activeIndex ? "current" : ""}>
            <b>{step.title}</b>
            {step.progress && <em>{step.progress.have}/{step.progress.need}</em>}
            {step.hands && <span>{step.hands}</span>}
            {step.note && <span className="muted-note">{step.note}</span>}
          </li>)}
        </ol>
        <div className="io-note">
          <b>Bottom to top, and the error goes on last.</b> Rings 1, 2 and 3 are stacked and
          measured as true as a hand can place them — each layer must be measured as it goes
          on, because layer N's measurement floor <i>is</i> layer N−1's measured top. Then ring
          4 goes on <b>deliberately off-centre</b>: the controlled error the paper's claim
          rests on. Nothing is ever slid or lifted off, so no ring is disturbed after it has
          been measured.
        </div>
        <div className="io-note">
          <b>What ring 4 is scored against.</b> It arrives already displaced, so it has no
          undisplaced take of its own to pair with. It is paired against the measured centre of
          <b> ring 3</b> instead — which is what your rule measured anyway: how far ring 4 sits
          from the ring it was stacked on. Scored against the plan centre instead, the whole
          stack's hand-placement error would be charged to the sensing chain.
        </div>
        <div className="io-note">
          <b>Two kinds of repeat, and why only some batch.</b> The <b>noise floor</b> sends the
          arm out and back {NOISE_TRIPS} times on one press: its spread contains the robot's
          re-approach as well as the camera, which is what makes it the floor everything else
          is read against. Every later condition takes {REPEATS} frames with the arm
          <b> parked</b> — one trip, seconds apart — so it measures the sensing chain alone,
          and the robot's contribution is already known from the noise floor. Only
          <b> Re-place</b> stays one press per take: your hand has to move between those.
        </div>
        <div className="io-note">
          <b>Measuring the offset.</b> Keep every offset ≤ 25 mm (the search band is ±30 mm).
          The work frame's axes run along the board edges, so a steel rule laid along an edge
          IS the frame axis: mark where the ring sits, slide it along the rule, and type the
          distance you actually achieved — 12 mm scores as well as 10. Do <b>not</b> count
          ChArUco squares — this board's are 40 mm, past the 25 mm cap, and the ring would fall
          outside the search band. Take one throwaway measurement first to learn which edge is
          +X: the reported offset tells you the sign.
        </div>
        <div className="io-note">
          <b>What the numbers mean.</b> A pure shift of d reads centre offset ≈ d, max ≈ d,
          mean ≈ 0.64 d, RMS ≈ 0.71 d — the summary checks that relation for you and warns
          if it disagrees. This cell's error floor is 1.26 mm (hand-eye board consistency),
          so nothing sub-millimetre can be claimed. Nothing here is extruded: this is a
          controlled validation of the sensing-and-comparison chain, not print deviation.
        </div>
        <div className="io-note">
          <b>The side photo.</b> After a layer's capture finishes and the arm is home, it goes
          out once more for one RGB photo of the stack from the side — the picture the paper
          shows next to the numbers. It routes <b>neutral → TowardsSideCapture → SideCapture</b>
          and comes back the same way in reverse; the approach target is taught around the
          things standing near the cell, so it is used in <b>both</b> directions. One photo per
          press, filed with that press's last take. It measures nothing, its cost is kept out
          of the inspection-cycle figure, and if either taught target is missing it is skipped
          with a note rather than failing the measurement.
        </div>
        <div className="io-note warn-text">
          <b>Collision validation is OFF</b> for these moves: the hand-placed stack is not in
          the station model, so RoboDK's check only sees the cell furniture and was rejecting
          good inspection poses. The pose is still IK-checked and reachable and the camera
          stops 300 mm above the ring, but nothing screens the path against the real cell.
          Keep hands clear and watch the first move — including the trip to the side pose.
        </div>
      </div>
    </dialog>
  </div>;
}
