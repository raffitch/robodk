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
  center_mm?: [number, number] | null;
}
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
  annotation: { introduced_offset_mm?: [number, number] | null; note?: string };
  metrics: { mean_absolute_mm: number; rms_mm: number; maximum_mm: number;
    center_offset_mm: [number, number]; center_offset_norm_mm: number; shape_rms_mm: number;
    measured_radius_mm: number; path_completeness: number };
  geometry: { height_mean_mm: number; height_min_mm: number; height_max_mm: number;
    bead_width_mean_mm: number } | null;
  timings_ms: { capture_ms: number; total_ms: number; acquisition_to_path_ms: number };
}
interface Characterization {
  index: number; radius_mm: number; center_mm: [number, number]; bead_width_mm: number;
  top_z_mean_mm: number; top_z_min_mm: number; top_z_max_mm: number;
}
interface MeasureSession {
  trial_id: string; takes: Record<string, number>; records: MeasureTake[];
  characterizations: Characterization[];
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

function BirdseyeStack({ plan, selectedLayer, onSelect }: {
  plan: Plan; selectedLayer: number; onSelect: (layer: number) => void;
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
  const bounds = [...plane, ...projectedLayers.flatMap((entry) => entry.rawPoints)];
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
      return <g key={item.layer_index} className="stack-layer"
                onClick={() => onSelect(item.layer_index)}>
        <path d={path(rawPoints)} fill={selected ? "rgba(76,154,255,.10)" : "none"}
              stroke={selected ? "#66a6ff" : "#39d0bd"}
              strokeOpacity={selected ? 1 : .42 + item.layer_index / plan.layers.length * .34}
              strokeWidth={selected ? 4 : 2} strokeLinecap="round" strokeLinejoin="round" />
        {selected && <circle cx={start.x} cy={start.y} r="5" fill="#f0a45d" />}
      </g>;
    })}
    {axes.map((axis) => <g key={axis.label}>
      <line x1={axisOrigin.x} y1={axisOrigin.y} x2={axis.end.x} y2={axis.end.y}
            stroke={axis.color} strokeWidth="1.6" />
      <text x={axis.end.x + 6} y={axis.end.y - 5} className="preview-axis"
            style={{ fill: axis.color }}>{axis.label}</text>
    </g>)}
    <text x="14" y={height - 14} className="preview-note">
      OBLIQUE XYZ · Z ×{zExaggeration.toFixed(1)} · ΔZ {layerHeight.toFixed(2)} mm
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
  const [confirmMotion, setConfirmMotion] = useState(false);
  const [paper, setPaper] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  const refreshStatus = useCallback(() => {
    api.get<Status>("/status").then((value) => {
      setStatus(value);
      if (value.result?.kind?.startsWith("cylinder_")) setResult(value.result);
    }).catch(() => {});
  }, []);
  const refreshSurface = useCallback(() => {
    api.get<ScanSurface>("/scan-surface").then(setSurface).catch(() => setSurface(null));
  }, []);
  useEffect(() => {
    api.get<Config>("/config").then((value) => {
      setConfig(value); setRecipe(value.defaults); setSetup(value.setup_defaults);
    }).catch((e) => setMessage(e.message));
    api.get<Plan>("/plan").then((value) => {
      setPlan(value); setRecipe(value.recipe); setSetup(value.setup);
      setSelectedLayer(1); setQuickLayers([1]);
    }).catch(() => {});
    refreshStatus(); refreshSurface();
  }, [refreshStatus, refreshSurface]);

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

  const centerOnScannedSurface = async () => {
    if (!setup || !recipe) return;
    setBusy(true); setMessage("Reading the applied scan surface…");
    try {
      const response = await api.post<{ setup: Partial<Setup>; surface: ScanSurface; fit: SurfaceFit }>(
        "/center-on-surface", { radius_mm: recipe.radius_mm, bead_diameter_mm: recipe.bead_diameter_mm });
      const frame = response.setup.work_frame || setup.work_frame;
      const pose = await api.post<{ xyz_mm: number[]; rpy_deg: number[] }>("/current-tcp", {
        print_tool: setup.print_tool, work_frame: frame,
      });
      setSurface({ ...response.surface, applied: true });
      setSetup({ ...setup, ...response.setup,
        orientation_rpy_deg: pose.rpy_deg as [number, number, number] });
      const size = response.surface.size_mm;
      invalidate(response.fit.inside
        ? `Centred on the scanned surface${size ? ` (${size[0].toFixed(0)} × ${size[1].toFixed(0)} mm)` : ""} in ${response.setup.work_frame}; current neutral TCP orientation captured as ${pose.rpy_deg.map((v) => v.toFixed(1)).join(", ")}° — generate coordinates next.`
        : `Centred, but the wall overhangs the measured surface by ${Math.abs(response.fit.minimum_margin_mm ?? 0).toFixed(1)} mm. Reduce the radius or re-scan a larger surface; preflight will reject it.`);
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };

  const connect = async () => {
    setBusy(true); setMessage("Loading RoboDK station…");
    try {
      await api.post("/connect");
      const discovered = await api.get<StationOptions>("/station-options");
      setOptions(discovered); setConnected(true); refreshSurface();
      setMessage("Station loaded. Select print/inspection tools, work frame, and inspection target.");
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
  const generate = async () => {
    if (!recipe || !setup) return;
    setBusy(true);
    try {
      const next = await api.post<Plan>("/generate", { recipe, setup });
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
  const newMeasureSession = async () => {
    setBusy(true);
    try {
      const data = await api.post<{ session: MeasureSession }>("/measure/session/new", { note: measureNote });
      setMeasureSession(data.session); setPaper(null);
      setMessage(`New measure-only session ${data.session.trial_id}.`);
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
  const characterize = async () => {
    setBusy(true);
    try {
      await api.post("/measure/characterize", {
        confirm_robot_motion: confirmMotion,
        collision_check_enabled: true,
      });
      setMessage("CHARACTERIZE started — the robot moves the camera over ring 1, measures it and returns.");
      refreshStatus();
    } catch (e: any) { setBusy(false); setMessage(e.message); }
  };
  const applyCharacterization = async () => {
    setBusy(true);
    try {
      const next = await api.post<Plan>("/measure/apply-characterization");
      setPlan(next); setRecipe(next.recipe); setSetup(next.setup);
      setPreflight(null); setResult(null); setSelectedLayer(1);
      setMessage(`Recipe and placement set from the measured ring: r ${next.recipe.radius_mm} mm, bead ${next.recipe.bead_diameter_mm} mm, layer ${next.recipe.layer_height_mm} mm.`);
      refreshStatus();
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
  const measure = async () => {
    if (!plan) return;
    setBusy(true);
    try {
      const shifted = offsetX !== 0 || offsetY !== 0;
      await api.post("/measure/layer", {
        fingerprint: plan.fingerprint, layer_index: measureLayer,
        annotation: { introduced_offset_mm: shifted ? [offsetX, offsetY] : null, note: measureNote },
        confirm_robot_motion: confirmMotion,
        collision_check_enabled: true,
      });
      setMessage(`MEASURE layer ${measureLayer} started — camera move only; no extrusion, no valve.`);
      refreshStatus();
    } catch (e: any) { setBusy(false); setMessage(e.message); }
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

  return <div>
    <h1 className="page-title">Cylinder Test</h1>
    <p className="page-sub">Top-down toolpath review → complete RoboDK dry run → layer print → one RGB-D inspection → archive.</p>

    <div className={`card conn-banner ${connected ? "ready" : ""}`}>
      <div className="conn-row"><span className="conn-label">RoboDK station</span>
        <span className="status-line">{connected ? "connected · station items discovered" : "not connected"}</span>
        <button className="secondary" disabled={busy || status?.running} onClick={connect}>{connected ? "Refresh items" : "Connect"}</button>
      </div>
    </div>

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
              onClick={centerOnScannedSurface}>Center on scanned surface</button>
            <button className="secondary" disabled={!connected || busy || !setup.print_tool || !setup.work_frame}
              onClick={captureCurrentOrientation}>Capture neutral orientation only</button>
            <button className="secondary" disabled={!connected || busy || !setup.print_tool || !setup.work_frame}
              onClick={seedFromCurrentTcp}>Seed path start from current TCP</button>
          </div>
          <div className={`surface-row ${surface?.available ? "ready" : "warn-text"}`}>
            <span className="conn-label">Scan surface</span>
            <span className="status-line">{surface?.applied
              ? `${surface.frame} · ${surface.size_mm ? `${surface.size_mm[0].toFixed(0)} × ${surface.size_mm[1].toFixed(0)} mm · ` : ""}run ${surface.run_id ?? "unknown"}${surface.applied_at ? ` · applied ${surface.applied_at.replace("T", " ")}` : ""}`
              : surface?.note || "No scanned surface applied — run the Scan module first."}</span>
            <button className="secondary" disabled={busy} onClick={refreshSurface}>Re-check</button>
          </div>
          {surface?.applied && !surface.available && <div className="hint warn-text">{surface.note}</div>}
          <div className="hint">Preferred flow: scan the surface, insert it, then <b>Center on scanned surface</b> — the cylinder lands in the middle of the measured rectangle and automatically captures the current neutral TCP orientation. <b>Capture neutral orientation only</b> refreshes orientation without changing the cylinder center or Z. Center, build-plane Z, and RoboDK XYZRPW are expressed in the selected work frame. The generated IK keeps the neutral front/elbow/wrist configuration, limits both axes 4 and 6 to ±90°, and samples the interpolated joint path to block hidden wrist flips before playback. These exact values are fingerprinted and simulated.</div>
          {setup.scan_run_id
            ? <div className="hint">Placed on scan run <code>{setup.scan_run_id}</code>. Re-scanning the surface invalidates this placement, and preflight rejects a wall that overhangs the measured rectangle.</div>
            : surface?.available && surface.frame !== setup.work_frame &&
              <div className="hint warn-text">A scanned surface is applied on <code>{surface.frame}</code> but this path is placed manually in <code>{setup.work_frame || "no frame"}</code>.</div>}
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
        {layer && <div className="layer-readout">LAYER {layer.layer_index}<b>Z {layer.nominal_z_mm.toFixed(2)} mm</b></div>}</div>
      {visualPlan && layer ? <div className="birdseye-layout">
        <BirdseyeStack plan={visualPlan} selectedLayer={visualSelectedLayer} onSelect={setSelectedLayer} />
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
        <div className="btn-row"><button disabled={!recipe || !selectionsReady || busy || status?.running} onClick={generate}>Generate coordinates & fingerprint</button>
          <button className="secondary" disabled={!plan || busy || status?.running} onClick={runPreflight}>Geometry & station preflight</button>
          <button className="secondary" disabled={!connected || busy || status?.running} onClick={resetGenerated}>Reset / clean RoboDK path</button></div>
      </div>
    </div>

    <div className={`card extrusion-safety ${status?.live_print_enabled ? "ready" : "locked"}`}>
      <h2>Safety workflow</h2>
      <div className="workflow-steps"><span className={plan ? "done" : ""}>1 Generate</span>
        <span className={preflight?.all_ok && stationReady ? "done" : ""}>2 Preflight</span>
        <span className={status?.quick_sim_passed ? "done" : ""}>3a Visual simulation</span>
        <span className={status?.dry_run_passed ? "done" : ""}>3b Collision-validated dry run</span>
        <span className={result?.kind === "cylinder_print" ? "done" : ""}>4 Print & record</span></div>
      <p className="status-line">{message}</p>
      {(busy || status?.running) && <><div className="progress"><div style={{ width: `${pct}%` }} /></div><div className="status-line">{progress.message}</div></>}
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
    </div>

    <div className="card">
      <h2>Ring stack — measure only (no extrusion)</h2>
      <p className="hint">Hand-placed dried rings. Each press moves ONLY the camera to the derived
        inspection pose (collision-checked), takes one RGB-D frame, measures it and returns.
        No layer program, no valve. Archived as <code>MEASURE_ONLY</code>, never counted as a print.</p>
      <div className="btn-row">
        <input placeholder="session / ring note" value={measureNote} onChange={(e) => setMeasureNote(e.target.value)} />
        <button className="secondary" disabled={!plan || busy || status?.running} onClick={newMeasureSession}>New session</button>
        <span className="hint">{measureSession ? `session ${measureSession.trial_id}` : "no session yet (one is created on first measure)"}</span>
      </div>
      <label><input type="checkbox" checked={confirmMotion} onChange={(e) => setConfirmMotion(e.target.checked)} />
        I confirm the robot may move the camera to the inspection pose (hands clear of the cell).</label>
      <div className="btn-row">
        <button disabled={!plan || !connected || !confirmMotion || busy || status?.running} onClick={characterize}>Characterize ring 1 — ROBOT MOVES</button>
        {measureSession?.characterizations?.length ? (() => {
          const c = measureSession.characterizations[measureSession.characterizations.length - 1];
          return <>
            <span className="hint">measured: r {c.radius_mm.toFixed(1)} mm · bead {c.bead_width_mm.toFixed(1)} mm ·
              height {c.top_z_min_mm.toFixed(1)}–{c.top_z_max_mm.toFixed(1)} (mean {c.top_z_mean_mm.toFixed(1)}) mm ·
              centre ({c.center_mm[0].toFixed(1)}, {c.center_mm[1].toFixed(1)})</span>
            <button className="secondary" disabled={busy || status?.running} onClick={applyCharacterization}>Apply to recipe &amp; placement</button>
          </>;
        })() : null}
      </div>
      <div className="btn-row">
        <label>Layer <select value={measureLayer} onChange={(e) => setMeasureLayer(Number(e.target.value))}>
          {(plan?.layers ?? []).map((l) => <option key={l.layer_index} value={l.layer_index}>{l.layer_index}</option>)}
        </select></label>
        <label>Introduced offset X <input type="number" step={1} value={offsetX} onChange={(e) => setOffsetX(Number(e.target.value))} /> mm</label>
        <label>Y <input type="number" step={1} value={offsetY} onChange={(e) => setOffsetY(Number(e.target.value))} /> mm</label>
        <button disabled={!plan || !connected || !confirmMotion || busy || status?.running} onClick={measure}>Measure layer {measureLayer} — ROBOT MOVES</button>
        {(busy || status?.running) && <button className="secondary" onClick={cancel}>Cancel</button>}
      </div>
      {measureSession?.records?.length ? <table className="metrics">
        <thead><tr><th>Layer</th><th>Take</th><th>Introduced</th><th>Offset dx/dy (|d|)</th><th>Mean |dev|</th><th>RMS</th><th>Max</th><th>Shape RMS</th><th>Height min/mean/max</th><th>Bead</th><th>Acq→path</th><th>Valid</th></tr></thead>
        <tbody>{measureSession.records.map((r) => <tr key={`${r.layer_index}-${r.take}`}>
          <td>{r.layer_index}</td><td>{r.take}</td>
          <td>{r.annotation?.introduced_offset_mm ? `(${r.annotation.introduced_offset_mm.join(", ")}) mm` : "—"}</td>
          <td className="num">{r.metrics.center_offset_mm[0].toFixed(1)} / {r.metrics.center_offset_mm[1].toFixed(1)} ({r.metrics.center_offset_norm_mm.toFixed(2)})</td>
          <td className="num">{r.metrics.mean_absolute_mm.toFixed(2)}</td><td className="num">{r.metrics.rms_mm.toFixed(2)}</td>
          <td className="num">{r.metrics.maximum_mm.toFixed(2)}</td><td className="num">{r.metrics.shape_rms_mm.toFixed(2)}</td>
          <td className="num">{r.geometry ? `${r.geometry.height_min_mm.toFixed(1)} / ${r.geometry.height_mean_mm.toFixed(1)} / ${r.geometry.height_max_mm.toFixed(1)}` : "—"}</td>
          <td className="num">{r.geometry ? r.geometry.bead_width_mean_mm.toFixed(1) : "—"}</td>
          <td className="num">{Math.round(r.timings_ms.acquisition_to_path_ms)} ms</td>
          <td><span className={`badge ${r.valid ? "good" : "bad"}`}>{r.valid ? "VALID" : "INVALID"}</span></td>
        </tr>)}</tbody></table> : null}
      <div className="btn-row">
        <button className="secondary" disabled={!measureSession} onClick={showPaper}>Paper summary</button>
      </div>
      {paper && <pre className="log" style={{ whiteSpace: "pre-wrap" }}>{paper}</pre>}
    </div>

    {headlineMetrics.length > 0 && <div className="card"><h2>Measured layers</h2>
      <table className="metrics"><thead><tr><th>Layer</th><th>Mean |dev|</th><th>RMS</th><th>Maximum</th><th>Completeness</th><th>Valid</th></tr></thead>
        <tbody>{headlineMetrics.map((item: any) => <tr key={item.layer_index}><td>{item.layer_index}</td>
          <td className="num">{item.metrics.mean_absolute_mm.toFixed(2)} mm</td><td className="num">{item.metrics.rms_mm.toFixed(2)} mm</td>
          <td className="num">{item.metrics.maximum_mm.toFixed(2)} mm</td><td className="num">{(item.metrics.path_completeness * 100).toFixed(1)}%</td>
          <td><span className={`badge ${item.metrics.valid ? "good" : "bad"}`}>{item.metrics.valid ? "VALID" : "INVALID"}</span></td></tr>)}</tbody></table>
      {result?.correction_available && <div className="btn-row"><button onClick={applyCorrection} disabled={busy || status?.running}>Create corrected cylinder plan</button>
        <span className="hint">Creates a new fingerprint; correction is not executed until it passes another dry run.</span></div>}
    </div>}
  </div>;
}
