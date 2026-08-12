import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { moduleApi } from "../api/client";
import { useEvents, type JobEvent } from "../api/events";

const api = moduleApi("extrusion");

interface Recipe {
  radius_mm: number; layer_count: number; layer_height_mm: number;
  bead_diameter_mm: number; robot_speed_mm_s: number;
  extrusion_rate_pct: number; points_per_circle: number;
  correction_enabled: boolean; material: unknown[];
}
interface Setup {
  print_tool: string; work_frame: string; inspection_tool: string;
  inspection_target: string; center_x_mm: number; center_y_mm: number;
  orientation_rpy_deg: [number, number, number];
  approach_clearance_mm: number; retreat_clearance_mm: number;
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
  live_print_enabled: boolean;
}
interface StationOptions { tools: string[]; frames: string[]; targets: string[]; programs: string[]; }
interface Status {
  status: string; running: boolean; result?: any; error?: string | null;
  fingerprint?: string | null; geometry_preflight_passed: boolean;
  dry_run_passed: boolean; hardware_io_test_approved: boolean;
  live_print_enabled: boolean;
}

const recipeFields: Array<{ key: keyof Recipe; label: string; min: number; max: number; step: number; unit: string }> = [
  { key: "radius_mm", label: "Radius", min: 5, max: 150, step: 1, unit: "mm" },
  { key: "layer_count", label: "Layers", min: 1, max: 30, step: 1, unit: "" },
  { key: "layer_height_mm", label: "Layer height", min: .5, max: 20, step: .5, unit: "mm" },
  { key: "bead_diameter_mm", label: "Bead diameter", min: .5, max: 30, step: .5, unit: "mm" },
  { key: "robot_speed_mm_s", label: "Robot speed", min: 5, max: 500, step: 5, unit: "mm/s" },
  { key: "extrusion_rate_pct", label: "Extrusion rate", min: 0, max: 100, step: 1, unit: "%" },
];

function Birdseye({ layer, radius, centerX, centerY, compact = false }: {
  layer: Layer; radius: number; centerX: number; centerY: number; compact?: boolean;
}) {
  const extent = Math.max(radius * 1.25, 1);
  const size = compact ? 88 : 420;
  const pad = compact ? 8 : 34;
  const scale = (size / 2 - pad) / extent;
  const center = size / 2;
  const d = layer.points.map((p, index) => {
    const x = center + (p.x_mm - centerX) * scale;
    const y = center - (p.y_mm - centerY) * scale;
    return `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
  return <svg viewBox={`0 0 ${size} ${size}`} className={compact ? "layer-mini-map" : "birdseye-map"}
              aria-label={`Bird's-eye path for layer ${layer.layer_index}`}>
    <rect width={size} height={size} fill="#090d14" />
    {!compact && <>
      {[-1, -.5, 0, .5, 1].map((v) => <g key={v}>
        <line x1={center + v * radius * scale} y1={pad} x2={center + v * radius * scale} y2={size - pad}
              stroke={v === 0 ? "#344055" : "#1b2331"} strokeWidth={v === 0 ? 1.2 : 1} />
        <line x1={pad} y1={center + v * radius * scale} x2={size - pad} y2={center + v * radius * scale}
              stroke={v === 0 ? "#344055" : "#1b2331"} strokeWidth={v === 0 ? 1.2 : 1} />
      </g>)}
      <text x={size - pad} y={center - 7} textAnchor="end" className="preview-axis">+X</text>
      <text x={center + 7} y={pad + 11} className="preview-axis">+Y</text>
    </>}
    <path d={d} fill="none" stroke="#39d0bd" strokeWidth={compact ? 2.2 : 4}
          strokeLinecap="round" strokeLinejoin="round" />
    <circle cx={center + (layer.points[0].x_mm - centerX) * scale}
            cy={center - (layer.points[0].y_mm - centerY) * scale}
            r={compact ? 2 : 5} fill="#f0a45d" />
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
  const [confirmLive, setConfirmLive] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  const refreshStatus = useCallback(() => {
    api.get<Status>("/status").then((value) => {
      setStatus(value);
      if (value.result?.kind?.startsWith("cylinder_")) setResult(value.result);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    api.get<Config>("/config").then((value) => {
      setConfig(value); setRecipe(value.defaults); setSetup(value.setup_defaults);
    }).catch((e) => setMessage(e.message));
    refreshStatus();
  }, [refreshStatus]);

  useEffect(() => subscribe((event: JobEvent) => {
    const name = event.payload?.name as string | undefined;
    if (event.type === "progress" && busy) {
      setProgress(event.payload); setMessage(event.payload.message || "Working…");
    } else if (event.type === "log" && busy) {
      setLogs((old) => [...old, event.payload.message]);
    } else if (event.type === "result" && name?.startsWith("extrusion-")) {
      setResult(event.payload.result); setBusy(false); setConfirmLive(false);
      setMessage(name === "extrusion-dry-run" ? "Dry run passed for this exact plan." : "Print and layer archive completed.");
      refreshStatus();
    } else if (event.type === "error" && name?.startsWith("extrusion-")) {
      setBusy(false); setMessage(event.payload.message); setLogs((old) => [...old, `ERROR: ${event.payload.message}`]);
      refreshStatus();
    } else if (event.type === "status" && name?.startsWith("extrusion-")) {
      refreshStatus();
    }
  }), [subscribe, refreshStatus, busy]);

  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [logs]);

  const invalidate = (note = "Inputs changed — generate again; prior checks were invalidated.") => {
    setPlan(null); setPreflight(null); setResult(null); setConfirmLive(false); setMessage(note);
    setStatus((old) => old ? { ...old, geometry_preflight_passed: false, dry_run_passed: false, live_print_enabled: false } : old);
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

  const connect = async () => {
    setBusy(true); setMessage("Loading RoboDK station…");
    try {
      await api.post("/connect");
      const discovered = await api.get<StationOptions>("/station-options");
      setOptions(discovered); setConnected(true);
      setMessage("Station loaded. Select print/inspection tools, work frame, and inspection target.");
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
  const generate = async () => {
    if (!recipe || !setup) return;
    setBusy(true);
    try {
      const next = await api.post<Plan>("/generate", { recipe, setup });
      setPlan(next); setSelectedLayer(1); setPreflight(null); setResult(null);
      setMessage(`Generated ${next.layers.length} complete closed paths.`); refreshStatus();
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
  const runPreflight = async () => {
    if (!plan) return;
    setBusy(true);
    try {
      const value = await api.post<any>("/preflight", { fingerprint: plan.fingerprint });
      setPreflight(value); setMessage(value.station?.ready ? value.note : "Geometry passed, but selected station items are missing.");
      refreshStatus();
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
  const startJob = async (kind: "dry-run" | "print") => {
    if (!plan) return;
    setBusy(true); setLogs([]); setProgress({ step: 0, total: plan.layers.length, message: "Starting…" });
    try {
      await api.post(`/${kind}`, kind === "print"
        ? { fingerprint: plan.fingerprint, confirm_live: confirmLive }
        : { fingerprint: plan.fingerprint });
      setMessage(kind === "dry-run" ? "DRY_RUN started — physical outputs blocked." : "LIVE_PRINT started.");
      refreshStatus();
    } catch (e: any) { setBusy(false); setMessage(e.message); }
  };
  const cancel = async () => { await api.post("/cancel"); setMessage("Cancellation requested; forcing the safe exit sequence."); };
  const applyCorrection = async () => {
    if (!plan) return;
    setBusy(true);
    try {
      const corrected = await api.post<Plan>("/correction/apply", { fingerprint: plan.fingerprint });
      setPlan(corrected); setSelectedLayer(1); setPreflight(null); setResult(null); setConfirmLive(false);
      setMessage("Corrected command generated. It has a new fingerprint and must pass preflight + dry run.");
      refreshStatus();
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };

  const layer = plan?.layers[Math.min(plan.layers.length - 1, Math.max(0, selectedLayer - 1))];
  const selectionsReady = Boolean(setup?.print_tool && setup?.work_frame && setup?.inspection_tool && setup?.inspection_target);
  const stationReady = Boolean(preflight?.station?.ready);
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
          <label>Inspection target<select value={setup?.inspection_target || ""} disabled={!options} onChange={(e) => updateSetup("inspection_target", e.target.value)}>
            <option value="">Select target…</option>{options?.targets.map((v) => <option key={v}>{v}</option>)}</select></label>
        </div>
        {setup && <>
          <div className="motion-number-grid">
            <label>Center X (mm)<input type="number" value={setup.center_x_mm} onChange={(e) => updateSetup("center_x_mm", Number(e.target.value))} /></label>
            <label>Center Y (mm)<input type="number" value={setup.center_y_mm} onChange={(e) => updateSetup("center_y_mm", Number(e.target.value))} /></label>
            {["A / roll", "B / pitch", "C / yaw"].map((label, i) => <label key={label}>{label} (°)<input type="number" value={setup.orientation_rpy_deg[i]} onChange={(e) => updateOrientation(i, Number(e.target.value))} /></label>)}
            <label>Approach (mm)<input type="number" min="1" value={setup.approach_clearance_mm} onChange={(e) => updateSetup("approach_clearance_mm", Number(e.target.value))} /></label>
            <label>Retreat (mm)<input type="number" min="1" value={setup.retreat_clearance_mm} onChange={(e) => updateSetup("retreat_clearance_mm", Number(e.target.value))} /></label>
          </div>
          <div className="hint">Orientation uses RoboDK XYZRPW in the selected work frame. These exact values are fingerprinted and dry-run.</div>
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
        <div className="btn-row"><button disabled={!recipe || !selectionsReady || busy || status?.running} onClick={generate}>Generate complete plan</button>
          <button className="secondary" disabled={!plan || busy || status?.running} onClick={runPreflight}>Geometry & station preflight</button></div>
      </div>
    </div>

    <div className="card birdseye-card">
      <div className="birdseye-head"><div><h2>Bird’s-eye layer review</h2><p>Exact XY geometry in the selected work frame. Choose a layer to inspect its path and Z height.</p></div>
        {layer && <div className="layer-readout">LAYER {layer.layer_index}<b>Z {layer.nominal_z_mm.toFixed(2)} mm</b></div>}</div>
      {plan && layer ? <div className="birdseye-layout">
        <Birdseye layer={layer} radius={plan.recipe.radius_mm}
          centerX={plan.setup.center_x_mm} centerY={plan.setup.center_y_mm} />
        <div className="layer-rail">{plan.layers.map((item) => <button key={item.layer_index}
          className={`layer-tile ${item.layer_index === selectedLayer ? "selected" : ""}`}
          onClick={() => setSelectedLayer(item.layer_index)}>
          <Birdseye layer={item} radius={plan.recipe.radius_mm}
            centerX={plan.setup.center_x_mm} centerY={plan.setup.center_y_mm} compact />
          <span>Layer {item.layer_index}</span><small>Z {item.nominal_z_mm.toFixed(1)} mm</small>
        </button>)}</div>
      </div> : <div className="empty cylinder-empty">Connect, select station items, and generate a plan.</div>}
      {plan && <div className="kv preview-kv"><span className="k">Plan fingerprint</span><span className="v">{plan.fingerprint.slice(0, 16)}</span>
        <span className="k">Total commanded path</span><span className="v">{plan.total_path_length_mm.toFixed(1)} mm</span>
        <span className="k">Points / layer</span><span className="v">{plan.layers[0].points.length}</span></div>}
    </div>

    <div className={`card extrusion-safety ${status?.live_print_enabled ? "ready" : "locked"}`}>
      <h2>Safety workflow</h2>
      <div className="workflow-steps"><span className={plan ? "done" : ""}>1 Generate</span>
        <span className={preflight?.all_ok && stationReady ? "done" : ""}>2 Preflight</span>
        <span className={status?.dry_run_passed ? "done" : ""}>3 Complete dry run</span>
        <span className={result?.kind === "cylinder_print" ? "done" : ""}>4 Print & record</span></div>
      <p className="status-line">{message}</p>
      {(busy || status?.running) && <><div className="progress"><div style={{ width: `${pct}%` }} /></div><div className="status-line">{progress.message}</div></>}
      {config && <div className="io-note">Valve: <code>{config.integration.valve_outputs.join(" + ")}</code> via {config.integration.air_on_program}/{config.integration.air_off_program}. Hardware approval: <b>{config.integration.hardware_io_test_approved ? "APPROVED" : "NOT APPROVED"}</b>.</div>}
      <div className="btn-row"><button disabled={!plan || !preflight?.all_ok || !stationReady || busy || status?.running} onClick={() => startJob("dry-run")}>Run complete RoboDK dry run</button>
        {(busy || status?.running) && <button className="secondary" onClick={cancel}>Cancel safely</button>}</div>
      <label className="live-confirm"><input type="checkbox" checked={confirmLive} onChange={(e) => setConfirmLive(e.target.checked)} disabled={!status?.dry_run_passed || status?.running} />
        I confirm the selected tool/frame/orientation, cell clearance, material system, and live extrusion run.</label>
      <button className="live-print-btn" disabled={!plan || !status?.live_print_enabled || !confirmLive || busy || status?.running} onClick={() => startJob("print")}>Print & record — LIVE ROBOT</button>
      <div className="log" ref={logRef}>{logs.length ? logs.map((line, i) => <div key={i} className={line.startsWith("ERROR") ? "err" : ""}>{line}</div>) : "Job timeline will appear here."}</div>
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
