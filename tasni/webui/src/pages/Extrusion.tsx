import { useEffect, useMemo, useState } from "react";
import { moduleApi } from "../api/client";

const api = moduleApi("extrusion");

interface Recipe {
  radius_mm: number; layer_count: number; layer_height_mm: number;
  bead_diameter_mm: number; robot_speed_mm_s: number;
  extrusion_rate_pct: number; points_per_circle: number;
  correction_enabled: boolean; material: unknown[];
}
interface Point { x_mm: number; y_mm: number; z_mm: number; }
interface Plan {
  fingerprint: string; recipe: Recipe; total_path_length_mm: number;
  layers: Array<{ layer_index: number; nominal_z_mm: number; points: Point[] }>;
}
interface Config {
  defaults: Recipe;
  integration: {
    extruder_tool: string; work_frame: string; inspection_target: string;
    air_on_program: string; air_off_program: string; valve_outputs: string[];
    mapping_source: string; mapping_verified: boolean; hardware_io_test_approved: boolean;
  };
  live_print_enabled: boolean;
}

const fields: Array<{ key: keyof Recipe; label: string; min: number; max: number; step: number; unit: string }> = [
  { key: "radius_mm", label: "Radius", min: 5, max: 150, step: 1, unit: "mm" },
  { key: "layer_count", label: "Layers", min: 1, max: 20, step: 1, unit: "" },
  { key: "layer_height_mm", label: "Layer height", min: .5, max: 20, step: .5, unit: "mm" },
  { key: "bead_diameter_mm", label: "Bead diameter", min: .5, max: 30, step: .5, unit: "mm" },
  { key: "robot_speed_mm_s", label: "Robot speed", min: 5, max: 500, step: 5, unit: "mm/s" },
  { key: "extrusion_rate_pct", label: "Extrusion rate", min: 0, max: 100, step: 1, unit: "%" },
];

export default function Extrusion() {
  const [config, setConfig] = useState<Config | null>(null);
  const [recipe, setRecipe] = useState<Recipe | null>(null);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [preflight, setPreflight] = useState<any>(null);
  const [message, setMessage] = useState("Configure a small cylinder, then generate its layer paths.");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.get<Config>("/config").then((value) => {
      setConfig(value); setRecipe(value.defaults);
    }).catch((e) => setMessage(e.message));
  }, []);

  const preview = useMemo(() => {
    if (!plan) return "";
    const all = plan.layers.flatMap((l) => l.points);
    const r = Math.max(...all.map((p) => Math.hypot(p.x_mm, p.y_mm)), 1);
    const colors = ["#39d0bd", "#4c9aff", "#b48cff", "#f0a45d"];
    return plan.layers.map((layer, index) => {
      const d = layer.points.map((p, i) => {
        const x = 160 + p.x_mm / r * 125;
        const y = 160 - p.y_mm / r * 125;
        return `${i ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
      }).join(" ");
      return <path key={layer.layer_index} d={d} fill="none" stroke={colors[index % colors.length]}
                   strokeWidth={Math.max(1.5, 6 - index * .35)} opacity={.92 - index * .025} />;
    });
  }, [plan]);

  const update = (key: keyof Recipe, value: number | boolean) => {
    setRecipe((r) => r ? { ...r, [key]: value } : r);
    setPlan(null); setPreflight(null);
    setMessage("Recipe changed — generate a new toolpath; prior checks were invalidated.");
  };
  const generate = async () => {
    if (!recipe) return;
    setBusy(true);
    try {
      const result = await api.post<Plan>("/generate", recipe);
      setPlan(result); setPreflight(null);
      setMessage(`Generated ${result.layers.length} closed layer paths.`);
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
  const runPreflight = async () => {
    if (!plan) return;
    setBusy(true);
    try {
      const result = await api.post<any>("/preflight", { fingerprint: plan.fingerprint });
      setPreflight(result); setMessage(result.note);
    } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };
  const dryRun = async () => {
    if (!plan) return;
    setBusy(true);
    try { await api.post("/dry-run", { fingerprint: plan.fingerprint }); }
    catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
  };

  return (
    <div>
      <h1 className="page-title">Cylinder Test</h1>
      <p className="page-sub">Circular layer generation, fail-safe dry run, single-frame inspection, and opt-in correction.</p>

      <div className="extrusion-layout">
        <div className="card">
          <h2>Recipe</h2>
          {recipe && fields.map((field) => (
            <label className="recipe-field" key={field.key}>
              <span>{field.label}</span>
              <input type="range" min={field.min} max={field.max} step={field.step}
                     value={recipe[field.key] as number}
                     onChange={(e) => update(field.key, Number(e.target.value))} />
              <input className="recipe-number" type="number" min={field.min} max={field.max}
                     step={field.step} value={recipe[field.key] as number}
                     onChange={(e) => update(field.key, Number(e.target.value))} />
              <small>{field.unit}</small>
            </label>
          ))}
          {recipe && (
            <label className="correction-toggle">
              <input type="checkbox" checked={recipe.correction_enabled}
                     onChange={(e) => update("correction_enabled", e.target.checked)} />
              Generate compensation after a valid measurement
            </label>
          )}
          <div className="btn-row">
            <button disabled={!recipe || busy} onClick={generate}>Generate toolpath</button>
            <button className="secondary" disabled={!plan || busy} onClick={runPreflight}>Geometry preflight</button>
          </div>
        </div>

        <div className="card cylinder-preview-card">
          <h2>Layer preview</h2>
          {plan ? <svg viewBox="0 0 320 320" className="cylinder-preview">
            <circle cx="160" cy="160" r="125" fill="none" stroke="#283244" strokeDasharray="4 5" />
            <line x1="20" y1="160" x2="300" y2="160" stroke="#202838" />
            <line x1="160" y1="20" x2="160" y2="300" stroke="#202838" />
            {preview}
          </svg> : <div className="empty cylinder-empty">Generate a recipe to preview its closed paths.</div>}
          {plan && <div className="kv">
            <span className="k">Fingerprint</span><span className="v">{plan.fingerprint.slice(0, 12)}</span>
            <span className="k">Total path</span><span className="v">{plan.total_path_length_mm.toFixed(1)} mm</span>
            <span className="k">Points / layer</span><span className="v">{plan.layers[0].points.length}</span>
          </div>}
        </div>
      </div>

      <div className={`card extrusion-safety ${config?.live_print_enabled ? "ready" : "locked"}`}>
        <h2>Safety workflow</h2>
        <div className="workflow-steps">
          <span className={plan ? "done" : ""}>1 Generate</span>
          <span className={preflight?.all_ok ? "done" : ""}>2 Geometry preflight</span>
          <span>3 RoboDK dry run</span><span>4 Print & record</span>
        </div>
        <p className="status-line">{message}</p>
        {config && <div className="io-note">
          Valve mapping recovered from <code>{config.integration.mapping_source}</code>:
          {" "}{config.integration.valve_outputs.join(" + ")} via {config.integration.air_on_program}/{config.integration.air_off_program}.
          Hardware approval: <b>{config.integration.hardware_io_test_approved ? "approved" : "NOT APPROVED — live output locked"}</b>.
        </div>}
        <div className="btn-row">
          <button disabled={!plan || !preflight?.all_ok || busy} onClick={dryRun}>Run complete RoboDK dry run</button>
          <button disabled title="Requires current-path dry run and approved hardware I/O test">Print & record</button>
        </div>
      </div>
    </div>
  );
}
