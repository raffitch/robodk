import { useCallback, useEffect, useRef, useState } from "react";
import { moduleApi } from "../api/client";
import { useEvents, type JobEvent } from "../api/events";
import CalibrationGuide from "./CalibrationGuide";

const api = moduleApi("calibration");

interface CalibConfig {
  robot: string;
  run_mode: string;
  target_prefix: string;
  board: { squares_x: number; squares_y: number; square_size_mm: number; marker_size_mm: number; dictionary: string };
  camera: { ip: string; port: number; resolution: string };
  calibration: { settle_s: number; holdout_count: number; refine: boolean; min_charuco_corners: number };
}
interface Split { rms_px: number; max_px: number; n_views: number; }
interface Report {
  refined: boolean;
  train: Split;
  validation: Split | null;
  board_consistency_mm: { rms: number; max: number };
}
interface RunResult {
  summary: string; report: Report; run_dir: string;
  tool_name: string | null; n_captured: number; n_skipped: string[]; can_apply: boolean;
}

const band = (px: number) => (px < 1 ? "good" : px < 3 ? "warn" : "bad");

export default function Calibration() {
  const { subscribe } = useEvents();
  const [config, setConfig] = useState<CalibConfig | null>(null);
  const [tools, setTools] = useState<string[]>([]);
  const [tool, setTool] = useState("");
  const [holdout, setHoldout] = useState(3);
  const [refine, setRefine] = useState(true);

  const [runMode, setRunMode] = useState("run_robot");   // calibration defaults to the real arm
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("idle");
  const [pct, setPct] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [frame, setFrame] = useState<string | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);
  const [canApply, setCanApply] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  const addLog = (msg: string, err = false) =>
    setLogs((l) => [...l, (err ? "ERROR: " : "") + msg]);

  const loadConfig = useCallback(() => {
    api.get<CalibConfig>("/config").then((c) => {
      setConfig(c);
      setHoldout(c.calibration.holdout_count);
      setRefine(c.calibration.refine);
    }).catch((e) => addLog(e.message, true));
  }, []);

  useEffect(() => {
    loadConfig();
    api.get<{ tools: string[] }>("/tools").then((d) => {
      setTools(d.tools);
      if (d.tools.length) setTool(d.tools[0]);
    }).catch((e) => addLog("tools: " + e.message, true));
  }, [loadConfig]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  useEffect(() => {
    return subscribe((ev: JobEvent) => {
      if (ev.type === "progress") {
        const { step, total, message } = ev.payload;
        setPct(total ? Math.round((step / total) * 100) : 0);
        setStatus(`${step}/${total}  ${message}`);
      } else if (ev.type === "log") {
        addLog(ev.payload.message);
      } else if (ev.type === "frame") {
        setFrame("data:image/jpeg;base64," + ev.payload.jpeg_b64);
      } else if (ev.type === "result") {
        setResult(ev.payload.result as RunResult);
        setCanApply(!!ev.payload.result?.can_apply);
        setStatus("done"); setPct(100); setRunning(false);
      } else if (ev.type === "error") {
        addLog(ev.payload.message, true); setStatus("error"); setRunning(false);
      } else if (ev.type === "status" && ev.payload.status === "cancelled") {
        addLog("cancelled."); setStatus("cancelled"); setRunning(false);
      }
    });
  }, [subscribe]);

  const run = async () => {
    if (runMode === "run_robot" &&
        !window.confirm("This will physically move the real robot through every calibration "
          + "pose. Make sure the cell is clear. Continue?")) return;
    setLogs([]); setResult(null); setCanApply(false); setPct(0);
    setStatus("starting…"); setRunning(true);
    try {
      await api.post("/run", { tool_name: tool || null, holdout_count: holdout, refine,
                               run_mode: runMode });
    } catch (e: any) { addLog("run: " + e.message, true); setRunning(false); }
  };
  const cancel = () => api.post("/cancel").catch(() => {});
  const apply = async () => {
    try {
      const r = await api.post<{ tool: string }>("/apply");
      addLog(`applied calibration to tool "${r.tool}".`);
      setCanApply(false);
    } catch (e: any) { addLog("apply: " + e.message, true); }
  };

  return (
    <div>
      <h1 className="page-title">🎯 Calibration</h1>
      <p className="page-sub">ChArUco eye-in-hand hand-eye calibration (TSAI) with quality metrics.</p>

      <div className="calib-layout">
       <div className="calib-main">
      <div className="card">
        <h2>Setup</h2>
        {config && (
          <div className="kv">
            <div className="k">Robot</div><div className="v">{config.robot}</div>
            <div className="k">Camera</div>
            <div className="v">{config.camera.ip}:{config.camera.port} @ {config.camera.resolution}</div>
            <div className="k">Board</div>
            <div className="v">
              {config.board.squares_x}×{config.board.squares_y}, {config.board.square_size_mm}/
              {config.board.marker_size_mm} mm, {config.board.dictionary}
            </div>
          </div>
        )}
        <div className="row" style={{ marginTop: 14 }}>
          <div className="field">
            <label>Tool to calibrate</label>
            <select value={tool} onChange={(e) => setTool(e.target.value)}>
              {tools.length === 0 && <option value="">(connect to RoboDK)</option>}
              {tools.map((t) => <option key={t}>{t}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Robot motion</label>
            <select value={runMode} onChange={(e) => setRunMode(e.target.value)}>
              <option value="run_robot">Real robot</option>
              <option value="simulate">Simulate (dry run)</option>
            </select>
          </div>
          <div className="field">
            <label>Validation poses (held out)</label>
            <input type="number" min={0} max={20} style={{ width: 80 }}
              value={holdout} onChange={(e) => setHoldout(parseInt(e.target.value, 10) || 0)} />
          </div>
          <div className="field">
            <label><input type="checkbox" checked={refine}
              onChange={(e) => setRefine(e.target.checked)} /> Reprojection refinement</label>
          </div>
        </div>
        {runMode === "run_robot"
          ? <div className="warn-text" style={{ marginTop: 8, fontSize: 12 }}>
              ⚠ Real robot: Run will physically move the KUKA through every pose. Clear the cell.
            </div>
          : <div className="hint">Simulate moves the robot in RoboDK only — the camera view
              won't change, so use this only for a dry run of the flow, not a real calibration.</div>}
        <div className="hint">
          Held-out poses validate the fit on data the solver never saw. Typical: ~12–20 total
          poses, hold out 3–5. (Need at least holdout + 3 to solve.)
        </div>
        <div className="btn-row">
          <button onClick={run} disabled={running}>Run calibration</button>
          <button className="secondary" onClick={cancel} disabled={!running}>Cancel</button>
        </div>
        <div className="hint">
          Drives the robot through every <code>{config?.target_prefix ?? "Target"}*</code> pose,
          detects the ChArUco board, solves TSAI, then reports quality. Nothing is written to the
          tool until you review the metrics and click Apply.
        </div>
      </div>

      <div className="card">
        <h2>Live preview</h2>
        {frame ? <img className="preview" src={frame} alt="camera preview" />
               : <div className="preview" />}
        <div className="progress"><div style={{ width: `${pct}%` }} /></div>
        <div className="status-line">{status}</div>
      </div>

      <div className="card">
        <h2>Quality metrics</h2>
        {result ? <Metrics result={result} /> :
          <div className="hint">Run a calibration to see reprojection and held-out validation errors.</div>}
        <div className="btn-row">
          <button onClick={apply} disabled={!canApply}>Apply to tool</button>
        </div>
      </div>

      <div className="card">
        <h2>Log</h2>
        <div className="log" ref={logRef}>
          {logs.map((l, i) => (
            <div key={i} className={l.startsWith("ERROR") ? "err" : ""}>{l}</div>
          ))}
        </div>
      </div>
       </div>
       <CalibrationGuide runMode={runMode} onConfigChanged={loadConfig} />
      </div>
    </div>
  );
}

function Metrics({ result }: { result: RunResult }) {
  const r = result.report;
  const rows: [string, string, string][] = [
    ["Solver", "TSAI" + (r.refined ? " + reprojection refinement" : ""), ""],
    [`Train fit (${r.train.n_views} poses)`,
      `RMS ${r.train.rms_px.toFixed(3)} px · max ${r.train.max_px.toFixed(3)} px`, band(r.train.rms_px)],
  ];
  if (r.validation)
    rows.push([`Held-out validation (${r.validation.n_views} poses)`,
      `RMS ${r.validation.rms_px.toFixed(3)} px · max ${r.validation.max_px.toFixed(3)} px`,
      band(r.validation.rms_px)]);
  rows.push(["Board consistency",
    `RMS ${r.board_consistency_mm.rms.toFixed(3)} mm · max ${r.board_consistency_mm.max.toFixed(3)} mm`, ""]);

  return (
    <>
      <table className="metrics"><tbody>
        {rows.map(([k, v, b], i) => (
          <tr key={i}>
            <th>{k}</th>
            <td className="num">{v}{b && <> <span className={`badge ${b}`}>{b}</span></>}</td>
          </tr>
        ))}
      </tbody></table>
      {result.n_skipped.length > 0 &&
        <div className="hint">Skipped (no board): {result.n_skipped.join(", ")}</div>}
      <div className="hint">Artifacts: <code>{result.run_dir}</code></div>
    </>
  );
}
