import { useCallback, useEffect, useRef, useState } from "react";
import { moduleApi } from "../api/client";
import { useEvents, type JobEvent } from "../api/events";
import AimHud, { type GateReading } from "./AimHud";
import CalibrationGuide from "./CalibrationGuide";
import ConeDiagram from "./ConeDiagram";

const api = moduleApi("calibration");

interface CalibConfig {
  robot: string;
  camera_tool: string;
  board: { squares_x: number; squares_y: number; square_size_mm: number; marker_size_mm: number; dictionary: string };
  camera: { ip: string; port: number; resolution: string };
  calibration: { holdout_count: number; refine: boolean; pose_count: number;
                 cone_half_angle_deg: number; roll_max_deg: number; distance_jitter: number };
  gate: { ideal_distance_mm: number; distance_tol_mm: number; max_tilt_deg: number };
}
interface Split { rms_px: number; max_px: number; n_views: number; }
interface Report {
  refined: boolean;
  method: string;
  train: Split;
  validation: Split | null;
  board_consistency_mm: { rms: number; max: number };
  motion_diversity?: { axis_spread: number; min_pair_deg: number; max_pair_deg: number; well_conditioned: boolean; note?: string };
  intrinsics_check?: { warn: boolean; note: string } | null;
  cross_val_rms_px?: number | null;
}
interface RunResult {
  summary: string;
  report?: Report;
  run_dir?: string;
  tool_name?: string;
  n_captured?: number;
  n_skipped?: string[];
  can_apply: boolean;
}

const band = (px: number) => (px < 1 ? "good" : px < 3 ? "warn" : "bad");

export default function Calibration() {
  const { subscribe } = useEvents();
  const [config, setConfig] = useState<CalibConfig | null>(null);
  const [holdout, setHoldout] = useState(3);
  const [refine, setRefine] = useState(true);

  const [conn, setConn] = useState<"idle" | "connecting" | "ready" | "error">("idle");
  const [connInfo, setConnInfo] = useState("");
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("idle");
  const ready = conn === "ready";
  const [pct, setPct] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [frame, setFrame] = useState<string | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);
  const [canApply, setCanApply] = useState(false);

  const [live, setLive] = useState(false);
  const [gate, setGate] = useState<GateReading | null>(null);
  const [targets, setTargets] = useState<number | null>(null);   // null = none created
  const [generating, setGenerating] = useState(false);
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

  useEffect(() => { loadConfig(); }, [loadConfig]);

  // Stop the live gate if we leave the page (frees the unicast camera).
  useEffect(() => () => { api.post("/live/stop").catch(() => {}); }, []);

  const connect = useCallback(async () => {
    setConn("connecting");
    setConnInfo("Opening the Tasni station… first load of the 117 MB station can take 1–2 min.");
    try {
      const r = await api.post<{ ready: boolean; tool: string; missing: string[] }>("/connect");
      if (r.ready) {
        setConn("ready");
        setConnInfo(`Ready — robot and the '${r.tool}' camera tool are present.`);
      } else {
        setConn("error");
        setConnInfo("Station opened but missing: " + r.missing.join(", ")
          + ". Mount the RealSense camera tool in RoboDK.");
      }
    } catch (e: any) {
      setConn("error");
      setConnInfo(e.message);
    }
  }, []);

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
      } else if (ev.type === "gate") {
        setGate(ev.payload as GateReading);
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

  const startLive = async () => {
    try {
      await api.post("/live/start");
      setLive(true);
      addLog("live aiming gate started — jog the robot until all lamps are green.");
    } catch (e: any) { addLog("live: " + e.message, true); }
  };
  const stopLive = async () => {
    try { await api.post("/live/stop"); } catch { /* ignore */ }
    setLive(false);
  };

  const generateTargets = async () => {
    setGenerating(true);
    try {
      const r = await api.post<{ created: number; look_distance_mm: number }>("/poses/generate");
      setTargets(r.created);
      setLive(false);   // generate stops the live gate server-side
      addLog(`created ${r.created} targets (working distance ~${Math.round(r.look_distance_mm)} mm)`
        + " — inspect them in RoboDK, then Run calibration.");
    } catch (e: any) { addLog("create targets: " + e.message, true); }
    finally { setGenerating(false); }
  };

  const run = async () => {
    if (!window.confirm("This will physically move the real robot through the "
        + (targets ?? "generated") + " calibration targets. Make sure the cell is clear. Continue?")) return;
    setLogs([]); setResult(null); setCanApply(false); setPct(0);
    setStatus("starting…"); setRunning(true); setLive(false);
    try {
      await api.post("/run", { holdout_count: holdout, refine });
    } catch (e: any) { addLog("run: " + e.message, true); setRunning(false); }
  };
  const clearPoses = async () => {
    try {
      const r = await api.post<{ cleared: number }>("/poses/clear");
      setTargets(null);
      addLog(`cleared ${r.cleared} generated targets from RoboDK.`);
    } catch (e: any) { addLog("clear: " + e.message, true); }
  };
  const cancel = () => api.post("/cancel").catch(() => {});
  const apply = async () => {
    try {
      const r = await api.post<{ tool: string }>("/apply");
      addLog(`applied calibration to tool "${r.tool}".`);
      setCanApply(false);
    } catch (e: any) { addLog("apply: " + e.message, true); }
  };

  const g = config?.gate;
  const lamps: [string, boolean | undefined][] = [
    ["DETECT", gate?.gates?.detected],
    ["DISTANCE", gate?.gates?.distance],
    ["ANGLE", gate?.gates?.angle],
  ];

  return (
    <div>
      <h1 className="page-title">🎯 Calibration</h1>
      <p className="page-sub">ChArUco eye-in-hand hand-eye calibration (TSAI) with quality metrics.</p>

      <div className="calib-layout">
       <div className="calib-main">

      <div className={"card conn-banner " + conn}>
        <div className="conn-row">
          <span className={"dot " + (ready ? "ok" : conn === "error" ? "bad" : "unknown")} />
          <span className="conn-label">
            {conn === "idle" && "Not connected to RoboDK"}
            {conn === "connecting" && "Connecting…"}
            {conn === "ready" && "Connected — cell ready"}
            {conn === "error" && "Connection problem"}
          </span>
          <button onClick={connect} disabled={conn === "connecting"} style={{ marginLeft: "auto" }}>
            {ready ? "Reconnect" : "Connect & open Tasni station"}
          </button>
        </div>
        {connInfo && <div className="hint">{connInfo}</div>}
        {!ready && conn !== "connecting" &&
          <div className="hint">Connecting opens RoboDK (if needed) and checks the robot + camera tool.
            You jog the robot yourself; creating targets stays locked until the gate is green.</div>}
      </div>

      {/* ---- Aiming gate ------------------------------------------------- */}
      <div className="card">
        <h2>Aim the camera</h2>
        <div className="hint" style={{ marginTop: 0, marginBottom: 10 }}>
          Start the camera and jog the robot until the board sits in the green band — ideal range
          <b> {g ? g.ideal_distance_mm : 450} ± {g ? g.distance_tol_mm : 80} mm</b>, tilt
          <b> ≤ {g ? g.max_tilt_deg : 25}°</b>. When all three lamps lock green, Create targets unlocks.
        </div>
        <div className="aim-wrap">
          {frame ? <img className="preview" src={frame} alt="camera" />
                 : <div className="preview" />}
          {live && <AimHud gate={gate} />}
          {!live && <div className="aim-off">camera off — press “Start camera”</div>}
        </div>

        <div className="lamps">
          {lamps.map(([name, on]) => (
            <span key={name} className={"lamp " + (on ? "on" : "off")}>
              <span className="dot" /> {name}
            </span>
          ))}
          <span className={"lamp lock " + (gate?.ok ? "on" : "off")}>
            {gate?.ok ? "● LOCK" : "○ NO LOCK"}
          </span>
        </div>

        <div className="btn-row">
          {!live
            ? <button onClick={startLive} disabled={running}>Start camera</button>
            : <button className="secondary" onClick={stopLive}>Stop camera</button>}
          <button onClick={generateTargets}
                  disabled={!ready || running || generating || !gate?.ok}>
            {generating ? "Creating…" : "Create targets"}
          </button>
          {targets != null &&
            <button className="secondary" onClick={clearPoses} disabled={running}>Clear targets</button>}
        </div>
        {targets != null
          ? <div className="ok-text" style={{ marginTop: 8, fontSize: 13 }}>
              ✓ {targets} targets created (TasniCalib_*). Inspect them in RoboDK, then Run calibration below.
            </div>
          : <div className="hint">Create targets needs the connection ready <i>and</i> a green lock.
              The robot's current pose becomes the seed; nothing is created until then.</div>}
      </div>

      {/* ---- How targets are generated --------------------------------- */}
      {config && (
        <div className="card">
          <h2>Target spread</h2>
          <div className="hint" style={{ marginTop: 0, marginBottom: 6 }}>
            Create targets orbits your aimed view in a cone (so the board stays in frame),
            with roll and distance variation — the rotational diversity the hand-eye solve
            needs. A wider cone gives a better-conditioned solve.
          </div>
          <ConeDiagram coneDeg={config.calibration.cone_half_angle_deg}
                       count={config.calibration.pose_count} />
          <div className="hint" style={{ marginTop: 6 }}>
            {config.calibration.pose_count} viewpoints · cone ±{config.calibration.cone_half_angle_deg}° ·
            roll ±{config.calibration.roll_max_deg}° · distance ±{Math.round(config.calibration.distance_jitter * 100)}%.
            Schematic only — the actual poses are reachability-filtered; inspect them in RoboDK.
          </div>
        </div>
      )}

      {/* ---- Run -------------------------------------------------------- */}
      <div className="card">
        <h2>Run calibration</h2>
        {config && (
          <div className="kv">
            <div className="k">Robot</div><div className="v">{config.robot}</div>
            <div className="k">Camera tool</div>
            <div className="v">{config.camera_tool} <span className="hint">(RealSense, fixed)</span></div>
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
            <label>Validation poses (held out)</label>
            <input type="number" min={0} max={20} style={{ width: 80 }}
              value={holdout} onChange={(e) => setHoldout(parseInt(e.target.value, 10) || 0)} />
          </div>
          <div className="field">
            <label><input type="checkbox" checked={refine}
              onChange={(e) => setRefine(e.target.checked)} /> Reprojection refinement</label>
          </div>
        </div>
        <div className="warn-text" style={{ marginTop: 8, fontSize: 12 }}>
          ⚠ Real robot: Run physically moves the KUKA through the created targets. Clear the cell.
        </div>
        <div className="btn-row">
          <button onClick={run} disabled={running || !ready || targets == null}>Run calibration</button>
          <button className="secondary" onClick={cancel} disabled={!running}>Cancel</button>
        </div>
        {targets == null && <div className="hint">Create targets (above) to enable Run.</div>}
        <div className="progress"><div style={{ width: `${pct}%` }} /></div>
        <div className="status-line">{status}</div>
      </div>

      <div className="card">
        <h2>Quality metrics</h2>
        {result?.report ? <Metrics result={result} />
          : result ? <div className="ok-text">{result.summary}</div>
          : <div className="hint">Run a calibration to see reprojection and held-out validation errors.</div>}
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
       <CalibrationGuide ready={ready} connState={conn} onConnect={connect} />
      </div>
    </div>
  );
}

function Metrics({ result }: { result: RunResult }) {
  const r = result.report;
  if (!r) return null;
  const rows: [string, string, string][] = [
    ["Solver", r.method + (r.refined ? " + reprojection refinement" : ""), ""],
    [`Train fit (${r.train.n_views} poses)`,
      `RMS ${r.train.rms_px.toFixed(3)} px · max ${r.train.max_px.toFixed(3)} px`, band(r.train.rms_px)],
  ];
  if (r.validation)
    rows.push([`Held-out validation (${r.validation.n_views} poses)`,
      `RMS ${r.validation.rms_px.toFixed(3)} px · max ${r.validation.max_px.toFixed(3)} px`,
      band(r.validation.rms_px)]);
  if (r.cross_val_rms_px != null)
    rows.push(["Cross-validation (k-fold)",
      `RMS ${r.cross_val_rms_px.toFixed(3)} px`, band(r.cross_val_rms_px)]);
  rows.push(["Board consistency",
    `RMS ${r.board_consistency_mm.rms.toFixed(3)} mm · max ${r.board_consistency_mm.max.toFixed(3)} mm`, ""]);
  if (r.motion_diversity)
    rows.push(["Motion diversity",
      `axis-spread ${r.motion_diversity.axis_spread.toFixed(2)} · rot ${Math.round(r.motion_diversity.min_pair_deg)}–${Math.round(r.motion_diversity.max_pair_deg)}°`,
      r.motion_diversity.well_conditioned ? "good" : "warn"]);
  if (r.intrinsics_check)
    rows.push(["Intrinsics check", r.intrinsics_check.note,
      r.intrinsics_check.warn ? "warn" : "good"]);

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
      {result.n_skipped && result.n_skipped.length > 0 &&
        <div className="hint">Skipped (no board): {result.n_skipped.join(", ")}</div>}
      <div className="hint">Artifacts: <code>{result.run_dir}</code></div>
    </>
  );
}
