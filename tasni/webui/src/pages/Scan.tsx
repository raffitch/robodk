import { useCallback, useEffect, useRef, useState } from "react";
import { moduleApi } from "../api/client";
import { useEvents, type JobEvent } from "../api/events";
import AimHud, { type GateReading } from "./AimHud";
import ScanViewer from "./ScanViewer";
import StreamStats, { useStreamStats } from "./StreamStats";

const api = moduleApi("scan");
const TARGET_PREFIX = "TasniScan_";          // must match service.py scan.target_prefix
const PREVIEW_URL = "/api/modules/scan/preview.bin";

interface ScanConfig {
  robot: string;
  camera_tool: string;
  camera: { ip: string; port: number; resolution: string };
  scan: { pose_count: number; cone_half_angle_deg: number; voxel_size_m: number;
          collision_self_pairs: boolean };
  gate: { ideal_distance_mm: number; distance_tol_mm: number; max_tilt_deg: number };
}
interface Plane {
  frame_T_mm: number[][];
  corners_mm: number[][];
  size_mm: [number, number];
  normal: number[];
  inlier_frac: number;
  inlier_count: number;
}
interface ScanResult {
  kind: "scan";
  run_dir: string;
  can_insert: boolean;
  n_views: number;
  n_points: number;
  mesh_vertices: number;
  mesh_triangles: number;
  stamp?: string;
  plane: Plane;
}
interface TourPose { name: string; reachable: boolean; collision: boolean | null; ok: boolean; transit?: boolean | null; }
interface TourResult {
  kind: "sim_tour"; total: number; passed: number; unreachable: number;
  collisions: number; transit_collisions?: number; collisions_checked: boolean;
  returned_to_start: boolean; all_ok: boolean; poses: TourPose[];
}

export default function Scan() {
  const { subscribe } = useEvents();
  const [config, setConfig] = useState<ScanConfig | null>(null);

  const [conn, setConn] = useState<"idle" | "connecting" | "ready" | "error">("idle");
  const [connInfo, setConnInfo] = useState("");
  const ready = conn === "ready";
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("idle");
  const [pct, setPct] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [frame, setFrame] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  const [live, setLive] = useState(false);
  const [gate, setGate] = useState<GateReading | null>(null);
  const { mark: markFrame, reset: resetStream, stat: streamStat } = useStreamStats();
  const [targets, setTargets] = useState<number | null>(null);
  const [generating, setGenerating] = useState(false);
  const [thumbs, setThumbs] = useState<string[]>([]);   // per-pose captures during a run

  const [result, setResult] = useState<ScanResult | null>(null);
  const [viewerNonce, setViewerNonce] = useState(0);
  const [inserted, setInserted] = useState(false);
  const [tour, setTour] = useState<TourResult | null>(null);

  const [showConfirm, setShowConfirm] = useState(false);
  const [cellClear, setCellClear] = useState(false);
  const runKindRef = useRef<"run" | "tour" | null>(null);
  const setRunKind = (k: "run" | "tour" | null) => { runKindRef.current = k; };
  const logRef = useRef<HTMLDivElement>(null);

  const addLog = (msg: string, err = false) =>
    setLogs((l) => [...l, (err ? "ERROR: " : "") + msg]);

  const loadConfig = useCallback(() => {
    api.get<ScanConfig>("/config").then(setConfig).catch((e) => addLog(e.message, true));
  }, []);
  const refreshJob = useCallback(async () => {
    try {
      const s = await api.get<{ result: ScanResult | null }>("/status");
      if (s.result?.can_insert) {
        setResult(s.result); setViewerNonce((n) => n + 1);
      }
    } catch { /* no prior scan */ }
  }, []);
  const refreshTargets = useCallback(async () => {
    try {
      const r = await api.get<{ targets: string[] }>("/targets");
      const n = r.targets.filter((t) => t.startsWith(TARGET_PREFIX)).length;
      setTargets(n > 0 ? n : null);
    } catch { /* RoboDK not ready */ }
  }, []);

  useEffect(() => { loadConfig(); refreshJob(); }, [loadConfig, refreshJob]);
  useEffect(() => () => { api.post("/live/stop").catch(() => {}); }, []);
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  const connect = useCallback(async () => {
    setConn("connecting");
    setConnInfo("Opening the Tasni station… first load of the 117 MB station can take 1–2 min.");
    try {
      const r = await api.post<{ ready: boolean; tool: string; missing: string[] }>("/connect");
      if (r.ready) {
        setConn("ready");
        setConnInfo(`Ready — robot and the '${r.tool}' camera tool are present.`);
        refreshTargets(); refreshJob();
      } else {
        setConn("error");
        setConnInfo("Station opened but missing: " + r.missing.join(", ")
          + ". Mount the RealSense camera tool in RoboDK.");
      }
    } catch (e: any) { setConn("error"); setConnInfo(e.message); }
  }, [refreshTargets, refreshJob]);

  useEffect(() => {
    return subscribe((ev: JobEvent) => {
      if (ev.type === "progress") {
        const { step, total, message } = ev.payload;
        setPct(total ? Math.round((step / total) * 100) : 0);
        setStatus(`${step}/${total}  ${message}`);
      } else if (ev.type === "log") {
        addLog(ev.payload.message);
      } else if (ev.type === "frame") {
        const src = "data:image/jpeg;base64," + ev.payload.jpeg_b64;
        setFrame(src);
        // During a real run each frame is one pose's capture — keep the strip.
        // Otherwise it's the live aiming stream, so clock its rate/jitter.
        if (runKindRef.current === "run") setThumbs((t) => [...t, src]);
        else markFrame();
      } else if (ev.type === "gate") {
        // Only real readings carry `gates`/`error`; ignore the color-only preview's
        // empty liveness pings so the HUD never shows a misleading SEARCHING.
        const p = ev.payload as GateReading;
        if (p && (p.gates || p.error)) setGate(p);
      } else if (ev.type === "result") {
        if (ev.payload.name === "sim_tour") {
          setTour(ev.payload.result as TourResult);
          setStatus("dry run complete"); setPct(100); setRunning(false); setRunKind(null);
        } else {
          setResult(ev.payload.result as ScanResult);
          setViewerNonce((n) => n + 1); setInserted(false);
          setStatus("done"); setPct(100); setRunning(false); setRunKind(null);
        }
      } else if (ev.type === "error") {
        addLog(ev.payload.message, true); setRunError(ev.payload.message);
        setStatus("error"); setRunning(false); setRunKind(null);
      } else if (ev.type === "status" && ev.payload.status === "cancelled") {
        addLog("cancelled."); setStatus("cancelled"); setRunning(false); setRunKind(null);
      }
    });
  }, [subscribe]);

  // Start (or resume) the smooth color preview. clearGate=true drops stale HUD
  // panels (a fresh "Start camera"); clearGate=false keeps the last depth reading
  // visible (resuming after a Create-targets check, so the operator keeps live
  // video + fps alongside the standoff/tilt guidance).
  const beginLive = async (clearGate: boolean) => {
    resetStream();
    if (clearGate) setGate(null);
    await api.post("/live/start");
    setLive(true);
  };
  const startLive = async () => {
    try {
      await beginLive(true);
      addLog("camera started — jog it to look down at the table, then Create targets.");
    } catch (e: any) { setLive(false); addLog("live: " + e.message, true); }
  };
  const stopLive = async () => {
    try { await api.post("/live/stop"); } catch { /* ignore */ }
    setLive(false); resetStream();
  };
  const generateTargets = async () => {
    setGenerating(true); setRunError(null);
    try {
      const r = await api.post<{ created: number; look_distance_mm: number;
        calibration_on_file: boolean; candidates_collided?: number;
        collisions_checked?: boolean }>("/poses/generate");
      setTargets(r.created); setTour(null);
      const cal = r.calibration_on_file ? "" :
        " ⚠ no calibration on file — the mesh/frame may be off; run Calibration once for accuracy.";
      addLog(`created ${r.created} scan targets (standoff ~${Math.round(r.look_distance_mm)} mm)`
        + (r.collisions_checked && r.candidates_collided
            ? ` (${r.candidates_collided} colliding filtered)` : "")
        + cal + " — inspect them in RoboDK, then Run.");
      // The depth check stopped the server-side preview; resume it so the operator
      // keeps live video + fps, with the green gate panels still shown as confirmation.
      beginLive(false).catch(() => setLive(false));
    } catch (e: any) {
      // The authoritative grab stops the server-side preview and publishes the gate
      // reading (distance/tilt + tilt-fix). Resume the live video so the operator can
      // re-aim, keeping the HUD panels (the last reading) as guidance — then Create
      // targets again to re-check.
      addLog("create targets: " + e.message, true);
      setRunError("Create targets: " + e.message);
      beginLive(false).catch(() => setLive(false));
    } finally { setGenerating(false); }
  };
  const dryRun = async () => {
    setTour(null); setPct(0); setStatus("starting dry run…"); setRunError(null);
    setRunning(true); setRunKind("tour"); setLive(false);
    try { await api.post("/poses/simulate"); }
    catch (e: any) {
      addLog("dry run: " + e.message, true); setRunError("Dry run: " + e.message);
      setRunning(false); setRunKind(null);
    }
  };
  const openRunConfirm = () => { setCellClear(false); setShowConfirm(true); };
  const doRun = async () => {
    setShowConfirm(false);
    setResult(null); setInserted(false); setPct(0); setRunError(null); setThumbs([]);
    setStatus("starting…"); setRunning(true); setRunKind("run"); setLive(false);
    try { await api.post("/run"); }
    catch (e: any) {
      addLog("run: " + e.message, true); setRunError("Run: " + e.message);
      setRunning(false); setRunKind(null);
    }
  };
  const clearPoses = async () => {
    try {
      const r = await api.post<{ cleared: number }>("/poses/clear");
      setTargets(null); setTour(null);
      addLog(`cleared ${r.cleared} scan targets from RoboDK.`);
    } catch (e: any) { addLog("clear: " + e.message, true); }
  };
  const cancel = () => api.post("/cancel").catch(() => {});
  const insert = async () => {
    try {
      const r = await api.post<{ frame: string; rectangle: string; mesh: string | null }>("/insert");
      setInserted(true);
      addLog(`inserted into RoboDK: frame "${r.frame}", rectangle "${r.rectangle}"`
        + (r.mesh ? `, mesh "${r.mesh}"` : "") + ".");
    } catch (e: any) { addLog("insert: " + e.message, true); }
  };

  useEffect(() => {
    if (!showConfirm) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setShowConfirm(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [showConfirm]);

  const g = config?.gate;
  const lamps: [string, boolean | undefined][] = [
    ["DETECT", gate?.gates?.detected],
    ["DISTANCE", gate?.gates?.distance],
    ["ANGLE", gate?.gates?.angle],
  ];
  const pl = result?.plane;

  return (
    <div>
      <h1 className="page-title">📷 Scan</h1>
      <p className="page-sub">3D-scan a work surface → fused mesh + a working frame + rectangle.</p>

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
        <div className="hint">The scan uses the stored camera calibration; it never runs one.
          If none is on file it warns and proceeds (the mesh/frame may be less accurate).</div>
      </div>

      {/* ---- Standoff gate ---------------------------------------------- */}
      <div className="card">
        <h2>Aim at the table</h2>
        <div className="hint" style={{ marginTop: 0, marginBottom: 10 }}>
          Start the camera (smooth color preview, same as calibration) and jog it to look
          down at the surface — ideal standoff
          <b> {g ? g.ideal_distance_mm : 500} ± {g ? g.distance_tol_mm : 150} mm</b>, tilt
          <b> ≤ {g ? g.max_tilt_deg : 35}°</b>. Standoff + tilt are checked when you
          <b> Create targets</b>: it refuses and shows how to fix the tilt (KUKA B/C) if
          you're out of band.
        </div>
        <div className="aim-wrap">
          {frame ? <img className="preview" src={frame} alt="camera" />
                 : <div className="preview" />}
          {/* The HUD panels need depth (slow on Wi-Fi), so they appear from the
              Create-targets check, not live. During smooth aiming we show just the
              video + fps; no misleading "SEARCHING". */}
          {gate && <AimHud gate={gate} />}
          {live && <StreamStats stat={streamStat} />}
          {!live && !gate && <div className="aim-off">camera off — press “Start camera”</div>}
        </div>

        <div className="lamps">
          {lamps.map(([name, on]) => {
            const state = on === undefined ? "unknown" : on ? "on" : "off";
            const glyph = on === undefined ? "·" : on ? "✓" : "✗";
            const word = on === undefined ? "—" : on ? "OK" : "NO";
            return (
              <span key={name} className={"lamp " + state}>
                <span className="glyph">{glyph}</span> {name}
                <span className="lamp-state">{word}</span>
              </span>
            );
          })}
          <span className={"lamp lock " + (gate?.ok ? "on" : "off")}>
            {gate?.ok ? "✓ ● LOCK" : "✗ ○ NO LOCK"}
          </span>
        </div>

        <div className="btn-row">
          {!live
            ? <button onClick={startLive} disabled={running}>Start camera</button>
            : <button className="secondary" onClick={stopLive}>Stop camera</button>}
          <button onClick={generateTargets} disabled={!ready || running || generating || !live}>
            {generating ? "Creating…" : "Create targets"}
          </button>
          {targets != null &&
            <button className="secondary" onClick={clearPoses} disabled={running}>Clear targets</button>}
        </div>
        {targets != null
          ? <div className="ok-text" style={{ marginTop: 8, fontSize: 13 }}>
              ✓ {targets} scan targets created (TasniScan_*). Inspect in RoboDK, then Run below.
            </div>
          : <div className="hint">Aim the camera at the table (live video), then Create targets — it
              checks standoff + tilt (refusing with a fix if out of band) and seeds pose generation
              from the robot's current pose.</div>}
      </div>

      {/* ---- Run -------------------------------------------------------- */}
      <div className="card">
        <h2>Run scan</h2>
        {config && (
          <div className="kv">
            <div className="k">Robot</div><div className="v">{config.robot}</div>
            <div className="k">Camera</div>
            <div className="v">{config.camera.ip}:{config.camera.port} @ {config.camera.resolution}</div>
            <div className="k">Fusion</div>
            <div className="v">TSDF · {Math.round(config.scan.voxel_size_m * 1000)} mm voxel ·
              {config.scan.pose_count} views · cone ±{config.scan.cone_half_angle_deg}°</div>
          </div>
        )}
        <div className="warn-text" style={{ marginTop: 10, fontSize: 12 }}>
          ⚠ Real robot: Run physically moves the KUKA through the created targets. Clear the cell.
        </div>
        <div className="btn-row">
          <button className="secondary" onClick={dryRun} disabled={running || !ready || targets == null}>
            {runKindRef.current === "tour" ? "Simulating…" : "Dry run (simulate)"}
          </button>
          <button onClick={openRunConfirm} disabled={running || !ready || targets == null}>Run scan</button>
          <button className="secondary" onClick={cancel} disabled={!running}>Cancel</button>
        </div>
        {targets == null && <div className="hint">Create targets (above) to enable Run.</div>}

        {runError && (
          <div className="run-error">
            <span className="run-error-tag">ERROR</span>
            <span>{runError}</span>
            <button className="run-error-x" onClick={() => setRunError(null)} aria-label="dismiss error">✕</button>
          </div>
        )}
        {tour && (
          <div className={"tour-result " + (tour.all_ok ? "ok" : "bad")}>
            <div className="tour-head">
              {tour.all_ok ? "✓ Dry run passed" : "⚠ Dry run found issues"} —
              {" "}{tour.passed}/{tour.total} poses OK, return-to-start {tour.returned_to_start ? "ok" : "FAILED"}.
            </div>
            {tour.poses.some((p) => !p.ok) && (
              <div className="tour-bad">
                Problem poses:{" "}
                {tour.poses.filter((p) => !p.ok)
                  .map((p) => `${p.name} (${!p.reachable ? "unreachable" : p.transit ? "transit collision" : "collision"})`)
                  .join(", ")}
              </div>
            )}
          </div>
        )}
        <div className="progress"><div style={{ width: `${pct}%` }} /></div>
        <div className="status-line">{status}</div>

        {thumbs.length > 0 && (
          <div className="thumb-strip">
            {thumbs.map((src, i) => (
              <img key={i} src={src} alt={`pose ${i + 1}`} title={`pose ${i + 1}`} />
            ))}
          </div>
        )}
      </div>

      {/* ---- Review + insert ------------------------------------------- */}
      <div className="card">
        <h2>Review &amp; insert</h2>
        {result ? (
          <>
            <ScanViewer nonce={viewerNonce} src={PREVIEW_URL}
                        frameT={pl?.frame_T_mm} corners={pl?.corners_mm} />
            <div className="kv" style={{ marginTop: 12 }}>
              <div className="k">Work surface</div>
              <div className="v">{Math.round(pl!.size_mm[0])} × {Math.round(pl!.size_mm[1])} mm
                <span className="hint"> (plane inliers {Math.round(pl!.inlier_frac * 100)}%)</span></div>
              <div className="k">Fused</div>
              <div className="v">{result.n_views} views · {result.n_points.toLocaleString()} points ·
                {result.mesh_vertices.toLocaleString()} mesh verts</div>
            </div>
            <div className="hint" style={{ marginTop: 6 }}>
              Orbit/zoom the cloud above. The blue rectangle + axes are the proposed work surface and
              frame. Insert creates them (and the mesh) in RoboDK — nothing is added until you do.
            </div>
            <div className="btn-row">
              <button onClick={insert} disabled={!result.can_insert || inserted}>
                {inserted ? "Inserted ✓" : "Insert into RoboDK"}
              </button>
            </div>
            <div className="hint">Artifacts: <code>{result.run_dir}</code></div>
          </>
        ) : (
          <div className="hint">Run a scan to fuse the surface and preview the proposed frame + rectangle here.</div>
        )}
      </div>

      <div className="card">
        <h2>Log</h2>
        <div className="log" ref={logRef}>
          {logs.map((l, i) => (
            <div key={i} className={l.startsWith("ERROR") ? "err" : ""}>{l}</div>
          ))}
        </div>
      </div>

      {showConfirm && (
        <div className="modal-backdrop" onClick={() => setShowConfirm(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}
               role="dialog" aria-modal="true" aria-labelledby="scan-confirm-title">
            <h2 id="scan-confirm-title">⚠ Move the real robot?</h2>
            <p>Run drives the <b>{config?.robot ?? "KUKA"}</b> through{" "}
              <b>{targets ?? "the generated"}</b> scan targets on the <b>real robot</b>,
              capturing depth at each. It returns to the start pose when finished.</p>
            <div className={"modal-tour " + (tour ? (tour.all_ok ? "ok" : "bad") : "none")}>
              {tour
                ? (tour.all_ok
                    ? `✓ Dry run passed: ${tour.passed}/${tour.total} poses reachable, return-to-start ok.`
                    : `⚠ Dry run found issues: ${tour.passed}/${tour.total} reachable, return-to-start ${tour.returned_to_start ? "ok" : "FAILED"}. Review in RoboDK first.`)
                : "No dry run performed. A dry run (simulate) is strongly recommended first."}
            </div>
            <label className="modal-ack">
              <input type="checkbox" checked={cellClear} onChange={(e) => setCellClear(e.target.checked)} />
              <span>The cell is clear and I am ready to move the real robot.</span>
            </label>
            <div className="btn-row">
              <button onClick={doRun} disabled={!cellClear}>Move robot &amp; scan</button>
              <button className="secondary" autoFocus onClick={() => setShowConfirm(false)}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
