import { useCallback, useEffect, useRef, useState } from "react";
import { apiGet, moduleApi } from "../api/client";
import { useEvents, type JobEvent } from "../api/events";
import AimHud, { type GateReading } from "./AimHud";
import { robotLinkNote } from "./Calibration";
import CollisionPanel, { type CollisionStatus } from "../components/CollisionPanel";
import ScanViewer from "./ScanViewer";
import StreamStats, { useStreamStats } from "./StreamStats";

const api = moduleApi("scan");
const TARGET_PREFIX = "TasniScan_";          // must match service.py scan.target_prefix
const PREVIEW_URL = "/api/modules/scan/preview.bin";
const STABLE_LOCK_MS = 1000;
const GATE_FRESH_MS = 1600;
// At ~1.5-2 fps a single noisy frame (one gate dipping below tolerance, or one
// slightly late frame) must NOT tear down a held "Surface ready": only a sustained
// loss of validity longer than this breaks the 1 s streak. Bridges ~one slow frame,
// so the readout stops bouncing "Surface ready" <-> "Hold position" on sensor noise.
const GATE_GRACE_MS = 1000;

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
  kind?: "scan";
  run_dir: string;
  can_insert: boolean;
  mode?: string;           // "quality" | "reference"
  n_views: number;
  n_points: number;
  mesh_vertices: number;
  mesh_triangles: number;
  stamp?: string;
  plane: Plane;
}
interface TourPose { name: string; reachable: boolean; collision: boolean | null; ok: boolean; transit?: boolean | null; collision_pairs?: string[] | null; }
interface TourResult {
  kind: "sim_tour"; total: number; passed: number; unreachable: number;
  collisions: number; transit_collisions?: number; collisions_checked: boolean;
  returned_to_start: boolean; all_ok: boolean; poses: TourPose[];
}
interface RdkStatus {
  connected: boolean;
  ready: boolean;
  tool: string;
  missing: string[];
  robot_link?: { connected: boolean; message: string; ip: string; configured: boolean } | null;
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
  const liveRef = useRef(false);
  const [gate, setGate] = useState<GateReading | null>(null);
  // Coverage accumulation: each live frame the RealSense reports a slightly
  // different set of valid-depth points (stereo dropouts at edges/low texture), so
  // a single frame under-shows coverage. We union the detected-surface dots over the
  // last COVERAGE_FRAMES live frames (deduped to a fine grid) so the whole board
  // fills in and a remaining gap is a true hole. Reset when the camera clearly moves.
  const COVERAGE_FRAMES = 18;
  const coverageRef = useRef<Array<Array<[number, number]>>>([]);
  const coverageCenterRef = useRef<[number, number] | null>(null);
  const [coverageDots, setCoverageDots] = useState<Array<[number, number]> | null>(null);
  const gateReceivedAtRef = useRef(0);
  const stableSinceRef = useRef<number | null>(null);
  const lastValidRef = useRef<number | null>(null);
  const [surfaceStable, setSurfaceStable] = useState(false);
  const [stableProgress, setStableProgress] = useState(0);
  const [surfaceLocked, setSurfaceLocked] = useState(false);
  const [locking, setLocking] = useState(false);
  const { mark: markFrame, reset: resetStream, stat: streamStat } = useStreamStats();
  const [targets, setTargets] = useState<number | null>(null);
  const [scanMode, setScanMode] = useState<"quality" | "reference" | null>(null);
  const [generating, setGenerating] = useState(false);
  const [thumbs, setThumbs] = useState<string[]>([]);   // per-pose captures during a run
  const [collision, setCollision] = useState<CollisionStatus | null>(null);
  const [collisionBusy, setCollisionBusy] = useState(false);
  const [recentCollisionPairs, setRecentCollisionPairs] = useState<string[]>([]);

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
  const hydrateConnection = useCallback(async () => {
    try {
      const r = await apiGet<RdkStatus>("/api/rdk/status");
      if (r.ready) {
        setConn("ready");
        setConnInfo(`Ready — robot and the '${r.tool}' camera tool are present.`
          + robotLinkNote(r.robot_link));
        refreshTargets(); refreshJob();
      }
    } catch { /* status hydration is opportunistic */ }
  }, [refreshTargets, refreshJob]);
  const checkCollision = useCallback(async () => {
    setCollisionBusy(true);
    try {
      const r = await api.get<CollisionStatus>("/collision/status");
      setCollision(r);
    } catch { setCollision(null); }
    finally { setCollisionBusy(false); }
  }, []);
  const ignoreCollisionPair = useCallback(async (pair: string) => {
    const r = await api.post<CollisionStatus>("/collision/ignore", { pair });
    setCollision(r);
    addLog(`ignored collision pair: ${pair}`);
  }, []);

  useEffect(() => { loadConfig(); refreshJob(); hydrateConnection(); },
            [loadConfig, refreshJob, hydrateConnection]);
  useEffect(() => { liveRef.current = live; }, [live]);
  useEffect(() => () => {
    if (liveRef.current) sessionStorage.setItem("tasni:autoStartCamera", "calibration");
    api.post("/live/stop").catch(() => {});
  }, []);
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  const connect = useCallback(async () => {
    setConn("connecting");
    setConnInfo("Opening the Tasni station… first load of the 117 MB station can take 1–2 min.");
    try {
      const r = await api.post<{ ready: boolean; tool: string; missing: string[];
        robot_link?: { connected: boolean; message: string; ip: string;
                       configured: boolean } | null }>("/connect");
      if (r.ready) {
        setConn("ready");
        setConnInfo(`Ready — robot and the '${r.tool}' camera tool are present.`
          + robotLinkNote(r.robot_link));
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
        // Accept compact live depth-plane telemetry and authoritative Create-targets
        // readings. Never replace valid guidance with a color transport error.
        const p = ev.payload as GateReading;
        if (p?.gates && !p.error) {
          gateReceivedAtRef.current = performance.now();
          setGate(p);
          // Accumulate the live aiming stream. On the authoritative (non-live) lock
          // snapshot we deliberately do NOT reset: a single frame's depth lands only
          // where the surface has texture (edges/low-texture drop out), so its dots
          // collapse toward the centre. Keeping the accumulated multi-frame union
          // frozen shows the real coverage the operator just saw. It is reset on the
          // next Start camera / Reposition / Create-targets (beginLive/stopLive).
          if (p.live && Array.isArray(p.points_uv) && p.points_uv.length) {
            accumulateCoverage(p.points_uv as Array<[number, number]>);
          }
        }
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

  useEffect(() => {
    const id = window.setInterval(() => {
      const now = performance.now();
      const fresh = now - gateReceivedAtRef.current <= GATE_FRESH_MS;
      const valid = !!(live && fresh && gate?.ok);
      if (valid) lastValidRef.current = now;
      // Debounce the lock streak: tolerate brief single-frame dips so the readout
      // does not bounce on sensor noise at low fps. Only a real loss of the pose
      // (invalid for longer than GATE_GRACE_MS) resets the 1 s "ready" timer.
      const recentlyValid = lastValidRef.current != null
        && now - lastValidRef.current <= GATE_GRACE_MS;
      if (!valid && !recentlyValid) {
        stableSinceRef.current = null;
        setStableProgress(0);
        setSurfaceStable(false);
        return;
      }
      if (stableSinceRef.current == null) stableSinceRef.current = now;
      const elapsed = now - stableSinceRef.current;
      setStableProgress(Math.min(1, elapsed / STABLE_LOCK_MS));
      setSurfaceStable(elapsed >= STABLE_LOCK_MS);
    }, 100);
    return () => window.clearInterval(id);
  }, [live, gate?.ok]);

  const resetCoverage = () => {
    coverageRef.current = [];
    coverageCenterRef.current = null;
    setCoverageDots(null);
  };

  // Fold one live frame's detected-surface dots into the rolling coverage union.
  const accumulateCoverage = (pts: Array<[number, number]>) => {
    if (!pts.length) return;
    let sx = 0, sy = 0;
    for (const [u, v] of pts) { sx += u; sy += v; }
    const center: [number, number] = [sx / pts.length, sy / pts.length];
    const last = coverageCenterRef.current;
    // A clear camera/surface move invalidates the accumulated dots (they were
    // anchored to the old view) — start fresh so coverage never smears.
    if (last && Math.hypot(center[0] - last[0], center[1] - last[1]) > 0.035) {
      coverageRef.current = [];
    }
    coverageCenterRef.current = center;
    const buf = coverageRef.current;
    buf.push(pts);
    while (buf.length > COVERAGE_FRAMES) buf.shift();
    // Union, snapped to a ~1/180 grid so repeated hits dedupe and the count stays
    // bounded (and renderable) while still resolving board-edge holes.
    const seen = new Set<string>();
    const union: Array<[number, number]> = [];
    for (const f of buf) {
      for (const [u, v] of f) {
        const key = `${Math.round(u * 180)},${Math.round(v * 180)}`;
        if (!seen.has(key)) { seen.add(key); union.push([u, v]); }
      }
    }
    setCoverageDots(union);
  };

  // Start (or resume) the smooth color preview. clearGate=true drops stale HUD
  // panels (a fresh "Start camera"); clearGate=false keeps the last depth reading
  // visible (resuming after a Create-targets check, so the operator keeps live
  // video + fps alongside the standoff/tilt guidance).
  const beginLive = async (clearGate: boolean) => {
    resetStream();
    resetCoverage();
    if (clearGate) {
      setGate(null);
      gateReceivedAtRef.current = 0;
      setSurfaceLocked(false);
    }
    stableSinceRef.current = null;
    lastValidRef.current = null;
    setSurfaceStable(false);
    setStableProgress(0);
    await api.post("/live/start");
    setLive(true);
  };
  const startLive = async () => {
    try {
      await beginLive(true);
      addLog("camera started — jog until the surface guidance is stable, then lock it.");
    } catch (e: any) { setLive(false); addLog("live: " + e.message, true); }
  };
  useEffect(() => {
    if (!ready || live || running) return;
    if (sessionStorage.getItem("tasni:autoStartCamera") !== "scan") return;
    sessionStorage.removeItem("tasni:autoStartCamera");
    startLive();
  }, [ready, live, running]);
  const stopLive = async () => {
    try { await api.post("/live/stop"); } catch { /* ignore */ }
    setLive(false); resetStream(); resetCoverage();
  };
  const lockSurface = async () => {
    setLocking(true); setRunError(null);
    try {
      const r = await api.post<{
        status: string; gate: GateReading;
        surface_mode: "full" | "crop";
        extent_mm?: [number, number] | null;
        crop_size_mm?: [number, number] | null;
      }>("/surface/lock");
      setLive(false); resetStream();
      setGate(r.gate);
      setSurfaceLocked(true);
      setSurfaceStable(false);
      addLog(r.surface_mode === "crop" && r.crop_size_mm
        ? `surface locked — review generic ${Math.round(r.crop_size_mm[0])} × ${Math.round(r.crop_size_mm[1])} mm work area (surface overruns the view)`
        : `surface locked — review full detected platform${
            r.extent_mm ? ` ${Math.round(r.extent_mm[0])} × ${Math.round(r.extent_mm[1])} mm` : ""}`);
    } catch (e: any) {
      addLog("lock surface: " + e.message, true);
      setRunError("Lock surface: " + e.message);
      beginLive(false).catch(() => setLive(false));
    } finally {
      setLocking(false);
    }
  };
  const repositionSurface = async () => {
    try { await api.post("/surface/unlock"); } catch { /* best effort */ }
    setSurfaceLocked(false);
    setGate(null);
    await beginLive(true);
  };
  const generateTargets = async () => {
    setGenerating(true); setRunError(null);
    try {
      const r = await api.post<{
        created: number;
        mode?: string;
        look_distance_mm?: number;
        extent_mm?: [number, number];
        voxel_size_m?: number;
        calibration_on_file: boolean;
        candidates_collided?: number;
        collisions_checked?: boolean;
        collision_filter_bypassed?: boolean;
        collision_pairs?: string[];
        can_insert?: boolean;
      }>("/poses/generate");
      const mode = (r.mode ?? "quality") as "quality" | "reference";
      setScanMode(mode);
      setSurfaceLocked(false);
      setTargets(r.created > 0 ? r.created : null);
      setTour(null);
      setRecentCollisionPairs(r.collision_pairs ?? []);
      checkCollision();

      if (mode === "reference") {
        const extTxt = r.extent_mm
          ? `${Math.round(r.extent_mm[0])} × ${Math.round(r.extent_mm[1])} mm`
          : "unknown size";
        addLog(
          `reference surface: ${extTxt} — too large / far for a quality scan tour. ` +
          "A reference rectangle was placed directly. Review below, then Insert."
        );
        // Fetch the ready reference result and show it in the Review section.
        try {
          const res = await api.get<ScanResult>("/result");
          setResult({ ...res, can_insert: true } as ScanResult);
          setViewerNonce((n) => n + 1); setInserted(false);
        } catch { /* not critical — user can re-check */ }
      } else {
        const cal = r.calibration_on_file ? "" :
          " ⚠ no calibration on file — the mesh/frame may be off; run Calibration once for accuracy.";
        addLog(
          `created ${r.created} scan targets` +
          (r.look_distance_mm != null ? ` (standoff ~${Math.round(r.look_distance_mm)} mm)` : "") +
          (r.extent_mm ? ` — surface ${Math.round(r.extent_mm[0])} × ${Math.round(r.extent_mm[1])} mm` : "") +
          (r.collision_filter_bypassed
            ? ` ⚠ collision filter bypassed after ${r.candidates_collided ?? 0} reported collisions; inspect/dry-run`
            : r.collisions_checked && r.candidates_collided
              ? ` (${r.candidates_collided} colliding filtered)` : "")
          + cal + " — inspect in RoboDK, then Run."
        );
      }
      // Resume smooth preview so the operator keeps live video + fps alongside the HUD.
      beginLive(false).catch(() => setLive(false));
    } catch (e: any) {
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
  const crop = gate?.crop_size_mm;
  const surfaceDescription = gate?.surface_mode === "crop"
    ? crop
      ? `Surface overruns view — generic ${Math.round(crop[0])} × ${Math.round(crop[1])} mm work area on the reticle`
      : "Surface overruns view — a generic work area will be projected on the reticle"
    : gate?.fully_framed === false
      ? "Full surface detected — move toward the recommended distance to include every edge"
    : gate?.fully_framed === true && gate.extent_mm
      ? `Full surface ${Math.round(gate.extent_mm[0])} × ${Math.round(gate.extent_mm[1])} mm`
      : "Aim the center reticle at the intended work surface";
  const lamps: [string, boolean | undefined][] = [
    ["DETECT", gate?.gates?.detected],
    ["DISTANCE", gate?.gates?.distance],
    ["ANGLE", gate?.gates?.angle],
    ["CENTER", gate?.gates?.center],
    ["EDGE A", gate?.gates?.edge],
    ["FRAMED", gate?.gates?.framed],
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

      {/* ---- Survey gate ------------------------------------------------ */}
      <div className="card">
        <h2>Survey the surface</h2>
        <div className="hint" style={{ marginTop: 0, marginBottom: 10 }}>
          Start the camera and jog the robot using the live range and tilt guidance.
          Hold a valid pose for one second, then <b>Lock surface &amp; create targets</b>.
          A fully visible platform (clear edges) uses its measured boundary; otherwise
          a generic 1 m work square is projected around the center reticle.
        </div>
        <div className="aim-wrap">
          {frame ? <img className="preview" src={frame} alt="camera" />
                 : <div className="preview" />}
          {/* Video/FPS and compact depth-plane telemetry use separate channels fed by
              one RealSense capture loop, so the guidance does not interrupt video. */}
          {/* Pass the accumulated coverage union whenever it exists — including the
              frozen union on the locked snapshot (live=false), so the locked dots show
              the real multi-frame coverage, not a sparse single-frame set. It is null
              once the camera is stopped / restarted (resetCoverage). */}
          {(live || gate) && <AimHud gate={gate} mode="scan"
                                     coverageDots={coverageDots} />}
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
          <span className={"lamp lock " + (surfaceStable ? "on" : gate?.ok ? "unknown" : "off")}>
            {surfaceStable ? "✓ ● SURFACE READY"
              : gate?.ok ? `◌ HOLD ${Math.round(stableProgress * 100)}%`
              : "✗ ○ POSITION"}
          </span>
        </div>

        <div className={"scan-ready " + (surfaceLocked || surfaceStable ? "ready" : gate?.ok ? "holding" : "")}>
          <span>{surfaceLocked ? "Surface locked — review region"
            : surfaceStable ? "Surface ready" : gate?.ok ? "Hold position…" : "Position surface"}</span>
          <span>{surfaceDescription}</span>
        </div>

        <CollisionPanel ready={ready} busy={collisionBusy} status={collision}
                        onRecheck={checkCollision}
                        onIgnore={ignoreCollisionPair}
                        recentPairs={recentCollisionPairs} />

        <div className="btn-row">
          {!live && !surfaceLocked
            ? <button onClick={startLive} disabled={running}>Start camera</button>
            : live
              ? <button className="secondary" onClick={stopLive}>Stop camera</button>
              : null}
          {!surfaceLocked
            ? <button onClick={lockSurface}
                      disabled={!ready || running || locking || !live || !surfaceStable}>
                {locking ? "Locking…" : "Lock surface"}
              </button>
            : <>
                <button className="secondary" onClick={repositionSurface}
                        disabled={running || generating}>Reposition</button>
                <button onClick={generateTargets}
                        disabled={!ready || running || generating}>
                  {generating ? "Creating…" : "Accept region & create targets"}
                </button>
              </>}
          {targets != null &&
            <button className="secondary" onClick={clearPoses} disabled={running}>Clear targets</button>}
        </div>
        {targets != null
          ? <div className="ok-text" style={{ marginTop: 8, fontSize: 13 }}>
              ✓ {targets} scan targets created (TasniScan_*). Inspect in RoboDK, then Run below.
            </div>
          : scanMode === "reference"
          ? <div className="ok-text" style={{ marginTop: 8, fontSize: 13 }}>
              ✓ Reference surface detected — rectangle placed directly. Review &amp; Insert below.
              Re-aim closer (300–800 mm) for a quality mesh tour.
            </div>
          : surfaceLocked
          ? <div className="hint">Review the frozen selected region. Reposition if the wrong
              plane or crop is highlighted; otherwise accept it to create the robot targets.</div>
          : <div className="hint">Jog until RANGE and ANGLE are valid and remain stable for
              one second. FRAMED may be red when the surface overruns the view; the scan
              will use the displayed generic 1 m work square instead.</div>}
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
                  .map((p) => {
                    const kind = !p.reachable ? "unreachable" : p.transit ? "transit collision" : "collision";
                    const pairs = p.collision_pairs?.length ? `: ${p.collision_pairs.slice(0, 2).join("; ")}` : "";
                    return `${p.name} (${kind}${pairs})`;
                  })
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
            {result.mode !== "reference" && (
              <ScanViewer nonce={viewerNonce} src={PREVIEW_URL}
                          frameT={pl?.frame_T_mm} corners={pl?.corners_mm} />
            )}
            <div className="kv" style={{ marginTop: 12 }}>
              <div className="k">Work surface</div>
              <div className="v">{Math.round(pl!.size_mm[0])} × {Math.round(pl!.size_mm[1])} mm
                <span className="hint"> (plane inliers {Math.round(pl!.inlier_frac * 100)}%)</span></div>
              {result.mode === "reference" ? (
                <>
                  <div className="k">Mode</div>
                  <div className="v">Reference — single-frame rectangle
                    <span className="hint"> (surface too large / far for a quality tour)</span></div>
                </>
              ) : (
                <>
                  <div className="k">Fused</div>
                  <div className="v">{result.n_views} views · {result.n_points.toLocaleString()} points ·
                    {result.mesh_vertices.toLocaleString()} mesh verts</div>
                </>
              )}
            </div>
            <div className="hint" style={{ marginTop: 6 }}>
              {result.mode === "reference"
                ? "The work rectangle + frame were placed from a single frame. Insert adds them to RoboDK."
                : "Orbit/zoom the cloud above. The blue rectangle + axes are the proposed work surface and frame. Insert creates them (and the mesh) in RoboDK — nothing is added until you do."}
            </div>
            <div className="btn-row">
              <button onClick={insert} disabled={!result.can_insert || inserted}>
                {inserted ? "Inserted ✓" : "Insert into RoboDK"}
              </button>
            </div>
            <div className="hint">Artifacts: <code>{result.run_dir}</code></div>
          </>
        ) : (
          <div className="hint">Run a scan to fuse the surface and preview the proposed frame + rectangle here.
            For a reference surface (too large for a tour), the rectangle appears here after Create targets.</div>
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
