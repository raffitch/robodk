import { useCallback, useEffect, useRef, useState } from "react";
import { apiGet, moduleApi } from "../api/client";
import { useEvents, type JobEvent } from "../api/events";
import AimHud, { type GateReading } from "./AimHud";
import { robotLinkNote } from "./Calibration";
import CollisionPanel, { type CollisionStatus } from "../components/CollisionPanel";
import ScanViewer from "./ScanViewer";
import StreamStats, { useStreamStats } from "./StreamStats";
import SurveyPanel, { type SurveyState } from "./SurveyPanel";

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
  mesh_kind?: string;
  coverage?: {
    weakest_edge: number;
    interior: number;
    fill: number;
    bin_mm?: number;
    edges?: Record<string, number>;
  };
  quality?: { voxel_size_mm: number; surface_mesh_spacing_mm: number; frames_per_pose: number };
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

function meanUv(points: Array<[number, number]> | null | undefined): [number, number] | null {
  if (!points?.length) return null;
  let sx = 0, sy = 0;
  for (const [u, v] of points) { sx += u; sy += v; }
  return [sx / points.length, sy / points.length];
}

// A tool rotation that levels the plane barely moves the projected rectangle's
// centroid, so the distance/centroid tests below miss it. Detect a genuine tilt
// change directly (signed B/C, so a flip through level also counts) with a
// deadband: per-frame RealSense plane noise stays under it (stays frozen), a real
// leveling move exceeds it (releases the freeze so the rectangle + tilt update).
const TILT_SHIFT_DEG = 1.5;
function tiltShifted(a: GateReading, b: GateReading): boolean {
  const ab = a.tilt_b_deg, bb = b.tilt_b_deg, ac = a.tilt_c_deg, bc = b.tilt_c_deg;
  if (ab != null && bb != null && ac != null && bc != null)
    return Math.hypot(bb - ab, bc - ac) > TILT_SHIFT_DEG;
  const at = a.tilt_deg, bt = b.tilt_deg;
  return at != null && bt != null && Math.abs(bt - at) > TILT_SHIFT_DEG;
}

function surfaceShifted(a: GateReading | null, b: GateReading): boolean {
  if (!a) return false;
  if (a.surface_mode !== b.surface_mode) return true;
  const da = a.distance_mm, db = b.distance_mm;
  if (da != null && db != null && Math.abs(db - da) > 80) return true;
  const ca = meanUv(a.outline_uv ?? a.points_uv);
  const cb = meanUv(b.outline_uv ?? b.points_uv);
  if (ca && cb && Math.hypot(cb[0] - ca[0], cb[1] - ca[1]) > 0.08) return true;
  if (tiltShifted(a, b)) return true;
  return false;
}

function freezeSurfaceGeometry(next: GateReading, frozen: GateReading): GateReading {
  const frozenMove = frozen.move_cam;
  const nextMove = next.move_cam;
  return {
    ...next,
    outline_uv: frozen.outline_uv ?? next.outline_uv,
    visible_outline_uv: frozen.visible_outline_uv ?? next.visible_outline_uv,
    points_uv: frozen.points_uv ?? next.points_uv,
    grid_uv: frozen.grid_uv ?? next.grid_uv,
    grid_spacing_mm: frozen.grid_spacing_mm ?? next.grid_spacing_mm,
    extent_mm: frozen.extent_mm ?? next.extent_mm,
    rectangle_size_mm: frozen.rectangle_size_mm ?? next.rectangle_size_mm,
    crop_size_mm: frozen.crop_size_mm ?? next.crop_size_mm,
    // Hold the tilt/level readouts too. While the surface is static the fitted
    // plane normal only wiggles from RealSense noise; freezing the rectangle/dots
    // but not these left the TILT readout + LEVEL (B/C/A) numbers jittering.
    // surfaceShifted() releases the whole freeze on a real leveling rotation, so
    // this guidance still updates the moment the operator actually tilts the tool.
    tilt_deg: frozen.tilt_deg ?? next.tilt_deg,
    tilt_b_deg: frozen.tilt_b_deg ?? next.tilt_b_deg,
    tilt_c_deg: frozen.tilt_c_deg ?? next.tilt_c_deg,
    yaw_a_deg: frozen.yaw_a_deg ?? next.yaw_a_deg,
    move_cam: frozenMove && nextMove
      ? [frozenMove[0], frozenMove[1], nextMove[2]]
      : next.move_cam,
    center_latched: true,
  };
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
  const autoConnectRef = useRef(false);
  const autoPreviewRef = useRef(false);
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
  // Live COLOR work boundary (server segments the object under the reticle from the color
  // frame every ~6 fps — see color_boundary.py). It IS the blue rectangle while aiming:
  // reliable + real-time, independent of the noisy 1 Hz depth telemetry + the freeze.
  const [liveBoundary, setLiveBoundary] =
    useState<{ outline: Array<[number, number]>; overruns: boolean } | null>(null);
  const boundaryTimerRef = useRef<number | null>(null);
  const frozenGateRef = useRef<GateReading | null>(null);
  const stableLiveFramesRef = useRef(0);
  const gateReceivedAtRef = useRef(0);
  const stableSinceRef = useRef<number | null>(null);
  const lastValidRef = useRef<number | null>(null);
  const [surfaceStable, setSurfaceStable] = useState(false);
  const [stableProgress, setStableProgress] = useState(0);
  const [surfaceLocked, setSurfaceLocked] = useState(false);
  const [surfaceMode, setSurfaceMode] = useState<"auto" | "crop">("auto");
  const [locking, setLocking] = useState(false);
  // Guided five-position workframe survey (spec §7): an alternative to Lock & create
  // targets for a platform too large for one camera view. surveyEvent forwards the
  // "survey" websocket event into SurveyPanel; it is reset to null at every survey
  // lifecycle transition (begin/finish/cancel) so a stale payload from a PRIOR survey
  // can never be misread as live state for a freshly-begun one (SurveyPanel's own
  // mount-time GET /survey/state is the real source of truth either way).
  const [surveyActive, setSurveyActive] = useState(false);
  const [surveyEvent, setSurveyEvent] = useState<SurveyState | null>(null);
  const [surveyStarting, setSurveyStarting] = useState(false);
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
  // Task 14 review Finding 2: a guided five-position survey can be left in progress
  // on the backend (a page refresh, a second tab, navigating away and back) with
  // nothing client-side to show for it — survey_begin's own docstring says it
  // REPLACES any prior in-progress survey and DISCARDS its captures, and each
  // capture cost a real jog-stop-measure cycle on the physical robot. On mount,
  // resume straight into SurveyPanel instead of ever offering a fresh "begin" that
  // would silently discard real work.
  const hydrateSurvey = useCallback(async () => {
    try {
      const r = await api.get<SurveyState>("/survey/state");
      if (r.step != null) {
        setSurveyEvent(r);
        setSurveyActive(true);
        addLog(`resuming an in-progress guided survey (step: ${r.step}).`);
      }
    } catch { /* opportunistic, same as hydrateConnection */ }
  }, []);

  useEffect(() => { loadConfig(); refreshJob(); hydrateConnection(); hydrateSurvey(); },
            [loadConfig, refreshJob, hydrateConnection, hydrateSurvey]);
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
    if (conn !== "idle" || autoConnectRef.current) return;
    autoConnectRef.current = true;
    connect();
  }, [conn, connect]);

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
          const frozenBefore = p.live ? frozenGateRef.current : null;
          if (frozenBefore && surfaceShifted(frozenBefore, p)) {
            resetCoverage();
          }
          setGate((prev) => {
            // Spec §11: no frontend display latch may replace the locked polygon —
            // a non-live (locked) gate is shown exactly as the backend sent it.
            let next: GateReading = p;
            if (p.live) {
              const frozen = frozenGateRef.current;
              if (frozen && !surfaceShifted(frozen, p)) {
                next = freezeSurfaceGeometry(next, frozen);
              } else {
                if (prev && p.ok && !surfaceShifted(prev, p)) stableLiveFramesRef.current += 1;
                else stableLiveFramesRef.current = p.ok ? 1 : 0;
                if (stableLiveFramesRef.current >= 5) {
                  frozenGateRef.current = next;
                }
              }
            }
            return next;
          });
          if (p.live && !frozenGateRef.current
              && Array.isArray(p.points_uv) && p.points_uv.length) {
            accumulateCoverage(p.points_uv as Array<[number, number]>);
          }
        }
      } else if (ev.type === "survey") {
        // Fired by the backend after every successful five-position capture
        // (module.py -> five_position_capture); SurveyPanel merges this as a
        // fallback/resync channel alongside its own ~1s poll (ambiguity #3).
        setSurveyEvent(ev.payload as SurveyState);
      } else if (ev.type === "boundary") {
        // Video-rate color boundary. Store it (with a staleness timeout so it clears if
        // the stream stalls); the render decides whether to show it (live + not locked).
        const p = ev.payload as { outline_uv?: Array<[number, number]>; overruns?: boolean };
        if (Array.isArray(p.outline_uv) && p.outline_uv.length >= 3) {
          setLiveBoundary({ outline: p.outline_uv, overruns: !!p.overruns });
          if (boundaryTimerRef.current) window.clearTimeout(boundaryTimerRef.current);
          boundaryTimerRef.current = window.setTimeout(() => setLiveBoundary(null), 1500);
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
    frozenGateRef.current = null;
    stableLiveFramesRef.current = 0;
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
    if (last && Math.hypot(center[0] - last[0], center[1] - last[1]) > 0.20) {
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
      addLog("surface feed started — jog TOOL X/Y/Z and A/B/C until the guidance is stable.");
    } catch (e: any) { setLive(false); addLog("live: " + e.message, true); }
  };
  useEffect(() => {
    if (!ready || live || running || autoPreviewRef.current) return;
    autoPreviewRef.current = true;
    sessionStorage.removeItem("tasni:autoStartCamera");
    startLive();
  }, [ready, live, running]);
  const stopLive = async () => {
    try { await api.post("/live/stop"); } catch { /* ignore */ }
    setLive(false); resetStream(); resetCoverage();
  };
  // Re-read the surface at the current robot pose: drops the anti-jitter hold so the
  // overlay/readouts re-settle where the arm is NOW (fixes a stale projection when
  // RoboDK is not mirroring the arm). Keeps the video streaming.
  const refreshLive = async () => {
    try {
      await api.post("/live/refresh");
      resetCoverage();
      stableSinceRef.current = null;
      setSurfaceStable(false);
      setStableProgress(0);
      addLog("re-reading the surface at the current pose…");
    } catch (e: any) { addLog("refresh: " + e.message, true); }
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
      setSurfaceLocked(false);
      beginLive(true).catch(() => setLive(false));
    } finally { setGenerating(false); }
  };
  const lockAndCreateTargets = async () => {
    setLocking(true); setRunError(null);
    try {
      const r = await api.post<{
        status: string; gate: GateReading;
        surface_mode: "full" | "crop";
        extent_mm?: [number, number] | null;
        crop_size_mm?: [number, number] | null;
      }>("/surface/lock", { mode: surfaceMode });
      setLive(false); resetStream();
      // The lock snapshot (r.gate) is the sole authoritative geometry — display it
      // as sent, with no client-side latch/repair (spec §11). resetCoverage() here
      // matters too: coverageDots is a LIVE-only accumulator (see accumulateCoverage
      // above), and AimHud renders it ahead of the gate's own points_uv whenever it
      // is non-empty. Without clearing it at the moment of lock, the HUD would keep
      // showing the pre-lock accumulated dot union — stale live geometry — on top of
      // the just-locked authoritative snapshot, for as long as the surface stays
      // locked (this call is the only place surfaceLocked ever becomes true, so it
      // covers every path into the locked-review state, including a manual
      // "Accept region & create targets" click).
      setGate(r.gate);
      setSurfaceLocked(true);
      setSurfaceStable(false);
      resetCoverage();
      addLog(r.surface_mode === "crop" && r.crop_size_mm
        ? `surface locked — generic ${Math.round(r.crop_size_mm[0])} x ${Math.round(r.crop_size_mm[1])} mm work area; creating targets`
        : `surface locked — detected platform${
            r.extent_mm ? ` ${Math.round(r.extent_mm[0])} x ${Math.round(r.extent_mm[1])} mm` : ""}; creating targets`);
    } catch (e: any) {
      addLog("lock surface: " + e.message, true);
      setRunError("Lock surface: " + e.message);
      beginLive(true).catch(() => setLive(false));
      setLocking(false);
      return;
    }
    setLocking(false);
    await generateTargets();
  };
  // Guided five-position survey: an alternative to lockAndCreateTargets for a
  // platform too large for one camera view (surface_mode === "crop"). Starts the
  // backend's FivePositionSurvey state machine and swaps SurveyPanel in for the
  // normal lock controls; SurveyPanel drives its own capture/recapture/finish loop.
  const beginSurvey = async () => {
    setSurveyStarting(true); setRunError(null);
    try {
      // Task 14 review Finding 2: hydrateSurvey() only runs once on mount, so it
      // cannot see a survey another tab/operator started AFTER this page loaded.
      // Re-check right before the action that would discard captures (survey_begin
      // replaces any in-progress survey) — this closes that window instead of
      // merely narrowing it, so there is no remaining path where clicking this
      // button can discard real jog-stop-measure work.
      const existing = await api.get<SurveyState>("/survey/state");
      if (existing.step != null) {
        setSurveyEvent(existing);
        setSurveyActive(true);
        addLog(`an in-progress guided survey was already active (step: ${existing.step}) — resuming it.`);
        return;
      }
      await api.post<SurveyState>("/survey/begin");
      setSurveyEvent(null);
      setSurveyActive(true);
      addLog("guided five-position survey started — jog to the surface CENTER, stop, then Measure.");
    } catch (e: any) {
      addLog("survey: " + e.message, true);
      setRunError("Survey: " + e.message);
    } finally { setSurveyStarting(false); }
  };
  // Ambiguity resolution #2: /survey/finish already locks the surface server-side
  // (module.py's survey_finish sets self._locked_surface directly), so this mirrors
  // ONLY lockAndCreateTargets's post-lock bookkeeping (no second /surface/lock call)
  // before handing off to the SAME generateTargets() the compact/crop lock path uses.
  const handleSurveyFinished = async () => {
    setSurveyActive(false);
    setSurveyEvent(null);
    setLive(false); resetStream();
    // Follow-up 3 (Task 14 review): lockAndCreateTargets seeds a fresh authoritative
    // `gate` from /surface/lock's own response (see its comment on why a stale
    // pre-lock reading must not linger in the HUD). /survey/finish has no gate/reading
    // field to source an equivalent snapshot from, so the closest match is to clear
    // it outright — the same "fresh start, no stale carryover" semantics beginLive's
    // own clearGate=true path already uses — rather than let a reading from
    // somewhere the operator jogged WHILE measuring corners (not the final locked
    // surface) linger until the next live gate tick happens to overwrite it.
    setGate(null);
    gateReceivedAtRef.current = 0;
    setSurfaceLocked(true);
    setSurfaceStable(false);
    resetCoverage();
    addLog("five-position survey complete — surface locked; creating targets");
    await generateTargets();
  };
  const handleSurveyCancelled = () => {
    setSurveyActive(false);
    setSurveyEvent(null);
    addLog("guided survey cancelled.");
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
  const manualCropReady = surfaceMode === "crop"
    && !!gate?.gates?.detected && !!gate?.gates?.distance && !!gate?.gates?.angle;
  const canLockSurface = surfaceStable || manualCropReady;
  const surfaceDescription = gate?.surface_mode === "crop"
    ? crop
      ? `Surface overruns view — generic ${Math.round(crop[0])} × ${Math.round(crop[1])} mm work area on the reticle`
      : "Surface overruns view — a generic work area will be projected on the reticle"
    : gate?.fully_framed === false
      ? "Full surface detected — move toward the recommended distance to include every edge"
    : gate?.fully_framed === true && gate.extent_mm
      ? `Full surface ${Math.round(gate.extent_mm[0])} × ${Math.round(gate.extent_mm[1])} mm`
      : "Aim the center reticle at the intended work surface";
  // Spec §12: CENTER and EDGE A are advisory — they inform aim but never block
  // locking (canLockSurface does not gate on them) — so they must not be presented
  // identically to the mandatory lamps.
  const lamps: [string, boolean | undefined, boolean?][] = [
    ["DETECT", gate?.gates?.detected],
    ["DISTANCE", gate?.gates?.distance],
    ["ANGLE", gate?.gates?.angle],
    ["CENTER", gate?.gates?.center, true],
    ["EDGE A", gate?.gates?.edge, true],
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
          The surface feed starts automatically. Jog in the robot TOOL frame until
          the live X/Y/Z and A/B/C guides are green, hold steady for one second, then
          lock the measured platform and create scan targets.
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
                                     coverageDots={coverageDots}
                                     liveBoundary={live && !surfaceLocked && !running
                                       ? liveBoundary : null} />}
          {live && <StreamStats stat={streamStat} />}
          {!live && !gate && <div className="aim-off">camera off — press “Start camera”</div>}
        </div>

        <SurfaceModeNotice gate={gate} mode={surfaceMode}
                           onModeChange={setSurfaceMode}
                           disabled={running || locking || generating || surfaceLocked || surveyActive}
                           onRegionApplied={refreshLive} />
        {gate?.surface_mode === "crop" && !surveyActive && !surfaceLocked && (
          <div className="btn-row" style={{ marginTop: 8 }}>
            <button className="secondary" onClick={beginSurvey}
                    disabled={!ready || running || locking || generating || surveyStarting}
                    title="Guided center + four-corner survey for a platform too large to fit in one camera view">
              {surveyStarting ? "Starting survey…" : "Large surface — guided survey"}
            </button>
          </div>
        )}
        <SurfaceGuide gate={gate} stable={surfaceStable} />

        <div className="lamps">
          {lamps.map(([name, on, advisory]) => {
            const state = on === undefined ? "unknown" : on ? "on" : "off";
            const glyph = on === undefined ? "·" : on ? "✓" : "✗";
            const word = on === undefined ? "—" : on ? "OK" : "NO";
            return (
              <span key={name} className={"lamp " + state + (advisory ? " advisory" : "")}
                    title={advisory ? "advisory — does not block lock" : undefined}>
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

        {/* Locked-gate boundary provenance (spec §2/§11): the only geometry the
            review UI may show is the locked polygon, so the operator must also see
            how that boundary was established. Absent when the backend deliberately
            declined to claim provenance (auto-overrun with no operator region
            declared) — that case surfaces its `warnings` instead, honestly. */}
        {gate?.live === false && gate?.boundary_provenance && (
          <div className={"provenance-chip "
              + (gate.boundary_provenance.startsWith("user specified") ? "declared" : "")}
               title="boundary provenance — how the locked work-region boundary was established">
            {gate.boundary_provenance}
          </div>
        )}
        {gate?.live === false && !gate?.boundary_provenance
          && gate?.warnings && gate.warnings.length > 0 && (
          <div className="warn-text" style={{ marginTop: 6, fontSize: 12 }}>
            {gate.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
          </div>
        )}

        <CollisionPanel ready={ready} busy={collisionBusy} status={collision}
                        onRecheck={checkCollision}
                        onIgnore={ignoreCollisionPair}
                        recentPairs={recentCollisionPairs} />

        {surveyActive ? (
          <SurveyPanel event={surveyEvent}
                       onFinished={handleSurveyFinished}
                       onCancelled={handleSurveyCancelled} />
        ) : (
          <>
            <div className="btn-row">
              {!live && !surfaceLocked
                ? <button onClick={startLive} disabled={running}>Start camera</button>
                : live
                  ? <button className="secondary" onClick={stopLive}>Stop camera</button>
                  : null}
              {live
                ? <button className="secondary" onClick={refreshLive} disabled={running}
                          title="Re-read the surface at the current robot pose (clears a stale overlay if the arm moved but the readout didn't)">
                    Refresh view
                  </button>
                : null}
              {!surfaceLocked
                ? <button onClick={lockAndCreateTargets}
                          disabled={!ready || running || locking || generating || !live || !canLockSurface}>
                    {locking ? "Locking…" : generating ? "Creating…" : "Lock & create targets"}
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
              ? <div className="hint">Surface is locked. Reposition if the wrong plane or crop
                  is highlighted; otherwise create the targets.</div>
              : <div className="hint">Jog until RANGE, TILT, CENTER and EDGE are valid and
                  remain stable for one second. FRAMED may be red when the surface overruns
                  the view; in that case the scan uses the displayed generic 1 m work square.</div>}
          </>
        )}
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
              <ScanViewer nonce={viewerNonce}
                          src={result.stamp
                            ? `${PREVIEW_URL}?run_id=${encodeURIComponent(result.stamp)}`
                            : PREVIEW_URL}
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
                    {result.mesh_vertices.toLocaleString()} fitted mesh verts ·
                    {result.mesh_triangles.toLocaleString()} tris</div>
                  {result.quality && (
                    <>
                      <div className="k">Quality</div>
                      <div className="v">{result.quality.voxel_size_mm.toFixed(1)} mm TSDF ·
                        fitted flat mesh insert ·
                        {result.quality.frames_per_pose} frame{result.quality.frames_per_pose === 1 ? "" : "s"}/target</div>
                    </>
                  )}
                  {result.coverage && (
                    <>
                      <div className="k">Coverage</div>
                      <div className="v">weakest edge {Math.round(result.coverage.weakest_edge * 100)}% ·
                        interior {Math.round(result.coverage.interior * 100)}%</div>
                    </>
                  )}
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


function SurfaceModeNotice({ gate, mode, onModeChange, disabled, onRegionApplied }:
  { gate: GateReading | null; mode: "auto" | "crop";
    onModeChange: (mode: "auto" | "crop") => void; disabled: boolean;
    onRegionApplied: () => void }) {
  const manualCrop = mode === "crop";
  const crop = manualCrop || gate?.surface_mode === "crop";
  const full = gate?.surface_mode === "full" && gate?.fully_framed === true;
  const cropSize = gate?.crop_size_mm;
  const extent = gate?.extent_mm;
  const modeText = manualCrop
    ? "Manual large platform mode"
    : crop
    ? "Large platform mode"
    : full
      ? "Rectangle tracking"
      : "Surface mode";
  const detail = manualCrop
    ? "ON - ignoring unstable rectangle/dot coverage; lock will use the center depth plane and fixed reticle work area."
    : crop
      ? cropSize
        ? `ON - surface exceeds the camera frame; using a fixed ${Math.round(cropSize[0])} x ${Math.round(cropSize[1])} mm reticle work area.`
        : "ON - surface exceeds the camera frame; using the reticle work area."
    : full && extent
      ? `ON - full rectangle tracked (${Math.round(extent[0])} x ${Math.round(extent[1])} mm). X/Y is advisory; targets use the measured rectangle center.`
      : "Auto will use a finite rectangle when the measured surface fits. Turn on User-specified region if dot coverage is incomplete or jittery.";

  // Operator-typed region size (mm). Defaults from the gate's current crop_size_mm
  // and keeps tracking it until the operator edits a field, so the inputs start
  // pre-filled with the live/measured value but never fight the operator's typing.
  const touchedRef = useRef(false);
  const [width, setWidth] = useState(() =>
    gate?.crop_size_mm ? String(Math.round(gate.crop_size_mm[0])) : "");
  const [height, setHeight] = useState(() =>
    gate?.crop_size_mm ? String(Math.round(gate.crop_size_mm[1])) : "");
  const [regionBusy, setRegionBusy] = useState(false);
  const [regionError, setRegionError] = useState<string | null>(null);
  const cropW = cropSize?.[0], cropH = cropSize?.[1];
  useEffect(() => {
    if (touchedRef.current || cropW == null || cropH == null) return;
    setWidth(String(Math.round(cropW)));
    setHeight(String(Math.round(cropH)));
  }, [cropW, cropH]);

  const applyRegion = async () => {
    const w = parseFloat(width), h = parseFloat(height);
    if (!Number.isFinite(w) || !Number.isFinite(h)) {
      setRegionError("enter numeric width and height (mm)."); return;
    }
    setRegionBusy(true); setRegionError(null);
    try {
      await api.post<{ user_region_mm: [number, number] }>(
        "/surface/region", { width_mm: w, height_mm: h });
      onRegionApplied();
    } catch (e: any) {
      setRegionError(e.message);
    } finally { setRegionBusy(false); }
  };

  return (
    <div className={"surface-mode " + (crop ? "crop" : full ? "full" : "")}>
      <button type="button"
              className={"surface-mode-toggle " + (manualCrop ? "on" : "")}
              disabled={disabled}
              onClick={() => onModeChange(manualCrop ? "auto" : "crop")}>
        <span className="surface-mode-knob" />
        {manualCrop ? "User-specified region" : "Auto"}
      </button>
      <div>
        <b>{modeText}</b>
        <span>{detail}</span>
        {crop && (
          <div className="row" style={{ gap: 10, alignItems: "flex-end", marginTop: 8 }}>
            <div className="field">
              <label>Region W (mm)</label>
              <input type="number" min={100} max={4000} style={{ width: 90 }}
                     value={width} disabled={disabled || regionBusy}
                     onChange={(e) => { touchedRef.current = true; setWidth(e.target.value); }} />
            </div>
            <div className="field">
              <label>Region H (mm)</label>
              <input type="number" min={100} max={4000} style={{ width: 90 }}
                     value={height} disabled={disabled || regionBusy}
                     onChange={(e) => { touchedRef.current = true; setHeight(e.target.value); }} />
            </div>
            <button type="button" className="secondary mini" disabled={disabled || regionBusy}
                    onClick={applyRegion}>
              {regionBusy ? "Applying…" : "Apply region"}
            </button>
            {regionError && <span className="warn-text" style={{ fontSize: 12 }}>{regionError}</span>}
          </div>
        )}
      </div>
    </div>
  );
}

function SurfaceGuide({ gate, stable }: { gate: GateReading | null; stable: boolean }) {
  const move = gate?.move_cam ?? null;
  const cropSurface = gate?.surface_mode === "crop";
  const trackedRectangle = gate?.surface_mode === "full" && gate?.fully_framed === true;
  const distanceTol = gate?.distance_tol_mm ?? 50;
  const centerTol = gate?.center_tol_mm ?? 30;
  // X/Y move_cam guidance only tracks the physical arm while RoboDK's driver is
  // connected and mirroring it (see GateReading.pose_live). Without that link the
  // reading can be stale, so Center X/Y degrade instead of reading as trustworthy
  // "OK"/instruction text. Range/Level/Edge are camera-derived and stay untouched —
  // true regardless of driver monitoring. Absent/true both mean "live" (no change).
  const poseLive = gate?.pose_live !== false;
  const range = move ? axisInstruction("Z", move[2], distanceTol, "mm") : "not measured";
  const centerX = !poseLive
    ? "model pose — not live"
    : trackedRectangle && !gate?.gates?.center
    ? "rectangle tracked"
    : move ? axisInstruction("X", move[0], centerTol, "mm") : "not measured";
  const centerY = !poseLive
    ? "model pose — not live"
    : trackedRectangle && !gate?.gates?.center
    ? "rectangle tracked"
    : move ? axisInstruction("Y", move[1], centerTol, "mm") : "not measured";
  const tiltB = rotationInstruction("B", gate?.tilt_b_deg ?? null);
  const tiltC = rotationInstruction("C", gate?.tilt_c_deg ?? null);
  const yawA = rotationInstruction("A", gate?.yaw_a_deg ?? null);
  const chips = [
    ["Range", range, !!gate?.gates?.distance],
    ["Center X", centerX,
      poseLive && !!gate && (gate.gates?.center || cropSurface || trackedRectangle)],
    ["Center Y", centerY,
      poseLive && !!gate && (gate.gates?.center || cropSurface || trackedRectangle)],
    ["Level B", tiltB, !!gate?.gates?.angle],
    ["Level C", tiltC, !!gate?.gates?.angle],
    ["Edge A", trackedRectangle && gate?.yaw_a_deg == null ? "rectangle tracked" : yawA,
      !!gate && (gate.gates?.edge || cropSurface || trackedRectangle)],
  ] as const;
  return (
    <div className={"surface-guide " + (stable ? "ready" : gate?.ok ? "holding" : "")}>
      <div className="surface-guide-head">
        <b>{stable ? "Surface ready" : gate?.detected ? "Review measured surface" : "Move near the platform"}</b>
        <span>{gate?.distance_mm != null
          ? `${Math.round(gate.distance_mm)} mm from surface, target ${Math.round(gate.ideal_distance_mm ?? 0)} mm`
          : "Waiting for live surface telemetry"}</span>
      </div>
      <div className="surface-guide-grid">
        {chips.map(([label, text, ok]) => {
          const degraded = !poseLive && (label === "Center X" || label === "Center Y");
          return (
            <div key={label} className={"surface-guide-chip " + (ok ? "ok" : "fix")}
                 title={degraded
                   ? "driver not monitoring — X/Y guidance is not real-time" : undefined}>
              <span>{label}</span>
              <b>{text}</b>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function axisInstruction(axis: string, value: number, tol: number, unit: string) {
  if (Math.abs(value) <= tol) return "OK";
  return `${axis}${value >= 0 ? "+" : "-"} ${Math.round(Math.abs(value))} ${unit}`;
}

function rotationInstruction(axis: string, value: number | null) {
  if (value == null) return "OK";
  if (Math.abs(value) < 1) return "OK";
  return `${axis}${value >= 0 ? "+" : "-"} ${Math.round(Math.abs(value))} deg`;
}
