import { useCallback, useEffect, useRef, useState } from "react";
import { apiGet, moduleApi, type ApiError } from "../api/client";
import { useEvents, type JobEvent } from "../api/events";
import AimHud, { type GateReading } from "./AimHud";
import { robotLinkNote } from "./Calibration";
import CollisionPanel, { type CollisionStatus } from "../components/CollisionPanel";
import ScanViewer from "./ScanViewer";
import StreamStats, { useStreamStats } from "./StreamStats";
import SurveyPanel, { type SurveyState, type SurveyReport } from "./SurveyPanel";

const api = moduleApi("scan");
const TARGET_PREFIX = "TasniScan_";          // must match service.py scan.target_prefix
const PREVIEW_URL = "/api/modules/scan/preview.bin";
const STABLE_LOCK_MS = 1000;
const GATE_FRESH_MS = 1600;
// Adaptive-scan plan Task 2: three INDEPENDENT concepts. `goal` is what the operator
// needs (a working frame, or also dense surface data), `scope` is what must be
// measured (every physical boundary, or a sized ROI), and the acquisition mode below
// is the provenance-bearing path that was actually used. Both goal and scope are sent
// on every lock — the backend binds them to the lock token, so changing either
// invalidates prepared results and generated targets server-side too.
type WorkflowGoal = "frame_only" | "full_scan";
type SurfaceScope = "entire_platform" | "declared_region";
// Mirrors survey_contract.PROVENANCE_BY_MODE — only used as a fallback label when a
// response carries the acquisition mode but not the provenance string it implies.
const PROVENANCE_BY_MODE: Record<string, string> = {
  compact: "camera measured - complete boundary",
  five_position: "camera measured - five-position boundary survey",
  user_specified: "user specified - plane measured, boundary declared",
};
const ACQUISITION_LABEL: Record<string, string> = {
  compact: "Compact — one authoritative view",
  five_position: "Five-position — center + four corners",
  user_specified: "User-specified region",
};
// service.py LOCK_MAX_AGE_S: prepare/generate refuse a lock older than this.
const LOCK_MAX_AGE_S = 120;
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
  // Absent on a directly-prepared result (reference / frame_only): there is no
  // fused cloud to fit a plane against, so there is no inlier statistic either.
  inlier_frac?: number;
  inlier_count?: number;
}
interface ScanResult {
  kind?: "scan";
  run_dir: string;
  can_insert: boolean;
  mode?: string;           // "quality" | "reference" | "frame_only"
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
  // The tour path reports fusion quality here; a frame-only report puts the locked
  // survey's own quality dict in the same field (plan Task 4), so every member is
  // optional and the review UI branches on `mode`.
  quality?: { voxel_size_mm?: number; surface_mesh_spacing_mm?: number;
              frames_per_pose?: number } & Record<string, unknown>;
  stamp?: string;
  plane: Plane;
  // frame_only reports (plan Task 4/10)
  workflow_goal?: string;
  surface_scope?: string;
  acquisition_mode?: string;
  boundary_provenance?: string;
  calibration_id?: string;
  captures?: string[];
  mesh_file?: string | null;
}
// Everything the operator must be able to check BEFORE preparing a working frame
// (plan Task 5), normalised across the two acquisition paths: /surface/lock echoes
// the whole locked survey inside its gate payload, while /survey/finish returns the
// same survey's quality flattened.
interface LockedSurveyInfo {
  acquisition_mode: string;
  boundary_provenance: string | null;
  size_mm: [number, number] | null;
  calibration_id: string | null;
  quality: Record<string, any> | null;
  can_prepare_frame: boolean;
  workflow_goal: string;
  surface_scope: string;
  warnings: string[];
  locked_at_ms: number;      // client clock — lock freshness is shown relative to it
}
// 409 detail from POST /surface/lock when the platform overruns the camera view under
// entire-platform scope (plan Task 3). NOT a generic failure: it has one primary
// recovery action, and the generic crop must never be presented as the full surface.
interface LargeSurfaceDetail {
  error: string;
  message: string;
  extent_mm?: [number, number] | null;
  surface_scope?: string;
  workflow_goal?: string;
  primary_action?: string;
  alternatives?: string[];
}
interface ActiveCalibration {
  run_id: string | null; applied_at: string; tool: string;
  quality?: { verdict?: string | null; val_rms_px?: number | null;
              train_rms_px?: number | null };
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
  // Plan Task 2/5: the operator's intent, chosen BEFORE acquiring and sent on every
  // lock. Defaults match the backend's own defaults (full_scan / entire_platform), so
  // the pre-existing target → dry tour → run → review → insert path is unchanged.
  const [goal, setGoal] = useState<WorkflowGoal>("full_scan");
  const [scope, setScope] = useState<SurfaceScope>("entire_platform");
  const [lockedSurvey, setLockedSurvey] = useState<LockedSurveyInfo | null>(null);
  const [largeSurface, setLargeSurface] = useState<LargeSurfaceDetail | null>(null);
  const [preparing, setPreparing] = useState(false);
  const [prepared, setPrepared] = useState(false);
  const [lockAgeS, setLockAgeS] = useState(0);
  const [calib, setCalib] = useState<ActiveCalibration | null>(null);
  const [locking, setLocking] = useState(false);
  const frameOnly = goal === "frame_only";
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
        setScope("entire_platform");     // see beginSurvey
        addLog(`resuming an in-progress guided survey (step: ${r.step}).`);
      }
    } catch { /* opportunistic, same as hydrateConnection */ }
  }, []);

  // Calibration identity/status shown before a frame-only preparation (plan Task 5):
  // the same applied-run provenance the Dashboard shows. The lock's own
  // `calibration_id` is the value the backend actually re-checks at prepare time;
  // this adds the human-readable "when/what verdict" alongside it.
  const hydrateCalibration = useCallback(async () => {
    try {
      const r = await apiGet<{ active: ActiveCalibration | null }>(
        "/api/runs/active?module=calibration");
      setCalib(r.active);
    } catch { /* opportunistic, same as hydrateConnection */ }
  }, []);

  useEffect(() => { loadConfig(); refreshJob(); hydrateConnection(); hydrateSurvey();
                    hydrateCalibration(); },
            [loadConfig, refreshJob, hydrateConnection, hydrateSurvey, hydrateCalibration]);
  // Lock freshness ticker — the backend refuses a prepare/generate on a lock older
  // than LOCK_MAX_AGE_S, so the operator must be able to see the age before clicking.
  useEffect(() => {
    if (!lockedSurvey) { setLockAgeS(0); return; }
    const t0 = lockedSurvey.locked_at_ms;
    const tick = () => setLockAgeS((Date.now() - t0) / 1000);
    tick();
    const id = window.setInterval(tick, 1000);
    return () => window.clearInterval(id);
  }, [lockedSurvey]);
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
    setLockedSurvey(null);
    setPrepared(false);
    setLargeSurface(null);
    setGate(null);
    await beginLive(true);
  };
  // Plan Task 5: changing goal or scope invalidates everything measured/planned under
  // the previous intent. The backend enforces this too (goal+scope are folded into the
  // lock fingerprint), but the UI must not keep showing a lock, a target count or a
  // prepared frame that no longer belongs to what the operator is now asking for.
  const clearIntentState = async (why: string) => {
    setLargeSurface(null);
    setLockedSurvey(null);
    setPrepared(false);
    setTour(null);
    setScanMode(null);
    setResult(null);
    setInserted(false);
    setRunError(null);
    addLog(why);
    if (surfaceLocked) {
      setSurfaceLocked(false);
      try { await api.post("/surface/unlock"); } catch { /* best effort */ }
    }
    if (targets != null) {
      // Leaving TasniScan_* in the station while the UI reports none would be a lie,
      // and they can never be run again anyway (run() rejects targets that predate the
      // current lock token, which the new intent has already invalidated).
      try {
        const r = await api.post<{ cleared: number }>("/poses/clear");
        addLog(`cleared ${r.cleared} scan targets created under the previous goal/scope.`);
      } catch (e: any) { addLog("clear targets: " + e.message, true); }
      setTargets(null);
    }
  };
  const changeGoal = (next: WorkflowGoal) => {
    if (next === goal) return;
    setGoal(next);
    clearIntentState(`goal → ${next === "frame_only" ? "working frame only" : "full scan"}`
      + " — the previous lock, targets and prepared result no longer apply.");
  };
  const changeScope = (next: SurfaceScope) => {
    if (next === scope) return;
    setScope(next);
    clearIntentState(`surface scope → ${next === "declared_region"
      ? "declared work region (boundary declared, not measured)"
      : "entire platform (every boundary must be measured)"}`
      + " — the previous lock, targets and prepared result no longer apply.");
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
      setLockedSurvey(null);      // the lock was consumed by generation
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
      // /poses/generate can raise the same structured large-surface refusal as
      // /surface/lock (module.py catches LargeSurfaceRequired ahead of RuntimeError
      // there too), so offer the same one primary recovery action rather than an
      // opaque error.
      const detail = (e as ApiError).detail as LargeSurfaceDetail | undefined;
      if (detail?.error === "large_surface_required") {
        setLargeSurface(detail);
        addLog("create targets: " + detail.message, true);
      } else {
        addLog("create targets: " + e.message, true);
        setRunError("Create targets: " + e.message);
      }
      setSurfaceLocked(false);
      setLockedSurvey(null);
      beginLive(true).catch(() => setLive(false));
    } finally { setGenerating(false); }
  };
  const lockSurface = async () => {
    setLocking(true); setRunError(null); setLargeSurface(null);
    try {
      const r = await api.post<{
        status: string; gate: GateReading;
        surface_mode: "full" | "crop";
        extent_mm?: [number, number] | null;
        crop_size_mm?: [number, number] | null;
        workflow_goal: string;
        surface_scope: string;
        boundary_provenance?: string | null;
        can_prepare_frame: boolean;
      }>("/surface/lock", { workflow_goal: goal, surface_scope: scope });
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
      setPrepared(false);
      resetCoverage();
      // /surface/lock echoes the whole immutable locked survey inside the gate
      // payload (service.py: gate_payload["survey"] = record.to_dict()), which is
      // where the pre-preparation facts below come from — acquisition mode,
      // provenance, size, calibration id and plane/corner residuals.
      // No record means the backend deliberately declined to claim a boundary (a
      // compact lock whose eligibility gates failed) — never fabricate provenance or
      // an acquisition mode for it; its `warnings` carry the honest reason instead.
      const rec = (r.gate?.survey ?? null) as Record<string, any> | null;
      const acq = (rec?.mode as string) ?? "";
      setLockedSurvey({
        acquisition_mode: acq,
        boundary_provenance: r.boundary_provenance
          ?? (rec?.boundary_provenance as string)
          ?? (rec ? PROVENANCE_BY_MODE[acq] ?? null : null),
        size_mm: (rec?.size_mm as [number, number]) ?? r.extent_mm ?? r.crop_size_mm ?? null,
        calibration_id: (rec?.calibration_id as string) ?? null,
        quality: (rec?.quality as Record<string, any>) ?? null,
        can_prepare_frame: !!r.can_prepare_frame,
        workflow_goal: r.workflow_goal, surface_scope: r.surface_scope,
        warnings: r.gate?.warnings ?? [],
        locked_at_ms: Date.now(),
      });
      const sizeTxt = r.surface_mode === "crop" && r.crop_size_mm
        ? `declared ${Math.round(r.crop_size_mm[0])} x ${Math.round(r.crop_size_mm[1])} mm work region`
        : `measured platform${r.extent_mm
            ? ` ${Math.round(r.extent_mm[0])} x ${Math.round(r.extent_mm[1])} mm` : ""}`;
      addLog(`surface locked — ${sizeTxt}; `
        + (goal === "frame_only"
            ? "review the survey below, then Prepare working frame (no robot motion)."
            : "creating targets"));
    } catch (e: any) {
      const detail = (e as ApiError).detail as LargeSurfaceDetail | undefined;
      if (detail?.error === "large_surface_required") {
        // Plan Task 3: a recoverable state with ONE primary action, not a failure.
        // Rendered by <LargeSurfaceNotice> rather than the generic error banner, and
        // deliberately never falls back to the generic crop.
        setLargeSurface(detail);
        addLog("lock surface: " + detail.message, true);
      } else {
        addLog("lock surface: " + e.message, true);
        setRunError("Lock surface: " + e.message);
      }
      // A failed lock leaves the backend with NO locked surface, so nothing here is
      // reviewable or preparable any more. An ALREADY prepared frame survives on
      // purpose (module.py clears _prepared_result only once a new lock succeeds), so
      // `result`/`prepared` are deliberately left alone — it stays insertable below.
      setLockedSurvey(null);
      beginLive(true).catch(() => setLive(false));
      setLocking(false);
      return;
    }
    setLocking(false);
    // Plan Task 4: frame-only creates no motion targets at all — the locked survey is
    // reviewed and turned straight into a working frame instead.
    if (goal === "full_scan") await generateTargets();
  };
  // Plan Task 4/5: build a reviewable working frame straight from the locked survey.
  // No robot motion, no targets, no fusion; insertion stays a separate explicit click.
  const prepareFrame = async () => {
    setPreparing(true); setRunError(null);
    try {
      const r = await api.post<{ status: string; report: ScanResult }>("/surface/prepare-frame");
      setResult({ ...r.report, can_insert: true });
      setViewerNonce((n) => n + 1);
      setInserted(false);
      setPrepared(true);
      const size = r.report.plane?.size_mm;
      addLog(`working frame prepared${size
        ? ` — ${Math.round(size[0])} x ${Math.round(size[1])} mm` : ""}`
        + " from the locked survey; no robot motion. Review & Insert below.");
    } catch (e: any) {
      addLog("prepare working frame: " + e.message, true);
      setRunError("Prepare working frame: " + e.message);
    } finally { setPreparing(false); }
  };
  // Guided five-position survey: an alternative to a compact lock for a
  // platform too large for one camera view (surface_mode === "crop"). Starts the
  // backend's FivePositionSurvey state machine and swaps SurveyPanel in for the
  // normal lock controls; SurveyPanel drives its own capture/recapture/finish loop.
  const beginSurvey = async () => {
    setSurveyStarting(true); setRunError(null); setLargeSurface(null);
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
      // A five-position survey exists precisely to measure a whole platform's real
      // boundary, and survey_finish locks it as entire_platform regardless — so keep
      // the selector honest rather than letting it claim a declared region.
      setScope("entire_platform");
      addLog("guided five-position survey started — jog to the surface CENTER, stop, then Measure.");
    } catch (e: any) {
      addLog("survey: " + e.message, true);
      setRunError("Survey: " + e.message);
    } finally { setSurveyStarting(false); }
  };
  // Ambiguity resolution #2: /survey/finish already locks the surface server-side
  // (module.py's survey_finish sets self._locked_surface directly), so this mirrors
  // ONLY lockSurface's post-lock bookkeeping (no second /surface/lock call) before
  // handing off to the SAME generateTargets() the compact/crop lock path uses — or,
  // for a frame-only goal, to the same review-then-Prepare panel (plan Task 5).
  const handleSurveyFinished = async (report: SurveyReport) => {
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
    setPrepared(false);
    setScope("entire_platform");   // what survey_finish actually locked (see beginSurvey)
    resetCoverage();
    // /survey/finish returns the locked survey's quality flattened (plus the intent it
    // was locked under) — everything the pre-preparation review below needs except the
    // calibration id, which that response does not carry.
    setLockedSurvey({
      acquisition_mode: "five_position",
      boundary_provenance: PROVENANCE_BY_MODE.five_position,
      size_mm: report.size_mm ?? null,
      calibration_id: null,
      quality: report as unknown as Record<string, any>,
      can_prepare_frame: report.can_prepare_frame ?? true,
      workflow_goal: report.workflow_goal ?? goal,
      surface_scope: report.surface_scope ?? "entire_platform",
      warnings: report.warnings ?? [],
      locked_at_ms: Date.now(),
    });
    addLog("five-position survey complete — surface locked; "
      + (goal === "frame_only"
          ? "review it below, then Prepare working frame (no robot motion)."
          : "creating targets"));
    if (goal === "full_scan") await generateTargets();
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
  const declaredRegion = scope === "declared_region";
  const manualCropReady = declaredRegion
    && !!gate?.gates?.detected && !!gate?.gates?.distance && !!gate?.gates?.angle;
  const canLockSurface = surfaceStable || manualCropReady;
  // Plan Task 3 item 5: never call an overrun crop the full surface. Under
  // entire-platform scope an overrun has no crop fallback at all — it needs the
  // five-position survey — and under declared-region scope the rectangle is the
  // operator's declaration, not a measurement.
  const surfaceDescription = declaredRegion
    ? crop
      ? `Declared work region ${Math.round(crop[0])} × ${Math.round(crop[1])} mm on the reticle — boundary declared, not measured`
      : "Declared work region on the reticle — boundary declared, not measured"
    : gate?.surface_mode === "crop"
      ? "Platform overruns the view — its full boundary cannot be measured from here; survey it from five positions"
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
  // A directly-prepared result (reference locate / frame-only) has no fused cloud, so
  // there is nothing for the 3D viewer to load and no fusion quality to report.
  const directResult = result?.mode === "reference" || result?.mode === "frame_only";
  const voxelMm = typeof result?.quality?.voxel_size_mm === "number"
    ? result.quality.voxel_size_mm : null;
  const framesPerPose = typeof result?.quality?.frames_per_pose === "number"
    ? result.quality.frames_per_pose : null;
  const intentBusy = running || locking || generating || preparing;

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

      {/* ---- Goal + scope (plan Task 2) ---------------------------------- */}
      <div className="card">
        <h2>What do you need?</h2>
        <div className="row" style={{ gap: 26 }}>
          <div className="field">
            <label>Workflow goal</label>
            <div className="btn-row" style={{ marginTop: 0 }}>
              <button className={frameOnly ? "" : "secondary"} aria-pressed={frameOnly}
                      disabled={intentBusy}
                      title="Locate the platform and insert its working frame — no scan tour, no robot motion"
                      onClick={() => changeGoal("frame_only")}>Working frame only</button>
              <button className={!frameOnly ? "" : "secondary"} aria-pressed={!frameOnly}
                      disabled={intentBusy}
                      title="Working frame plus a dense fused surface mesh (moves the robot through a scan tour)"
                      onClick={() => changeGoal("full_scan")}>Full scan</button>
            </div>
          </div>
          <div className="field">
            <label>Surface scope</label>
            <div className="btn-row" style={{ marginTop: 0 }}>
              <button className={!declaredRegion ? "" : "secondary"} aria-pressed={!declaredRegion}
                      disabled={intentBusy || surveyActive}
                      title="Every physical boundary must be measured — a platform that overruns the view needs the five-position survey"
                      onClick={() => changeScope("entire_platform")}>Entire platform</button>
              <button className={declaredRegion ? "" : "secondary"} aria-pressed={declaredRegion}
                      disabled={intentBusy || surveyActive}
                      title="A sized rectangle you declare on the measured plane — provenance stays 'declared', not 'measured'"
                      onClick={() => changeScope("declared_region")}>Declared work region</button>
            </div>
          </div>
        </div>
        <div className="hint">
          {frameOnly
            ? "Working frame only: measure the platform, review it, then insert its frame + "
              + "rectangle into RoboDK. No scan targets are created and the robot never moves."
            : "Full scan: measure the platform, create scan targets, dry-run them, then move the "
              + "robot through the tour to fuse a surface mesh."}
          {" "}
          {declaredRegion
            ? "Declared work region: you type the rectangle size; only its plane is measured, "
              + "so its boundary is recorded as declared."
            : "Entire platform: every boundary is measured — a platform that overruns the camera "
              + "view must be surveyed from five positions."}
        </div>
      </div>

      {/* ---- Survey gate ------------------------------------------------ */}
      <div className="card">
        <h2>Survey the surface</h2>
        <div className="hint" style={{ marginTop: 0, marginBottom: 10 }}>
          The surface feed starts automatically. Jog in the robot TOOL frame until
          the live X/Y/Z and A/B/C guides are green, hold steady for one second, then
          lock the measured platform{frameOnly ? "" : " and create scan targets"}.
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

        <SurfaceModeNotice gate={gate} scope={scope}
                           disabled={running || locking || generating || preparing
                                     || surfaceLocked || surveyActive}
                           onRegionApplied={refreshLive} />
        {largeSurface && !surveyActive && (
          <LargeSurfaceNotice detail={largeSurface}
                              busy={!ready || running || locking || generating || surveyStarting}
                              onSurvey={beginSurvey}
                              onDeclareRegion={() => changeScope("declared_region")} />
        )}
        {gate?.surface_mode === "crop" && !surveyActive && !surfaceLocked && !largeSurface && (
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
          <SurveyPanel event={surveyEvent} goal={goal}
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
                ? <button onClick={lockSurface}
                          disabled={!ready || running || locking || generating || !live || !canLockSurface}>
                    {locking ? "Locking…" : generating ? "Creating…"
                      : frameOnly ? "Lock & review surface" : "Lock & create targets"}
                  </button>
                : <>
                    <button className="secondary" onClick={repositionSurface}
                            disabled={running || generating || preparing}>Reposition</button>
                    {!frameOnly && (
                      <button onClick={generateTargets}
                              disabled={!ready || running || generating}>
                        {generating ? "Creating…" : "Accept region & create targets"}
                      </button>
                    )}
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
              ? <div className="hint">Surface is locked. Reposition if the wrong plane or region
                  is highlighted; otherwise {frameOnly
                    ? "review it below and prepare the working frame." : "create the targets."}</div>
              : <div className="hint">Jog until RANGE, TILT, CENTER and EDGE are valid and
                  remain stable for one second. FRAMED goes red when the platform overruns the
                  view: under <b>Entire platform</b> that boundary must be measured with the
                  guided five-position survey, and only under <b>Declared work region</b> is the
                  sized rectangle you typed used instead (recorded as declared, not measured).</div>}
          </>
        )}
      </div>

      {/* ---- Frame-only preparation (plan Task 4/5) ---------------------- */}
      {frameOnly && (
        <div className="card">
          <h2>Prepare working frame</h2>
          {lockedSurvey ? (
            <FramePrepPanel info={lockedSurvey} calib={calib} ageS={lockAgeS}
                            preparing={preparing} prepared={prepared}
                            disabled={!ready || running || locking || generating}
                            onPrepare={prepareFrame} />
          ) : (
            <div className="hint">Lock the surface above (compact view or guided five-position
              survey) — its locked polygon and quality appear here for review before the frame
              is prepared. Preparation moves nothing: it converts the locked survey straight
              into a frame + rectangle you then insert.</div>
          )}
        </div>
      )}

      {/* ---- Run -------------------------------------------------------- */}
      <div className="card">
        <h2>Run scan</h2>
        {/* Frame-only creates no motion targets at all (plan Task 4), so the tour
            controls are not offered — not merely disabled. The error banner and
            progress line stay, since lock/prepare failures report through them. */}
        {frameOnly ? (
          <div className="hint" style={{ marginTop: 0 }}>
            Not used for a working frame: the goal above is <b>Working frame only</b>, which
            creates no scan targets and never moves the robot. Switch the goal to
            <b> Full scan</b> if you also need a fused surface mesh.
          </div>
        ) : (
          <>
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
          </>
        )}

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
            {!directResult && (
              <ScanViewer nonce={viewerNonce}
                          src={result.stamp
                            ? `${PREVIEW_URL}?run_id=${encodeURIComponent(result.stamp)}`
                            : PREVIEW_URL}
                          frameT={pl?.frame_T_mm} corners={pl?.corners_mm} />
            )}
            <div className="kv" style={{ marginTop: 12 }}>
              <div className="k">Work surface</div>
              <div className="v">{Math.round(pl!.size_mm[0])} × {Math.round(pl!.size_mm[1])} mm
                {pl?.inlier_frac != null &&
                  <span className="hint"> (plane inliers {Math.round(pl.inlier_frac * 100)}%)</span>}</div>
              {result.mode === "frame_only" ? (
                <>
                  <div className="k">Mode</div>
                  <div className="v">Working frame only — no scan tour
                    <span className="hint"> (converted from the locked survey; no robot motion)</span></div>
                  <div className="k">Boundary</div>
                  <div className="v">{ACQUISITION_LABEL[result.acquisition_mode ?? ""]
                    ?? result.acquisition_mode ?? "—"}
                    <span className="hint"> {result.boundary_provenance}</span></div>
                  {result.captures?.length != null && (
                    <>
                      <div className="k">Captures</div>
                      <div className="v">{result.captures.join(", ")}
                        <span className="hint"> · calibration {result.calibration_id ?? "—"}</span></div>
                    </>
                  )}
                </>
              ) : result.mode === "reference" ? (
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
                  {voxelMm != null && (
                    <>
                      <div className="k">Quality</div>
                      <div className="v">{voxelMm.toFixed(1)} mm TSDF ·
                        fitted flat mesh insert ·
                        {framesPerPose} frame{framesPerPose === 1 ? "" : "s"}/target</div>
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
              {result.mode === "frame_only"
                ? "This is exactly the geometry you reviewed at lock time — the frame + rectangle are converted from the locked survey, never refitted. Insert creates them in RoboDK; nothing is added until you do."
                : result.mode === "reference"
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
          <div className="hint">{frameOnly
            ? "Prepare the working frame above — the frame + rectangle appear here for review before you insert them."
            : "Run a scan to fuse the surface and preview the proposed frame + rectangle here. "
              + "For a reference surface (too large for a tour), the rectangle appears here after Create targets."}</div>
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


// The scope chooser itself lives in the goal/scope card above (plan Task 5); this
// notice explains what the CURRENT scope means for the live reading, and hosts the
// sized-region inputs that only a declared-region scope may use (Task 3 item 4).
function SurfaceModeNotice({ gate, scope, disabled, onRegionApplied }:
  { gate: GateReading | null; scope: SurfaceScope; disabled: boolean;
    onRegionApplied: () => void }) {
  const declared = scope === "declared_region";
  const overruns = gate?.surface_mode === "crop";
  const crop = declared || overruns;
  const full = gate?.surface_mode === "full" && gate?.fully_framed === true;
  const cropSize = gate?.crop_size_mm;
  const extent = gate?.extent_mm;
  const modeText = declared
    ? "Declared work region"
    : overruns
    ? "Platform overruns the camera view"
    : full
      ? "Rectangle tracking"
      : "Surface mode";
  const detail = declared
    ? cropSize
      ? `Boundary DECLARED, not measured: the lock uses the measured centre depth plane and the ${Math.round(cropSize[0])} x ${Math.round(cropSize[1])} mm rectangle you set below.`
      : "Boundary DECLARED, not measured: the lock uses the measured centre depth plane and the rectangle size you set below."
    : overruns
      ? "The platform continues past the image edge, so its full boundary cannot be measured from here. Under Entire platform scope this cannot be locked from one view — run the guided five-position survey, or switch the scope to Declared work region."
    : full && extent
      ? `Full rectangle tracked (${Math.round(extent[0])} x ${Math.round(extent[1])} mm) — every boundary is measured. X/Y is advisory; targets use the measured rectangle center.`
      : "Entire platform: the lock uses the measured rectangle once the whole platform fits in the view. Switch to Declared work region if it never will, or if dot coverage is incomplete or jittery.";

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
      <div>
        <b>{modeText}</b>
        <span>{detail}</span>
        {/* Task 3 item 4: the sized inputs exist ONLY under an explicitly selected
            declared-region scope — an overrun under entire-platform scope must not be
            offered a crop at all. */}
        {declared && (
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

// Plan Task 3: the platform overruns the camera view under entire-platform scope.
// ONE primary action (the five-position survey, which can actually measure that
// boundary) and one explicit alternative (declare a region, which relabels the
// geometry as declared). The generic work crop is never offered as the full surface.
function LargeSurfaceNotice({ detail, busy, onSurvey, onDeclareRegion }:
  { detail: LargeSurfaceDetail; busy: boolean; onSurvey: () => void;
    onDeclareRegion: () => void }) {
  const ext = detail.extent_mm;
  return (
    <div className="verdict borderline" style={{ marginTop: 12, marginBottom: 0 }}>
      <div className="verdict-head">
        <span className="verdict-tag">LARGE SURFACE</span>
        <span>{detail.message}</span>
      </div>
      {ext && (
        <div className="hint">Visible extent so far: {Math.round(ext[0])} × {Math.round(ext[1])} mm
          — at least one edge is outside the frame, so this is a lower bound, not the
          platform size.</div>
      )}
      <div className="btn-row">
        <button onClick={onSurvey} disabled={busy}>
          Survey full platform — center + four corners
        </button>
        <button className="secondary" onClick={onDeclareRegion} disabled={busy}
                title="Measure only the plane and declare a sized rectangle on it — the boundary is then recorded as declared, not measured">
          Use a declared work region
        </button>
      </div>
    </div>
  );
}

function fmtMm(v: unknown, digits = 2) {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : null;
}

// Plan Task 5: everything the operator must be able to check BEFORE preparing a
// working frame, plus the explicit statement that preparation moves nothing.
function FramePrepPanel({ info, calib, ageS, preparing, prepared, disabled, onPrepare }:
  { info: LockedSurveyInfo; calib: ActiveCalibration | null; ageS: number;
    preparing: boolean; prepared: boolean; disabled: boolean; onPrepare: () => void }) {
  const q = info.quality ?? {};
  const declaredBoundary = !!info.boundary_provenance?.startsWith("user specified");
  const stale = ageS > LOCK_MAX_AGE_S;
  const planeRms = fmtMm(q.plane_rms_mm);
  const planeMax = fmtMm(q.plane_max_residual_mm);
  const cornerAgreement = fmtMm(q.corner_agreement_mm);
  const discrepancy = fmtMm(q.discrepancy_mm);
  const edgeRms = Array.isArray(q.edge_rms_mm)
    ? q.edge_rms_mm.map((v: number) => fmtMm(v) ?? "—").join(" / ") : null;
  const standoff = fmtMm(q.standoff_mm, 0);
  const tilt = fmtMm(q.tilt_deg, 1);
  const validFrac = typeof q.valid_frac === "number" ? Math.round(q.valid_frac * 100) : null;
  const flags: string[] = Array.isArray(q.flags) ? q.flags : [];
  return (
    <>
      <div className="req-note">
        <b>No robot movement.</b> Preparing a working frame converts the locked survey
        you reviewed straight into a frame + rectangle: no scan targets are created, no
        dry tour is needed and the KUKA does not move. Insertion into RoboDK stays a
        separate click afterwards.
      </div>
      <div className={"provenance-chip " + (declaredBoundary ? "declared" : "")}
           title="boundary provenance — how the locked work-region boundary was established">
        {info.boundary_provenance ?? "boundary provenance not claimed"}
      </div>
      <table className="metrics" style={{ marginTop: 10 }}>
        <tbody>
          <tr>
            <th>Acquisition</th>
            <td className="num">{ACQUISITION_LABEL[info.acquisition_mode]
              ?? (info.acquisition_mode || "no boundary survey on this lock")}</td>
          </tr>
          <tr>
            <th>Boundary</th>
            <td className="num">{info.boundary_provenance == null
              ? "not claimed — nothing measured or declared"
              : declaredBoundary ? "declared by operator" : "measured by camera"}</td>
          </tr>
          <tr>
            <th>Size</th>
            <td className="num">{info.size_mm
              ? `${Math.round(info.size_mm[0])} × ${Math.round(info.size_mm[1])} mm` : "—"}</td>
          </tr>
          <tr>
            <th>Calibration</th>
            <td className="num">{info.calibration_id ?? "checked at preparation"}
              {calib && <span className="hint" style={{ marginTop: 0 }}>
                {" "}applied {calib.applied_at.replace("T", " ")}
                {calib.quality?.verdict ? ` · ${calib.quality.verdict}` : ""}</span>}
              {!calib && <span className="hint" style={{ marginTop: 0 }}>
                {" "}no applied calibration run on file</span>}</td>
          </tr>
          <tr>
            <th>Plane residual</th>
            <td className="num">{planeRms ? `${planeRms} mm RMS` : "—"}
              {planeMax && ` · max ${planeMax} mm`}
              {edgeRms && <span className="hint" style={{ marginTop: 0 }}>
                {" "}edges {edgeRms} mm</span>}</td>
          </tr>
          <tr>
            <th>Corner residual</th>
            <td className="num">{cornerAgreement
              ? `${cornerAgreement} mm`
              : "not applicable — single-view boundary"}
              {discrepancy && ` · discrepancy ${discrepancy} mm`}</td>
          </tr>
          {(standoff || tilt || validFrac != null) && (
            <tr>
              <th>Measurement</th>
              <td className="num">{[standoff && `${standoff} mm standoff`,
                                   tilt && `${tilt}° tilt`,
                                   validFrac != null && `${validFrac}% valid depth`]
                                    .filter(Boolean).join(" · ")}</td>
            </tr>
          )}
          <tr>
            <th>Lock freshness</th>
            <td className="num">
              <span className={stale ? "warn-text" : "ok-text"}>
                {Math.round(ageS)} s old{stale ? ` — expired (> ${LOCK_MAX_AGE_S} s)` : ""}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
      {flags.length > 0 && (
        <div className="warn-text" style={{ marginTop: 8, fontSize: 12 }}>
          ⚠ flagged for review: {flags.join(", ")}
        </div>
      )}
      {info.warnings.length > 0 && (
        <div className="warn-text" style={{ marginTop: 6, fontSize: 12 }}>
          {info.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
        </div>
      )}
      {!info.can_prepare_frame && (
        <div className="hint">This lock carries no measured boundary survey, so there is no
          trustworthy frame to prepare. Fully frame the platform and lock again, run the
          five-position survey, or declare a work region.</div>
      )}
      {stale && (
        <div className="hint">The backend refuses a lock older than {LOCK_MAX_AGE_S} s (and any
          lock the robot has moved away from) — Reposition and lock again.</div>
      )}
      <div className="btn-row">
        <button onClick={onPrepare} disabled={disabled || preparing || !info.can_prepare_frame}>
          {preparing ? "Preparing…" : prepared ? "Prepare again" : "Prepare working frame"}
        </button>
      </div>
      {prepared && (
        <div className="ok-text" style={{ marginTop: 8, fontSize: 13 }}>
          ✓ Working frame prepared — review it below, then Insert into RoboDK.
        </div>
      )}
    </>
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
