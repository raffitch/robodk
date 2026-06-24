// Step 0 — dedicated RGB intrinsic calibration (camera-only, no robot motion).
//
// The D435i ships its colour stream with ZERO distortion (Intel calibrates
// depth/IR, not the RGB lens), so the hand-eye solve silently absorbs the real
// lens distortion — the "intrinsics disagree / borderline" verdict. This panel
// drives the existing /intrinsics/* backend: the operator waves the ChArUco board
// across the whole frame, views auto-capture as new image cells are covered, then
// Solve + Apply writes K + distortion into the camera config (live + persisted).
//
// Self-contained: it owns its own WS subscription (frames + status tagged
// mode==="intrinsics"), so it never cross-feeds the aiming HUD. The parent
// suppresses the aiming stream while this capture is live (onLiveChange).
import { useEffect, useRef, useState } from "react";
import { moduleApi } from "../api/client";
import { useEvents, type JobEvent } from "../api/events";

const api = moduleApi("calibration");
const band = (px: number) => (px < 1 ? "good" : px < 3 ? "warn" : "bad");

interface IntrStatus {
  mode?: string;
  detected: boolean; n_corners: number; stable: boolean; captured: boolean;
  cell: [number, number] | null;
  count: number; coverage_pct: number;
  cells: number[][]; grid: [number, number];
  max_views: number; min_views: number; have_solve: boolean;
}
interface IntrReport {
  rms_px: number; n_views: number; coverage_pct: number; fix_k3: boolean;
  image_size: [number, number];
  K: number[][]; dist: number[];
  fx: number; fy: number; cx: number; cy: number;
  delta_fx: number; delta_fy: number; delta_cx: number; delta_cy: number;
  per_view: { rms_px: number; max_px: number; n_corners: number }[];
  cells: number[][]; grid: [number, number]; run_dir?: string;
}

export default function IntrinsicsPanel(
  { disabled, onLiveChange, onApplied }:
  { disabled?: boolean; onLiveChange?: (live: boolean) => void; onApplied?: () => void },
) {
  const { subscribe } = useEvents();
  const [open, setOpen] = useState(false);
  const [running, setRunning] = useState(false);
  const runningRef = useRef(false);
  const setRun = (v: boolean) => { runningRef.current = v; setRunning(v); onLiveChange?.(v); };
  const [st, setSt] = useState<IntrStatus | null>(null);
  const [frame, setFrame] = useState<string | null>(null);
  const [report, setReport] = useState<IntrReport | null>(null);
  const [fixK3, setFixK3] = useState(true);
  const [busy, setBusy] = useState(false);
  const [applied, setApplied] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // Rehydrate the capture progress whenever the panel opens (a reopened panel
  // shows what's already captured instead of resetting to 0).
  useEffect(() => {
    if (!open) return;
    api.get<IntrStatus>("/intrinsics/status").then(setSt).catch(() => {});
  }, [open]);

  // Frames + status ride the shared WS. React only to OUR frames (while our
  // capture is live) and to gate events tagged mode==="intrinsics", so this panel
  // and the aiming HUD never cross-feed.
  useEffect(() => subscribe((ev: JobEvent) => {
    if (ev.type === "frame") {
      if (runningRef.current) setFrame("data:image/jpeg;base64," + ev.payload.jpeg_b64);
    } else if (ev.type === "gate" && ev.payload?.mode === "intrinsics") {
      setSt(ev.payload as IntrStatus);
    }
  }), [subscribe]);

  // Free the camera if the panel unmounts / the page is left mid-capture.
  useEffect(() => () => { api.post("/intrinsics/live/stop").catch(() => {}); }, []);

  const start = async () => {
    setErr(null); setFrame(null);
    try {
      const r = await api.post<IntrStatus & { status: string }>("/intrinsics/live/start");
      setSt(r); setRun(true);   // seed the grid now; gate events refresh it per frame
    } catch (e: any) { setErr(e.message); setRun(false); }
  };
  const stop = async () => {
    try { await api.post("/intrinsics/live/stop"); } catch { /* ignore */ }
    setRun(false); setFrame(null);
  };
  const reset = async () => {
    setErr(null);
    try { setSt(await api.post<IntrStatus>("/intrinsics/reset")); setReport(null); setApplied(false); }
    catch (e: any) { setErr(e.message); }
  };
  const solve = async () => {
    setBusy(true); setErr(null);
    try {
      const r = await api.post<IntrReport>("/intrinsics/solve", { fix_k3: fixK3 });
      setReport(r); setRun(false); setApplied(false);   // solve stops capture server-side
    } catch (e: any) { setErr(e.message); }
    finally { setBusy(false); }
  };
  const apply = async () => {
    setBusy(true); setErr(null);
    try { await api.post("/intrinsics/apply"); setApplied(true); onApplied?.(); }
    catch (e: any) { setErr(e.message); }
    finally { setBusy(false); }
  };

  const filledCells = st ? st.cells.flat().filter((n) => n > 0).length : 0;
  const totalCells = st ? st.grid[0] * st.grid[1] : 0;
  const enoughViews = !!st && st.count >= st.min_views;
  const sign = (v: number) => (v >= 0 ? "+" : "") + v.toFixed(1);

  return (
    <div className="card intr-card">
      <div className="intr-head" role="button" tabIndex={0}
           onClick={() => setOpen((o) => !o)}
           onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") setOpen((o) => !o); }}>
        <h2 style={{ margin: 0 }}>
          Step 0 — Camera intrinsics
          <span className="hint" style={{ fontWeight: 400 }}> (optional · do once per camera)</span>
        </h2>
        <span className="intr-toggle">{open ? "▾" : "▸"}</span>
      </div>

      {!open && (
        <div className="hint" style={{ marginTop: 6 }}>
          Calibrate the RGB lens (K + distortion) by waving the board across the whole frame —
          camera-only, no robot motion. Clears the “intrinsics disagree” warning and improves scan
          accuracy. Click to expand.
        </div>
      )}

      {open && (<>
        <div className="hint" style={{ marginTop: 6, marginBottom: 10 }}>
          The D435i ships its colour stream with <b>zero</b> distortion (Intel calibrates depth/IR,
          not the RGB lens), so the hand-eye solve silently absorbs the real lens distortion. Capture
          many ChArUco views spread across the frame — <b>especially the corners</b> — then Solve &amp;
          Apply. The board auto-captures when held still in a new cell. The robot does not move.
        </div>

        <div className="aim-wrap">
          {frame ? <img className="preview" src={frame} alt="intrinsics capture" />
                 : <div className="preview" />}
          {!running && <div className="aim-off">camera off — press “Start capture”</div>}
        </div>

        {st && (
          <div className="intr-cov">
            <div className="intr-grid" style={{ gridTemplateColumns: `repeat(${st.grid[0]}, 1fr)` }}>
              {st.cells.flatMap((row, y) => row.map((n, x) => {
                const here = !!st.cell && st.cell[0] === x && st.cell[1] === y;
                return (
                  <div key={`${x}-${y}`}
                       className={"intr-cell" + (n > 0 ? " filled" : "") + (here ? " here" : "")}>
                    {n > 0 ? n : ""}
                  </div>
                );
              }))}
            </div>
            <div className="intr-stats">
              <div><b>{st.count}</b> views <span className="hint">/ min {st.min_views}</span></div>
              <div>coverage <b>{Math.round(st.coverage_pct * 100)}%</b>{" "}
                <span className="hint">({filledCells}/{totalCells} cells)</span></div>
              <div className="hint">
                {!running ? "press Start capture"
                  : st.captured ? "✓ captured"
                  : !st.detected ? "show the board to the camera"
                  : !st.stable ? "hold still…"
                  : "move to an empty cell / tilt the board"}
              </div>
            </div>
          </div>
        )}

        {err && <div className="warn-text" style={{ marginTop: 8 }}>{err}</div>}

        <div className="btn-row">
          {!running
            ? <button onClick={start} disabled={disabled || busy}>Start capture</button>
            : <button className="secondary" onClick={stop}>Stop capture</button>}
          <button className="secondary" onClick={reset} disabled={busy || !st?.count}>Reset</button>
          <label className="intr-k3" title="Hold the high-order radial term at 0 (recommended for the low-distortion D4xx RGB lens)">
            <input type="checkbox" checked={fixK3} onChange={(e) => setFixK3(e.target.checked)} /> fix k3
          </label>
          <button onClick={solve} disabled={busy || !enoughViews}>
            {busy ? "Solving…" : `Solve (${st?.count ?? 0} views)`}
          </button>
        </div>
        {st && !enoughViews &&
          <div className="hint">Need ≥ {st.min_views} views to solve — keep moving the board across the frame.</div>}

        {report && (
          <div className="intr-report">
            <h3 style={{ margin: "14px 0 6px" }}>Result</h3>
            <div className="kv">
              <div className="k">Fit RMS</div>
              <div className="v">
                <span className={"badge " + band(report.rms_px)}>{report.rms_px.toFixed(3)} px</span>
                <span className="hint"> · {report.n_views} views · coverage {Math.round(report.coverage_pct * 100)}% · k3 {report.fix_k3 ? "fixed" : "free"}</span>
              </div>
              <div className="k">Focal fx, fy</div>
              <div className="v">{report.fx.toFixed(1)}, {report.fy.toFixed(1)}
                <span className="hint"> · Δ {sign(report.delta_fx)}, {sign(report.delta_fy)} px vs config</span></div>
              <div className="k">Principal cx, cy</div>
              <div className="v">{report.cx.toFixed(1)}, {report.cy.toFixed(1)}
                <span className="hint"> · Δ {sign(report.delta_cx)}, {sign(report.delta_cy)} px</span></div>
              <div className="k">Distortion</div>
              <div className="v">k1 {report.dist[0].toFixed(3)} · k2 {report.dist[1].toFixed(3)} · p1 {report.dist[2].toFixed(4)} · p2 {report.dist[3].toFixed(4)} · k3 {report.dist[4].toFixed(3)}</div>
            </div>
            {report.coverage_pct < 0.6 &&
              <div className="warn-text" style={{ marginTop: 6 }}>
                ⚠ Coverage {Math.round(report.coverage_pct * 100)}% is thin — the distortion at the edges
                is under-constrained. Capture more corner cells before relying on this.
              </div>}
            <div className="btn-row">
              <button onClick={apply} disabled={busy || applied}>
                {applied ? "✓ Applied" : "Apply to camera config"}
              </button>
            </div>
            {applied &&
              <div className="ok-text" style={{ fontSize: 13 }}>
                ✓ Intrinsics applied (live + saved to tasni.config.json). Re-run the hand-eye
                calibration to clear the “intrinsics disagree” warning.
              </div>}
          </div>
        )}
      </>)}
    </div>
  );
}
