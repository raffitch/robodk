// Guided five-position workframe survey (spec §7): CENTER + four corners, each an
// authoritative jog->stop->measure capture, then a fitted-rectangle quality report
// the operator accepts before targets are generated. Mounted by Scan.tsx in place
// of the normal lock controls whenever the operator starts a guided survey
// (gate.surface_mode === "crop" — a platform too large for one camera view).
//
// State comes from three places that must agree: (1) this panel's own REST calls
// (POST begin/capture/recapture/finish all return the fresh survey state or, for
// finish, the quality report — applied directly, no extra round trip); (2) the
// `survey` websocket event Scan.tsx forwards (fired after every successful capture,
// so a second tab / the pendant operator's own reconnect stays in sync); (3) a ~1s
// poll fallback while mounted, matching the rest of the app's "never trust a single
// channel" pattern (see Scan.tsx's own gate-freshness handling).
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { moduleApi } from "../api/client";

const api = moduleApi("scan");

export interface SurveyState {
  step: string | null;
  accepted: string[];
  corners_base: number[][];
  warnings: string[];
}

interface SurveyReport {
  status: string;
  plane_rms_mm: number;
  plane_max_residual_mm: number;
  per_position_rms_mm: number[];
  edge_rms_mm: number[];
  parallelism_deg: number;
  perpendicularity_deg: number;
  discrepancy_mm: number;
  corner_agreement_mm: number;
  size_mm: [number, number];
  flags: string[];
  warnings: string[];
}

const STEPS = ["center", "corner1", "corner2", "corner3", "corner4", "review"] as const;
const STEP_LABEL: Record<string, string> = {
  center: "CENTER", corner1: "C1", corner2: "C2", corner3: "C3", corner4: "C4", review: "REVIEW",
};
const STEP_HINT: Record<string, string> = {
  center: "Jog the pendant to the work-surface CENTER, stop the robot, then Measure.",
  corner1: "Jog to CORNER 1 — nearest the robot base — stop the robot, then Measure.",
  corner2: "Jog clockwise to CORNER 2, stop the robot, then Measure.",
  corner3: "Jog clockwise to CORNER 3, stop the robot, then Measure.",
  corner4: "Jog clockwise to CORNER 4, stop the robot, then Measure.",
};
const POLL_MS = 1000;

interface Props {
  // Latest "survey" websocket payload Scan.tsx has forwarded (or null before the
  // first one arrives) — a fallback/resync channel, not the sole source of truth.
  event: SurveyState | null;
  // Called once after a successful POST /survey/finish AND the operator has
  // reviewed the quality report and chosen to continue — Scan.tsx re-uses its
  // existing generateTargets() flow from here (the survey already locked the
  // surface server-side; see lockAndCreateTargets's post-lock steps).
  onFinished: () => void;
  // Called after Cancel succeeds, or if a state read/event ever reports no active
  // survey (server restart, a second tab cancelling it, etc.) — Scan.tsx unmounts
  // this panel and returns to the normal lock controls.
  onCancelled: () => void;
}

export default function SurveyPanel({ event, onFinished, onCancelled }: Props) {
  const [state, setState] = useState<SurveyState | null>(null);
  const [capturing, setCapturing] = useState(false);
  const [captureError, setCaptureError] = useState<string | null>(null);
  const [recapturing, setRecapturing] = useState<string | null>(null);
  const [recaptureError, setRecaptureError] = useState<string | null>(null);
  const [finishing, setFinishing] = useState(false);
  const [finishError, setFinishError] = useState<string | null>(null);
  const [report, setReport] = useState<SurveyReport | null>(null);
  const [cancelling, setCancelling] = useState(false);
  // Once finish() succeeds the backend clears its survey object, so the very next
  // poll/ws tick would otherwise read back {step: null} and (via the guard below)
  // fire onCancelled() right on top of the report we just showed. This latches the
  // panel into "showing the report" and makes every later resync a no-op.
  const doneRef = useRef(false);
  // onCancelled() must fire at most once per unmount (a poll tick and a ws event
  // can both observe "no active survey" in the same render pass).
  const endedRef = useRef(false);
  // Scan.tsx does not memoize onCancelled (it is a plain const recreated on every
  // render, and Scan.tsx re-renders often while surveying — every live frame/gate
  // websocket event). Routing it through a ref, rather than a useCallback dependency,
  // keeps noticeEnded/refresh's own identity stable so the mount-effect and the 1s
  // poll below do not tear down and restart on every parent render (they would
  // otherwise fire far more often than the ~1s cadence ambiguity resolution #3 asks
  // for, and could machine-gun GET /survey/state during live streaming).
  const onCancelledRef = useRef(onCancelled);
  useEffect(() => { onCancelledRef.current = onCancelled; }, [onCancelled]);

  const noticeEnded = useCallback(() => {
    if (endedRef.current) return;
    endedRef.current = true;
    onCancelledRef.current();
  }, []);

  const refresh = useCallback(async () => {
    try {
      const s = await api.get<SurveyState>("/survey/state");
      if (doneRef.current) return;
      if (s.step == null) { noticeEnded(); return; }
      setState(s);
    } catch { /* transient network hiccup — the next poll/ws tick catches up */ }
  }, [noticeEnded]);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (report) return;
    const id = window.setInterval(refresh, POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh, report]);
  useEffect(() => {
    if (!event || doneRef.current) return;
    if (event.step == null) { noticeEnded(); return; }
    setState(event);
  }, [event, noticeEnded]);

  const doCapture = async () => {
    setCapturing(true); setCaptureError(null);
    try {
      const s = await api.post<SurveyState>("/survey/capture");
      setState(s);
    } catch (e: any) { setCaptureError(e.message); }
    finally { setCapturing(false); }
  };

  const doRecapture = async (kind: string) => {
    setRecapturing(kind); setRecaptureError(null);
    try {
      const s = await api.post<SurveyState>("/survey/recapture", { kind });
      setState(s);
    } catch (e: any) { setRecaptureError(e.message); }
    finally { setRecapturing(null); }
  };

  const doFinish = async () => {
    setFinishing(true); setFinishError(null);
    try {
      const r = await api.post<SurveyReport>("/survey/finish");
      doneRef.current = true;
      setReport(r);
    } catch (e: any) { setFinishError(e.message); }
    finally { setFinishing(false); }
  };

  const doCancel = async () => {
    setCancelling(true);
    try { await api.post("/survey/cancel"); }
    catch { /* best effort — even a failed cancel call still leaves the operator's
               "Cancel" intent honoured client-side below */ }
    finally { setCancelling(false); noticeEnded(); }
  };

  const currentStep = state?.step ?? null;

  return (
    <div className="survey-panel">
      <StepStrip currentStep={report ? "review" : currentStep}
                 accepted={state?.accepted ?? []} done={!!report} />
      <SurveyDiagram state={state} />
      <div className="hint" style={{ marginTop: 6 }}>
        Walk the four corners clockwise, starting nearest the robot base (C1).
      </div>

      {!state && !report && <div className="hint">Loading survey…</div>}

      {state && !report && (
        <>
          {currentStep && currentStep !== "review" && (
            <div className="hint" style={{ marginTop: 10 }}>{STEP_HINT[currentStep]}</div>
          )}

          {capturing && (
            <div className="survey-busy">
              Measuring — capturing frames and reading the robot pose. This stops the
              live preview briefly and can take several seconds; it is working, not stuck.
            </div>
          )}
          {captureError && (
            <div className="run-error">
              <span className="run-error-tag">ERROR</span>
              <span>{captureError}</span>
              <button className="run-error-x" onClick={() => setCaptureError(null)}
                      aria-label="dismiss error">✕</button>
            </div>
          )}

          {finishing && (
            <div className="survey-busy">
              Fitting the rectangle and locking the surface…
            </div>
          )}
          {finishError && (
            <div className="run-error">
              <span className="run-error-tag">ERROR</span>
              <span>{finishError}</span>
              <button className="run-error-x" onClick={() => setFinishError(null)}
                      aria-label="dismiss error">✕</button>
            </div>
          )}

          <div className="btn-row">
            {currentStep && currentStep !== "review" && (
              <button onClick={doCapture} disabled={capturing || finishing}>
                {capturing ? "Measuring…" : `Measure ${STEP_LABEL[currentStep]}`}
              </button>
            )}
            {currentStep === "review" && (
              <button onClick={doFinish} disabled={finishing || capturing}>
                {finishing ? "Locking…" : "Accept & lock"}
              </button>
            )}
            <button className="secondary" onClick={doCancel}
                    disabled={cancelling || capturing || finishing}>
              {cancelling ? "Cancelling…" : "Cancel survey"}
            </button>
          </div>

          {state.accepted.length > 0 && (
            <div className="survey-recaptures">
              <div className="hint" style={{ marginTop: 12, marginBottom: 6 }}>
                Recapture a position:
              </div>
              <div className="btn-row" style={{ marginTop: 0 }}>
                {state.accepted.map((k) => (
                  <button key={k} className="secondary mini"
                          disabled={!!recapturing || capturing || finishing}
                          onClick={() => doRecapture(k)}>
                    {recapturing === k ? "Recapturing…" : `Recapture ${STEP_LABEL[k]}`}
                  </button>
                ))}
              </div>
              {recaptureError && (
                <div className="run-error" style={{ marginTop: 8 }}>
                  <span className="run-error-tag">ERROR</span>
                  <span>{recaptureError}</span>
                  <button className="run-error-x" onClick={() => setRecaptureError(null)}
                          aria-label="dismiss error">✕</button>
                </div>
              )}
            </div>
          )}
        </>
      )}

      {report && <QualityReport report={report} onContinue={onFinished} />}
    </div>
  );
}

function StepStrip({ currentStep, accepted, done }:
  { currentStep: string | null; accepted: string[]; done: boolean }) {
  const acceptedSet = new Set(accepted);
  const nodes: ReactNode[] = [];
  STEPS.forEach((s, i) => {
    if (i > 0) {
      nodes.push(<span key={`arrow-${s}`} className="survey-step-arrow" aria-hidden="true">→</span>);
    }
    const isReview = s === "review";
    const stepDone = isReview ? done : acceptedSet.has(s);
    const isCurrent = !done && currentStep === s;
    const cls = ["", stepDone ? "done" : "", isCurrent ? "current" : ""].filter(Boolean).join(" ");
    nodes.push(
      <span key={s} className={cls.trim()}>
        {stepDone ? "✓ " : ""}{STEP_LABEL[s]}
      </span>
    );
  });
  return <div className="workflow-steps survey-steps">{nodes}</div>;
}

// Small top-down sanity check: connects the captured corners in CAPTURE order (the
// literal order the operator walked them), autoscaled from state.corners_base — NOT
// the backend's internally re-derived canonical C1..C4 (order_corners_clockwise
// re-sorts geometrically for the fit and may not match capture order 1:1). This is
// deliberately the raw capture sequence: its job is to catch an operator who visited
// corners out of order, which a geometrically "corrected" re-sort would hide.
function SurveyDiagram({ state }: { state: SurveyState | null }) {
  const W = 220, H = 180, PAD = 26;
  const cornerKinds = (state?.accepted ?? []).filter((k) => k !== "center").sort();
  const points = state?.corners_base ?? [];
  const labeled = cornerKinds
    .map((kind, i) => ({ kind, pt: points[i] as number[] | undefined }))
    .filter((e): e is { kind: string; pt: number[] } => !!e.pt && e.pt.length >= 2);

  if (labeled.length === 0) {
    return (
      <svg className="survey-diagram" viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="No corners captured yet">
        <rect x={PAD} y={PAD} width={W - PAD * 2} height={H - PAD * 2} rx={8}
              fill="none" stroke="var(--border)" strokeDasharray="6 8" />
        <text x={W / 2} y={H / 2} textAnchor="middle" fontSize={11} fill="var(--muted)">
          awaiting corner captures
        </text>
      </svg>
    );
  }

  const xs = labeled.map((e) => e.pt[0]), ys = labeled.map((e) => e.pt[1]);
  const minx = Math.min(...xs), maxx = Math.max(...xs);
  const miny = Math.min(...ys), maxy = Math.max(...ys);
  const spanx = Math.max(1, maxx - minx), spany = Math.max(1, maxy - miny);
  const scale = Math.min((W - PAD * 2) / spanx, (H - PAD * 2) / spany);
  const cx = (minx + maxx) / 2, cy = (miny + maxy) / 2;
  const project = (pt: number[]): [number, number] =>
    [W / 2 + (pt[0] - cx) * scale, H / 2 + (pt[1] - cy) * scale];
  const pts = labeled.map((e) => project(e.pt));
  const path = pts.map((p) => p.join(",")).join(" ");
  const closed = labeled.length === 4;

  return (
    <svg className="survey-diagram" viewBox={`0 0 ${W} ${H}`} role="img"
         aria-label={`Surveyed corners so far, in capture order: ${labeled.map((e) => e.kind).join(", ")}`}>
      {closed
        ? <polygon points={path} fill="rgba(76,154,255,.16)" stroke="var(--accent)" strokeWidth={2} />
        : <polyline points={path} fill="none" stroke="var(--accent)" strokeWidth={2} strokeDasharray="5 5" />}
      {labeled.map((e, i) => {
        const [x, y] = pts[i];
        return (
          <g key={e.kind}>
            <circle cx={x} cy={y} r={5} fill="var(--accent)" />
            <text x={x} y={y - 10} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--text)">
              {STEP_LABEL[e.kind]}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function fmt(v: number, digits = 2) { return Number.isFinite(v) ? v.toFixed(digits) : "—"; }

function QualityReport({ report, onContinue }:
  { report: SurveyReport; onContinue: () => void }) {
  const hasFlags = report.flags.length > 0;
  const positionLabels = ["CENTER", "C1", "C2", "C3", "C4"];
  return (
    <div className="survey-report">
      {/* §5: non-empty flags must be visually prominent, not buried in the table —
          reuses the same "verdict" banner treatment Calibration's diagnosis uses. */}
      {hasFlags && (
        <div className="verdict borderline">
          <div className="verdict-head">
            <span className="verdict-tag">{report.flags.join(", ").toUpperCase()}</span>
            <span>The survey passed its hard gates but is flagged for review.</span>
          </div>
          {report.warnings.length > 0 && (
            <ul className="verdict-causes">
              {report.warnings.map((w, i) => <li key={i}>{w}</li>)}
            </ul>
          )}
        </div>
      )}
      {!hasFlags && report.warnings.length > 0 && (
        <div className="hint">{report.warnings.join(" ")}</div>
      )}
      <table className="metrics">
        <tbody>
          <tr>
            <th>Plane fit</th>
            <td className="num">{fmt(report.plane_rms_mm)} mm RMS
              · max {fmt(report.plane_max_residual_mm)} mm</td>
          </tr>
          <tr>
            <th>Per-position plane RMS</th>
            <td className="num">
              {report.per_position_rms_mm.map((v, i) =>
                `${positionLabels[i] ?? i}: ${fmt(v)}`).join("  ·  ")} mm
            </td>
          </tr>
          <tr>
            <th>Edge RMS (4 edges)</th>
            <td className="num">{report.edge_rms_mm.map((v) => fmt(v)).join(" / ")} mm</td>
          </tr>
          <tr><th>Parallelism</th><td className="num">{fmt(report.parallelism_deg)}°</td></tr>
          <tr><th>Perpendicularity</th><td className="num">{fmt(report.perpendicularity_deg)}°</td></tr>
          <tr>
            <th>Discrepancy</th>
            <td className="num">{fmt(report.discrepancy_mm)} mm
              <span className="hint" style={{ marginTop: 0 }}> unconstrained vs constrained fit</span></td>
          </tr>
          <tr>
            <th>Corner agreement</th>
            <td className="num">{fmt(report.corner_agreement_mm)} mm
              <span className="hint" style={{ marginTop: 0 }}> surveyed corners vs fitted rectangle</span></td>
          </tr>
          <tr>
            <th>Size</th>
            <td className="num">{Math.round(report.size_mm[0])} × {Math.round(report.size_mm[1])} mm</td>
          </tr>
        </tbody>
      </table>
      <div className="ok-text" style={{ marginTop: 10, fontSize: 13 }}>
        ✓ Surface locked from the five-position survey.
      </div>
      <div className="btn-row">
        <button onClick={onContinue}>Create scan targets →</button>
      </div>
    </div>
  );
}
