// Fighter-jet style aiming HUD over the live camera frame. Driven purely by the
// backend "gate" event (distance/tilt/offset + per-gate booleans + camera-frame
// jog deltas + thresholds), so it needs no other config. The camera is 16:9 and
// the preview box is 16:9, so the 1280x720 viewBox lines up 1:1 with the image.
//
// Wrapped in an error boundary: a malformed/partial frame can at worst blank the
// overlay for one frame, never the whole page.
import { Component, type ReactNode } from "react";

export interface GateReading {
  detected: boolean;
  n_corners?: number;
  distance_mm?: number | null;
  tilt_deg?: number | null;
  offset?: [number, number] | null;
  gates?: { detected: boolean; distance: boolean; angle: boolean };
  ok: boolean;
  ideal_distance_mm?: number;
  distance_tol_mm?: number;
  max_tilt_deg?: number;
  move_cam?: [number, number, number] | null;  // camera-frame mm to reach ideal
  center_tol_mm?: number;
  live?: boolean;
  error?: string;
}

const W = 1280, H = 720, CX = W / 2, CY = H / 2;
const OK = "#46d160", WARN = "#e6a93a", BAD = "#f5564d";
const DIM = "rgba(125,235,160,.55)", INK = "rgba(6,11,8,.64)";
const MONO = "ui-monospace, Consolas, monospace";
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const r = (n: number) => Math.round(n);

function Hud({ gate }: { gate: GateReading | null }) {
  const detected = !!gate?.detected;
  const locked = !!gate?.ok;
  const main = locked ? OK : detected ? WARN : BAD;
  const ox = gate?.offset?.[0] ?? 0, oy = gate?.offset?.[1] ?? 0;
  const bx = clamp((0.5 + ox / 2) * W, 80, W - 80);
  const by = clamp((0.5 + oy / 2) * H, 80, H - 80);
  const offCenter = Math.hypot(bx - CX, by - CY) > 45;
  const status = gate?.error ? "NO SIGNAL"
    : locked ? "● LOCK" : detected ? "AIMING" : "SEARCHING";

  return (
    <svg className="aim-hud" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
         xmlns="http://www.w3.org/2000/svg" fontFamily={MONO}>
      <defs>
        <marker id="ah-arrow" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill={main} />
        </marker>
      </defs>

      {/* 'place the board here' guide + fixed centre reticle */}
      <rect x={CX - 300} y={CY - 225} width={600} height={450} rx={12}
            fill="none" stroke={DIM} strokeWidth={2.5} strokeDasharray="10 16" />
      <g fill="none" stroke={detected ? main : DIM} strokeWidth={3}>
        <line x1={CX - 48} y1={CY} x2={CX - 14} y2={CY} />
        <line x1={CX + 14} y1={CY} x2={CX + 48} y2={CY} />
        <line x1={CX} y1={CY - 48} x2={CX} y2={CY - 14} />
        <line x1={CX} y1={CY + 14} x2={CX} y2={CY + 48} />
        <circle cx={CX} cy={CY} r={7} fill={detected ? main : "none"} />
      </g>

      {/* fly-to vector + board lock bracket */}
      {detected && (
        <>
          {offCenter && (
            <line x1={CX} y1={CY} x2={bx} y2={by} stroke={main} strokeWidth={4}
                  markerEnd="url(#ah-arrow)" opacity={0.9} />
          )}
          <Bracket x={bx} y={by} color={main} />
        </>
      )}

      {/* status chip */}
      <rect x={26} y={24} width={status.length * 27 + 46} height={64} rx={10} fill={INK} />
      <text x={48} y={68} fontSize={44} fontWeight={800} fill={main}>{status}</text>

      {/* readouts */}
      {detected && gate && (
        <>
          <Readout y={104} label="RANGE" value={`${r(gate.distance_mm ?? 0)}`}
            unit={`mm  target ${r(gate.ideal_distance_mm ?? 450)}`}
            color={gate.gates?.distance ? OK : WARN} />
          <Readout y={200} label="TILT" value={`${(gate.tilt_deg ?? 0).toFixed(1)}`}
            unit={`deg  max ${r(gate.max_tilt_deg ?? 25)}`}
            color={gate.gates?.angle ? OK : WARN} />
        </>
      )}

      {gate?.error && (
        <text x={48} y={120} fontSize={28} fill={BAD}>{gate.error}</text>
      )}

      {/* jog guidance (camera/TOOL frame) */}
      {detected && gate?.move_cam && (
        <JogBar move={gate.move_cam} ctol={gate.center_tol_mm ?? 40}
                dtol={gate.distance_tol_mm ?? 80} />
      )}
    </svg>
  );
}

function Bracket({ x, y, color }: { x: number; y: number; color: string }) {
  const h = 85, a = 34;
  const c = (sx: number, sy: number) =>
    `M ${x + sx * h} ${y + sy * (h - a)} L ${x + sx * h} ${y + sy * h} L ${x + sx * (h - a)} ${y + sy * h}`;
  return (
    <path d={[c(-1, -1), c(1, -1), c(-1, 1), c(1, 1)].join(" ")}
          stroke={color} strokeWidth={5} fill="none" />
  );
}

function Readout({ y, label, value, unit, color }:
  { y: number; label: string; value: string; unit: string; color: string }) {
  return (
    <g>
      <rect x={26} y={y} width={384} height={84} rx={10} fill={INK} />
      <text x={44} y={y + 30} fontSize={23} fill={DIM}>{label}</text>
      <text x={44} y={y + 73} fontSize={48} fontWeight={800} fill={color}>
        {value}<tspan fontSize={25} fontWeight={500} fill={DIM} dx="12">{unit}</tspan>
      </text>
    </g>
  );
}

function JogBar({ move, ctol, dtol }:
  { move: [number, number, number]; ctol: number; dtol: number }) {
  const axes = [
    { k: "X", v: move[0], tol: ctol },
    { k: "Y", v: move[1], tol: ctol },
    { k: "Z", v: move[2], tol: dtol },
  ];
  const x0 = 40, x1 = W - 40, cw = (x1 - x0) / 3, BY = H - 116, BH = 96;
  return (
    <g>
      <rect x={x0} y={BY} width={x1 - x0} height={BH} rx={12} fill={INK} />
      <text x={x0 + 20} y={BY + 26} fontSize={20} fill={DIM}>
        JOG — TOOL frame (X right · Y down · Z forward)
      </text>
      {[1, 2].map((i) => (
        <line key={i} x1={x0 + cw * i} y1={BY + 36} x2={x0 + cw * i} y2={BY + BH - 12}
              stroke={DIM} strokeWidth={1} />
      ))}
      {axes.map((a, i) => {
        const cx = x0 + cw * (i + 0.5);
        const ok = Math.abs(a.v) <= a.tol;
        const col = ok ? OK : WARN;
        const sign = a.v >= 0 ? "+" : "-";
        return (
          <g key={a.k}>
            <text x={cx} y={BY + 76} fontSize={50} fontWeight={800} fill={col} textAnchor="middle">
              {ok ? `${a.k} OK` : `${a.k}${sign} ${r(Math.abs(a.v))}`}
              {!ok && <tspan fontSize={26} fill={DIM} dx="6">mm</tspan>}
            </text>
          </g>
        );
      })}
    </g>
  );
}

// -- error boundary: never let a bad frame blank the page; self-heal next frame.
export default class AimHud extends Component<{ gate: GateReading | null }, { err: boolean }> {
  state = { err: false };
  static getDerivedStateFromError() { return { err: true }; }
  componentDidUpdate(prev: { gate: GateReading | null }) {
    if (this.state.err && prev.gate !== this.props.gate) this.setState({ err: false });
  }
  render(): ReactNode {
    if (this.state.err) return null;
    return <Hud gate={this.props.gate} />;
  }
}
