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
  gates?: { detected: boolean; distance: boolean; angle: boolean; framed?: boolean;
            center?: boolean; edge?: boolean; coverage?: boolean; stable?: boolean };
  ok: boolean;
  ideal_distance_mm?: number;
  distance_tol_mm?: number;
  max_tilt_deg?: number;
  move_cam?: [number, number, number] | null;  // camera-frame mm to reach ideal
  center_tol_mm?: number;
  board_area_frac?: number | null;
  min_board_area_frac?: number;
  max_board_area_frac?: number;
  stable_for_s?: number;
  stable_required_s?: number;
  yaw_a_deg?: number | null;
  edge_align_tol_deg?: number;
  // Scan standoff gate only: how to correct a surface tilt, as TOOL-frame rotations
  // (KUKA A/B/C: A=about Z, B=about Y, C=about X). Absent for the calibration board gate.
  tilt_b_deg?: number | null;
  tilt_c_deg?: number | null;
  live?: boolean;
  error?: string;
  // Survey fields (full-frame surface measurement — scan module only)
  fully_framed?: boolean | null;
  outline_uv?: Array<[number, number]> | null;   // 4 plane-rectangle corners, normalized 0-1
  visible_outline_uv?: Array<[number, number]> | null; // raw surviving depth silhouette
  points_uv?: Array<[number, number]> | null;    // decimated detected-surface dots, normalized 0-1
  grid_uv?: Array<[[number, number], [number, number]]> | null;  // metric grid segments
  grid_spacing_mm?: number | null;
  extent_mm?: [number, number] | null;           // (longer, shorter) surface size mm
  rectangle_size_mm?: [number, number] | null;   // lengths of outline edges 0->1, 1->2
  crop_size_mm?: [number, number] | null;        // bounded region for an oversized plane
  surface_mode?: "full" | "crop" | null;
  measurement_ts?: number | null;
}

const W = 1280, H = 720, CX = W / 2, CY = H / 2;
const OK = "#46d160", WARN = "#e6a93a", BAD = "#f5564d";
const DIM = "rgba(125,235,160,.55)", INK = "rgba(6,11,8,.64)";
const MONO = "ui-monospace, Consolas, monospace";
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const r = (n: number) => Math.round(n);

function projectivePoint(
  quad: Array<[number, number]>, x: number, y: number,
): [number, number] {
  const [p0, p1, p2, p3] = quad;
  const dx1 = p1[0] - p2[0], dx2 = p3[0] - p2[0];
  const dy1 = p1[1] - p2[1], dy2 = p3[1] - p2[1];
  const dx3 = p0[0] - p1[0] + p2[0] - p3[0];
  const dy3 = p0[1] - p1[1] + p2[1] - p3[1];
  const det = dx1 * dy2 - dx2 * dy1;
  if (Math.abs(det) < 1e-12 || (Math.abs(dx3) < 1e-12 && Math.abs(dy3) < 1e-12)) {
    return [
      p0[0] + x * (p1[0] - p0[0]) + y * (p3[0] - p0[0]),
      p0[1] + x * (p1[1] - p0[1]) + y * (p3[1] - p0[1]),
    ];
  }
  const g = (dx3 * dy2 - dx2 * dy3) / det;
  const h = (dx1 * dy3 - dx3 * dy1) / det;
  const a = p1[0] - p0[0] + g * p1[0];
  const b = p3[0] - p0[0] + h * p3[0];
  const d = p1[1] - p0[1] + g * p1[1];
  const e = p3[1] - p0[1] + h * p3[1];
  const den = g * x + h * y + 1;
  return [(a * x + b * y + p0[0]) / den, (d * x + e * y + p0[1]) / den];
}

function cropQuad(
  source: Array<[number, number]>,
  crop: [number, number],
  rectangleSize: [number, number],
): Array<[number, number]> {
  const edge0IsLong = rectangleSize[0] >= rectangleSize[1];
  const crop0 = edge0IsLong ? crop[0] : crop[1];
  const crop1 = edge0IsLong ? crop[1] : crop[0];
  const sx = Math.min(1, crop0 / Math.max(rectangleSize[0], 1e-9));
  const sy = Math.min(1, crop1 / Math.max(rectangleSize[1], 1e-9));
  const x0 = (1 - sx) / 2, x1 = 1 - x0;
  const y0 = (1 - sy) / 2, y1 = 1 - y0;
  return [
    projectivePoint(source, x0, y0),
    projectivePoint(source, x1, y0),
    projectivePoint(source, x1, y1),
    projectivePoint(source, x0, y1),
  ];
}

function Hud({ gate, mode = "scan" }: { gate: GateReading | null; mode?: "calibration" | "scan" }) {
  const detected = !!gate?.detected;
  const locked = !!gate?.ok;
  const main = locked ? OK : detected ? WARN : BAD;
  const ox = gate?.offset?.[0] ?? 0, oy = gate?.offset?.[1] ?? 0;
  const bx = clamp((0.5 + ox / 2) * W, 80, W - 80);
  const by = clamp((0.5 + oy / 2) * H, 80, H - 80);
  const offCenter = Math.hypot(bx - CX, by - CY) > 45;
  const waitingForScanCheck = mode === "scan" && !gate;
  const calibrationGeometryReady = mode === "calibration" && !!gate?.gates
    && !!gate.gates.detected && !!gate.gates.distance && !!gate.gates.angle
    && !!gate.gates.center && !!gate.gates.coverage;
  const status = gate?.error ? "NO SIGNAL"
    : waitingForScanCheck ? "DEPTH STARTING"
    : calibrationGeometryReady && !gate?.gates?.stable ? "HOLD STEADY"
    : locked ? (mode === "scan" ? "IN RANGE" : "● LOCK")
    : detected ? "AIMING" : "SEARCHING";

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

      {/* detected-surface dots over the RGB — where depth actually landed on the plane.
          Sparse gaps here are real coverage holes (e.g. an unseen far edge). Drawn
          behind the outline/grid; capped defensively so the SVG stays light. */}
      {mode === "scan" && gate?.points_uv && gate.points_uv.length > 0 && (
        <g fill="#ff453a" opacity={0.62}>
          {gate.points_uv.slice(0, 800).map(([u, v], i) => (
            <circle key={i} cx={u * W} cy={v * H} r={2.4} />
          ))}
        </g>
      )}

      {/* survey surface overlay (outline + metric grid) — behind all other HUD elements */}
      {mode === "scan" && gate?.outline_uv && gate.outline_uv.length >= 3 && (() => {
        const source = gate.outline_uv!;
        const selected = gate.fully_framed === false && gate.crop_size_mm
          && gate.rectangle_size_mm && source.length === 4
          ? cropQuad(source, gate.crop_size_mm, gate.rectangle_size_mm)
          : source;
        const pts = source.map(([u, v]) =>
          `${(u * W).toFixed(1)},${(v * H).toFixed(1)}`).join(" ");
        const selectedPts = selected.map(([u, v]) =>
          `${(u * W).toFixed(1)},${(v * H).toFixed(1)}`).join(" ");
        const visiblePts = gate.visible_outline_uv?.map(([u, v]) =>
          `${(u * W).toFixed(1)},${(v * H).toFixed(1)}`).join(" ");
        const col = gate.fully_framed == null ? DIM
          : gate.fully_framed ? OK : WARN;
        return (
          <>
            {visiblePts && (
              <polygon points={visiblePts} fill="none" stroke={DIM} strokeWidth={1.5}
                       strokeDasharray="4 7" opacity={0.45} />
            )}
            <polygon points={pts} fill="none" stroke={col} strokeWidth={2.5}
                     strokeDasharray="10 6" opacity={0.8} />
            <polygon points={selectedPts} fill="rgba(53,194,255,.16)"
                     stroke="#35c2ff" strokeWidth={4} opacity={0.95} />
            {gate.grid_uv && gate.grid_uv.map(([[u1, v1], [u2, v2]], i) => (
              <line key={i} x1={u1 * W} y1={v1 * H} x2={u2 * W} y2={v2 * H}
                    stroke={col} strokeWidth={1} opacity={0.35} />
            ))}
          </>
        );
      })()}

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
      {waitingForScanCheck && (
        <>
          <PendingReadout y={104} label="RANGE" text="WAITING FOR DEPTH" />
          <PendingReadout y={200} label="TILT" text="WAITING FOR DEPTH" />
          <PendingReadout y={296} label="LEVEL" text="WAITING FOR DEPTH" tall />
          <PendingReadout y={440} label="FRAMED" text="FINAL SNAPSHOT" />
        </>
      )}
      {detected && gate && (
        <>
          <Readout y={104} label="RANGE" value={`${r(gate.distance_mm ?? 0)}`}
            unit={`mm  target ${r(gate.ideal_distance_mm ?? 450)}`}
            ok={!!gate.gates?.distance} />
          <Readout y={200} label="TILT" value={`${(gate.tilt_deg ?? 0).toFixed(1)}`}
            unit={`deg  max ${r(gate.max_tilt_deg ?? 25)}`}
            ok={!!gate.gates?.angle} />
          {/* Tilt-correction reference (scan only): which TOOL rotation levels it. */}
          {(gate.tilt_b_deg != null || gate.tilt_c_deg != null || gate.yaw_a_deg != null) && (
            <TiltFix a={gate.yaw_a_deg} b={gate.tilt_b_deg ?? 0} c={gate.tilt_c_deg ?? 0}
                     ok={!!gate.gates?.angle} edgeOk={gate.gates?.edge} />
          )}
          {/* Surface framing readout (survey mode only — scan module). */}
          {mode === "scan" && gate.fully_framed != null && (
            <Readout y={440} label="FRAMED"
              value={gate.extent_mm
                ? `${r(gate.extent_mm[0])}×${r(gate.extent_mm[1])}`
                : gate.fully_framed ? "FULL" : "OVER"}
              unit={gate.extent_mm ? "mm" : ""}
              ok={!!gate.gates?.framed} />
          )}
          {mode === "scan" && gate.fully_framed == null && (
            <PendingReadout y={440} label="FRAMED" text="FINAL SNAPSHOT" />
          )}
          {mode === "calibration" && gate.board_area_frac != null && (
            <Readout y={440} label="BOARD SIZE"
              value={`${Math.round(gate.board_area_frac * 100)}`}
              unit={`% image  hold ${(gate.stable_for_s ?? 0).toFixed(1)}/${(gate.stable_required_s ?? 1).toFixed(1)}s`}
              ok={!!gate.gates?.coverage && !!gate.gates?.stable} />
          )}
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

function PendingReadout({ y, label, text, tall = false }:
  { y: number; label: string; text: string; tall?: boolean }) {
  return (
    <g>
      <rect x={26} y={y} width={384} height={tall ? 132 : 84} rx={10} fill={INK} />
      <text x={44} y={y + 30} fontSize={23} fill={DIM}>{label}</text>
      <text x={44} y={y + 70} fontSize={27} fontWeight={700} fill={DIM}>
        {text}
      </text>
    </g>
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

function Readout({ y, label, value, unit, ok }:
  { y: number; label: string; value: string; unit: string; ok: boolean }) {
  // Colour-blind-safe: the ✓/✗ glyph + IN/OUT word carry the in-band state, not
  // just the green/amber colour.
  const color = ok ? OK : WARN;
  return (
    <g>
      <rect x={26} y={y} width={384} height={84} rx={10} fill={INK} />
      <text x={44} y={y + 30} fontSize={23} fill={DIM}>
        {label}<tspan dx="10" fill={color} fontWeight={800}>{ok ? "✓ IN" : "✗ OUT"}</tspan>
      </text>
      <text x={44} y={y + 73} fontSize={48} fontWeight={800} fill={color}>
        {value}<tspan fontSize={25} fontWeight={500} fill={DIM} dx="12">{unit}</tspan>
      </text>
    </g>
  );
}

function TiltFix({ a, b, c, ok, edgeOk }:
  { a?: number | null; b: number; c: number; ok: boolean; edgeOk?: boolean }) {
  // Tells the operator which TOOL rotation makes the surface fronto-parallel, in
  // KUKA A/B/C terms. B = rotate about Y (left/right), C = rotate about X (fwd/back);
  // A = rotate about Z and does NOT change tilt. Signed degrees; small = leave it.
  const color = ok ? OK : WARN;
  const dir = (v: number, neg: string, pos: string) =>
    Math.abs(v) < 1 ? "·" : v > 0 ? pos : neg;
  return (
    <g>
      <rect x={26} y={296} width={384} height={132} rx={10} fill={INK} />
      <text x={44} y={326} fontSize={23} fill={DIM}>
        LEVEL — ROTATE TOOL
        <tspan dx="10" fill={color} fontWeight={800}>{ok ? "✓ LEVEL" : "✗ TILTED"}</tspan>
      </text>
      <text x={44} y={366} fontSize={34} fontWeight={800} fill={color}>
        B {dir(b, "◀", "▶")} {r(Math.abs(b))}°
      </text>
      <text x={210} y={366} fontSize={34} fontWeight={800} fill={color}>
        C {dir(c, "▼", "▲")} {r(Math.abs(c))}°
      </text>
      {a != null && (
        <text x={44} y={402} fontSize={30} fontWeight={800}
              fill={edgeOk ? OK : WARN}>
          A {a >= 0 ? "▶" : "◀"} {r(Math.abs(a))}°
          <tspan fontSize={18} fontWeight={500} fill={DIM} dx="10">align platform edge</tspan>
        </text>
      )}
      <text x={44} y={422} fontSize={17} fill={DIM}>
        A aligns edge · B/C level the plane
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
export default class AimHud extends Component<
  { gate: GateReading | null; mode?: "calibration" | "scan" },
  { err: boolean }
> {
  state = { err: false };
  static getDerivedStateFromError() { return { err: true }; }
  componentDidUpdate(prev: { gate: GateReading | null }) {
    if (this.state.err && prev.gate !== this.props.gate) this.setState({ err: false });
  }
  render(): ReactNode {
    if (this.state.err) return null;
    return <Hud gate={this.props.gate} mode={this.props.mode} />;
  }
}
