/**
 * Schematic of the calibration viewpoint cone, driven by the live config.
 *
 * The generator orbits the operator's aimed ("seed") view in a cone, every pose
 * aimed at the board, with roll + distance variation. This is a SIDE-VIEW
 * schematic of the intended spread (the actual poses are reachability-filtered in
 * RoboDK) — it widens visibly when the cone half-angle is increased. The viewBox
 * is sized so the fan fits without clipping for cone half-angles up to ~60°.
 */
interface Props {
  coneDeg: number;
  count: number;
}

const W = 320;
const H = 210;
const BX = 268;            // board centre x (right side)
const BY = H / 2;          // board centre y
const R0 = 96;             // nominal camera distance (px)
const EDGE = R0 * 1.18;    // cone edge ray length

export default function ConeDiagram({ coneDeg, count }: Props) {
  const cone = Math.max(5, Math.min(60, coneDeg));
  const rad = (d: number) => (d * Math.PI) / 180;

  // A point on a ray leaving the board to the left at angle `sign*cone`.
  const ray = (sign: number, r: number): [number, number] => {
    const a = rad(sign * cone);
    return [BX - r * Math.cos(a), BY + r * Math.sin(a)];
  };
  const [ux, uy] = ray(-1, EDGE);
  const [lx, ly] = ray(1, EDGE);

  // Viewpoint dots fanned across the cone with a little distance jitter; the
  // centre dot (the aimed seed view) is highlighted.
  const n = Math.max(3, Math.min(count, 21));
  const dots = Array.from({ length: n }, (_, i) => {
    const t = n === 1 ? 0.5 : i / (n - 1);
    const a = rad((2 * t - 1) * cone);
    const jig = (((i * 0.3819660113) % 1) * 2 - 1) * 0.1;
    const r = R0 * (1 + jig);
    return {
      x: BX - r * Math.cos(a),
      y: BY + r * Math.sin(a),
      seed: Math.abs(2 * t - 1) < 1e-6,
    };
  });

  const arcR = 28;
  const [atx, aty] = ray(-1, arcR);
  const [abx, aby] = ray(1, arcR);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
      aria-label={`Calibration viewpoints fanned within plus or minus ${Math.round(cone)} degrees, all aimed at the board`}
      style={{ display: "block", maxWidth: 360 }}>
      {/* the cone */}
      <polygon points={`${BX},${BY} ${ux},${uy} ${lx},${ly}`} fill="var(--accent)" opacity={0.1} />
      <line x1={BX} y1={BY} x2={ux} y2={uy} stroke="var(--accent)" strokeOpacity={0.5} />
      <line x1={BX} y1={BY} x2={lx} y2={ly} stroke="var(--accent)" strokeOpacity={0.5} />
      {/* aimed (seed) axis */}
      <line x1={BX} y1={BY} x2={BX - EDGE} y2={BY} stroke="var(--muted)" strokeDasharray="4 4" />
      {/* cone half-angle arc + label */}
      <path d={`M ${atx} ${aty} A ${arcR} ${arcR} 0 0 1 ${abx} ${aby}`}
        fill="none" stroke="var(--accent)" strokeWidth={1.2} />
      <text x={BX - arcR - 6} y={BY + 4} fontSize={12} fill="var(--accent)" textAnchor="end">
        ±{Math.round(cone)}°
      </text>
      {/* sightlines + viewpoint dots */}
      {dots.map((d, i) => (
        <g key={i}>
          <line x1={d.x} y1={d.y} x2={BX} y2={BY} stroke="var(--border)" strokeWidth={0.8} />
          <circle cx={d.x} cy={d.y} r={d.seed ? 5 : 3.4}
            fill={d.seed ? "var(--ok)" : "var(--accent)"} />
        </g>
      ))}
      {/* board */}
      <rect x={BX - 3} y={BY - 26} width={12} height={52} rx={2}
        fill="var(--panel-2)" stroke="var(--text)" />
      {[0, 1, 2, 3].map((k) => (
        <rect key={k} x={BX - 1} y={BY - 22 + k * 11} width={8} height={5.5}
          fill={k % 2 ? "var(--text)" : "var(--muted)"} opacity={0.5} />
      ))}
      <text x={BX + 14} y={BY + 4} fontSize={11} fill="var(--muted)">board</text>
      {/* seed label */}
      <text x={BX - EDGE} y={BY - 9} fontSize={11} fill="var(--ok)" textAnchor="middle">
        seed view
      </text>
    </svg>
  );
}
