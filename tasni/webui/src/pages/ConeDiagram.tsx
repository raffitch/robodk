/**
 * Bird's-eye 3D schematic of the calibration setup, driven by the live config.
 *
 * Matches the real cell: a ChArUco board on a pedestal, with the robot's camera
 * sweeping a cone of viewpoints ABOVE it, every pose aimed down at the board
 * centre (the "seed" view is straight overhead). The cone widens visibly when the
 * half-angle grows. A tiny orthographic projector renders an elevated 3/4 view;
 * world axes: X,Y = floor, Z = up. The projection auto-fits, so nothing clips.
 */
type V3 = [number, number, number];

const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 =>
  [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V3): V3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
const lerp = (a: V3, b: V3, t: number): V3 =>
  [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
const rad = (d: number) => (d * Math.PI) / 180;

// Elevated 3/4 ("bird's-eye") view, looking down onto the cell.
const EYE = norm([0.95, -0.85, 0.72]);
const F = norm([-EYE[0], -EYE[1], -EYE[2]]);   // forward: eye -> origin
const RIGHT = norm(cross(F, [0, 0, 1]));
const UP = cross(RIGHT, F);
const raw = (p: V3): [number, number] => [dot(p, RIGHT), -dot(p, UP)];

interface Props {
  coneDeg: number;
  count: number;
  squaresX?: number;
  squaresY?: number;
}

export default function ConeDiagram({ coneDeg, count, squaresX = 8, squaresY = 6 }: Props) {
  const W = 340;
  const H = 250;
  const pad = 30;
  const cone = Math.max(5, Math.min(60, coneDeg));

  // Board lies flat (z = 0) on top of a pedestal; cameras orbit above (+Z).
  const HX = 1.5, HY = 1.05;          // board half-extents on the floor
  const Rd = 2.4;                     // camera distance above the board centre
  const Hped = 1.7, pc = 0.24, base = 0.72;   // pedestal height / column / foot

  const c00: V3 = [-HX, -HY, 0], c10: V3 = [HX, -HY, 0];
  const c11: V3 = [HX, HY, 0], c01: V3 = [-HX, HY, 0];
  const bpt = (u: number, v: number): V3 => lerp(lerp(c00, c10, u), lerp(c01, c11, u), v);

  // Viewpoints on the cone above (golden-angle spiral, denser near the seed).
  const n = Math.max(3, Math.min(count, 21));
  const GA = Math.PI * (3 - Math.sqrt(5));
  const cams = Array.from({ length: n }, (_, i) => {
    const th = rad(cone) * Math.sqrt((i + 0.5) / n);
    const ph = i * GA;
    const p: V3 = [Rd * Math.sin(th) * Math.cos(ph), Rd * Math.sin(th) * Math.sin(ph), Rd * Math.cos(th)];
    return { p, seed: i === 0 };
  });
  const ring = Array.from({ length: 64 }, (_, k): V3 => {
    const ph = (2 * Math.PI * k) / 64, th = rad(cone);
    return [Rd * Math.sin(th) * Math.cos(ph), Rd * Math.sin(th) * Math.sin(ph), Rd * Math.cos(th)];
  });
  const centre: V3 = [0, 0, 0];

  // Pedestal: a square column from the board down to a foot on the floor.
  const col = (s: number, z: number): V3[] =>
    [[-s, -s, z], [s, -s, z], [s, s, z], [-s, s, z]];
  const top = col(pc, 0), bot = col(pc, -Hped), foot = col(base, -Hped);
  const colFaces: V3[][] = [
    [top[0], top[1], bot[1], bot[0]], [top[1], top[2], bot[2], bot[1]],
    [top[2], top[3], bot[3], bot[2]], [top[3], top[0], bot[0], bot[3]],
  ];

  // Auto-fit everything into the padded viewBox.
  const all: V3[] = [c00, c10, c11, c01, centre, ...cams.map((c) => c.p), ...ring, ...bot, ...foot];
  const rs = all.map(raw);
  const xs = rs.map((r) => r[0]), ys = rs.map((r) => r[1]);
  const minx = Math.min(...xs), maxx = Math.max(...xs), miny = Math.min(...ys), maxy = Math.max(...ys);
  const s = Math.min((W - 2 * pad) / (maxx - minx), (H - 2 * pad) / (maxy - miny));
  const ox = (W - (minx + maxx) * s) / 2, oy = (H - (miny + maxy) * s) / 2;
  const T = (p: V3): [number, number] => { const [a, b] = raw(p); return [a * s + ox, b * s + oy]; };
  const P = (arr: V3[]) => arr.map((p) => T(p).join(",")).join(" ");

  const C = T(centre);
  const camDraw = cams
    .map((c) => ({ s: T(c.p), seed: c.seed, depth: dot(c.p, F) }))
    .sort((a, b) => b.depth - a.depth);   // far first
  const ringTop = ring.map(T).reduce((m, p) => (p[1] < m[1] ? p : m), [0, 1e9] as [number, number]);
  const seedS = T(cams[0].p);
  const c01S = T(c01);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
      aria-label={`Bird's-eye 3D view: a ChArUco board on a pedestal with ${n} camera viewpoints sweeping a cone of half-angle ${Math.round(cone)} degrees above it, each aimed down at the board`}
      style={{ display: "block", maxWidth: 380 }}>
      {/* pedestal: foot + column */}
      <polygon points={P(foot)} fill="var(--muted)" opacity={0.22} />
      {colFaces.map((f, i) => (
        <polygon key={i} points={P(f)} fill="var(--muted)" opacity={0.16}
          stroke="var(--muted)" strokeOpacity={0.3} />
      ))}

      {/* board plane: checker cells + outline */}
      {Array.from({ length: squaresY }, (_, j) =>
        Array.from({ length: squaresX }, (_, i) =>
          (i + j) % 2 === 0 ? null : (
            <polygon key={`${i}-${j}`} fill="var(--text)" opacity={0.18}
              points={P([bpt(i / squaresX, j / squaresY), bpt((i + 1) / squaresX, j / squaresY),
                bpt((i + 1) / squaresX, (j + 1) / squaresY), bpt(i / squaresX, (j + 1) / squaresY)])} />
          )))}
      <polygon points={P([c00, c10, c11, c01])} fill="none" stroke="var(--text)" strokeOpacity={0.6} strokeWidth={1.2} />

      {/* cone above: rim ellipse + edge lines from the board centre */}
      <polygon points={P(ring)} fill="var(--accent)" opacity={0.08} />
      <polygon points={P(ring)} fill="none" stroke="var(--accent)" strokeOpacity={0.35} />
      {[0, 16, 32, 48].map((k) => (
        <line key={k} x1={C[0]} y1={C[1]} x2={T(ring[k])[0]} y2={T(ring[k])[1]}
          stroke="var(--accent)" strokeOpacity={0.22} />
      ))}

      {/* sightlines + viewpoint dots (near ones on top) */}
      {camDraw.map((c, i) => (
        <line key={i} x1={c.s[0]} y1={c.s[1]} x2={C[0]} y2={C[1]}
          stroke="var(--border)" strokeWidth={0.7} strokeOpacity={0.5} />
      ))}
      {camDraw.map((c, i) => (
        <circle key={i} cx={c.s[0]} cy={c.s[1]} r={c.seed ? 5 : 3.4}
          fill={c.seed ? "var(--ok)" : "var(--accent)"} />
      ))}

      {/* labels */}
      <text x={ringTop[0]} y={ringTop[1] - 7} fontSize={12} fill="var(--accent)" textAnchor="middle">
        ±{Math.round(cone)}° cone
      </text>
      <text x={seedS[0] + 8} y={seedS[1] + 3} fontSize={11} fill="var(--ok)">seed view</text>
      <text x={c01S[0]} y={c01S[1] + 13} fontSize={11} fill="var(--muted)" textAnchor="middle">board on pedestal</text>
    </svg>
  );
}
