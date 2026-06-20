/**
 * Bird's-eye 3D schematic of the calibration viewpoint cone, driven by the live
 * config. The board is drawn as a 3D plane; the generated viewpoints are placed
 * on the 3D cone in front of it (golden-angle spiral, denser near the seed), each
 * aimed at the board centre. It widens visibly when the cone half-angle grows.
 *
 * A tiny orthographic projector renders the scene from an elevated 3/4 view. World
 * axes: X = board normal (toward the cameras), Y = board width, Z = board up. The
 * projection auto-fits the viewBox, so no cone angle clips.
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

// Elevated 3/4 ("bird's-eye") view basis.
const EYE = norm([0.95, -0.78, 0.82]);
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
  const H = 234;
  const pad = 30;
  const cone = Math.max(5, Math.min(60, coneDeg));
  const HW = 1.45;          // board half-width (Y)
  const HH = 1.08;          // board half-height (Z)
  const Rd = 2.5;           // camera distance from the board centre

  // Board corners (x = 0 plane): bottom-left, bottom-right, top-right, top-left.
  const bl: V3 = [0, -HW, -HH], br: V3 = [0, HW, -HH];
  const tr: V3 = [0, HW, HH], tl: V3 = [0, -HW, HH];
  const bpt = (u: number, v: number): V3 => lerp(lerp(bl, br, u), lerp(tl, tr, u), v);

  // Viewpoints on the cone (matches the generator: golden-angle azimuth, polar
  // angle ~ sqrt(frac) so it's denser near the seed).
  const n = Math.max(3, Math.min(count, 21));
  const GA = Math.PI * (3 - Math.sqrt(5));
  const cams = Array.from({ length: n }, (_, i) => {
    const th = rad(cone) * Math.sqrt((i + 0.5) / n);
    const ph = i * GA;
    const p: V3 = [Rd * Math.cos(th), Rd * Math.sin(th) * Math.cos(ph), Rd * Math.sin(th) * Math.sin(ph)];
    return { p, seed: i === 0 };
  });
  // Cone-edge ring at the camera distance (the cap rim).
  const ring = Array.from({ length: 64 }, (_, k): V3 => {
    const ph = (2 * Math.PI * k) / 64;
    const th = rad(cone);
    return [Rd * Math.cos(th), Rd * Math.sin(th) * Math.cos(ph), Rd * Math.sin(th) * Math.sin(ph)];
  });
  const centre: V3 = [0, 0, 0];

  // Auto-fit: project everything, scale + offset into the padded viewBox.
  const everything: V3[] = [bl, br, tr, tl, centre, ...cams.map((c) => c.p), ...ring];
  const rs = everything.map(raw);
  const xs = rs.map((r) => r[0]), ys = rs.map((r) => r[1]);
  const minx = Math.min(...xs), maxx = Math.max(...xs);
  const miny = Math.min(...ys), maxy = Math.max(...ys);
  const s = Math.min((W - 2 * pad) / (maxx - minx), (H - 2 * pad) / (maxy - miny));
  const ox = (W - (minx + maxx) * s) / 2, oy = (H - (miny + maxy) * s) / 2;
  const T = (p: V3): [number, number] => {
    const [a, b] = raw(p);
    return [a * s + ox, b * s + oy];
  };
  const pts = (arr: V3[]) => arr.map((p) => T(p).join(",")).join(" ");

  const C = T(centre);
  const ringTop = ring.map(T).reduce((m, p) => (p[1] < m[1] ? p : m), [0, 1e9] as [number, number]);
  const camDraw = cams
    .map((c) => ({ s: T(c.p), seed: c.seed, depth: dot(c.p, F) }))
    .sort((a, b) => b.depth - a.depth);   // far first, so near dots land on top
  const tlS = T(tl);
  const seedS = T(cams[0].p);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
      aria-label={`Bird's-eye 3D view: ${n} calibration viewpoints on a cone of half-angle ${Math.round(cone)} degrees in front of the board, each aimed at the board`}
      style={{ display: "block", maxWidth: 380 }}>
      {/* board plane: checker cells + outline */}
      {Array.from({ length: squaresY }, (_, j) =>
        Array.from({ length: squaresX }, (_, i) =>
          (i + j) % 2 === 0 ? null : (
            <polygon key={`${i}-${j}`}
              points={pts([bpt(i / squaresX, j / squaresY), bpt((i + 1) / squaresX, j / squaresY),
                bpt((i + 1) / squaresX, (j + 1) / squaresY), bpt(i / squaresX, (j + 1) / squaresY)])}
              fill="var(--text)" opacity={0.16} />
          )))}
      <polygon points={pts([bl, br, tr, tl])} fill="none" stroke="var(--muted)" strokeWidth={1.2} />

      {/* cone surface: rim ellipse + a few edge lines from the board centre */}
      <polygon points={pts(ring)} fill="var(--accent)" opacity={0.08} />
      <polygon points={pts(ring)} fill="none" stroke="var(--accent)" strokeOpacity={0.35} />
      {[0, 16, 32, 48].map((k) => (
        <line key={k} x1={C[0]} y1={C[1]} x2={T(ring[k])[0]} y2={T(ring[k])[1]}
          stroke="var(--accent)" strokeOpacity={0.25} />
      ))}

      {/* sightlines (every viewpoint aims at the board centre) */}
      {camDraw.map((c, i) => (
        <line key={i} x1={c.s[0]} y1={c.s[1]} x2={C[0]} y2={C[1]}
          stroke="var(--border)" strokeWidth={0.7} strokeOpacity={0.55} />
      ))}
      {/* viewpoint dots (near ones on top) */}
      {camDraw.map((c, i) => (
        <circle key={i} cx={c.s[0]} cy={c.s[1]} r={c.seed ? 5 : 3.4}
          fill={c.seed ? "var(--ok)" : "var(--accent)"} />
      ))}

      {/* labels */}
      <text x={ringTop[0]} y={ringTop[1] - 7} fontSize={12} fill="var(--accent)" textAnchor="middle">
        ±{Math.round(cone)}° cone
      </text>
      <text x={tlS[0] - 4} y={tlS[1] - 6} fontSize={11} fill="var(--muted)" textAnchor="end">board</text>
      <text x={seedS[0]} y={seedS[1] - 9} fontSize={11} fill="var(--ok)" textAnchor="middle">seed view</text>
    </svg>
  );
}
