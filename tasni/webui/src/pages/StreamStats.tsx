// Stream health indicator overlaid on the live camera frame: actual FPS, an
// inter-frame jitter sparkline (intermittence), and a quality dot. Measured
// client-side from frame *arrival* times — so it reflects what the operator
// actually sees end-to-end (Jetson -> backend pace -> websocket -> browser),
// including stalls and network hiccups, with no backend changes.
import { useEffect, useRef, useState } from "react";

const WINDOW_MS = 3000;   // rolling window for fps + jitter
const STALL_MS = 1200;    // no frame for this long => "no signal"
const SPARK_N = 28;       // inter-frame gaps shown in the sparkline

export interface StreamStat {
  fps: number;            // frames/sec over the rolling window
  jitterMs: number;       // stdev of inter-frame interval — the "intermittence"
  gapMs: number;          // ms since the most recent frame
  live: boolean;          // a frame arrived within STALL_MS
  quality: "good" | "fair" | "poor" | "stalled";
  gaps: number[];         // recent inter-frame intervals (ms) for the sparkline
}

const EMPTY: StreamStat = { fps: 0, jitterMs: 0, gapMs: Infinity, live: false, quality: "stalled", gaps: [] };

// Hook: call mark() on each received frame, reset() when (re)starting the stream.
// `stat` is refreshed on every frame and on a timer (so a stall is noticed even
// when frames stop arriving).
export function useStreamStats(): { mark: () => void; reset: () => void; stat: StreamStat } {
  const times = useRef<number[]>([]);
  const [stat, setStat] = useState<StreamStat>(EMPTY);

  const compute = (): StreamStat => {
    const now = performance.now();
    const t = (times.current = times.current.filter((x) => now - x <= WINDOW_MS));
    if (t.length === 0) return EMPTY;

    const gapMs = now - t[t.length - 1];
    const live = gapMs < STALL_MS;

    let fps = 0;
    const gaps: number[] = [];
    for (let i = 1; i < t.length; i++) gaps.push(t[i] - t[i - 1]);
    if (gaps.length > 0) {
      const span = t[t.length - 1] - t[0];
      fps = span > 0 ? gaps.length / (span / 1000) : 0;
    }
    const mean = gaps.length ? gaps.reduce((a, b) => a + b, 0) / gaps.length : 0;
    const jitterMs = gaps.length
      ? Math.sqrt(gaps.reduce((a, b) => a + (b - mean) ** 2, 0) / gaps.length)
      : 0;

    let quality: StreamStat["quality"];
    if (!live) quality = "stalled";
    else if (fps >= 5 && jitterMs < 120) quality = "good";
    else if (fps >= 2.5 && jitterMs < 280) quality = "fair";
    else quality = "poor";

    return { fps, jitterMs, gapMs, live, quality, gaps: gaps.slice(-SPARK_N) };
  };

  const mark = () => {
    times.current.push(performance.now());
    setStat(compute());
  };
  const reset = () => {
    times.current = [];
    setStat(EMPTY);
  };

  useEffect(() => {
    const id = setInterval(() => setStat(compute()), 400);
    return () => clearInterval(id);
  }, []);

  return { mark, reset, stat };
}

// Compact overlay badge. Sparkline bars grow taller/redder as the gap between
// frames grows, so a stuttery stream reads at a glance.
export default function StreamStats({ stat }: { stat: StreamStat }) {
  const target = stat.fps > 0 ? Math.max(1000 / stat.fps, 1) : 200;
  const cap = target * 2.5;   // a gap of 2.5x the typical interval = full-height bar
  return (
    <div className={"stream-stat " + stat.quality} title="live stream rate · inter-frame jitter">
      <span className="ss-dot" />
      <span className="ss-fps">
        {stat.live
          ? stat.gaps.length > 0
            ? <>{stat.fps.toFixed(1)}<span className="ss-unit">fps</span></>
            : "starting…"
          : "no signal"}
      </span>
      {stat.live && (
        <>
          <span className="ss-jit">±{Math.round(stat.jitterMs)}ms</span>
          <span className="ss-spark" aria-hidden>
            {stat.gaps.map((gap, i) => {
              const h = Math.max(0.12, Math.min(1, gap / cap));
              const hot = gap > target * 1.8;
              return <i key={i} style={{ height: `${(h * 100).toFixed(0)}%` }}
                        className={hot ? "hot" : ""} />;
            })}
          </span>
        </>
      )}
    </div>
  );
}
