import { useEffect, useState } from "react";
import { apiGet } from "./client";

export interface Health {
  robodk: { ok: boolean | null; detail: string };
  camera: { ok: boolean | null; detail: string };
  job: { status: string; running: boolean };
}

// Polls /api/health. Pauses to a slower cadence is unnecessary — the backend
// already skips the camera probe while a job runs.
export function useHealth(intervalMs = 4000): Health | null {
  const [health, setHealth] = useState<Health | null>(null);
  useEffect(() => {
    let alive = true;
    const tick = () =>
      apiGet<Health>("/api/health").then((h) => alive && setHealth(h)).catch(() => {});
    tick();
    const t = setInterval(tick, intervalMs);
    return () => { alive = false; clearInterval(t); };
  }, [intervalMs]);
  return health;
}
