import { useEffect, useState } from "react";
import { apiGet } from "./client";

export interface Health {
  robodk: { ok: boolean | null; detail: string };
  camera: {
    ok: boolean | null;
    state: "connected" | "offline" | "in_use";
    detail: string;
    route: string;
    endpoint: string;
  };
  job: { status: string; running: boolean };
}

// Polls /api/health. Pausing to a slower cadence is unnecessary — the backend
// already skips the camera probe while a job runs.
//
// `offline` exists because the failure used to be swallowed by a bare
// `.catch(() => {})`: when the backend died the last good Health object simply
// stayed on screen, so every pill kept reading green and the operator had no way
// to tell a healthy cell from a dead server. Two consecutive failures are
// required so a single dropped poll cannot flash the overlay, and any success
// clears it immediately.
export function useHealth(intervalMs = 4000): { health: Health | null; offline: boolean } {
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  useEffect(() => {
    let alive = true;
    let misses = 0;
    const tick = () =>
      apiGet<Health>("/api/health")
        .then((h) => {
          if (!alive) return;
          misses = 0;
          setHealth(h);
          setOffline(false);
        })
        .catch(() => {
          if (!alive) return;
          misses += 1;
          if (misses >= 2) setOffline(true);
        });
    tick();
    const t = setInterval(tick, intervalMs);
    return () => { alive = false; clearInterval(t); };
  }, [intervalMs]);
  return { health, offline };
}
