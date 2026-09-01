import { useCallback, useEffect, useState } from "react";
import { apiGet } from "../api/client";

export interface FilterChain {
  available: boolean;
  state: "ok" | "in_use" | "offline" | "refused";
  detail?: string;
  arm?: string;
  stock?: boolean;
  filters?: string[];
  options?: Record<string, number | null>;
}

// Which depth-filter arm the camera is ACTUALLY on, read back off the device.
//
// Why this exists: a runtime filter override dies on restart, and the Jetson's
// auto-pull timer restarts the camera whenever `server/` changes — so an A/B arm
// can silently revert to the unit file's default in the middle of a sweep.
// Without this, that is only discoverable afterwards, by reading each take's
// archived `filter_options`, i.e. after the robot time is already spent.
//
// Deliberately quiet when the chain is stock (the normal case) and loud when it
// is not: a non-stock chain means measurements are not comparable with anything
// captured before, which is the thing worth interrupting someone about.
//
// Read-only by design. Changing the chain stays a deliberate act at the terminal
// (`tools/camera_set.py`), because a write retires the camera generation — a
// mis-click mid-capture would kill the take.
export default function DepthChainBadge({ refreshKey }: { refreshKey?: unknown }) {
  const [chain, setChain] = useState<FilterChain | null>(null);
  const [loading, setLoading] = useState(false);

  const read = useCallback(() => {
    setLoading(true);
    apiGet<FilterChain>("/api/camera/filter-chain")
      .then(setChain)
      .catch(() => setChain({ available: false, state: "offline" }))
      .finally(() => setLoading(false));
  }, []);

  // On mount, and again whenever `refreshKey` changes — the caller passes the
  // job-running flag, so the chain is re-read as soon as the camera is free
  // again. It cannot be read at all while a capture holds the unicast server.
  useEffect(() => { read(); }, [read, refreshKey]);

  if (!chain) return null;

  // Stock is the overwhelmingly common case; say so briefly and move on.
  if (chain.available && chain.stock) {
    return (
      <span className="pill" title={`depth filters: ${chain.filters?.join(" → ")}`}>
        <span className="dot ok" />
        depth chain: stock
      </span>
    );
  }

  if (chain.available) {
    const opts = Object.entries(chain.options ?? {})
      .map(([k, v]) => `${k}=${v ?? "off"}`).join("\n");
    return (
      <button
        type="button"
        onClick={read}
        className="pill"
        style={{ borderColor: "#b8862b", background: "#2a2213", cursor: "pointer" }}
        title={`NOT the stock chain — measurements are not comparable with takes\n` +
               `captured under stock. Restore with:\n` +
               `  py -3.10 tools/camera_set.py --restore\n\n${opts}`}
      >
        <span className="dot bad" />
        depth chain: {chain.arm} — not stock
      </button>
    );
  }

  // Unavailable is not a fault: mid-capture the camera is legitimately busy, and
  // probing it would steal the frame the capture is waiting for.
  const why = chain.state === "in_use" ? "camera busy" : chain.state;
  return (
    <button type="button" onClick={read} className="pill"
            style={{ cursor: "pointer" }}
            title={`${chain.detail ?? why} — click to re-read`}>
      <span className="dot unknown" />
      depth chain: {loading ? "reading…" : why}
    </button>
  );
}
