import { useState } from "react";

export interface CollisionStatus {
  available: boolean;
  count: number | null;
  pairs?: string[];
  guarded_tools?: string[];
  guarded_pairs?: number;
  ignored_pairs?: string[];
  ignored_pair_count?: number;
}

interface Props {
  ready: boolean;
  busy: boolean;
  status: CollisionStatus | null;
  onRecheck: () => void;
  onIgnore: (pair: string) => Promise<void>;
  recentPairs?: string[];
}

export default function CollisionPanel({
  ready, busy, status, onRecheck, onIgnore, recentPairs = [],
}: Props) {
  const [ignoring, setIgnoring] = useState<string | null>(null);
  const shownPairs = unique([...(status?.pairs ?? []), ...recentPairs]);
  const ignored = new Set(status?.ignored_pairs ?? []);

  const ignore = async (pair: string) => {
    setIgnoring(pair);
    try { await onIgnore(pair); }
    finally { setIgnoring(null); }
  };

  return (
    <div className={"collision-chip " + (status == null ? "unknown"
      : status.available ? "ok" : "bad")}>
      <div className="collision-main">
        {status == null ? (
          <span>Collision check: <b>unknown</b> - connect, or recheck.</span>
        ) : status.available ? (
          <>
            <span>Collision checking <b>active</b>.
              {status.guarded_tools?.length
                ? ` Guarding ${status.guarded_tools.join(", ")} against the arm.`
                : ""}
              {status.count ? ` Current pose: ${status.count} colliding pair${status.count === 1 ? "" : "s"}.` : ""}
            </span>
            {shownPairs.length > 0 && (
              <div className="collision-pairs">
                {shownPairs.map((p) => (
                  <div className="collision-pair" key={p}>
                    <code>{p}</code>
                    {ignored.has(p) ? (
                      <span className="ignored-tag">ignored</span>
                    ) : (
                      <button className="secondary mini" disabled={!!ignoring}
                              onClick={() => ignore(p)}>
                        {ignoring === p ? "Ignoring..." : "Ignore pair"}
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </>
        ) : (
          <span><b>No collision map detected</b> - Create targets cannot filter colliding poses.
            Set it up in RoboDK (<b>Tools - Collision Map</b>), enable the required pairs,
            and save the station.</span>
        )}
      </div>
      <button className="secondary" onClick={onRecheck}
              disabled={busy || !ready}>
        {busy ? "Checking..." : "Recheck"}
      </button>
    </div>
  );
}

function unique(values: string[]) {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const v of values) {
    const s = String(v || "").trim();
    if (s && !seen.has(s)) {
      seen.add(s);
      out.push(s);
    }
  }
  return out;
}
