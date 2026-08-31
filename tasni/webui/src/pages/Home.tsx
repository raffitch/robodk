import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { apiDelete, apiGet, type ApiError, type ModuleMeta } from "../api/client";
import { useHealth } from "../api/useHealth";
import StatusPill from "../components/StatusPill";

interface Run {
  module: string; stamp: string; path: string;
  files?: number; bytes?: number; active?: boolean;
}
interface ActiveRun {
  module: string; run_id: string | null; applied_at: string; tool: string;
  source: string; refined?: boolean | null; method?: string | null;
  quality?: { verdict?: string | null; train_rms_px?: number | null;
              val_rms_px?: number | null; board_consistency_rms_mm?: number | null };
}

// Run folders are stamped `YYYYMMDD-HHMMSS[-suffix]`, which is sortable but not
// readable at a glance. Parsed here rather than added to the API because the
// stamp already carries it -- the backend has nothing extra to send. Anything
// that does not match the pattern falls through to the raw stamp rather than
// rendering "Invalid Date".
const STAMP = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})/;
const fmtStamp = (stamp: string): string | null => {
  const m = STAMP.exec(stamp);
  if (!m) return null;
  const [, y, mo, d, h, mi] = m;
  const dt = new Date(+y, +mo - 1, +d, +h, +mi);
  if (Number.isNaN(dt.getTime())) return null;
  return dt.toLocaleString(undefined, {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
};

const fmtSize = (b?: number) =>
  b == null ? "" : b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB`
    : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB`
    : b >= 1e3 ? `${Math.round(b / 1e3)} kB` : `${b} B`;

export default function Home() {
  const nav = useNavigate();
  const { health } = useHealth();
  const [modules, setModules] = useState<ModuleMeta[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [calib, setCalib] = useState<ActiveRun | null>(null);
  const [limit, setLimit] = useState(20);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);
  const [runNote, setRunNote] = useState<string | null>(null);

  const loadCalib = useCallback(() => {
    apiGet<{ active: ActiveRun | null }>("/api/runs/active?module=calibration")
      .then((d) => setCalib(d.active)).catch(() => {});
  }, []);

  // Re-read the list after a delete rather than splicing it client-side: the limit
  // means dropping rows can reveal older ones, and `active` may have moved.
  const loadRuns = useCallback(() => {
    apiGet<{ runs: Run[] }>(`/api/runs?limit=${limit}`)
      .then((d) => {
        setRuns(d.runs);
        const alive = new Set(d.runs.map((r) => r.path));
        setPicked((prev) => new Set([...prev].filter((k) => alive.has(k))));
      })
      .catch(() => {});
  }, [limit]);

  useEffect(() => {
    apiGet<{ modules: ModuleMeta[] }>("/api/modules").then((d) => setModules(d.modules)).catch(() => {});
    loadCalib();
  }, [loadCalib]);

  useEffect(() => { loadRuns(); }, [loadRuns]);

  const toggle = (path: string) =>
    setPicked((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path); else next.add(path);
      return next;
    });

  const selected = runs.filter((r) => picked.has(r.path));
  const selectedBytes = selected.reduce((n, r) => n + (r.bytes ?? 0), 0);

  // One run -> one DELETE. `force` is only ever sent after the operator answers a
  // second prompt naming the run that is applied to the cell.
  const removeOne = (r: Run, force: boolean) =>
    apiDelete(`/api/runs/${encodeURIComponent(r.module)}/${encodeURIComponent(r.stamp)}`
              + (force ? "?force=true" : ""));

  async function deleteSelected() {
    if (selected.length === 0) return;
    const names = selected.slice(0, 8).map((r) => `  ${r.module} / ${r.stamp}`).join("\n")
      + (selected.length > 8 ? `\n  ...and ${selected.length - 8} more` : "");
    if (!window.confirm(
      `Permanently delete ${selected.length} run folder(s) and every file in them`
      + `${selectedBytes ? ` (${fmtSize(selectedBytes)})` : ""}?\n\n${names}`
      + `\n\nThis cannot be undone.`)) return;

    setDeleting(true);
    setRunNote(null);
    let freed = 0, gone = 0;
    const failed: string[] = [];
    const blocked: Run[] = [];
    for (const r of selected) {
      try {
        await removeOne(r, false);
        gone += 1;
        freed += r.bytes ?? 0;
      } catch (e) {
        const err = e as ApiError;
        if (err.detail?.code === "run_is_active") blocked.push(r);
        else failed.push(`${r.module}/${r.stamp}: ${err.message}`);
      }
    }
    // The applied run gets its own answer, one at a time -- never folded into a bulk
    // "yes" the operator gave for a pile of old runs.
    for (const r of blocked) {
      if (window.confirm(
        `${r.module} / ${r.stamp} is the run currently applied to the cell.\n\n`
        + `Deleting it leaves the cell's settings in place but throws away the files`
        + ` behind them (it can no longer be re-applied or re-read).\n\nDelete anyway?`)) {
        try {
          await removeOne(r, true);
          gone += 1;
          freed += r.bytes ?? 0;
        } catch (e) {
          failed.push(`${r.module}/${r.stamp}: ${(e as ApiError).message}`);
        }
      }
    }
    setDeleting(false);
    setRunNote(`Deleted ${gone} run${gone === 1 ? "" : "s"}`
      + (freed ? `, freed ${fmtSize(freed)}` : "")
      + (failed.length ? ` — ${failed.length} failed: ${failed.join("; ")}` : "."));
    loadRuns();
    loadCalib();
  }

  // "cell calibrated: <date> · <verdict>" — provenance of the live calibration.
  const calibLine = calib
    ? `Cell calibrated: ${calib.applied_at.replace("T", " ")}`
      + (calib.quality?.verdict ? ` · ${calib.quality.verdict}` : "")
      + (calib.quality?.val_rms_px != null
          ? ` · ${calib.quality.val_rms_px.toFixed(2)} px val`
          : calib.quality?.train_rms_px != null
            ? ` · ${calib.quality.train_rms_px.toFixed(2)} px train` : "")
      + ` · tool ${calib.tool}`
    : null;

  return (
    <div>
      <h1 className="page-title">Dashboard</h1>
      <p className="page-sub">Cell status and workflow modules. Pick a module to begin.</p>

      <div className="card">
        <h2>Cell status</h2>
        <div className="pills">
          <StatusPill label="RoboDK" ok={health?.robodk.ok} detail={health?.robodk.detail} />
          <StatusPill label="Camera" ok={health?.camera.ok} detail={health?.camera.detail} />
          <span className="pill">
            job: {health?.job.running ? "running" : (health?.job.status ?? "idle")}
          </span>
        </div>
        {calibLine && (
          <div className={"calib-stamp " + (calib?.quality?.verdict ?? "")}>
            ✓ {calibLine}
          </div>
        )}
        {!calibLine && (
          <div className="hint">Cell not calibrated yet — run the Calibration module and apply.</div>
        )}
        <div className="hint">
          RoboDK must be open with the station loaded (Target* poses + the tool) before a
          real run; the Jetson camera server listens on TCP 1024.
        </div>
      </div>

      <h2 style={{ fontSize: 15, margin: "18px 0 10px" }}>Modules</h2>
      <div className="grid">
        {modules.map((m) => (
          <div key={m.id} className="card module-card" onClick={() => nav(`/m/${m.id}`)}>
            <div className="mc-icon">{m.icon}</div>
            <div className="mc-title">{m.title}</div>
            <div className="mc-desc">{m.description}</div>
          </div>
        ))}
        {modules.length === 0 && <div className="empty">No modules registered.</div>}
      </div>

      <div className="card" style={{ marginTop: 18 }}>
        <div className="runs-head">
          <h2>Recent runs</h2>
          <div className="runs-actions">
            {runs.length > 0 && (
              <button className="mini secondary" disabled={deleting} onClick={() =>
                setPicked(picked.size === runs.length
                  ? new Set() : new Set(runs.map((r) => r.path)))}>
                {picked.size === runs.length ? "Clear selection" : "Select all"}
              </button>
            )}
            <button className="mini danger" disabled={selected.length === 0 || deleting}
                    onClick={deleteSelected}>
              {deleting ? "Deleting…"
                : `Delete${selected.length ? ` ${selected.length}` : ""}`
                  + (selectedBytes ? ` (${fmtSize(selectedBytes)})` : "")}
            </button>
          </div>
        </div>
        {runs.length === 0 ? (
          <div className="empty">No runs yet.</div>
        ) : (
          <div className="runs">
            {runs.map((r) => (
              <label className={"run" + (picked.has(r.path) ? " picked" : "")} key={r.path}>
                <input type="checkbox" checked={picked.has(r.path)}
                       disabled={deleting} onChange={() => toggle(r.path)} />
                <span className="run-module">{r.module}</span>
                <span className="run-when" title={r.stamp}>
                  {fmtStamp(r.stamp) ?? <span className="mono">{r.stamp}</span>}
                </span>
                {r.active && <span className="run-tag">applied</span>}
                <span className="run-size">{fmtSize(r.bytes)}</span>
              </label>
            ))}
          </div>
        )}
        {runNote && <div className="hint">{runNote}</div>}
        {runs.length >= limit && (
          <button className="mini secondary" style={{ marginTop: 10 }}
                  onClick={() => setLimit(limit + 40)}>Show more</button>
        )}
        <div className="hint">
          Deleting a run removes its folder under <span className="mono">runs/</span> and
          every file in it — point clouds, reports, images. Not undoable.
        </div>
      </div>
    </div>
  );
}
