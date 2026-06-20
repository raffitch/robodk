import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { apiGet, type ModuleMeta } from "../api/client";
import { useHealth } from "../api/useHealth";
import StatusPill from "../components/StatusPill";

interface Run { module: string; stamp: string; path: string; }
interface ActiveRun {
  module: string; run_id: string | null; applied_at: string; tool: string;
  source: string; refined?: boolean | null; method?: string | null;
  quality?: { verdict?: string | null; train_rms_px?: number | null;
              val_rms_px?: number | null; board_consistency_rms_mm?: number | null };
}

export default function Home() {
  const nav = useNavigate();
  const health = useHealth();
  const [modules, setModules] = useState<ModuleMeta[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [calib, setCalib] = useState<ActiveRun | null>(null);

  useEffect(() => {
    apiGet<{ modules: ModuleMeta[] }>("/api/modules").then((d) => setModules(d.modules)).catch(() => {});
    apiGet<{ runs: Run[] }>("/api/runs").then((d) => setRuns(d.runs)).catch(() => {});
    apiGet<{ active: ActiveRun | null }>("/api/runs/active?module=calibration")
      .then((d) => setCalib(d.active)).catch(() => {});
  }, []);

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
        <h2>Recent runs</h2>
        {runs.length === 0 ? (
          <div className="empty">No runs yet.</div>
        ) : (
          <div className="runs">
            {runs.map((r) => (
              <div className="run" key={r.path}>
                <span>{r.module}</span>
                <span className="mono">{r.stamp}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
