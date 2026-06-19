import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { apiGet, type ModuleMeta } from "../api/client";
import { useHealth } from "../api/useHealth";
import StatusPill from "../components/StatusPill";

interface Run { module: string; stamp: string; path: string; }

export default function Home() {
  const nav = useNavigate();
  const health = useHealth();
  const [modules, setModules] = useState<ModuleMeta[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);

  useEffect(() => {
    apiGet<{ modules: ModuleMeta[] }>("/api/modules").then((d) => setModules(d.modules)).catch(() => {});
    apiGet<{ runs: Run[] }>("/api/runs").then((d) => setRuns(d.runs)).catch(() => {});
  }, []);

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
