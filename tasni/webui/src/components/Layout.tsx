import { type ReactNode, useEffect, useState } from "react";
import { NavLink } from "react-router-dom";
import { apiGet, type ModuleMeta } from "../api/client";
import { useEvents } from "../api/events";
import { useHealth } from "../api/useHealth";
import DepthChainBadge from "./DepthChainBadge";
import StatusPill from "./StatusPill";

export default function Layout({ children }: { children: ReactNode }) {
  const [modules, setModules] = useState<ModuleMeta[]>([]);
  const { connected } = useEvents();
  const { health, offline } = useHealth();

  useEffect(() => {
    apiGet<{ modules: ModuleMeta[] }>("/api/modules")
      .then((d) => setModules(d.modules))
      .catch(() => setModules([]));
  }, []);

  return (
    <div className="app">
      {offline && (
        // The backend is gone (uvicorn stopped, .\start.ps1 killed, a crash).
        // Every pill would otherwise keep showing its LAST value, so the cell
        // reads healthy while nothing is listening. Block the UI rather than let
        // an operator press Run against a dead server.
        <div role="alertdialog" aria-modal="true" aria-labelledby="offline-title"
             style={{ position: "fixed", inset: 0, zIndex: 9999,
                      background: "rgba(8,12,18,.86)", backdropFilter: "blur(3px)",
                      display: "grid", placeItems: "center", padding: 24 }}>
          <div style={{ maxWidth: 460, textAlign: "center", color: "#e8ecf2",
                        background: "#161c25", border: "1px solid #34404f",
                        borderRadius: 6, padding: "26px 28px",
                        boxShadow: "0 20px 60px -20px rgba(0,0,0,.8)" }}>
            <div style={{ fontSize: 30, marginBottom: 10 }} aria-hidden="true">⚡</div>
            <h2 id="offline-title" style={{ margin: "0 0 8px", fontSize: 19 }}>
              Backend not responding
            </h2>
            <p style={{ margin: "0 0 14px", fontSize: 14, lineHeight: 1.55, color: "#9fb0c2" }}>
              The app cannot reach the Tasni server. Nothing on screen is live, and
              the cell may still be moving.
            </p>
            <p style={{ margin: 0, fontSize: 13, lineHeight: 1.6, color: "#9fb0c2" }}>
              Restart it with <span className="mono" style={{ color: "#e8ecf2" }}>
              .\start.ps1</span> in the project folder. This overlay clears by
              itself as soon as the server answers.
            </p>
          </div>
        </div>
      )}
      <header className="topbar">
        <div className="brand">
          tasni<span className="brand-sub">robotic fabrication cell</span>
        </div>
        <div className="pills">
          <StatusPill label="robodk" ok={health?.robodk.ok} detail={health?.robodk.detail} />
          <StatusPill label="camera" ok={health?.camera.ok} detail={health?.camera.detail}
            summary={health?.camera
              ? `${health.camera.route} · ${health.camera.endpoint}`
              : "checking…"} />
          <StatusPill label="link" ok={connected} detail="job event stream" />
          {/* Re-read once the camera is free again: it cannot be read while a
              capture holds the unicast server, and a job restart is exactly
              when an override may have silently reverted. */}
          <DepthChainBadge refreshKey={health?.job.running} />
        </div>
      </header>
      <div className="layout">
        <nav className="sidebar">
          <div className="side-title">Cell</div>
          <NavLink to="/" className={({ isActive }) => "navlink" + (isActive ? " active" : "")} end>
            <span className="ic">▦</span>
            <span><div className="m-title">Dashboard</div></span>
          </NavLink>
          <div className="side-title" style={{ marginTop: 14 }}>Modules</div>
          {modules.map((m) => (
            <NavLink key={m.id} to={`/m/${m.id}`}
              className={({ isActive }) => "navlink" + (isActive ? " active" : "")}>
              <span className="ic">{m.icon}</span>
              <span>
                <div className="m-title">{m.title}</div>
                <div className="m-desc">{m.description}</div>
              </span>
            </NavLink>
          ))}
        </nav>
        <main className="content">{children}</main>
      </div>
    </div>
  );
}
