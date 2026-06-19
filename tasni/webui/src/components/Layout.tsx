import { type ReactNode, useEffect, useState } from "react";
import { NavLink } from "react-router-dom";
import { apiGet, type ModuleMeta } from "../api/client";
import { useEvents } from "../api/events";
import { useHealth } from "../api/useHealth";
import StatusPill from "./StatusPill";

export default function Layout({ children }: { children: ReactNode }) {
  const [modules, setModules] = useState<ModuleMeta[]>([]);
  const { connected } = useEvents();
  const health = useHealth();

  useEffect(() => {
    apiGet<{ modules: ModuleMeta[] }>("/api/modules")
      .then((d) => setModules(d.modules))
      .catch(() => setModules([]));
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          tasni<span className="brand-sub">robotic fabrication cell</span>
        </div>
        <div className="pills">
          <StatusPill label="robodk" ok={health?.robodk.ok} detail={health?.robodk.detail} />
          <StatusPill label="camera" ok={health?.camera.ok} detail={health?.camera.detail} />
          <StatusPill label="link" ok={connected} detail="job event stream" />
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
