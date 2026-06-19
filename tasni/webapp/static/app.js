// tasni shell — module-agnostic. Renders registered modules, owns the WebSocket,
// and hands each module an `api` bound to its own REST prefix + the event stream.
(() => {
  const listeners = new Set();          // active module's event callbacks
  window.TasniModules = window.TasniModules || {};

  // -- WebSocket event stream ------------------------------------------------
  function connectWs() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    const conn = document.getElementById("conn");
    ws.onopen = () => { conn.textContent = "link: live"; conn.className = "conn conn--on"; };
    ws.onclose = () => {
      conn.textContent = "link: reconnecting…"; conn.className = "conn conn--off";
      setTimeout(connectWs, 1500);
    };
    ws.onmessage = (msg) => {
      const event = JSON.parse(msg.data);
      listeners.forEach((cb) => { try { cb(event); } catch (e) { console.error(e); } });
    };
  }

  // -- per-module api --------------------------------------------------------
  function makeApi(moduleId, root) {
    const base = `/api/modules/${moduleId}`;
    return {
      moduleId,
      el: (sel) => root.querySelector(sel),
      els: (sel) => root.querySelectorAll(sel),
      async get(path) {
        const r = await fetch(base + path);
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        return r.json();
      },
      async post(path, body) {
        const r = await fetch(base + path, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body || {}),
        });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        return r.json();
      },
      onEvent(cb) { listeners.add(cb); },
    };
  }

  function loadScriptOnce(id, src) {
    return new Promise((resolve, reject) => {
      if (document.getElementById(id)) return resolve();
      const s = document.createElement("script");
      s.id = id; s.src = src; s.onload = resolve; s.onerror = reject;
      document.body.appendChild(s);
    });
  }

  async function activate(meta, liEl) {
    document.querySelectorAll(".module-list li").forEach((li) => li.classList.remove("active"));
    liEl.classList.add("active");
    listeners.clear();
    const panel = document.getElementById("panel");
    panel.innerHTML = await (await fetch(`/api/modules/${meta.id}/panel.html`)).text();
    await loadScriptOnce(`mod-js-${meta.id}`, `/api/modules/${meta.id}/panel.js`);
    const mod = window.TasniModules[meta.id];
    if (mod && typeof mod.init === "function") mod.init(makeApi(meta.id, panel));
  }

  async function boot() {
    connectWs();
    const { modules } = await (await fetch("/api/modules")).json();
    const list = document.getElementById("module-list");
    modules.forEach((meta, i) => {
      const li = document.createElement("li");
      li.innerHTML = `<div class="m-title">${meta.title}</div><div class="m-desc">${meta.description || ""}</div>`;
      li.onclick = () => activate(meta, li);
      list.appendChild(li);
      if (i === 0) activate(meta, li);   // open the first module by default
    });
  }

  boot().catch((e) => {
    document.getElementById("panel").innerHTML =
      `<div class="empty">Failed to load: ${e.message}</div>`;
  });
})();
