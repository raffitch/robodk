// Calibration module panel controller. Registered for the shell to init().
window.TasniModules = window.TasniModules || {};
window.TasniModules["calibration"] = {
  init(api) {
    const el = api.el.bind(api);
    const run = el("#run"), cancel = el("#cancel"), apply = el("#apply");
    const log = el("#log"), status = el("#status"), bar = el("#bar");

    const addLog = (msg, cls) => {
      const line = document.createElement("div");
      if (cls) line.className = cls;
      line.textContent = msg;
      log.appendChild(line);
      log.scrollTop = log.scrollHeight;
    };
    const setRunning = (on) => {
      run.disabled = on; cancel.disabled = !on;
      if (on) apply.disabled = true;
    };

    // -- load config + tools -------------------------------------------------
    api.get("/config").then((c) => {
      el("#cfg").innerHTML = [
        ["Robot", c.robot],
        ["Run mode", c.run_mode === "run_robot"
          ? '<span class="badge bad">real robot</span>'
          : '<span class="badge good">simulate</span>'],
        ["Camera", `${c.camera.ip}:${c.camera.port} @ ${c.camera.resolution}`],
        ["Board", `${c.board.squares_x}×${c.board.squares_y}, ` +
          `${c.board.square_size_mm}/${c.board.marker_size_mm} mm, ${c.board.dictionary}`],
      ].map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join("");
      el("#prefix").textContent = c.target_prefix + "*";
      el("#holdout").value = c.calibration.holdout_count;
      el("#refine").checked = c.calibration.refine;
    }).catch((e) => addLog("config error: " + e.message, "err"));

    api.get("/tools").then(({ tools }) => {
      el("#tool").innerHTML = tools.length
        ? tools.map((t) => `<option>${t}</option>`).join("")
        : '<option value="">(no tools — is RoboDK open?)</option>';
    }).catch((e) => {
      el("#tool").innerHTML = '<option value="">(RoboDK unavailable)</option>';
      addLog("tools: " + e.message, "err");
    });

    // -- actions -------------------------------------------------------------
    run.onclick = async () => {
      log.innerHTML = ""; el("#metrics").innerHTML = "";
      bar.style.width = "0%"; status.textContent = "starting…";
      setRunning(true);
      try {
        await api.post("/run", {
          tool_name: el("#tool").value || null,
          holdout_count: parseInt(el("#holdout").value, 10),
          refine: el("#refine").checked,
        });
      } catch (e) { addLog("run: " + e.message, "err"); setRunning(false); }
    };
    cancel.onclick = () => api.post("/cancel").catch(() => {});
    apply.onclick = async () => {
      try {
        const r = await api.post("/apply");
        addLog(`applied calibration to tool "${r.tool}".`);
        apply.disabled = true;
      } catch (e) { addLog("apply: " + e.message, "err"); }
    };

    // -- live events ---------------------------------------------------------
    api.onEvent((ev) => {
      if (ev.type === "progress") {
        const { step, total, message } = ev.payload;
        bar.style.width = total ? `${Math.round((step / total) * 100)}%` : "0%";
        status.textContent = `${step}/${total}  ${message}`;
      } else if (ev.type === "log") {
        addLog(ev.payload.message);
      } else if (ev.type === "frame") {
        el("#preview").src = "data:image/jpeg;base64," + ev.payload.jpeg_b64;
      } else if (ev.type === "result") {
        renderMetrics(el("#metrics"), ev.payload.result);
        status.textContent = "done";
        bar.style.width = "100%";
        setRunning(false);
        apply.disabled = !(ev.payload.result && ev.payload.result.can_apply);
      } else if (ev.type === "error") {
        addLog("ERROR: " + ev.payload.message, "err");
        status.textContent = "error";
        setRunning(false);
      } else if (ev.type === "status" && ev.payload.status === "cancelled") {
        addLog("cancelled.");
        status.textContent = "cancelled";
        setRunning(false);
      }
    });
  },
};

// reproj-error quality bands (px), tuned for a D435i color stream.
function band(px) {
  if (px < 1.0) return "good";
  if (px < 3.0) return "warn";
  return "bad";
}
function renderMetrics(root, result) {
  if (!result || !result.report) { root.innerHTML = "<div class='hint'>no result.</div>"; return; }
  const r = result.report;
  const rows = [];
  rows.push(["Solver", "TSAI" + (r.refined ? " + reprojection refinement" : ""), ""]);
  const t = r.train;
  rows.push([`Train fit (${t.n_views} poses)`,
    `RMS ${t.rms_px.toFixed(3)} px · max ${t.max_px.toFixed(3)} px`, band(t.rms_px)]);
  if (r.validation) {
    const v = r.validation;
    rows.push([`Held-out validation (${v.n_views} poses)`,
      `RMS ${v.rms_px.toFixed(3)} px · max ${v.max_px.toFixed(3)} px`, band(v.rms_px)]);
  }
  const bc = r.board_consistency_mm;
  rows.push(["Board consistency",
    `RMS ${bc.rms.toFixed(3)} mm · max ${bc.max.toFixed(3)} mm`, ""]);

  let html = "<table class='metrics'><tbody>";
  for (const [k, v, b] of rows) {
    const badge = b ? ` <span class="badge ${b}">${b}</span>` : "";
    html += `<tr><th>${k}</th><td class="num">${v}${badge}</td></tr>`;
  }
  html += "</tbody></table>";
  if (result.n_skipped && result.n_skipped.length)
    html += `<div class="hint">Skipped (no board): ${result.n_skipped.join(", ")}</div>`;
  html += `<div class="hint">Artifacts: <code>${result.run_dir}</code></div>`;
  root.innerHTML = html;
}
