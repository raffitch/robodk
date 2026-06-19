import { useEffect, useState } from "react";
import { moduleApi } from "../api/client";

const api = moduleApi("calibration");
const PDF_URL = "/api/modules/calibration/board.pdf";

interface BoardSpec {
  dictionary: string;
  squares_x: number;
  squares_y: number;
  square_size_mm: number;
  marker_size_mm: number;
  page: string;
  landscape: boolean;
  board_w_mm: number;
  board_h_mm: number;
  matches_config: boolean;
  pages: string[];
}

const STEPS = [
  "Print the ChArUco board (below) at 100% scale and verify the ruler.",
  "Mount it rigidly where the camera can see it — flat, no glare.",
  "Open the Tasni station in RoboDK (loads the robot, poses and tool).",
  "Make sure the Jetson camera server is up (the Camera pill turns green).",
  "Position the board in view — move to the first pose and check framing.",
  "Choose robot motion — calibration needs the real robot to move.",
  "Run — the robot visits each pose; watch the preview detect the board.",
  "Review the metrics: reprojection px, held-out validation, board consistency.",
  "Apply to the tool once the numbers look good.",
];

interface GuideProps {
  runMode: string;
  ready: boolean;
  connState: "idle" | "connecting" | "ready" | "error";
  onConnect: () => void;
  onConfigChanged: () => void;
}

export default function CalibrationGuide(
  { runMode, ready, connState, onConnect, onConfigChanged }: GuideProps,
) {
  const [page, setPage] = useState("A4");
  const [spec, setSpec] = useState<BoardSpec | null>(null);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<boolean[]>(() => STEPS.map(() => false));
  const [previewMsg, setPreviewMsg] = useState<string>("");

  const loadSpec = (p: string) =>
    api.get<BoardSpec>(`/board/spec?page=${p}`).then(setSpec).catch(() => setSpec(null));
  useEffect(() => { loadSpec(page); }, [page]);

  const toggle = (i: number) => setDone((d) => d.map((v, j) => (j === i ? !v : v)));

  const useDims = async () => {
    setBusy(true);
    try {
      await api.post("/board/use", { page });
      await loadSpec(page);
      onConfigChanged();
    } finally { setBusy(false); }
  };

  const preview = async () => {
    if (!ready) return;
    if (runMode === "run_robot" &&
        !window.confirm("This moves the real robot to the first pose. Cell clear?")) return;
    setBusy(true); setPreviewMsg("Moving to first pose…");
    try {
      const r = await api.post<{ target: string; detected: boolean; n_corners: number }>(
        "/preview", { run_mode: runMode });
      setPreviewMsg(r.detected
        ? `✓ board detected at ${r.target} (${r.n_corners} corners) — see Live preview.`
        : `✗ no board at ${r.target} — reposition the board and retry.`);
    } catch (e: any) {
      setPreviewMsg("✗ " + e.message);
    } finally { setBusy(false); }
  };

  return (
    <div className="card calib-guide">
      <h2>How to calibrate</h2>
      <ol className="checklist">
        {STEPS.map((text, i) => (
          <li key={i}>
            <input type="checkbox" checked={done[i]} onChange={() => toggle(i)} />
            <div style={{ flex: 1 }}>
              <span className={"check-txt" + (done[i] ? " done" : "")}>
                <b>{i + 1}.</b> {text}
              </span>

              {i === 0 && (
                <div className="board-tools">
                  <div className="row" style={{ gap: 10, alignItems: "flex-end" }}>
                    <div className="field">
                      <label>Page</label>
                      <select value={page} onChange={(e) => setPage(e.target.value)}>
                        {(spec?.pages ?? ["A4", "A3", "Letter"]).map((p) => <option key={p}>{p}</option>)}
                      </select>
                    </div>
                    <a className="linkbtn" href={`${PDF_URL}?page=${page}`} target="_blank" rel="noreferrer">Open PDF</a>
                    <a className="linkbtn" href={`${PDF_URL}?page=${page}&download=true`}>Download</a>
                  </div>
                  {spec && (
                    <div className="board-dims" style={{ marginTop: 8 }}>
                      {spec.squares_x}×{spec.squares_y}, square <b>{spec.square_size_mm} mm</b>,
                      marker <b>{spec.marker_size_mm} mm</b> ({spec.landscape ? "landscape" : "portrait"}).
                      {spec.matches_config ? (
                        <div className="ok-text" style={{ marginTop: 6 }}>✓ matches detection config</div>
                      ) : (
                        <div style={{ marginTop: 6 }}>
                          <div className="warn-text">⚠ detection config differs — sync so the solved
                            scale is correct (a size mismatch is NOT caught by the metrics).</div>
                          <button className="secondary" style={{ marginTop: 6 }} onClick={useDims} disabled={busy}>
                            Match config to this board
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {i === 2 && (
                <div className="board-tools">
                  <button className="secondary" onClick={onConnect} disabled={connState === "connecting"}>
                    {ready ? "Reconnect" : "Open Tasni station"}
                  </button>
                  <span style={{ marginLeft: 10 }}
                    className={ready ? "ok-text" : connState === "error" ? "warn-text" : ""}>
                    {connState === "connecting" ? "connecting…"
                      : ready ? "✓ connected"
                      : connState === "error" ? "✗ not connected" : "not connected"}
                  </span>
                </div>
              )}

              {i === 4 && (
                <div className="board-tools">
                  <button className="secondary" onClick={preview} disabled={busy || !ready}>
                    Move to first pose &amp; check framing
                  </button>
                  {!ready && <span className="hint" style={{ marginLeft: 10 }}>connect first</span>}
                  {previewMsg && <div className="board-dims" style={{ marginTop: 6 }}>{previewMsg}</div>}
                </div>
              )}
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
