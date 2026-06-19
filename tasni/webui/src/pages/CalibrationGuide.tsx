import { useEffect, useState } from "react";
import { moduleApi } from "../api/client";

const api = moduleApi("calibration");
const PDF_URL = "/api/modules/calibration/board.pdf";
const PNG_URL = "/api/modules/calibration/board.png";

interface BoardSpec {
  dictionary: string;
  squares_x: number;
  squares_y: number;
  square_size_mm: number;
  marker_size_mm: number;
  board_w_mm: number;
  board_h_mm: number;
  page: string;
  landscape: boolean;
  fits: boolean;
  pages: string[];
}

const STEPS = [
  "Print the ChArUco board (below) at 100% scale and verify the ruler.",
  "Mount it rigidly where the camera can see it — flat, no glare.",
  "Open the Tasni station — checks the robot, the Realsense tool and NEUTRAL.",
  "Make sure the Jetson camera server is up (the Camera pill turns green).",
  "Check framing — move to NEUTRAL and confirm the board is detected.",
  "Run — auto-generates poses around NEUTRAL, visits each on the real robot.",
  "Review the metrics: reprojection px, held-out validation, board consistency.",
  "Apply to the Realsense tool once the numbers look good.",
];

interface GuideProps {
  ready: boolean;
  connState: "idle" | "connecting" | "ready" | "error";
  onConnect: () => void;
}

export default function CalibrationGuide({ ready, connState, onConnect }: GuideProps) {
  const [page, setPage] = useState("A4");
  const [spec, setSpec] = useState<BoardSpec | null>(null);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<boolean[]>(() => STEPS.map(() => false));
  const [previewMsg, setPreviewMsg] = useState<string>("");

  const loadSpec = (p: string) =>
    api.get<BoardSpec>(`/board/spec?page=${p}`).then(setSpec).catch(() => setSpec(null));
  useEffect(() => { loadSpec(page); }, [page]);

  const toggle = (i: number) => setDone((d) => d.map((v, j) => (j === i ? !v : v)));

  const preview = async () => {
    if (!ready) return;
    if (!window.confirm("This moves the real robot to NEUTRAL. Cell clear?")) return;
    setBusy(true); setPreviewMsg("Moving to NEUTRAL…");
    try {
      const r = await api.post<{ target: string; detected: boolean; n_corners: number }>("/preview");
      setPreviewMsg(r.detected
        ? `✓ board detected at ${r.target} (${r.n_corners} corners) — see Live preview.`
        : `✗ no board at ${r.target} — reposition the board or NEUTRAL and retry.`);
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
                  <img className="board-preview" src={PNG_URL} alt="calibration board" />
                  {spec && (
                    <div className="board-dims">
                      This exact board is what detection expects <i>and</i> what prints —
                      <b> {spec.squares_x}×{spec.squares_y}</b>, square <b>{spec.square_size_mm} mm</b>,
                      marker <b>{spec.marker_size_mm} mm</b>, {spec.dictionary}.
                      Printed size <b>{spec.board_w_mm}×{spec.board_h_mm} mm</b>.
                      <span className={spec.fits ? "ok-text" : "warn-text"} style={{ marginLeft: 6 }}>
                        {spec.fits ? `✓ fits ${page}` : `⚠ too big for ${page} — try A3`}
                      </span>
                    </div>
                  )}
                  <div className="row" style={{ gap: 10, alignItems: "flex-end", marginTop: 8 }}>
                    <div className="field">
                      <label>Paper</label>
                      <select value={page} onChange={(e) => setPage(e.target.value)}>
                        {(spec?.pages ?? ["A4", "A3", "Letter"]).map((p) => <option key={p}>{p}</option>)}
                      </select>
                    </div>
                    <a className="linkbtn" href={`${PDF_URL}?page=${page}`} target="_blank" rel="noreferrer">Open PDF</a>
                    <a className="linkbtn" href={`${PDF_URL}?page=${page}&download=true`}>Download</a>
                  </div>
                  <div className="hint">Print at 100% (Actual size). The dimensions don't change with
                    paper size — the page just needs to be big enough; verify with the 100 mm ruler.</div>
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
                    Move to NEUTRAL &amp; check the board
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
