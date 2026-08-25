# `models/` — SAM weights for the live scan work boundary

The scan module's live blue work-rectangle is segmented by a **point-prompted SAM model**
running host-side (`tasni/modules/scan/sam_boundary.py`). This directory holds its ONNX
weights.

**The weight files are NOT committed** (`.gitignore` excludes `*.onnx` / `*.pt`). That keeps
the repo + the Jetson clone light (the Jetson never runs SAM) and avoids vendoring
non-commercial weights into the repo. Fetch them on setup:

```powershell
py -3.10 -m pip install -e .[sam]        # onnxruntime (host inference)
py -3.10 tools/download_sam.py           # EdgeSAM ONNX (default) -> models/
```

## Which model / licensing (read before shipping)

| Model | Files | License | Notes |
|---|---|---|---|
| **EdgeSAM** (default) | `edge_sam_encoder.onnx`, `edge_sam_decoder.onnx` | **S-Lab 1.0 — non-commercial research only** | Ready-to-run ONNX; validated on the real green-mat cell (hugged the mat, score ~0.98). Fine for internal eval; a legal risk for a commercial product. |
| **MobileSAM** | `mobile_sam.encoder.onnx`, `mobile_sam.decoder.onnx` | **Apache-2.0** (commercial-friendly) | Ships as `mobile_sam.pt`; needs a one-time ONNX export (`tools/download_sam.py --model mobilesam` downloads the `.pt` and prints the steps). Standard SAM decoder — `sam_boundary.py` reads the graph signature, so it drops in by pointing config at the files. |

`sam_boundary.py` is **model-agnostic**: it reads the encoder input size and the decoder's
input/output names off the ONNX graph, so EdgeSAM's simplified decoder and MobileSAM's
standard SAM decoder both work. Swap models via `tasni.config.json`:

```jsonc
"scan": {
  "boundary_engine": "sam_then_color",          // "color" | "sam" | "sam_then_color"
  "sam_encoder_file": "edge_sam_encoder.onnx",  // or mobile_sam.encoder.onnx
  "sam_decoder_file": "edge_sam_decoder.onnx"
}
```

If the weights (or `onnxruntime`) are absent, the scan module logs once and falls back to
the classical colour boundary (`boundary_engine="sam_then_color"`) — the app still runs.
