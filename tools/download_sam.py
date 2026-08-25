"""Fetch the SAM ONNX weights for the live scan work-boundary into ``models/``.

The scan module segments the work surface under the reticle with a point-prompted model
(``tasni/modules/scan/sam_boundary.py``) so the live blue rectangle is reliable even on
low-contrast scenes the classical colour layer abstains on. The weights are NOT vendored in
the repo (keeps the clone + the Jetson pull light; the Jetson never runs SAM) — this tool
downloads them on setup.

    py -3.10 tools/download_sam.py                 # EdgeSAM (default; matches config defaults)
    py -3.10 tools/download_sam.py --model mobilesam   # MobileSAM checkpoint (Apache-2.0)

LICENSING — read before you ship:
  * EdgeSAM weights are **S-Lab License 1.0 = non-commercial research only**. Great for
    internal evaluation; a legal risk for a commercial product. (This is what the config
    defaults to because it runs out of the box.)
  * MobileSAM weights are **Apache-2.0** (clean for commercial use), but ship as a PyTorch
    ``.pt`` — exporting it to the encoder/decoder ONNX this module consumes needs the
    ``mobile_sam`` package + torch. This tool downloads the ``.pt`` and prints the export
    steps; point the config's ``sam_encoder_file``/``sam_decoder_file`` at the result.

Everything is host-side; no Jetson deploy.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

try:  # the Windows console is cp1252; keep prints ASCII-safe regardless
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MODELS = Path(__file__).resolve().parents[1] / "models"

# EdgeSAM 3x official ONNX (encoder + simplified point decoder). S-Lab 1.0 (non-commercial).
EDGESAM = {
    "edge_sam_encoder.onnx":
        "https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x_encoder.onnx",
    "edge_sam_decoder.onnx":
        "https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x_decoder.onnx",
}
# MobileSAM checkpoint (Apache-2.0). Needs a one-time ONNX export (see notes below).
MOBILESAM_PT = {
    "mobile_sam.pt":
        "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
}


def _download(name: str, url: str) -> None:
    dst = MODELS / name
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  [ok] {name} already present ({dst.stat().st_size/1e6:.1f} MB)")
        return
    print(f"  ... downloading {name}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)
    print(f"  [ok] {name} ({dst.stat().st_size/1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["edgesam", "mobilesam"], default="edgesam",
                    help="edgesam = ready-to-run ONNX (non-commercial); "
                         "mobilesam = Apache-2.0 checkpoint (needs ONNX export)")
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)

    if args.model == "edgesam":
        print("EdgeSAM ONNX  (LICENSE: S-Lab 1.0 — NON-COMMERCIAL research only)")
        for name, url in EDGESAM.items():
            _download(name, url)
        print("\nDone. The scan module's config defaults already point at these files.")
        print("Verify: py -3.10 -c \"from tasni.modules.scan.sam_boundary import _sessions; "
              "_sessions('models','edge_sam_encoder.onnx','edge_sam_decoder.onnx'); print('SAM OK')\"")
        return 0

    print("MobileSAM checkpoint  (LICENSE: Apache-2.0 — commercial-friendly)")
    for name, url in MOBILESAM_PT.items():
        _download(name, url)
    print(
        "\nMobileSAM ships as PyTorch weights. Export the ONNX pair once (needs the\n"
        "`mobile_sam` package + torch — NOT a runtime dependency of tasni):\n\n"
        "  pip install git+https://github.com/ChaoningZhang/MobileSAM.git\n"
        "  # encoder -> models/mobile_sam.encoder.onnx  (input 'image' 1x3x1024x1024)\n"
        "  # decoder -> models/mobile_sam.decoder.onnx  (standard SAM decoder)\n"
        "  #   use MobileSAM's scripts/export_onnx_model.py for the decoder, and export\n"
        "  #   model.image_encoder for the encoder.\n\n"
        "Then set in tasni.config.json (scan):\n"
        '  "sam_encoder_file": "mobile_sam.encoder.onnx",\n'
        '  "sam_decoder_file": "mobile_sam.decoder.onnx"\n'
        "sam_boundary.py reads the graph signature, so the standard SAM decoder just works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
