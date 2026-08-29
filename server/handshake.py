"""Client handshake parsing for the camera server (pure; importable on the host).

One line, sent right after connect, declares the stream a client wants:

    MODE FULL V2                  depth+colour, protocol 2 (greeting, raw depth)
    MODE BURST V2                 burst capture of protocol-2 frames
    MODE COLOR [Q<n>] [H264 [B<kbps>]] [SCAN]   colour-only paths (unchanged)
    MODE TELEMETRY                scan telemetry side-channel (unchanged)

Anything else -- including NO line and the pre-V2 "MODE FULL" -- parses as a
depth request WITHOUT v2. The server refuses those: a host that did not restart
after the protocol change must fail loudly at the handshake, not misread the JSON
greeting as a frame length and hang.
"""
from __future__ import annotations

DEFAULT_H264_BITRATE_KBPS = 4000


def parse_handshake(req: bytes) -> dict:
    req = bytes(req).strip().upper()
    tokens = req.split()
    mode = "full"
    if req.startswith(b"MODE BURST"):
        mode = "burst"
    elif req.startswith(b"MODE TELEMETRY"):
        mode = "telemetry"
    elif req.startswith(b"MODE COLOR") or req == b"C":
        mode = "color"
    codec, quality, bitrate = "jpeg", None, DEFAULT_H264_BITRATE_KBPS
    for tok in tokens:
        if tok == b"H264":
            codec = "h264"
        elif tok.startswith(b"Q") and tok[1:].isdigit():
            quality = max(10, min(100, int(tok[1:])))
        elif tok.startswith(b"B") and tok[1:].isdigit():
            bitrate = max(500, min(20000, int(tok[1:])))
    return {
        "mode": mode,
        "v2": b"V2" in tokens,
        "codec": codec,
        "quality": quality,
        "bitrate": bitrate,
        "scan_telemetry": b"SCAN" in tokens,
        "depth_requested": mode in ("full", "burst"),
    }
