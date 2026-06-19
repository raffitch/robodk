"""Launch the tasni web app:  py -3.10 -m tasni  (then open the printed URL)."""
from __future__ import annotations

import argparse

import uvicorn

from .core.config import load_config


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="tasni control panel (web)")
    ap.add_argument("--host", default=cfg.web.host)
    ap.add_argument("--port", type=int, default=cfg.web.port)
    ap.add_argument("--reload", action="store_true", help="dev autoreload")
    args = ap.parse_args(argv)

    print(f"tasni -> http://{args.host}:{args.port}")
    if args.reload:
        uvicorn.run("tasni.webapp.server:create_app", host=args.host,
                    port=args.port, reload=True, factory=True)
    else:
        from .webapp.server import create_app
        uvicorn.run(create_app(cfg), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
