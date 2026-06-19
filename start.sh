#!/usr/bin/env bash
# Start the tasni control panel.
#
#   ./start.sh          dev  — FastAPI (:8000) + Vite (:5173, hot reload).  Open :5173
#   ./start.sh prod     build the React app, then serve everything from FastAPI (:8000)
#
# On Windows run from Git Bash. Python launcher + node/npm must be on PATH.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-py -3.10}"
WEBUI="tasni/webui"
MODE="${1:-dev}"

ensure_deps() {
  if [ ! -d "$WEBUI/node_modules" ]; then
    echo "[start] installing web UI deps…"
    (cd "$WEBUI" && npm install)
  fi
}

if [ "$MODE" = "prod" ]; then
  ensure_deps
  echo "[start] building web UI…"
  (cd "$WEBUI" && npm run build)
  echo "[start] serving on http://127.0.0.1:8000"
  exec $PY -m tasni --port 8000
fi

# dev: backend + Vite together; kill the backend when Vite exits.
ensure_deps
echo "[start] backend → http://127.0.0.1:8000"
$PY -m tasni --port 8000 &
BACKEND=$!
trap 'kill $BACKEND 2>/dev/null || true' EXIT INT TERM

echo "[start] UI (dev) → http://127.0.0.1:5173"
(cd "$WEBUI" && npm run dev)
