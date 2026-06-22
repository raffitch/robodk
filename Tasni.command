#!/bin/bash
# ===== Tasni remote launcher (run on the Mac) ==========================
# Double-click to: start the headless Tasni server on the Windows cell over
# SSH/Tailscale, open it in a dedicated app window here, and stop the Windows
# server when you close that window (or this Terminal).
#
# Requires: Tailscale up on both machines, key-based SSH to Windows, and RoboDK
# already open on the Windows cell (Tasni drives the real robot/camera there).
# ----------------------------------------------------------------------
WIN="user@fablab.taile3c54a.ts.net"     # Windows SSH target (or user@100.127.132.93)
KEY="$HOME/.ssh/tasni_ed25519"          # dedicated key (passwordless)
SERVE='D:\DesktopStuff\RAFFI NO TOUCH\backuprobodk\RoboDkClaude\serve.ps1'
PORT=8000
URL="http://127.0.0.1:$PORT"
PS="powershell -NoProfile -ExecutionPolicy Bypass -File \"$SERVE\""
LOG="/tmp/tasni-ssh.log"
SSHOPTS=(-i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)

cleanup() {
  [ -n "$SSH_PID" ] && kill "$SSH_PID" 2>/dev/null
  # explicit teardown so the Windows server always stops, pty or not
  ssh "${SSHOPTS[@]}" -o ConnectTimeout=8 "$WIN" "$PS -Stop" >/dev/null 2>&1
  echo "Tasni server stopped."
}
trap cleanup EXIT INT TERM

echo "Starting Tasni on the Windows cell over Tailscale (first run rebuilds the UI)..."
# ONE plain SSH connection carries both the port-forward (-L) and the foreground
# server (it blocks on uvicorn, keeping the tunnel up). NO pseudo-tty (-t/-tt):
# backgrounding a pty SSH is what made the session close instantly.
ssh "${SSHOPTS[@]}" -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
    -L "127.0.0.1:$PORT:127.0.0.1:$PORT" "$WIN" "$PS" >"$LOG" 2>&1 &
SSH_PID=$!

# Wait until the Windows server actually ANSWERS. With -L the local port is open
# the instant SSH connects (before the server binds), so polling the bare TCP
# port opens the browser onto an empty response -- poll real HTTP instead. First
# run builds the UI, so allow ~120s; show the log if SSH dies early.
echo "Waiting for the server (first run builds the UI)..."
ready=0
for i in $(seq 1 240); do
  if curl -fs -o /dev/null --max-time 2 "$URL/api/health" 2>/dev/null; then ready=1; break; fi
  kill -0 "$SSH_PID" 2>/dev/null || { echo "--- SSH ended early; $LOG: ---"; cat "$LOG"; exit 1; }
  sleep 0.5
done
[ "$ready" = 1 ] || { echo "Server didn't respond in ~120s; $LOG:"; cat "$LOG"; }

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ -x "$CHROME" ]; then
  "$CHROME" --app="$URL" --user-data-dir="$HOME/.tasni-appwin" \
            --no-first-run --no-default-browser-check >/dev/null 2>&1
  for i in $(seq 1 40); do pgrep -f tasni-appwin >/dev/null && break; sleep 0.5; done
  while pgrep -f tasni-appwin >/dev/null; do sleep 1; done   # block until the window closes
else
  open "$URL"
  echo "Tasni is open in your browser. CLOSE THIS TERMINAL WINDOW to stop the server."
  wait "$SSH_PID"
fi
# window/terminal closed -> the EXIT trap stops the Windows server
