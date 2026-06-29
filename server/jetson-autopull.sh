#!/bin/bash
# jetson-autopull.sh — keep the Jetson's monorepo clone in lock-step with its
# checked-out origin branch and restart the camera server when its code changes.
#
# Run by jetson-autopull.timer every couple of minutes. Design goals:
#   * Cheap no-op when there's nothing new (a quiet fetch + a SHA compare).
#   * Safe over the flaky Wi-Fi: any failure just exits 0 and retries next tick.
#   * Never interrupt a live capture: if a client is connected on :1024 we defer
#     the WHOLE update to a later tick, so the running service and the on-disk code
#     never drift (we only ever update when it's safe to restart).
#   * Only bounce the camera service when the pulled change actually touched
#     server/ — an unrelated commit (web UI, docs) shouldn't blip the camera.
#
# Runs as root (to `systemctl restart`); git runs as the repo owner `jetson` via a
# login shell so its GitHub credentials (~/.git-credentials) and config are used.
set -u

REPO=/home/jetson/robodk
UNIT=realsense-camera
GIT() { runuser -l jetson -c "cd '$REPO' && $1"; }

GIT "git fetch --quiet origin" || exit 0
BRANCH=$(GIT "git rev-parse --abbrev-ref HEAD" 2>/dev/null) || BRANCH=main
[ -z "$BRANCH" ] || [ "$BRANCH" = "HEAD" ] && BRANCH=main
GIT "git rev-parse --verify --quiet origin/$BRANCH" >/dev/null || BRANCH=main
GIT "git checkout --quiet '$BRANCH'" 2>/dev/null || \
    GIT "git checkout --quiet -B '$BRANCH' --track 'origin/$BRANCH'" || exit 0

LOCAL=$(GIT "git rev-parse HEAD" 2>/dev/null) || exit 0
REMOTE=$(GIT "git rev-parse 'origin/$BRANCH'" 2>/dev/null) || exit 0
[ -z "$REMOTE" ] && exit 0
[ "$LOCAL" = "$REMOTE" ] && exit 0      # already up to date — nothing to do

# A new commit is waiting. If a client is mid-capture, leave everything untouched
# and try again next tick (keeps running code == on-disk code).
if ss -tn 2>/dev/null | grep -q ':1024'; then
    echo "auto-pull: client connected on :1024 — deferring $LOCAL -> $REMOTE"
    exit 0
fi

CHANGED=$(GIT "git diff --name-only '$LOCAL' '$REMOTE'" 2>/dev/null)
echo "auto-pull: updating $LOCAL -> $REMOTE"
GIT "git reset --hard --quiet '$REMOTE'" || { echo "auto-pull: reset failed"; exit 0; }

if printf '%s\n' "$CHANGED" | grep -q '^server/'; then
    echo "auto-pull: server/ changed -> restarting $UNIT"
    systemctl restart "$UNIT" || echo "auto-pull: restart failed"
else
    echo "auto-pull: no server/ changes; leaving $UNIT running"
fi
