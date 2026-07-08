#!/usr/bin/env bash
set -euo pipefail

REMOTE=root@47.77.202.240
RECONNECT_DELAY_SECONDS=${RECONNECT_DELAY_SECONDS:-5}

# The cloud side enables BBR for newly accepted SSH connections and routes
# public ports 8082/8084 through these loopback-only reverse forwards.
while true; do
  echo "[$(date '+%F %T %Z')] starting policy reverse tunnel to ${REMOTE}"
  ssh -N -T \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o TCPKeepAlive=yes \
    -o ConnectTimeout=15 \
    -R 127.0.0.1:29682:127.0.0.1:8082 \
    -R 127.0.0.1:29684:127.0.0.1:8084 \
    "${REMOTE}" || rc=$?
  rc=${rc:-0}
  echo "[$(date '+%F %T %Z')] tunnel exited with code ${rc}; reconnecting in ${RECONNECT_DELAY_SECONDS}s"
  sleep "${RECONNECT_DELAY_SECONDS}"
  unset rc
done
