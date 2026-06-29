#!/usr/bin/env bash
set -euo pipefail

# The cloud side enables BBR for newly accepted SSH connections and routes
# public ports 8082/8084 through these loopback-only reverse forwards.
exec ssh -N -T \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -R 127.0.0.1:29682:127.0.0.1:8082 \
  -R 127.0.0.1:29684:127.0.0.1:8084 \
  root@47.77.202.240
