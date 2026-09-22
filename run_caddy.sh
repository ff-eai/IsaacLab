#!/bin/bash
# Reverse proxy that serves the WebXR client AND forwards CloudXR signaling
# endpoints over the same HTTPS origin (fixes the mixed-content / CORS gap that
# made the Pico's CONNECT button return 0xC0F22213).
# Replaces serve_xr_client.sh when CloudXR is running.
# Stop with Ctrl-C.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Caddy reverse proxy"
echo "  WebXR client: https://10.66.0.71:8443/"
echo "  CloudXR proxy: /attachment /head /pose -> http://127.0.0.1:49100"
echo "  Ctrl-C to stop."

exec ./bin/caddy run --config Caddyfile --adapter caddyfile
