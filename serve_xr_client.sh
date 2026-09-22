#!/bin/bash
# Serve the WebXR client over HTTPS so the Pico 4 Ultra can load it.
# Cert: mkcert-generated, trusted if you install webxr_certs/rootCA... on the Pico.
# Stop with Ctrl-C.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEB_ROOT="${SCRIPT_DIR}/webxr_client"
CERT="${SCRIPT_DIR}/webxr_certs/10.66.0.71+2.pem"
KEY="${SCRIPT_DIR}/webxr_certs/10.66.0.71+2-key.pem"
PORT="${PORT:-8443}"
BIND="${BIND:-0.0.0.0}"

if [ ! -f "$CERT" ] || [ ! -f "$KEY" ]; then
    echo "Cert/key missing at $CERT / $KEY" >&2
    echo "Regenerate with: cd webxr_certs && ~/.local/bin/mkcert 10.66.0.71 localhost 127.0.0.1" >&2
    exit 1
fi

echo "Serving ${WEB_ROOT} at https://${BIND}:${PORT}/  (Ctrl-C to stop)"
echo "From Pico:   https://10.66.0.71:${PORT}/"

exec python3 - "$WEB_ROOT" "$BIND" "$PORT" "$CERT" "$KEY" <<'PY'
import http.server, ssl, sys, os

root, bind, port, cert, key = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
os.chdir(root)

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

httpd = http.server.ThreadingHTTPServer((bind, port), Handler)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(certfile=cert, keyfile=key)
httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
httpd.serve_forever()
PY
