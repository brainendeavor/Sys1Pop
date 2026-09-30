#!/usr/bin/env bash
# ==============================================================================
# Sys1Pop Local Development Server
# Runs Miniflare / workerd with Kick the Tires UI and local example models
# ==============================================================================
set -euo pipefail

PORT=8787
DIST_DIR="./dist/models"

print_help() {
  cat << EOF
Sys1Pop Local Development Runner

USAGE:
  ./scripts/dev.sh [options]

OPTIONS:
  --port <number>       Local dev server port (default: 8787)
  --help, -h            Show this help message

EXAMPLES:
  ./scripts/dev.sh
  ./scripts/dev.sh --port 9000
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      print_help
      exit 1
      ;;
  esac
done

echo ""
echo "🚀 ========================================================"
echo "   Sys1Pop Local Development Server"
echo "   UI Mode:     Enabled (ENABLE_UI=true)"
echo "   Local Port:  $PORT"
echo "   Local URL:   http://localhost:$PORT"
echo "=========================================================="
echo ""

# 1. Ensure local model bundles are present
if [[ ! -f "$DIST_DIR/catalog.json" ]]; then
  echo "📦 Generating local example models in $DIST_DIR..."
  python3 tools/export_examples.py --output-dir "$DIST_DIR"
fi

# 2. Check cargo target build
echo "🔨 Verifying local Worker build..."
CARGO_TARGET_DIR=./target cargo check -p sys1pop-worker

# 3. Launch Wrangler dev with UI enabled
echo ""
echo "🌐 Starting Miniflare local runtime..."
echo "👉 Open http://localhost:$PORT in your browser to kick the tires!"
echo ""

export ENABLE_UI="true"
exec npx wrangler dev --port "$PORT" --var ENABLE_UI:true
