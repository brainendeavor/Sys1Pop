#!/usr/bin/env bash
# ==============================================================================
# Sys1Pop Local Development Server
# Runs Miniflare / workerd with Kick the Tires UI and local example models
# ==============================================================================
set -euo pipefail

PORT=6061
DIST_DIR="./dist/models"
SECURE_DECIDE_API="false"
API_TOKEN=""

print_help() {
  cat << EOF
Sys1Pop Local Development Runner

USAGE:
  ./scripts/dev.sh [options]

OPTIONS:
  --port <number>       Local dev server port (default: 6061)
  --secure-decide-api   Enforce API_TOKEN authentication on /v1/decide
  --api-token <token>   Secret API_TOKEN value to configure in dev environment
  --help, -h            Show this help message

EXAMPLES:
  ./scripts/dev.sh
  ./scripts/dev.sh --port 5751
  ./scripts/dev.sh --api-token "dev-secret" --secure-decide-api
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --secure-decide-api|--secure-all-apis)
      SECURE_DECIDE_API="true"
      shift
      ;;
    --api-token)
      API_TOKEN="$2"
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
echo "   UI Mode:             Enabled (ENABLE_UI=true)"
echo "   Secure /decide API:  $SECURE_DECIDE_API"
echo "   Local Port:          $PORT"
echo "   Local URL:           http://localhost:$PORT"
echo "=========================================================="
echo ""

# 1. Ensure local model bundles are present
if [[ ! -f "$DIST_DIR/catalog.json" ]]; then
  echo "📦 Generating local model bundles in $DIST_DIR..."
  if command -v uv >/dev/null 2>&1; then
    uv run --with "torch,transformers,scikit-learn,safetensors" tools/export_models.py --output-dir "$DIST_DIR"
  else
    python3 tools/export_models.py --output-dir "$DIST_DIR"
  fi
fi

# 2. Check and compile Worker WASM bundle
if [[ ! -f "build/worker/shim.mjs" ]]; then
  echo "🔨 Compiling Worker WASM bundle with worker-build..."
  (cd crates/sys1pop-worker && worker-build --dev)
  mkdir -p build && cp -R crates/sys1pop-worker/build/ build/
else
  echo "🔨 Worker WASM bundle ready at build/worker/shim.mjs."
fi

# 3. Launch Wrangler dev with UI enabled
echo ""
echo "🌐 Starting Miniflare local runtime..."
echo "👉 Open http://localhost:$PORT in your browser to kick the tires!"
echo ""

export ENABLE_UI="true"
DEV_CMD=("npx" "wrangler" "dev" "--port" "$PORT" "--var" "ENABLE_UI:true")

if [[ "$SECURE_DECIDE_API" == "true" ]]; then
  DEV_CMD+=("--var" "SECURE_DECIDE_API:true")
fi

if [[ -n "$API_TOKEN" ]]; then
  DEV_CMD+=("--var" "API_TOKEN:${API_TOKEN}")
fi

exec "${DEV_CMD[@]}"
