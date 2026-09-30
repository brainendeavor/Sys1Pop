#!/usr/bin/env bash
# ==============================================================================
# Sys1Pop Worker Deployment Script
# Deploys Sys1Pop Cloudflare Worker with embedded Kick-the-Tires UI
# ==============================================================================
set -euo pipefail

ENABLE_UI="true"
API_TOKEN=""
CF_ENV=""
EXTRA_ARGS=()

print_help() {
  cat << EOF
Sys1Pop Worker Deployment Tool

USAGE:
  ./scripts/deploy-worker.sh [options]

OPTIONS:
  --enable-ui             Enable embedded Kick the Tires interactive UI (default)
  --disable-ui            Deploy as headless API microservice without UI
  --api-token <token>     Configure API_TOKEN for protected lifecycle endpoints
  --env <name>            Target Cloudflare environment (e.g., staging, production)
  --help, -h              Show this help message

EXAMPLES:
  ./scripts/deploy-worker.sh
  ./scripts/deploy-worker.sh --disable-ui
  ./scripts/deploy-worker.sh --api-token "my-secret-token" --env production
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --enable-ui)
      ENABLE_UI="true"
      shift
      ;;
    --disable-ui)
      ENABLE_UI="false"
      shift
      ;;
    --api-token)
      API_TOKEN="$2"
      shift 2
      ;;
    --env)
      CF_ENV="$2"
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

echo ""
echo "🚀 ========================================================"
echo "   Sys1Pop Worker Deployment"
echo "   UI Enabled:    $ENABLE_UI"
echo "   SIMD128 Flags: +simd128 (wasm32-unknown-unknown)"
if [[ -n "$CF_ENV" ]]; then
  echo "   Environment:   $CF_ENV"
fi
echo "=========================================================="
echo ""

# 1. Prerequisite checks
if ! command -v cargo &> /dev/null; then
  echo "❌ Error: Rust 'cargo' is required but not found."
  exit 1
fi

if ! rustup target list --installed | grep -q "wasm32-unknown-unknown"; then
  echo "📦 Installing wasm32-unknown-unknown target..."
  rustup target add wasm32-unknown-unknown
fi

# Ensure SIMD128 target flag is set
if [[ ! -f ".cargo/config.toml" ]] || ! grep -q "target-feature=+simd128" ".cargo/config.toml"; then
  echo "⚠️ Warning: .cargo/config.toml missing SIMD128 target flag. Creating..."
  mkdir -p .cargo
  cat << 'EOF' > .cargo/config.toml
[target.wasm32-unknown-unknown]
rustflags = ["-C", "target-feature=+simd128"]
EOF
fi

# 2. Build worker WASM
echo "🔨 Compiling Sys1Pop Worker with SIMD128 vector acceleration..."
CARGO_TARGET_DIR=./target cargo check -p sys1pop-worker

# 3. Construct Wrangler Deploy command
DEPLOY_CMD=("npx" "wrangler" "deploy" "--var" "ENABLE_UI:${ENABLE_UI}")

if [[ -n "$API_TOKEN" ]]; then
  DEPLOY_CMD+=("--var" "API_TOKEN:${API_TOKEN}")
fi

if [[ -n "$CF_ENV" ]]; then
  DEPLOY_CMD+=("--env" "$CF_ENV")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  DEPLOY_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "📡 Deploying to Cloudflare Workers via Wrangler..."
"${DEPLOY_CMD[@]}"

echo ""
echo "✅ Sys1Pop Worker deployed successfully!"
if [[ "$ENABLE_UI" == "true" ]]; then
  echo "🌐 Kick the Tires UI is ACTIVE. Open your Worker URL in browser to test!"
else
  echo "🔒 Kick the Tires UI is DISABLED. Operating in headless API mode."
fi
echo ""
