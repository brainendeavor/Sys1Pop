#!/usr/bin/env bash
# ==============================================================================
# Sys1Pop Example Models Deployment Script
# Deploys INT8 model bundles and catalog to Cloudflare R2 bucket(s)
# ==============================================================================
set -euo pipefail

BUCKET="sys1-models"
ENDPOINT=""
DIST_DIR="./dist/models"
SKIP_EXPORT=false

print_help() {
  cat << EOF
Sys1Pop Example Models R2 Deployment Tool

USAGE:
  ./scripts/deploy-models.sh [options]

OPTIONS:
  --bucket <name>       Target Cloudflare R2 bucket name (default: sys1-models)
  --endpoint <url>      Sys1Pop worker URL to verify models against after upload
  --dist-dir <path>     Directory containing or receiving model bundles (default: ./dist/models)
  --skip-export         Skip python bundle generation step and upload existing dist
  --help, -h            Show this help message

EXAMPLES:
  ./scripts/deploy-models.sh
  ./scripts/deploy-models.sh --bucket custom-models-bucket
  ./scripts/deploy-models.sh --endpoint https://sys1pop.example.workers.dev
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket)
      BUCKET="$2"
      shift 2
      ;;
    --endpoint)
      ENDPOINT="$2"
      shift 2
      ;;
    --dist-dir)
      DIST_DIR="$2"
      shift 2
      ;;
    --skip-export)
      SKIP_EXPORT=true
      shift
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
echo "📦 ========================================================"
echo "   Sys1Pop Example Models R2 Deployment"
echo "   R2 Bucket:    $BUCKET"
echo "   Model Dir:    $DIST_DIR"
if [[ -n "$ENDPOINT" ]]; then
  echo "   Verify Host:  $ENDPOINT"
fi
echo "=========================================================="
echo ""

# 1. Ensure R2 bucket exists
echo "🔍 Checking Cloudflare R2 bucket '$BUCKET'..."
if ! npx wrangler r2 bucket list 2>/dev/null | grep -q "\"name\": \"$BUCKET\""; then
  echo "📦 Bucket '$BUCKET' not found or not visible. Attempting creation..."
  npx wrangler r2 bucket create "$BUCKET" 2>/dev/null || echo "   (Bucket may already exist or will be verified on put)"
else
  echo "  ✓ Bucket '$BUCKET' confirmed."
fi

# 2. Generate model bundles & catalog if needed
if [[ "$SKIP_EXPORT" == "false" ]]; then
  echo ""
  echo "⚙️ Generating spec-compliant neural model bundles and catalog..."
  if command -v uv >/dev/null 2>&1; then
    uv run --with "torch,transformers,scikit-learn,safetensors" tools/export_models.py --output-dir "$DIST_DIR"
  else
    python3 tools/export_models.py --output-dir "$DIST_DIR"
  fi
fi

# 3. Upload model bundles and catalog to R2
echo ""
echo "🚀 Uploading model artifacts to R2 bucket '$BUCKET'..."

# Upload catalog.json
CATALOG_FILE="$DIST_DIR/catalog.json"
if [[ -f "$CATALOG_FILE" ]]; then
  echo "  ↑ Uploading models/catalog.json -> r2://$BUCKET/models/catalog.json"
  npx wrangler r2 object put "$BUCKET/models/catalog.json" --file="$CATALOG_FILE" --remote
fi

# Upload each model directory
for MODEL_PATH in "$DIST_DIR"/*; do
  if [[ -d "$MODEL_PATH" ]]; then
    MODEL_ID=$(basename "$MODEL_PATH")
    echo "  📦 Uploading model '$MODEL_ID'..."

    for FILE_NAME in "manifest.json" "model.safetensors" "tokenizer.json" "config.json"; do
      SRC_FILE="$MODEL_PATH/$FILE_NAME"
      if [[ -f "$SRC_FILE" ]]; then
        R2_KEY="models/$MODEL_ID/$FILE_NAME"
        echo "    ↑ $FILE_NAME -> r2://$BUCKET/$R2_KEY"
        npx wrangler r2 object put "$BUCKET/$R2_KEY" --file="$SRC_FILE" --remote
      fi
    done
  fi
done

echo ""
echo "✅ All example models published to R2 bucket '$BUCKET'!"

# 4. Verification if endpoint provided
if [[ -n "$ENDPOINT" ]]; then
  echo ""
  echo "🧪 Verifying deployed models against Worker endpoint: $ENDPOINT..."
  curl -sSf "$ENDPOINT/v1/models" | jq . || curl -sSf "$ENDPOINT/v1/models"
  echo ""
  echo "✅ Worker successfully connected and serving model catalog!"
fi

echo ""
echo "Ready for edge inference! You can select and test these models in the Kick the Tires UI."
echo ""
