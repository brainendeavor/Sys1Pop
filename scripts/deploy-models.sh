#!/usr/bin/env bash
# ==============================================================================
# Sys1Pop Example Models Deployment Script
# Deploys INT8 model bundles and catalog to Cloudflare R2 bucket(s)
# ==============================================================================
set -euo pipefail

BUCKET="sys1pop-models"
ENDPOINT=""
DIST_DIR="./dist/models"
SKIP_EXPORT=false
MODELS=""
PRUNE=false

print_help() {
  cat << EOF
Sys1Pop Example Models R2 Deployment Tool

USAGE:
  ./scripts/deploy-models.sh [options]

OPTIONS:
  --models, -m <list>   Specify which models to upload (e.g. spam-detector-v1,sys1-base)
  --prune, -p           Remove models not specified from R2 bucket and catalog
  --bucket <name>       Target Cloudflare R2 bucket name (default: sys1pop-models)
  --endpoint <url>      Sys1Pop worker URL to verify models against after upload
  --dist-dir <path>     Directory containing or receiving model bundles (default: ./dist/models)
  --skip-export         Skip python bundle generation step and upload existing dist
  --help, -h            Show this help message

EXAMPLES:
  ./scripts/deploy-models.sh
  ./scripts/deploy-models.sh --models spam-detector-v1
  ./scripts/deploy-models.sh --models spam-detector-v1 --prune
  ./scripts/deploy-models.sh --models "spam-detector-v1, sys1-base"
  ./scripts/deploy-models.sh --bucket custom-models-bucket
  ./scripts/deploy-models.sh --endpoint https://sys1pop.example.workers.dev
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models|--model|-m)
      if [[ -z "$MODELS" ]]; then
        MODELS="$2"
      else
        MODELS="$MODELS,$2"
      fi
      shift 2
      ;;
    --prune|-p)
      PRUNE=true
      shift
      ;;
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
echo "   R2 Bucket:     $BUCKET"
echo "   Model Dir:     $DIST_DIR"
if [[ -n "$MODELS" ]]; then
  echo "   Target Models: $MODELS"
else
  echo "   Target Models: all"
fi
echo "   Prune:         $PRUNE"
if [[ -n "$ENDPOINT" ]]; then
  echo "   Verify Host:   $ENDPOINT"
fi
echo "=========================================================="
echo ""

# 1. Ensure R2 bucket exists
echo "🔍 Checking Cloudflare R2 bucket '$BUCKET'..."
if ! npx wrangler r2 bucket list 2>/dev/null | grep -E "(name:[[:space:]]+$BUCKET|\"$BUCKET\")" >/dev/null; then
  echo "📦 Bucket '$BUCKET' not found or not visible. Attempting creation..."
  npx wrangler r2 bucket create "$BUCKET" 2>/dev/null || echo "   (Bucket may already exist or will be verified on put)"
else
  echo "  ✓ Bucket '$BUCKET' confirmed."
fi

# 2. Generate model bundles & catalog if needed
if [[ "$SKIP_EXPORT" == "false" ]]; then
  echo ""
  echo "⚙️ Generating spec-compliant neural model bundles and catalog..."
  EXPORT_CMD=()
  if command -v uv >/dev/null 2>&1; then
    EXPORT_CMD=("uv" "run" "--with" "torch,transformers,scikit-learn,safetensors" "tools/export_models.py" "--output-dir" "$DIST_DIR")
  else
    EXPORT_CMD=("python3" "tools/export_models.py" "--output-dir" "$DIST_DIR")
  fi

  if [[ -n "$MODELS" ]]; then
    EXPORT_CMD+=("--models" "$MODELS")
  fi

  if [[ "$PRUNE" == "true" ]]; then
    EXPORT_CMD+=("--prune")
  fi

  "${EXPORT_CMD[@]}"
fi

# 3. Determine models to upload
UPLOAD_MODELS=()
if [[ -n "$MODELS" ]]; then
  # Parse comma or space separated list
  CLEAN_MODELS="${MODELS//,/ }"
  read -r -a PARSED_MODELS <<< "$CLEAN_MODELS"
  for M in "${PARSED_MODELS[@]}"; do
    M=$(echo "$M" | tr -d '[:space:]')
    if [[ -n "$M" ]]; then
      if [[ -d "$DIST_DIR/$M" ]]; then
        UPLOAD_MODELS+=("$M")
      else
        echo "⚠️ Warning: Model directory '$DIST_DIR/$M' not found. Skipping."
      fi
    fi
  done
else
  for MODEL_PATH in "$DIST_DIR"/*; do
    if [[ -d "$MODEL_PATH" ]]; then
      UPLOAD_MODELS+=("$(basename "$MODEL_PATH")")
    fi
  done
fi

# Prune unspecified models from Cloudflare R2 bucket and local catalog if requested
if [[ "$PRUNE" == "true" ]]; then
  echo ""
  echo "🧹 Checking for unspecified models to prune..."

  # Discover currently deployed models in R2 from remote catalog
  TEMP_REMOTE_CAT=$(mktemp)
  npx wrangler r2 object get "$BUCKET/models/catalog.json" --file="$TEMP_REMOTE_CAT" --remote >/dev/null 2>&1 || true

  # Determine models to prune using python (avoids bash set -u empty array bugs)
  PRUNE_OUTPUT=$(python3 -c '
import json, sys

remote_cat_file = sys.argv[1]
upload_models = set(sys.argv[2].split()) if len(sys.argv) > 2 and sys.argv[2] else set()
known_models = ["sys1-base", "spam-detector-v1", "rag-triage-v1", "intent-router-v1"]

candidates = set(known_models)
try:
    with open(remote_cat_file, "r", encoding="utf-8") as f:
        d = json.load(f)
    for m in d.get("models", []):
        mid = m.get("model_id")
        if mid:
            candidates.add(mid)
except Exception:
    pass

to_prune = sorted(list(candidates - upload_models))
print(" ".join(to_prune))
' "$TEMP_REMOTE_CAT" "${UPLOAD_MODELS[*]}" 2>/dev/null || true)
  rm -f "$TEMP_REMOTE_CAT"

  if [[ -z "${PRUNE_OUTPUT// /}" ]]; then
    echo "  ✓ No unspecified models to prune."
  else
    read -r -a MODELS_TO_PRUNE <<< "$PRUNE_OUTPUT"
    echo "  🗑️ Pruning ${#MODELS_TO_PRUNE[@]} unspecified model(s) from R2: ${MODELS_TO_PRUNE[*]}"
    for P_MODEL in "${MODELS_TO_PRUNE[@]}"; do
      echo "  🗑️ Removing r2://$BUCKET/models/$P_MODEL/..."
      (
        for FILE_NAME in "manifest.json" "model.safetensors" "tokenizer.json" "config.json" "tokenizer_config.json" "special_tokens_map.json"; do
          npx wrangler r2 object delete "$BUCKET/models/$P_MODEL/$FILE_NAME" --remote >/dev/null 2>&1 &
        done
        wait
      )
      echo "    ✓ Deleted r2://$BUCKET/models/$P_MODEL/"

      # Also clean up local directory if present
      if [[ -d "$DIST_DIR/$P_MODEL" ]]; then
        rm -rf "$DIST_DIR/$P_MODEL"
      fi

      # Evict from warm worker isolate if endpoint provided
      if [[ -n "$ENDPOINT" ]]; then
        curl -s -X POST "$ENDPOINT/v1/models/unload" \
          -H "Content-Type: application/json" \
          -d "{\"model\":\"$P_MODEL\"}" >/dev/null 2>&1 || true
      fi
    done
  fi

  # Ensure local catalog only contains specified models
  CATALOG_FILE="$DIST_DIR/catalog.json"
  if [[ -f "$CATALOG_FILE" ]]; then
    python3 -c '
import json, sys
cat_path = sys.argv[1]
allowed = set(sys.argv[2:])
try:
    with open(cat_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    d["models"] = [m for m in d.get("models", []) if m.get("model_id") in allowed]
    with open(cat_path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
except Exception:
    pass
' "$CATALOG_FILE" "${UPLOAD_MODELS[@]}"
  fi
fi

# 4. Upload model bundles and catalog to R2
echo ""
echo "🚀 Uploading model artifacts to R2 bucket '$BUCKET'..."

# Upload catalog.json
CATALOG_FILE="$DIST_DIR/catalog.json"
if [[ -f "$CATALOG_FILE" ]]; then
  echo "  ↑ Uploading models/catalog.json -> r2://$BUCKET/models/catalog.json"
  npx wrangler r2 object put "$BUCKET/models/catalog.json" --file="$CATALOG_FILE" --remote
fi

# Upload each selected model directory
for MODEL_ID in "${UPLOAD_MODELS[@]}"; do
  MODEL_PATH="$DIST_DIR/$MODEL_ID"
  echo "  📦 Uploading model '$MODEL_ID'..."

  for FILE_NAME in "manifest.json" "model.safetensors" "tokenizer.json" "config.json"; do
    SRC_FILE="$MODEL_PATH/$FILE_NAME"
    if [[ -f "$SRC_FILE" ]]; then
      R2_KEY="models/$MODEL_ID/$FILE_NAME"
      echo "    ↑ $FILE_NAME -> r2://$BUCKET/$R2_KEY"
      npx wrangler r2 object put "$BUCKET/$R2_KEY" --file="$SRC_FILE" --remote
    fi
  done
done

echo ""
echo "✅ Selected models (${UPLOAD_MODELS[*]}) published to R2 bucket '$BUCKET'!"

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
