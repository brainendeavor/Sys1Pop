# Sys1Pop Setup & Configuration Guide

This guide walks you through setting up, configuring, running locally, and deploying the **Sys1Pop** autonomous edge decision engine to Cloudflare Workers and Cloudflare R2.

---

## Table of Contents

- [Overview & Architecture](#overview--architecture)
- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Environment Variables & Secrets Reference](#environment-variables--secrets-reference)
- [Cloudflare R2 Bucket Provisioning & Model Deployment](#cloudflare-r2-bucket-provisioning--model-deployment)
- [Worker Production Deployment](#worker-production-deployment)
- [Integration Guide](#integration-guide)
  - [Cloudflare Worker Service Binding (Zero-Hop IPC)](#cloudflare-worker-service-binding-zero-hop-ipc)
  - [External HTTP API Client](#external-http-api-client)
- [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Overview & Architecture

Sys1Pop executes ultra-lean, sub-35ms neural decisions directly inside **Cloudflare Worker V8 Isolates** across global edge PoPs (Points of Presence).

```
┌────────────────────────────────────────────────────────────┐
│                    Cloudflare Edge PoP                     │
│                                                            │
│  HTTP / Service Binding ──► [ sys1pop Worker Isolate ]     │
│                                   │                        │
│                         In-Isolate LRU Cache               │
│                                   │                        │
│                       Candle WASM SIMD128 Engine           │
│                                   ▲                        │
│                                   │ on-demand stream       │
└───────────────────────────────────┼────────────────────────┘
                                    │
                         [ Cloudflare R2 Bucket ]
                         (models/{id}/model.safetensors)
```

- **Runtime:** Rust WebAssembly (`wasm32-unknown-unknown`) compiled with `-C target-feature=+simd128` vector acceleration and Hugging Face Candle.
- **Model Storage:** Cloudflare R2 bucket (`sys1pop-models`) storing INT8 quantized safetensors, configurations, tokenizers, and dynamic catalog metadata.
- **Memory & Cache:** 512MB isolate RAM headroom with warm model residency and microsecond in-isolate LRU decision caching (<0.05ms).

---

## Prerequisites

Before starting, ensure you have the following installed:

1. **Rust Toolchain & WASM Target:**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup target add wasm32-unknown-unknown
   ```

2. **Cloudflare Worker Build Tool (`worker-build`):**
   ```bash
   cargo install worker-build
   ```

3. **Node.js (18+) & Wrangler:**
   ```bash
   node --version # >= 18.0.0
   npm --version
   ```

4. **Python 3.10+ / uv (Optional for model calibration & ModelForge compilation):**
   ```bash
   # uv is recommended for fast dependency resolution
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

---

## Local Development Setup

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/brainendeavor/Sys1Pop.git
cd Sys1Pop
npm install
```

### 2. Configure Local Environment Variables

Sys1Pop includes a template file [`.dev.vars.example`](file:///.dev.vars.example) for local configuration. Copy it to `.dev.vars` (which is git-ignored):

```bash
cp .dev.vars.example .dev.vars
```

Edit `.dev.vars` to customize your local settings:

```ini
# Enables Kick the Tires interactive test UI at http://localhost:6061
ENABLE_UI=true

# Secret token for authenticating protected routes
API_TOKEN=dev-secret-token-123

# Keep false for unauthenticated /v1/decide testing in the playground
SECURE_DECIDE_API=false

# Keep admin lifecycle routes (/v1/models/unload, /v1/cache/clear) active
ENABLE_ADMIN_API=true

# Fallback catalog model IDs if R2 bucket is empty
CONFIGURED_MODELS=sys1-base,spam-detector-v1,rag-triage-v1,intent-router-v1
```

### 3. Launch Local Development Server & Studio

Start the local server using the dev runner script:

```bash
./scripts/dev.sh
# or using npm:
npm run dev
```

The script automatically:
1. Verifies or generates local model bundles (`./dist/models`) using `tools/export_models.py`.
2. Compiles the Rust worker WASM bundle using `worker-build --dev`.
3. Launches Miniflare / workerd on port **`6061`** with `.dev.vars` environment variables loaded.

### 4. Kick the Tires in the Browser

Open **`http://localhost:6061`** in your browser. The embedded **Kick the Tires** testing studio allows you to:
- Inspect active and warm model bundles.
- Select presets or input custom states.
- Evaluate real-time neural decisions, boolean flags, choice classifications, and continuous scores.
- Monitor forward-pass and tokenization latency metrics.

### 5. Test Inference via Curl

In another terminal, test the decision API directly:

```bash
curl -X POST http://localhost:6061/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "model": "spam-detector-v1",
    "state": "Site: example.com | Name: Alex | Email: alex@example.com | Message: Low Google ranking? We offer high DA backlinks!",
    "questions": [
      { "type": "boolean", "id": "is_spam" },
      { "type": "choice", "id": "spam_category", "options": ["legitimate_inquiry", "seo_backlink_spam", "crypto_phishing"] },
      { "type": "score", "id": "risk_score", "min": 1, "max": 5 }
    ]
  }'
```

---

## Environment Variables & Secrets Reference

All environment variables, secrets, and bindings supported by Sys1Pop are summarized below:

| Variable / Binding | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `API_TOKEN` | Secret / Var | **Required** in prod (if auth enabled) | `""` | Secret token used to authenticate calls to protected endpoints (`/v1/models/unload`, `/v1/cache/clear`) and `/v1/decide` (when `SECURE_DECIDE_API=true`). Accepted via `Authorization: Bearer <token>` or `X-API-Token: <token>`. |
| `SECURE_DECIDE_API` | Var | Optional | `false` | When set to `true`, **enforces `API_TOKEN` authentication on `POST /v1/decide`**. If `false`, inference is public (ideal for dev/testing or internal service networks). *(Aliases: `SECURE_ALL_APIS`, `REQUIRE_AUTH`)* |
| `ENABLE_ADMIN_API` | Var | Optional | `true` | When set to `false`, completely disables admin lifecycle endpoints (`/v1/models/unload`, `/v1/cache/clear`) with HTTP 403 Forbidden. |
| `ENABLE_UI` | Var | Optional | `false` (Worker default) / `true` (Dev script) | Controls whether the embedded Kick the Tires interactive HTML/SVG test playground is served at `GET /` and `GET /ui`. |
| `CONFIGURED_MODELS` | Var | Optional | `sys1-base, spam-detector-v1, rag-triage-v1` | Comma-separated list of model IDs to include in the `GET /v1/models` catalog response when no dynamic `models/catalog.json` exists in R2. |
| `MODELS` | R2 Bucket Binding | **Required** in prod | - | Cloudflare R2 bucket binding configured in `wrangler.toml` (`[[r2_buckets]] binding = "MODELS" bucket_name = "sys1pop-models"`). Stores model safetensors, configurations, tokenizers, and catalog. |
| `SYS1POP` | Service Binding | Optional (Caller-side) | - | Cloudflare Service Binding identifier when calling Sys1Pop from another Worker isolate (`[[services]] binding = "SYS1POP" service = "sys1pop"`). |

---

## Cloudflare R2 Bucket Provisioning & Model Deployment

Sys1Pop streams quantized INT8 transformer weights on demand from Cloudflare R2.

### 1. Create the R2 Bucket

```bash
npx wrangler r2 bucket create sys1pop-models
```

### 2. Export & Deploy Model Bundles

Use `./scripts/deploy-models.sh` or the Sys1Pop CLI to calibrate, export, and upload model bundles directly to R2:

```bash
# Upload all standard foundation models
./scripts/deploy-models.sh

# Or upload specific models selectively
./scripts/deploy-models.sh --models "sys1-base,spam-detector-v1"

# Prune unspecified models from remote R2 bucket and catalog
./scripts/deploy-models.sh --models "spam-detector-v1" --prune
```

Alternatively, use the Sys1Pop CLI (`npx sys1pop`):

```bash
# Seed standard catalog to R2
npx sys1pop seed-catalog --bucket sys1pop-models

# Or compile and push a custom ModelForge model
npx sys1pop model build ./examples/spam-detector-v1.spec.json --output ./dist/models/spam-detector-v1
npx sys1pop model push ./dist/models/spam-detector-v1 --bucket sys1pop-models
```

### 3. Verify Models in R2

Inspect the models registered with the remote catalog:

```bash
npx wrangler r2 object get sys1pop-models/models/catalog.json --file=catalog.json
cat catalog.json | jq .
```

---

## Worker Production Deployment

### 1. Configure Production Secrets

Never commit production secret tokens to version control. Set them in Cloudflare KMS via Wrangler:

```bash
# Configure the secret API token for authentication
npx wrangler secret put API_TOKEN
```

### 2. Deploy Worker

Deploy Sys1Pop to Cloudflare Workers with SIMD128 acceleration and production variables:

```bash
# Deploy with secured decide API and headless mode (UI disabled in production)
./scripts/deploy-worker.sh --api-token "your-production-token" --secure-decide-api --disable-ui

# Or deploy to a specific Cloudflare environment (staging/production)
./scripts/deploy-worker.sh --env production
```

Alternatively, deploy using `wrangler`:

```bash
cd crates/sys1pop-worker
worker-build --release
cd ../..
mkdir -p build && cp -R crates/sys1pop-worker/build/ build/
npx wrangler deploy --var SECURE_DECIDE_API:true --var ENABLE_UI:false
```

### 3. Verify Live Worker Health

```bash
# Verify health status
curl https://sys1pop.your-domain.workers.dev/health

# Verify available models
curl https://sys1pop.your-domain.workers.dev/v1/models
```

---

## Integration Guide

### Cloudflare Worker Service Binding (Zero-Hop IPC)

When calling Sys1Pop from another Worker (e.g. FreeFormer, an API gateway, or a backend microservice), use Cloudflare **Service Bindings**. Service Bindings run in-memory within the same edge PoP without HTTP/TLS overhead (<0.2ms latency).

1. In your caller worker's `wrangler.toml`:

```toml
[[services]]
binding = "SYS1POP"
service = "sys1pop"
```

2. In your caller worker TypeScript code:

```typescript
import { Sys1Pop } from "@sys1pop/sdk";

interface Env {
  SYS1POP: Fetcher;
  SYS1POP_API_TOKEN?: string; // Optional if SECURE_DECIDE_API is enabled
}

export default {
  async fetch(req: Request, env: Env) {
    const sys1 = new Sys1Pop({
      binding: env.SYS1POP,
      token: env.SYS1POP_API_TOKEN,
    });

    const decision = await sys1.decide({
      model: "spam-detector-v1",
      state: "Inquiry text from user contact form",
      questions: [
        { type: "boolean", id: "is_spam" },
        { type: "score", id: "risk_score", min: 1, max: 5 }
      ]
    });

    const isSpam = decision.getBoolean("is_spam");
    const risk = decision.getScore("risk_score");

    return Response.json({ isSpam, risk });
  }
};
```

### External HTTP API Client

For Node.js, Python, or client applications calling Sys1Pop over HTTPS:

```typescript
import { Sys1Pop } from "@sys1pop/sdk";

const sys1 = new Sys1Pop({
  endpoint: "https://sys1pop.your-domain.workers.dev",
  token: process.env.API_TOKEN, // Pass Bearer token
});

const result = await sys1.decide({
  model: "sys1-base",
  state: "Database connection timeouts on us-east cluster",
  questions: [
    { type: "choice", id: "primary_intent", options: ["lookup", "escalation", "feedback"] },
    { type: "boolean", id: "requires_action" }
  ]
});

console.log("Intent:", result.getChoice("primary_intent"));
console.log("Action needed:", result.getBoolean("requires_action"));
```

---

## Troubleshooting & FAQ

### 1. `Error: Model 'xyz' not found in R2 bucket 'MODELS'`
- **Cause:** The requested model bundle has not been uploaded to the R2 bucket.
- **Solution:** Ensure the model exists in R2 at `models/{model_id}/model.safetensors`, `config.json`, and `tokenizer.json`. Run `./scripts/deploy-models.sh --models xyz` or `npx sys1pop seed-catalog`.

### 2. `HTTP 401 Unauthorized`
- **Cause:** Either the endpoint requires `API_TOKEN` (admin routes or `SECURE_DECIDE_API=true`), but the request is missing or has an incorrect token.
- **Solution:** Pass the token via `Authorization: Bearer <API_TOKEN>` or header `X-API-Token: <API_TOKEN>`. Ensure `API_TOKEN` is configured in `.dev.vars` locally or via `npx wrangler secret put API_TOKEN` remotely.

### 3. `WASM SIMD128 Build Errors`
- **Cause:** Missing `wasm32-unknown-unknown` target or missing `-C target-feature=+simd128` rustflags.
- **Solution:** Run `rustup target add wasm32-unknown-unknown`. Verify `.cargo/config.toml` contains:
  ```toml
  [target.wasm32-unknown-unknown]
  rustflags = ["-C", "target-feature=+simd128"]
  ```

### 4. Memory Headroom & Model Eviction
- Sys1Pop keeps models warm in isolate RAM across requests. If you deploy many models and need to evict a specific model from warm isolate memory, call the unload endpoint:
  ```bash
  curl -X POST https://sys1pop.your-domain.workers.dev/v1/models/unload \
    -H "Authorization: Bearer $API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"model":"spam-detector-v1"}'
  ```
  Or use the CLI: `npx sys1pop model unload spam-detector-v1`.
