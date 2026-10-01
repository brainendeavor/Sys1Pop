# Sys1Pop CLI (`npx sys1pop`)

The developer command-line interface for **Sys1Pop** — the autonomous, zero-latency "System 1" decision engine and RAG triage microservice running in Cloudflare Workers.

---

## Installation & Usage

You can run the CLI directly via `npx`:

```bash
npx sys1pop <command> [options]
```

Or install it globally:

```bash
npm install -g sys1pop
```

---

## Available Commands

### 1. `sys1pop model init [path]`
Generates a pre-populated declarative `model.spec.json` template with schema validation:

```bash
npx sys1pop model init ./support-triage-v1.spec.json --name support-triage-v1 --type multi-head
```

### 2. `sys1pop model validate <spec>`
Validates specification syntax, unique question IDs, options, and dataset balances in milliseconds with zero ML dependencies:

```bash
npx sys1pop model validate ./support-triage-v1.spec.json
```

### 3. `sys1pop model build <spec>`
Compiles the specification into a production-ready edge bundle (`manifest.json`, `model.safetensors`, `tokenizer.json`, `config.json`) under the 35MB Cloudflare Worker limit:

```bash
npx sys1pop model build ./support-triage-v1.spec.json --output ./dist/models/support-triage-v1
```

### 4. `sys1pop model push <dir>`
Validates any model directory, uploads assets directly to Cloudflare R2, and atomically updates the remote `catalog.json`:

```bash
npx sys1pop model push ./dist/models/support-triage-v1 --bucket sys1pop-models
```

### 5. `sys1pop model test <model-id>`
Runs a live synthetic latency and memory audit against the edge worker:
* Measures cold-start forward pass latency (~25–35 ms).
* Measures in-isolate LRU cache hit latency (< 1 ms at $0.00 CPU cost).
* Asserts decision head output calibration.

```bash
npx sys1pop model test support-triage-v1 --endpoint http://localhost:6061
```

### 6. `sys1pop model list`
Lists all models currently resident in warm isolate RAM across active edge worker instances.

```bash
npx sys1pop model list [--endpoint <url>]
```

### 7. `sys1pop model unload <model-id>`
Evicts a warm model bundle from isolate RAM to free memory.

```bash
npx sys1pop model unload support-triage-v1 [--endpoint <url>]
```

### 8. `sys1pop deploy`
Builds and deploys the turnkey Sys1Pop worker microservice directly into your Cloudflare account with SIMD128 vector acceleration enabled.

```bash
npx sys1pop deploy
```

### 9. `sys1pop seed-catalog`
Syncs the pre-quantized foundation models into your Cloudflare R2 bucket (`sys1pop-models`):
* `sys1-base`: 33.4M parameter general semantic representation & decision backbone.
* `rag-reranker`: 33.4M parameter passage cross-encoder for RAG triage.
* `intent-router`: 33.4M parameter multi-action intent routing backbone.

```bash
npx sys1pop seed-catalog [--bucket <name>]
```

---

## Detailed ModelForge Documentation

For the complete guide on declarative edge model design, probe head math, dataset ingestion, and active edge learning loops, see:
👉 **[`docs/MODEL_FORGE.md`](../../docs/MODEL_FORGE.md)**
