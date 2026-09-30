<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand_kit/sys1pop_banner_dark.jpg">
    <source media="(prefers-color-scheme: light)" srcset="docs/brand_kit/sys1pop_banner_light.jpg">
    <img alt="Sys1Pop: Autonomous Edge Decision Engine" src="docs/brand_kit/sys1pop_banner_dark.jpg" width="100%">
  </picture>
</p>

# Sys1Pop: Autonomous Edge Decision Engine in Cloudflare Workers

An ultra-lean, autonomous "System 1" decision engine and RAG triage pipeline deployed inside Cloudflare Workers across global **PoPs** (Points of Presence) using pure Rust, Hugging Face **Candle**, and WebAssembly (`wasm32-unknown-unknown` + SIMD128).

---

## Key Highlights

- **Pure In-Worker Inference:** No external API roundtrips; executes directly within the Cloudflare V8 isolate at every edge PoP.
- **The "Cost Hack":** Consumes ~20–40ms of Worker CPU time (**~$1.00 per 1M decisions**), compared to $12.00 on Workers AI, $35–$70 on GPT-4o-mini, or recurring fees on Akismet/CleanTalk.
- **400MB+ Memory Headroom:** Sized around a 33.4M–70M parameter backbone (MiniLM cross-encoders) so the remaining isolate RAM can power multi-chunk RAG triage and text pre-processing pipelines.
- **WASM SIMD128 Accelerated:** Vector operations compiled with `-C target-feature=+simd128` for 3x–4x throughput on V8.
- **Service Bindings Support:** Call Sys1Pop from other Cloudflare Workers with zero network hop and microsecond IPC latency (<0.2ms).

---

## Detailed Documentation

- 📊 **[Economics & Benchmarks](docs/economics_and_benchmarks.md):** Side-by-side cost breakdown comparing Worker CPU time, Workers AI Neurons, and cloud LLMs across 100K, 1M, and 10M requests.
- 📖 **[Frontier LLM Distillation Spec](docs/SPEC.md):** Specification for Claude, Gemini, and GPT to distill custom edge models.
- 🎨 **[Brand Guidelines & Design Kit](docs/brand_kit/BRAND_GUIDELINES.md):** Official banners, shield emblems, Nixie logotype, and color tokens.

---

## Project Structure

```
Sys1Pop
├── .cargo/
│   └── config.toml                  # SIMD128 target flags
├── Cargo.toml                       # Virtual workspace root
├── wrangler.toml                    # Cloudflare Worker config & R2 bindings
├── docs/
│   ├── SPEC.md                      # Promptable LLM Teacher specification
│   ├── economics_and_benchmarks.md  # Detailed financial & latency metrics
│   ├── brand_kit/                   # Official logo emblems, banners, and design guidelines
│   └── prompts/
│       └── distill_teacher.md       # Synthetic dataset generation prompt templates
├── examples/
│   ├── spam_detection_request.json              # Sample form spam payload
│   ├── rag_triage.json                          # RAG sufficiency & relevance test
│   ├── agent_intent_routing.json                # Assistant / smart home tool dispatch
│   ├── cloudflare_worker_binding_caller.ts      # Cloudflare Service Binding integration
│   └── Sys1Pop_Studio.ipynb                      # LLM distillation Colab notebook
├── tools/
│   └── distill_and_export.py        # Python INT8 model packaging & verification tool
├── packages/
│   ├── sdk/                         # @sys1pop/sdk (TypeScript client for Service Bindings)
│   └── cli/                         # npx sys1pop (Worker deploy, model push/test)
└── crates/
    ├── sys1pop-core/                 # Pure Rust algorithms, heads & contracts
    │   ├── build.rs                 # SIMD128 compile-time assertion
    │   └── src/
    │       ├── contract.rs          # Typed System 1 JSON request & response schemas
    │       ├── engine.rs            # Core decision engine execution
    │       ├── model/               # QuantizedBackbone, ChoiceHead, BooleanHead, ScoreHead
    │       └── pipeline/            # RAGTriagePipeline
    └── sys1pop-worker/               # Turnkey Cloudflare Worker microservice
        ├── wrangler.toml            # Worker config & R2 bindings
        └── src/
            ├── cache.rs             # In-isolate 200MB LRU decision cache
            ├── loader.rs            # Dynamic R2 model streaming & residency
            └── lib.rs               # Worker HTTP & Service Binding entrypoint (/v1/decide)
```

---

## Example Use Cases

### 1. Web Form Spam & Phishing Classification (`examples/spam_detection_request.json`)
Evaluates user messages, sender emails, and site context to classify:
- **`is_spam`**: Calibrated boolean probability.
- **`spam_category`**: Multi-class categorization (`legitimate_inquiry`, `commercial_sales_pitch`, `seo_backlink_spam`, `crypto_phishing`, `automated_bot_gibberish`).
- **`risk_score`**: Expected value score from 1 (clean) to 5 (malicious).

### 2. Multi-Chunk RAG Triage (`examples/rag_triage.json`)
Batch-scores retrieved passages against a user prompt:
- Prunes 80% of irrelevant chunks before hitting downstream generative models.
- Returns a boolean `sufficient_context` confidence score to gate generation.

### 3. Agent Intent Routing (`examples/agent_intent_routing.json`)
Routes omni-box prompts across IoT device control, calendar scheduling, or conversational agents in under 40ms.

---

## Local Development & Testing

### Prerequisites
- Rust with `wasm32-unknown-unknown` target:
  ```bash
  rustup target add wasm32-unknown-unknown
  ```
- Wrangler CLI (`npm install -g wrangler` or `npx wrangler`)
- Worker-build (`cargo install worker-build`)

### Run Tests Natively
```bash
cargo test --workspace
```

### Run Locally with Miniflare / workerd
```bash
npx wrangler dev
```

### Test Decision Endpoint
```bash
curl -X POST http://localhost:8787/v1/decide \
  -H "Content-Type: application/json" \
  --data @examples/spam_detection_request.json
```

---

## License

Sys1Pop is dual-licensed under:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

You may choose to use either license at your option.

