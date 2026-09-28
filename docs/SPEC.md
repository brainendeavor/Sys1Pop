# Sys1Pop Model Specification & Frontier LLM Distillation Guide (v1.0)

> **Audience:** Frontier LLMs (Claude 3.5 Sonnet, Gemini 1.5 Pro, GPT-4o) and Machine Learning Engineers designing custom edge models for the **Sys1Pop** edge decision engine.

---

## 1. Edge Runtime Environment & Hard Constraints

Sys1Pop executes directly inside **Cloudflare Worker V8 Isolates** across global edge PoPs (Points of Presence) via pure Rust WebAssembly (`wasm32-unknown-unknown` + SIMD128) and Hugging Face **Candle**.

When generating weights, architectures, or distillation scripts, you **MUST** strictly adhere to these physical hardware and isolate limits:

| Parameter | Constraint | Rationale |
| :--- | :--- | :--- |
| **Max Model Weight Size** | **≤ 35 MB** | Cloudflare Workers enforce script/asset limits. Model files stream from R2 into isolate RAM. |
| **Quantization Format** | **INT8 Safetensors (`Q8_0` or INT8 weights)** | Reduces 130MB FP32 MiniLM to ~33MB; accelerates Candle SIMD128 dot products by 3.5x. |
| **Max Isolate RAM Allocation** | **100 MB** for weights & tensors | Worker paid limit is 512MB; remaining >400MB is reserved for isolate LRU cache and RAG chunks. |
| **Target CPU Execution Budget** | **≤ 35 ms** | Cloudflare Unbound bills $0.02 per 1M CPU-seconds. 35ms keeps edge cost at ~$1.00 per 1M decisions. |
| **Target Sequence Length** | **128 – 512 tokens** | Max sequence length for real-time edge decision triage. |
| **Tokenizer Engine** | **`tokenizers-rs` (HuggingFace `tokenizer.json`)** | Must be exportable as standard fast tokenizer JSON without Python-specific regex dependencies. |

### Supported Base Transformer Backbones
* **Standard Representation:** `sentence-transformers/all-MiniLM-L6-v2` (384 hidden dim, 6 layers, 12 heads, 33.4M parameters).
* **High-Accuracy Representation:** `BAAI/bge-small-en-v1.5` (384 hidden dim, 33.4M parameters).
* **Cross-Encoder Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (for passage cross-attention scoring).

---

## 2. Decision Heads & Mathematical Formulation

Sys1Pop maps representation vectors into discrete, calibrated outputs via three fundamental decision heads:

### 1. `ChoiceHead` (Multi-Class Categorization)
Computes a normalized probability distribution over discrete candidate options via temperature-scaled softmax:
$$P(y_i) = \frac{\exp(z_i / T)}{\sum_{j=1}^K \exp(z_j / T)}$$
* Logits $z \in \mathbb{R}^{1 \times K}$ projected from the pooled transformer representation $h \in \mathbb{R}^{384}$.
* $T$: Calibration temperature (default $T = 1.0$).
* Outputs the winning option string, probability confidence score, and complete distribution map.

### 2. `BooleanHead` (Binary Decision)
Computes a calibrated binary probability using sigmoid:
$$P(\text{true}) = \sigma(z) = \frac{1}{1 + \exp(-z)}$$
* Threshold: $P(\text{true}) \ge 0.5 \implies \text{true}$.

### 3. `ScoreHead` (Expected Value / Ordinal Rating)
Computes the expected value across bounded integer bins (e.g. risk score 1 to 5):
$$\mathbb{E}[S] = \sum_{i=\min}^{\max} i \cdot P(S = i)$$
where probabilities $P(S = i)$ are derived via softmax over ordinal projection logits.

---

## 3. The Model Bundle Specification

Every model published to Cloudflare R2 (`r2://sys1-models/models/{model_id}/`) must contain four files:

```
models/{model_id}/
├── manifest.json        # Contract metadata, head configs, and default questions
├── model.safetensors    # INT8 quantized weights
├── tokenizer.json       # HuggingFace fast tokenizer definition
└── config.json          # Transformer backbone configuration
```

### `manifest.json` Schema Specification:
```json
{
  "schema_version": "1.0",
  "model_id": "customer-support-triage-v1",
  "architecture": "minilm_l6_v2",
  "hidden_dim": 384,
  "max_seq_len": 512,
  "quantization": "int8_q8_0",
  "supported_heads": ["choice", "boolean", "score"],
  "default_questions": [
    {
      "type": "boolean",
      "id": "requires_escalation"
    },
    {
      "type": "choice",
      "id": "department",
      "options": ["billing", "technical_support", "account_security", "general_inquiry"]
    },
    {
      "type": "score",
      "id": "urgency",
      "min": 1,
      "max": 5
    }
  ],
  "calibration": {
    "temperature": 1.0,
    "default_threshold": 0.5
  }
}
```

---

## 4. The 1-Minute Distillation Recipe for Frontier LLMs

When prompted to build an Sys1Pop model for a specific domain:

1. **Step 1: Domain Modeling:** Define the `Choice`, `Boolean`, and `Score` questions in `manifest.json`.
2. **Step 2: Synthetic Data Generation:** Generate 30 to 50 diverse synthetic examples per label with realistic edge cases and hard negatives.
3. **Step 3: Few-Shot Parameter-Efficient Distillation:**
   * Freeze the base backbone (`all-MiniLM-L6-v2`).
   * Fit linear classification heads using logistic regression or SetFit contrastive learning.
   * Training takes **under 45 seconds on a laptop CPU** without requiring a GPU.
4. **Step 4: INT8 Safetensors Export:** Quantize the resulting weights and export the four bundle files.
