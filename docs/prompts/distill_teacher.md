# Frontier LLM Teacher Prompt Template: Sys1Pop Model Distiller

Copy and paste the prompt below into **Claude 3.5 Sonnet**, **Gemini 1.5 Pro**, or **GPT-4o** to generate a complete Sys1Pop model bundle for any custom domain.

---

```markdown
You are an expert Machine Learning Engineer specializing in edge distillation and model compression for Cloudflare Workers using Sys1Pop (Hugging Face Candle Rust WASM).

I need to create a custom Sys1Pop edge model for the following domain:
[DESCRIBE YOUR DOMAIN HERE, e.g.: "Healthcare customer support triage to detect HIPAA privacy risks, route across billing/clinical/insurance, and score urgency from 1 to 5."]

Please follow the Sys1Pop Model Specification (docs/SPEC.md) and provide:

1. **Decision Taxonomy & Default Questions:**
   - Define the Boolean, Choice, and Score questions formatted for `manifest.json`.

2. **Synthetic Training Dataset:**
   - Generate 30 to 50 realistic, diverse examples with hard negatives and edge cases.
   - Output formatted as JSON with fields: `text`, `boolean_labels`, `choice_labels`, `score_labels`.

3. **1-Click Python Distillation Script:**
   - A self-contained script using `sentence-transformers` and `safetensors`.
   - Fits the linear projection heads on `all-MiniLM-L6-v2` in <60 seconds on CPU.
   - Quantizes weights to INT8 and outputs the Sys1Pop bundle:
     - `manifest.json`
     - `model.safetensors`
     - `tokenizer.json`
     - `config.json`

4. **Edge Verification & Service Binding Example:**
   - A short TypeScript snippet demonstrating how to call the model via `@sys1pop/sdk`.
```
