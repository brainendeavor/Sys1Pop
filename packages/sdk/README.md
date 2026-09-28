# @sys1pop/sdk

Official TypeScript/JavaScript SDK for **Sys1Pop** — the autonomous, zero-latency "System 1" decision engine and RAG triage microservice running in Cloudflare Workers.

---

## Installation

```bash
npm install @sys1pop/sdk
```

---

## Universal Decision API (`.decide()`)

Sys1Pop is a general-purpose edge decision runtime. All domain classification, intent routing, risk scoring, and spam detection are evaluated through the universal `.decide()` method.

### In a Cloudflare Worker (via Service Binding)

Configure the Service Binding in your caller's `wrangler.toml`:

```toml
[[services]]
binding = "SYS1POP"
service = "sys1pop"
```

Then in your TypeScript worker:

```typescript
import { Sys1Pop } from "@sys1pop/sdk";

export default {
  async fetch(req: Request, env: { SYS1POP: Fetcher }) {
    const sys1 = new Sys1Pop(env.SYS1POP); // In-memory isolate IPC (<0.2ms)

    // Evaluate typed decision questions
    const res = await sys1.decide({
      state: "User feedback: fast shipping but package arrived damaged.",
      questions: [
        { type: "boolean", id: "damaged" },
        { type: "choice", id: "category", options: ["shipping", "product_quality", "billing"] },
        { type: "score", id: "urgency", min: 1, max: 5 }
      ]
    });

    // Ergonomic typed getters
    const isDamaged = res.getBoolean("damaged");          // true
    const category  = res.getChoice("category");           // "shipping"
    const urgency   = res.getScore("urgency");             // 4.2
    const isCached  = res.cached;                          // false (or true on repeat)

    return Response.json({ isDamaged, category, urgency, isCached });
  }
};
```

---

## Multi-Chunk RAG Triage

Prune irrelevant passages before hitting downstream generative LLMs:

```typescript
const triage = await sys1.triageRAG(
  "enterprise refund SLA",
  ["Passage 1: Standard refunds...", "Passage 2: Enterprise SLA requirements..."],
  { relevance_threshold: 0.6 }
);

console.log(triage.retained_chunks);  // Indices of relevant chunks
console.log(triage.sufficiency_score); // 0.0 - 1.0 confidence
```

---

## External HTTP / Node.js Usage

```typescript
import { Sys1Pop } from "@sys1pop/sdk";

const sys1 = new Sys1Pop("https://sys1pop.your-domain.workers.dev");

const res = await sys1.decide({
  model: "spam-classifier",
  state: "Form submission: check out our crypto presale",
  questions: [{ type: "boolean", id: "is_spam" }]
});

console.log("Is spam:", res.getBoolean("is_spam"));
```
