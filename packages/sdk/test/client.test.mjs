import test from "node:test";
import assert from "node:assert/strict";
import { Sys1Pop } from "../dist/index.js";

test("Sys1Pop SDK handles service binding and calls /v1/decide with typed getters", async () => {
  let capturedUrl = "";
  let capturedBody = null;

  const mockServiceBinding = {
    async fetch(url, init) {
      capturedUrl = url.toString();
      capturedBody = JSON.parse(init.body);
      return new Response(
        JSON.stringify({
          decisions: {
            is_spam: {
              type: "boolean",
              value: true,
              probability: 0.94,
            },
            category: {
              type: "choice",
              winner: "phishing",
              confidence: 0.88,
              distribution: { phishing: 0.88, sales: 0.12 },
            },
            urgency: {
              type: "score",
              expected_value: 4.5,
              distribution: [0.0, 0.0, 0.1, 0.3, 0.6],
            },
          },
          metrics: {
            tokenize_ms: 1.2,
            forward_pass_ms: 15.0,
            total_ms: 18.0,
          },
          cached: false,
          model_id: "sys1-base",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    },
  };

  const sys1 = new Sys1Pop(mockServiceBinding);
  const response = await sys1.decide({
    state: "Suspicious message requesting crypto transfer",
    questions: [
      { type: "boolean", id: "is_spam" },
      { type: "choice", id: "category", options: ["phishing", "sales"] },
      { type: "score", id: "urgency", min: 1, max: 5 },
    ],
  });

  assert.equal(capturedUrl, "http://sys1pop/v1/decide");
  assert.equal(capturedBody.state, "Suspicious message requesting crypto transfer");
  assert.equal(response.cached, false);
  assert.equal(response.modelId, "sys1-base");
  
  // Test typed getters
  assert.equal(response.getBoolean("is_spam"), true);
  assert.equal(response.getProbability("is_spam"), 0.94);
  assert.equal(response.getChoice("category"), "phishing");
  assert.equal(response.getConfidence("category"), 0.88);
  assert.equal(response.getScore("urgency"), 4.5);
  assert.equal(response.getBoolean("non_existent"), undefined);
});

test("Sys1Pop SDK triageRAG helper calls /v1/decide with chunks", async () => {
  const mockServiceBinding = {
    async fetch(url, init) {
      return new Response(
        JSON.stringify({
          decisions: {},
          triage: {
            chunks_evaluated: 2,
            retained_chunks: [0],
            relevance_scores: [0.85, 0.1],
            sufficiency_score: 0.5,
          },
          metrics: { tokenize_ms: 1.0, forward_pass_ms: 10.0, total_ms: 12.0 },
          cached: false,
          model_id: "sys1-base",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    },
  };

  const sys1 = new Sys1Pop(mockServiceBinding);
  const triage = await sys1.triageRAG("query", ["chunk1", "chunk2"]);
  assert.equal(triage.chunks_evaluated, 2);
  assert.deepEqual(triage.retained_chunks, [0]);
  assert.equal(triage.sufficiency_score, 0.5);
});

test("Sys1Pop SDK handles unconfigured graceful degradation", async () => {
  const sys1 = new Sys1Pop(undefined);
  assert.equal(sys1.isConfigured(), false);

  await assert.rejects(
    async () => {
      await sys1.decide({ state: "test" });
    },
    {
      name: "Sys1PopNotConfiguredError",
    }
  );
});
