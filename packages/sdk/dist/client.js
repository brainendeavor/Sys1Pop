"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Sys1Pop = exports.DecisionResultHelper = exports.Sys1PopNotConfiguredError = void 0;
class Sys1PopNotConfiguredError extends Error {
    constructor(message = "Sys1Pop is not configured: missing Service Binding or endpoint") {
        super(message);
        this.name = "Sys1PopNotConfiguredError";
    }
}
exports.Sys1PopNotConfiguredError = Sys1PopNotConfiguredError;
function isServiceBinding(target) {
    return (typeof target === "object" &&
        target !== null &&
        "fetch" in target &&
        typeof target.fetch === "function");
}
/**
 * Ergonomic wrapper around the edge DecisionResponse providing typed getters
 */
class DecisionResultHelper {
    raw;
    decisions;
    triage;
    metrics;
    cached;
    modelId;
    constructor(raw) {
        this.raw = raw;
        this.decisions = raw.decisions || {};
        this.triage = raw.triage ?? null;
        this.metrics = raw.metrics;
        this.cached = raw.cached ?? false;
        this.modelId = raw.model_id;
    }
    getBoolean(id) {
        const d = this.decisions[id];
        return d && d.type === "boolean" ? d.value : undefined;
    }
    getProbability(id) {
        const d = this.decisions[id];
        return d && d.type === "boolean" ? d.probability : undefined;
    }
    getChoice(id) {
        const d = this.decisions[id];
        return d && d.type === "choice" ? d.winner : undefined;
    }
    getConfidence(id) {
        const d = this.decisions[id];
        return d && d.type === "choice" ? d.confidence : undefined;
    }
    getScore(id) {
        const d = this.decisions[id];
        return d && d.type === "score" ? d.expected_value : undefined;
    }
}
exports.DecisionResultHelper = DecisionResultHelper;
class Sys1Pop {
    binding;
    baseUrl;
    defaultHeaders;
    configured;
    constructor(target) {
        if (!target) {
            this.binding = undefined;
            this.baseUrl = "http://sys1pop";
            this.defaultHeaders = {};
            this.configured = false;
        }
        else if (typeof target === "string") {
            this.baseUrl = target.replace(/\/+$/, "");
            this.defaultHeaders = {};
            this.configured = true;
        }
        else if (isServiceBinding(target)) {
            this.binding = target;
            this.baseUrl = "http://sys1pop";
            this.defaultHeaders = {};
            this.configured = true;
        }
        else {
            const opts = target;
            this.binding = opts.binding;
            this.baseUrl = (opts.endpoint || "http://sys1pop").replace(/\/+$/, "");
            this.defaultHeaders = { ...(opts.headers || {}) };
            if (opts.token) {
                this.defaultHeaders["Authorization"] = `Bearer ${opts.token}`;
            }
            this.configured = !!(opts.binding || opts.endpoint);
        }
    }
    /**
     * Returns true if a valid Service Binding or HTTP endpoint is configured
     */
    isConfigured() {
        return this.configured;
    }
    async fetchInternal(path, init) {
        if (!this.configured && !this.binding) {
            throw new Sys1PopNotConfiguredError();
        }
        const url = `${this.baseUrl}${path}`;
        const headers = {
            "Content-Type": "application/json",
            ...this.defaultHeaders,
            ...(init?.headers || {}),
        };
        if (this.binding) {
            return this.binding.fetch(url, { ...init, headers });
        }
        return fetch(url, { ...init, headers });
    }
    /**
     * Universal decision method evaluating questions and RAG triage against a given state
     */
    async decide(request) {
        const res = await this.fetchInternal("/v1/decide", {
            method: "POST",
            body: JSON.stringify(request),
        });
        if (!res.ok) {
            const errText = await res.text();
            throw new Error(`Sys1Pop decide failed [${res.status}]: ${errText}`);
        }
        const raw = (await res.json());
        return new DecisionResultHelper(raw);
    }
    /**
     * Evaluates a decision request pre-configured from a declarative ModelSpec
     */
    async decideWithSpec(spec, state, overrides) {
        const questions = spec.questions.map((q) => {
            if (q.type === "choice") {
                return { type: "choice", id: q.id, options: q.options };
            }
            else if (q.type === "score") {
                return { type: "score", id: q.id, min: q.min, max: q.max };
            }
            else {
                return { type: "boolean", id: q.id };
            }
        });
        return this.decide({
            model: spec.model_id,
            state,
            questions,
            context_chunks: overrides?.context_chunks || spec.default_chunks || [],
            triage_config: overrides?.triage_config || spec.triage_config || undefined,
            ...overrides,
        });
    }
    /**
     * Helper for multi-chunk RAG relevance and sufficiency triage
     */
    async triageRAG(query, chunks, config) {
        const res = await this.decide({
            state: query,
            context_chunks: chunks,
            triage_config: config,
            questions: [
                { id: "sufficient_context", type: "boolean" },
            ],
        });
        if (!res.triage) {
            throw new Error("Sys1Pop did not return triage evaluation for chunks");
        }
        return res.triage;
    }
    /**
     * Health and isolate status check
     */
    async health() {
        const res = await this.fetchInternal("/health", { method: "GET" });
        if (!res.ok) {
            throw new Error(`Sys1Pop health check failed: status ${res.status}`);
        }
        return (await res.json());
    }
    /**
     * Lists models warm in isolate memory
     */
    async getModels() {
        const res = await this.fetchInternal("/v1/models", { method: "GET" });
        if (!res.ok) {
            throw new Error(`Sys1Pop getModels failed: status ${res.status}`);
        }
        const data = (await res.json());
        return data.models;
    }
}
exports.Sys1Pop = Sys1Pop;
