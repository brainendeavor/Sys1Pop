import {
  DecisionRequest,
  DecisionResponse,
  DecisionResult,
  ExecutionMetrics,
  HealthResponse,
  RAGTriageResult,
  TriageConfig,
} from "./types.js";

/**
 * Minimal Cloudflare Worker Fetcher interface for Service Bindings
 */
export interface ServiceBinding {
  fetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response>;
}

export interface Sys1PopOptions {
  endpoint?: string;
  binding?: ServiceBinding;
  headers?: Record<string, string>;
  token?: string;
}

export type Sys1PopTarget = ServiceBinding | string | Sys1PopOptions | undefined | null;

export class Sys1PopNotConfiguredError extends Error {
  constructor(message = "Sys1Pop is not configured: missing Service Binding or endpoint") {
    super(message);
    this.name = "Sys1PopNotConfiguredError";
  }
}

function isServiceBinding(target: unknown): target is ServiceBinding {
  return (
    typeof target === "object" &&
    target !== null &&
    "fetch" in target &&
    typeof (target as { fetch: unknown }).fetch === "function"
  );
}

/**
 * Ergonomic wrapper around the edge DecisionResponse providing typed getters
 */
export class DecisionResultHelper {
  readonly raw: DecisionResponse;
  readonly decisions: Record<string, DecisionResult>;
  readonly triage: RAGTriageResult | null;
  readonly metrics: ExecutionMetrics;
  readonly cached: boolean;
  readonly modelId: string;

  constructor(raw: DecisionResponse) {
    this.raw = raw;
    this.decisions = raw.decisions || {};
    this.triage = raw.triage ?? null;
    this.metrics = raw.metrics;
    this.cached = raw.cached ?? false;
    this.modelId = raw.model_id;
  }

  getBoolean(id: string): boolean | undefined {
    const d = this.decisions[id];
    return d && d.type === "boolean" ? d.value : undefined;
  }

  getProbability(id: string): number | undefined {
    const d = this.decisions[id];
    return d && d.type === "boolean" ? d.probability : undefined;
  }

  getChoice(id: string): string | undefined {
    const d = this.decisions[id];
    return d && d.type === "choice" ? d.winner : undefined;
  }

  getConfidence(id: string): number | undefined {
    const d = this.decisions[id];
    return d && d.type === "choice" ? d.confidence : undefined;
  }

  getScore(id: string): number | undefined {
    const d = this.decisions[id];
    return d && d.type === "score" ? d.expected_value : undefined;
  }
}

export class Sys1Pop {
  private binding?: ServiceBinding;
  private baseUrl: string;
  private defaultHeaders: Record<string, string>;
  private configured: boolean;

  constructor(target?: Sys1PopTarget) {
    if (!target) {
      this.binding = undefined;
      this.baseUrl = "http://sys1pop";
      this.defaultHeaders = {};
      this.configured = false;
    } else if (typeof target === "string") {
      this.baseUrl = target.replace(/\/+$/, "");
      this.defaultHeaders = {};
      this.configured = true;
    } else if (isServiceBinding(target)) {
      this.binding = target;
      this.baseUrl = "http://sys1pop";
      this.defaultHeaders = {};
      this.configured = true;
    } else {
      const opts = target as Sys1PopOptions;
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
  isConfigured(): boolean {
    return this.configured;
  }

  private async fetchInternal(path: string, init?: RequestInit): Promise<Response> {
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
  async decide(request: DecisionRequest): Promise<DecisionResultHelper> {
    const res = await this.fetchInternal("/v1/decide", {
      method: "POST",
      body: JSON.stringify(request),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Sys1Pop decide failed [${res.status}]: ${errText}`);
    }

    const raw = (await res.json()) as DecisionResponse;
    return new DecisionResultHelper(raw);
  }

  /**
   * Helper for multi-chunk RAG relevance and sufficiency triage
   */
  async triageRAG(
    query: string,
    chunks: string[],
    config?: TriageConfig
  ): Promise<RAGTriageResult> {
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
  async health(): Promise<HealthResponse> {
    const res = await this.fetchInternal("/health", { method: "GET" });
    if (!res.ok) {
      throw new Error(`Sys1Pop health check failed: status ${res.status}`);
    }
    return (await res.json()) as HealthResponse;
  }

  /**
   * Lists models warm in isolate memory
   */
  async getModels(): Promise<string[]> {
    const res = await this.fetchInternal("/v1/models", { method: "GET" });
    if (!res.ok) {
      throw new Error(`Sys1Pop getModels failed: status ${res.status}`);
    }
    const data = (await res.json()) as { models: string[] };
    return data.models;
  }
}
