import { DecisionRequest, DecisionResponse, DecisionResult, ExecutionMetrics, HealthResponse, RAGTriageResult, TriageConfig } from "./types.js";
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
}
export type Sys1PopTarget = ServiceBinding | string | Sys1PopOptions | undefined | null;
export declare class Sys1PopNotConfiguredError extends Error {
    constructor(message?: string);
}
/**
 * Ergonomic wrapper around the edge DecisionResponse providing typed getters
 */
export declare class DecisionResultHelper {
    readonly raw: DecisionResponse;
    readonly decisions: Record<string, DecisionResult>;
    readonly triage: RAGTriageResult | null;
    readonly metrics: ExecutionMetrics;
    readonly cached: boolean;
    readonly modelId: string;
    constructor(raw: DecisionResponse);
    getBoolean(id: string): boolean | undefined;
    getProbability(id: string): number | undefined;
    getChoice(id: string): string | undefined;
    getConfidence(id: string): number | undefined;
    getScore(id: string): number | undefined;
}
export declare class Sys1Pop {
    private binding?;
    private baseUrl;
    private defaultHeaders;
    private configured;
    constructor(target?: Sys1PopTarget);
    /**
     * Returns true if a valid Service Binding or HTTP endpoint is configured
     */
    isConfigured(): boolean;
    private fetchInternal;
    /**
     * Universal decision method evaluating questions and RAG triage against a given state
     */
    decide(request: DecisionRequest): Promise<DecisionResultHelper>;
    /**
     * Helper for multi-chunk RAG relevance and sufficiency triage
     */
    triageRAG(query: string, chunks: string[], config?: TriageConfig): Promise<RAGTriageResult>;
    /**
     * Health and isolate status check
     */
    health(): Promise<HealthResponse>;
    /**
     * Lists models warm in isolate memory
     */
    getModels(): Promise<string[]>;
}
