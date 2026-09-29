/**
 * Decision Question Types
 */
export type ChoiceQuestion = {
    type: "choice";
    id: string;
    options: string[];
};
export type BooleanQuestion = {
    type: "boolean";
    id: string;
};
export type ScoreQuestion = {
    type: "score";
    id: string;
    min?: number;
    max?: number;
};
export type Question = ChoiceQuestion | BooleanQuestion | ScoreQuestion;
/**
 * Dynamic RAG Triage Configuration
 */
export interface TriageConfig {
    relevance_threshold?: number;
    sufficiency_threshold?: number;
    strategy?: string;
    max_retained_chunks?: number;
}
/**
 * Inbound Decision Request
 */
export interface DecisionRequest {
    state: string;
    model?: string;
    context_chunks?: string[];
    triage_config?: TriageConfig;
    questions?: Question[];
}
/**
 * Decision Result Types
 */
export interface ChoiceDecision {
    type: "choice";
    winner: string;
    confidence: number;
    distribution: Record<string, number>;
}
export interface BooleanDecision {
    type: "boolean";
    value: boolean;
    probability: number;
}
export interface ScoreDecision {
    type: "score";
    expected_value: number;
    distribution: number[];
}
export type DecisionResult = ChoiceDecision | BooleanDecision | ScoreDecision;
/**
 * RAG Triage Results
 */
export interface RAGTriageResult {
    chunks_evaluated: number;
    retained_chunks: number[];
    relevance_scores: number[];
    sufficiency_score: number;
}
/**
 * Execution Performance Metrics
 */
export interface ExecutionMetrics {
    tokenize_ms: number;
    forward_pass_ms: number;
    total_ms: number;
}
/**
 * Raw Outbound Decision Response from Edge Worker
 */
export interface DecisionResponse {
    decisions: Record<string, DecisionResult>;
    triage?: RAGTriageResult | null;
    metrics: ExecutionMetrics;
    cached: boolean;
    model_id: string;
}
/**
 * Health Status Response
 */
export interface HealthResponse {
    status: string;
    service: string;
    engine: string;
    cache_entries: number;
    warm_models: string[];
}
