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
/**
 * ModelForge Declarative Specification Types (RFC-002)
 */
export interface ModelSpecCalibration {
    temperature?: number;
    regularization_c?: number;
    max_iterations?: number;
    default_threshold?: number;
}
export type ModelSpecChoiceQuestion = ChoiceQuestion;
export type ModelSpecBooleanQuestion = BooleanQuestion & {
    threshold?: number;
};
export type ModelSpecScoreQuestion = ScoreQuestion;
export type ModelSpecQuestion = ModelSpecChoiceQuestion | ModelSpecBooleanQuestion | ModelSpecScoreQuestion;
export interface ModelSpecExample {
    state: string;
    [questionId: string]: string | number | boolean;
}
export interface ModelSpec {
    $schema?: string;
    model_id: string;
    name: string;
    version?: string;
    description?: string;
    icon?: string;
    base_backbone?: string;
    quantization?: "fp32_edge" | "int8_q8_0";
    calibration?: ModelSpecCalibration;
    default_state?: string;
    default_chunks?: string[];
    triage_config?: TriageConfig;
    sample_presets?: Array<{
        label: string;
        state: string;
    }>;
    questions: ModelSpecQuestion[];
    training_examples?: ModelSpecExample[];
    dataset_path?: string;
}
