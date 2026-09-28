use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Inbound System 1 Decision Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRequest {
    /// Contextual state or prompt to evaluate
    pub state: String,
    
    /// Optional model identifier (e.g., "sys1-base", "freeformer-spam-v2", "legal-triage-v1")
    #[serde(default)]
    pub model: Option<String>,

    /// Optional context passages / retrieved RAG chunks to evaluate
    #[serde(default)]
    pub context_chunks: Vec<String>,
    
    /// Optional dynamic RAG triage parameters
    #[serde(default)]
    pub triage_config: Option<TriageConfig>,

    /// Targeted decision questions
    #[serde(default)]
    pub questions: Vec<Question>,
}

/// Dynamic RAG triage parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriageConfig {
    #[serde(default = "default_relevance_threshold")]
    pub relevance_threshold: f32,
    #[serde(default = "default_sufficiency_threshold")]
    pub sufficiency_threshold: f32,
    #[serde(default)]
    pub strategy: Option<String>,
    #[serde(default)]
    pub max_retained_chunks: Option<usize>,
}

fn default_relevance_threshold() -> f32 {
    0.3
}

fn default_sufficiency_threshold() -> f32 {
    0.5
}

/// A typed decision question
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Question {
    Choice {
        id: String,
        options: Vec<String>,
    },
    Boolean {
        id: String,
    },
    Score {
        id: String,
        min: Option<i32>,
        max: Option<i32>,
    },
}

/// Outbound System 1 Decision Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionResponse {
    pub decisions: HashMap<String, DecisionResult>,
    pub triage: Option<RAGTriageResult>,
    pub metrics: ExecutionMetrics,
    #[serde(default)]
    pub cached: bool,
    #[serde(default = "default_model_id")]
    pub model_id: String,
}

fn default_model_id() -> String {
    "sys1-base".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionResult {
    Choice {
        winner: String,
        confidence: f32,
        distribution: HashMap<String, f32>,
    },
    Boolean {
        value: bool,
        probability: f32,
    },
    Score {
        expected_value: f32,
        distribution: Vec<f32>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RAGTriageResult {
    pub chunks_evaluated: usize,
    pub retained_chunks: Vec<usize>,
    pub relevance_scores: Vec<f32>,
    pub sufficiency_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    pub tokenize_ms: f64,
    pub forward_pass_ms: f64,
    pub total_ms: f64,
}
