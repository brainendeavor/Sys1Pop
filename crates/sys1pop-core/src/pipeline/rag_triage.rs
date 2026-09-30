use crate::contract::{RAGTriageResult, TriageConfig};

/// Evaluates retrieved context chunks for relevance and sufficiency.
pub struct RAGTriagePipeline {
    pub relevance_threshold: f32,
    pub sufficiency_threshold: f32,
}

impl Default for RAGTriagePipeline {
    fn default() -> Self {
        Self::new(0.3, 0.5)
    }
}

impl RAGTriagePipeline {
    pub fn new(relevance_threshold: f32, sufficiency_threshold: f32) -> Self {
        Self {
            relevance_threshold,
            sufficiency_threshold,
        }
    }

    /// Evaluates pre-computed neural relevance scores against triage thresholds.
    pub fn evaluate_scores(&self, scores: &[f32], config: Option<&TriageConfig>) -> RAGTriageResult {
        let (rel_thresh, _suff_thresh, max_retained) = if let Some(cfg) = config {
            (cfg.relevance_threshold, cfg.sufficiency_threshold, cfg.max_retained_chunks)
        } else {
            (self.relevance_threshold, self.sufficiency_threshold, None)
        };

        let mut retained_chunks = Vec::new();
        for (idx, &score) in scores.iter().enumerate() {
            if score >= rel_thresh {
                if let Some(max_k) = max_retained {
                    if retained_chunks.len() < max_k {
                        retained_chunks.push(idx);
                    }
                } else {
                    retained_chunks.push(idx);
                }
            }
        }

        let sufficiency_score = if scores.is_empty() {
            0.0
        } else {
            (retained_chunks.len() as f32 / scores.len() as f32).min(1.0)
        };

        RAGTriageResult {
            chunks_evaluated: scores.len(),
            retained_chunks,
            relevance_scores: scores.to_vec(),
            sufficiency_score,
        }
    }

    /// Evaluates chunks against the query state with an optional dynamic triage configuration.
    pub fn evaluate_with_config(&self, query: &str, chunks: &[String], config: Option<&TriageConfig>) -> RAGTriageResult {
        let (rel_thresh, _suff_thresh, max_retained) = if let Some(cfg) = config {
            (cfg.relevance_threshold, cfg.sufficiency_threshold, cfg.max_retained_chunks)
        } else {
            (self.relevance_threshold, self.sufficiency_threshold, None)
        };

        let mut relevance_scores = Vec::with_capacity(chunks.len());
        let mut retained_chunks = Vec::new();
        
        let query_words: Vec<&str> = query.split_whitespace().collect();

        for (idx, chunk) in chunks.iter().enumerate() {
            let mut matches = 0;
            for word in &query_words {
                if chunk.to_lowercase().contains(&word.to_lowercase()) {
                    matches += 1;
                }
            }
            
            let score = if query_words.is_empty() {
                0.0
            } else {
                (matches as f32 / query_words.len() as f32).min(1.0)
            };

            relevance_scores.push(score);

            if score >= rel_thresh {
                if let Some(max_k) = max_retained {
                    if retained_chunks.len() < max_k {
                        retained_chunks.push(idx);
                    }
                } else {
                    retained_chunks.push(idx);
                }
            }
        }

        let sufficiency_score = if chunks.is_empty() {
            0.0
        } else {
            (retained_chunks.len() as f32 / chunks.len() as f32).min(1.0)
        };

        RAGTriageResult {
            chunks_evaluated: chunks.len(),
            retained_chunks,
            relevance_scores,
            sufficiency_score,
        }
    }

    /// Evaluates chunks against the query state using default pipeline thresholds.
    pub fn evaluate(&self, query: &str, chunks: &[String]) -> RAGTriageResult {
        self.evaluate_with_config(query, chunks, None)
    }
}
