use crate::contract::{DecisionRequest, DecisionResponse, DecisionResult, ExecutionMetrics, Question};
use crate::error::Result;
use crate::model::{BooleanHead, ChoiceHead, QuantizedBackbone, ScoreHead};
use crate::pipeline::RAGTriagePipeline;
use candle_core::{Device, Tensor};
use std::collections::HashMap;

pub struct Sys1Engine {
    pub model_id: String,
    pub backbone: QuantizedBackbone,
    pub choice_head: ChoiceHead,
    pub triage: RAGTriagePipeline,
}

impl Default for Sys1Engine {
    fn default() -> Self {
        Self::new("sys1-base")
    }
}

impl Sys1Engine {
    pub fn new(model_id: impl Into<String>) -> Self {
        let device = Device::Cpu;
        Self {
            model_id: model_id.into(),
            backbone: QuantizedBackbone::new(384, device),
            choice_head: ChoiceHead::new(1.0),
            triage: RAGTriagePipeline::new(0.3, 0.5),
        }
    }

    pub fn execute(&self, req: DecisionRequest) -> Result<DecisionResponse> {
        let mut decisions = HashMap::new();

        // Evaluate RAG chunks if provided
        let triage_result = if !req.context_chunks.is_empty() {
            Some(self.triage.evaluate_with_config(&req.state, &req.context_chunks, req.triage_config.as_ref()))
        } else {
            None
        };

        // Evaluate typed questions
        for question in req.questions {
            match question {
                Question::Choice { id, options } => {
                    let num_opts = options.len();
                    let mock_logits = Tensor::zeros((1, num_opts), candle_core::DType::F32, &self.backbone.device)?;
                    
                    let (winner, conf, dist) = self.choice_head.forward(&mock_logits, &options)?;

                    decisions.insert(id, DecisionResult::Choice {
                        winner,
                        confidence: conf,
                        distribution: dist,
                    });
                }
                Question::Boolean { id } => {
                    let (val, prob) = BooleanHead::forward(0.75);
                    decisions.insert(id, DecisionResult::Boolean {
                        value: val,
                        probability: prob,
                    });
                }
                Question::Score { id, min, max } => {
                    let min_val = min.unwrap_or(1);
                    let max_val = max.unwrap_or(5);
                    let count = (max_val - min_val + 1) as usize;
                    
                    let mock_logits = Tensor::zeros((1, count), candle_core::DType::F32, &self.backbone.device)?;

                    let (expected, dist) = ScoreHead::forward(&mock_logits, min_val, max_val)?;

                    decisions.insert(id, DecisionResult::Score {
                        expected_value: expected,
                        distribution: dist,
                    });
                }
            }
        }

        Ok(DecisionResponse {
            decisions,
            triage: triage_result,
            metrics: ExecutionMetrics {
                tokenize_ms: 1.5,
                forward_pass_ms: 18.2,
                total_ms: 22.0,
            },
            cached: false,
            model_id: self.model_id.clone(),
        })
    }
}
