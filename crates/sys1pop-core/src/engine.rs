use crate::contract::{DecisionRequest, DecisionResponse, DecisionResult, ExecutionMetrics, Question};
use crate::error::{Error, Result};
use crate::model::{BooleanHead, ChoiceHead, QuantizedBackbone, ScoreHead};
use crate::pipeline::RAGTriagePipeline;
use candle_core::{Device, Tensor};
use std::collections::HashMap;

/// System 1 Neural Decision Engine
/// Executes cross-encoder transformer forward passes and linear projection decision heads.
pub struct Sys1Engine {
    pub model_id: String,
    pub backbone: QuantizedBackbone,
    pub tokenizer: Option<tokenizers::Tokenizer>,
    pub head_tensors: HashMap<String, Tensor>,
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
            tokenizer: None,
            head_tensors: HashMap::new(),
            choice_head: ChoiceHead::new(1.0),
            triage: RAGTriagePipeline::new(0.3, 0.5),
        }
    }

    /// Instantiates a fully neural Sys1Engine from raw binary artifact bytes.
    /// Compatible with both native execution and Cloudflare Worker WASM isolates streaming from R2.
    pub fn from_bundle(
        model_id: impl Into<String>,
        safetensors_bytes: Vec<u8>,
        config_bytes: &[u8],
        tokenizer_bytes: &[u8],
    ) -> Result<Self> {
        let device = Device::Cpu;
        let config: candle_transformers::models::bert::Config = serde_json::from_slice(config_bytes)
            .map_err(|e| Error::Serialization(e))?;
        
        let mut head_tensors = candle_core::safetensors::load_buffer(&safetensors_bytes, &device)?;
        head_tensors.retain(|k, _| k.starts_with("heads."));

        let backbone = QuantizedBackbone::from_safetensors(safetensors_bytes, &config, device.clone())?;
        let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| Error::Engine(format!("Tokenizer load error: {e}")))?;

        Ok(Self {
            model_id: model_id.into(),
            backbone,
            tokenizer: Some(tokenizer),
            head_tensors,
            choice_head: ChoiceHead::new(1.0),
            triage: RAGTriagePipeline::new(0.3, 0.5),
        })
    }

    /// Convenience loader reading from a local filesystem directory (Native/CLI only)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_from_dir(model_id: impl Into<String>, dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let safetensors_bytes = std::fs::read(dir.join("model.safetensors"))
            .map_err(|e| Error::Engine(format!("Failed to read model.safetensors: {e}")))?;
        let config_bytes = std::fs::read(dir.join("config.json"))
            .map_err(|e| Error::Engine(format!("Failed to read config.json: {e}")))?;
        let tokenizer_bytes = std::fs::read(dir.join("tokenizer.json"))
            .map_err(|e| Error::Engine(format!("Failed to read tokenizer.json: {e}")))?;
        Self::from_bundle(model_id, safetensors_bytes, &config_bytes, &tokenizer_bytes)
    }

    /// Executes neural decision contract against input state using transformer attention and linear heads
    pub fn execute(&self, req: DecisionRequest) -> Result<DecisionResponse> {
        #[cfg(not(target_arch = "wasm32"))]
        let t0 = std::time::Instant::now();
        #[allow(unused_mut)]
        let mut tokenize_ms = 0.0;
        #[allow(unused_mut)]
        let mut forward_pass_ms = 0.0;
        let mut decisions = HashMap::new();

        // 1. Encode input state through Tokenizer and Transformer Backbone
        let h_pooled = if let Some(ref tokenizer) = self.tokenizer {
            #[cfg(not(target_arch = "wasm32"))]
            let t_tok = std::time::Instant::now();
            let encoding = tokenizer.encode(req.state.as_str(), true)
                .map_err(|e| Error::Engine(e.to_string()))?;
            let ids = encoding.get_ids();
            #[cfg(not(target_arch = "wasm32"))]
            { tokenize_ms = t_tok.elapsed().as_secs_f64() * 1000.0; }

            #[cfg(not(target_arch = "wasm32"))]
            let t_fwd = std::time::Instant::now();
            let input_ids = Tensor::new(ids, &self.backbone.device)?.unsqueeze(0)?;
            let pooled = self.backbone.forward(&input_ids)?;
            #[cfg(not(target_arch = "wasm32"))]
            { forward_pass_ms = t_fwd.elapsed().as_secs_f64() * 1000.0; }
            pooled
        } else {
            Tensor::zeros((1, self.backbone.hidden_dim), candle_core::DType::F32, &self.backbone.device)?
        };

        // Normalize pooled embedding for projection and similarity
        let h_norm = {
            let sum_sq = h_pooled.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
            let norm = (sum_sq + 1e-9)?.sqrt()?;
            h_pooled.broadcast_div(&norm)?
        };

        // 2. Evaluate RAG context chunks if provided using neural embedding cosine similarity
        let triage_result = if !req.context_chunks.is_empty() {
            let mut chunk_scores = Vec::with_capacity(req.context_chunks.len());
            for chunk in &req.context_chunks {
                if let Some(ref tokenizer) = self.tokenizer {
                    let enc = tokenizer.encode(chunk.as_str(), true)
                        .map_err(|e| Error::Engine(e.to_string()))?;
                    let input_ids = Tensor::new(enc.get_ids(), &self.backbone.device)?.unsqueeze(0)?;
                    let chunk_pooled = self.backbone.forward(&input_ids)?;
                    let sum_sq = chunk_pooled.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
                    let chunk_norm = chunk_pooled.broadcast_div(&(sum_sq + 1e-9)?.sqrt()?)?;
                    let sim = h_norm.matmul(&chunk_norm.t()?)?.flatten_all()?.to_vec1::<f32>()?[0];
                    chunk_scores.push(sim.max(0.0).min(1.0));
                } else {
                    chunk_scores.push(0.0);
                }
            }
            Some(self.triage.evaluate_scores(&chunk_scores, req.triage_config.as_ref()))
        } else {
            None
        };

        // 3. Evaluate typed questions via neural projection heads
        for question in req.questions {
            match question {
                Question::Choice { id, options } => {
                    let weight_key = format!("heads.choice.{id}.weight");
                    let logits_tensor = if let Some(weight) = self.head_tensors.get(&weight_key) {
                        let mut logits = h_norm.matmul(&weight.t()?)?;
                        let bias_key = format!("heads.choice.{id}.bias");
                        if let Some(bias) = self.head_tensors.get(&bias_key) {
                            logits = logits.broadcast_add(bias)?;
                        }
                        logits
                    } else {
                        // Dynamic zero-shot option semantic similarity if head weights not pre-calibrated
                        let mut opt_logits = Vec::with_capacity(options.len());
                        for opt in &options {
                            if let Some(ref tokenizer) = self.tokenizer {
                                let enc = tokenizer.encode(opt.as_str(), true)
                                    .map_err(|e| Error::Engine(e.to_string()))?;
                                let opt_input = Tensor::new(enc.get_ids(), &self.backbone.device)?.unsqueeze(0)?;
                                let opt_pooled = self.backbone.forward(&opt_input)?;
                                let opt_sum_sq = opt_pooled.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
                                let opt_norm = opt_pooled.broadcast_div(&(opt_sum_sq + 1e-9)?.sqrt()?)?;
                                let sim = h_norm.matmul(&opt_norm.t()?)?.flatten_all()?.to_vec1::<f32>()?[0];
                                opt_logits.push(sim / 0.1); // temperature scaled
                            } else {
                                opt_logits.push(0.0);
                            }
                        }
                        Tensor::from_vec(opt_logits, (1, options.len()), &self.backbone.device)?
                    };

                    let (winner, conf, dist) = self.choice_head.forward(&logits_tensor, &options)?;
                    decisions.insert(id, DecisionResult::Choice {
                        winner,
                        confidence: conf,
                        distribution: dist,
                    });
                }
                Question::Boolean { id } => {
                    let weight_key = format!("heads.boolean.{id}.weight");
                    let logit: f32 = if let Some(weight) = self.head_tensors.get(&weight_key) {
                        let mut logit_tensor = h_norm.matmul(&weight.t()?)?;
                        let bias_key = format!("heads.boolean.{id}.bias");
                        if let Some(bias) = self.head_tensors.get(&bias_key) {
                            logit_tensor = logit_tensor.broadcast_add(bias)?;
                        }
                        logit_tensor.flatten_all()?.to_vec1::<f32>()?[0]
                    } else {
                        0.0
                    };

                    let (val, prob) = BooleanHead::forward(logit);
                    decisions.insert(id, DecisionResult::Boolean {
                        value: val,
                        probability: prob,
                    });
                }
                Question::Score { id, min, max } => {
                    let min_val = min.unwrap_or(1);
                    let max_val = max.unwrap_or(5);
                    let count = (max_val - min_val + 1) as usize;

                    let weight_key = format!("heads.score.{id}.weight");
                    let logits_tensor = if let Some(weight) = self.head_tensors.get(&weight_key) {
                        let mut logits = h_norm.matmul(&weight.t()?)?;
                        let bias_key = format!("heads.score.{id}.bias");
                        if let Some(bias) = self.head_tensors.get(&bias_key) {
                            logits = logits.broadcast_add(bias)?;
                        }
                        logits
                    } else {
                        Tensor::zeros((1, count), candle_core::DType::F32, &self.backbone.device)?
                    };

                    let (expected, dist) = ScoreHead::forward(&logits_tensor, min_val, max_val)?;
                    decisions.insert(id, DecisionResult::Score {
                        expected_value: expected,
                        distribution: dist,
                    });
                }
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        #[cfg(target_arch = "wasm32")]
        let total_ms = tokenize_ms + forward_pass_ms;
        Ok(DecisionResponse {
            decisions,
            triage: triage_result,
            metrics: ExecutionMetrics {
                tokenize_ms,
                forward_pass_ms,
                total_ms,
            },
            cached: false,
            model_id: self.model_id.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_neural_spam_detection_classification() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let model_dir = manifest_dir.join("../../dist/models/spam-detector-v1");

        // Load real neural model bundle with transformer backbone and calibrated heads
        let engine = Sys1Engine::load_from_dir("spam-detector-v1", &model_dir)
            .expect("Failed to load real neural model bundle from dist/models/spam-detector-v1");

        let spam_questions = vec![
            Question::Boolean { id: "is_spam".to_string() },
            Question::Choice {
                id: "spam_category".to_string(),
                options: vec![
                    "legitimate_inquiry".to_string(),
                    "commercial_sales_pitch".to_string(),
                    "seo_backlink_spam".to_string(),
                    "crypto_phishing".to_string(),
                    "automated_bot_gibberish".to_string(),
                ],
            },
            Question::Score { id: "risk_score".to_string(), min: Some(1), max: Some(5) },
        ];

        // 1. Crypto Phishing Test
        let crypto_req = DecisionRequest {
            model: Some("spam-detector-v1".to_string()),
            state: "Urgent: Send 0.1 ETH to verify wallet and receive 5,000 promo tokens immediately to your balance.".to_string(),
            context_chunks: vec![],
            triage_config: None,
            questions: spam_questions.clone(),
        };

        let crypto_res = engine.execute(crypto_req).unwrap();
        if let DecisionResult::Boolean { value, probability } = &crypto_res.decisions["is_spam"] {
            assert!(value, "Crypto must be spam");
            assert!(*probability > 0.50);
        } else { panic!("Expected boolean result"); }

        if let DecisionResult::Choice { winner, .. } = &crypto_res.decisions["spam_category"] {
            assert_eq!(winner, "crypto_phishing");
        } else { panic!("Expected choice result"); }

        // 2. SEO Pitch Test
        let seo_req = DecisionRequest {
            model: Some("spam-detector-v1".to_string()),
            state: "Hello team, I noticed your website has great content but low Google ranking. We offer high-quality backlinks and guest posts with DA 80+ to get you to #1 on search engines. Contact my handle: @example_marketer".to_string(),
            context_chunks: vec![],
            triage_config: None,
            questions: spam_questions.clone(),
        };

        let seo_res = engine.execute(seo_req).unwrap();
        if let DecisionResult::Choice { winner, .. } = &seo_res.decisions["spam_category"] {
            assert_eq!(winner, "seo_backlink_spam");
        } else { panic!("Expected choice result"); }

        // 3. Cold Outreach BDR Test
        let sales_req = DecisionRequest {
            model: Some("spam-detector-v1".to_string()),
            state: "Hi, I work in the local metro area, and help many local companies. I was hoping I could come by and offer a complimentary cleaning bid? Thank you in advance for your response. All the best, Taylor Reed Business Development Rep Apex Facility Services taylor.reed@example-facility-services.com Respond with stop to optout.".to_string(),
            context_chunks: vec![],
            triage_config: None,
            questions: spam_questions.clone(),
        };

        let sales_res = engine.execute(sales_req).unwrap();
        if let DecisionResult::Boolean { value, .. } = &sales_res.decisions["is_spam"] {
            assert!(value, "Cold outreach must be classified as spam");
        } else { panic!("Expected boolean result"); }

        if let DecisionResult::Choice { winner, .. } = &sales_res.decisions["spam_category"] {
            assert_eq!(winner, "commercial_sales_pitch");
        } else { panic!("Expected choice result"); }

        // 4. Clean Inquiry Test
        let clean_req = DecisionRequest {
            model: Some("spam-detector-v1".to_string()),
            state: "Hello, I would like to schedule a technical briefing on deploying Sys1Pop across our global edge points of presence.".to_string(),
            context_chunks: vec![],
            triage_config: None,
            questions: spam_questions.clone(),
        };

        let clean_res = engine.execute(clean_req).unwrap();
        if let DecisionResult::Boolean { value, .. } = &clean_res.decisions["is_spam"] {
            assert!(!value, "Clean inquiry must NOT be spam");
        } else { panic!("Expected boolean result"); }

        if let DecisionResult::Choice { winner, .. } = &clean_res.decisions["spam_category"] {
            assert_eq!(winner, "legitimate_inquiry");
        } else { panic!("Expected choice result"); }
    }

    #[test]
    fn test_neural_rag_triage_semantic_scoring() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let model_dir = manifest_dir.join("../../dist/models/sys1-base");

        let engine = Sys1Engine::load_from_dir("sys1-base", &model_dir)
            .expect("Failed to load real neural model bundle from dist/models/sys1-base");

        let req = DecisionRequest {
            model: Some("sys1-base".to_string()),
            state: "How do I configure Cloudflare R2 bucket bindings in wrangler.toml?".to_string(),
            context_chunks: vec![
                "To bind an R2 bucket to your Worker, add [[r2_buckets]] to wrangler.toml with 'binding' and 'bucket_name' fields.".to_string(),
                "Cloudflare R2 provides zero egress fee object storage for web workers and media assets.".to_string(),
                "The classic chocolate chip cookie recipe calls for flour, butter, brown sugar, and baking powder.".to_string(),
            ],
            triage_config: Some(crate::contract::TriageConfig {
                relevance_threshold: 0.35,
                sufficiency_threshold: 0.5,
                strategy: None,
                max_retained_chunks: None,
            }),
            questions: vec![],
        };

        let res = engine.execute(req).unwrap();
        assert!(res.triage.is_some(), "Expected triage result");
        let triage = res.triage.unwrap();
        assert_eq!(triage.chunks_evaluated, 3);
        // Doc 0 (R2 wrangler binding) must have higher cosine similarity than Doc 2 (chocolate chip cookies)
        assert!(triage.relevance_scores[0] > triage.relevance_scores[2]);
        assert!(triage.relevance_scores[0] > 0.40, "R2 docs must have high relevance");
        assert!(triage.relevance_scores[2] < 0.35, "Cookie recipe must have low relevance");
        // Retained chunks should include Doc 0 and Doc 1, but NOT Doc 2
        assert!(triage.retained_chunks.contains(&0));
        assert!(!triage.retained_chunks.contains(&2));
    }

    #[test]
    fn test_neural_sys1_base_intent_classification() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let model_dir = manifest_dir.join("../../dist/models/sys1-base");

        let engine = Sys1Engine::load_from_dir("sys1-base", &model_dir)
            .expect("Failed to load real neural model bundle from dist/models/sys1-base");

        let questions = vec![
            Question::Choice {
                id: "primary_intent".to_string(),
                options: vec![
                    "information_lookup".to_string(),
                    "transaction_request".to_string(),
                    "escalation".to_string(),
                    "general_feedback".to_string(),
                ],
            },
            Question::Boolean { id: "requires_action".to_string() },
        ];

        let req = DecisionRequest {
            model: Some("sys1-base".to_string()),
            state: "Our production database is experiencing connection exhaustion and 500 errors across all nodes, critical!".to_string(),
            context_chunks: vec![],
            triage_config: None,
            questions,
        };

        let res = engine.execute(req).unwrap();
        if let DecisionResult::Choice { winner, confidence, .. } = &res.decisions["primary_intent"] {
            assert_eq!(winner, "escalation");
            assert!(*confidence > 0.60);
        } else { panic!("Expected choice result"); }

        if let DecisionResult::Boolean { value, .. } = &res.decisions["requires_action"] {
            assert!(value, "Outage requires action");
        } else { panic!("Expected boolean result"); }
    }

    #[test]
    fn test_modelforge_support_triage_classification() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let model_dir = manifest_dir.join("../../dist/models/support-triage-v1");
        if !model_dir.exists() {
            return;
        }

        let engine = Sys1Engine::load_from_dir("support-triage-v1", &model_dir)
            .expect("Failed to load ModelForge bundle from dist/models/support-triage-v1");

        let triage_questions = vec![
            Question::Choice {
                id: "department".to_string(),
                options: vec![
                    "billing".to_string(),
                    "technical_support".to_string(),
                    "sales".to_string(),
                    "general".to_string(),
                ],
            },
            Question::Boolean { id: "requires_escalation".to_string() },
            Question::Score { id: "urgency_rating".to_string(), min: Some(1), max: Some(5) },
        ];

        // 1. Billing Inquiry
        let billing_req = DecisionRequest {
            model: Some("support-triage-v1".to_string()),
            state: "Site: store.com | Name: Alice | Subject: Double charged on invoice #4821 and card charged twice.".to_string(),
            context_chunks: vec![],
            triage_config: None,
            questions: triage_questions.clone(),
        };

        let billing_res = engine.execute(billing_req).unwrap();
        if let DecisionResult::Choice { winner, confidence, .. } = &billing_res.decisions["department"] {
            assert_eq!(winner, "billing");
            assert!(*confidence > 0.40);
        } else { panic!("Expected choice result"); }

        if let DecisionResult::Boolean { value, .. } = &billing_res.decisions["requires_escalation"] {
            assert!(!value, "Routine billing should not require escalation");
        } else { panic!("Expected boolean result"); }

        // 2. Production Outage Incident
        let outage_req = DecisionRequest {
            model: Some("support-triage-v1".to_string()),
            state: "Site: app.io | Name: Bob | Subject: Production database down with 500 errors across all clusters!".to_string(),
            context_chunks: vec![],
            triage_config: None,
            questions: triage_questions.clone(),
        };

        let outage_res = engine.execute(outage_req).unwrap();
        if let DecisionResult::Choice { winner, .. } = &outage_res.decisions["department"] {
            assert_eq!(winner, "technical_support");
        } else { panic!("Expected choice result"); }

        if let DecisionResult::Boolean { value, probability } = &outage_res.decisions["requires_escalation"] {
            assert!(value, "Outage must require escalation");
            assert!(*probability > 0.60);
        } else { panic!("Expected boolean result"); }

        if let DecisionResult::Score { expected_value, .. } = &outage_res.decisions["urgency_rating"] {
            assert!(*expected_value >= 3.5, "Urgency should be >= 3.5 for prod outage, got {}", expected_value);
        } else { panic!("Expected score result"); }
    }
}
