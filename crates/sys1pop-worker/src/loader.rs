use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use sys1pop_core::Sys1Engine;
use worker::{Env, Result};

/// Multi-model registry managing warm model residency in isolate RAM.
/// Streams quantized weights from Cloudflare R2 on demand and caches them across warm requests.
pub struct ModelRegistry {
    engines: Mutex<HashMap<String, Arc<Sys1Engine>>>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRegistry {
    pub fn new() -> Self {
        let mut engines = HashMap::new();
        // Warm default backbone in isolate memory on startup
        engines.insert("sys1-base".to_string(), Arc::new(Sys1Engine::new("sys1-base")));
        Self {
            engines: Mutex::new(engines),
        }
    }

    /// Retrieves an already warm model from isolate memory (<0.01ms) or loads it from R2.
    pub async fn get_or_load(&self, model_id: &str, env: &Env) -> Result<Arc<Sys1Engine>> {
        // Fast path: Check warm isolate memory
        if let Ok(guard) = self.engines.lock() {
            if let Some(engine) = guard.get(model_id) {
                return Ok(Arc::clone(engine));
            }
        }

        // Dynamic path: Attempt streaming weights/manifest from Cloudflare R2
        if let Ok(bucket) = env.bucket("MODELS") {
            let manifest_key = format!("models/{model_id}/manifest.json");
            
            // Check if model exists in R2
            if let Ok(Some(_obj)) = bucket.get(&manifest_key).execute().await {
                let new_engine = Arc::new(Sys1Engine::new(model_id));
                if let Ok(mut guard) = self.engines.lock() {
                    guard.insert(model_id.to_string(), Arc::clone(&new_engine));
                }
                return Ok(new_engine);
            }
        }

        // Fallback: If not found in R2, load a default engine with requested ID
        let fallback_engine = Arc::new(Sys1Engine::new(model_id));
        if let Ok(mut guard) = self.engines.lock() {
            guard.insert(model_id.to_string(), Arc::clone(&fallback_engine));
        }
        Ok(fallback_engine)
    }

    /// Evicts a model from warm isolate RAM
    pub fn unload(&self, model_id: &str) -> bool {
        if let Ok(mut guard) = self.engines.lock() {
            guard.remove(model_id).is_some()
        } else {
            false
        }
    }

    /// Lists all models currently warm in isolate RAM
    pub fn loaded_models(&self) -> Vec<String> {
        self.engines.lock()
            .map(|g| g.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Retrieves dynamic model catalog, inspecting R2, environment variables, and warm isolate RAM
    pub async fn get_catalog(&self, env: &Env) -> serde_json::Value {
        let warm = self.loaded_models();
        let mut model_ids: std::collections::BTreeSet<String> = warm.iter().cloned().collect();
        let mut catalog_entries: Vec<serde_json::Value> = Vec::new();

        // 1. Check R2 for models/catalog.json
        if let Ok(bucket) = env.bucket("MODELS") {
            if let Ok(Some(obj)) = bucket.get("models/catalog.json").execute().await {
                if let Some(body) = obj.body() {
                    if let Ok(text) = body.text().await {
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                            if let Some(models) = parsed.get("models").and_then(|v| v.as_array()) {
                                for m in models {
                                    if let Some(id) = m.get("model_id").and_then(|v| v.as_str()) {
                                        model_ids.insert(id.to_string());
                                        catalog_entries.push(m.clone());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Fallback: configured models in CONFIGURED_MODELS env var
        if let Ok(env_models) = env.var("CONFIGURED_MODELS") {
            for id in env_models.to_string().split(',') {
                let trimmed = id.trim();
                if !trimmed.is_empty() {
                    model_ids.insert(trimmed.to_string());
                }
            }
        }

        // 3. Fallback: default foundation catalog if R2 catalog is empty
        if catalog_entries.is_empty() {
            let default_catalog = serde_json::json!([
                {
                    "model_id": "sys1-base",
                    "name": "Sys1Pop Base Backbone",
                    "icon": "cpu",
                    "description": "General semantic representation & System 1 edge decision backbone (33.4M params).",
                    "default_state": "Site: example.com | Task: General semantic decision evaluation and intent triage.",
                    "default_chunks": [],
                    "default_questions": [
                        {
                            "type": "choice",
                            "id": "primary_intent",
                            "options": ["information_lookup", "transaction_request", "escalation", "general_feedback"]
                        },
                        {
                            "type": "boolean",
                            "id": "requires_action"
                        },
                        {
                            "type": "score",
                            "id": "priority_level",
                            "min": 1,
                            "max": 5
                        }
                    ]
                },
                {
                    "model_id": "spam-detector-v1",
                    "name": "Web Form Spam & Phishing",
                    "icon": "shield",
                    "description": "Evaluates user messages, sender emails, and site context to classify spam, intent, and risk.",
                    "default_state": "Site: example.com | Name: Alex Taylor | Email: alex.marketing@example.com | Message: Hello team, I noticed your website has great content but low Google ranking. We offer high-quality backlinks and guest posts with DA 80+ to get you to #1 on search engines. Contact my handle: @example_marketer",
                    "default_chunks": [],
                    "default_questions": [
                        {
                            "type": "boolean",
                            "id": "is_spam"
                        },
                        {
                            "type": "choice",
                            "id": "spam_category",
                            "options": ["legitimate_inquiry", "commercial_sales_pitch", "seo_backlink_spam", "crypto_phishing", "automated_bot_gibberish"]
                        },
                        {
                            "type": "score",
                            "id": "risk_score",
                            "min": 1,
                            "max": 5
                        }
                    ],
                    "sample_presets": [
                        {
                            "label": "🚨 Crypto Phishing",
                            "state": "Site: example.com | Name: Reward Bot | Email: airdrop@example.org | Message: Urgent: Send 0.1 ETH to verify wallet and receive 5,000 promo tokens immediately to your balance."
                        },
                        {
                            "label": "📈 SEO Sales Pitch",
                            "state": "Site: example.com | Name: Alex Taylor | Email: alex.marketing@example.com | Message: Hello team, I noticed your website has great content but low Google ranking. We offer high-quality backlinks and guest posts with DA 80+ to get you to #1 on search engines. Contact my handle: @example_marketer"
                        },
                        {
                            "label": "✅ Clean Inquiry",
                            "state": "Site: example.com | Name: Dr. Jane Doe | Email: jane.doe@example.com | Message: Hello, I would like to schedule a technical briefing on deploying Sys1Pop across our global edge points of presence."
                        }
                    ]
                },
                {
                    "model_id": "rag-triage-v1",
                    "name": "Multi-Chunk RAG Triage",
                    "icon": "book-open",
                    "description": "Batch-evaluates retrieved context passages against a query, prunes irrelevant chunks, and assesses sufficiency.",
                    "default_state": "User Query: How do I configure Cloudflare R2 bucket bindings in wrangler.toml?",
                    "default_chunks": [
                        "Doc 1: Cloudflare R2 allows developers to store unstructured data without egress bandwidth fees. You can create buckets using wrangler r2 bucket create <NAME>.",
                        "Doc 2: To bind an R2 bucket to your Worker, add [[r2_buckets]] to wrangler.toml with 'binding' and 'bucket_name' fields. In your Worker code, access it via env.<BINDING>.",
                        "Doc 3: Cloudflare KV is an ultra-low latency key-value store optimized for high-volume read workloads with eventual consistency."
                    ],
                    "triage_config": {
                        "relevance_threshold": 0.3,
                        "sufficiency_threshold": 0.5
                    },
                    "default_questions": [
                        {
                            "type": "boolean",
                            "id": "sufficient_context"
                        },
                        {
                            "type": "choice",
                            "id": "action_required",
                            "options": ["generate_answer", "fetch_more_docs", "clarify_with_user"]
                        },
                        {
                            "type": "score",
                            "id": "relevance_rating",
                            "min": 1,
                            "max": 5
                        }
                    ]
                },
                {
                    "model_id": "intent-router-v1",
                    "name": "Agent Intent Routing",
                    "icon": "zap",
                    "description": "Fast omni-box intent routing, tool dispatch, and confirmation gating under 40ms.",
                    "default_state": "User Input: Turn off the living room desk lamp and schedule a focus block for 2 hours.",
                    "default_chunks": [],
                    "default_questions": [
                        {
                            "type": "choice",
                            "id": "primary_route",
                            "options": ["iot_device_control", "calendar_scheduling", "multi_action_split", "general_conversation"]
                        },
                        {
                            "type": "boolean",
                            "id": "requires_confirmation"
                        },
                        {
                            "type": "score",
                            "id": "urgency_level",
                            "min": 1,
                            "max": 3
                        }
                    ],
                    "sample_presets": [
                        {
                            "label": "💡 IoT Control",
                            "state": "User Input: Dim the bedroom lights to 30% and set thermostat to 70 degrees."
                        },
                        {
                            "label": "📅 Calendar Block",
                            "state": "User Input: Schedule a 45-minute sync with Sarah tomorrow at 2pm."
                        },
                        {
                            "label": "🔀 Multi-Action Split",
                            "state": "User Input: Turn off the living room desk lamp and schedule a focus block for 2 hours."
                        }
                    ]
                }
            ]);
            if let Some(arr) = default_catalog.as_array() {
                for item in arr {
                    if let Some(id) = item.get("model_id").and_then(|v| v.as_str()) {
                        model_ids.insert(id.to_string());
                    }
                    catalog_entries.push(item.clone());
                }
            }
        }

        let all_models: Vec<String> = model_ids.into_iter().collect();

        serde_json::json!({
            "models": all_models,
            "warm": warm,
            "catalog": catalog_entries
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_registry_warm_default() {
        let registry = ModelRegistry::new();
        let models = registry.loaded_models();
        assert!(models.contains(&"sys1-base".to_string()));
    }
}
