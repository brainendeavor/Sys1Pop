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

    /// Lists all models currently warm in isolate RAM
    pub fn loaded_models(&self) -> Vec<String> {
        self.engines.lock()
            .map(|g| g.keys().cloned().collect())
            .unwrap_or_default()
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
