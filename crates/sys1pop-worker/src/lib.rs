pub mod cache;
pub mod loader;

use cache::DecisionCache;
use loader::ModelRegistry;
use sys1pop_core::contract::DecisionRequest;
use once_cell::sync::OnceCell;
use worker::*;

pub static CACHE: OnceCell<DecisionCache> = OnceCell::new();
pub static REGISTRY: OnceCell<ModelRegistry> = OnceCell::new();

pub fn get_or_init_cache() -> &'static DecisionCache {
    CACHE.get_or_init(DecisionCache::default)
}

pub fn get_or_init_registry() -> &'static ModelRegistry {
    REGISTRY.get_or_init(ModelRegistry::default)
}

#[event(fetch)]
pub async fn main(mut req: Request, env: Env, _ctx: Context) -> Result<Response> {
    let method = req.method();
    let path = req.path();

    match (method, path.as_str()) {
        (Method::Get, "/health") => {
            let registry = get_or_init_registry();
            let cache = get_or_init_cache();
            let warm_models = registry.loaded_models();
            let cache_entries = cache.len();

            let payload = serde_json::json!({
                "status": "healthy",
                "service": "sys1pop",
                "engine": "candle-wasm",
                "cache_entries": cache_entries,
                "warm_models": warm_models,
            });

            Response::from_json(&payload)
        }
        (Method::Get, "/v1/models") => {
            let registry = get_or_init_registry();
            let models = registry.loaded_models();
            Response::from_json(&serde_json::json!({ "models": models }))
        }
        (Method::Post, "/v1/decide") => {
            let body: DecisionRequest = match req.json().await {
                Ok(b) => b,
                Err(err) => return Response::error(format!("Invalid JSON request: {err}"), 400),
            };

            let cache = get_or_init_cache();
            // Fast Path: Check in-isolate LRU cache (<0.05ms, $0.00 CPU)
            if let Some(cached_res) = cache.get(&body) {
                return Response::from_json(&cached_res);
            }

            let model_id = body.model.as_deref().unwrap_or("sys1-base");
            let registry = get_or_init_registry();
            let engine = registry.get_or_load(model_id, &env).await?;

            let response = engine.execute(body.clone())
                .map_err(|e| worker::Error::RustError(e.to_string()))?;

            // Store in in-isolate LRU cache
            cache.insert(&body, &response);

            Response::from_json(&response)
        }
        _ => Response::error("Not Found", 404),
    }
}
