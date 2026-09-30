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

fn verify_admin_auth(req: &Request, env: &Env) -> Result<Option<Response>> {
    // Check if admin API is explicitly disabled
    if let Ok(enabled_var) = env.var("ENABLE_ADMIN_API") {
        let val = enabled_var.to_string().to_lowercase();
        if val == "false" || val == "0" || val == "no" {
            return Ok(Some(Response::error("Admin lifecycle API is disabled", 403)?));
        }
    }

    // Check API_TOKEN if configured
    if let Ok(expected_token) = env.var("API_TOKEN").or_else(|_| env.secret("API_TOKEN")) {
        let expected = expected_token.to_string();
        if !expected.is_empty() {
            let headers = req.headers();
            let auth_header = headers.get("Authorization").ok().flatten();
            let x_token = headers.get("X-API-Token").ok().flatten();

            let token = auth_header
                .and_then(|h| h.strip_prefix("Bearer ").map(|s| s.to_string()))
                .or(x_token);

            match token {
                Some(t) if t == expected => (),
                _ => return Ok(Some(Response::error("Unauthorized: Invalid or missing API token", 401)?)),
            }
        }
    }
    Ok(None)
}

#[event(fetch)]
pub async fn main(mut req: Request, env: Env, _ctx: Context) -> Result<Response> {
    std::panic::set_hook(Box::new(|info| {
        worker::console_error!("Sys1Pop Panic: {}", info);
    }));

    let method = req.method();
    let path = req.path();

    match (method, path.as_str()) {
        (Method::Get, "/") | (Method::Get, "/ui") => {
            let enable_ui = env.var("ENABLE_UI")
                .map(|v| {
                    let val = v.to_string().to_lowercase();
                    val == "true" || val == "1" || val == "yes" || val == "on"
                })
                .unwrap_or(false);

            if enable_ui {
                let html = include_str!("../ui/index.html");
                let headers = Headers::new();
                let _ = headers.set("Content-Type", "text/html; charset=utf-8");
                let _ = headers.set("Cache-Control", "no-cache");
                return Response::ok(html).map(|r| r.with_headers(headers));
            }

            Response::from_json(&serde_json::json!({
                "service": "sys1pop",
                "version": "0.1.0",
                "status": "online",
                "engine": "candle-wasm",
                "ui": "disabled",
                "endpoints": {
                    "decide": "POST /v1/decide",
                    "models": "GET /v1/models",
                    "health": "GET /health"
                },
                "hint": "Set ENABLE_UI=true in wrangler.toml or Cloudflare environment variables to activate the interactive test playground."
            }))
        }
        (Method::Get, "/sys1pop_robot_blowing_bubbles.svg") | (Method::Get, "/ui/sys1pop_robot_blowing_bubbles.svg") => {
            let svg = include_str!("../ui/sys1pop_robot_blowing_bubbles.svg");
            let headers = Headers::new();
            let _ = headers.set("Content-Type", "image/svg+xml");
            let _ = headers.set("Cache-Control", "public, max-age=86400");
            Response::ok(svg).map(|r| r.with_headers(headers))
        }
        (Method::Get, "/sys1pop_wordmark_nixie.svg") | (Method::Get, "/ui/sys1pop_wordmark_nixie.svg") => {
            let svg = include_str!("../ui/sys1pop_wordmark_nixie.svg");
            let headers = Headers::new();
            let _ = headers.set("Content-Type", "image/svg+xml");
            let _ = headers.set("Cache-Control", "public, max-age=86400");
            Response::ok(svg).map(|r| r.with_headers(headers))
        }
        (Method::Get, "/sys1pop_wordmark_nixie_dark.svg") | (Method::Get, "/ui/sys1pop_wordmark_nixie_dark.svg") => {
            let svg = include_str!("../ui/sys1pop_wordmark_nixie_dark.svg");
            let headers = Headers::new();
            let _ = headers.set("Content-Type", "image/svg+xml");
            let _ = headers.set("Cache-Control", "public, max-age=86400");
            Response::ok(svg).map(|r| r.with_headers(headers))
        }
        (Method::Get, "/sys1pop_banner_dark_pure.svg") | (Method::Get, "/ui/sys1pop_banner_dark_pure.svg") => {
            let svg = include_str!("../ui/sys1pop_banner_dark_pure.svg");
            let headers = Headers::new();
            let _ = headers.set("Content-Type", "image/svg+xml");
            let _ = headers.set("Cache-Control", "public, max-age=86400");
            Response::ok(svg).map(|r| r.with_headers(headers))
        }
        (Method::Get, "/favicon.ico") | (Method::Get, "/favicon.png") => {
            let bytes = include_bytes!("../ui/favicon.png");
            let headers = Headers::new();
            let _ = headers.set("Content-Type", "image/png");
            let _ = headers.set("Cache-Control", "public, max-age=86400");
            Response::from_bytes(bytes.to_vec()).map(|r| r.with_headers(headers))
        }
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
            let catalog = registry.get_catalog(&env).await;
            Response::from_json(&catalog)
        }
        (Method::Post, "/v1/models/unload") => {
            if let Some(err_resp) = verify_admin_auth(&req, &env)? {
                return Ok(err_resp);
            }
            #[derive(serde::Deserialize)]
            struct UnloadReq {
                model: String,
            }
            let unload_body: UnloadReq = match req.json().await {
                Ok(b) => b,
                Err(err) => return Response::error(format!("Invalid JSON request: {err}"), 400),
            };
            let registry = get_or_init_registry();
            let unloaded = registry.unload(&unload_body.model);
            Response::from_json(&serde_json::json!({
                "status": if unloaded { "unloaded" } else { "not_found" },
                "model": unload_body.model,
                "remaining_warm": registry.loaded_models()
            }))
        }
        (Method::Post, "/v1/cache/clear") => {
            if let Some(err_resp) = verify_admin_auth(&req, &env)? {
                return Ok(err_resp);
            }
            let cache = get_or_init_cache();
            cache.clear();
            Response::from_json(&serde_json::json!({
                "status": "cleared",
                "cache_entries": 0
            }))
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
            worker::console_log!("Resolving engine for model: {}", model_id);
            let registry = get_or_init_registry();
            let engine = registry.get_or_load(model_id, &env).await?;

            worker::console_log!("Executing decision on engine for model: {}", model_id);
            let response = engine.execute(body.clone())
                .map_err(|e| {
                    worker::console_error!("Engine execute error: {}", e);
                    worker::Error::RustError(e.to_string())
                })?;
            worker::console_log!("Decision execution completed successfully!");

            // Store in in-isolate LRU cache
            cache.insert(&body, &response);

            Response::from_json(&response)
        }
        _ => Response::error("Not Found", 404),
    }
}
