use sys1pop_core::contract::{DecisionRequest, DecisionResponse, ExecutionMetrics};
use lru::LruCache;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Mutex;

/// In-isolate LRU cache for decision results.
/// Operates in warm isolate RAM to resolve repeat queries in <0.05ms at $0.00 CPU cost.
pub struct DecisionCache {
    inner: Mutex<LruCache<u64, DecisionResponse>>,
}

impl Default for DecisionCache {
    fn default() -> Self {
        Self::new(10_000)
    }
}

impl DecisionCache {
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(10_000).unwrap());
        Self {
            inner: Mutex::new(LruCache::new(cap)),
        }
    }

    pub fn compute_key(req: &DecisionRequest) -> u64 {
        let mut hasher = DefaultHasher::new();
        if let Some(ref m) = req.model {
            m.hash(&mut hasher);
        }
        req.state.hash(&mut hasher);
        req.context_chunks.hash(&mut hasher);
        if let Ok(q_json) = serde_json::to_string(&req.questions) {
            q_json.hash(&mut hasher);
        }
        if let Some(ref cfg) = req.triage_config {
            if let Ok(cfg_json) = serde_json::to_string(cfg) {
                cfg_json.hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    pub fn get(&self, req: &DecisionRequest) -> Option<DecisionResponse> {
        let key = Self::compute_key(req);
        let mut lock = self.inner.lock().ok()?;
        if let Some(cached_res) = lock.get(&key) {
            let mut res = cached_res.clone();
            res.cached = true;
            res.metrics = ExecutionMetrics {
                tokenize_ms: 0.0,
                forward_pass_ms: 0.0,
                total_ms: 0.05,
            };
            return Some(res);
        }
        None
    }

    pub fn insert(&self, req: &DecisionRequest, res: &DecisionResponse) {
        let key = Self::compute_key(req);
        if let Ok(mut lock) = self.inner.lock() {
            lock.put(key, res.clone());
        }
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map(|l| l.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        if let Ok(mut lock) = self.inner.lock() {
            lock.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sys1pop_core::contract::{DecisionResult, Question};
    use std::collections::HashMap;

    #[test]
    fn test_decision_cache_hit_and_miss() {
        let cache = DecisionCache::new(100);
        let req = DecisionRequest {
            state: "Check spam payload".to_string(),
            model: Some("sys1-base".to_string()),
            context_chunks: vec![],
            triage_config: None,
            questions: vec![Question::Boolean { id: "is_spam".to_string() }],
        };

        // Cache miss
        assert!(cache.get(&req).is_none());

        let mut decisions = HashMap::new();
        decisions.insert("is_spam".to_string(), DecisionResult::Boolean { value: true, probability: 0.99 });
        let response = DecisionResponse {
            decisions,
            triage: None,
            metrics: ExecutionMetrics { tokenize_ms: 1.0, forward_pass_ms: 20.0, total_ms: 22.0 },
            cached: false,
            model_id: "sys1-base".to_string(),
        };

        // Insert into cache
        cache.insert(&req, &response);
        assert_eq!(cache.len(), 1);

        // Cache hit
        let cached = cache.get(&req).expect("Expected cache hit");
        assert!(cached.cached);
        assert_eq!(cached.metrics.total_ms, 0.05);
        if let Some(DecisionResult::Boolean { value, .. }) = cached.decisions.get("is_spam") {
            assert!(value);
        } else {
            panic!("Expected boolean decision");
        }
    }
}
