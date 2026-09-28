pub mod contract;
pub mod engine;
pub mod error;
pub mod model;
pub mod pipeline;

pub use contract::{
    DecisionRequest, DecisionResponse, DecisionResult, ExecutionMetrics, Question, RAGTriageResult,
    TriageConfig,
};
pub use engine::Sys1Engine;
pub use error::{Error, Result};
pub use model::{BooleanHead, ChoiceHead, QuantizedBackbone, ScoreHead};
pub use pipeline::RAGTriagePipeline;
