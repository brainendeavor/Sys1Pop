use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Engine error: {0}")]
    Engine(String),
}

pub type Result<T> = std::result::Result<T, Error>;
