use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};

pub struct QuantizedBackbone {
    pub hidden_dim: usize,
    pub device: Device,
    pub model: Option<BertModel>,
}

impl QuantizedBackbone {
    pub fn new(hidden_dim: usize, device: Device) -> Self {
        Self { hidden_dim, device, model: None }
    }

    pub fn from_safetensors(bytes: Vec<u8>, config: &Config, device: Device) -> Result<Self> {
        let vb = VarBuilder::from_buffered_safetensors(bytes, DType::F32, &device)?;
        let model = BertModel::load(vb, config)?;
        Ok(Self {
            hidden_dim: config.hidden_size,
            device,
            model: Some(model),
        })
    }

    /// Forward pass executing transformer attention layers and mean pooling token embeddings
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        if let Some(ref model) = self.model {
            let token_type_ids = token_ids.zeros_like()?;
            let hidden_states = model.forward(token_ids, &token_type_ids, None)?;
            // Sentence-Transformers standard: Mean pooling over token sequence (dimension 1)
            hidden_states.mean(1)
        } else {
            let (batch_size, _seq_len) = token_ids.dims2()?;
            Tensor::zeros((batch_size, self.hidden_dim), DType::F32, &self.device)
        }
    }
}
