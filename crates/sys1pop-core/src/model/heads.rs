use candle_core::{Result, Tensor};
use std::collections::HashMap;

/// Decision head computing a probability distribution over discrete candidate options.
pub struct ChoiceHead {
    temperature: f64,
}

impl ChoiceHead {
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    /// Evaluates candidate logits via temperature-scaled softmax.
    pub fn forward(&self, logits: &Tensor, options: &[String]) -> Result<(String, f32, HashMap<String, f32>)> {
        let scaled_logits = (logits / self.temperature)?;
        let probs = candle_nn::ops::softmax(&scaled_logits, candle_core::D::Minus1)?;
        let probs_vec: Vec<f32> = probs.flatten_all()?.to_vec1()?;

        let mut distribution = HashMap::new();
        let mut max_prob = -1.0;
        let mut winner = String::new();

        for (idx, opt) in options.iter().enumerate() {
            let prob = probs_vec.get(idx).copied().unwrap_or(0.0);
            distribution.insert(opt.clone(), prob);
            if prob > max_prob {
                max_prob = prob;
                winner = opt.clone();
            }
        }

        Ok((winner, max_prob, distribution))
    }
}

/// Decision head computing a calibrated boolean probability using sigmoid.
pub struct BooleanHead;

impl BooleanHead {
    pub fn forward(logit: f32) -> (bool, f32) {
        let prob = 1.0 / (1.0 + (-logit).exp());
        let value = prob >= 0.5;
        (value, prob)
    }
}

/// Decision head computing expected value over ordinal bins.
pub struct ScoreHead;

impl ScoreHead {
    pub fn forward(logits: &Tensor, min: i32, max: i32) -> Result<(f32, Vec<f32>)> {
        let probs = candle_nn::ops::softmax(logits, candle_core::D::Minus1)?;
        let probs_vec: Vec<f32> = probs.flatten_all()?.to_vec1()?;
        
        let count = (max - min + 1) as usize;
        let mut expected_value = 0.0;
        
        for (i, &p) in probs_vec.iter().take(count).enumerate() {
            let score_val = (min + i as i32) as f32;
            expected_value += score_val * p;
        }

        Ok((expected_value, probs_vec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_boolean_head() {
        let (val, prob) = BooleanHead::forward(2.0);
        assert!(val);
        assert!(prob > 0.85);

        let (val_neg, prob_neg) = BooleanHead::forward(-2.0);
        assert!(!val_neg);
        assert!(prob_neg < 0.15);
    }

    #[test]
    fn test_choice_head() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0f32, 5.0, 2.0], (1, 3), &device).unwrap();
        let options = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let head = ChoiceHead::new(1.0);
        let (winner, max_prob, dist) = head.forward(&logits, &options).unwrap();
        assert_eq!(winner, "B");
        assert!(max_prob > 0.90);
        assert_eq!(dist.len(), 3);
    }

    #[test]
    fn test_score_head() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4, 5.0], (1, 5), &device).unwrap();
        let (score, _) = ScoreHead::forward(&logits, 1, 5).unwrap();
        assert!(score > 4.5);
    }
}
