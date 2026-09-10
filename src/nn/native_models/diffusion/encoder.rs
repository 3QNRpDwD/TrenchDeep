use super::*;

pub const MAX_PERIOD: f32 = 10000.0;

// TODO(LatentDiffusion): VAE 인코더 구현 예정. 삭제 금지.
#[allow(dead_code)]
pub struct Encoder;

/// Sinusoidal Positional Embedding.
///
/// 입력 shape: `[B, 1]` (timestep 스칼라 배치).
/// 출력 shape: `[B, dim]` (sin half + cos half concat).
#[derive(Debug)]
pub struct SinusoidalPE {
    inv_freq: Variable,
    concat_axis: Variable, // Concat 연산자가 요구하는 axis 스칼라 ([1,1], value=1.0 = 마지막 축)
    label: String,
}

impl SinusoidalPE {
    pub fn new(dim: usize, label: &str) -> MlResult<Self> {
        let half_dim = dim / 2;
        let data: Vec<f32> = (0..half_dim)
            .map(|i| (MAX_PERIOD).powf(2.0 * i as f32 / dim as f32).recip())
            .collect();
        Ok(SinusoidalPE {
            inv_freq: Variable::new(Tensor::from_vec(data, &[1, half_dim])?),
            concat_axis: Variable::new(Tensor::from_vec(vec![1.0], &[1, 1])?),
            label: label.to_string(),
        })
    }
}

impl Layer for SinusoidalPE {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let mut sin = Sin::new()?;
        let mut cos = Cos::new()?;
        let mut matmul = Matmul::new()?;
        let mut concat = Concat::new()?;

        // (B, 1) · (1, half) → (B, half)
        let args = matmul.apply(&[input, &self.inv_freq])?;
        let sin_features = sin.apply(&[&args])?;
        let cos_features = cos.apply(&[&args])?;

        // axis=1 로 concat → (B, dim)
        concat.apply(&[&sin_features, &cos_features, &self.concat_axis])
    }

    fn predict(&self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let sin = Sin::new()?;
        let cos = Cos::new()?;
        let matmul = Matmul::new()?;
        let concat = Concat::new()?;

        let args = matmul.forward(&[input, self.inv_freq.tensor()])?;
        let sin_features = sin.forward(&[&args[0]])?;
        let cos_features = cos.forward(&[&args[0]])?;

        Ok(concat
            .forward(&[&sin_features[0], &cos_features[0], self.concat_axis.tensor()])?
            .remove(0))
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        vec![]
    }

    fn label(&self) -> &str {
        self.label.as_str()
    }
}

#[cfg(test)]
#[path = "../../../tests/nn/native_models/diffusion/encoder_tests.rs"]
mod tests;
