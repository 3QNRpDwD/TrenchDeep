use super::*;

pub const MAX_PERIOD: f32 = 10000.0;

pub struct Encoder;

#[derive(Debug)]
pub struct SinusoidalPE {
    inv_freq: Variable,
    label: String
}

impl SinusoidalPE {
    pub fn new(dim: usize, label: &str) -> MlResult<Self> {
        // max_period = 보통 10000
        let half_dim = dim / 2;
        let data: Vec<f32> = (0..half_dim)
            .map(|i| (MAX_PERIOD).powf(2.0 * i as f32 / dim as f32).recip())
            .collect();
        Ok(SinusoidalPE { inv_freq: Variable::new(Tensor::from_vec(data, &[1, half_dim])?), label: label.to_string() })
    }
}

impl Layer for SinusoidalPE {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let mut sin = Sin::new()?;
        let mut cos = Cos::new()?;
        let mut matmul = Matmul::new()?;
        let mut concat = Concat::new()?;
        // 1. 분모(Inverse Frequencies) 미리 계산
        // weights = [exp(i * -ln(10000) / (dim/2))] for i in 0..dim/2

        // 2. 외적(Outer Product) 수행: (Batch, 1) * (1, dim/2) -> (Batch, dim/2)
        // t_tensor: [t1, t2, t3, t4]
        let args = matmul.apply(&[input, &self.inv_freq])?;

        // 3. Sin과 Cos을 각각 적용
        let sin_features = sin.apply(&[&args])?;
        let cos_features = cos.apply(&[&args])?;

        // 4. 마지막 차원을 기준으로 결합 (Concatenate)
        // 결과: (Batch, dim) 형태의 텐서 하나를 반환

        Ok(concat.apply(&[&sin_features, &cos_features])?)
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let sin = Sin::new()?;
        let cos = Cos::new()?;
        let matmul = Matmul::new()?;
        let concat = Concat::new()?;
        // 1. 분모(Inverse Frequencies) 미리 계산 (한 번만 수행하거나 생성자에서 처리 권장)
        // weights = [exp(i * -ln(10000) / (dim/2))] for i in 0..dim/2

        // 2. 외적(Outer Product) 수행: (Batch, 1) * (1, dim/2) -> (Batch, dim/2)
        // t_tensor: [t1, t2, t3, t4]
        let args = matmul.forward(&[input, self.inv_freq.tensor()])?;

        // 3. Sin과 Cos을 각각 적용
        let sin_features = sin.forward(&[&args[0]])?;
        let cos_features = cos.forward(&[&args[0]])?;

        // 4. 마지막 차원을 기준으로 결합 (Concatenate)
        // 결과: (Batch, dim) 형태의 텐서 하나를 반환

        Ok(concat.forward(&[&sin_features[0], &cos_features[0]])?.remove(0))
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        vec![]
    }

    fn label(&self) -> &str {
        self.label.as_str()
    }
}
