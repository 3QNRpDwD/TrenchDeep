use super::*;

layer_params!(Linear, "Linear", [weight, bias], |s| serde_json::json!({
    "in_features":  s.weight.tensor().shape()[0],
    "out_features": s.weight.tensor().shape()[1],
}));

impl Linear {
    /// 새로운 Linear 레이어를 생성합니다.
    ///
    /// # Arguments
    /// * `in_features` - 입력 텐서의 특성(피처) 수
    /// * `out_features` - 출력 텐서의 특성(피처) 수
    /// * `label` - 시각화 및 디버깅을 위한 레이어의 이름
    pub fn new(in_features: usize, out_features: usize, label: &str) -> MlResult<Self> {
        // 가중치(W) 초기화: (in_features, out_features) 형태
        // Kaiming He 초기화와 유사하게 표준편차를 조절하여 가중치를 초기화합니다.
        let k = 1.0 / (in_features as f32).sqrt();
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| rand::random::<f32>() * 2.0 * k - k)
            .collect();
        let weight_tensor = Tensor::from_vec(weight_data, &[in_features, out_features])?;

        // 편향(b) 초기화: (out_features) 형태
        let bias_data = vec![0.0; out_features];
        let bias_tensor = Tensor::from_vec(bias_data, &[out_features])?;

        Ok(Self {
            label: label.to_string(),
            weight: var_weight!(weight_tensor),
            bias: var_bias!(bias_tensor),
        })
    }
}


impl Layer for Linear {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let mut matmul = Matmul::new()?;
        // y = xW + b
        let x = matmul.apply(&[input, &self.weight])?;
        Ok(&x + &self.bias)
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let matmul = Matmul::new()?;
        let add = Add::new()?;

        let weight_tensor = self.weight.tensor();
        let bias_tensor = self.bias.tensor();

        let x = matmul.forward(&[input, weight_tensor])?.remove(0);
        let output = add.forward(&[&x, bias_tensor])?.remove(0);

        Ok(output)
    }

    fn params(&self) -> Vec<&dyn Parameter> { self._params() }
    fn label(&self) -> &str { &self.label }
    fn save_state(&self) -> LayerState { self._save_state() }
    fn load_state(&mut self, state: &LayerState) -> MlResult<()> { self._load_state(state) }
}
