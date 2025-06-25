use super::*;

impl Linear {
    /// 새로운 Linear 레이어를 생성합니다.
    ///
    /// # Arguments
    /// * `in_features` - 입력 텐서의 특성(피처) 수
    /// * `out_features` - 출력 텐서의 특성(피처) 수
    /// * `label` - 시각화 및 디버깅을 위한 레이어의 이름
    pub fn new(in_features: usize, out_features: usize, label: &str) -> MlResult<Self> {
        let k = 1.0 / (in_features as f32).sqrt();
        let weight_data: Vec<f32> = (0..out_features * in_features)
            .map(|_| rand::random::<f32>() * 2.0 * k - k)
            .collect();
        let weight_tensor = Tensor::from_vec(weight_data, &[out_features, in_features])?;

        // 편향(b) 초기화: (out_features) 형태
        let bias_data = vec![0.0; out_features];
        let bias_tensor = Tensor::from_vec(bias_data, &[out_features, 1])?;

        Ok(Self {
            label: label.to_string(),
            weight: var_weight!(weight_tensor),
            bias: var_bias!(bias_tensor),
            cache: HashMap::new(),
            matmul: Matmul::new()?,
            add: Add::new()?,
        })
    }
}


impl Layer for Linear {
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let x = self.matmul.forward(&[self.weight.tensor(), input.tensor()])?.remove(0);
        let output = self.add.forward(&[&x, self.bias.tensor()])?.remove(0);

        x.with_grad_fn(self.type_name(), &[&input]); // 입출력에 대한 계산그래프 구성을 재설계 해야함
        output.with_grad_fn(self.type_name(), &[&x, self.bias.tensor()]);
        Ok(applied)
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let x = self.matmul.forward(&[self.weight.tensor(), input])?.remove(0);
        let output = self.add.forward(&[&x, self.bias.tensor()])?.remove(0);

        Ok(output)
    }

    /// 이 레이어가 소유한 모든 파라미터(가중치, 편향)의 참조를 반환합니다.
    fn params(&self) -> Vec<&dyn Parameter> { vec![&self.weight, &self.bias] }
    fn label(&self) -> &str { &self.label }
}