use super::*;

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
        // 이는 학습 초기 단계에서 그래디언트가 소실되거나 폭발하는 것을 방지하는 데 도움이 됩니다.
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

    /// 이 레이어가 소유한 모든 파라미터(가중치, 편향)의 참조를 반환합니다.
    fn params(&self) -> Vec<&dyn Parameter> {
        vec![&self.weight, &self.bias]
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn save_state(&self) -> LayerState {
        let w = self.weight.tensor();
        let b = self.bias.tensor();
        LayerState {
            layer_type: "Linear".to_string(),
            label: self.label.clone(),
            config: serde_json::json!({
                "in_features":  w.shape()[0],
                "out_features": w.shape()[1],
            }),
            params: vec![
                ParamState { name: "weight".to_string(), shape: w.shape().to_vec(), data: w.data().to_vec(), blob_offset: None, blob_length: None },
                ParamState { name: "bias".to_string(),   shape: b.shape().to_vec(), data: b.data().to_vec(), blob_offset: None, blob_length: None },
            ],
        }
    }

    fn load_state(&mut self, state: &LayerState) -> MlResult<()> {
        if state.layer_type != "Linear" {
            return Err(MlError::StringError(format!(
                "레이어 타입 불일치: 파일='{}', 현재='Linear'", state.layer_type
            )));
        }
        let w = crate::nn::checkpoint::find_param(&state.params, "weight")?;
        crate::nn::checkpoint::validate_shape(w, self.weight.tensor().shape())?;
        self.weight.tensor().replace(GlobalTensor::from_vec(w.data.clone(), &w.shape)?);

        let b = crate::nn::checkpoint::find_param(&state.params, "bias")?;
        crate::nn::checkpoint::validate_shape(b, self.bias.tensor().shape())?;
        self.bias.tensor().replace(GlobalTensor::from_vec(b.data.clone(), &b.shape)?);

        Ok(())
    }
}