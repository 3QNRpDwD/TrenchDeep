use super::*;
use crate::tensor::operators::GroupNorm;

impl GroupNormLayer {
    /// 새로운 GroupNorm 레이어를 생성합니다.
    ///
    /// # Arguments
    /// * `num_groups`   - 채널을 나눌 그룹 수. `num_channels % num_groups == 0` 필요
    /// * `num_channels` - 입력 채널 수 C
    /// * `eps`          - 수치 안정성을 위한 소수 (기본값: 1e-5)
    /// * `label`        - 레이어 레이블
    pub fn new(
        num_groups: usize,
        num_channels: usize,
        eps: f32,
        label: &str,
    ) -> MlResult<Self> {
        if num_channels % num_groups != 0 {
            return Err(MlError::StringError(format!(
                "GroupNormLayer: num_channels({}) % num_groups({}) != 0",
                num_channels, num_groups
            )));
        }

        // γ = 1, β = 0 초기화
        let gamma_tensor = Tensor::from_vec(vec![1.0f32; num_channels], &[num_channels])?;
        let beta_tensor  = Tensor::from_vec(vec![0.0f32; num_channels], &[num_channels])?;

        Ok(Self {
            label: label.to_string(),
            gamma: var_weight!(gamma_tensor),
            beta:  var_bias!(beta_tensor),
            num_groups,
            num_channels,
            eps,
        })
    }

    fn param_scalars(&self) -> MlResult<[GlobalTensor<f32>; 2]> {
        Ok([
            GlobalTensor::from_vec(vec![self.num_groups as f32], &[1, 1])?,
            GlobalTensor::from_vec(vec![self.eps],               &[1, 1])?,
        ])
    }
}

impl Layer for GroupNormLayer {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let g_var   = Variable::new(Tensor::from_vec(vec![self.num_groups as f32], &[1, 1])?);
        let eps_var = Variable::new(Tensor::from_vec(vec![self.eps],               &[1, 1])?);

        // apply_with_saved: forward()[1..] (x_hat, mean, var) 를 saved tensors로
        // 자동 보존하고 with_grad_fn inputs 뒤에 이어붙임.
        // → backward targets = [X, γ, β, g, eps, x_hat, mean, var]
        let mut op = GroupNorm::new()?;
        op.apply_with_saved(&[input, &self.gamma, &self.beta, &g_var, &eps_var])
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let [g, e] = self.param_scalars()?;
        let op = GroupNorm::new()?;
        // forward()[0] = Y
        Ok(op.forward(&[
            input,
            self.gamma.tensor(),
            self.beta.tensor(),
            &g, &e,
        ])?.remove(0))
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        vec![&self.gamma, &self.beta]
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn save_state(&self) -> LayerState {
        let g = self.gamma.tensor();
        let b = self.beta.tensor();
        LayerState {
            layer_type: "GroupNormLayer".to_string(),
            label: self.label.clone(),
            config: serde_json::json!({
                "num_groups":   self.num_groups,
                "num_channels": self.num_channels,
                "eps":          self.eps,
            }),
            params: vec![
                ParamState { name: "gamma".to_string(), shape: g.shape().to_vec(), data: g.data().to_vec(), blob_offset: None, blob_length: None },
                ParamState { name: "beta".to_string(),  shape: b.shape().to_vec(), data: b.data().to_vec(), blob_offset: None, blob_length: None },
            ],
        }
    }

    fn load_state(&mut self, state: &LayerState) -> MlResult<()> {
        if state.layer_type != "GroupNormLayer" {
            return Err(MlError::StringError(format!(
                "레이어 타입 불일치: 파일='{}', 현재='GroupNormLayer'", state.layer_type
            )));
        }
        let g = crate::nn::checkpoint::find_param(&state.params, "gamma")?;
        crate::nn::checkpoint::validate_shape(g, self.gamma.tensor().shape())?;
        self.gamma.tensor().replace(GlobalTensor::from_vec(g.data.clone(), &g.shape)?);

        let b = crate::nn::checkpoint::find_param(&state.params, "beta")?;
        crate::nn::checkpoint::validate_shape(b, self.beta.tensor().shape())?;
        self.beta.tensor().replace(GlobalTensor::from_vec(b.data.clone(), &b.shape)?);

        Ok(())
    }
}
