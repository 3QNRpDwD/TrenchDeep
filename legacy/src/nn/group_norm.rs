use super::*;
use crate::tensor::operators::GroupNormOp;

layer_params!(GroupNorm, "GroupNorm", [gamma, beta], |s| serde_json::json!({
    "num_groups":   s.num_groups,
    "num_channels": s.num_channels,
    "eps":          s.eps,
}));

impl GroupNorm {
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
                "GroupNorm: num_channels({}) % num_groups({}) != 0",
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

impl Layer for GroupNorm {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let g_var   = Variable::new(Tensor::from_vec(vec![self.num_groups as f32], &[1, 1])?);
        let eps_var = Variable::new(Tensor::from_vec(vec![self.eps],               &[1, 1])?);

        let mut op = GroupNormOp::new()?;
        op.apply_with_saved(&[input, &self.gamma, &self.beta, &g_var, &eps_var])
    }

    fn predict(&self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let [g, e] = self.param_scalars()?;
        let op = GroupNormOp::new()?;
        Ok(op.forward(&[
            input,
            self.gamma.tensor(),
            self.beta.tensor(),
            &g, &e,
        ])?.remove(0))
    }

    fn params(&self) -> Vec<&dyn Parameter> { self._params() }
    fn label(&self) -> &str { &self.label }
    fn save_state(&self) -> LayerState { self._save_state() }
    fn load_state(&mut self, state: &LayerState) -> MlResult<()> { self._load_state(state) }
}
