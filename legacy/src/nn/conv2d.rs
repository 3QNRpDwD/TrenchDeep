use super::*;
use crate::tensor::operators::Conv2dOp;

layer_params!(Conv2D, "Conv2D", [weight, bias], |s| serde_json::json!({
    "in_channels":  s.in_channels,
    "out_channels": s.out_channels,
    "kernel_h": s.kernel_size.0, "kernel_w": s.kernel_size.1,
    "stride_h": s.stride.0,      "stride_w": s.stride.1,
    "pad_h":    s.padding.0,     "pad_w":    s.padding.1,
}));

impl Conv2D {
    /// 새로운 Conv 레이어를 생성.
    ///
    /// # Arguments
    /// * `in_channels`  - 입력 채널 수
    /// * `out_channels` - 출력 채널 수 (필터 수)
    /// * `kernel_size`  - 커널 크기 (kH, kW)
    /// * `stride`       - 스트라이드 (sH, sW)
    /// * `padding`      - 패딩 크기 (pH, pW)
    /// * `label`        - 레이어 레이블
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        label: &str,
    ) -> MlResult<Self> {
        // Kaiming He 초기화 (fan_in = C_in * kH * kW)
        let fan_in = in_channels * kernel_size.0 * kernel_size.1;
        let k = 1.0 / (fan_in as f32).sqrt();
        let w_size = out_channels * in_channels * kernel_size.0 * kernel_size.1;
        let weight_data: Vec<f32> = (0..w_size)
            .map(|_| rand::random::<f32>() * 2.0 * k - k)
            .collect();
        let weight_tensor = Tensor::from_vec(
            weight_data,
            &[out_channels, in_channels, kernel_size.0, kernel_size.1],
        )?;
        let bias_tensor = Tensor::from_vec(vec![0.0f32; out_channels], &[out_channels])?;

        Ok(Self {
            label: label.to_string(),
            weight: var_weight!(weight_tensor),
            bias: var_bias!(bias_tensor),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        })
    }

    fn stride_padding_scalars(&self) -> MlResult<[GlobalTensor<f32>; 4]> {
        Ok([
            GlobalTensor::from_vec(vec![self.stride.0  as f32], &[1, 1])?,
            GlobalTensor::from_vec(vec![self.stride.1  as f32], &[1, 1])?,
            GlobalTensor::from_vec(vec![self.padding.0 as f32], &[1, 1])?,
            GlobalTensor::from_vec(vec![self.padding.1 as f32], &[1, 1])?,
        ])
    }
}

impl Layer for Conv2D {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let sh = Variable::new(Tensor::from_vec(vec![self.stride.0  as f32], &[1, 1])?);
        let sw = Variable::new(Tensor::from_vec(vec![self.stride.1  as f32], &[1, 1])?);
        let ph = Variable::new(Tensor::from_vec(vec![self.padding.0 as f32], &[1, 1])?);
        let pw = Variable::new(Tensor::from_vec(vec![self.padding.1 as f32], &[1, 1])?);

        let mut op = Conv2dOp::new()?;
        op.apply(&[input, &self.weight, &self.bias, &sh, &sw, &ph, &pw])
    }

    fn predict(&self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let [sh, sw, ph, pw] = self.stride_padding_scalars()?;
        let op = Conv2dOp::new()?;
        let mut result = op.forward(&[
            input,
            self.weight.tensor(),
            self.bias.tensor(),
            &sh, &sw, &ph, &pw,
        ])?;
        Ok(result.remove(0))
    }

    fn params(&self) -> Vec<&dyn Parameter> { self._params() }
    fn label(&self) -> &str { &self.label }
    fn save_state(&self) -> LayerState { self._save_state() }
    fn load_state(&mut self, state: &LayerState) -> MlResult<()> { self._load_state(state) }
}
