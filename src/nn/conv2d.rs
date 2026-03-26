use super::*;
use crate::tensor::operators::Conv2d;

impl Conv2DLayer {
    /// 새로운 Conv 레이어를 생성합니다.
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

impl Layer for Conv2DLayer {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        // 스칼라 하이퍼파라미터를 Variable로 래핑 (그래프에 leaf로 등록되지만 grad 불필요)
        let sh = Variable::new(Tensor::from_vec(vec![self.stride.0  as f32], &[1, 1])?);
        let sw = Variable::new(Tensor::from_vec(vec![self.stride.1  as f32], &[1, 1])?);
        let ph = Variable::new(Tensor::from_vec(vec![self.padding.0 as f32], &[1, 1])?);
        let pw = Variable::new(Tensor::from_vec(vec![self.padding.1 as f32], &[1, 1])?);

        let mut op = Conv2d::new()?;
        op.apply(&[input, &self.weight, &self.bias, &sh, &sw, &ph, &pw])
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let [sh, sw, ph, pw] = self.stride_padding_scalars()?;
        let op = Conv2d::new()?;
        let mut result = op.forward(&[
            input,
            self.weight.tensor(),
            self.bias.tensor(),
            &sh, &sw, &ph, &pw,
        ])?;
        Ok(result.remove(0))
    }

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
            layer_type: "Conv2DLayer".to_string(),
            label: self.label.clone(),
            config: serde_json::json!({
                "in_channels":  self.in_channels,
                "out_channels": self.out_channels,
                "kernel_h": self.kernel_size.0, "kernel_w": self.kernel_size.1,
                "stride_h": self.stride.0,      "stride_w": self.stride.1,
                "pad_h":    self.padding.0,     "pad_w":    self.padding.1,
            }),
            params: vec![
                ParamState { name: "weight".to_string(), shape: w.shape().to_vec(), data: w.data().to_vec(), blob_offset: None, blob_length: None },
                ParamState { name: "bias".to_string(),   shape: b.shape().to_vec(), data: b.data().to_vec(), blob_offset: None, blob_length: None },
            ],
        }
    }

    fn load_state(&mut self, state: &LayerState) -> MlResult<()> {
        if state.layer_type != "Conv2DLayer" {
            return Err(MlError::StringError(format!(
                "레이어 타입 불일치: 파일='{}', 현재='Conv2DLayer'", state.layer_type
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
