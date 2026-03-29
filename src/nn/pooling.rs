use super::*;
use crate::tensor::operators::{MaxPool2d, AvgPool2d};

impl Pooling {
    /// MaxPool2d 레이어를 생성.
    pub fn new_max(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        label: &str,
    ) -> Self {
        Self {
            label: label.to_string(),
            kernel_size,
            stride,
            mode: PoolingMode::Max,
        }
    }

    /// AvgPool2d 레이어를 생성.
    pub fn new_avg(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        label: &str,
    ) -> Self {
        Self {
            label: label.to_string(),
            kernel_size,
            stride,
            mode: PoolingMode::Average,
        }
    }

    fn kernel_stride_scalars(&self) -> MlResult<[GlobalTensor<f32>; 4]> {
        Ok([
            GlobalTensor::from_vec(vec![self.kernel_size.0 as f32], &[1, 1])?,
            GlobalTensor::from_vec(vec![self.kernel_size.1 as f32], &[1, 1])?,
            GlobalTensor::from_vec(vec![self.stride.0     as f32], &[1, 1])?,
            GlobalTensor::from_vec(vec![self.stride.1     as f32], &[1, 1])?,
        ])
    }
}

impl Layer for Pooling {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let kh = Variable::new(Tensor::from_vec(vec![self.kernel_size.0 as f32], &[1, 1])?);
        let kw = Variable::new(Tensor::from_vec(vec![self.kernel_size.1 as f32], &[1, 1])?);
        let sh = Variable::new(Tensor::from_vec(vec![self.stride.0      as f32], &[1, 1])?);
        let sw = Variable::new(Tensor::from_vec(vec![self.stride.1      as f32], &[1, 1])?);

        match self.mode {
            PoolingMode::Max => {
                // forward()[0]=Y, forward()[1]=mask
                // apply_with_saved: mask를 saved tensor로 보존 →
                // backward targets = [input, kH, kW, sh, sw, mask]
                let mut op = MaxPool2d::new()?;
                op.apply_with_saved(&[input, &kh, &kw, &sh, &sw])
            }
            PoolingMode::Average => {
                let mut op = AvgPool2d::new()?;
                op.apply(&[input, &kh, &kw, &sh, &sw])
            }
        }
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let [kh, kw, sh, sw] = self.kernel_stride_scalars()?;
        match self.mode {
            PoolingMode::Max => {
                let op = MaxPool2d::new()?;
                // forward returns [Y, mask]; 첫 번째가 출력
                let mut result = op.forward(&[input, &kh, &kw, &sh, &sw])?;
                result.remove(1); // mask 제거
                Ok(result.remove(0))
            }
            PoolingMode::Average => {
                let op = AvgPool2d::new()?;
                Ok(op.forward(&[input, &kh, &kw, &sh, &sw])?.remove(0))
            }
        }
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        vec![] // Pooling은 학습 파라미터 없음
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn save_state(&self) -> LayerState {
        LayerState {
            layer_type: "Pooling".to_string(),
            label: self.label.clone(),
            config: serde_json::json!({
                "mode":     if self.mode == PoolingMode::Max { "max" } else { "avg" },
                "kernel_h": self.kernel_size.0, "kernel_w": self.kernel_size.1,
                "stride_h": self.stride.0,      "stride_w": self.stride.1,
            }),
            params: vec![],
        }
    }
    // load_state: 파라미터 없으므로 기본 구현(no-op) 사용
}
