use super::*;

/// axis 기준 row-wise softmax를 계산합니다.
/// 음수 axis를 지원합니다 (axis=-1 → 마지막 축).
fn softmax_along_axis(data: &[f32], shape: &[usize], axis: usize) -> Vec<f32> {
    let ndim = shape.len();
    let axis_dim = shape[axis];
    let outer_size: usize = shape[..axis].iter().product();
    let inner_size: usize = shape[axis + 1..ndim].iter().product();

    let mut out = vec![0.0f32; data.len()];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // axis 축을 따른 stride 기반 row 추출
            let base = outer * axis_dim * inner_size + inner;

            // max (수치 안정성)
            let mut max_val = f32::NEG_INFINITY;
            for k in 0..axis_dim {
                let idx = base + k * inner_size;
                if data[idx] > max_val { max_val = data[idx]; }
            }

            // exp + sum
            let mut sum_exp = 0.0f32;
            for k in 0..axis_dim {
                let idx = base + k * inner_size;
                let e = (data[idx] - max_val).exp();
                out[idx] = e;
                sum_exp += e;
            }

            // normalize
            for k in 0..axis_dim {
                let idx = base + k * inner_size;
                out[idx] /= sum_exp;
            }
        }
    }
    out
}

/// axis 기준 softmax backward를 계산합니다.
/// s = softmax output, g = upstream gradient
/// ∂L/∂x_i = s_i * (g_i - dot(s, g))   (row 단위)
#[cfg(feature = "enableBackward")]
fn softmax_backward_along_axis(s: &[f32], g: &[f32], shape: &[usize], axis: usize) -> Vec<f32> {
    let ndim = shape.len();
    let axis_dim = shape[axis];
    let outer_size: usize = shape[..axis].iter().product();
    let inner_size: usize = shape[axis + 1..ndim].iter().product();

    let mut dx = vec![0.0f32; s.len()];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let base = outer * axis_dim * inner_size + inner;

            // dot(s, g) for this row
            let mut dot = 0.0f32;
            for k in 0..axis_dim {
                let idx = base + k * inner_size;
                dot += s[idx] * g[idx];
            }

            // dx_i = s_i * (g_i - dot)
            for k in 0..axis_dim {
                let idx = base + k * inner_size;
                dx[idx] = s[idx] * (g[idx] - dot);
            }
        }
    }
    dx
}

impl Function for SoftmaxOp {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(SoftmaxOp)
    }

    /// Softmax forward.
    ///
    /// - `targets = [input]`         → 전역 softmax (하위 호환)
    /// - `targets = [input, axis]`   → axis 기준 row-wise softmax
    ///
    /// axis 스칼라는 음수를 허용합니다 (-1 = 마지막 축).
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let input = targets[0];
        let input_data = input.data();
        let shape = input.shape();

        if targets.len() >= 2 {
            // axis-aware softmax
            let raw_axis = targets[1].data()[0] as i32;
            let ndim = shape.len() as i32;
            let axis = if raw_axis < 0 { (ndim + raw_axis) as usize } else { raw_axis as usize };
            if axis >= shape.len() {
                return Err(MlError::StringError(format!(
                    "SoftmaxOp: axis {} 는 {}차원 텐서 범위를 벗어났습니다.", axis, shape.len()
                )));
            }
            let out = softmax_along_axis(input_data, shape, axis);
            Ok(vec![GlobalTensor::from_vec(out, shape)?])
        } else {
            // 전역 softmax (기존 동작)
            let max_val = input_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_values: Vec<f32> = input_data.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_of_exps: f32 = exp_values.iter().sum();
            let softmax_output: Vec<f32> = exp_values.iter().map(|&exp_val| exp_val / sum_of_exps).collect();
            Ok(vec![GlobalTensor::from_vec(softmax_output, shape)?])
        }
    }

    /// Softmax backward.
    ///
    /// targets 구조는 forward와 동일:
    /// - `targets = [softmax_output]`       → 전역 backward
    /// - `targets = [softmax_output, axis]` → axis 기준 row-wise backward
    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let softmax_output = targets[0];
        let s = softmax_output.data();
        let g = grad.data();
        let shape = softmax_output.shape();

        if targets.len() >= 2 {
            // axis-aware backward
            let raw_axis = targets[1].data()[0] as i32;
            let ndim = shape.len() as i32;
            let axis = if raw_axis < 0 { (ndim + raw_axis) as usize } else { raw_axis as usize };
            let dx = softmax_backward_along_axis(s, g, shape, axis);
            // axis 스칼라에 대한 gradient는 없음
            Ok(vec![
                GlobalTensor::from_vec(dx, shape)?,
                GlobalTensor::from_vec(vec![0.0], &[1, 1])?,
            ])
        } else {
            // 전역 backward (기존 동작)
            let dot_product: f32 = s.iter().zip(g.iter()).map(|(&sv, &gv)| sv * gv).sum();
            let input_grad: Vec<f32> = s.iter().zip(g.iter())
                .map(|(&sv, &gv)| sv * (gv - dot_product))
                .collect();
            Ok(vec![GlobalTensor::from_vec(input_grad, shape)?])
        }
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
}
