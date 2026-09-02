use super::*;

impl Function for MeanSquaredError {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(MeanSquaredError)
    }
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let pred = targets[0];
        let target = targets[1];

        if pred.shape() != target.shape() {
            return Err(LossError::InvalidShape {
                expected: pred.shape().to_vec(),
                got: target.shape().to_vec(),
            }.into());
        }

        let n = pred.data().len() as f32;
        let diff = pred.data().iter().zip(target.data().iter()).map(|(&p, &t)| p - t);
        let squared_error = diff.map(|d| d * d).sum::<f32>();

        Ok(vec![scalar!(squared_error / n)])
    }

    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let pred = targets[0];
        let target = targets[1];
        let n = pred.data().len() as f32;
        let grad_val = grad.data()[0];

        let grad_pred_data: Vec<f32> = pred.data()
            .iter()
            .zip(target.data().iter())
            .map(|(&p, &t)| grad_val * 2.0 * (p - t) / n)
            .collect();

        let grad_target_data: Vec<f32> = grad_pred_data.iter().map(|&g| -g).collect();

        let grad_pred = GlobalTensor::from_vec(grad_pred_data, pred.shape())?;
        let grad_target = GlobalTensor::from_vec(grad_target_data, target.shape())?;

        Ok(vec![grad_pred, grad_target])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &NodeId {
        &self.node_id
    }
}


impl Function for MeanAbsoluteError {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(MeanAbsoluteError)
    }
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let pred = targets[0];
        let target = targets[1];

        if pred.shape() != target.shape() {
            return Err(LossError::InvalidShape {
                expected: pred.shape().to_vec(),
                got: target.shape().to_vec(),
            }.into());
        }

        let n = pred.data().len() as f32;
        let abs_error = pred.data().iter().zip(target.data().iter()).map(|(&p, &t)| (p - t).abs()).sum::<f32>();

        Ok(vec![scalar!(abs_error / n)])
    }

    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let pred = targets[0];
        let target = targets[1];
        let n = pred.data().len() as f32;
        let grad_val = grad.data()[0];

        let grad_pred_data: Vec<f32> = pred.data()
            .iter()
            .zip(target.data().iter())
            .map(|(&p, &t)| grad_val * (p - t).signum() / n)
            .collect();

        let grad_target_data: Vec<f32> = grad_pred_data.iter().map(|&g| -g).collect();

        let grad_pred = GlobalTensor::from_vec(grad_pred_data, pred.shape())?;
        let grad_target = GlobalTensor::from_vec(grad_target_data, target.shape())?;

        Ok(vec![grad_pred, grad_target])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &NodeId {
        &self.node_id
    }
}


impl Function for HuberLoss {
    fn new() -> MlResult<GlobalFunction> {
        OPERATOR_STORAGE.with(|ops| {
            let my = "HuberLoss";
            let mut ops = ops.borrow_mut();
            match ops.contains_key(my) {
                true => Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id())),
                false => {
                    ops.insert(
                        String::from(my),
                        Box::new(HuberLoss { backend: Arc::new(CpuBackend::new()?), node_id: NODE_ID_GEN.next(), delta: 1.0})
                    );
                    Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id()))
                }
            }
        })
    }

    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let pred = targets[0];
        let target = targets[1];

        if pred.shape() != target.shape() {
            return Err(LossError::InvalidShape {
                expected: pred.shape().to_vec(),
                got: target.shape().to_vec(),
            }.into());
        }

        let n = pred.data().len() as f32;
        let delta = self.delta;

        let huber_error = pred.data().iter().zip(target.data().iter()).map(|(&p, &t)| {
            let diff = (p - t).abs();
            if diff <= delta {
                0.5 * diff.powi(2)
            } else {
                delta * (diff - 0.5 * delta)
            }
        }).sum::<f32>();

        Ok(vec![scalar!(huber_error / n)])
    }

    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let pred = targets[0];
        let target = targets[1];
        let n = pred.data().len() as f32;
        let grad_val = grad.data()[0];
        let delta = self.delta;

        let grad_pred_data: Vec<f32> = pred.data().iter().zip(target.data().iter()).map(|(&p, &t)| {
            let diff = p - t;
            let grad_elem = if diff.abs() <= delta {
                diff
            } else {
                delta * diff.signum()
            };
            grad_val * grad_elem / n
        }).collect();

        let grad_target_data: Vec<f32> = grad_pred_data.iter().map(|&g| -g).collect();

        let grad_pred = GlobalTensor::from_vec(grad_pred_data, pred.shape())?;
        let grad_target = GlobalTensor::from_vec(grad_target_data, target.shape())?;

        Ok(vec![grad_pred, grad_target])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &NodeId {
        &self.node_id
    }
}

impl Function for BinaryCrossEntropyLoss {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(BinaryCrossEntropyLoss)
    }

    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let pred = targets[0];
        let target = targets[1];

        if pred.shape() != target.shape() {
            return Err(LossError::InvalidShape {
                expected: pred.shape().to_vec(),
                got: target.shape().to_vec(),
            }.into());
        }

        let n = pred.data().len() as f32;
        let bce_loss = pred.data().iter().zip(target.data().iter()).map(|(&p, &t)| {
            let p_clipped = p.max(EPSILON).min(1.0 - EPSILON);
            - (t * p_clipped.ln() + (1.0 - t) * (1.0 - p_clipped).ln())
        }).sum::<f32>();

        Ok(vec![scalar!(bce_loss / n)])
    }

    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let pred = targets[0];
        let target = targets[1];
        let n = pred.data().len() as f32;
        let grad_val = grad.data()[0];

        let grad_pred_data: Vec<f32> = pred.data().iter().zip(target.data().iter()).map(|(&p, &t)| {
            let p_clipped = p.max(EPSILON).min(1.0 - EPSILON);
            grad_val * ((p_clipped - t) / (p_clipped * (1.0 - p_clipped))) / n
        }).collect();

        let grad_target_data: Vec<f32> = pred.data().iter().zip(target.data().iter()).map(|(&p, &t)| {
            let p_clipped = p.max(EPSILON).min(1.0 - EPSILON);
            grad_val * -((1.0 - p_clipped).ln() - p_clipped.ln()) / n
        }).collect();

        let grad_pred = GlobalTensor::from_vec(grad_pred_data, pred.shape())?;
        let grad_target = GlobalTensor::from_vec(grad_target_data, target.shape())?;

        Ok(vec![grad_pred, grad_target])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &NodeId {
        &self.node_id
    }
}

// ===== CategoricalCrossEntropy =====

impl CrossEntropyLoss {
    /// 텐서의 shape을 기반으로 배치 크기를 계산합니다.
    /// shape이 [batch_size, num_classes]인 경우 batch_size를,
    /// [num_classes]인 경우 1을 반환합니다.
    fn get_batch_size(shape: &[usize]) -> f32 {
        if shape.len() > 1 {
            shape[0] as f32
        } else {
            1.0
        }
    }
}

impl SoftmaxCrossEntropyLoss {
    fn get_batch_size(shape: &[usize]) -> f32 {
        if shape.len() > 1 {
            shape[0] as f32
        } else {
            1.0
        }
    }
}

impl Function for CrossEntropyLoss {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(CrossEntropyLoss)
    }
    
    fn forward(&self, inputs: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        // 1. 입력 유효성 검사 강화
        let (pred, target) = match inputs {
            [p, t] => (*p, *t),
            _ => return Err(MlError::TensorError(InvalidInputCount {
                expected: 2,
                got: inputs.len(),
            }.into()))
        };

        if pred.shape() != target.shape() {
            return Err(LossError::InvalidShape {
                expected: pred.shape().to_vec(),
                got: target.shape().to_vec(),
            }.into());
        }

        // 2. 배치 크기 계산 로직 재사용
        let batch_size = Self::get_batch_size(pred.shape());
        if batch_size == 0.0 {
            return Ok(vec![scalar!(0.0)]);
        }

        // 3. 손실 계산
        let cce_loss: f32 = pred.data().iter()
            .zip(target.data().iter())
            .map(|(&p, &t)| {
                // p가 0에 가까워지는 것을 방지하여 log(0)으로 인한 NaN 방지
                let p_clipped = p.max(EPSILON);
                -t * p_clipped.ln()
            })
            .sum();

        Ok(vec![scalar!(cce_loss / batch_size)])
    }

    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, inputs: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        // 1. 입력 유효성 검사 강화
        let (pred, target) = match inputs {
            [p, t] => (*p, *t),
            _ => return Err(MlError::TensorError(InvalidInputCount {
                expected: 2,
                got: inputs.len(),
            }.into()))
        };
        
        let grad_val = grad.data().get(0).copied().unwrap_or(1.0);

        // 2. 배치 크기 계산 로직 재사용
        let batch_size = Self::get_batch_size(pred.shape());
        if batch_size == 0.0 {
            let zero_grad = GlobalTensor::from_vec(vec![0.0; pred.data().len()], pred.shape())?;
            return Ok(vec![zero_grad.clone(), zero_grad]);
        }

        // 3. Gradient 계산
        // ∂L/∂p = (∂L/∂out) * (∂out/∂p)
        // 여기서 out = cce_loss, ∂L/∂out = grad_val
        // ∂(cce_loss)/∂p = -t/p
        let grad_pred_data: Vec<f32> = pred.data().iter()
            .zip(target.data().iter())
            .map(|(&p, &t)| {
                let p_clipped = p.max(EPSILON);
                grad_val * (-t / p_clipped) / batch_size
            })
            .collect();
        
        let grad_target_data: Vec<f32> = pred.data().iter()
            .map(|&p| {
                let p_clipped = p.max(EPSILON);
                grad_val * -p_clipped.ln() / batch_size
            })
            .collect();

        let grad_pred = GlobalTensor::from_vec(grad_pred_data, pred.shape())?;
        let grad_target = GlobalTensor::from_vec(grad_target_data, target.shape())?;

        Ok(vec![grad_pred, grad_target])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &NodeId {
        &self.node_id
    }
}

// 기존에 소프트맥스 함수와 크로스엔트로피 로스를 따로따로 사용했을때 기울기가 폭발하는 현상이 매우 빈번하여, 각각 따로 계산되던 함수를 하나로 융합함.
impl Function for SoftmaxCrossEntropyLoss {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(SoftmaxCrossEntropyLoss)
    }

    /// 순전파를 계산합니다.
    ///
    /// # Arguments
    /// * `inputs`: `[&Tensor(logits), &Tensor(target)]` 형태의 슬라이스.
    ///   - `logits`: 모델의 마지막 선형 계층에서 나온 원시 점수 (Softmax 적용 전).
    ///   - `target`: 실제 값 (원-핫 인코딩된 벡터).
    fn forward(&self, inputs: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let (logits, target) = match inputs {
            [l, t] => (*l, *t),
            _ => return Err(MlError::TensorError(InvalidInputCount { expected: 2, got: inputs.len() }.into())),
        };

        let logits_data = logits.data();
        let target_data = target.data();
        let shape       = logits.shape();
        let batch_size  = Self::get_batch_size(shape);
        // `num_classes` 는 마지막 축 크기. `[B, V]` 는 V, `[V]` 는 V, `[B, L, V]` 도 V.
        let num_classes = *shape.last().unwrap_or(&logits_data.len());
        let n_rows      = if num_classes == 0 { 0 } else { logits_data.len() / num_classes };

        // 행(=샘플) 단위로 log-sum-exp + dot(z, t) 를 누적한다.
        // 전체 텐서를 하나의 분포로 취급하던 이전 구현은 `[B, V]` 에서 잘못된 값을 냈다.
        let mut loss_sum = 0.0f32;
        for row in 0..n_rows {
            let lo = row * num_classes;
            let hi = lo + num_classes;
            let row_logits = &logits_data[lo..hi];
            let row_target = &target_data[lo..hi];

            let max_logit = row_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum_exp   = row_logits.iter().map(|&z| (z - max_logit).exp()).sum::<f32>();
            let log_sum_exp = sum_exp.ln();
            let dot_product = row_logits.iter().zip(row_target.iter()).map(|(&z, &t)| z * t).sum::<f32>();

            loss_sum += (max_logit + log_sum_exp) - dot_product;
        }

        Ok(vec![scalar!(loss_sum / batch_size)])
    }

    #[cfg(all(feature = "enableBackward"))]
    /// 역전파를 계산합니다.
    /// 로짓에 대한 그래디언트는 (p - t) 형태로 매우 안정적입니다.
    fn backward(&self, inputs: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let (logits, target) = match inputs {
            [l, t] => (*l, *t),
            _ => return Err(MlError::TensorError(InvalidInputCount { expected: 2, got: inputs.len() }.into())),
        };

        let grad_val    = grad.data().get(0).copied().unwrap_or(1.0);
        let shape       = logits.shape();
        let batch_size  = Self::get_batch_size(shape);
        let logits_data = logits.data();
        let target_data = target.data();
        let num_classes = *shape.last().unwrap_or(&logits_data.len());
        let n_rows      = if num_classes == 0 { 0 } else { logits_data.len() / num_classes };

        // 행 단위로 softmax(p) 계산 후 (p - t) 를 스케일해 기록한다.
        let mut grad_logits_data = vec![0.0f32; logits_data.len()];
        for row in 0..n_rows {
            let lo = row * num_classes;
            let hi = lo + num_classes;
            let row_logits = &logits_data[lo..hi];
            let row_target = &target_data[lo..hi];
            let out        = &mut grad_logits_data[lo..hi];

            let max_logit  = row_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_values: Vec<f32> = row_logits.iter().map(|&z| (z - max_logit).exp()).collect();
            let sum_exp    = exp_values.iter().sum::<f32>();

            for i in 0..num_classes {
                let p = exp_values[i] / sum_exp;
                out[i] = grad_val * (p - row_target[i]) / batch_size;
            }
        }

        let grad_logits = GlobalTensor::from_vec(grad_logits_data, shape)?;
        let grad_target = GlobalTensor::zeros(target.shape());

        Ok(vec![grad_logits, grad_target])
    }


    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &NodeId {
        &self.node_id
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 테스트: SoftmaxCrossEntropyLoss — per-row log-sum-exp 회귀 테스트
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod softmax_ce_tests {
    use super::*;
    use crate::tensor::{operators::Function, Tensor};

    /// [1, V] 단일 행: 균등 logit → loss = log V.
    #[test]
    fn sce_single_row_uniform_logits() -> MlResult<()> {
        let sce = SoftmaxCrossEntropyLoss::new()?;
        let logits = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[1, 4])?;
        let target = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0], &[1, 4])?;

        let out = sce.forward(&[&logits, &target])?.remove(0);
        let loss = out.data()[0];
        let expected = (4.0f32).ln();
        assert!((loss - expected).abs() < 1e-5, "expected {expected}, got {loss}");
        Ok(())
    }

    /// [B>1, V]: 모든 행이 균등 logit 이면 평균 loss = log V.
    /// 이전 구현은 flat log-sum-exp 로 잘못된 값을 반환했다.
    #[test]
    fn sce_multi_row_uniform_logits_per_row() -> MlResult<()> {
        let sce = SoftmaxCrossEntropyLoss::new()?;
        let logits = Tensor::from_vec(vec![0.0; 12], &[3, 4])?;
        let mut target_data = vec![0.0; 12];
        target_data[0]  = 1.0;
        target_data[5]  = 1.0;
        target_data[10] = 1.0;
        let target = Tensor::from_vec(target_data, &[3, 4])?;

        let out = sce.forward(&[&logits, &target])?.remove(0);
        let loss = out.data()[0];
        let expected = (4.0f32).ln();
        assert!((loss - expected).abs() < 1e-5, "expected {expected}, got {loss}");
        Ok(())
    }

    /// 확신적 예측 → loss ≥ 0 이며 매우 작아야 함.
    #[test]
    fn sce_multi_row_nonnegative_and_small_when_confident() -> MlResult<()> {
        let sce = SoftmaxCrossEntropyLoss::new()?;
        let logits_data = vec![
            10.0, -10.0, -10.0, -10.0,
            -10.0, 10.0, -10.0, -10.0,
            -10.0, -10.0, 10.0, -10.0,
        ];
        let mut target_data = vec![0.0; 12];
        target_data[0]  = 1.0;
        target_data[5]  = 1.0;
        target_data[10] = 1.0;

        let logits = Tensor::from_vec(logits_data, &[3, 4])?;
        let target = Tensor::from_vec(target_data, &[3, 4])?;
        let out = sce.forward(&[&logits, &target])?.remove(0);
        let loss = out.data()[0];
        assert!(loss >= 0.0, "loss must be non-negative, got {loss}");
        assert!(loss < 1e-3, "confident predictions should yield near-zero loss, got {loss}");
        Ok(())
    }

    /// Backward: 행별 (p - t) / batch_size 가 배치 전반에 걸쳐 정확히 계산되어야 한다.
    #[cfg(feature = "enableBackward")]
    #[test]
    fn sce_multi_row_backward_per_row() -> MlResult<()> {
        let sce = SoftmaxCrossEntropyLoss::new()?;
        let logits = Tensor::from_vec(vec![0.0; 8], &[2, 4])?;
        let mut target_data = vec![0.0; 8];
        target_data[0] = 1.0;
        target_data[7] = 1.0;
        let target = Tensor::from_vec(target_data, &[2, 4])?;

        let grad = Tensor::from_vec(vec![1.0], &[1])?;
        let out  = sce.backward(&[&logits, &target], &grad)?;
        let grad_logits = &out[0];
        let data = grad_logits.data();

        // 균등 softmax = 1/4, batch_size = 2.
        // row 0 target class 0: [-0.375, 0.125, 0.125, 0.125]
        // row 1 target class 3: [0.125, 0.125, 0.125, -0.375]
        let expected = [
            -0.375, 0.125, 0.125, 0.125,
             0.125, 0.125, 0.125, -0.375,
        ];
        for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-5, "grad_logits[{i}] expected {exp}, got {got}");
        }
        Ok(())
    }
}