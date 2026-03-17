use super::*;

impl Function for MeanSquaredError {
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
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

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
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

        let grad_pred = PooledTensor::from_vec(grad_pred_data, pred.shape())?;
        let grad_target = PooledTensor::from_vec(grad_target_data, target.shape())?;

        Ok(vec![grad_pred, grad_target])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &HandleId {
        &self.node_id
    }
}


impl Function for MeanAbsoluteError {
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
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

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
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

        let grad_pred = PooledTensor::from_vec(grad_pred_data, pred.shape())?;
        let grad_target = PooledTensor::from_vec(grad_target_data, target.shape())?;

        Ok(vec![grad_pred, grad_target])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &HandleId {
        &self.node_id
    }
}


impl Function for HuberLoss {
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
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

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
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

        let grad_pred = PooledTensor::from_vec(grad_pred_data, pred.shape())?;
        let grad_target = PooledTensor::from_vec(grad_target_data, target.shape())?;

        Ok(vec![grad_pred, grad_target])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &HandleId {
        &self.node_id
    }
}

impl Function for BinaryCrossEntropyLoss {
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
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

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
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

        let grad_pred = PooledTensor::from_vec(grad_pred_data, pred.shape())?;
        let grad_target = PooledTensor::from_vec(grad_target_data, target.shape())?;

        Ok(vec![grad_pred, grad_target])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &HandleId {
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
    fn forward(&self, inputs: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
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

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, inputs: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
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
            let zero_grad = PooledTensor::from_vec(vec![0.0; pred.data().len()], pred.shape())?;
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

        let grad_pred = PooledTensor::from_vec(grad_pred_data, pred.shape())?;
        let grad_target = PooledTensor::from_vec(grad_target_data, target.shape())?;

        Ok(vec![grad_pred, grad_target])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &HandleId {
        &self.node_id
    }
}

// 기존에 소프트맥스 함수와 크로스엔트로피 로스를 따로따로 사용했을때 기울기가 폭발하는 현상이 매우 빈번하여, 각각 따로 계산되던 함수를 하나로 융합함.
impl Function for SoftmaxCrossEntropyLoss {
    /// 순전파를 계산합니다.
    ///
    /// # Arguments
    /// * `inputs`: `[&Tensor(logits), &Tensor(target)]` 형태의 슬라이스.
    ///   - `logits`: 모델의 마지막 선형 계층에서 나온 원시 점수 (Softmax 적용 전).
    ///   - `target`: 실제 값 (원-핫 인코딩된 벡터).
    fn forward(&self, inputs: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        let (logits, target) = match inputs {
            [l, t] => (*l, *t),
            _ => return Err(MlError::TensorError(InvalidInputCount { expected: 2, got: inputs.len() }.into())),
        };

        let logits_data = logits.data();
        let target_data = target.data();
        let batch_size = Self::get_batch_size(logits.shape());

        // Log-Sum-Exp 트릭을 사용한 안정적인 손실 계산
        // loss = log(sum(exp(z_i))) - z_k (여기서 k는 정답 클래스 인덱스)
        //      = log(sum(exp(z_i))) - dot(z, t)

        // 1. 오버플로우 방지를 위해 로짓의 최댓값을 뺍니다.
        let max_logit = logits_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // 2. log(sum(exp(z_i))) 계산
        let sum_exp = logits_data.iter().map(|&z| (z - max_logit).exp()).sum::<f32>();
        let log_sum_exp = sum_exp.ln();

        // 3. 실제 로짓과 타겟의 내적(dot product) 계산
        let dot_product = logits_data.iter().zip(target.data().iter()).map(|(&z, &t)| z * t).sum::<f32>();

        // 4. 최종 손실 계산 (뺐던 최댓값을 다시 더해줌)
        let loss = (max_logit + log_sum_exp) - dot_product;

        Ok(vec![scalar!(loss / batch_size)])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    /// 역전파를 계산합니다.
    /// 로짓에 대한 그래디언트는 (p - t) 형태로 매우 안정적입니다.
    fn backward(&self, inputs: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        let (logits, target) = match inputs {
            [l, t] => (*l, *t),
            _ => return Err(MlError::TensorError(InvalidInputCount { expected: 2, got: inputs.len() }.into())),
        };

        let grad_val = grad.data().get(0).copied().unwrap_or(1.0);
        let batch_size = Self::get_batch_size(logits.shape());

        // --- 그래디언트 계산을 위해 먼저 Softmax 확률(p)을 계산 ---
        let logits_data = logits.data();
        let max_logit = logits_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f32> = logits_data.iter().map(|&z| (z - max_logit).exp()).collect();
        let sum_exp = exp_values.iter().sum::<f32>();

        // p (확률) 계산
        let probabilities: Vec<f32> = exp_values.iter().map(|&exp_val| exp_val / sum_exp).collect();

        // --- 로짓에 대한 그래디언트 (p - t) 계산 ---
        let grad_logits_data: Vec<f32> = probabilities.iter()
            .zip(target.data().iter())
            .map(|(&p, &t)| grad_val * (p - t) / batch_size)
            .collect();

        let grad_logits = PooledTensor::from_vec(grad_logits_data, logits.shape())?;

        // target에 대한 그래디언트는 필요 없는 경우가 많지만, 완전성을 위해 계산 (보통 0으로 처리)
        let grad_target = PooledTensor::zeros(target.shape());

        Ok(vec![grad_logits, grad_target])
    }


    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &HandleId {
        &self.node_id
    }
}
