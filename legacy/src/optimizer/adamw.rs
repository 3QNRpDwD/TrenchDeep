use super::*;


/// AdamW (Adam with Decoupled Weight Decay).
///
/// Adam에서 weight decay를 gradient에 포함시키는 대신 가중치에 직접 적용한다.
/// 이 분리(decoupled)가 일반화 성능을 개선한다.
///
/// 업데이트 규칙:
/// ```text
/// m = β1 * m + (1 - β1) * grad
/// v = β2 * v + (1 - β2) * grad²
/// m̂ = m / (1 - β1^t)
/// v̂ = v / (1 - β2^t)
/// W = W - lr * (m̂ / (sqrt(v̂) + ε) + weight_decay * W)
/// ```
pub struct AdamW {
    lr:           f32,
    beta1:        f32,
    beta2:        f32,
    eps:          f32,
    weight_decay: f32,
    t:            u32,
    params:       Vec<crate::optimizer::adam::AdamState>,
}

impl AdamW {
    /// * `lr`           - 학습률 (일반적으로 1e-3)
    /// * `beta1`        - 1차 모멘트 감쇠 계수 (일반적으로 0.9)
    /// * `beta2`        - 2차 모멘트 감쇠 계수 (일반적으로 0.999)
    /// * `eps`          - 수치 안정성 소량값 (일반적으로 1e-8)
    /// * `weight_decay` - 가중치 감쇠 계수 (일반적으로 1e-2)
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self { lr, beta1, beta2, eps, weight_decay, t: 0, params: vec![] }
    }
}

impl Optimizer for AdamW {
    fn register(&mut self, param: &dyn Parameter) {
        let size = TENSOR_STORAGE.with_borrow(|s| {
            s.get(&param.node_id()).map(|w| w.data.len()).unwrap_or(0)
        });
        self.params.push(crate::optimizer::adam::AdamState::new(param.node_id(), param.grad().id(), size));
    }

    fn step(&mut self) -> MlResult<()> {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        let grad_ids: Vec<NodeId> = self.params.iter().map(|e| e.grad_id).collect();
        let grads = snapshot_grads(&grad_ids);

        let lr           = self.lr;
        let beta1        = self.beta1;
        let beta2        = self.beta2;
        let eps          = self.eps;
        let weight_decay = self.weight_decay;

        TENSOR_STORAGE.with_borrow_mut(|storage| {
            for (e, grad) in self.params.iter_mut().zip(grads.iter()) {
                if grad.is_empty() { continue; }

                // m = β1*m + (1-β1)*grad,  v = β2*v + (1-β2)*grad²
                for i in 0..e.m.len() {
                    e.m[i] = beta1 * e.m[i] + (1.0 - beta1) * grad[i];
                    e.v[i] = beta2 * e.v[i] + (1.0 - beta2) * grad[i] * grad[i];
                }

                // W = W - lr * (m̂ / (sqrt(v̂) + ε) + weight_decay * W)
                if let Some(w) = storage.get_mut(&e.weight_id) {
                    w.data.iter_mut().enumerate().for_each(|(i, p)| {
                        let m_hat = e.m[i] / bc1;
                        let v_hat = e.v[i] / bc2;
                        *p -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * *p);
                    });
                }
            }
        });
        Ok(())
    }

    fn zero_grad(&self) -> MlResult<()> {
        let grad_ids: Vec<NodeId> = self.params.iter().map(|e| e.grad_id).collect();
        clear_grads(&grad_ids);
        Ok(())
    }

    fn lr(&self) -> f32 { self.lr }
    fn set_lr(&mut self, lr: f32) { self.lr = lr; }
    fn registered_param_count(&self) -> usize { self.params.len() }
}
