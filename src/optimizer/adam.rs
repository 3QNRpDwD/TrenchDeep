use crate::MlResult;
use crate::nn::Parameter;
use crate::tensor::{NodeId, TENSOR_STORAGE};
use super::{clear_grads, snapshot_grads, Optimizer};

pub(super) struct AdamState {
    pub(super) weight_id: NodeId,
    pub(super) grad_id:   NodeId,
    /// 1차 모멘트 (편향 미보정)
    pub(super) m: Vec<f32>,
    /// 2차 모멘트 (편향 미보정)
    pub(super) v: Vec<f32>,
}

impl AdamState {
    pub(super) fn new(weight_id: NodeId, grad_id: NodeId, size: usize) -> Self {
        Self {
            weight_id,
            grad_id,
            m: vec![0.0; size],
            v: vec![0.0; size],
        }
    }
}

/// Adam (Adaptive Moment Estimation).
///
/// 업데이트 규칙:
/// ```text
/// m = β1 * m + (1 - β1) * grad
/// v = β2 * v + (1 - β2) * grad²
/// m̂ = m / (1 - β1^t)
/// v̂ = v / (1 - β2^t)
/// W = W - lr * m̂ / (sqrt(v̂) + ε)
/// ```
pub struct Adam {
    lr:    f32,
    beta1: f32,
    beta2: f32,
    eps:   f32,
    /// 현재 스텝 수 (bias correction에 사용)
    t:     u32,
    params: Vec<AdamState>,
}

impl Adam {
    /// * `lr`    - 학습률 (일반적으로 1e-3)
    /// * `beta1` - 1차 모멘트 감쇠 계수 (일반적으로 0.9)
    /// * `beta2` - 2차 모멘트 감쇠 계수 (일반적으로 0.999)
    /// * `eps`   - 수치 안정성 소량값 (일반적으로 1e-8)
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        Self { lr, beta1, beta2, eps, t: 0, params: vec![] }
    }
}

impl Optimizer for Adam {
    fn register(&mut self, param: &dyn Parameter) {
        let size = TENSOR_STORAGE.with_borrow(|s| {
            s.get(&param.node_id()).map(|w| w.data.len()).unwrap_or(0)
        });
        self.params.push(AdamState::new(param.node_id(), param.grad().id(), size));
    }

    fn step(&mut self) -> MlResult<()> {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        let grad_ids: Vec<NodeId> = self.params.iter().map(|e| e.grad_id).collect();
        let grads = snapshot_grads(&grad_ids);

        let lr    = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps   = self.eps;

        TENSOR_STORAGE.with_borrow_mut(|storage| {
            for (e, grad) in self.params.iter_mut().zip(grads.iter()) {
                if grad.is_empty() { continue; }

                // m = β1*m + (1-β1)*grad,  v = β2*v + (1-β2)*grad²
                for i in 0..e.m.len() {
                    e.m[i] = beta1 * e.m[i] + (1.0 - beta1) * grad[i];
                    e.v[i] = beta2 * e.v[i] + (1.0 - beta2) * grad[i] * grad[i];
                }

                // W = W - lr * m̂ / (sqrt(v̂) + ε)
                if let Some(w) = storage.get_mut(&e.weight_id) {
                    w.data.iter_mut().enumerate().for_each(|(i, p)| {
                        let m_hat = e.m[i] / bc1;
                        let v_hat = e.v[i] / bc2;
                        *p -= lr * m_hat / (v_hat.sqrt() + eps);
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
}
