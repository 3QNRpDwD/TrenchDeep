use super::*;

struct MomentumState {
    weight_id: NodeId,
    grad_id:   NodeId,
    velocity:  Vec<f32>,
}

/// SGD with Momentum.
///
/// 업데이트 규칙:
/// ```text
/// v = momentum * v + grad
/// W = W - lr * v
/// ```
pub struct Momentum {
    lr:       f32,
    momentum: f32,
    params:   Vec<MomentumState>,
}

impl Momentum {
    /// * `lr`       - 학습률
    /// * `momentum` - 모멘텀 계수 (일반적으로 0.9)
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self { lr, momentum, params: vec![] }
    }
}

impl Optimizer for Momentum {
    fn register(&mut self, param: &dyn Parameter) {
        let size = TENSOR_STORAGE.with_borrow(|s| {
            s.get(&param.node_id()).map(|w| w.data.len()).unwrap_or(0)
        });
        self.params.push(MomentumState {
            weight_id: param.node_id(),
            grad_id:   param.grad().id(),
            velocity:  vec![0.0; size],
        });
    }

    fn step(&mut self) -> MlResult<()> {
        let grad_ids: Vec<NodeId> = self.params.iter().map(|e| e.grad_id).collect();
        let grads = snapshot_grads(&grad_ids);

        let momentum = self.momentum;
        let lr = self.lr;

        TENSOR_STORAGE.with_borrow_mut(|storage| {
            for (e, grad) in self.params.iter_mut().zip(grads.iter()) {
                if grad.is_empty() { continue; }

                // v = momentum * v + grad
                e.velocity.iter_mut()
                    .zip(grad.iter())
                    .for_each(|(v, &g)| *v = momentum * *v + g);

                // W = W - lr * v
                if let Some(w) = storage.get_mut(&e.weight_id) {
                    w.data.iter_mut()
                        .zip(e.velocity.iter())
                        .for_each(|(p, &v)| *p -= lr * v);
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
