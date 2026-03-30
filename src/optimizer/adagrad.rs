use super::*;


struct AdaGradState {
    weight_id: NodeId,
    grad_id:   NodeId,
    /// 누적 제곱 그래디언트
    g_accum:   Vec<f32>,
}

/// AdaGrad (Adaptive Gradient).
///
/// 업데이트 규칙:
/// ```text
/// G += grad²
/// W = W - lr * grad / sqrt(G + eps)
/// ```
///
/// 파라미터마다 학습률을 자동 조정한다.
/// 단, G가 계속 누적되므로 학습이 장기화될수록 업데이트가 소멸할 수 있다.
pub struct AdaGrad {
    lr:     f32,
    eps:    f32,
    params: Vec<AdaGradState>,
}

impl AdaGrad {
    /// * `lr`  - 초기 학습률 (일반적으로 0.01)
    /// * `eps` - 수치 안정성을 위한 소량값 (일반적으로 1e-8)
    pub fn new(lr: f32, eps: f32) -> Self {
        Self { lr, eps, params: vec![] }
    }
}

impl Optimizer for AdaGrad {
    fn register(&mut self, param: &dyn Parameter) {
        let size = TENSOR_STORAGE.with_borrow(|s| {
            s.get(&param.node_id()).map(|w| w.data.len()).unwrap_or(0)
        });
        self.params.push(AdaGradState {
            weight_id: param.node_id(),
            grad_id:   param.grad().id(),
            g_accum:   vec![0.0; size],
        });
    }

    fn step(&mut self) -> MlResult<()> {
        let grad_ids: Vec<NodeId> = self.params.iter().map(|e| e.grad_id).collect();
        let grads = snapshot_grads(&grad_ids);

        let lr = self.lr;
        let eps = self.eps;

        TENSOR_STORAGE.with_borrow_mut(|storage| {
            for (e, grad) in self.params.iter_mut().zip(grads.iter()) {
                if grad.is_empty() { continue; }

                // G += grad²
                e.g_accum.iter_mut()
                    .zip(grad.iter())
                    .for_each(|(g, &gi)| *g += gi * gi);

                // W = W - lr * grad / sqrt(G + eps)
                if let Some(w) = storage.get_mut(&e.weight_id) {
                    w.data.iter_mut()
                        .zip(grad.iter())
                        .zip(e.g_accum.iter())
                        .for_each(|((p, &gi), &acc)| {
                            *p -= lr * gi / (acc + eps).sqrt();
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
