use super::*;


struct ParamEntry {
    weight_id: NodeId,
    grad_id:   NodeId,
}

/// Stochastic Gradient Descent.
///
/// 업데이트 규칙: `W = W - lr * grad`
///
/// BGD/MiniBGD와 업데이트 규칙은 동일하며, 배치 구성은 학습 루프 측에서 담당한다.
pub struct SGD {
    lr:     f32,
    params: Vec<ParamEntry>,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self { lr, params: vec![] }
    }
}

impl Optimizer for SGD {
    fn register(&mut self, param: &dyn Parameter) {
        self.params.push(ParamEntry {
            weight_id: param.node_id(),
            grad_id:   param.grad().id(),
        });
    }

    fn step(&mut self) -> MlResult<()> {
        let grad_ids: Vec<NodeId> = self.params.iter().map(|e| e.grad_id).collect();
        let grads = snapshot_grads(&grad_ids);

        TENSOR_STORAGE.with_borrow_mut(|storage| {
            for (e, grad) in self.params.iter().zip(grads.iter()) {
                if grad.is_empty() { continue; }
                if let Some(w) = storage.get_mut(&e.weight_id) {
                    w.data.iter_mut()
                        .zip(grad.iter())
                        .for_each(|(p, &g)| *p -= self.lr * g);
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
