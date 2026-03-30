use super::*;

struct RMSPropState {
    weight_id: NodeId,
    grad_id:   NodeId,
    /// 지수이동평균 제곱 그래디언트
    g_avg:     Vec<f32>,
}

/// RMSProp (Root Mean Square Propagation).
///
/// 업데이트 규칙:
/// ```text
/// G = rho * G + (1 - rho) * grad²
/// W = W - lr * grad / sqrt(G + eps)
/// ```
///
/// AdaGrad의 G 누적 소멸 문제를 지수이동평균으로 해결한다.
pub struct RMSProp {
    lr:     f32,
    rho:    f32,
    eps:    f32,
    params: Vec<RMSPropState>,
}

impl RMSProp {
    /// * `lr`  - 학습률 (일반적으로 1e-3)
    /// * `rho` - 감쇠 계수 (일반적으로 0.9)
    /// * `eps` - 수치 안정성 소량값 (일반적으로 1e-8)
    pub fn new(lr: f32, rho: f32, eps: f32) -> Self {
        Self { lr, rho, eps, params: vec![] }
    }
}

impl Optimizer for RMSProp {
    fn register(&mut self, param: &dyn Parameter) {
        let size = TENSOR_STORAGE.with_borrow(|s| {
            s.get(&param.node_id()).map(|w| w.data.len()).unwrap_or(0)
        });
        self.params.push(RMSPropState {
            weight_id: param.node_id(),
            grad_id:   param.grad().id(),
            g_avg:     vec![0.0; size],
        });
    }

    fn step(&mut self) -> MlResult<()> {
        let grad_ids: Vec<NodeId> = self.params.iter().map(|e| e.grad_id).collect();
        let grads = snapshot_grads(&grad_ids);

        let lr  = self.lr;
        let rho = self.rho;
        let eps = self.eps;

        TENSOR_STORAGE.with_borrow_mut(|storage| {
            for (e, grad) in self.params.iter_mut().zip(grads.iter()) {
                if grad.is_empty() { continue; }

                // G = rho * G + (1 - rho) * grad²
                e.g_avg.iter_mut()
                    .zip(grad.iter())
                    .for_each(|(g, &gi)| *g = rho * *g + (1.0 - rho) * gi * gi);

                // W = W - lr * grad / sqrt(G + eps)
                if let Some(w) = storage.get_mut(&e.weight_id) {
                    w.data.iter_mut()
                        .zip(grad.iter())
                        .zip(e.g_avg.iter())
                        .for_each(|((p, &gi), &avg)| {
                            *p -= lr * gi / (avg + eps).sqrt();
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
