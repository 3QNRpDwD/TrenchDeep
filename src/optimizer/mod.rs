pub mod sgd;
pub mod momentum;
pub mod adagrad;
pub mod rmsprop;
pub mod adam;
pub mod adamw;

pub use sgd::SGD;
pub use momentum::Momentum;
pub use adagrad::AdaGrad;
pub use rmsprop::RMSProp;
pub use adam::Adam;
pub use adamw::AdamW;

use crate::MlResult;
use crate::nn::Parameter;
use crate::tensor::{NodeId, TENSOR_STORAGE};

#[derive(thiserror::Error, Debug)]
pub enum OptimError {
    #[error("Gradient Error: {0}")]
    GradientError(String),
}

/// 모든 옵티마이저가 구현해야 하는 공통 인터페이스.
///
/// # 사용 예시
/// ```no_run
/// let mut opt = SGD::new(0.01);
/// opt.register(&model.w1);
/// opt.register(&model.b1);
///
/// // 학습 루프
/// loss.backward()?;
/// opt.step()?;
/// opt.zero_grad()?;
/// ```
pub trait Optimizer {
    /// 파라미터를 옵티마이저에 등록한다.
    /// register 시점의 weight shape로 내부 상태(velocity, moment 등) 버퍼를 초기화한다.
    fn register(&mut self, param: &dyn Parameter);

    /// 등록된 모든 파라미터에 대해 1 스텝 업데이트를 수행한다.
    fn step(&mut self) -> MlResult<()>;

    /// 등록된 모든 파라미터의 그래디언트를 0으로 초기화한다.
    fn zero_grad(&self) -> MlResult<()>;

    fn lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);
}

/// 등록된 grad NodeId 목록의 그래디언트를 일괄 초기화한다.
/// 모든 옵티마이저의 zero_grad 구현에서 공유한다.
pub(crate) fn clear_grads(grad_ids: &[NodeId]) {
    TENSOR_STORAGE.with_borrow_mut(|storage| {
        for &id in grad_ids {
            if let Some(g) = storage.get_mut(&id) {
                if g.dirty {
                    g.data.iter_mut().for_each(|x| *x = 0.0);
                    g.dirty = false;
                }
            }
        }
    });
}

/// 등록된 grad NodeId 목록의 grad 데이터를 일괄 스냅샷으로 가져온다 (clone).
/// step() 내부에서 read/write borrow 충돌을 방지하기 위해 사용한다.
pub(crate) fn snapshot_grads(grad_ids: &[NodeId]) -> Vec<Vec<f32>> {
    grad_ids.iter().map(|id| {
        TENSOR_STORAGE.with_borrow(|storage| {
            storage.get(id).map(|g| g.data.clone()).unwrap_or_default()
        })
    }).collect()
}
