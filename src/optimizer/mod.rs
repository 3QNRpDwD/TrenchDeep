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

use crate::{
    MlResult,
    nn::Parameter,
    tensor::{NodeId, TENSOR_STORAGE}
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerKind { Sgd, Momentum, AdaGrad, RmsProp, Adam, AdamW }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterOptimizerState {
    pub shape: Vec<usize>,
    pub buffers: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerSnapshot {
    pub version: u32,
    pub kind: OptimizerKind,
    pub lr: f32,
    pub scalars: Vec<f32>,
    pub parameters: Vec<ParameterOptimizerState>,
}

#[derive(thiserror::Error, Debug)]
pub enum OptimError {
    #[error("Gradient Error: {0}")]
    GradientError(String),
}

/// 모든 옵티마이저가 구현해야 하는 공통 인터페이스.
///
/// # 사용 예시
/// ```ignore
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

    /// TODO(Phase-6): each optimizer must serialize its state buffers.
    fn snapshot(&self) -> MlResult<OptimizerSnapshot> {
        Err(crate::MlError::StringError("optimizer snapshot is not implemented (Phase-6)".into()))
    }

    /// TODO(Phase-6): restore after parameters have been registered in stable order.
    fn restore(&mut self, _snapshot: &OptimizerSnapshot) -> MlResult<()> {
        Err(crate::MlError::StringError("optimizer restore is not implemented (Phase-6)".into()))
    }

    fn registered_param_count(&self) -> usize { 0 }
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

/// 전체 파라미터의 gradient에 대해 L2 norm을 계산하고,
/// `max_norm`을 초과하면 모든 gradient를 비례 축소한다.
///
/// PyTorch의 `torch.nn.utils.clip_grad_norm_` 과 동일한 동작.
///
/// # Arguments
/// * `params` - gradient clipping 대상 파라미터 목록
/// * `max_norm` - 허용할 최대 L2 norm
///
/// # Returns
/// clipping 전 총 gradient norm (모니터링용)
pub fn clip_grad_norm(params: &[&dyn Parameter], max_norm: f32) -> f32 {
    // 1) 전체 grad 의 L2 norm 계산
    let total_norm_sq: f32 = TENSOR_STORAGE.with_borrow(|storage| {
        params.iter().map(|p| {
            storage.get(&p.grad().id())
                .filter(|g| g.dirty)
                .map(|g| g.data.iter().map(|&v| v * v).sum::<f32>())
                .unwrap_or(0.0)
        }).sum()
    });
    let total_norm = total_norm_sq.sqrt();

    // 2) max_norm 초과 시 비례 축소
    if total_norm > max_norm {
        let clip_coef = max_norm / (total_norm + 1e-6);
        TENSOR_STORAGE.with_borrow_mut(|storage| {
            for p in params {
                if let Some(g) = storage.get_mut(&p.grad().id()) {
                    if g.dirty {
                        g.data.iter_mut().for_each(|v| *v *= clip_coef);
                    }
                }
            }
        });
    }

    total_norm
}
