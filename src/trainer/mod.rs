//! One training service shared by every paradigm.
use crate::{ExecutionContext,MlResult,MlError,Variable,Tensor,Parameter,TensorBuffer,ContextId,ContextError};
pub mod api;
pub mod core;
pub mod data;
pub mod checkpoint;
pub(crate) mod progress;
mod service;
mod runners;
mod reinforcement;
pub use api::*;
pub use core::*;
pub use data::*;
pub use runners::*;
pub use reinforcement::*;
pub trait TrainableModel {
    fn context_id(&self)->ContextId;
    fn parameters(&self)->Vec<&Parameter>;
}
pub trait SupervisedModel:TrainableModel {fn forward_loss(&mut self,input:&Variable,target:&Tensor)->MlResult<(Variable,Variable)>;}
pub trait UnsupervisedModel:TrainableModel {fn forward_loss(&mut self,input:&Variable)->MlResult<(Variable,Variable)>;}
pub trait SemiSupervisedModel:TrainableModel {fn forward_loss(&mut self,labeled_input:&Variable,labeled_target:&Tensor,unlabeled_input:&Variable,lambda:f32)->MlResult<(Variable,Variable)>;}
pub trait AutoregressiveModel:TrainableModel {fn forward_loss(&mut self,input:&Variable)->MlResult<(Variable,Variable,usize)>;}
pub struct Trainer {
    pub(crate) core: TrainerCore,
}

impl Trainer {
    /// 커스텀 빌더를 반환.
    pub fn builder() -> TrainerBuilder {
        TrainerBuilder::new()
    }

    pub fn with_observer(self, observer: Box<dyn TrainingObserver>) -> Self {
        self.core.add_observer(observer);
        self
    }

    // ── 프리셋 ────────────────────────────────────────────────────────────

    /// 최대 성능 모드. 모든 로그·NaN 검사가 비활성화.
    ///
    /// # 주의
    /// NaN 검사가 꺼져 있으므로 발산이 발생해도 감지되지 않음.
    /// 완전히 검증된 모델과 학습률 조합에서만 사용.
    pub fn silent() -> Self {
        Self::builder()
            .log_every_n_batches(0)
            .summarize_every_n_batches(0)
            .log_every_n_epochs(0)
            .nan_check(false)
            .metrics(Metrics::none())
            .show_progress(false)
            .build()
    }

    /// 핵심 메트릭 모드. progress bar에 배치 손실을 표시하고 NaN 검사를 유지한다.
    pub fn minimal() -> Self {
        Self::builder()
            .log_every_n_batches(1)
            .summarize_every_n_batches(0)
            .log_every_n_epochs(10)
            .nan_check(true)
            .metrics(Metrics::none())
            .show_progress(true)
            .build()
    }

    /// 기본 모드. 핵심 메트릭과 패러다임 대표 메트릭을 표시한다.
    pub fn default() -> Self {
        Self::builder()
            .log_every_n_batches(1)
            .summarize_every_n_batches(0)
            .log_every_n_epochs(10)
            .nan_check(true)
            .metrics(Metrics::default())
            .show_progress(true)
            .build()
    }

    /// 상세 진단 모드. 모든 메트릭을 활성화하고 완료 후 배치 요약을
    /// 100배치 간격으로 발행한다.
    pub fn verbose() -> Self {
        Self::builder()
            .log_every_n_batches(1)
            .summarize_every_n_batches(100)
            .log_every_n_epochs(1)
            .nan_check(true)
            .metrics(Metrics::all())
            .show_progress(true)
            .build()
    }

    pub fn supervised(self, context: &ExecutionContext) -> SupervisedTrainer { SupervisedTrainer::from_trainer(context,self) }
    pub fn unsupervised(self, context: &ExecutionContext) -> UnsupervisedTrainer { UnsupervisedTrainer::from_trainer(context,self) }
    pub fn semi_supervised(self, context: &ExecutionContext) -> SemiSupervisedTrainer { SemiSupervisedTrainer::from_trainer(context,self) }
    pub fn autoregressive(self, context: &ExecutionContext) -> AutoregressiveTrainer { AutoregressiveTrainer::from_trainer(context,self) }
    pub fn reinforcement(self, context: &ExecutionContext) -> RLTrainer { RLTrainer::from_trainer(context,self) }
}

#[derive(Debug, Clone, Copy)]
pub enum ConsistencyRamp {
    /// 고정 가중치. 전 에폭 동안 `w` 를 사용.
    Constant(f32),
    /// `0 → max_weight` 로 `ramp_epochs` 동안 시그모이드 램프업.
    /// Pi-model 논문의 기본 스케줄.
    Sigmoid { max_weight: f32, ramp_epochs: usize },
}

impl ConsistencyRamp {
    /// 주어진 에폭에서의 가중치 계산.
    pub fn value(&self, epoch: usize) -> f32 {
        match *self {
            ConsistencyRamp::Constant(w) => w,
            ConsistencyRamp::Sigmoid {
                max_weight,
                ramp_epochs,
            } => {
                if ramp_epochs == 0 {
                    return max_weight;
                }
                let e = epoch.min(ramp_epochs) as f32;
                let r = ramp_epochs as f32;
                // Pi-model 의 exp(-5·(1 - t)²) 스케줄
                let phase = 1.0 - e / r;
                max_weight * (-5.0 * phase * phase).exp()
            }
        }
    }
}

impl Default for ConsistencyRamp {
    /// 기본 스케줄: 30 에폭 동안 0 → 1.0 으로 시그모이드 램프업.
    fn default() -> Self {
        ConsistencyRamp::Sigmoid {
            max_weight: 1.0,
            ramp_epochs: 30,
        }
    }
}

