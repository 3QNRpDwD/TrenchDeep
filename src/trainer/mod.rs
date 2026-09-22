//! One training service shared by every paradigm.
use crate::{
    ContextError, ContextId, ExecutionContext, MlError, MlResult, Parameter, Tensor, TensorBuffer,
    Variable,
};
pub mod api;
pub mod checkpoint;
pub mod core;
pub mod data;
mod prepared;
pub(crate) mod objective;
pub use objective::{Objective, PreparedObjective, ForwardModel, Supervised, Autoregressive, DiffusionObjective, SemiSupervised};
pub(crate) mod progress;
mod reinforcement;
mod runners;
mod service;
pub use api::*;
pub use core::*;
pub use data::*;
pub use reinforcement::*;
pub use service::BatchInputs;
pub trait TrainableModel {
    fn context_id(&self) -> ContextId;
    fn parameters(&self) -> Vec<&Parameter>;
}
/// Low-level execution contract used by the internal model/objective adapter.
pub trait TrainingModel: TrainableModel {
    type Batch: BatchInputs;
    const PARADIGM: &'static str;
    fn forward_batch(
        &mut self,
        batch: &Self::Batch,
        step: &TrainingStepContext,
    ) -> MlResult<TrainingOutput>;
}
/// Zero-based model-side indices. Observer indices remain one-based.
#[derive(Debug, Clone, Copy, Default)]
pub struct TrainingStepContext {
    pub epoch: usize,
    pub batch: usize,
    pub lambda: Option<f32>,
}
/// Loss and metadata consumed by the common optimizer/metrics pipeline.
pub struct TrainingOutput {
    pub loss: Variable,
    pub prediction: Option<Variable>,
    pub target: Option<Tensor>,
    pub weight: usize,
    pub tokens: Option<usize>,
    pub lambda: Option<f32>,
}
#[derive(Debug)]
pub struct Eager;
#[derive(Debug)]
pub struct Prepared;
/// Context-bound training with an explicit execution mode.
pub struct Trainer<Mode = Eager> {
    pub(crate) service: service::TrainingService,
    ramp: ConsistencyRamp,
    mode: std::marker::PhantomData<Mode>,
}
impl Trainer<Eager> {
    pub fn builder(context: &ExecutionContext) -> TrainerBuilder {
        TrainerBuilder::new(context)
    }
    pub fn new(context: &ExecutionContext) -> Self {
        Self::silent(context)
    }
    /// Static execution; unsupported models/providers never fall back.
    pub fn prepared(self) -> Trainer<Prepared> {
        Trainer {
            service: self.service,
            ramp: self.ramp,
            mode: std::marker::PhantomData,
        }
    }
    // ── 프리셋 ────────────────────────────────────────────────────────────

    /// 최대 성능 모드. 모든 로그·NaN 검사가 비활성화.
    ///
    /// # 주의
    /// NaN 검사가 꺼져 있으므로 발산이 발생해도 감지되지 않음.
    /// 완전히 검증된 모델과 학습률 조합에서만 사용.
    pub fn silent(context: &ExecutionContext) -> Self {
        Self::builder(context)
            .log_every_n_batches(0)
            .summarize_every_n_batches(0)
            .log_every_n_epochs(0)
            .nan_check(false)
            .metrics(Metrics::none())
            .show_progress(false)
            .build()
    }

    /// 핵심 메트릭 모드. progress bar에 배치 손실을 표시하고 NaN 검사를 유지한다.
    pub fn minimal(context: &ExecutionContext) -> Self {
        Self::builder(context)
            .log_every_n_batches(1)
            .summarize_every_n_batches(0)
            .log_every_n_epochs(10)
            .nan_check(true)
            .metrics(Metrics::none())
            .show_progress(true)
            .build()
    }

    /// 기본 모드. 핵심 메트릭과 패러다임 대표 메트릭을 표시한다.
    pub fn default(context: &ExecutionContext) -> Self {
        Self::builder(context)
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
    pub fn verbose(context: &ExecutionContext) -> Self {
        Self::builder(context)
            .log_every_n_batches(1)
            .summarize_every_n_batches(100)
            .log_every_n_epochs(1)
            .nan_check(true)
            .metrics(Metrics::all())
            .show_progress(true)
            .build()
    }

    pub fn reinforcement(self, context: &ExecutionContext) -> RLTrainer {
        RLTrainer::from_trainer(context, self)
    }
}
impl<Mode> Trainer<Mode> {
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.service.core.config.seed = seed;
        self.service.core.runtime.reseed(seed);
        self
    }
    pub fn with_hook(self, hook: Box<dyn MetricHook>) -> Self {
        self.service.core.add_hook(hook);
        self
    }
    pub fn with_observer(self, observer: Box<dyn TrainingObserver>) -> Self {
        self.service.core.add_observer(observer);
        self
    }
    pub fn check_finite_gradients(mut self, enabled: bool) -> Self {
        self.service.core.config.nan_check_interval = if enabled { 1 } else { usize::MAX };
        self
    }
    pub fn with_max_grad_norm(mut self, max: f32) -> MlResult<Self> {
        if !max.is_finite() || max <= 0.0 {
            return Err(MlError::StringError(
                "max_grad_norm must be finite and positive".into(),
            ));
        }
        self.service.max_grad_norm = Some(max);
        Ok(self)
    }
    /// Applied only to models whose PARADIGM is "semi_supervised".
    pub fn with_ramp(mut self, ramp: ConsistencyRamp) -> Self {
        self.ramp = ramp;
        self
    }
    fn step_context<M: TrainingModel>(&self, epoch: usize, batch: usize) -> TrainingStepContext {
        TrainingStepContext {
            epoch,
            batch,
            lambda: (M::PARADIGM == "semi_supervised").then(|| self.ramp.value(epoch)),
        }
    }
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
