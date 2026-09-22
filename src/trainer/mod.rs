//! One training service shared by every paradigm.
use crate::{
    ContextError, ContextId, ExecutionContext, MlError, MlResult, Parameter, Tensor, TensorBuffer,
    Variable,
};
pub mod api;
pub mod checkpoint;
pub use checkpoint::ParadigmTag;
pub mod core;
pub mod data;
mod prepared;
pub(crate) mod strategy;
pub use strategy::{
    Autoregressive, DiffusionTraining, ForwardModel, PreparedTrainingStrategy, SemiSupervised,
    Supervised, TrainingStrategy,
};
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
/// Low-level execution contract used by the internal model/strategy/loss adapter.
pub trait TrainingModel: TrainableModel {
    type Batch: BatchInputs;
    const PARADIGM: ParadigmTag;
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
/// Context-bound training with a typed strategy and execution mode.
///
/// ```no_run
/// use trench_deep::{ExecutionContext, trainer::Trainer};
/// let ctx = ExecutionContext::new();
/// let trainer = Trainer::supervised(&ctx).minimal().prepared().verbose();
/// ```
///
/// Strategies reject incompatible models at compile time:
/// ```compile_fail
/// use trench_deep::{nn::Mlp, trainer::{Autoregressive, TrainingStrategy}};
/// fn requires_strategy<S: TrainingStrategy<Mlp>>() {}
/// requires_strategy::<Autoregressive>();
/// ```
pub struct Trainer<Mode = Eager, Strategy = Supervised> {
    pub(crate) service: service::TrainingService,
    strategy: Strategy,
    ramp: ConsistencyRamp,
    mode: std::marker::PhantomData<Mode>,
}
impl Trainer<Eager, Supervised> {
    pub fn supervised(context: &ExecutionContext) -> Self {
        Self::from_strategy(context, Supervised)
    }
    pub fn autoregressive(context: &ExecutionContext) -> Trainer<Eager, Autoregressive> {
        Self::from_strategy(context, Autoregressive)
    }
    pub fn diffusion(context: &ExecutionContext) -> Trainer<Eager, DiffusionTraining> {
        Self::from_strategy(context, DiffusionTraining)
    }
    pub fn semi_supervised(context: &ExecutionContext) -> Trainer<Eager, SemiSupervised> {
        Self::from_strategy(context, SemiSupervised { noise_scale: 0.1 })
    }
    /// Extension point for custom batch preparation and numeric training operations.
    pub fn from_strategy<S>(context: &ExecutionContext, strategy: S) -> Trainer<Eager, S> {
        Trainer {
            service: service::TrainingService::new(context, TrainerCore::new(LogConfig::default())),
            strategy,
            ramp: ConsistencyRamp::default(),
            mode: std::marker::PhantomData,
        }
    }
}
impl<S> Trainer<Eager, S> {
    /// Static execution; unsupported strategies/providers never fall back.
    pub fn prepared(self) -> Trainer<Prepared, S> {
        Trainer {
            service: self.service,
            strategy: self.strategy,
            ramp: self.ramp,
            mode: std::marker::PhantomData,
        }
    }
    pub fn reinforcement(self, context: &ExecutionContext) -> RLTrainer {
        RLTrainer::from_trainer(context, self)
    }
}
impl<Mode> Trainer<Mode, SemiSupervised> {
    pub fn with_noise_scale(mut self, noise_scale: f32) -> MlResult<Self> {
        self.strategy = SemiSupervised::new(noise_scale)?;
        Ok(self)
    }
}
impl<Mode, S> Trainer<Mode, S> {
    /// Disable metric logging, progress and finite-gradient checks.
    pub fn silent(self) -> Self {
        self.log_every_n_batches(0)
            .summarize_every_n_batches(0)
            .log_every_n_epochs(0)
            .nan_check(false)
            .metrics(Metrics::none())
            .show_progress(false)
    }
    /// Loss progress and finite-gradient checks, without additional metrics.
    pub fn minimal(self) -> Self {
        self.log_every_n_batches(1)
            .summarize_every_n_batches(0)
            .log_every_n_epochs(10)
            .nan_check(true)
            .metrics(Metrics::none())
            .show_progress(true)
    }
    /// Standard logging plus the strategy's representative metrics.
    pub fn default(self) -> Self {
        self.minimal().metrics(Metrics::default())
    }
    /// All metrics and detailed epoch/batch summaries.
    pub fn verbose(self) -> Self {
        self.log_every_n_batches(1)
            .summarize_every_n_batches(100)
            .log_every_n_epochs(1)
            .nan_check(true)
            .metrics(Metrics::all())
            .show_progress(true)
    }
    /// 몇 배치마다 메트릭을 계산하고 progress bar 메시지를 갱신할지 설정.
    ///
    /// `0`을 입력하면 배치 레벨 메시지가 완전히 비활성화.
    /// 예: `50` → 50배치마다 grad_norm 계산 + progress bar 갱신.
    pub fn log_every_n_batches(mut self, n: usize) -> Self {
        self.service.core.config.batch_log_interval = if n == 0 { usize::MAX } else { n };
        self
    }

    /// Progress 종료 후 발행할 배치 요약 간격. `0`이면 비활성화한다.
    pub fn summarize_every_n_batches(mut self, n: usize) -> Self {
        self.service.core.config.batch_summary_interval = if n == 0 { usize::MAX } else { n };
        self
    }

    /// 몇 에폭마다 에폭 요약 로그를 출력할지 설정.
    pub fn log_every_n_epochs(mut self, n: usize) -> Self {
        self.service.core.config.epoch_log_interval = if n == 0 { usize::MAX } else { n };
        self
    }

    /// NaN/Inf 그래디언트 검사 활성화 여부.
    ///
    /// `false`로 설정하면 성능이 향상되지만 발산 감지가 불가능.
    /// 완전히 검증된 모델·학습률 조합에서만 비활성화를 권장.
    pub fn nan_check(mut self, enabled: bool) -> Self {
        self.service.core.config.nan_check_interval = if enabled { 1 } else { usize::MAX };
        self
    }

    /// 활성화할 메트릭 집합을 설정.
    ///
    /// ```no_run
    /// use trench_deep::trainer::{Metrics, Trainer};
    /// let ctx = trench_deep::ExecutionContext::new();
    /// let trainer = Trainer::supervised(&ctx)
    ///     .metrics(Metrics::none().grad_norm().accuracy());
    /// ```
    pub fn metrics(mut self, m: Metrics) -> Self {
        self.service.core.config.metrics = m;
        self
    }

    /// 터미널 progress bar 출력 여부.
    pub fn show_progress(mut self, show: bool) -> Self {
        self.service.core.config.show_progress = show;
        self
    }

    /// 체크포인트 저장 디렉토리를 설정한다.
    ///
    /// 설정하면 학습 중 Ctrl+C 인터럽트 시 모델 가중치와 학습 상태를
    /// 이 디렉토리에 저장한다. 완전한 학습 상태 재개는 아직 지원하지 않는다.
    ///
    /// ```no_run
    /// use trench_deep::trainer::Trainer;
    /// let ctx = trench_deep::ExecutionContext::new();
    /// let trainer = Trainer::supervised(&ctx)
    ///     .checkpoint_dir("checkpoints/my_model");
    /// ```
    pub fn checkpoint_dir(mut self, dir: &str) -> Self {
        self.service.core.config.checkpoint_dir = Some(dir.to_string());
        self
    }

    pub fn checkpoint(self, dir: &str) -> Self {
        self.checkpoint_dir(dir)
    }

    /// Sets the deterministic training RNG seed.
    pub fn seed(self, seed: u64) -> Self {
        self.with_seed(seed)
    }

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
    /// Applied to strategies tagged `ParadigmTag::SemiSupervised`.
    pub fn with_ramp(mut self, ramp: ConsistencyRamp) -> Self {
        self.ramp = ramp;
        self
    }
    fn step_context<M: TrainingModel>(&self, epoch: usize, batch: usize) -> TrainingStepContext {
        TrainingStepContext {
            epoch,
            batch,
            lambda: (M::PARADIGM == ParadigmTag::SemiSupervised).then(|| self.ramp.value(epoch)),
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

#[cfg(all(test, feature = "builtinStorage", feature = "builtinKernels"))]
#[path = "../tests/trainer/preset_tests.rs"]
mod preset_tests;
