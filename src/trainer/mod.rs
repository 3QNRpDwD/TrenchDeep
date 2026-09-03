//! 학습 루프 인프라. 패러다임별로 모델 트레잇과 트레이너가 분리되어 있다.
//!
//! | 패러다임   | 모델 트레잇              | 트레이너                  | 손실 시그니처            |
//! |-----------|-------------------------|--------------------------|-------------------------|
//! | 지도학습   | [`SupervisedModel`]     | [`SupervisedTrainer`]    | `forward_loss(x, t)`    |
//! | 비지도학습 | [`UnsupervisedModel`]   | [`UnsupervisedTrainer`]  | `forward_loss(x)`       |
//! | 반지도학습 | [`SemiSupervisedModel`] | [`SemiSupervisedTrainer`]| `forward_loss(x_l, t_l, x_u, λ)` |
//! | 강화학습   | [`RLModel`] + [`Environment`] | [`RLTrainer`]      | REINFORCE 내부 구성     |
//! | 자기회귀   | [`AutoregressiveModel`] | [`AutoregressiveTrainer`]| `forward_loss(x) → (y, loss, n_tokens)` |
//!
//! 공통 인프라:
//! - [`TrainerCore`] — 로그 설정, 메트릭 훅, 체크포인트 인터럽트.
//! - [`TrainResult`] — 모든 트레이너의 공통 반환 타입.
//! - [`Convergence`] — 연속 두 에폭의 손실 변화 기준 조기 종료.
//! - [`MetricHook`]  — 배치마다 커스텀 메트릭을 주입하는 훅.
//!
//! `Trainer` 는 공통 설정을 소유하는 단일 facade다. `supervised()` 등으로
//! 정적 타입이 보존된 패러다임별 runner를 선택한다.
//!
//! `debugging` feature를 활성화하면 모델 파라미터 구조와 에폭/배치 실행
//! 컨텍스트가 기존 연산별 forward/backward trace에 자동으로 추가된다.
//! 동적 progress bar는 상세 trace와 터미널 행이 충돌하지 않도록 비활성화된다.

pub mod core;
pub mod api;
pub mod checkpoint;
pub mod supervised;
pub mod unsupervised;
pub mod semi_supervised;
pub mod reinforcement;
pub mod autoregressive;
pub mod data;
pub(crate) mod progress;

pub use core::{
    TrainerCore, LogConfig, TrainerConfig, TrainingRuntime, Metrics, TrainerBuilder,
    MetricHook, BatchContext, Convergence,
    grad_norm, weight_norm, update_ratio, has_invalid_grad,
    argmax, ClassificationAccuracy, Perplexity,
    EpochStep, StepOutput, StepDiagnostics, BatchObservations, EpochOutcome,
    TrainingObserver, BatchStartContext, BatchEndContext, EpochContext, TrainStartContext,
    TrainEndContext,
};
#[cfg(feature = "enableVisualization")]
pub use core::{CaptureSelector, GraphVisualizationObserver, GraphVisualizationObserverBuilder};
pub use api::{TrainableModel, CheckpointableModel, StopReason, StepUnit, MetricValues,
    CheckpointPaths, TrainResult, EpochSchedule, EpisodeSchedule, SupervisedDataset,
    UnsupervisedDataset, SemiSupervisedDataset, AutoregressiveDataset};
pub use api::{SupervisedOptions, SemiSupervisedOptions, AutoregressiveOptions, ReinforcementOptions};
pub use checkpoint::{TrainingCheckpoint, ParadigmTag, CheckpointManager, CHECKPOINT_SCHEMA_VERSION};
pub use data::{
    AutoregressiveBatch, AutoregressiveSample, AutoregressiveStackCollator, BatchLoader,
    Collator, CsvRecord, CsvSource, DataError, DataLoader, DataLoaderBuilder, Dataset,
    DatasetBuilder, InMemoryDataset, IntoBatchLoader, JsonLinesSource, JsonRecord, MemorySource,
    RecordSource, SemiSupervisedBatch, SemiSupervisedDataLoader, Transform,
    SemiSupervisedDataLoaderBuilder, SupervisedBatch, SupervisedSample, SupervisedStackCollator,
    UnsupervisedBatch, UnsupervisedSample, UnsupervisedStackCollator,
};

// ── 지도학습 아키텍처 ──────────────────────────────────────────────────────
#[cfg(feature = "enableBackward")]
pub use supervised::SupervisedModel;
pub use supervised::SupervisedTrainer;

// ── 비지도학습 아키텍처 ────────────────────────────────────────────────────
#[cfg(feature = "enableBackward")]
pub use unsupervised::UnsupervisedModel;
pub use unsupervised::UnsupervisedTrainer;

// ── 반지도학습 아키텍처 ────────────────────────────────────────────────────
#[cfg(feature = "enableBackward")]
pub use semi_supervised::SemiSupervisedModel;
pub use semi_supervised::{SemiSupervisedTrainer, ConsistencyRamp};

// ── 강화학습 아키텍처 ──────────────────────────────────────────────────────
#[cfg(feature = "enableBackward")]
pub use reinforcement::RLModel;
pub use reinforcement::{Environment, StepResult, RLTrainer};

// ── 자기회귀 아키텍처 ──────────────────────────────────────────────────────
#[cfg(feature = "enableBackward")]
pub use autoregressive::AutoregressiveModel;
pub use autoregressive::AutoregressiveTrainer;

use progress::EpochProgress;

// ── trainer 하위 모듈 공통 import ────────────────────────────────────────────
pub(crate) use crate::{MlError, MlResult};
pub(crate) use crate::nn::Parameter;
pub(crate) use crate::tensor::TensorBase;
// ────────────────────────────────────────────────────────────────────────────
// Trainer — 범용 팩토리
// ────────────────────────────────────────────────────────────────────────────

/// 로그 설정과 훅 프리셋을 구성해 각 패러다임 트레이너로 변환하는 범용 팩토리.
///
/// 지도학습은 [`SupervisedTrainer`], 비지도학습은 [`UnsupervisedTrainer`] 등
/// 전용 트레이너가 실제 `fit` 루프를 가진다. `Trainer` 는 `TrainerCore` 구성을
/// 편리하게 생성하고 패러다임별 runner를 선택하는 출발점 역할을 한다.
///
/// # 프리셋 (권장 진입점)
/// ```no_run
/// use trench_deep::trainer::{SupervisedTrainer, Trainer};
/// let t: SupervisedTrainer = Trainer::default().supervised();
/// ```
///
/// # 커스텀 빌더
/// ```no_run
/// let trainer: trench_deep::trainer::SupervisedTrainer =
///     trench_deep::trainer::Trainer::builder()
///         .log_every_n_batches(50)
///         .metrics(trench_deep::trainer::Metrics::none().grad_norm().accuracy())
///         .show_progress(true)
///         .build()
///         .supervised();
/// ```
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

    pub fn supervised(self) -> SupervisedTrainer { self.into() }
    pub fn unsupervised(self) -> UnsupervisedTrainer { self.into() }
    pub fn semi_supervised(self) -> SemiSupervisedTrainer { self.into() }
    pub fn autoregressive(self) -> AutoregressiveTrainer { self.into() }
    pub fn reinforcement(self) -> RLTrainer { self.into() }
}

#[cfg(test)]
mod preset_tests {
    use super::*;

    #[test]
    fn logging_presets_keep_metric_layers_separate() {
        let silent = Trainer::silent().core;
        assert!(!silent.config().show_progress);
        assert_eq!(silent.config().batch_log_interval, usize::MAX);
        assert_eq!(silent.config().epoch_log_interval, usize::MAX);
        assert!(!silent.config().metrics.paradigm);

        let minimal = Trainer::minimal().core;
        assert!(minimal.config().show_progress);
        assert_eq!(minimal.config().batch_log_interval, 1);
        assert_eq!(minimal.config().batch_summary_interval, usize::MAX);
        assert_eq!(minimal.config().epoch_log_interval, 10);
        assert!(!minimal.config().metrics.paradigm);
        assert!(!minimal.config().metrics.grad_norm);
        assert!(!minimal.config().metrics.update_ratio);
        assert!(!minimal.config().metrics.fw_bw_timing);

        let default = Trainer::default().core;
        assert!(default.config().metrics.paradigm);
        assert!(!default.config().metrics.grad_norm);
        assert!(!default.config().metrics.update_ratio);
        assert!(!default.config().metrics.fw_bw_timing);
        assert_eq!(default.config().batch_summary_interval, usize::MAX);
        assert_eq!(default.config().epoch_log_interval, 10);

        let verbose = Trainer::verbose().core;
        assert!(verbose.config().metrics.paradigm);
        assert!(verbose.config().metrics.grad_norm);
        assert!(verbose.config().metrics.update_ratio);
        assert!(verbose.config().metrics.accuracy);
        assert!(verbose.config().metrics.fw_bw_timing);
        assert_eq!(verbose.config().batch_summary_interval, 100);
        assert_eq!(verbose.config().epoch_log_interval, 1);
    }
}
