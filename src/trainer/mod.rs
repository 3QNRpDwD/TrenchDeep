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

pub mod core;
pub mod api;
pub mod checkpoint;
pub mod supervised;
pub mod unsupervised;
pub mod semi_supervised;
pub mod reinforcement;
pub mod autoregressive;
pub(crate) mod progress;

pub use core::{
    TrainerCore, LogConfig, TrainerConfig, TrainingRuntime, Metrics, TrainerBuilder,
    MetricHook, BatchContext, Convergence,
    grad_norm, weight_norm, update_ratio, has_invalid_grad,
    argmax, ClassificationAccuracy, Perplexity,
    EpochStep, StepOutput, StepDiagnostics, BatchObservations, EpochOutcome,
};
pub use api::{TrainableModel, CheckpointableModel, StopReason, StepUnit, MetricValues,
    CheckpointPaths, TrainResult, EpochSchedule, EpisodeSchedule, SupervisedDataset,
    UnsupervisedDataset, SemiSupervisedDataset, AutoregressiveDataset};
pub use api::{SupervisedOptions, SemiSupervisedOptions, AutoregressiveOptions, ReinforcementOptions};
pub use checkpoint::{TrainingCheckpoint, ParadigmTag, CheckpointManager, CHECKPOINT_SCHEMA_VERSION};

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

    // ── 프리셋 ────────────────────────────────────────────────────────────

    /// 최대 성능 모드. 모든 로그·NaN 검사가 비활성화.
    ///
    /// # 주의
    /// NaN 검사가 꺼져 있으므로 발산이 발생해도 감지되지 않음.
    /// 완전히 검증된 모델과 학습률 조합에서만 사용.
    pub fn silent() -> Self {
        Self::builder()
            .log_every_n_batches(0)
            .log_every_n_epochs(0)
            .nan_check(false)
            .metrics(Metrics::none())
            .show_progress(false)
            .build()
    }

    /// 최소 로그 모드. 에폭 평균 손실만 출력한다. NaN 검사는 유지.
    pub fn minimal() -> Self {
        Self::builder()
            .log_every_n_batches(0)
            .nan_check(true)
            .metrics(Metrics::none().accuracy())
            .show_progress(false)
            .build()
    }

    /// 기본 모드. FW/BW 타이밍, GradNorm, Accuracy, progress bar 포함.
    pub fn default() -> Self {
        Self::builder()
            .log_every_n_batches(1)
            .nan_check(true)
            .metrics(Metrics::none().grad_norm().accuracy().fw_bw_timing())
            .show_progress(true)
            .build()
    }

    /// 전체 디버그 모드. 모든 메트릭 활성화.
    pub fn verbose() -> Self {
        Self::builder()
            .log_every_n_batches(1)
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
