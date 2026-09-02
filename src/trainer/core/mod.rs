//! 학습 루프 공통 인프라.
//!
//! 모든 패러다임별 트레이너(`SupervisedTrainer`, `UnsupervisedTrainer`,
//! `SemiSupervisedTrainer`, `RLTrainer`) 가 공유하는 타입·함수를 모아둔다.
//!
//! - [`TrainerCore`] : 로그 설정 + 메트릭 훅 보관 (아키텍처-불문 공용 상태).
//! - [`LogConfig`], [`Metrics`], [`TrainerBuilder`] — 학습 루프 구성.
//! - [`MetricHook`], [`BatchContext`] — 플러그인 메트릭 프로토콜.
//! - [`Convergence`] — 조기 종료 판정.
//! - `grad_norm`, `weight_norm`, `update_ratio`, `ClassificationAccuracy` 등 내장 메트릭.

pub mod config;
pub mod metric_hook;
pub mod metrics;
pub mod convergence;
pub mod epoch_loop;
pub mod runtime;

use std::cell::RefCell;

// 부모(`trainer`) 로부터 상속받는 공통 심볼.
// 하위 모듈이 `use super::*;` 로 가져다 쓴다.
pub(crate) use super::{MlError, MlResult, Parameter, TensorBase};

// 공용 API 재수출.
pub use config::{LogConfig, TrainerConfig, Metrics, TrainerBuilder};
pub use metric_hook::{MetricHook, BatchContext};
pub use metrics::{
    grad_norm, weight_norm, update_ratio, has_invalid_grad,
    argmax, ClassificationAccuracy, Perplexity,
};
pub use convergence::Convergence;
pub use epoch_loop::{EpochStep, StepOutput, StepDiagnostics, BatchObservations, EpochOutcome};
pub use runtime::TrainingRuntime;

/// 아키텍처-불문 공용 상태.
///
/// - `config` : 기존 `LogConfig` 그대로. 내장 메트릭 플래그와 로그 주기를 담는다.
/// - `hooks`  : `MetricHook` 의 동적 디스패치 목록. `RefCell` 로 감싸두어
///              `&TrainerCore` 에서도 훅을 업데이트할 수 있게 한다
///              (Phase 3 에서 `run_epoch(&self, ...)` 가 훅을 돌리기 위함).
pub struct TrainerCore {
    pub(crate) config: LogConfig,
    pub(crate) hooks:  RefCell<Vec<Box<dyn MetricHook>>>,
    pub(crate) runtime: TrainingRuntime,
}

impl TrainerCore {
    /// 빈 훅 목록과 함께 `LogConfig` 로부터 새 `TrainerCore` 를 생성함.
    pub fn new(config: LogConfig) -> Self {
        let runtime = TrainingRuntime::new(config.seed);
        Self { config, hooks: RefCell::new(Vec::new()), runtime }
    }

    /// 훅을 하나 추가함. 에폭당 순서대로 `update` → `format` 이 호출된다.
    ///
    /// `&self` 를 취하는 이유는 훅 장착이 `Trainer::fit()` 직전의 `&Trainer`
    /// 체이닝 스타일에서도 자연스럽게 동작하도록 하기 위함이다.
    pub fn add_hook(&self, hook: Box<dyn MetricHook>) {
        self.hooks.borrow_mut().push(hook);
    }

    /// 현재 등록된 훅 개수.
    pub fn hook_count(&self) -> usize {
        self.hooks.borrow().len()
    }

    /// 모든 훅을 제거한다. 테스트에서 상태를 리셋할 때 유용.
    pub fn clear_hooks(&self) {
        self.hooks.borrow_mut().clear();
    }

    /// 내부 `LogConfig` 에 대한 읽기 전용 접근.
    pub fn config(&self) -> &LogConfig {
        &self.config
    }

    pub(crate) fn shuffle<T>(&self, values: &mut [T]) {
        self.runtime.shuffle(values);
    }

    /// Epoch-derived streams make uninterrupted and resumed epoch ordering identical.
    pub(crate) fn begin_epoch(&self, epoch: usize) {
        const EPOCH_MIX: u64 = 0x9E37_79B9_7F4A_7C15;
        self.runtime.reseed(self.config.seed ^ (epoch as u64).wrapping_mul(EPOCH_MIX));
    }

    pub(crate) fn random_f32(&self) -> f32 {
        self.runtime.random_f32()
    }
}
