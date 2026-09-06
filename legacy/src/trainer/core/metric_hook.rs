//! 기존 `LogConfig.metrics` 비트플래그는 내장 지표(grad_norm, accuracy 등)만 다룰 수 있었다.
//! 아키텍처별 트레이너(비지도·반지도·강화학습)가 추가되면서 각기 다른 지표
//! (diffusion 의 SNR, RL 의 평균 보상, 반지도의 consistency loss 등)를 필요로 하므로,
//! 플러그인 방식으로 확장 가능한 훅 프로토콜을 제공한다.
//!
//! ## 동작 경로 (Phase 3 이후)
//!
//! 1. 사용자 또는 트레이너 프리셋이 `TrainerCore::add_hook(Box::new(MyHook::new()))` 로 장착.
//! 2. 매 배치마다 `TrainerCore::run_epoch` 내부에서 [`BatchContext`] 를 조립해
//!    `hook.update(&ctx)` 호출. NaN 감지로 조기 종료 시에는 호출하지 않는다.
//! 3. 에폭 경계에서 `hook.format()` 결과가 에폭 요약 로그 뒷부분에 합류.
//! 4. `hook.reset()` 은 다음 에폭 시작 시 자동 호출된다.
//!
//! ## 컨텍스트 노출 규약
//!
//! 패러다임별 [`crate::trainer::EpochStep`] 구현이 `last_pred / last_target /
//! last_n_tokens / last_lambda / current_lr` 메서드를 통해 노출하는 값만
//! `BatchContext` 에 채워진다. 패러다임에 의미 없는 필드는 `None` 이다.

use super::*;

/// 배치 한 번의 순방향·역방향이 끝난 시점에 훅으로 전달되는 문맥.
///
/// 모든 필드는 **패러다임이 노출한 경우에만** 값이 들어온다. 훅은 필요한
/// 필드만 꺼내 쓰고 나머지는 무시한다.
///
/// 수명 파라미터 `'a` 는 `EpochStep` 내부에서 stash 된 참조(예: 마지막
/// 배치의 `pred` Variable) 의 수명과 연동된다. 따라서 `BatchContext` 는
/// `update` 호출 범위를 넘어 저장해서는 안 된다.
pub struct BatchContext<'a> {
    /// 현재 에폭 내의 0-indexed 배치 번호.
    pub batch_idx: usize,
    /// 모델 순방향 출력 텐서. 스텝이 노출하지 않으면 `None`.
    pub pred:      Option<&'a dyn TensorBase>,
    /// 정답 텐서. 비지도/자기회귀처럼 명시적 타깃이 없으면 `None`.
    pub target:    Option<&'a dyn TensorBase>,
    /// 이번 배치의 스칼라 손실값 (`StepOutput::loss`).
    pub loss:      f32,
    /// 이 배치에서 유효한 타깃 토큰 수. 자기회귀(AR) 외에는 일반적으로 `None`.
    pub n_tokens:  Option<usize>,
    /// 반지도 학습의 현재 일관성 가중치. 해당 없으면 `None`.
    pub lambda:    Option<f32>,
    /// 옵티마이저의 현재 학습률.
    pub lr:        f32,
}

/// 학습 루프의 매 배치마다 호출되는 플러그인 가능한 메트릭.
///
/// # 설계 원칙
/// - **Dyn-compatible**: 제네릭 메서드·연관 타입 없이 `Box<dyn MetricHook>` 으로 보관 가능해야 한다.
/// - **Side-effect free between batches**: `update` 는 내부 상태만 누적하고, 외부 I/O 를 수행하지 않는다.
/// - **Cheap `compute`**: 에폭 요약 시 한 번 호출되므로 복잡한 연산을 피한다.
///
/// # Send 경계 미부여 사유
/// 현재 학습 루프는 단일 스레드이며, 훅이 `Variable` 이나 `GlobalTensor` 같은
/// 비-Send 타입을 내부에 붙잡을 자유를 유지하려고 의도적으로 `Send` 를 요구하지 않는다.
/// 병렬 학습을 도입할 때 별도의 `MetricHookSend` 를 만들거나 bound 를 추가한다.
pub trait MetricHook {
    /// 배치 한 번의 결과를 누적한다.
    fn update(&mut self, ctx: &BatchContext<'_>) -> MlResult<()>;

    /// 현재까지 누적된 값을 단일 f32 로 환산해 반환한다.
    ///
    /// 일반적으로 에폭 종료 시 한 번 호출되며, 로그 포맷터가 이 값을
    /// `format()` 과 함께 사용해 표시 문자열을 만든다.
    fn compute(&self) -> f32;

    /// 에폭 경계에서 내부 상태를 초기화한다.
    fn reset(&mut self) -> MlResult<()>;

    /// 로그 라인에 표시할 짧은 식별자 (예: `"acc"`, `"grad_norm"`).
    fn name(&self) -> &str;

    /// 기본 표시 포맷. 필요 시 구현체에서 override 한다.
    ///
    /// 기본은 `"<name>: <compute():.4>"`.
    fn format(&self) -> String {
        format!("{}: {:.4}", self.name(), self.compute())
    }
}
