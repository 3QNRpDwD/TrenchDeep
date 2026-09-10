// 학습 루프의 조기 종료(수렴) 판정 추상화.
//
// 기존 `fit`/`resume` API는 `tolerance: f32` 하나로 수렴을 표현했다.
// 앞으로 아키텍처별 트레이너(비지도·반지도·강화학습)에서
// 서로 다른 수렴 기준이 필요해지므로, 판정 규칙을 enum 으로 캡슐화함.
//
// P0 단계에서는 기존 동작을 보존하기 위해 두 가지 변종만 제공함.
//
// ```text
// Off                  — 조기 종료하지 않음 (tolerance == 0 또는 음수)
// AbsDelta(tol)        — |last_loss - avg_loss| < tol 이면 종료
// ```
//
// 향후 `RelDelta`, `Patience`, `ValLoss` 등을 확장 지점으로 추가할 수 있다.
// 에폭 경계에서 학습을 조기 종료할지 판정하는 규칙.
//
// `Trainer::fit` 은 외부에서 `tolerance: f32` 를 받아
// `Convergence::from_tolerance(tolerance)` 로 변환한 뒤 내부 루프에 전달함.
// 체크포인트 직렬화는 여전히 `tolerance: f32` 로 왕복하므로
// `tolerance()` 메서드로 원래 값을 복원할 수 있다.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Convergence {
    /// 조기 종료를 수행하지 않음.
    Off,
    /// 연속 두 에폭 평균 손실의 절대차가 `tol` 미만이면 종료.
    AbsDelta(f32),
}

impl Convergence {
    // 기존 `tolerance: f32` 인자와의 호환 변환.
    //
    // - `tolerance <= 0.0` 또는 NaN 은 `Off` 로 해석함.
    // - 그 외에는 `AbsDelta(tolerance)`.
    pub fn from_tolerance(tolerance: f32) -> Self {
        if tolerance.is_nan() || tolerance <= 0.0 {
            Convergence::Off
        } else {
            Convergence::AbsDelta(tolerance)
        }
    }

    /// 체크포인트 직렬화용 원값(tolerance) 복원.
    ///
    /// `Off` 는 `0.0` 으로 반환함 (기존 스키마와의 왕복 호환).
    pub fn tolerance(&self) -> f32 {
        match self {
            Convergence::Off => 0.0,
            Convergence::AbsDelta(tol) => *tol,
        }
    }

    /// 조기 종료 판정.
    ///
    /// # 파라미터
    /// - `last_loss`: 직전 에폭의 평균 손실. 첫 에폭에서는 `f32::INFINITY` 를 전달할 수 있다.
    /// - `current_loss`: 방금 끝난 에폭의 평균 손실.
    pub fn should_stop(&self, last_loss: f32, current_loss: f32) -> bool {
        match self {
            Convergence::Off => false,
            Convergence::AbsDelta(tol) => {
                // last_loss 가 INFINITY 면 첫 에폭이므로 종료하지 않음.
                if !last_loss.is_finite() {
                    false
                } else {
                    (last_loss - current_loss).abs() < *tol
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/trainer/core/convergence_tests.rs"]
mod tests;
