use super::*;

// ────────────────────────────────────────────────────────────────────────────
// 수치 유틸 함수 — 학습 루프 안팎에서 독립적으로 사용 가능
// ────────────────────────────────────────────────────────────────────────────

/// 등록된 파라미터 전체의 그래디언트 L2 노름.
///
/// `params`가 비어 있거나 grad가 아직 계산되지 않았으면 `0.0`을 반환.
pub fn grad_norm(params: &[&dyn Parameter]) -> f32 {
    params.iter()
        .flat_map(|p| p.grad().data().iter().map(|&g| g * g))
        .sum::<f32>()
        .sqrt()
}

/// 등록된 파라미터 전체의 가중치 L2 노름.
pub fn weight_norm(params: &[&dyn Parameter]) -> f32 {
    params.iter()
        .flat_map(|p| p.tensor().data().iter().map(|&w| w * w))
        .sum::<f32>()
        .sqrt()
}

/// Update Ratio = ||lr · grad|| / ||W||.
///
/// 현재 학습률에서 한 스텝의 업데이트 크기가 가중치 크기 대비 얼마나 되는지를 나타냄.
/// 일반적으로 `1e-3` 근처가 건강한 범위.
/// `||W|| ≤ 1e-12`이면 `0.0`을 반환 (제로 나눗셈 방지).
pub fn update_ratio(params: &[&dyn Parameter], lr: f32) -> f32 {
    let update_sq: f32 = params.iter()
        .flat_map(|p| p.grad().data().iter().map(|&g| { let u = lr * g; u * u }))
        .sum();
    let weight_sq: f32 = params.iter()
        .flat_map(|p| p.tensor().data().iter().map(|&w| w * w))
        .sum();
    if weight_sq > 1e-12 { update_sq.sqrt() / weight_sq.sqrt() } else { 0.0 }
}

/// 모든 파라미터의 그래디언트에 NaN 또는 Inf가 있는지 검사.
///
/// 조기 종료(short-circuit)로 구현되어 첫 비정상값 발견 시 즉시 반환.
/// Gradient explosion 시 NaN보다 Inf가 먼저 발생하는 경우가 많으므로
/// 두 조건을 모두 검사.
pub fn has_invalid_grad(params: &[&dyn Parameter]) -> bool {
    params.iter().any(|p|
        p.grad().data().iter().any(|x| x.is_nan() || x.is_infinite())
    )
}

// ────────────────────────────────────────────────────────────────────────────
// ClassificationAccuracy — argmax 기반 분류 정확도 누적기
// ────────────────────────────────────────────────────────────────────────────

/// argmax 비교로 배치 단위 분류 정확도를 누적한다.
///
/// # 주의
/// argmax가 유효하지 않은 샘플(all-zero 등)은 분모에서 제외.
/// 이는 손실의 분모(`total_loss_count`)와 다를 수 있으므로 두 값을 혼용하지 않도록 함.
#[derive(Default)]
pub struct ClassificationAccuracy {
    correct: usize,
    total:   usize,
}

impl ClassificationAccuracy {
    pub fn new() -> Self { Self::default() }

    /// 예측 텐서와 정답 텐서를 받아 내부 상태를 갱신.
    pub fn update(&mut self, pred: &dyn TensorBase, target: &dyn TensorBase) {
        if let (Some(p), Some(t)) = (argmax(pred.data()), argmax(target.data())) {
            if p == t { self.correct += 1; }
            self.total += 1;
        }
    }

    /// 현재까지 누적된 정확도를 백분율(0.0 ~ 100.0)로 반환.
    pub fn compute(&self) -> f32 {
        if self.total > 0 {
            (self.correct as f32 / self.total as f32) * 100.0
        } else {
            0.0
        }
    }

    /// 에폭 시작 시 상태를 초기화.
    pub fn reset(&mut self) {
        self.correct = 0;
        self.total   = 0;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 공통 유틸
// ────────────────────────────────────────────────────────────────────────────

/// 데이터 슬라이스에서 최대값의 인덱스를 반환.
///
/// 동률 시 첫 번째 최대값의 인덱스를 반환.
/// 슬라이스가 비어 있으면 `None`을 반환.
pub fn argmax(data: &[f32]) -> Option<usize> {
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
}
