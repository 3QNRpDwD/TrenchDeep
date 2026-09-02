use super::*;
use super::metric_hook::{MetricHook, BatchContext};

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
// MetricHook 어댑터
//
// `ClassificationAccuracy` 는 고유 `update/compute/reset` 메서드를 그대로 두고
// 훅 트레잇 경로는 경로 수식(Self::update)으로 위임한다. 이렇게 해야 기존
// 호출자(`accuracy.update(y.tensor(), t.tensor())`)가 그대로 동작하면서도
// `Box<dyn MetricHook>` 컨테이너에도 넣을 수 있다.
// ────────────────────────────────────────────────────────────────────────────

impl MetricHook for ClassificationAccuracy {
    fn update(&mut self, ctx: &BatchContext<'_>) -> MlResult<()> {
        // pred/target 가 모두 노출된 경우에만 누적.
        if let (Some(p), Some(t)) = (ctx.pred, ctx.target) {
            ClassificationAccuracy::update(self, p, t);
        }
        Ok(())
    }

    fn compute(&self) -> f32 {
        ClassificationAccuracy::compute(self)
    }

    fn reset(&mut self) -> MlResult<()> {
        ClassificationAccuracy::reset(self);
        Ok(())
    }

    fn name(&self) -> &str {
        "accuracy"
    }

    fn format(&self) -> String {
        format!("AC: {:>6.2}%", MetricHook::compute(self))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Perplexity — Autoregressive/LM 평가용 누적기
// ────────────────────────────────────────────────────────────────────────────

/// 토큰 단위 평균 음 로그우도(NLL)로부터 Perplexity 를 계산하는 누적기.
///
/// `update_loss(batch_loss, token_count)` 를 각 배치마다 호출하면,
/// `compute()` 가 `exp(Σ(loss_i · n_i) / Σ n_i)` 를 반환한다.
///
/// `token_count` 는 해당 배치의 **유효 타깃 토큰 수** (padding 제외).
/// 배치 손실이 `mean_over_tokens` 이면 `count` 를 `1` 로 넣어도 되지만,
/// 배치별 길이가 다르면 실제 토큰 수를 넘겨야 정확한 평균이 나온다.
#[derive(Default)]
pub struct Perplexity {
    nll_sum: f64,
    token_sum: usize,
}

impl Perplexity {
    pub fn new() -> Self { Self::default() }

    /// 배치의 평균 NLL 과 해당 배치의 유효 토큰 수를 받아 누적한다.
    pub fn update_loss(&mut self, mean_nll: f32, token_count: usize) {
        if token_count == 0 || !mean_nll.is_finite() { return; }
        self.nll_sum   += mean_nll as f64 * token_count as f64;
        self.token_sum += token_count;
    }

    /// 현재까지 누적된 평균 NLL.
    pub fn mean_nll(&self) -> f32 {
        if self.token_sum == 0 { 0.0 } else { (self.nll_sum / self.token_sum as f64) as f32 }
    }

    /// 에폭 경계에서 상태를 초기화.
    pub fn reset(&mut self) {
        self.nll_sum   = 0.0;
        self.token_sum = 0;
    }
}

impl MetricHook for Perplexity {
    fn update(&mut self, ctx: &BatchContext<'_>) -> MlResult<()> {
        // ctx.loss 는 스칼라 평균 NLL. token_count 가 노출되지 않으면 1 로
        // 간주해 배치 평균을 그대로 누적 (가변 길이 시퀀스에서는 정확도 떨어짐).
        let n = ctx.n_tokens.unwrap_or(1);
        self.update_loss(ctx.loss, n);
        Ok(())
    }

    fn compute(&self) -> f32 {
        self.mean_nll().exp()
    }

    fn reset(&mut self) -> MlResult<()> {
        Perplexity::reset(self);
        Ok(())
    }

    fn name(&self) -> &str {
        "perplexity"
    }

    fn format(&self) -> String {
        format!("PPL: {:>8.3}", MetricHook::compute(self))
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

// ────────────────────────────────────────────────────────────────────────────
// Tests — 훅 경로가 직접 호출 경로와 동일 결과를 내는지 확인.
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn classification_accuracy_hook_matches_direct() {
        // 3 개 배치. 정답 예측 2 개, 오답 예측 1 개 → 66.67%.
        let preds  = [
            Tensor::from_vec(vec![0.1, 0.9, 0.0], &[1, 3]).unwrap(),  // 1
            Tensor::from_vec(vec![0.7, 0.2, 0.1], &[1, 3]).unwrap(),  // 0
            Tensor::from_vec(vec![0.2, 0.3, 0.5], &[1, 3]).unwrap(),  // 2
        ];
        let targets = [
            Tensor::from_vec(vec![0.0, 1.0, 0.0], &[1, 3]).unwrap(),  // 1 ✓
            Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]).unwrap(),  // 0 ✓
            Tensor::from_vec(vec![0.0, 1.0, 0.0], &[1, 3]).unwrap(),  // 1 ✗
        ];

        // 직접 경로
        let mut direct = ClassificationAccuracy::new();
        for (p, t) in preds.iter().zip(targets.iter()) {
            direct.update(p, t);
        }

        // 훅 경로
        let mut hook = ClassificationAccuracy::new();
        for (i, (p, t)) in preds.iter().zip(targets.iter()).enumerate() {
            let ctx = BatchContext {
                batch_idx: i,
                pred:      Some(p as &dyn TensorBase),
                target:    Some(t as &dyn TensorBase),
                loss:      0.0,
                n_tokens:  None,
                lambda:    None,
                lr:        1e-3,
            };
            <ClassificationAccuracy as MetricHook>::update(&mut hook, &ctx);
        }

        assert!((direct.compute() - MetricHook::compute(&hook)).abs() < 1e-6);
        assert!((MetricHook::compute(&hook) - 200.0 / 3.0).abs() < 1e-4);
    }

    #[test]
    fn perplexity_hook_weighted_by_tokens() {
        // 두 배치: loss 1.0 @ 10 tokens, loss 2.0 @ 20 tokens
        // mean_nll = (1*10 + 2*20) / 30 = 50/30
        let mut hook = Perplexity::new();
        for (loss, n) in [(1.0_f32, 10usize), (2.0_f32, 20usize)] {
            let ctx = BatchContext {
                batch_idx: 0,
                pred:      None,
                target:    None,
                loss,
                n_tokens:  Some(n),
                lambda:    None,
                lr:        1e-3,
            };
            <Perplexity as MetricHook>::update(&mut hook, &ctx);
        }
        let expected = (50.0_f32 / 30.0).exp();
        let actual   = MetricHook::compute(&hook);
        assert!((actual - expected).abs() < 1e-4, "expected {}, got {}", expected, actual);
    }

    #[test]
    fn classification_accuracy_hook_skips_when_target_missing() {
        // target 이 None 이면 호출이 no-op 이어야 함 (비지도/AR 에 훅이 붙어도 안전).
        let mut hook = ClassificationAccuracy::new();
        let pred = Tensor::from_vec(vec![0.1, 0.9, 0.0], &[1, 3]).unwrap();
        let ctx = BatchContext {
            batch_idx: 0,
            pred:      Some(&pred as &dyn TensorBase),
            target:    None,
            loss:      0.0,
            n_tokens:  None,
            lambda:    None,
            lr:        1e-3,
        };
        <ClassificationAccuracy as MetricHook>::update(&mut hook, &ctx);
        assert_eq!(MetricHook::compute(&hook), 0.0);
    }
}
