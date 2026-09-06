use super::*;
#[derive(Default)]
pub struct ClassificationAccuracy {
    correct: usize,
    total:   usize,
}

impl ClassificationAccuracy {
    pub fn new() -> Self { Self::default() }

    /// 예측 텐서와 정답 텐서를 받아 내부 상태를 갱신.
    pub fn update(&mut self, pred: &TensorBuffer, target: &TensorBuffer) {
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


pub fn grad_norm(parameters:&[&Parameter])->MlResult<f32> {let mut n=0.0;for p in parameters {if let Some(g)=p.grad()? {n+=g.data().iter().map(|x|x*x).sum::<f32>();}}Ok(n.sqrt())}
pub fn weight_norm(parameters:&[&Parameter])->MlResult<f32> {let mut n=0.0;for p in parameters {n+=p.tensor().with_view(|v|v.data().iter().map(|x|x*x).sum::<f32>())?;}Ok(n.sqrt())}
/// Estimate based on the unclipped lr * gradient, not the optimizer's actual update.
pub fn update_ratio(parameters:&[&Parameter],lr:f32)->MlResult<f32> {let w=weight_norm(parameters)?;Ok(if w>1e-12 {lr*grad_norm(parameters)?/w}else{0.0})}
pub fn has_invalid_grad(parameters:&[&Parameter])->MlResult<bool> {for p in parameters {if let Some(g)=p.grad()? {if g.data().iter().any(|x|!x.is_finite()){return Ok(true);}}}Ok(false)}
