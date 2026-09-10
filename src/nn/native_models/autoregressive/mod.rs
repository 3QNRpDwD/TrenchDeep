//! 자기회귀학습 파일럿: **Bigram 언어모델**.
//!
//! 이 모듈은 Phase 1 에서 도입된 `AutoregressiveTrainer` + `AutoregressiveModel`
//! 인터페이스가 end-to-end 로 동작함을 최소 예제로 증명한다.
//!
//! ## 구조
//!
//! - 어휘 크기 `V`, 시퀀스 길이 `L+1` (입력 `L` 토큰 + 타깃 `L` 토큰)
//! - 파라미터: `W: [V, V]` 단일 전이 행렬(bigram transition logits)
//! - 입력은 한 시퀀스당 `[L+1, V]` one-hot 텐서 하나로 패킹된다:
//!   - `rows[0..L]`  → 입력 토큰
//!   - `rows[1..L+1]` → 타깃 토큰 (shift-by-one)
//! - 모델이 내부에서 입력/타깃을 분리한 뒤, 입력 leaf Variable 로부터
//!   `logits = input @ W` 를 계산하고 `SoftmaxCrossEntropyLoss` 로 손실을 구한다.
//!
//! 이 수준의 모델은 *실제 LM* 이라기보다는 **AR 트레이너의 제어 흐름을
//! 검증하는 도구**이다. 실 LM 은 임베딩 + RNN/Transformer 로 구현된다.

use super::*;

use crate::legacy::{
    MlResult,
    loss::SoftmaxCrossEntropyLoss,
    nn::Variable,
    tensor::{
        GlobalFunction, GlobalTensor, Tensor, TensorBase,
        operators::{Function, Matmul},
    },
    var_with_label,
};

// ────────────────────────────────────────────────────────────────────────────
// BigramLM
// ────────────────────────────────────────────────────────────────────────────

/// Bigram 토이 언어모델.
pub struct BigramLM {
    pub w: Variable,
    loss_fn: GlobalFunction,
    vocab: usize,
}

impl BigramLM {
    /// `vocab` 크기의 전이 행렬을 작은 무작위 값으로 초기화하여 구성한다.
    pub fn new(vocab: usize) -> MlResult<Self> {
        let loss_fn = SoftmaxCrossEntropyLoss::new()?;
        let w_data: Vec<f32> = (0..vocab * vocab)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.2)
            .collect();
        let w = var_with_label!(Tensor::from_vec(w_data, &[vocab, vocab])?, "bigram_w");
        Ok(Self { w, loss_fn, vocab })
    }

    /// 한 토큰 위치의 one-hot row 를 `[1, V]` Variable 로 만든다.
    fn row_var(&self, data: &[f32], row: usize) -> MlResult<Variable> {
        let v = self.vocab;
        let slice: Vec<f32> = data[row * v..(row + 1) * v].to_vec();
        Ok(Variable::new(Tensor::from_vec(slice, &[1, v])?))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AutoregressiveModel impl
// ────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "enableBackward")]
impl crate::legacy::trainer::AutoregressiveModel for BigramLM {
    fn forward_loss(&mut self, x: &Variable) -> MlResult<(Variable, Variable, usize)> {
        // 입력은 `[L+1, V]` one-hot 시퀀스.
        // `SoftmaxCrossEntropyLoss::forward` 가 `[1, V]` 단일 행 입력에만
        // 올바른 per-row log-sum-exp 를 계산하므로, 각 (t, t+1) 쌍마다
        // 개별 loss 를 구한 뒤 Variable 수준에서 합산한다.
        let shape = x.tensor().shape();
        let l_plus_1 = shape[0];
        let v = shape[1];
        assert_eq!(v, self.vocab, "vocab 차원 불일치");
        assert!(l_plus_1 >= 2, "시퀀스 길이는 최소 2 (L≥1)");
        let l = l_plus_1 - 1;

        let data = x.tensor().data().to_vec();

        let mut matmul = Matmul::new()?;
        let mut last_logits: Option<Variable> = None;
        let mut total_loss: Option<Variable> = None;

        for t in 0..l {
            let input_t = self.row_var(&data, t)?;
            let target_t = self.row_var(&data, t + 1)?;
            let logits_t = matmul.apply(&[&input_t, &self.w])?;
            let loss_t = self
                .loss_fn
                .apply_with_label(&[&logits_t, &target_t], "bigram_loss")?;
            total_loss = Some(match total_loss {
                Some(acc) => &acc + &loss_t,
                None => loss_t,
            });
            last_logits = Some(logits_t);
        }

        // 평균 NLL 로 변환: total_loss / L
        let loss_sum = total_loss.expect("l>=1 이면 최소 하나의 loss 가 누적됨");
        let scale = Variable::new(Tensor::from_vec(vec![1.0 / l as f32], &[1, 1])?);
        let mut mul = crate::legacy::tensor::operators::Mul::new()?;
        let loss_mean = mul.apply(&[&loss_sum, &scale])?;

        Ok((last_logits.unwrap(), loss_mean, l))
    }

    fn predict_raw(&mut self, x: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let matmul = Matmul::new()?;
        let logits = matmul.forward(&[x, self.w.tensor()])?.remove(0);
        Ok(logits)
    }
}

impl crate::legacy::trainer::TrainableModel for BigramLM {
    fn params(&self) -> Vec<&dyn Parameter> {
        vec![&self.w]
    }
}
impl crate::legacy::trainer::CheckpointableModel for BigramLM {}

// ────────────────────────────────────────────────────────────────────────────
// 테스트
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "enableBackward")]
#[path = "../../../tests/nn/native_models/autoregressive/mod_tests.rs"]
mod tests;
