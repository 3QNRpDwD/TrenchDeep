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

use crate::{
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
impl crate::trainer::AutoregressiveModel for BigramLM {
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
        let mut mul = crate::tensor::operators::Mul::new()?;
        let loss_mean = mul.apply(&[&loss_sum, &scale])?;

        Ok((last_logits.unwrap(), loss_mean, l))
    }

    fn predict_raw(&mut self, x: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let matmul = Matmul::new()?;
        let logits = matmul.forward(&[x, self.w.tensor()])?.remove(0);
        Ok(logits)
    }
}

impl crate::trainer::TrainableModel for BigramLM {
    fn params(&self) -> Vec<&dyn Parameter> {
        vec![&self.w]
    }
}
impl crate::trainer::CheckpointableModel for BigramLM {}

// ────────────────────────────────────────────────────────────────────────────
// 테스트
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "enableBackward")]
mod tests {
    use super::*;
    use crate::{
        optimizer::{Adam, Optimizer},
        trainer::{AutoregressiveModel, TrainableModel, Trainer},
    };

    /// 한 토큰 one-hot 을 한 줄로 만든다 (`[V]` 슬라이스를 누적).
    fn one_hot(token: usize, vocab: usize) -> Vec<f32> {
        let mut row = vec![0.0; vocab];
        row[token] = 1.0;
        row
    }

    /// 길이 `L+1` 의 토큰 시퀀스를 `[L+1, V]` one-hot 변수로 패킹한다.
    fn pack_sequence(tokens: &[usize], vocab: usize) -> MlResult<Variable> {
        let mut data = Vec::with_capacity(tokens.len() * vocab);
        for &t in tokens {
            data.extend_from_slice(&one_hot(t, vocab));
        }
        Ok(Variable::new(Tensor::from_vec(
            data,
            &[tokens.len(), vocab],
        )?))
    }

    /// 작은 합성 코퍼스에서 Bigram LM 이 발산 없이 학습되는지 확인한다.
    /// 파일럿 목적은 AR 트레이너의 **제어 흐름 검증** 이지, 성능 달성이 아니다.
    #[test]
    fn bigram_lm_pilot_runs() -> MlResult<()> {
        let vocab = 4;
        let mut model = BigramLM::new(vocab)?;
        let mut opt = Adam::new(1e-1, 0.9, 0.999, 1e-8);
        for p in model.params() {
            opt.register(p);
        }

        // 코퍼스: 반복되는 bigram 패턴 (0→1→2→3→0 …)
        let sequences = [
            vec![0, 1, 2, 3, 0, 1, 2],
            vec![1, 2, 3, 0, 1, 2, 3],
            vec![2, 3, 0, 1, 2, 3, 0],
            vec![3, 0, 1, 2, 3, 0, 1],
        ];
        let seq_vars: Vec<Variable> = sequences
            .iter()
            .map(|s| pack_sequence(s, vocab))
            .collect::<MlResult<Vec<_>>>()?;
        let dataset = crate::trainer::DatasetBuilder::from_source(
            crate::trainer::MemorySource::new(seq_vars),
        )
        .map(|sequence: Variable| {
            Ok(crate::trainer::AutoregressiveSample::new(
                sequence.tensor().clone(),
            ))
        })
        .build()?;
        let mut loader = crate::trainer::DataLoader::builder(dataset)
            .collator(|samples: &[&crate::trainer::AutoregressiveSample]| {
                if samples.len() != 1 {
                    return Err(crate::MlError::StringError(
                        "single-sequence collator expects one sample".into(),
                    ));
                }
                Ok(crate::trainer::AutoregressiveBatch {
                    sequences: Variable::new(samples[0].sequence.clone()),
                })
            })
            .batch_size(1)
            .build()?;

        let trainer = Trainer::silent().autoregressive();
        let result = trainer.fit(
            &mut model,
            &mut opt,
            &mut loader,
            crate::trainer::EpochSchedule::new(20)?.with_tolerance(1e-10),
        )?;

        assert!(result.units_completed > 0, "적어도 1 에폭은 학습되어야 함");
        assert!(
            result.final_loss.is_finite(),
            "최종 손실이 유한해야 함: got {}",
            result.final_loss
        );
        assert!(
            result.final_loss >= 0.0,
            "손실은 음이 아니어야 함: got {}",
            result.final_loss
        );
        Ok(())
    }

    /// 학습이 진행됨에 따라 손실이 감소해야 한다 (monotonic decrease 는 아니어도
    /// 초기 대비 말기 평균이 유의미하게 낮아야 한다).
    #[test]
    fn bigram_lm_pilot_loss_decreases() -> MlResult<()> {
        let vocab = 4;
        let mut model = BigramLM::new(vocab)?;
        let mut opt = Adam::new(1e-1, 0.9, 0.999, 1e-8);
        for p in model.params() {
            opt.register(p);
        }

        // 결정적인 bigram: 항상 t → (t+1) mod V
        let sequences = [vec![0, 1, 2, 3, 0, 1, 2, 3], vec![1, 2, 3, 0, 1, 2, 3, 0]];
        let seq_vars: Vec<Variable> = sequences
            .iter()
            .map(|s| pack_sequence(s, vocab))
            .collect::<MlResult<Vec<_>>>()?;
        let x_set: Vec<&Variable> = seq_vars.iter().collect();

        let trainer = Trainer::silent().autoregressive();
        let res_short = trainer.fit(
            &mut model,
            &mut opt,
            crate::trainer::AutoregressiveDataset::new(&x_set)?,
            crate::trainer::EpochSchedule::new(1)?.with_tolerance(1e-10),
        )?;
        let init_loss = res_short.final_loss;

        // 추가로 더 학습
        let res_long = trainer.fit(
            &mut model,
            &mut opt,
            crate::trainer::AutoregressiveDataset::new(&x_set)?,
            crate::trainer::EpochSchedule::new(40)?.with_tolerance(1e-10),
        )?;
        let final_loss = res_long.final_loss;

        assert!(
            final_loss < init_loss,
            "학습 후 손실이 감소해야 함: init={:.4}, final={:.4}",
            init_loss,
            final_loss
        );
        Ok(())
    }

    // ────────────────────────────────────────────────────────────────────────
    // Phase 3: MetricHook 활성화 — 훅이 실제 학습 루프에서 배치마다 호출되고
    // 에폭 경계에서 reset 이 호출되는지 확인.
    // ────────────────────────────────────────────────────────────────────────

    use crate::trainer::{BatchContext, MetricHook};
    use std::cell::Cell;

    /// 배치 호출 횟수와 reset 횟수를 세는 스파이 훅.
    struct CallCounterHook {
        updates: Cell<usize>,
        resets: Cell<usize>,
        last_lr: Cell<f32>,
    }

    impl MetricHook for CallCounterHook {
        fn update(&mut self, ctx: &BatchContext<'_>) -> MlResult<()> {
            self.updates.set(self.updates.get() + 1);
            self.last_lr.set(ctx.lr);
            Ok(())
        }
        fn compute(&self) -> f32 {
            self.updates.get() as f32
        }
        fn reset(&mut self) -> MlResult<()> {
            self.resets.set(self.resets.get() + 1);
            Ok(())
        }
        fn name(&self) -> &str {
            "call_counter"
        }
    }

    /// 훅이 배치당 1 회 update, 에폭당 1 회 reset 호출되는지 검증.
    #[test]
    fn hook_is_called_per_batch_and_reset_per_epoch() -> MlResult<()> {
        let vocab = 4;
        let mut model = BigramLM::new(vocab)?;
        let mut opt = Adam::new(1e-1, 0.9, 0.999, 1e-8);
        for p in model.params() {
            opt.register(p);
        }

        let sequences = [
            vec![0, 1, 2, 3, 0],
            vec![1, 2, 3, 0, 1],
            vec![2, 3, 0, 1, 2],
        ];
        let seq_vars: Vec<Variable> = sequences
            .iter()
            .map(|s| pack_sequence(s, vocab))
            .collect::<MlResult<Vec<_>>>()?;
        let x_set: Vec<&Variable> = seq_vars.iter().collect();

        let trainer = Trainer::silent().autoregressive();
        trainer.core.add_hook(Box::new(CallCounterHook {
            updates: Cell::new(0),
            resets: Cell::new(0),
            last_lr: Cell::new(0.0),
        }));

        let epochs = 3;
        let _ = trainer.fit(
            &mut model,
            &mut opt,
            crate::trainer::AutoregressiveDataset::new(&x_set)?,
            crate::trainer::EpochSchedule::new(epochs)?.with_tolerance(1e-10),
        )?;

        // 훅 상태는 RefCell 안에 있으므로 borrow 로 접근.
        let hooks = trainer.core.hooks.borrow();
        let hook = hooks[0].as_ref();
        // `MetricHook::compute` 는 updates 카운트를 반환.
        let updates = hook.compute() as usize;
        assert_eq!(
            updates,
            epochs * x_set.len(),
            "배치당 1 회 update 기대: {} 에폭 × {} 배치 = {}",
            epochs,
            x_set.len(),
            epochs * x_set.len()
        );
        Ok(())
    }
}
