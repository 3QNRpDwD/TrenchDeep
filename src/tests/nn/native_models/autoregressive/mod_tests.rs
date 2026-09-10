use super::*;
use crate::legacy::{
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
    let dataset = crate::legacy::trainer::DatasetBuilder::from_source(
        crate::legacy::trainer::MemorySource::new(seq_vars),
    )
    .map(|sequence: Variable| {
        Ok(crate::legacy::trainer::AutoregressiveSample::new(
            sequence.tensor().clone(),
        ))
    })
    .build()?;
    let mut loader = crate::legacy::trainer::DataLoader::builder(dataset)
        .collator(|samples: &[&crate::legacy::trainer::AutoregressiveSample]| {
            if samples.len() != 1 {
                return Err(crate::legacy::MlError::StringError(
                    "single-sequence collator expects one sample".into(),
                ));
            }
            Ok(crate::legacy::trainer::AutoregressiveBatch {
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
        crate::legacy::trainer::EpochSchedule::new(20)?.with_tolerance(1e-10),
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
        crate::legacy::trainer::AutoregressiveDataset::new(&x_set)?,
        crate::legacy::trainer::EpochSchedule::new(1)?.with_tolerance(1e-10),
    )?;
    let init_loss = res_short.final_loss;

    // 추가로 더 학습
    let res_long = trainer.fit(
        &mut model,
        &mut opt,
        crate::legacy::trainer::AutoregressiveDataset::new(&x_set)?,
        crate::legacy::trainer::EpochSchedule::new(40)?.with_tolerance(1e-10),
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

use crate::legacy::trainer::{BatchContext, MetricHook};
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
        crate::legacy::trainer::AutoregressiveDataset::new(&x_set)?,
        crate::legacy::trainer::EpochSchedule::new(epochs)?.with_tolerance(1e-10),
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
