use super::*;
use crate::legacy::{
    optimizer::{Adam, Optimizer},
    trainer::{ConsistencyRamp, SemiSupervisedModel, Trainer, TrainableModel},
};

/// labeled 4 개 + unlabeled 8 개로 Pi-model 학습이 **발산 없이** 동작하는지 확인.
/// 파일럿의 목적은 트레이너 인터페이스 검증이지 정확도 달성이 아니다.
#[test]
fn pi_model_pilot_runs() -> MlResult<()> {
    // ── 모델 + 옵티마이저 ────────────────────────────────────────────
    let mut model = PiToyClassifier::new(2, 2, 0.1)?;
    let mut opt   = Adam::new(1e-2, 0.9, 0.999, 1e-8);
    for p in model.params() {
        opt.register(p);
    }

    // ── 데이터 (2D 이진 분류: 사분면 기반) ──────────────────────────
    // class 0 : (+1, +1) 근처
    // class 1 : (-1, -1) 근처
    let labeled = vec![
        ([ 1.0,  1.0], [1.0, 0.0]),
        ([ 0.9,  1.1], [1.0, 0.0]),
        ([-1.0, -1.0], [0.0, 1.0]),
        ([-1.1, -0.9], [0.0, 1.0]),
    ];
    // unlabeled: 두 군집 주변에 흩뿌린 8 개 점
    let unlabeled = vec![
        [ 1.2,  0.8], [ 0.7,  1.3], [ 1.1,  1.0], [ 0.8,  0.9],
        [-1.2, -0.8], [-0.7, -1.3], [-1.1, -1.0], [-0.8, -0.9],
    ];
    let labeled_dataset = crate::legacy::trainer::DatasetBuilder::from_source(
        crate::legacy::trainer::MemorySource::new(labeled),
    )
    .map(|(input, target): ([f32; 2], [f32; 2])| Ok(crate::legacy::trainer::SupervisedSample::new(
        Tensor::from_vec(input.to_vec(), &[2])?,
        Tensor::from_vec(target.to_vec(), &[2])?,
    )))
    .build()?;
    let unlabeled_dataset = crate::legacy::trainer::DatasetBuilder::from_source(
        crate::legacy::trainer::MemorySource::new(unlabeled),
    )
    .map(|input: [f32; 2]| Ok(crate::legacy::trainer::UnsupervisedSample::new(
        Tensor::from_vec(input.to_vec(), &[2])?,
    )))
    .build()?;
    let mut loader = crate::legacy::trainer::SemiSupervisedDataLoader::builder(
        labeled_dataset,
        unlabeled_dataset,
    )
    .labeled_collator(crate::legacy::trainer::SupervisedStackCollator::new())
    .unlabeled_collator(crate::legacy::trainer::UnsupervisedStackCollator::new())
    .labeled_batch_size(2)
    .unlabeled_batch_size(4)
    .build()?;

    // ── 트레이너: silent + 짧은 램프 ────────────────────────────────
    let trainer = Trainer::silent().semi_supervised()
        .with_ramp(ConsistencyRamp::Sigmoid { max_weight: 1.0, ramp_epochs: 5 });

    let result = trainer.fit(&mut model, &mut opt,
        &mut loader,
        crate::legacy::trainer::EpochSchedule::new(8)?.with_tolerance(1e-10))?;

    // ── 검증 ────────────────────────────────────────────────────────
    assert!(result.units_completed > 0, "적어도 1 에폭은 학습되어야 함");
    assert!(
        result.final_loss.is_finite(),
        "최종 손실이 유한해야 함: got {}", result.final_loss
    );
    assert!(
        result.final_loss >= 0.0,
        "손실은 음이 아니어야 함: got {}", result.final_loss
    );

    Ok(())
}

/// `ConsistencyRamp::Constant(0.0)` 이면 일관성 손실 기여가 0 이어야 하고,
/// 파일럿이 그 경우도 정상 동작하는지 확인.
#[test]
fn pi_model_pilot_zero_ramp_is_supervised_only() -> MlResult<()> {
    let mut model = PiToyClassifier::new(2, 2, 0.1)?;
    let mut opt   = Adam::new(1e-2, 0.9, 0.999, 1e-8);
    for p in model.params() {
        opt.register(p);
    }

    let x_l = Variable::new(Tensor::from_vec(vec![1.0, 1.0], &[1, 2])?);
    let t_l = Variable::new(Tensor::from_vec(vec![1.0, 0.0], &[1, 2])?);
    let x_u = Variable::new(Tensor::from_vec(vec![0.5, 0.5], &[1, 2])?);

    let x_l_slice = [&x_l];
    let t_l_slice = [&t_l];
    let x_u_slice = [&x_u];

    let trainer = Trainer::silent().semi_supervised()
        .with_ramp(ConsistencyRamp::Constant(0.0));

    let result = trainer.fit(&mut model, &mut opt,
        crate::legacy::trainer::SemiSupervisedDataset::new(&x_l_slice, &t_l_slice, &x_u_slice)?,
        crate::legacy::trainer::EpochSchedule::new(3)?.with_tolerance(1e-10))?;

    assert!(result.final_loss.is_finite());
    Ok(())
}
