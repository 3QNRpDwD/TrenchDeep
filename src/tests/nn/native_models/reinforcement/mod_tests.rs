use super::*;
use crate::legacy::{
    optimizer::{Adam, Optimizer},
    trainer::{RLModel, Trainer, TrainableModel},
};

/// REINFORCE 가 발산 없이 돌고, 결과적으로 최적 행동(1)의 로짓이 더 커지는지 검증.
/// 파일럿 목적은 트레이너 인터페이스 검증이므로 엄격한 통계 테스트는 하지 않는다.
#[test]
fn reinforce_bandit_pilot_runs() -> MlResult<()> {
    let mut policy = LinearPolicy::new(1, 2)?;
    let mut env    = TwoArmedBandit::default();
    let mut opt    = Adam::new(1e-1, 0.9, 0.999, 1e-8);
    for p in policy.params() {
        opt.register(p);
    }

    let trainer = Trainer::silent().reinforcement()
        .with_gamma(1.0)        // 1-step bandit, 할인 불필요
        .with_baseline(true);

    let result = trainer.fit(&mut policy, &mut env, &mut opt,
        crate::legacy::trainer::EpisodeSchedule::new(100, 1)?)?;

    assert_eq!(result.units_completed, 100);
    assert!(
        result.final_loss.is_finite(),
        "최종 손실이 유한해야 함: got {}", result.final_loss
    );

    // 학습 후 정책이 최적 행동(1) 쪽으로 치우쳐야 함.
    let obs = Tensor::from_vec(vec![1.0], &[1, 1])?;
    let logits = policy.predict_policy_raw(&obs)?;
    let data   = logits.data.as_slice();
    assert_eq!(data.len(), 2);
    assert!(
        data[1] > data[0],
        "학습 후 action 1 의 로짓이 더 커야 함: logits = {:?}", data
    );

    Ok(())
}

/// `use_baseline=false` 일 때도 학습이 발산 없이 동작하는지 확인.
#[test]
fn reinforce_bandit_no_baseline() -> MlResult<()> {
    let mut policy = LinearPolicy::new(1, 2)?;
    let mut env    = TwoArmedBandit::default();
    let mut opt    = Adam::new(5e-2, 0.9, 0.999, 1e-8);
    for p in policy.params() {
        opt.register(p);
    }

    let trainer = Trainer::silent().reinforcement()
        .with_gamma(1.0)
        .with_baseline(false);

    let result = trainer.fit(&mut policy, &mut env, &mut opt,
        crate::legacy::trainer::EpisodeSchedule::new(10, 1)?)?;

    assert_eq!(result.units_completed, 10);
    assert!(result.final_loss.is_finite());
    Ok(())
}

/// `with_gamma` 가 의도대로 반영되는지 확인.
#[test]
fn rl_trainer_config_methods() {
    let t = Trainer::silent().reinforcement().with_gamma(0.5).with_baseline(false);
    assert_eq!(t.gamma, 0.5);
    assert!(!t.use_baseline);
}
