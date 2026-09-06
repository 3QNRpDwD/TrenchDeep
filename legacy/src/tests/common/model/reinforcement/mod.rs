//! 강화학습 파일럿: **2-armed stochastic bandit** + **선형 정책**.
//!
//! P5 의 `RLTrainer` + `Environment` + `RLModel` 인터페이스가 end-to-end 로
//! 동작함을 최소 예제로 증명한다.
//!
//! ## 환경: `TwoArmedBandit`
//!
//! - 상태: 고정 [1.0] (contextless)
//! - 행동 0: 평균 보상 `0.2`
//! - 행동 1: 평균 보상 `0.8`
//! - 보상: 평균 ± 잡음 (`uniform(-0.1, 0.1)`)
//! - 에피소드 길이: 1 스텝
//!
//! 최적 정책은 `action=1` 을 항상 선택. 학습 후 정책이 그쪽으로 치우치는지 확인.
//!
//! ## 정책: `LinearPolicy`
//!
//! `logits = Wx + b`, shape `[1] → [2]`.
//! Softmax 는 트레이너가 롤아웃 단계에서 자체 수행.

use super::*;

use crate::{
    nn::Variable,
    tensor::{
        operators::{Add, Function, Matmul},
        AutogradFunction,
        GlobalTensor,
        Tensor,
        TensorBase,
    },
    trainer::{Environment, StepResult},
    var_with_label,
    MlResult,
};

// ────────────────────────────────────────────────────────────────────────────
// TwoArmedBandit
// ────────────────────────────────────────────────────────────────────────────

/// 2-arm 확률적 밴딧 환경. 최적 행동은 `1` (평균 보상 0.8).
pub struct TwoArmedBandit {
    pub mean_rewards: [f32; 2],
    pub noise_scale:  f32,
}

impl Default for TwoArmedBandit {
    fn default() -> Self {
        Self { mean_rewards: [0.2, 0.8], noise_scale: 0.1 }
    }
}

impl Environment for TwoArmedBandit {
    fn reset(&mut self) -> MlResult<Tensor> {
        Tensor::from_vec(vec![1.0], &[1, 1])
    }

    fn step(&mut self, action: usize) -> MlResult<StepResult> {
        let base  = self.mean_rewards.get(action).copied().unwrap_or(0.0);
        let noise = (rand::random::<f32>() - 0.5) * 2.0 * self.noise_scale;
        Ok(StepResult {
            next_observation: Tensor::from_vec(vec![1.0], &[1, 1])?,
            reward:           base + noise,
            done:             true, // 1-step bandit
        })
    }

    fn num_actions(&self) -> usize { 2 }

    fn observation_shape(&self) -> Vec<usize> { vec![1, 1] }
}

// ────────────────────────────────────────────────────────────────────────────
// LinearPolicy
// ────────────────────────────────────────────────────────────────────────────

/// 선형 정책. 관측치 `[1, 1]` 에서 로짓 `[1, 2]` 를 계산.
pub struct LinearPolicy {
    pub w: Variable,
    pub b: Variable,
}

impl LinearPolicy {
    pub fn new(obs_dim: usize, n_actions: usize) -> MlResult<Self> {
        let w_data: Vec<f32> = (0..obs_dim * n_actions)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.1)
            .collect();
        let w = var_with_label!(
            Tensor::from_vec(w_data, &[obs_dim, n_actions])?,
            "policy_w"
        );
        let b_data: Vec<f32> = vec![0.0; n_actions];
        let b = var_with_label!(
            Tensor::from_vec(b_data, &[n_actions])?,
            "policy_b"
        );
        Ok(Self { w, b })
    }
}

#[cfg(feature = "enableBackward")]
impl crate::trainer::RLModel for LinearPolicy {
    fn policy_logits(&mut self, obs: &Variable) -> MlResult<Variable> {
        let mut matmul = Matmul::new()?;
        let pre = matmul.apply(&[obs, &self.w])?;
        Ok(&pre + &self.b)
    }

    fn predict_policy_raw(&mut self, obs: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let matmul = Matmul::new()?;
        let add    = Add::new()?;
        let pre    = matmul.forward(&[obs, self.w.tensor()])?.remove(0);
        let out    = add.forward(&[&pre, self.b.tensor()])?.remove(0);
        Ok(out)
    }

}

impl crate::trainer::TrainableModel for LinearPolicy {
    fn params(&self) -> Vec<&dyn Parameter> { vec![&self.w, &self.b] }
}
impl crate::trainer::CheckpointableModel for LinearPolicy {}

// ────────────────────────────────────────────────────────────────────────────
// 테스트
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "enableBackward")]
mod tests {
    use super::*;
    use crate::{
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
            crate::trainer::EpisodeSchedule::new(100, 1)?)?;

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
            crate::trainer::EpisodeSchedule::new(10, 1)?)?;

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
}
