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

use crate::legacy::{
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
impl crate::legacy::trainer::RLModel for LinearPolicy {
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

impl crate::legacy::trainer::TrainableModel for LinearPolicy {
    fn params(&self) -> Vec<&dyn Parameter> { vec![&self.w, &self.b] }
}
impl crate::legacy::trainer::CheckpointableModel for LinearPolicy {}

// ────────────────────────────────────────────────────────────────────────────
// 테스트
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "enableBackward")]
#[path = "../../../tests/nn/native_models/reinforcement/mod_tests.rs"]
mod tests;
