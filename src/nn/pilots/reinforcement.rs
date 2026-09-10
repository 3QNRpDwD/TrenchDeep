//! Explicit-context two-armed-bandit and linear-policy pilot.

use crate::nn::{Layer, Linear, Parameter};
use crate::tensor::TensorBuffer;
use crate::trainer::{Environment, RLModel, StepResult, TrainableModel};
use crate::{ContextId, ExecutionContext, MlResult, Tensor, Variable};
use rand::{Rng, SeedableRng, rngs::StdRng};

pub struct TwoArmedBandit {
    pub mean_rewards: [f32; 2],
    pub noise_scale: f32,
    rng: StdRng,
}

impl Default for TwoArmedBandit {
    fn default() -> Self {
        Self {
            mean_rewards: [0.2, 0.8],
            noise_scale: 0.1,
            rng: StdRng::seed_from_u64(0),
        }
    }
}
impl TwoArmedBandit {
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = StdRng::seed_from_u64(seed);
        self
    }
}

impl Environment for TwoArmedBandit {
    fn reset(&mut self) -> MlResult<TensorBuffer> {
        TensorBuffer::from_vec(vec![1.0], &[1, 1])
    }

    fn step(&mut self, action: usize) -> MlResult<StepResult> {
        let mean = self.mean_rewards.get(action).copied().ok_or_else(|| {
            crate::TensorError::InvalidOperation {
                op: "bandit",
                reason: "invalid action".into(),
            }
        })?;
        let reward = mean + (self.rng.random::<f32>() - 0.5) * 2.0 * self.noise_scale;
        Ok(StepResult {
            next_observation: TensorBuffer::from_vec(vec![1.0], &[1, 1])?,
            reward,
            done: true,
        })
    }

    fn num_actions(&self) -> usize {
        2
    }
    fn observation_shape(&self) -> Vec<usize> {
        vec![1, 1]
    }
}

#[derive(Debug)]
pub struct LinearPolicy {
    context: ExecutionContext,
    linear: Linear,
}

impl LinearPolicy {
    pub fn new(context: &ExecutionContext, observations: usize, actions: usize) -> MlResult<Self> {
        Ok(Self {
            context: context.clone(),
            linear: Linear::new(context, observations, actions, "policy")?,
        })
    }

    pub fn linear(&self) -> &Linear {
        &self.linear
    }
}

impl TrainableModel for LinearPolicy {
    fn context_id(&self) -> ContextId {
        self.context.id()
    }
    fn parameters(&self) -> Vec<&Parameter> {
        self.linear.parameters()
    }
}

impl RLModel for LinearPolicy {
    fn policy_logits(&mut self, observation: &Variable) -> MlResult<Variable> {
        self.linear.apply(observation)
    }
}

impl crate::trainer::CheckpointableModel for LinearPolicy {
    fn save_checkpoint(&self, path: &std::path::Path) -> MlResult<()> {
        let path = path
            .to_str()
            .ok_or_else(|| crate::MlError::StringError("invalid checkpoint path".into()))?;
        crate::nn::ModelState::new(vec![self.linear.save_state()?]).save(path)
    }
    fn load_checkpoint(&mut self, path: &std::path::Path) -> MlResult<()> {
        let path = path
            .to_str()
            .ok_or_else(|| crate::MlError::StringError("invalid checkpoint path".into()))?;
        let state = crate::nn::ModelState::load(path)?;
        if state.layers.len() != 1 {
            return Err(crate::MlError::StringError(
                "invalid policy checkpoint".into(),
            ));
        }
        self.linear.load_state(&state.layers[0])
    }
}
