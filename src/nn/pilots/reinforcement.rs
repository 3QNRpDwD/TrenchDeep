//! Explicit-context two-armed-bandit and linear-policy pilot.

use crate::nn::{Layer, Linear, Parameter};
use crate::tensor::{TensorBuffer};
use crate::trainer::{
    Environment, RLModel, StepResult, TrainableModel,
};
use crate::{ContextId, Tensor, Variable, ExecutionContext, MlResult};

pub struct TwoArmedBandit {
    pub mean_rewards: [f32; 2],
    pub noise_scale: f32,
}

impl Default for TwoArmedBandit {
    fn default() -> Self {
        Self { mean_rewards: [0.2, 0.8], noise_scale: 0.1 }
    }
}

impl Environment for TwoArmedBandit {
    fn reset(&mut self) -> MlResult<TensorBuffer> {
        TensorBuffer::from_vec(vec![1.0], &[1, 1])
    }

    fn step(&mut self, action: usize) -> MlResult<StepResult> {
        let reward = self.mean_rewards.get(action).copied().unwrap_or(0.0)
            + (rand::random::<f32>() - 0.5) * 2.0 * self.noise_scale;
        Ok(StepResult {
            next_observation: TensorBuffer::from_vec(vec![1.0], &[1, 1])?,
            reward,
            done: true,
        })
    }

    fn num_actions(&self) -> usize { 2 }
    fn observation_shape(&self) -> Vec<usize> { vec![1, 1] }
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

    pub fn linear(&self) -> &Linear { &self.linear }
}

impl TrainableModel for LinearPolicy {
    fn context_id(&self) -> ContextId { self.context.id() }
    fn parameters(&self) -> Vec<&Parameter> { self.linear.parameters() }
}

impl RLModel for LinearPolicy {
    fn policy_logits(&mut self, observation: &Variable) -> MlResult<Variable> {
        self.linear.apply(observation)
    }

    fn predict_policy_raw(&mut self, observation: &Tensor) -> MlResult<TensorBuffer> {
        let output = self.linear.predict(observation)?;
        TensorBuffer::from_vec(output.to_vec()?, &output.shape()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{Adam, Optimizer};
    use crate::trainer::{RLTrainer, EpisodeSchedule};

    #[test]
    fn bandit_policy_pilot_trains_end_to_end() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = LinearPolicy::new(&context, 1, 2)?;
        let mut environment = TwoArmedBandit {
            mean_rewards: [-1.0, 1.0],
            noise_scale: 0.0,
        };
        let mut optimizer = Adam::new(&context, 0.05, 0.9, 0.999, 1e-8)?;
        optimizer.register_all(&model.parameters())?;
        let result = RLTrainer::silent(&context)
            .with_seed(19)
            .with_baseline(false)
            .fit(
                &mut model,
                &mut environment,
                &mut optimizer,
                EpisodeSchedule::new(20, 1)?,
            )?;
        assert!(result.final_loss.is_finite());
        assert_eq!(result.units_completed, 20);
        assert_eq!(context.graph_stats()?.graph_nodes, 0);
        Ok(())
    }
}
