//! Explicit-context two-armed-bandit and linear-policy pilot.

use crate::nn::{ContextLayer, ContextLinear, ContextParameter};
use crate::tensor::{GlobalTensor, TensorBase};
use crate::trainer::{
    ContextEnvironment, ContextRLModel, ContextStepResult, ContextTrainableModel,
};
use crate::{ContextId, ContextTensor, ContextVariable, ExecutionContext, MlResult};

pub struct ContextTwoArmedBandit {
    pub mean_rewards: [f32; 2],
    pub noise_scale: f32,
}

impl Default for ContextTwoArmedBandit {
    fn default() -> Self {
        Self { mean_rewards: [0.2, 0.8], noise_scale: 0.1 }
    }
}

impl ContextEnvironment for ContextTwoArmedBandit {
    fn reset(&mut self) -> MlResult<GlobalTensor<f32>> {
        GlobalTensor::from_vec(vec![1.0], &[1, 1])
    }

    fn step(&mut self, action: usize) -> MlResult<ContextStepResult> {
        let reward = self.mean_rewards.get(action).copied().unwrap_or(0.0)
            + (rand::random::<f32>() - 0.5) * 2.0 * self.noise_scale;
        Ok(ContextStepResult {
            next_observation: GlobalTensor::from_vec(vec![1.0], &[1, 1])?,
            reward,
            done: true,
        })
    }

    fn num_actions(&self) -> usize { 2 }
    fn observation_shape(&self) -> Vec<usize> { vec![1, 1] }
}

#[derive(Debug)]
pub struct ContextLinearPolicy {
    context: ExecutionContext,
    linear: ContextLinear,
}

impl ContextLinearPolicy {
    pub fn new(context: &ExecutionContext, observations: usize, actions: usize) -> MlResult<Self> {
        Ok(Self {
            context: context.clone(),
            linear: ContextLinear::new(context, observations, actions, "policy")?,
        })
    }

    pub fn linear(&self) -> &ContextLinear { &self.linear }
}

impl ContextTrainableModel for ContextLinearPolicy {
    fn context_id(&self) -> ContextId { self.context.id() }
    fn parameters(&self) -> Vec<&ContextParameter> { self.linear.parameters() }
}

impl ContextRLModel for ContextLinearPolicy {
    fn policy_logits(&mut self, observation: &ContextVariable) -> MlResult<ContextVariable> {
        self.linear.apply(observation)
    }

    fn predict_policy_raw(&mut self, observation: &ContextTensor) -> MlResult<GlobalTensor<f32>> {
        let output = self.linear.predict(observation)?;
        GlobalTensor::from_vec(output.to_vec()?, &output.shape()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{ContextAdam, ContextOptimizer};
    use crate::trainer::{ContextRLTrainer, EpisodeSchedule};

    #[test]
    fn bandit_policy_pilot_trains_end_to_end() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = ContextLinearPolicy::new(&context, 1, 2)?;
        let mut environment = ContextTwoArmedBandit {
            mean_rewards: [-1.0, 1.0],
            noise_scale: 0.0,
        };
        let mut optimizer = ContextAdam::new(&context, 0.05, 0.9, 0.999, 1e-8)?;
        optimizer.register_all(&model.parameters())?;
        let result = ContextRLTrainer::silent(&context)
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
