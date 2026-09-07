//! Explicit-context Pi-model style semi-supervised pilot.

use crate::loss::Reduction;
use crate::nn::{Layer, Linear, Parameter};
use crate::trainer::{SemiSupervisedModel, TrainableModel};
use crate::{ContextId, ExecutionContext, MlResult, Tensor, Variable};

#[derive(Debug)]
pub struct PiClassifier {
    context: ExecutionContext,
    linear: Linear,
    noise_scale: f32,
}

impl PiClassifier {
    pub fn new(
        context: &ExecutionContext,
        inputs: usize,
        outputs: usize,
        noise_scale: f32,
    ) -> MlResult<Self> {
        if !noise_scale.is_finite() || noise_scale < 0.0 {
            return Err(crate::TensorError::InvalidOperation {
                op: "pi_model",
                reason: "noise_scale must be finite and non-negative".into(),
            }
            .into());
        }
        Ok(Self {
            context: context.clone(),
            linear: Linear::new(context, inputs, outputs, "pi_linear")?,
            noise_scale,
        })
    }

    fn noisy(&self, input: &Variable) -> MlResult<Variable> {
        let shape = input.tensor().shape()?;
        let noise = self
            .context
            .model_uniform(input.tensor().numel()?, self.noise_scale)?;
        let noise = self.context.input(noise, &shape)?;
        input.add(noise.tensor())
    }

    pub fn predict(&self, input: &Tensor) -> MlResult<Tensor> {
        self.linear.predict(input)
    }
}

impl TrainableModel for PiClassifier {
    fn context_id(&self) -> ContextId {
        self.context.id()
    }
    fn parameters(&self) -> Vec<&Parameter> {
        self.linear.parameters()
    }
}

impl SemiSupervisedModel for PiClassifier {
    fn forward_loss(
        &mut self,
        labeled_input: &Variable,
        labeled_target: &Tensor,
        unlabeled_input: &Variable,
        lambda: f32,
    ) -> MlResult<(Variable, Variable)> {
        let first = self.noisy(unlabeled_input)?;
        let second = self.noisy(unlabeled_input)?;
        self.forward_loss_with_augmentations(labeled_input, labeled_target, &first, &second, lambda)
    }
}

impl PiClassifier {
    /// Train on two explicitly supplied augmented views (for reproducible comparisons).
    pub fn forward_loss_with_augmentations(
        &self,
        labeled_input: &Variable,
        labeled_target: &Tensor,
        first_augmentation: &Variable,
        second_augmentation: &Variable,
        lambda: f32,
    ) -> MlResult<(Variable, Variable)> {
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(crate::TensorError::InvalidOperation {
                op: "pi_model",
                reason: "lambda must be finite and nonnegative".into(),
            }
            .into());
        }
        if first_augmentation.tensor().shape()? != second_augmentation.tensor().shape()?
            || first_augmentation.tensor().numel()? == 0
        {
            return Err(crate::TensorError::InvalidOperation {
                op: "pi_model",
                reason: "augmented views must have identical, nonempty shapes".into(),
            }
            .into());
        }
        let labeled_logits = self.linear.apply(labeled_input)?;
        let supervised = labeled_logits.softmax_cross_entropy(labeled_target, Reduction::Mean)?;

        let first = self.linear.apply(first_augmentation)?;
        let second = self.linear.apply(second_augmentation)?;
        let difference = first.sub(second.tensor())?;
        let squared = difference.square()?;
        let sum = squared.sum()?;
        let count = squared.tensor().numel()?;
        let mean_scale = self.context.input(vec![1.0 / count as f32], &[])?;
        let consistency = sum.mul(mean_scale.tensor())?;
        let lambda = self.context.input(vec![lambda], &[])?;
        let weighted = consistency.mul(lambda.tensor())?;
        let total = supervised.add(weighted.tensor())?;
        Ok((labeled_logits, total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{Adam, Optimizer};
    use crate::trainer::{
        ConsistencyRamp, EpochSchedule, SemiSupervisedDataset, SemiSupervisedTrainer,
    };

    #[test]
    fn supplied_views_reject_broadcasting_and_empty_batches_before_recording() -> MlResult<()> {
        let context = ExecutionContext::new();
        let model = PiClassifier::new(&context, 2, 2, 0.1)?;
        let input = context.input(vec![1.0, 2.0], &[1, 2])?;
        let target = context.tensor(vec![1.0, 0.0], &[1, 2])?;
        let other = context.input(vec![1.0; 4], &[2, 2])?;
        let empty = context.input(vec![], &[0, 2])?;
        for (first, second) in [(&input, &other), (&empty, &empty)] {
            assert!(matches!(
                model.forward_loss_with_augmentations(&input, &target, first, second, 0.4),
                Err(crate::MlError::TensorError(
                    crate::TensorError::InvalidOperation { op: "pi_model", .. }
                ))
            ));
            assert_eq!(context.graph_stats()?.graph_nodes, 0);
        }
        Ok(())
    }

    #[test]
    fn pi_model_pilot_trains_end_to_end() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = PiClassifier::new(&context, 2, 2, 0.1)?;
        let labeled = [
            context.input(vec![1.0, 1.0], &[1, 2])?,
            context.input(vec![-1.0, -1.0], &[1, 2])?,
        ];
        let targets = [
            context.tensor(vec![1.0, 0.0], &[1, 2])?,
            context.tensor(vec![0.0, 1.0], &[1, 2])?,
        ];
        let unlabeled = [
            context.input(vec![0.9, 1.1], &[1, 2])?,
            context.input(vec![-0.9, -1.1], &[1, 2])?,
        ];
        let labeled_refs = labeled.iter().collect::<Vec<_>>();
        let target_refs = targets.iter().collect::<Vec<_>>();
        let unlabeled_refs = unlabeled.iter().collect::<Vec<_>>();
        let dataset =
            SemiSupervisedDataset::new(&context, &labeled_refs, &target_refs, &unlabeled_refs)?;
        let mut optimizer = Adam::new(&context, 0.02, 0.9, 0.999, 1e-8)?;
        optimizer.register_all(&model.parameters())?;
        let result = SemiSupervisedTrainer::silent(&context)
            .with_ramp(ConsistencyRamp::Sigmoid {
                max_weight: 1.0,
                ramp_epochs: 2,
            })
            .fit(
                &mut model,
                &mut optimizer,
                &dataset,
                EpochSchedule::new(3)?.with_tolerance(0.0),
            )?;
        assert!(result.final_loss.is_finite());
        assert_eq!(context.graph_stats()?.graph_nodes, 0);
        Ok(())
    }
}
