//! Explicit-context Pi-model style semi-supervised pilot.

use crate::loss::Reduction;
use crate::nn::{ContextLayer, ContextLinear, ContextParameter};
use crate::trainer::{ContextSemiSupervisedModel, ContextTrainableModel};
use crate::{ContextId, ContextTensor, ContextVariable, ExecutionContext, MlResult};

#[derive(Debug)]
pub struct ContextPiClassifier {
    context: ExecutionContext,
    linear: ContextLinear,
    noise_scale: f32,
}

impl ContextPiClassifier {
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
            }.into());
        }
        Ok(Self {
            context: context.clone(),
            linear: ContextLinear::new(context, inputs, outputs, "pi_linear")?,
            noise_scale,
        })
    }

    fn noisy(&self, input: &ContextVariable) -> MlResult<ContextVariable> {
        let shape = input.tensor().shape()?;
        let noise = (0..input.tensor().numel()?)
            .map(|_| (rand::random::<f32>() - 0.5) * 2.0 * self.noise_scale)
            .collect();
        let noise = self.context.input(noise, &shape)?;
        input.add(noise.tensor())
    }

    pub fn predict(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.linear.predict(input)
    }
}

impl ContextTrainableModel for ContextPiClassifier {
    fn context_id(&self) -> ContextId { self.context.id() }
    fn parameters(&self) -> Vec<&ContextParameter> { self.linear.parameters() }
}

impl ContextSemiSupervisedModel for ContextPiClassifier {
    fn forward_loss(
        &mut self,
        labeled_input: &ContextVariable,
        labeled_target: &ContextTensor,
        unlabeled_input: &ContextVariable,
        lambda: f32,
    ) -> MlResult<(ContextVariable, ContextVariable)> {
        let labeled_logits = self.linear.apply(labeled_input)?;
        let supervised = labeled_logits.softmax_cross_entropy(labeled_target, Reduction::Mean)?;

        let first = self.linear.apply(&self.noisy(unlabeled_input)?)?;
        let second = self.linear.apply(&self.noisy(unlabeled_input)?)?;
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
    use crate::optimizer::{ContextAdam, ContextOptimizer};
    use crate::trainer::{
        ConsistencyRamp, ContextSemiSupervisedDataset, ContextSemiSupervisedTrainer, EpochSchedule,
    };

    #[test]
    fn pi_model_pilot_trains_end_to_end() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = ContextPiClassifier::new(&context, 2, 2, 0.1)?;
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
        let dataset = ContextSemiSupervisedDataset::new(
            &context, &labeled_refs, &target_refs, &unlabeled_refs,
        )?;
        let mut optimizer = ContextAdam::new(&context, 0.02, 0.9, 0.999, 1e-8)?;
        optimizer.register_all(&model.parameters())?;
        let result = ContextSemiSupervisedTrainer::silent(&context)
            .with_ramp(ConsistencyRamp::Sigmoid { max_weight: 1.0, ramp_epochs: 2 })
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
