use super::*;
use crate::nn::PiClassifier;

#[derive(Debug, Clone, Copy)]
pub struct SemiSupervised<L> { loss: L, noise_scale: f32 }
fn invalid(reason: &str) -> MlError {
    crate::TensorError::InvalidOperation { op: "pi_model", reason: reason.into() }.into()
}
impl<L> SemiSupervised<L> {
    pub fn new(loss: L, noise_scale: f32) -> MlResult<Self> {
        if !noise_scale.is_finite() || noise_scale < 0.0 {
            return Err(invalid("noise_scale must be finite and non-negative"));
        }
        Ok(Self { loss, noise_scale })
    }
    fn noise(&self, input: &Variable) -> MlResult<Tensor> {
        let ctx = input.tensor().execution_context()?;
        ctx.tensor(ctx.model_uniform(input.tensor().numel()?, self.noise_scale)?, &input.tensor().shape()?)
    }
}
impl<L: Loss> SemiSupervised<L> {
    /// Explicit views for custom augmentation and reproducible comparisons.
    pub fn forward_loss_with_augmentations(&self, model: &PiClassifier, labeled: &Variable, target: &Tensor, first: &Variable, second: &Variable, lambda: f32) -> MlResult<(Variable, Variable)> {
        if !lambda.is_finite() || lambda < 0.0 { return Err(invalid("lambda must be finite and nonnegative")); }
        let ctx = labeled.tensor().execution_context()?;
        self.forward_with_lambda(model, labeled, target, first, second, &ctx.tensor(vec![lambda], &[])?)
    }
    fn forward_with_lambda(&self, model: &PiClassifier, labeled: &Variable, target: &Tensor, first: &Variable, second: &Variable, lambda: &Tensor) -> MlResult<(Variable, Variable)> {
        if first.tensor().shape()? != second.tensor().shape()? || first.tensor().numel()? == 0 {
            return Err(invalid("augmented views must have identical, nonempty shapes"));
        }
        let ctx = labeled.tensor().execution_context()?;
        let prediction = model.forward(labeled)?;
        let supervised = self.loss.compute(&ctx, &prediction, target)?;
        let first = model.forward(first)?;
        let second = model.forward(second)?;
        // Both branches differentiate. An MSE operation would stop the second branch.
        let difference = first.sub(second.tensor())?;
        let squared = difference.square()?;
        let sum = squared.sum()?;
        let mean_scale = ctx.input(vec![1.0 / squared.tensor().numel()? as f32], &[])?;
        let consistency = sum.mul(mean_scale.tensor())?;
        let weighted = consistency.mul(lambda)?;
        Ok((prediction, supervised.add(weighted.tensor())?))
    }
}
impl<L: Loss> Objective<PiClassifier> for SemiSupervised<L> {
    type Batch = SemiSupervisedBatch;
    const PARADIGM: &'static str = "semi_supervised";
    fn forward_batch(&self, model: &mut PiClassifier, batch: &Self::Batch, step: &TrainingStepContext) -> MlResult<TrainingOutput> {
        let weight = sample_count(batch.labeled_inputs.tensor())?;
        let lambda = step.lambda.ok_or_else(|| invalid("semi-supervised step requires lambda"))?;
        let first = batch.unlabeled_inputs.add(&self.noise(&batch.unlabeled_inputs)?)?;
        let second = batch.unlabeled_inputs.add(&self.noise(&batch.unlabeled_inputs)?)?;
        let (prediction, loss) = self.forward_loss_with_augmentations(model, &batch.labeled_inputs, &batch.labeled_targets, &first, &second, lambda)?;
        Ok(TrainingOutput { loss, prediction: Some(prediction), target: Some(batch.labeled_targets.clone()), weight, tokens: None, lambda: Some(lambda) })
    }
}
impl<L: Loss> PreparedObjective<PiClassifier> for SemiSupervised<L> {
    fn execution_batch(&self, _: &mut PiClassifier, batch: &Self::Batch, step: &TrainingStepContext) -> MlResult<PreparedBatch> {
        let lambda = step.lambda.ok_or_else(|| invalid("semi-supervised step requires lambda"))?;
        if !lambda.is_finite() || lambda < 0.0 { return Err(invalid("lambda must be finite and nonnegative")); }
        let ctx = batch.labeled_inputs.tensor().execution_context()?;
        let inputs = ExecutionInputs::new("semi_supervised")
            .with("labeled", batch.labeled_inputs.tensor().clone())?
            .with("target", batch.labeled_targets.clone())?
            .with("unlabeled", batch.unlabeled_inputs.tensor().clone())?
            .with("first_noise", self.noise(&batch.unlabeled_inputs)?)?
            .with("second_noise", self.noise(&batch.unlabeled_inputs)?)?
            .with("lambda", ctx.tensor(vec![lambda], &[])?)?;
        let mut result = PreparedBatch::new(inputs, sample_count(batch.labeled_inputs.tensor())?);
        result.target = Some(batch.labeled_targets.clone());
        result.lambda = Some(lambda);
        Ok(result)
    }
    fn forward_inputs(&self, model: &PiClassifier, inputs: &ExecutionInputs) -> MlResult<ModelOutput> {
        let unlabeled = inputs.get("unlabeled")?.as_variable()?;
        let first = unlabeled.add(inputs.get("first_noise")?)?;
        let second = unlabeled.add(inputs.get("second_noise")?)?;
        let (prediction, loss) = self.forward_with_lambda(model, &inputs.get("labeled")?.as_variable()?, inputs.get("target")?, &first, &second, inputs.get("lambda")?)?;
        Ok(ModelOutput::new(loss, Some(prediction)))
    }
}
