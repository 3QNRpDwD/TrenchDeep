//! Batch preparation and training objectives, separate from prediction models.
use super::*;
use crate::{loss::Loss, runtime::prepared::{ExecutionInputs, ModelOutput, PreparedBatch, PreparedModel}};

mod autoregressive;
mod diffusion;
mod semi_supervised;
pub use autoregressive::Autoregressive;
pub use diffusion::DiffusionObjective;
pub use semi_supervised::SemiSupervised;

/// Single-input differentiable prediction. Host sampling belongs in an Objective.
pub trait ForwardModel: TrainableModel {
    fn forward(&self, input: &Variable) -> MlResult<Variable>;
}

/// Defines batch preparation, model calls, loss and reporting metadata.
pub trait Objective<M: TrainableModel> {
    type Batch: BatchInputs;
    const PARADIGM: &'static str;
    fn forward_batch(&self, model: &mut M, batch: &Self::Batch, step: &TrainingStepContext) -> MlResult<TrainingOutput>;
}

/// Opt-in capture contract. Sample randomness in execution_batch, not forward_inputs.
/// Keep structural settings fixed; feed changing values such as lambda as tensors.
pub trait PreparedObjective<M: TrainableModel>: Objective<M> {
    fn execution_batch(&self, model: &mut M, batch: &Self::Batch, step: &TrainingStepContext) -> MlResult<PreparedBatch>;
    fn forward_inputs(&self, model: &M, inputs: &ExecutionInputs) -> MlResult<ModelOutput>;
}

/// A borrowed bridge to the existing training loop and prepared model executor.
pub(crate) struct BoundObjective<'a, M, O> {
    pub model: &'a mut M,
    pub objective: &'a O,
}
impl<M: TrainableModel, O> TrainableModel for BoundObjective<'_, M, O> {
    fn context_id(&self) -> ContextId { self.model.context_id() }
    fn parameters(&self) -> Vec<&Parameter> { self.model.parameters() }
}
impl<M: TrainableModel, O: Objective<M>> TrainingModel for BoundObjective<'_, M, O> {
    type Batch = O::Batch;
    const PARADIGM: &'static str = O::PARADIGM;
    fn forward_batch(&mut self, batch: &Self::Batch, step: &TrainingStepContext) -> MlResult<TrainingOutput> {
        self.objective.forward_batch(self.model, batch, step)
    }
}
impl<M: TrainableModel, O: PreparedObjective<M>> PreparedModel for BoundObjective<'_, M, O> {
    fn execution_batch(&mut self, batch: &Self::Batch, step: &TrainingStepContext) -> MlResult<PreparedBatch> {
        self.objective.execution_batch(self.model, batch, step)
    }
    fn forward_inputs(&self, inputs: &ExecutionInputs) -> MlResult<ModelOutput> {
        self.objective.forward_inputs(self.model, inputs)
    }
}
impl<M: CheckpointableModel, O> CheckpointableModel for BoundObjective<'_, M, O> {
    fn save_checkpoint(&self, path: &std::path::Path) -> MlResult<()> { self.model.save_checkpoint(path) }
    fn load_checkpoint(&mut self, path: &std::path::Path) -> MlResult<()> { self.model.load_checkpoint(path) }
}

fn sample_count(input: &Tensor) -> MlResult<usize> {
    let shape = input.shape()?;
    Ok(if shape.len() > 1 { shape[0] } else { 1 })
}

#[derive(Debug, Clone, Copy)]
pub struct Supervised<L> { loss: L }
impl<L> Supervised<L> {
    pub fn new(loss: L) -> Self { Self { loss } }
}
impl<M: ForwardModel, L: Loss> Objective<M> for Supervised<L> {
    type Batch = SupervisedBatch;
    const PARADIGM: &'static str = "supervised";
    fn forward_batch(&self, model: &mut M, batch: &Self::Batch, _: &TrainingStepContext) -> MlResult<TrainingOutput> {
        let weight = sample_count(batch.inputs.tensor())?;
        let prediction = model.forward(&batch.inputs)?;
        let ctx = batch.inputs.tensor().execution_context()?;
        let loss = self.loss.compute(&ctx, &prediction, &batch.targets)?;
        Ok(TrainingOutput { loss, prediction: Some(prediction), target: Some(batch.targets.clone()), weight, tokens: None, lambda: None })
    }
}
impl<M: ForwardModel, L: Loss> PreparedObjective<M> for Supervised<L> {
    fn execution_batch(&self, _: &mut M, batch: &Self::Batch, _: &TrainingStepContext) -> MlResult<PreparedBatch> {
        let inputs = ExecutionInputs::new("supervised")
            .with("x", batch.inputs.tensor().clone())?
            .with("target", batch.targets.clone())?;
        let mut batch_out = PreparedBatch::new(inputs, sample_count(batch.inputs.tensor())?);
        batch_out.target = Some(batch.targets.clone());
        Ok(batch_out)
    }
    fn forward_inputs(&self, model: &M, inputs: &ExecutionInputs) -> MlResult<ModelOutput> {
        let input = inputs.get("x")?;
        let prediction = model.forward(&input.as_variable()?)?;
        let loss = self.loss.compute(&input.execution_context()?, &prediction, inputs.get("target")?)?;
        Ok(ModelOutput::new(loss, Some(prediction)))
    }
}
