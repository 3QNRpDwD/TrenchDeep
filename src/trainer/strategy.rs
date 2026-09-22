//! Batch preparation and training strategies, separate from prediction models.
use super::*;
use crate::{
    loss::Loss,
    runtime::prepared::{ExecutionInputs, ModelOutput, PreparedBatch, PreparedModel},
};

mod autoregressive;
mod diffusion;
mod semi_supervised;
pub use autoregressive::Autoregressive;
pub use diffusion::DiffusionTraining;
pub use semi_supervised::SemiSupervised;

/// Single-input differentiable prediction. Host sampling belongs in a TrainingStrategy.
pub trait ForwardModel: TrainableModel {
    fn forward(&self, input: &Variable) -> MlResult<Variable>;
}

/// Defines batch preparation, model calls, loss and reporting metadata.
pub trait TrainingStrategy<M: TrainableModel> {
    type Batch: BatchInputs;
    const PARADIGM: ParadigmTag;
    fn forward_batch<L: Loss + ?Sized>(
        &self,
        loss: &L,
        model: &mut M,
        batch: &Self::Batch,
        step: &TrainingStepContext,
    ) -> MlResult<TrainingOutput>;
}

/// Opt-in capture contract. Sample randomness in execution_batch, not forward_inputs.
/// Keep structural settings fixed; feed changing values such as lambda as tensors.
pub trait PreparedTrainingStrategy<M: TrainableModel>: TrainingStrategy<M> {
    fn execution_batch(
        &self,
        model: &mut M,
        batch: &Self::Batch,
        step: &TrainingStepContext,
    ) -> MlResult<PreparedBatch>;
    fn forward_inputs<L: Loss + ?Sized>(
        &self,
        loss: &L,
        model: &M,
        inputs: &ExecutionInputs,
    ) -> MlResult<ModelOutput>;
}

/// A borrowed bridge to the existing training loop and prepared model executor.
pub(crate) struct BoundTraining<'a, M, O, L: ?Sized> {
    pub model: &'a mut M,
    pub strategy: &'a O,
    pub loss: &'a L,
}
impl<M: TrainableModel, O, L: ?Sized> TrainableModel for BoundTraining<'_, M, O, L> {
    fn context_id(&self) -> ContextId {
        self.model.context_id()
    }
    fn parameters(&self) -> Vec<&Parameter> {
        self.model.parameters()
    }
}
impl<M: TrainableModel, O: TrainingStrategy<M>, L: Loss + ?Sized> TrainingModel
    for BoundTraining<'_, M, O, L>
{
    type Batch = O::Batch;
    const PARADIGM: ParadigmTag = O::PARADIGM;
    fn forward_batch(
        &mut self,
        batch: &Self::Batch,
        step: &TrainingStepContext,
    ) -> MlResult<TrainingOutput> {
        self.strategy
            .forward_batch(self.loss, self.model, batch, step)
    }
}
impl<M: TrainableModel, O: PreparedTrainingStrategy<M>, L: Loss + ?Sized> PreparedModel
    for BoundTraining<'_, M, O, L>
{
    fn execution_batch(
        &mut self,
        batch: &Self::Batch,
        step: &TrainingStepContext,
    ) -> MlResult<PreparedBatch> {
        self.strategy.execution_batch(self.model, batch, step)
    }
    fn forward_inputs(&self, inputs: &ExecutionInputs) -> MlResult<ModelOutput> {
        self.strategy.forward_inputs(self.loss, self.model, inputs)
    }
}
impl<M: CheckpointableModel, O, L: ?Sized> CheckpointableModel for BoundTraining<'_, M, O, L> {
    fn save_checkpoint(&self, path: &std::path::Path) -> MlResult<()> {
        self.model.save_checkpoint(path)
    }
    fn load_checkpoint(&mut self, path: &std::path::Path) -> MlResult<()> {
        self.model.load_checkpoint(path)
    }
}

fn sample_count(input: &Tensor) -> MlResult<usize> {
    let shape = input.shape()?;
    Ok(if shape.len() > 1 { shape[0] } else { 1 })
}

#[derive(Debug, Clone, Copy)]
pub struct Supervised;
impl<M: ForwardModel> TrainingStrategy<M> for Supervised {
    type Batch = SupervisedBatch;
    const PARADIGM: ParadigmTag = ParadigmTag::Supervised;
    fn forward_batch<L: Loss + ?Sized>(
        &self,
        loss: &L,
        model: &mut M,
        batch: &Self::Batch,
        _: &TrainingStepContext,
    ) -> MlResult<TrainingOutput> {
        let weight = sample_count(batch.inputs.tensor())?;
        let prediction = model.forward(&batch.inputs)?;
        let ctx = batch.inputs.tensor().execution_context()?;
        let loss = loss.compute(&ctx, &prediction, &batch.targets)?;
        Ok(TrainingOutput {
            loss,
            prediction: Some(prediction),
            target: Some(batch.targets.clone()),
            weight,
            tokens: None,
            lambda: None,
        })
    }
}
impl<M: ForwardModel> PreparedTrainingStrategy<M> for Supervised {
    fn execution_batch(
        &self,
        _: &mut M,
        batch: &Self::Batch,
        _: &TrainingStepContext,
    ) -> MlResult<PreparedBatch> {
        let inputs = ExecutionInputs::new("supervised")
            .with("x", batch.inputs.tensor().clone())?
            .with("target", batch.targets.clone())?;
        let mut batch_out = PreparedBatch::new(inputs, sample_count(batch.inputs.tensor())?);
        batch_out.target = Some(batch.targets.clone());
        Ok(batch_out)
    }
    fn forward_inputs<L: Loss + ?Sized>(
        &self,
        loss: &L,
        model: &M,
        inputs: &ExecutionInputs,
    ) -> MlResult<ModelOutput> {
        let input = inputs.get("x")?;
        let prediction = model.forward(&input.as_variable()?)?;
        let loss = loss.compute(
            &input.execution_context()?,
            &prediction,
            inputs.get("target")?,
        )?;
        Ok(ModelOutput::new(loss, Some(prediction)))
    }
}
