//! Model-level training facade shared by direct callers and the Trainer.
use super::{ExecutionInputs, PreparedExecutor, PreparedMode, invalid};
use crate::{
    trainer::{TrainingModel, TrainingStepContext},
    *,
};
/// Host work (random inputs, scheduling and batch metadata) is kept outside capture.
/// A topology variant must change whenever configuration changes the numeric graph.
/// Epoch-dependent values such as `step.lambda` must be supplied as input tensors,
/// not captured as host constants. Random sampling belongs in `execution_batch`.
pub trait PreparedModel: TrainingModel {
    fn execution_batch(
        &mut self,
        batch: &Self::Batch,
        step: &TrainingStepContext,
    ) -> MlResult<PreparedBatch>;
    /// Numeric operations only; called once per prepared input signature.
    fn forward_inputs(&self, inputs: &ExecutionInputs) -> MlResult<ModelOutput>;
}
pub struct PreparedBatch {
    pub inputs: ExecutionInputs,
    pub weight: usize,
    pub target: Option<Tensor>,
    pub tokens: Option<usize>,
    pub lambda: Option<f32>,
}
impl PreparedBatch {
    pub fn new(inputs: ExecutionInputs, weight: usize) -> Self {
        Self {
            inputs,
            weight,
            target: None,
            tokens: None,
            lambda: None,
        }
    }
}
pub struct ModelOutput {
    pub loss: Variable,
    pub prediction: Option<Variable>,
}
impl ModelOutput {
    pub fn new(loss: Variable, prediction: Option<Variable>) -> Self {
        Self { loss, prediction }
    }
}
#[derive(Debug)]
pub struct PreparedModelExecutor {
    executor: PreparedExecutor,
    prediction: bool,
}
impl ExecutionContext {
    /// Prepare from already-created example inputs; this does not generate noise.
    pub fn prepare_model<M: PreparedModel>(
        &self,
        model: &M,
        inputs: &ExecutionInputs,
    ) -> MlResult<PreparedModelExecutor> {
        self.prepare_model_impl(model, inputs, false)
    }
    /// Keep prediction exports, but prepare backward only from the model loss.
    /// Use `prepare_model` when a caller also needs prediction-root backward.
    pub fn prepare_model_for_loss<M: PreparedModel>(
        &self,
        model: &M,
        inputs: &ExecutionInputs,
    ) -> MlResult<PreparedModelExecutor> {
        self.prepare_model_impl(model, inputs, true)
    }
    fn prepare_model_impl<M: PreparedModel>(
        &self,
        model: &M,
        inputs: &ExecutionInputs,
        loss_only: bool,
    ) -> MlResult<PreparedModelExecutor> {
        if model.context_id() != self.id() {
            return Err(ContextError::Mismatch.into());
        }
        let mut prediction = false;
        let describe = |inputs: &ExecutionInputs| {
            let output = model.forward_inputs(inputs)?;
            let mut tensors = vec![output.loss.tensor().clone()];
            if let Some(value) = output.prediction {
                prediction = true;
                tensors.push(value.tensor().clone());
            }
            Ok(tensors)
        };
        let plan = if loss_only {
            self.prepare_forward_with_roots(
                inputs,
                &model.parameters(),
                PreparedMode::Training,
                &[0],
                describe,
            )?
        } else {
            self.prepare_forward(
                inputs,
                &model.parameters(),
                PreparedMode::Training,
                describe,
            )?
        };
        Ok(PreparedModelExecutor {
            executor: plan.into_executor(self)?,
            prediction,
        })
    }
}
impl PreparedModelExecutor {
    pub fn executor(&self) -> &PreparedExecutor {
        &self.executor
    }
    pub fn run<M: PreparedModel, T>(
        &mut self,
        model: &M,
        inputs: &ExecutionInputs,
        callback: impl FnOnce(ModelOutput) -> MlResult<T>,
    ) -> MlResult<T> {
        if model.context_id() != self.executor.context_id() {
            return Err(ContextError::Mismatch.into());
        }
        let prediction = self.prediction;
        self.executor
            .with_inputs(inputs, &model.parameters(), |out| {
                if out.len() != if prediction { 2 } else { 1 } {
                    return Err(invalid("model output contract changed"));
                }
                callback(ModelOutput {
                    loss: out[0].as_variable()?,
                    prediction: if prediction {
                        Some(out[1].as_variable()?)
                    } else {
                        None
                    },
                })
            })
    }
}
