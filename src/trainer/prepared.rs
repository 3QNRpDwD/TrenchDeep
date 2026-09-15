//! Prepared training uses the same model executor as user-owned loops.
use super::{
    service::{BatchInputs, StepData, TrainingService},
    *,
};
use crate::{
    optimizer::Optimizer,
    runtime::prepared::{PreparedModel, PreparedModelExecutor},
};
use std::time::Instant;
pub struct PreparedTrainer {
    service: TrainingService,
}
impl Trainer {
    /// Explicit static execution; unsupported models/providers never fall back.
    pub fn prepared(self, context: &ExecutionContext) -> PreparedTrainer {
        PreparedTrainer {
            service: TrainingService::new(context, self),
        }
    }
}
impl PreparedTrainer {
    pub fn with_max_grad_norm(mut self, max: f32) -> MlResult<Self> {
        if !max.is_finite() || max <= 0.0 {
            return Err(MlError::StringError(
                "max_grad_norm must be finite and positive".into(),
            ));
        }
        self.service.max_grad_norm = Some(max);
        Ok(self)
    }
    pub fn fit<M, I>(
        &self,
        model: &mut M,
        optimizer: &mut dyn Optimizer,
        input: I,
        schedule: EpochSchedule,
    ) -> MlResult<TrainResult>
    where
        M: PreparedModel,
        I: IntoBatchLoader<Batch = M::Batch>,
        M::Batch: BatchInputs,
    {
        self.fit_impl(model, optimizer, input, schedule, None)
    }
    pub fn fit_checkpointed<M, I>(
        &self,
        model: &mut M,
        optimizer: &mut dyn Optimizer,
        input: I,
        schedule: EpochSchedule,
    ) -> MlResult<TrainResult>
    where
        M: PreparedModel + CheckpointableModel,
        I: IntoBatchLoader<Batch = M::Batch>,
        M::Batch: BatchInputs,
    {
        checkpoint::install_interrupt_handler()?;
        self.fit_impl(
            model,
            optimizer,
            input,
            schedule,
            Some(|m, p| m.save_checkpoint(p)),
        )
    }
    fn fit_impl<M, I>(
        &self,
        model: &mut M,
        optimizer: &mut dyn Optimizer,
        input: I,
        schedule: EpochSchedule,
        save: Option<fn(&M, &std::path::Path) -> MlResult<()>>,
    ) -> MlResult<TrainResult>
    where
        M: PreparedModel,
        I: IntoBatchLoader<Batch = M::Batch>,
        M::Batch: BatchInputs,
    {
        let mut plans: Vec<(
            Vec<(String, Vec<usize>, bool)>,
            String,
            PreparedModelExecutor,
        )> = Vec::new();
        self.service.fit_steps(
            model,
            optimizer,
            input,
            schedule,
            M::PARADIGM,
            |model, batch, _, optimizer, context| {
                // Host input creation is cleaned up on errors and does not run during capture.
                let batch = self
                    .service
                    .context
                    .with_training_scope(|| model.execution_batch(&batch))?;
                let signature = batch.inputs.tensor_signature()?;
                let variant = batch.inputs.variant().to_owned();
                let index = if let Some(index) = plans
                    .iter()
                    .position(|(s, v, _)| *s == signature && *v == variant)
                {
                    index
                } else {
                    let prepared = self.service.context.prepare_model(model, &batch.inputs)?;
                    plans.push((signature, variant, prepared));
                    plans.len() - 1
                };
                let start = Instant::now();
                plans[index].2.run(model, &batch.inputs, |output| {
                    self.service.finish_step(
                        model,
                        optimizer,
                        StepData {
                            loss: output.loss,
                            prediction: output.prediction,
                            target: batch.target,
                            weight: batch.weight,
                            tokens: batch.tokens,
                            lambda: batch.lambda,
                        },
                        context,
                        start.elapsed(),
                    )
                })
            },
            save,
            true,
        )
    }
}
