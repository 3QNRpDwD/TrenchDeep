//! Prepared training uses the same model executor as user-owned loops.
use super::{
    service::{BatchInputs, StepData},
    *,
};
use crate::{
    optimizer::Optimizer,
    runtime::prepared::{PreparedModel, PreparedModelExecutor},
};
use std::time::Instant;
impl<S> Trainer<Prepared, S> {
    pub fn fit<M, L: crate::loss::Loss + ?Sized, I>(
        &self,
        model: &mut M,
        loss: &L,
        optimizer: &mut dyn Optimizer,
        input: I,
        schedule: EpochSchedule,
    ) -> MlResult<TrainResult>
    where
        M: TrainableModel,
        S: PreparedTrainingStrategy<M>,
        I: IntoBatchLoader<Batch = S::Batch>,
    {
        let mut bound = strategy::BoundTraining {
            model,
            strategy: &self.strategy,
            loss,
        };
        self.fit_impl(&mut bound, optimizer, input, schedule, None)
    }
    pub fn fit_checkpointed<M, L: crate::loss::Loss + ?Sized, I>(
        &self,
        model: &mut M,
        loss: &L,
        optimizer: &mut dyn Optimizer,
        input: I,
        schedule: EpochSchedule,
    ) -> MlResult<TrainResult>
    where
        M: TrainableModel + CheckpointableModel,
        S: PreparedTrainingStrategy<M>,
        I: IntoBatchLoader<Batch = S::Batch>,
    {
        checkpoint::install_interrupt_handler()?;
        let mut bound = strategy::BoundTraining {
            model,
            strategy: &self.strategy,
            loss,
        };
        self.fit_impl(
            &mut bound,
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
            |model, batch, epoch, optimizer, context| {
                let step = self.step_context::<M>(epoch, context.batch - 1);
                // Host input creation is cleaned up on errors and does not run during capture.
                let batch = self
                    .service
                    .context
                    .with_training_scope(|| model.execution_batch(&batch, &step))?;
                let signature = batch.inputs.tensor_signature()?;
                let variant = batch.inputs.variant().to_owned();
                let index = if let Some(index) = plans
                    .iter()
                    .position(|(s, v, _)| *s == signature && *v == variant)
                {
                    index
                } else {
                    let prepared = self
                        .service
                        .context
                        .prepare_model_for_loss(model, &batch.inputs)?;
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
