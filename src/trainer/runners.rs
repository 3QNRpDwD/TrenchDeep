use super::*;
use crate::optimizer::Optimizer;
use std::{path::Path, time::Instant};

impl Trainer<Eager> {
    pub fn fit<M, I>(
        &self,
        model: &mut M,
        optimizer: &mut dyn Optimizer,
        input: I,
        schedule: EpochSchedule,
    ) -> MlResult<TrainResult>
    where
        M: TrainingModel,
        I: IntoBatchLoader<Batch = M::Batch>,
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
        M: TrainingModel + CheckpointableModel,
        I: IntoBatchLoader<Batch = M::Batch>,
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
        save: Option<fn(&M, &Path) -> MlResult<()>>,
    ) -> MlResult<TrainResult>
    where
        M: TrainingModel,
        I: IntoBatchLoader<Batch = M::Batch>,
    {
        self.service.fit_steps(
            model,
            optimizer,
            input,
            schedule,
            M::PARADIGM,
            |model, batch, epoch, optimizer, context| {
                let start = Instant::now();
                let step = self.step_context::<M>(epoch, context.batch - 1);
                let output = model.forward_batch(&batch, &step)?;
                self.service
                    .finish_step(model, optimizer, output, context, start.elapsed())
            },
            save,
            false,
        )
    }
}
