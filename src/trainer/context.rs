//! Explicit-context training entry points used during the P1 migration.

use crate::nn::ContextParameter;
use crate::optimizer::{clip_context_grad_norm, ContextOptimizer};
use crate::tensor::{GlobalTensor, TensorBase};
use crate::{ContextError, ContextId, ContextTensor, ContextVariable, ExecutionContext, MlError, MlResult};
use std::time::Instant;

use super::{
    BatchContext, BatchEndContext, BatchStartContext, DataError, EpochContext, EpochSchedule,
    MetricHook, StopReason, TrainEndContext, Trainer, TrainerCore, TrainingObserver,
    TrainingRuntime, TrainResult, TrainStartContext,
};

pub trait ContextTrainableModel {
    fn context_id(&self) -> ContextId;
    fn parameters(&self) -> Vec<&ContextParameter>;
}

pub trait ContextSupervisedModel: ContextTrainableModel {
    fn forward_loss(
        &mut self,
        input: &ContextVariable,
        target: &ContextTensor,
    ) -> MlResult<(ContextVariable, ContextVariable)>;
}

pub struct ContextSupervisedDataset<'a> {
    pub inputs: &'a [&'a ContextVariable],
    pub targets: &'a [&'a ContextTensor],
    context_id: ContextId,
}

impl<'a> ContextSupervisedDataset<'a> {
    pub fn new(
        context: &ExecutionContext,
        inputs: &'a [&'a ContextVariable],
        targets: &'a [&'a ContextTensor],
    ) -> MlResult<Self> {
        if inputs.is_empty() {
            return Err(MlError::StringError("context supervised dataset must not be empty".into()));
        }
        if inputs.len() != targets.len() {
            return Err(MlError::StringError("context supervised input/target length mismatch".into()));
        }
        if inputs.iter().any(|input| input.tensor().context_id() != context.id())
            || targets.iter().any(|target| target.context_id() != context.id())
        {
            return Err(ContextError::Mismatch.into());
        }
        Ok(Self { inputs, targets, context_id: context.id() })
    }

    pub fn len(&self) -> usize { self.inputs.len() }
    pub fn is_empty(&self) -> bool { self.inputs.is_empty() }
}

pub struct ContextSupervisedBatch {
    pub inputs: ContextVariable,
    pub targets: ContextTensor,
    pub samples: usize,
}

pub struct ContextSupervisedDataLoader<'a> {
    context: ExecutionContext,
    dataset: ContextSupervisedDataset<'a>,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    indices: Vec<usize>,
    cursor: usize,
}

impl<'a> ContextSupervisedDataLoader<'a> {
    pub fn new(context: &ExecutionContext, dataset: ContextSupervisedDataset<'a>) -> MlResult<Self> {
        if dataset.context_id != context.id() { return Err(ContextError::Mismatch.into()); }
        let indices = (0..dataset.len()).collect();
        Ok(Self {
            context: context.clone(), dataset, batch_size: 1, shuffle: true,
            drop_last: false, indices, cursor: 0,
        })
    }

    pub fn batch_size(mut self, batch_size: usize) -> MlResult<Self> {
        if batch_size == 0 { return Err(DataError::InvalidBatchSize.into()); }
        self.batch_size = batch_size;
        Ok(self)
    }
    pub fn shuffle(mut self, shuffle: bool) -> Self { self.shuffle = shuffle; self }
    pub fn drop_last(mut self, drop_last: bool) -> MlResult<Self> {
        if drop_last && self.dataset.len() < self.batch_size {
            return Err(DataError::NoBatches.into());
        }
        self.drop_last = drop_last;
        Ok(self)
    }
    pub fn batch_count(&self) -> usize {
        if self.drop_last { self.dataset.len() / self.batch_size }
        else { self.dataset.len().div_ceil(self.batch_size) }
    }

    fn begin_epoch(&mut self, runtime: &TrainingRuntime) {
        self.indices.clear();
        self.indices.extend(0..self.dataset.len());
        if self.shuffle { runtime.shuffle(&mut self.indices); }
        self.cursor = 0;
    }

    fn next_batch(&mut self) -> MlResult<Option<ContextSupervisedBatch>> {
        if self.cursor >= self.indices.len() { return Ok(None); }
        let end = (self.cursor + self.batch_size).min(self.indices.len());
        if self.drop_last && end - self.cursor < self.batch_size {
            self.cursor = self.indices.len();
            return Ok(None);
        }
        let selected = &self.indices[self.cursor..end];
        self.cursor = end;
        let inputs = selected.iter().map(|&index| self.dataset.inputs[index].tensor()).collect::<Vec<_>>();
        let targets = selected.iter().map(|&index| self.dataset.targets[index]).collect::<Vec<_>>();
        let (input_data, input_shape) = stack_context_tensors(&inputs)?;
        let (target_data, target_shape) = stack_context_tensors(&targets)?;
        Ok(Some(ContextSupervisedBatch {
            inputs: self.context.input(input_data, &input_shape)?,
            targets: self.context.tensor(target_data, &target_shape)?,
            samples: selected.len(),
        }))
    }
}

fn stack_context_tensors(tensors: &[&ContextTensor]) -> MlResult<(Vec<f32>, Vec<usize>)> {
    let first = tensors.first().ok_or(DataError::EmptyBatch)?;
    let expected = first.shape()?;
    let mut data = Vec::new();
    for (sample_index, tensor) in tensors.iter().enumerate() {
        let shape = tensor.shape()?;
        if shape != expected {
            return Err(DataError::ShapeMismatch { sample_index, expected, got: shape }.into());
        }
        data.extend(tensor.to_vec()?);
    }
    let mut shape = Vec::with_capacity(expected.len() + 1);
    shape.push(tensors.len());
    shape.extend(expected);
    Ok((data, shape))
}

pub struct ContextSupervisedTrainer {
    context: ExecutionContext,
    core: TrainerCore,
    max_grad_norm: Option<f32>,
    nan_check_interval: usize,
}

impl ContextSupervisedTrainer {
    pub fn new(context: &ExecutionContext) -> Self {
        Self::from_trainer(context, Trainer::silent())
    }

    /// Reuses the legacy supervised preset conversion so hooks and logging flags
    /// retain exactly the same meaning on the explicit-context path.
    pub fn from_trainer(context: &ExecutionContext, trainer: Trainer) -> Self {
        let trainer: super::SupervisedTrainer = trainer.into();
        let nan_check_interval = trainer.core.config.nan_check_interval;
        Self {
            context: context.clone(), core: trainer.core,
            max_grad_norm: None, nan_check_interval,
        }
    }

    pub fn silent(context: &ExecutionContext) -> Self {
        Self::from_trainer(context, Trainer::silent())
    }

    pub fn minimal(context: &ExecutionContext) -> Self {
        Self::from_trainer(context, Trainer::minimal())
    }

    pub fn default(context: &ExecutionContext) -> Self {
        Self::from_trainer(context, Trainer::default())
    }

    pub fn verbose(context: &ExecutionContext) -> Self {
        Self::from_trainer(context, Trainer::verbose())
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.core.config.seed = seed;
        self.core.runtime.reseed(seed);
        self
    }
    pub fn with_max_grad_norm(mut self, max_norm: f32) -> MlResult<Self> {
        if !max_norm.is_finite() || max_norm <= 0.0 {
            return Err(MlError::StringError("max_grad_norm must be finite and positive".into()));
        }
        self.max_grad_norm = Some(max_norm);
        Ok(self)
    }
    pub fn check_finite_gradients(mut self, enabled: bool) -> Self {
        self.nan_check_interval = if enabled { 1 } else { usize::MAX };
        self
    }
    pub fn with_hook(self, hook: Box<dyn MetricHook>) -> Self {
        self.core.add_hook(hook);
        self
    }
    pub fn with_observer(self, observer: Box<dyn TrainingObserver>) -> Self {
        self.core.add_observer(observer);
        self
    }

    pub fn train_epoch<M: ContextSupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        dataset: &ContextSupervisedDataset<'_>,
    ) -> MlResult<f32> {
        self.train_dataset_epoch(model, optimizer, dataset, 1, 1)
    }

    fn train_dataset_epoch<M: ContextSupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        dataset: &ContextSupervisedDataset<'_>,
        epoch: usize,
        total_epochs: usize,
    ) -> MlResult<f32> {
        self.validate(model, optimizer, dataset)?;
        self.reset_hooks()?;
        let mut total_loss = 0.0;
        for (batch_index, (&input, &target)) in dataset.inputs.iter().zip(dataset.targets).enumerate() {
            let batch_context = BatchStartContext {
                paradigm: "supervised",
                epoch,
                batch: batch_index + 1,
                total_epochs,
                total_batches: Some(dataset.len()),
                episode: None,
            };
            total_loss += self.run_batch(model, optimizer, input, target, &batch_context)?;
        }
        Ok(total_loss / dataset.len() as f32)
    }

    pub fn train_loader_epoch<M: ContextSupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        loader: &mut ContextSupervisedDataLoader<'_>,
        epoch: usize,
    ) -> MlResult<f32> {
        self.train_loader_epoch_inner(model, optimizer, loader, epoch, epoch + 1)
    }

    fn train_loader_epoch_inner<M: ContextSupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        loader: &mut ContextSupervisedDataLoader<'_>,
        epoch_index: usize,
        total_epochs: usize,
    ) -> MlResult<f32> {
        self.validate(model, optimizer, &loader.dataset)?;
        const EPOCH_MIX: u64 = 0x9E37_79B9_7F4A_7C15;
        self.reset_hooks()?;
        self.core.runtime.reseed(self.core.config.seed ^ (epoch_index as u64).wrapping_mul(EPOCH_MIX));
        loader.begin_epoch(&self.core.runtime);
        let mut weighted_loss = 0.0;
        let mut samples = 0usize;
        let mut batch_index = 0usize;
        while let Some(batch) = loader.next_batch()? {
            let batch_context = BatchStartContext {
                paradigm: "supervised",
                epoch: epoch_index + 1,
                batch: batch_index + 1,
                total_epochs,
                total_batches: Some(loader.batch_count()),
                episode: None,
            };
            let loss = self.run_batch(
                model,
                optimizer,
                &batch.inputs,
                &batch.targets,
                &batch_context,
            )?;
            weighted_loss += loss * batch.samples as f32;
            samples += batch.samples;
            batch_index += 1;
        }
        if samples == 0 { return Err(DataError::NoBatches.into()); }
        Ok(weighted_loss / samples as f32)
    }

    fn run_batch<M: ContextSupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        input: &ContextVariable,
        target: &ContextTensor,
        batch_context: &BatchStartContext,
    ) -> MlResult<f32> {
        let batch: MlResult<f32> = (|| {
            let (prediction, loss) = model.forward_loss(input, target)?;
            if loss.tensor().context_id() != self.context.id() { return Err(ContextError::Mismatch.into()); }
            let value = loss.tensor().item()?;
            if !value.is_finite() { return Err(MlError::StringError("non-finite context loss".into())); }
            loss.backward()?;
            let parameters = model.parameters();
            if self.nan_check_interval != usize::MAX
                && batch_context.batch % self.nan_check_interval == 0
            {
                for parameter in &parameters {
                    if parameter.grad()?.is_some_and(|gradient| gradient.data.iter().any(|value| !value.is_finite())) {
                        return Err(MlError::StringError("non-finite context gradient".into()));
                    }
                }
            }
            if let Some(max_norm) = self.max_grad_norm {
                clip_context_grad_norm(&self.context, &parameters, max_norm)?;
            }
            if self.core.hook_count() != 0 {
                let prediction_snapshot = GlobalTensor::from_vec(
                    prediction.tensor().to_vec()?, &prediction.tensor().shape()?,
                )?;
                let target_snapshot = GlobalTensor::from_vec(target.to_vec()?, &target.shape()?)?;
                let hook_context = BatchContext {
                    batch_idx: batch_context.batch - 1,
                    pred: Some(&prediction_snapshot), target: Some(&target_snapshot), loss: value,
                    n_tokens: None, lambda: None, lr: optimizer.lr(),
                };
                for hook in self.core.hooks.borrow_mut().iter_mut() { hook.update(&hook_context)?; }
            }
            optimizer.step()?;
            optimizer.zero_grad()?;
            self.core.notify_batch_end(&BatchEndContext {
                batch: batch_context.clone(),
                loss: value,
            });
            Ok(value)
        })();
        let cleanup = self.context.clear_graph();
        match (batch, cleanup) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
        }
    }

    fn reset_hooks(&self) -> MlResult<()> {
        for hook in self.core.hooks.borrow_mut().iter_mut() { hook.reset()?; }
        Ok(())
    }

    fn metric_values(&self) -> super::MetricValues {
        self.core.hooks.borrow().iter().map(|hook| (hook.name().to_string(), hook.compute())).collect()
    }

    pub fn fit<M: ContextSupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        dataset: &ContextSupervisedDataset<'_>,
        schedule: EpochSchedule,
    ) -> MlResult<TrainResult> {
        let started = Instant::now();
        self.core.notify_train_start(&TrainStartContext {
            paradigm: "supervised", total_units: schedule.epochs,
        });
        let mut previous = f32::INFINITY;
        let mut final_loss = f32::INFINITY;
        for epoch in 0..schedule.epochs {
            let epoch_context = EpochContext {
                paradigm: "supervised", epoch: epoch + 1, total_epochs: schedule.epochs,
                total_batches: Some(dataset.len()),
            };
            self.core.notify_epoch_start(&epoch_context);
            final_loss = match self.train_dataset_epoch(
                model, optimizer, dataset, epoch + 1, schedule.epochs,
            ) {
                Ok(loss) => loss,
                Err(error) => {
                    self.core.notify_train_error(&error.to_string());
                    return Err(error);
                }
            };
            self.core.notify_epoch_end(&epoch_context);
            if !final_loss.is_finite() {
                return Err(MlError::StringError(format!(
                    "context trainer produced non-finite loss at epoch {}", epoch + 1
                )));
            }
            if schedule.convergence.should_stop(previous, final_loss) {
                self.core.notify_train_end(&TrainEndContext {
                    paradigm: "supervised", units_completed: epoch + 1, interrupted: false,
                });
                return Ok(TrainResult::epochs(
                    StopReason::Converged, epoch + 1, final_loss, started.elapsed(),
                ).with_metrics(self.metric_values()));
            }
            previous = final_loss;
        }
        self.core.notify_train_end(&TrainEndContext {
            paradigm: "supervised", units_completed: schedule.epochs, interrupted: false,
        });
        Ok(TrainResult::epochs(
            StopReason::Completed, schedule.epochs, final_loss, started.elapsed(),
        ).with_metrics(self.metric_values()))
    }

    pub fn fit_loader<M: ContextSupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        loader: &mut ContextSupervisedDataLoader<'_>,
        schedule: EpochSchedule,
    ) -> MlResult<TrainResult> {
        let started = Instant::now();
        self.core.notify_train_start(&TrainStartContext {
            paradigm: "supervised", total_units: schedule.epochs,
        });
        let mut previous = f32::INFINITY;
        let mut final_loss = f32::INFINITY;
        for epoch in 0..schedule.epochs {
            let epoch_context = EpochContext {
                paradigm: "supervised", epoch: epoch + 1, total_epochs: schedule.epochs,
                total_batches: Some(loader.batch_count()),
            };
            self.core.notify_epoch_start(&epoch_context);
            final_loss = match self.train_loader_epoch_inner(
                model, optimizer, loader, epoch, schedule.epochs,
            ) {
                Ok(loss) => loss,
                Err(error) => {
                    self.core.notify_train_error(&error.to_string());
                    return Err(error);
                }
            };
            self.core.notify_epoch_end(&epoch_context);
            if schedule.convergence.should_stop(previous, final_loss) {
                self.core.notify_train_end(&TrainEndContext {
                    paradigm: "supervised", units_completed: epoch + 1, interrupted: false,
                });
                return Ok(TrainResult::epochs(
                    StopReason::Converged, epoch + 1, final_loss, started.elapsed(),
                ).with_metrics(self.metric_values()));
            }
            previous = final_loss;
        }
        self.core.notify_train_end(&TrainEndContext {
            paradigm: "supervised", units_completed: schedule.epochs, interrupted: false,
        });
        Ok(TrainResult::epochs(
            StopReason::Completed, schedule.epochs, final_loss, started.elapsed(),
        ).with_metrics(self.metric_values()))
    }

    fn validate<M: ContextTrainableModel + ?Sized>(
        &self,
        model: &M,
        optimizer: &dyn ContextOptimizer,
        dataset: &ContextSupervisedDataset<'_>,
    ) -> MlResult<()> {
        if model.context_id() != self.context.id()
            || optimizer.context_id() != self.context.id()
            || dataset.inputs.iter().any(|input| input.tensor().context_id() != self.context.id())
            || dataset.targets.iter().any(|target| target.context_id() != self.context.id())
            || model.parameters().iter().any(|parameter| parameter.context_id() != self.context.id())
        {
            return Err(ContextError::Mismatch.into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::Reduction;
    use crate::nn::{ContextLayer, ContextLinear};
    use crate::optimizer::{ContextOptimizer, ContextSGD};
    use crate::tensor::{GlobalTensor, TensorBase};
    use std::{cell::RefCell, rc::Rc};

    #[derive(Debug)]
    struct RegressionModel { context: ExecutionContext, linear: ContextLinear }

    impl ContextTrainableModel for RegressionModel {
        fn context_id(&self) -> ContextId { self.context.id() }
        fn parameters(&self) -> Vec<&ContextParameter> { self.linear.parameters() }
    }

    impl ContextSupervisedModel for RegressionModel {
        fn forward_loss(&mut self, input: &ContextVariable, target: &ContextTensor) -> MlResult<(ContextVariable, ContextVariable)> {
            let prediction = self.linear.apply(input)?;
            let loss = self.context.mse_loss_variable(&prediction, target, Reduction::Mean)?;
            Ok((prediction, loss))
        }
    }

    fn regression_model(context: &ExecutionContext) -> MlResult<RegressionModel> {
        let model = RegressionModel {
            context: context.clone(),
            linear: ContextLinear::new(context, 1, 1, "linear")?,
        };
        context.replace_parameter(model.linear.weight().variable(), GlobalTensor::from_vec(vec![0.0], &[1, 1])?)?;
        context.replace_parameter(model.linear.bias().variable(), GlobalTensor::from_vec(vec![0.0], &[1])?)?;
        Ok(model)
    }

    #[test]
    fn epoch_updates_parameters_and_clears_graph() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = regression_model(&context)?;
        let input = context.input(vec![2.0], &[1, 1])?;
        let target = context.tensor(vec![4.0], &[1, 1])?;
        let inputs = [&input];
        let targets = [&target];
        let dataset = ContextSupervisedDataset::new(&context, &inputs, &targets)?;
        let mut optimizer = ContextSGD::new(&context, 0.1)?;
        optimizer.register_all(&model.parameters())?;
        let loss = ContextSupervisedTrainer::new(&context).train_epoch(&mut model, &mut optimizer, &dataset)?;
        assert_eq!(loss, 16.0);
        assert_eq!(model.linear.weight().tensor().to_vec()?, vec![1.6]);
        assert_eq!(context.graph_stats()?.graph_nodes, 0);
        assert!(model.parameters().iter().all(|parameter| parameter.grad().unwrap().is_none()));
        Ok(())
    }

    #[test]
    fn trainer_rejects_foreign_optimizer() -> MlResult<()> {
        let context = ExecutionContext::new();
        let foreign = ExecutionContext::new();
        let mut model = regression_model(&context)?;
        let input = context.input(vec![1.0], &[1, 1])?;
        let target = context.tensor(vec![1.0], &[1, 1])?;
        let inputs = [&input];
        let targets = [&target];
        let dataset = ContextSupervisedDataset::new(&context, &inputs, &targets)?;
        let mut optimizer = ContextSGD::new(&foreign, 0.1)?;
        assert!(matches!(
            ContextSupervisedTrainer::new(&context).train_epoch(&mut model, &mut optimizer, &dataset),
            Err(MlError::ContextError(ContextError::Mismatch))
        ));
        Ok(())
    }

    #[test]
    fn context_loader_stacks_drops_and_shuffles_deterministically() -> MlResult<()> {
        let context = ExecutionContext::new();
        let inputs = [
            context.input(vec![1.0, 10.0], &[2])?,
            context.input(vec![2.0, 20.0], &[2])?,
            context.input(vec![3.0, 30.0], &[2])?,
        ];
        let targets = [
            context.tensor(vec![1.0], &[1])?,
            context.tensor(vec![2.0], &[1])?,
            context.tensor(vec![3.0], &[1])?,
        ];
        let input_refs: Vec<_> = inputs.iter().collect();
        let target_refs: Vec<_> = targets.iter().collect();
        let mut ordered = ContextSupervisedDataLoader::new(
            &context,
            ContextSupervisedDataset::new(&context, &input_refs, &target_refs)?,
        )?.batch_size(2)?.shuffle(false);
        assert_eq!(ordered.batch_count(), 2);
        ordered.begin_epoch(&TrainingRuntime::new(9));
        let first = ordered.next_batch()?.expect("first batch");
        assert_eq!(first.inputs.tensor().shape()?, vec![2, 2]);
        assert_eq!(first.targets.shape()?, vec![2, 1]);
        assert_eq!(first.inputs.tensor().to_vec()?, vec![1.0, 10.0, 2.0, 20.0]);

        let mut dropped = ContextSupervisedDataLoader::new(
            &context,
            ContextSupervisedDataset::new(&context, &input_refs, &target_refs)?,
        )?.batch_size(2)?.drop_last(true)?;
        assert_eq!(dropped.batch_count(), 1);
        let left_runtime = TrainingRuntime::new(77);
        let right_runtime = TrainingRuntime::new(77);
        let mut shuffled = ContextSupervisedDataLoader::new(
            &context,
            ContextSupervisedDataset::new(&context, &input_refs, &target_refs)?,
        )?.batch_size(3)?;
        dropped.begin_epoch(&left_runtime);
        shuffled.begin_epoch(&right_runtime);
        assert_eq!(dropped.indices, shuffled.indices);
        Ok(())
    }

    #[test]
    fn loader_fit_matches_epoch_accounting_and_cleans_temporary_batches() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = regression_model(&context)?;
        let inputs = [
            context.input(vec![1.0], &[1])?,
            context.input(vec![2.0], &[1])?,
            context.input(vec![3.0], &[1])?,
        ];
        let targets = [
            context.tensor(vec![2.0], &[1])?,
            context.tensor(vec![4.0], &[1])?,
            context.tensor(vec![6.0], &[1])?,
        ];
        let input_refs: Vec<_> = inputs.iter().collect();
        let target_refs: Vec<_> = targets.iter().collect();
        let mut loader = ContextSupervisedDataLoader::new(
            &context,
            ContextSupervisedDataset::new(&context, &input_refs, &target_refs)?,
        )?.batch_size(2)?.shuffle(false);
        let mut optimizer = ContextSGD::new(&context, 0.05)?;
        optimizer.register_all(&model.parameters())?;
        let result = ContextSupervisedTrainer::new(&context).fit_loader(
            &mut model, &mut optimizer, &mut loader, EpochSchedule::new(3)?,
        )?;
        assert_eq!(result.units_completed, 3);
        assert!(result.final_loss.is_finite());
        assert_eq!(context.graph_stats()?.graph_nodes, 0);
        Ok(())
    }

    struct RecordingHook {
        updates: Rc<RefCell<Vec<(usize, Vec<f32>, Vec<f32>)>>>,
    }

    impl MetricHook for RecordingHook {
        fn update(&mut self, context: &BatchContext<'_>) -> MlResult<()> {
            self.updates.borrow_mut().push((
                context.batch_idx,
                context.pred.map(|tensor| tensor.data().to_vec()).unwrap_or_default(),
                context.target.map(|tensor| tensor.data().to_vec()).unwrap_or_default(),
            ));
            Ok(())
        }

        fn compute(&self) -> f32 { self.updates.borrow().len() as f32 }
        fn reset(&mut self) -> MlResult<()> { Ok(()) }
        fn name(&self) -> &str { "recorded_batches" }
    }

    struct RecordingObserver {
        events: Rc<RefCell<Vec<String>>>,
    }

    impl TrainingObserver for RecordingObserver {
        fn on_train_start(&mut self, context: &TrainStartContext) {
            self.events.borrow_mut().push(format!("start:{}", context.total_units));
        }
        fn on_epoch_start(&mut self, context: &EpochContext) {
            self.events.borrow_mut().push(format!("epoch-start:{}", context.epoch));
        }
        fn on_batch_end(&mut self, context: &BatchEndContext) {
            self.events.borrow_mut().push(format!(
                "batch:{}:{}/{}",
                context.batch.epoch,
                context.batch.batch,
                context.batch.total_batches.unwrap_or_default(),
            ));
        }
        fn on_epoch_end(&mut self, context: &EpochContext) {
            self.events.borrow_mut().push(format!("epoch-end:{}", context.epoch));
        }
        fn on_train_end(&mut self, context: &TrainEndContext) {
            self.events.borrow_mut().push(format!("end:{}", context.units_completed));
        }
    }

    #[test]
    fn context_trainer_reuses_hooks_and_observer_lifecycle() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = regression_model(&context)?;
        let inputs = [
            context.input(vec![1.0], &[1, 1])?,
            context.input(vec![2.0], &[1, 1])?,
        ];
        let targets = [
            context.tensor(vec![2.0], &[1, 1])?,
            context.tensor(vec![4.0], &[1, 1])?,
        ];
        let input_refs: Vec<_> = inputs.iter().collect();
        let target_refs: Vec<_> = targets.iter().collect();
        let dataset = ContextSupervisedDataset::new(&context, &input_refs, &target_refs)?;
        let mut optimizer = ContextSGD::new(&context, 0.01)?;
        optimizer.register_all(&model.parameters())?;

        let updates = Rc::new(RefCell::new(Vec::new()));
        let events = Rc::new(RefCell::new(Vec::new()));
        let trainer = ContextSupervisedTrainer::silent(&context)
            .with_hook(Box::new(RecordingHook { updates: updates.clone() }))
            .with_observer(Box::new(RecordingObserver { events: events.clone() }));
        let result = trainer.fit(
            &mut model,
            &mut optimizer,
            &dataset,
            EpochSchedule::new(2)?,
        )?;

        assert_eq!(result.metrics.get("recorded_batches"), Some(&4.0));
        let updates = updates.borrow();
        assert_eq!(updates.iter().map(|entry| entry.0).collect::<Vec<_>>(), vec![0, 1, 0, 1]);
        assert!(updates.iter().all(|entry| !entry.1.is_empty() && !entry.2.is_empty()));
        assert_eq!(
            events.borrow().as_slice(),
            [
                "start:2", "epoch-start:1", "batch:1:1/2", "batch:1:2/2", "epoch-end:1",
                "epoch-start:2", "batch:2:1/2", "batch:2:2/2", "epoch-end:2", "end:2",
            ]
        );
        Ok(())
    }

    #[test]
    fn context_presets_reuse_supervised_hook_policy() {
        let context = ExecutionContext::new();
        let silent = ContextSupervisedTrainer::silent(&context);
        let default = ContextSupervisedTrainer::default(&context);
        assert_eq!(silent.core.hook_count(), 0);
        assert_eq!(silent.nan_check_interval, usize::MAX);
        assert_eq!(default.core.hook_count(), 1);
        assert_eq!(default.nan_check_interval, 1);
    }
}
