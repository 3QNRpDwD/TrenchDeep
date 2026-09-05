//! Explicit-context training entry points used during the P1 migration.

mod autoregressive;
mod reinforcement;
mod semi_supervised;
mod unsupervised;
pub use autoregressive::{
    ContextAutoregressiveBatch, ContextAutoregressiveDataLoader, ContextAutoregressiveDataset,
    ContextAutoregressiveModel, ContextAutoregressiveTrainer,
};
pub use unsupervised::{
    ContextUnsupervisedBatch, ContextUnsupervisedDataLoader, ContextUnsupervisedDataset,
    ContextUnsupervisedModel, ContextUnsupervisedTrainer,
};
pub use semi_supervised::{
    ContextSemiSupervisedBatch, ContextSemiSupervisedDataLoader,
    ContextSemiSupervisedDataset, ContextSemiSupervisedModel, ContextSemiSupervisedTrainer,
};
pub use reinforcement::{
    ContextEnvironment, ContextRLModel, ContextRLTrainer, ContextStepResult,
};

use crate::nn::ContextParameter;
use crate::optimizer::{clip_context_grad_norm, ContextOptimizer};
use crate::tensor::{GlobalTensor, TensorBase};
use crate::{ContextError, ContextId, ContextTensor, ContextVariable, ExecutionContext, MlError, MlResult};
use std::time::{Duration, Instant};

use super::{
    BatchContext, BatchEndContext, BatchLoader, BatchStartContext, DataError, EpochContext, EpochSchedule,
    MetricHook, StopReason, TrainEndContext, Trainer, TrainerCore, TrainingObserver,
    TrainingRuntime, TrainResult, TrainStartContext,
};

pub trait ContextTrainableModel {
    fn context_id(&self) -> ContextId;
    fn parameters(&self) -> Vec<&ContextParameter>;
}

fn validate_training_parameters(
    context: &ExecutionContext,
    model: &impl ContextTrainableModel,
    optimizer: &dyn ContextOptimizer,
) -> MlResult<()> {
    use std::collections::HashSet;
    if model.context_id() != context.id() || optimizer.context_id() != context.id() {
        return Err(ContextError::Mismatch.into());
    }
    let model_parameters = model.parameters();
    let registered = optimizer.registered_parameters();
    for parameter in model_parameters.iter().chain(registered.iter()) {
        if parameter.context_id() != context.id() { return Err(ContextError::Mismatch.into()); }
        parameter.tensor().numel()?;
    }
    let expected = model_parameters.iter().map(|p| p.node_id()).collect::<HashSet<_>>();
    let actual = registered.iter().map(|p| p.node_id()).collect::<HashSet<_>>();
    if expected != actual {
        return Err(crate::optimizer::OptimError::ParameterSetMismatch {
            missing: expected.difference(&actual).copied().collect(),
            extra: actual.difference(&expected).copied().collect(),
        }.into());
    }
    Ok(())
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
        if self.drop_last && self.dataset.len() < batch_size {
            return Err(DataError::NoBatches.into());
        }
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

impl BatchLoader for ContextSupervisedDataLoader<'_> {
    type Batch = ContextSupervisedBatch;

    fn begin_epoch(&mut self, _epoch: usize, runtime: &TrainingRuntime) -> MlResult<()> {
        ContextSupervisedDataLoader::begin_epoch(self, runtime);
        Ok(())
    }

    fn next_batch(&mut self) -> MlResult<Option<Self::Batch>> {
        ContextSupervisedDataLoader::next_batch(self)
    }

    fn batch_count(&self) -> Option<usize> {
        Some(ContextSupervisedDataLoader::batch_count(self))
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

struct ContextBatchOutcome {
    loss: f32,
    forward: Option<Duration>,
    backward: Option<Duration>,
    grad_norm: Option<f32>,
    update_ratio: Option<f32>,
}

#[derive(Default)]
struct ContextEpochDiagnostics {
    forward_secs: f32,
    forward_count: usize,
    backward_secs: f32,
    backward_count: usize,
    grad_norm: f32,
    grad_norm_count: usize,
    update_ratio: f32,
    update_ratio_count: usize,
}

impl ContextEpochDiagnostics {
    fn record(&mut self, outcome: &ContextBatchOutcome) {
        if let Some(duration) = outcome.forward {
            self.forward_secs += duration.as_secs_f32();
            self.forward_count += 1;
        }
        if let Some(duration) = outcome.backward {
            self.backward_secs += duration.as_secs_f32();
            self.backward_count += 1;
        }
        if let Some(value) = outcome.grad_norm {
            self.grad_norm += value;
            self.grad_norm_count += 1;
        }
        if let Some(value) = outcome.update_ratio {
            self.update_ratio += value;
            self.update_ratio_count += 1;
        }
    }

    fn finish(
        self,
        avg_loss: f32,
        epoch_duration: Duration,
        hooks: &std::cell::RefCell<Vec<Box<dyn MetricHook>>>,
    ) -> super::MetricValues {
        let mut metrics = super::MetricValues::new();
        metrics.insert("avg_loss".into(), avg_loss);
        metrics.insert("epoch_duration_secs".into(), epoch_duration.as_secs_f32());
        if self.grad_norm_count != 0 {
            metrics.insert("grad_norm".into(), self.grad_norm / self.grad_norm_count as f32);
        }
        if self.update_ratio_count != 0 {
            metrics.insert(
                "update_ratio".into(),
                self.update_ratio / self.update_ratio_count as f32,
            );
        }
        if self.forward_count != 0 {
            metrics.insert("forward_secs".into(), self.forward_secs / self.forward_count as f32);
        }
        if self.backward_count != 0 {
            metrics.insert("backward_secs".into(), self.backward_secs / self.backward_count as f32);
        }
        for hook in hooks.borrow().iter() {
            metrics.insert(hook.name().to_string(), hook.compute());
        }
        metrics
    }
}

struct ContextEpochOutcome {
    avg_loss: f32,
    metrics: super::MetricValues,
}

fn run_context_epoch<B>(
    core: &TrainerCore,
    epoch: &EpochContext,
    mut next_batch: impl FnMut() -> MlResult<Option<B>>,
    mut run_batch: impl FnMut(B, &BatchStartContext) -> MlResult<(ContextBatchOutcome, usize)>,
) -> MlResult<ContextEpochOutcome> {
    for hook in core.hooks.borrow_mut().iter_mut() { hook.reset()?; }
    let started = Instant::now();
    let mut weighted_loss = 0.0;
    let mut total_weight = 0usize;
    let mut diagnostics = ContextEpochDiagnostics::default();
    let mut batch_index = 0usize;
    while let Some(batch) = next_batch()? {
        let batch_context = BatchStartContext {
            paradigm: epoch.paradigm,
            epoch: epoch.epoch,
            batch: batch_index + 1,
            total_epochs: epoch.total_epochs,
            total_batches: epoch.total_batches,
            episode: None,
        };
        let (outcome, weight) = run_batch(batch, &batch_context)?;
        let weight = weight.max(1);
        weighted_loss += outcome.loss * weight as f32;
        total_weight += weight;
        diagnostics.record(&outcome);
        batch_index += 1;
    }
    if batch_index == 0 { return Err(DataError::NoBatches.into()); }
    let avg_loss = weighted_loss / total_weight as f32;
    Ok(ContextEpochOutcome {
        avg_loss,
        metrics: diagnostics.finish(avg_loss, started.elapsed(), &core.hooks),
    })
}

fn context_grad_norm(parameters: &[&ContextParameter]) -> MlResult<f32> {
    let mut squared = 0.0;
    for parameter in parameters {
        if let Some(gradient) = parameter.grad()? {
            squared += gradient.data.iter().map(|value| value * value).sum::<f32>();
        }
    }
    Ok(squared.sqrt())
}

fn context_has_invalid_grad(parameters: &[&ContextParameter]) -> MlResult<bool> {
    for parameter in parameters {
        if let Some(gradient) = parameter.grad()? {
            if gradient.data.iter().any(|value| !value.is_finite()) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn context_update_ratio(parameters: &[&ContextParameter], learning_rate: f32) -> MlResult<f32> {
    let mut update_squared = 0.0;
    let mut weight_squared = 0.0;
    for parameter in parameters {
        if let Some(gradient) = parameter.grad()? {
            update_squared += gradient.data.iter().map(|value| {
                let update = learning_rate * value;
                update * update
            }).sum::<f32>();
        }
        weight_squared += parameter.tensor().to_vec()?.iter().map(|value| value * value).sum::<f32>();
    }
    Ok(if weight_squared > 1e-12 {
        update_squared.sqrt() / weight_squared.sqrt()
    } else {
        0.0
    })
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
        Ok(self.train_dataset_epoch(model, optimizer, dataset, 1, 1)?.avg_loss)
    }

    fn train_dataset_epoch<M: ContextSupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        dataset: &ContextSupervisedDataset<'_>,
        epoch: usize,
        total_epochs: usize,
    ) -> MlResult<ContextEpochOutcome> {
        self.validate(model, optimizer, dataset)?;
        let mut order = (0..dataset.len()).collect::<Vec<_>>();
        self.core.runtime.shuffle(&mut order);
        let mut cursor = 0usize;
        let epoch_context = EpochContext {
            paradigm: "supervised", epoch, total_epochs, total_batches: Some(dataset.len()),
        };
        run_context_epoch(
            &self.core,
            &epoch_context,
            || {
                let Some(&sample_index) = order.get(cursor) else { return Ok(None); };
                cursor += 1;
                Ok(Some((dataset.inputs[sample_index], dataset.targets[sample_index])))
            },
            |(input, target), batch| {
                let outcome = self.run_batch(model, optimizer, input, target, batch)?;
                Ok((outcome, target.shape()?.first().copied().unwrap_or(1)))
            },
        )
    }

    pub fn train_loader_epoch<M: ContextSupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        loader: &mut ContextSupervisedDataLoader<'_>,
        epoch: usize,
    ) -> MlResult<f32> {
        Ok(self.train_loader_epoch_inner(model, optimizer, loader, epoch, epoch + 1)?.avg_loss)
    }

    fn train_loader_epoch_inner<M: ContextSupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        loader: &mut ContextSupervisedDataLoader<'_>,
        epoch_index: usize,
        total_epochs: usize,
    ) -> MlResult<ContextEpochOutcome> {
        self.validate(model, optimizer, &loader.dataset)?;
        const EPOCH_MIX: u64 = 0x9E37_79B9_7F4A_7C15;
        self.core.runtime.reseed(self.core.config.seed ^ (epoch_index as u64).wrapping_mul(EPOCH_MIX));
        loader.begin_epoch(&self.core.runtime);
        let epoch_context = EpochContext {
            paradigm: "supervised",
            epoch: epoch_index + 1,
            total_epochs,
            total_batches: Some(ContextSupervisedDataLoader::batch_count(loader)),
        };
        run_context_epoch(
            &self.core,
            &epoch_context,
            || loader.next_batch(),
            |batch, context| {
                let outcome = self.run_batch(
                    model, optimizer, &batch.inputs, &batch.targets, context,
                )?;
                Ok((outcome, batch.samples))
            },
        )
    }

    fn run_batch<M: ContextSupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        input: &ContextVariable,
        target: &ContextTensor,
        batch_context: &BatchStartContext,
    ) -> MlResult<ContextBatchOutcome> {
        validate_training_parameters(&self.context, model, optimizer)?;
        let scope = self.context.begin_training_scope()?;
        let batch: MlResult<ContextBatchOutcome> = (|| {
            let track_timing = self.core.config.metrics.fw_bw_timing;
            let forward_started = track_timing.then(Instant::now);
            let (prediction, loss) = model.forward_loss(input, target)?;
            let forward = forward_started.map(|started| started.elapsed());
            if loss.tensor().context_id() != self.context.id()
                || prediction.tensor().context_id() != self.context.id()
            { return Err(ContextError::Mismatch.into()); }
            let value = loss.tensor().item()?;
            if !value.is_finite() { return Err(MlError::StringError("non-finite context loss".into())); }
            let backward_started = track_timing.then(Instant::now);
            loss.backward()?;
            let backward = backward_started.map(|started| started.elapsed());
            let parameters = model.parameters();
            if self.nan_check_interval != usize::MAX
                && batch_context.batch % self.nan_check_interval == 0
            {
                if context_has_invalid_grad(&parameters)? {
                    return Err(MlError::StringError("non-finite context gradient".into()));
                }
            }
            let should_measure = self.core.config.batch_log_interval != usize::MAX
                && batch_context.batch % self.core.config.batch_log_interval == 0;
            let grad_norm = if should_measure && self.core.config.metrics.grad_norm {
                Some(context_grad_norm(&parameters)?)
            } else {
                None
            };
            let update_ratio = if should_measure && self.core.config.metrics.update_ratio {
                Some(context_update_ratio(&parameters, optimizer.lr())?)
            } else {
                None
            };
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

            Ok(ContextBatchOutcome { loss: value, forward, backward, grad_norm, update_ratio })
        })();
        let outcome = scope.finish(batch)?;
        self.core.notify_batch_end(&BatchEndContext { batch: batch_context.clone(), loss: outcome.loss });
        Ok(outcome)
    }

    fn reset_hooks(&self) -> MlResult<()> {
        for hook in self.core.hooks.borrow_mut().iter_mut() { hook.reset()?; }
        Ok(())
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
        let mut final_metrics = super::MetricValues::new();
        for epoch in 0..schedule.epochs {
            let epoch_context = EpochContext {
                paradigm: "supervised", epoch: epoch + 1, total_epochs: schedule.epochs,
                total_batches: Some(dataset.len()),
            };
            self.core.notify_epoch_start(&epoch_context);
            let outcome = match self.train_dataset_epoch(
                model, optimizer, dataset, epoch + 1, schedule.epochs,
            ) {
                Ok(outcome) => outcome,
                Err(error) => {
                    self.core.notify_train_error(&error.to_string());
                    return Err(error);
                }
            };
            final_loss = outcome.avg_loss;
            final_metrics = outcome.metrics;
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
                ).with_metrics(final_metrics));
            }
            previous = final_loss;
        }
        self.core.notify_train_end(&TrainEndContext {
            paradigm: "supervised", units_completed: schedule.epochs, interrupted: false,
        });
        Ok(TrainResult::epochs(
            StopReason::Completed, schedule.epochs, final_loss, started.elapsed(),
        ).with_metrics(final_metrics))
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
        let mut final_metrics = super::MetricValues::new();
        for epoch in 0..schedule.epochs {
            let epoch_context = EpochContext {
                paradigm: "supervised", epoch: epoch + 1, total_epochs: schedule.epochs,
                total_batches: Some(ContextSupervisedDataLoader::batch_count(loader)),
            };
            self.core.notify_epoch_start(&epoch_context);
            let outcome = match self.train_loader_epoch_inner(
                model, optimizer, loader, epoch, schedule.epochs,
            ) {
                Ok(outcome) => outcome,
                Err(error) => {
                    self.core.notify_train_error(&error.to_string());
                    return Err(error);
                }
            };
            final_loss = outcome.avg_loss;
            final_metrics = outcome.metrics;
            self.core.notify_epoch_end(&epoch_context);
            if schedule.convergence.should_stop(previous, final_loss) {
                self.core.notify_train_end(&TrainEndContext {
                    paradigm: "supervised", units_completed: epoch + 1, interrupted: false,
                });
                return Ok(TrainResult::epochs(
                    StopReason::Converged, epoch + 1, final_loss, started.elapsed(),
                ).with_metrics(final_metrics));
            }
            previous = final_loss;
        }
        self.core.notify_train_end(&TrainEndContext {
            paradigm: "supervised", units_completed: schedule.epochs, interrupted: false,
        });
        Ok(TrainResult::epochs(
            StopReason::Completed, schedule.epochs, final_loss, started.elapsed(),
        ).with_metrics(final_metrics))
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
    fn trainer_rejects_missing_and_extra_registered_parameters() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = regression_model(&context)?;
        let input = context.input(vec![1.0], &[1, 1])?;
        let target = context.tensor(vec![1.0], &[1, 1])?;
        let inputs = [&input];
        let targets = [&target];
        let dataset = ContextSupervisedDataset::new(&context, &inputs, &targets)?;
        let trainer = ContextSupervisedTrainer::silent(&context);
        let mut optimizer = ContextSGD::new(&context, 0.1)?;
        assert!(matches!(trainer.train_epoch(&mut model, &mut optimizer, &dataset),
            Err(MlError::OptimError(crate::optimizer::OptimError::ParameterSetMismatch { .. }))));
        optimizer.register_all(&model.parameters())?;
        let extra = ContextParameter::new(context.parameter(vec![0.0], &[])?);
        optimizer.register(&extra)?;
        assert!(matches!(trainer.train_epoch(&mut model, &mut optimizer, &dataset),
            Err(MlError::OptimError(crate::optimizer::OptimError::ParameterSetMismatch { .. }))));
        assert_eq!(context.graph_stats()?.graph_nodes, 0);
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

    struct NoOpOptimizer {
        parameters: Vec<ContextParameter>,
        context_id: ContextId,
        learning_rate: f32,
    }

    impl ContextOptimizer for NoOpOptimizer {
        fn register(&mut self, _parameter: &ContextParameter) -> MlResult<()> { Ok(()) }
        fn step(&mut self) -> MlResult<()> { Ok(()) }
        fn zero_grad(&self) -> MlResult<()> { Ok(()) }
        fn lr(&self) -> f32 { self.learning_rate }
        fn set_lr(&mut self, learning_rate: f32) -> MlResult<()> {
            self.learning_rate = learning_rate;
            Ok(())
        }
        fn registered_param_count(&self) -> usize { self.parameters.len() }
        fn registered_parameters(&self) -> Vec<&ContextParameter> { self.parameters.iter().collect() }
        fn context_id(&self) -> ContextId { self.context_id }
    }

    #[test]
    fn prebatched_loss_uses_legacy_sample_weighting() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = regression_model(&context)?;
        let inputs = [
            context.input(vec![0.0], &[1, 1])?,
            context.input(vec![0.0, 0.0, 0.0], &[3, 1])?,
        ];
        let targets = [
            context.tensor(vec![2.0], &[1, 1])?,
            context.tensor(vec![4.0, 4.0, 4.0], &[3, 1])?,
        ];
        let input_refs: Vec<_> = inputs.iter().collect();
        let target_refs: Vec<_> = targets.iter().collect();
        let dataset = ContextSupervisedDataset::new(&context, &input_refs, &target_refs)?;
        let mut optimizer = NoOpOptimizer { parameters: model.parameters().into_iter().cloned().collect(), context_id: context.id(), learning_rate: 0.1 };

        let loss = ContextSupervisedTrainer::silent(&context)
            .train_epoch(&mut model, &mut optimizer, &dataset)?;
        assert_eq!(loss, 13.0);
        Ok(())
    }

    #[test]
    fn verbose_preset_reports_legacy_diagnostic_metric_keys() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = regression_model(&context)?;
        let input = context.input(vec![2.0], &[1, 1])?;
        let target = context.tensor(vec![4.0], &[1, 1])?;
        let inputs = [&input];
        let targets = [&target];
        let dataset = ContextSupervisedDataset::new(&context, &inputs, &targets)?;
        let mut optimizer = ContextSGD::new(&context, 0.1)?;
        optimizer.register_all(&model.parameters())?;

        let result = ContextSupervisedTrainer::verbose(&context).fit(
            &mut model,
            &mut optimizer,
            &dataset,
            EpochSchedule::new(1)?,
        )?;
        for key in [
            "avg_loss",
            "epoch_duration_secs",
            "grad_norm",
            "update_ratio",
            "forward_secs",
            "backward_secs",
        ] {
            assert!(result.metrics.contains_key(key), "missing metric {key}");
        }
        Ok(())
    }
}
