//! Explicit-context semi_supervised training adapter.

use super::*;
use crate::trainer::ConsistencyRamp;

pub trait ContextSemiSupervisedModel: ContextTrainableModel {
    fn forward_loss(
        &mut self,
        labeled_input: &ContextVariable,
        labeled_target: &ContextTensor,
        unlabeled_input: &ContextVariable,
        lambda: f32,
    ) -> MlResult<(ContextVariable, ContextVariable)>;
}

pub struct ContextSemiSupervisedDataset<'a> {
    pub labeled_inputs: &'a [&'a ContextVariable],
    pub labeled_targets: &'a [&'a ContextTensor],
    pub unlabeled_inputs: &'a [&'a ContextVariable],
    context_id: ContextId,
}

impl<'a> ContextSemiSupervisedDataset<'a> {
    pub fn new(
        context: &ExecutionContext,
        labeled_inputs: &'a [&'a ContextVariable],
        labeled_targets: &'a [&'a ContextTensor],
        unlabeled_inputs: &'a [&'a ContextVariable],
    ) -> MlResult<Self> {
        if labeled_inputs.is_empty() || unlabeled_inputs.is_empty() {
            return Err(MlError::StringError("context semi-supervised datasets must not be empty".into()));
        }
        if labeled_inputs.len() != labeled_targets.len() {
            return Err(MlError::StringError("labeled input/target length mismatch".into()));
        }
        if labeled_inputs.iter().any(|value| value.tensor().context_id() != context.id())
            || labeled_targets.iter().any(|value| value.context_id() != context.id())
            || unlabeled_inputs.iter().any(|value| value.tensor().context_id() != context.id())
        {
            return Err(ContextError::Mismatch.into());
        }
        Ok(Self { labeled_inputs, labeled_targets, unlabeled_inputs, context_id: context.id() })
    }

    pub fn len(&self) -> usize { self.labeled_inputs.len().max(self.unlabeled_inputs.len()) }
    pub fn is_empty(&self) -> bool { self.labeled_inputs.is_empty() || self.unlabeled_inputs.is_empty() }
}

pub struct ContextSemiSupervisedBatch {
    pub labeled_inputs: ContextVariable,
    pub labeled_targets: ContextTensor,
    pub unlabeled_inputs: ContextVariable,
}

/// Context equivalent of the legacy same-shape semi_supervised stack collator.
pub struct ContextSemiSupervisedDataLoader<'a> {
    context: ExecutionContext,
    dataset: ContextSemiSupervisedDataset<'a>,
    labeled_batch_size: usize,
    unlabeled_batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    labeled_indices: Vec<usize>,
    unlabeled_indices: Vec<usize>,
    cursor: usize,
}

impl<'a> ContextSemiSupervisedDataLoader<'a> {
    pub fn new(context: &ExecutionContext, dataset: ContextSemiSupervisedDataset<'a>) -> MlResult<Self> {
        if dataset.context_id != context.id() { return Err(ContextError::Mismatch.into()); }
        let labeled_indices = (0..dataset.labeled_inputs.len()).collect();
        let unlabeled_indices = (0..dataset.unlabeled_inputs.len()).collect();
        Ok(Self {
            context: context.clone(), dataset, labeled_batch_size: 1, unlabeled_batch_size: 1,
            shuffle: true, drop_last: false, labeled_indices, unlabeled_indices, cursor: 0,
        })
    }

    pub fn labeled_batch_size(mut self, batch_size: usize) -> MlResult<Self> {
        if batch_size == 0 { return Err(DataError::InvalidBatchSize.into()); }
        if self.drop_last && self.dataset.labeled_inputs.len() < batch_size {
            return Err(DataError::NoBatches.into());
        }
        self.labeled_batch_size = batch_size;
        Ok(self)
    }
    pub fn unlabeled_batch_size(mut self, batch_size: usize) -> MlResult<Self> {
        if batch_size == 0 { return Err(DataError::InvalidBatchSize.into()); }
        if self.drop_last && self.dataset.unlabeled_inputs.len() < batch_size {
            return Err(DataError::NoBatches.into());
        }
        self.unlabeled_batch_size = batch_size;
        Ok(self)
    }
    pub fn shuffle(mut self, shuffle: bool) -> Self { self.shuffle = shuffle; self }
    pub fn drop_last(mut self, drop_last: bool) -> MlResult<Self> {
        if drop_last && (self.dataset.labeled_inputs.len() < self.labeled_batch_size
            || self.dataset.unlabeled_inputs.len() < self.unlabeled_batch_size)
        {
            return Err(DataError::NoBatches.into());
        }
        self.drop_last = drop_last;
        Ok(self)
    }
    pub fn batch_count(&self) -> usize {
        context_batch_count(self.dataset.labeled_inputs.len(), self.labeled_batch_size, self.drop_last)
            .max(context_batch_count(
                self.dataset.unlabeled_inputs.len(), self.unlabeled_batch_size, self.drop_last,
            ))
    }

    fn begin_epoch(&mut self, runtime: &TrainingRuntime) {
        self.labeled_indices.clear();
        self.labeled_indices.extend(0..self.dataset.labeled_inputs.len());
        self.unlabeled_indices.clear();
        self.unlabeled_indices.extend(0..self.dataset.unlabeled_inputs.len());
        if self.shuffle {
            runtime.shuffle(&mut self.labeled_indices);
            runtime.shuffle(&mut self.unlabeled_indices);
        }
        self.cursor = 0;
    }

    fn next_batch(&mut self) -> MlResult<Option<ContextSemiSupervisedBatch>> {
        if self.cursor >= ContextSemiSupervisedDataLoader::batch_count(self) { return Ok(None); }
        let labeled = context_batch_indices(
            &self.labeled_indices, self.labeled_batch_size, self.cursor, self.drop_last,
        );
        let unlabeled = context_batch_indices(
            &self.unlabeled_indices, self.unlabeled_batch_size, self.cursor, self.drop_last,
        );
        self.cursor += 1;
        let labeled_inputs = labeled.iter()
            .map(|&index| self.dataset.labeled_inputs[index].tensor())
            .collect::<Vec<_>>();
        let labeled_targets = labeled.iter()
            .map(|&index| self.dataset.labeled_targets[index])
            .collect::<Vec<_>>();
        let unlabeled_inputs = unlabeled.iter()
            .map(|&index| self.dataset.unlabeled_inputs[index].tensor())
            .collect::<Vec<_>>();
        let (labeled_data, labeled_shape) = stack_context_tensors(&labeled_inputs)?;
        let (target_data, target_shape) = stack_context_tensors(&labeled_targets)?;
        let (unlabeled_data, unlabeled_shape) = stack_context_tensors(&unlabeled_inputs)?;
        Ok(Some(ContextSemiSupervisedBatch {
            labeled_inputs: self.context.input(labeled_data, &labeled_shape)?,
            labeled_targets: self.context.tensor(target_data, &target_shape)?,
            unlabeled_inputs: self.context.input(unlabeled_data, &unlabeled_shape)?,
        }))
    }
}

impl BatchLoader for ContextSemiSupervisedDataLoader<'_> {
    type Batch = ContextSemiSupervisedBatch;

    fn begin_epoch(&mut self, _epoch: usize, runtime: &TrainingRuntime) -> MlResult<()> {
        ContextSemiSupervisedDataLoader::begin_epoch(self, runtime);
        Ok(())
    }

    fn next_batch(&mut self) -> MlResult<Option<Self::Batch>> {
        ContextSemiSupervisedDataLoader::next_batch(self)
    }

    fn batch_count(&self) -> Option<usize> {
        Some(ContextSemiSupervisedDataLoader::batch_count(self))
    }
}

fn context_batch_count(len: usize, size: usize, drop_last: bool) -> usize {
    if drop_last { len / size } else { len.div_ceil(size) }
}

fn context_batch_indices(indices: &[usize], size: usize, batch: usize, drop_last: bool) -> &[usize] {
    let count = context_batch_count(indices.len(), size, drop_last);
    let logical = batch % count;
    let start = logical * size;
    &indices[start..(start + size).min(indices.len())]
}

pub struct ContextSemiSupervisedTrainer {
    context: ExecutionContext,
    core: TrainerCore,
    nan_check_interval: usize,
    max_grad_norm: Option<f32>,
    ramp: ConsistencyRamp,
}

impl ContextSemiSupervisedTrainer {
    pub fn from_trainer(context: &ExecutionContext, trainer: Trainer) -> Self {
        let trainer: crate::trainer::SemiSupervisedTrainer = trainer.into();
        let nan_check_interval = trainer.core.config.nan_check_interval;
        Self {
            context: context.clone(),
            core: trainer.core,
            nan_check_interval,
            max_grad_norm: None,
            ramp: trainer.ramp,
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
    pub fn with_hook(self, hook: Box<dyn MetricHook>) -> Self {
        self.core.add_hook(hook);
        self
    }
    pub fn with_observer(self, observer: Box<dyn TrainingObserver>) -> Self {
        self.core.add_observer(observer);
        self
    }
    pub fn with_ramp(mut self, ramp: ConsistencyRamp) -> Self {
        self.ramp = ramp;
        self
    }
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.core.config.seed = seed;
        self.core.runtime.reseed(seed);
        self
    }
    pub fn check_finite_gradients(mut self, enabled: bool) -> Self {
        self.nan_check_interval = if enabled { 1 } else { usize::MAX };
        self
    }
    pub fn with_max_grad_norm(mut self, max_norm: f32) -> MlResult<Self> {
        if !max_norm.is_finite() || max_norm <= 0.0 {
            return Err(MlError::StringError("max_grad_norm must be finite and positive".into()));
        }
        self.max_grad_norm = Some(max_norm);
        Ok(self)
    }

    pub fn fit<M: ContextSemiSupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        dataset: &ContextSemiSupervisedDataset<'_>,
        schedule: EpochSchedule,
    ) -> MlResult<TrainResult> {
        self.validate(model, optimizer, dataset)?;
        let started = Instant::now();
        self.core.notify_train_start(&TrainStartContext {
            paradigm: "semi_supervised",
            total_units: schedule.epochs,
        });
        let mut previous = f32::INFINITY;
        let mut final_loss = f32::INFINITY;
        let mut final_metrics = super::super::MetricValues::new();
        for epoch_index in 0..schedule.epochs {
            self.core.begin_epoch(epoch_index);
            let epoch = EpochContext {
                paradigm: "semi_supervised",
                epoch: epoch_index + 1,
                total_epochs: schedule.epochs,
                total_batches: Some(dataset.len()),
            };
            self.core.notify_epoch_start(&epoch);
            let outcome = match self.run_epoch(model, optimizer, dataset, &epoch) {
                Ok(outcome) => outcome,
                Err(error) => {
                    self.core.notify_train_error(&error.to_string());
                    return Err(error);
                }
            };
            self.core.notify_epoch_end(&epoch);
            final_loss = outcome.avg_loss;
            final_metrics = outcome.metrics;
            if schedule.convergence.should_stop(previous, final_loss) {
                self.core.notify_train_end(&TrainEndContext {
                    paradigm: "semi_supervised",
                    units_completed: epoch_index + 1,
                    interrupted: false,
                });
                return Ok(TrainResult::epochs(
                    StopReason::Converged,
                    epoch_index + 1,
                    final_loss,
                    started.elapsed(),
                ).with_metrics(final_metrics));
            }
            previous = final_loss;
        }
        self.core.notify_train_end(&TrainEndContext {
            paradigm: "semi_supervised",
            units_completed: schedule.epochs,
            interrupted: false,
        });
        Ok(TrainResult::epochs(
            StopReason::Completed,
            schedule.epochs,
            final_loss,
            started.elapsed(),
        ).with_metrics(final_metrics))
    }

    pub fn fit_loader<M: ContextSemiSupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        loader: &mut ContextSemiSupervisedDataLoader<'_>,
        schedule: EpochSchedule,
    ) -> MlResult<TrainResult> {
        self.validate(model, optimizer, &loader.dataset)?;
        let started = Instant::now();
        self.core.notify_train_start(&TrainStartContext {
            paradigm: "semi_supervised",
            total_units: schedule.epochs,
        });
        let mut previous = f32::INFINITY;
        let mut final_loss = f32::INFINITY;
        let mut final_metrics = super::super::MetricValues::new();
        for epoch_index in 0..schedule.epochs {
            self.core.begin_epoch(epoch_index);
            loader.begin_epoch(&self.core.runtime);
            let epoch = EpochContext {
                paradigm: "semi_supervised",
                epoch: epoch_index + 1,
                total_epochs: schedule.epochs,
                total_batches: Some(ContextSemiSupervisedDataLoader::batch_count(loader)),
            };
            self.core.notify_epoch_start(&epoch);
            let outcome = match self.run_loader_epoch(model, optimizer, loader, &epoch) {
                Ok(outcome) => outcome,
                Err(error) => {
                    self.core.notify_train_error(&error.to_string());
                    return Err(error);
                }
            };
            self.core.notify_epoch_end(&epoch);
            final_loss = outcome.avg_loss;
            final_metrics = outcome.metrics;
            if schedule.convergence.should_stop(previous, final_loss) {
                self.core.notify_train_end(&TrainEndContext {
                    paradigm: "semi_supervised",
                    units_completed: epoch_index + 1,
                    interrupted: false,
                });
                return Ok(TrainResult::epochs(
                    StopReason::Converged, epoch_index + 1, final_loss, started.elapsed(),
                ).with_metrics(final_metrics));
            }
            previous = final_loss;
        }
        self.core.notify_train_end(&TrainEndContext {
            paradigm: "semi_supervised",
            units_completed: schedule.epochs,
            interrupted: false,
        });
        Ok(TrainResult::epochs(
            StopReason::Completed, schedule.epochs, final_loss, started.elapsed(),
        ).with_metrics(final_metrics))
    }

    fn run_epoch<M: ContextSemiSupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        dataset: &ContextSemiSupervisedDataset<'_>,
        epoch: &EpochContext,
    ) -> MlResult<ContextEpochOutcome> {
        for hook in self.core.hooks.borrow_mut().iter_mut() { hook.reset()?; }
        let started = Instant::now();
        let mut weighted_loss = 0.0;
        let mut samples = 0usize;
        let mut diagnostics = ContextEpochDiagnostics::default();
        let lambda = self.ramp.value(epoch.epoch - 1);
        let mut labeled_order = (0..dataset.labeled_inputs.len()).collect::<Vec<_>>();
        let mut unlabeled_order = (0..dataset.unlabeled_inputs.len()).collect::<Vec<_>>();
        self.core.runtime.shuffle(&mut labeled_order);
        self.core.runtime.shuffle(&mut unlabeled_order);
        for batch_index in 0..dataset.len() {
            let labeled_index = labeled_order[batch_index % labeled_order.len()];
            let unlabeled_index = unlabeled_order[batch_index % unlabeled_order.len()];
            let batch = BatchStartContext {
                paradigm: "semi_supervised",
                epoch: epoch.epoch,
                batch: batch_index + 1,
                total_epochs: epoch.total_epochs,
                total_batches: epoch.total_batches,
                episode: None,
            };
            let outcome = self.run_batch(
                model,
                optimizer,
                dataset.labeled_inputs[labeled_index],
                dataset.labeled_targets[labeled_index],
                dataset.unlabeled_inputs[unlabeled_index],
                lambda,
                &batch,
            )?;
            let weight = outcome.samples.max(1);
            weighted_loss += outcome.batch.loss * weight as f32;
            samples += weight;
            diagnostics.record(&outcome.batch);
        }
        let avg_loss = weighted_loss / samples as f32;
        Ok(ContextEpochOutcome {
            avg_loss,
            metrics: diagnostics.finish(avg_loss, started.elapsed(), &self.core.hooks),
        })
    }

    fn run_loader_epoch<M: ContextSemiSupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        loader: &mut ContextSemiSupervisedDataLoader<'_>,
        epoch: &EpochContext,
    ) -> MlResult<ContextEpochOutcome> {
        for hook in self.core.hooks.borrow_mut().iter_mut() { hook.reset()?; }
        let started = Instant::now();
        let mut weighted_loss = 0.0;
        let mut samples = 0usize;
        let mut diagnostics = ContextEpochDiagnostics::default();
        let mut batch_index = 0usize;
        let lambda = self.ramp.value(epoch.epoch - 1);
        while let Some(batch) = loader.next_batch()? {
            let context = BatchStartContext {
                paradigm: "semi_supervised",
                epoch: epoch.epoch,
                batch: batch_index + 1,
                total_epochs: epoch.total_epochs,
                total_batches: epoch.total_batches,
                episode: None,
            };
            let outcome = self.run_batch(
                model,
                optimizer,
                &batch.labeled_inputs,
                &batch.labeled_targets,
                &batch.unlabeled_inputs,
                lambda,
                &context,
            )?;
            let weight = outcome.samples.max(1);
            weighted_loss += outcome.batch.loss * weight as f32;
            samples += weight;
            diagnostics.record(&outcome.batch);
            batch_index += 1;
        }
        if batch_index == 0 { return Err(DataError::NoBatches.into()); }
        let avg_loss = weighted_loss / samples as f32;
        Ok(ContextEpochOutcome {
            avg_loss,
            metrics: diagnostics.finish(avg_loss, started.elapsed(), &self.core.hooks),
        })
    }

    fn run_batch<M: ContextSemiSupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        labeled_input: &ContextVariable,
        labeled_target: &ContextTensor,
        unlabeled_input: &ContextVariable,
        lambda: f32,
        batch_context: &BatchStartContext,
    ) -> MlResult<SemiSupervisedBatchOutcome> {
        validate_training_parameters(&self.context, model, optimizer)?;
        let scope = self.context.begin_training_scope()?;
        let result = (|| {
            let track_timing = self.core.config.metrics.fw_bw_timing;
            let forward_started = track_timing.then(Instant::now);
            let (prediction, loss) = model.forward_loss(
                labeled_input, labeled_target, unlabeled_input, lambda,
            )?;
            let forward = forward_started.map(|started| started.elapsed());
            if prediction.tensor().context_id() != self.context.id()
                || loss.tensor().context_id() != self.context.id()
            {
                return Err(ContextError::Mismatch.into());
            }
            let samples = labeled_target.shape()?.first().copied().unwrap_or(1).max(1);
            let value = loss.tensor().item()?;
            if !value.is_finite() {
                return Err(MlError::StringError("non-finite context semi_supervised loss".into()));
            }
            let backward_started = track_timing.then(Instant::now);
            loss.backward()?;
            let backward = backward_started.map(|started| started.elapsed());
            let parameters = model.parameters();
            if self.nan_check_interval != usize::MAX
                && batch_context.batch % self.nan_check_interval == 0
                && context_has_invalid_grad(&parameters)?
            {
                return Err(MlError::StringError("non-finite context gradient".into()));
            }
            let should_measure = self.core.config.batch_log_interval != usize::MAX
                && batch_context.batch % self.core.config.batch_log_interval == 0;
            let grad_norm = if should_measure && self.core.config.metrics.grad_norm {
                Some(context_grad_norm(&parameters)?)
            } else { None };
            let update_ratio = if should_measure && self.core.config.metrics.update_ratio {
                Some(context_update_ratio(&parameters, optimizer.lr())?)
            } else { None };
            if let Some(max_norm) = self.max_grad_norm {
                clip_context_grad_norm(&self.context, &parameters, max_norm)?;
            }
            if self.core.hook_count() != 0 {
                let prediction = GlobalTensor::from_vec(
                    prediction.tensor().to_vec()?,
                    &prediction.tensor().shape()?,
                )?;
                let target = GlobalTensor::from_vec(
                    labeled_target.to_vec()?,
                    &labeled_target.shape()?,
                )?;
                let hook_context = BatchContext {
                    batch_idx: batch_context.batch - 1,
                    pred: Some(&prediction),
                    target: Some(&target),
                    loss: value,
                    n_tokens: None,
                    lambda: Some(lambda),
                    lr: optimizer.lr(),
                };
                for hook in self.core.hooks.borrow_mut().iter_mut() { hook.update(&hook_context)?; }
            }
            optimizer.step()?;
            optimizer.zero_grad()?;

            Ok(SemiSupervisedBatchOutcome {
                batch: ContextBatchOutcome {
                    loss: value, forward, backward, grad_norm, update_ratio,
                },
                samples,
            })
        })();
        let outcome = scope.finish(result)?;
        self.core.notify_batch_end(&BatchEndContext { batch: batch_context.clone(), loss: outcome.batch.loss });
        Ok(outcome)
    }

    fn validate<M: ContextTrainableModel + ?Sized>(
        &self,
        model: &M,
        optimizer: &dyn ContextOptimizer,
        dataset: &ContextSemiSupervisedDataset<'_>,
    ) -> MlResult<()> {
        if model.context_id() != self.context.id()
            || optimizer.context_id() != self.context.id()
            || dataset.context_id != self.context.id()
            || model.parameters().iter().any(|parameter| parameter.context_id() != self.context.id())
        {
            return Err(ContextError::Mismatch.into());
        }
        Ok(())
    }
}

struct SemiSupervisedBatchOutcome {
    batch: ContextBatchOutcome,
    samples: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::Reduction;
    use crate::nn::{ContextLayer, ContextLinear};
    use crate::RequiresGrad;

    struct ToySemiSupervisedModel {
        context: ExecutionContext,
        linear: ContextLinear,
        lambdas: Vec<f32>,
    }

    impl ContextTrainableModel for ToySemiSupervisedModel {
        fn context_id(&self) -> ContextId { self.context.id() }
        fn parameters(&self) -> Vec<&ContextParameter> { self.linear.parameters() }
    }

    impl ContextSemiSupervisedModel for ToySemiSupervisedModel {
        fn forward_loss(
            &mut self,
            labeled_input: &ContextVariable,
            labeled_target: &ContextTensor,
            unlabeled_input: &ContextVariable,
            lambda: f32,
        ) -> MlResult<(ContextVariable, ContextVariable)> {
            self.lambdas.push(lambda);
            let prediction = self.linear.apply(labeled_input)?;
            let supervised = self.context.mse_loss_variable(
                &prediction, labeled_target, Reduction::Mean,
            )?;
            let unlabeled_prediction = self.linear.apply(unlabeled_input)?;
            let consistency = self.context.mse_loss_variable(
                &unlabeled_prediction, unlabeled_input.tensor(), Reduction::Mean,
            )?;
            let scale = self.context.variable(vec![lambda], &[], RequiresGrad::No)?;
            let weighted = self.context.mul_variable(&consistency, &scale)?;
            let total = self.context.add_variable(&supervised, &weighted)?;
            Ok((prediction, total))
        }
    }

    struct NoOpOptimizer { context_id: ContextId, parameters: Vec<ContextParameter> }
    impl ContextOptimizer for NoOpOptimizer {
        fn register(&mut self, _parameter: &ContextParameter) -> MlResult<()> { Ok(()) }
        fn step(&mut self) -> MlResult<()> { Ok(()) }
        fn zero_grad(&self) -> MlResult<()> { Ok(()) }
        fn lr(&self) -> f32 { 0.1 }
        fn set_lr(&mut self, _learning_rate: f32) -> MlResult<()> { Ok(()) }
        fn registered_param_count(&self) -> usize { self.parameters.len() }
        fn registered_parameters(&self) -> Vec<&ContextParameter> { self.parameters.iter().collect() }
        fn context_id(&self) -> ContextId { self.context_id }
    }

    fn model(context: &ExecutionContext) -> MlResult<ToySemiSupervisedModel> {
        let model = ToySemiSupervisedModel {
            context: context.clone(),
            linear: ContextLinear::new(context, 1, 1, "semi.linear")?,
            lambdas: Vec::new(),
        };
        context.replace_parameter(
            model.linear.weight().variable(), GlobalTensor::from_vec(vec![0.0], &[1, 1])?,
        )?;
        context.replace_parameter(
            model.linear.bias().variable(), GlobalTensor::from_vec(vec![0.0], &[1])?,
        )?;
        Ok(model)
    }

    #[test]
    fn preset_and_ramp_policy_match_legacy() {
        let context = ExecutionContext::new();
        assert_eq!(ContextSemiSupervisedTrainer::silent(&context).core.hook_count(), 0);
        assert_eq!(ContextSemiSupervisedTrainer::default(&context).core.hook_count(), 1);
        let ramp = ConsistencyRamp::Sigmoid { max_weight: 1.0, ramp_epochs: 30 };
        assert!(ramp.value(0) < ramp.value(15));
        assert!(ramp.value(15) < ramp.value(30));
    }

    #[test]
    fn prebatched_path_wraps_shorter_side_and_applies_lambda() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = model(&context)?;
        let labeled = [context.input(vec![0.0], &[1, 1])?];
        let targets = [context.tensor(vec![1.0], &[1, 1])?];
        let unlabeled = [
            context.input(vec![1.0], &[1, 1])?,
            context.input(vec![2.0], &[1, 1])?,
            context.input(vec![3.0], &[1, 1])?,
        ];
        let labeled_refs: Vec<_> = labeled.iter().collect();
        let target_refs: Vec<_> = targets.iter().collect();
        let unlabeled_refs: Vec<_> = unlabeled.iter().collect();
        let dataset = ContextSemiSupervisedDataset::new(
            &context, &labeled_refs, &target_refs, &unlabeled_refs,
        )?;
        let mut optimizer = NoOpOptimizer { parameters: model.parameters().into_iter().cloned().collect(), context_id: context.id() };
        let result = ContextSemiSupervisedTrainer::silent(&context)
            .with_ramp(ConsistencyRamp::Constant(0.5))
            .fit(&mut model, &mut optimizer, &dataset, EpochSchedule::new(1)?)?;
        assert!((result.final_loss - 10.0 / 3.0).abs() < 1e-6);
        assert_eq!(model.lambdas, vec![0.5; 3]);
        assert_eq!(context.graph_stats()?.graph_nodes, 0);
        Ok(())
    }

    #[test]
    fn loader_cycles_shorter_batch_stream() -> MlResult<()> {
        let context = ExecutionContext::new();
        let labeled = [
            context.input(vec![1.0], &[1])?, context.input(vec![2.0], &[1])?,
        ];
        let targets = [
            context.tensor(vec![1.0], &[1])?, context.tensor(vec![2.0], &[1])?,
        ];
        let unlabeled = [
            context.input(vec![1.0], &[1])?, context.input(vec![2.0], &[1])?,
            context.input(vec![3.0], &[1])?, context.input(vec![4.0], &[1])?,
            context.input(vec![5.0], &[1])?,
        ];
        let labeled_refs: Vec<_> = labeled.iter().collect();
        let target_refs: Vec<_> = targets.iter().collect();
        let unlabeled_refs: Vec<_> = unlabeled.iter().collect();
        let mut loader = ContextSemiSupervisedDataLoader::new(
            &context,
            ContextSemiSupervisedDataset::new(
                &context, &labeled_refs, &target_refs, &unlabeled_refs,
            )?,
        )?.labeled_batch_size(2)?.unlabeled_batch_size(2)?.shuffle(false);
        assert_eq!(loader.batch_count(), 3);
        loader.begin_epoch(&TrainingRuntime::new(1));
        for _ in 0..3 {
            let batch = loader.next_batch()?.expect("cycled batch");
            assert_eq!(batch.labeled_inputs.tensor().shape()?, vec![2, 1]);
        }
        assert!(loader.next_batch()?.is_none());
        Ok(())
    }
}
