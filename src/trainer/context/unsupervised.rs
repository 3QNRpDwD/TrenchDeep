//! Explicit-context unsupervised training adapter.

use super::*;

pub trait ContextUnsupervisedModel: ContextTrainableModel {
    fn forward_loss(
        &mut self,
        sample: &ContextVariable,
    ) -> MlResult<(ContextVariable, ContextVariable)>;
}

pub struct ContextUnsupervisedDataset<'a> {
    pub samples: &'a [&'a ContextVariable],
    context_id: ContextId,
}

impl<'a> ContextUnsupervisedDataset<'a> {
    pub fn new(context: &ExecutionContext, samples: &'a [&'a ContextVariable]) -> MlResult<Self> {
        if samples.is_empty() {
            return Err(MlError::StringError("context unsupervised dataset must not be empty".into()));
        }
        if samples.iter().any(|sample| sample.tensor().context_id() != context.id()) {
            return Err(ContextError::Mismatch.into());
        }
        Ok(Self { samples, context_id: context.id() })
    }

    pub fn len(&self) -> usize { self.samples.len() }
    pub fn is_empty(&self) -> bool { self.samples.is_empty() }
}

pub struct ContextUnsupervisedBatch {
    pub samples: ContextVariable,
}

/// Context equivalent of the legacy same-shape unsupervised stack collator.
pub struct ContextUnsupervisedDataLoader<'a> {
    context: ExecutionContext,
    dataset: ContextUnsupervisedDataset<'a>,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    indices: Vec<usize>,
    cursor: usize,
}

impl<'a> ContextUnsupervisedDataLoader<'a> {
    pub fn new(context: &ExecutionContext, dataset: ContextUnsupervisedDataset<'a>) -> MlResult<Self> {
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

    fn next_batch(&mut self) -> MlResult<Option<ContextUnsupervisedBatch>> {
        if self.cursor >= self.indices.len() { return Ok(None); }
        let end = (self.cursor + self.batch_size).min(self.indices.len());
        if self.drop_last && end - self.cursor < self.batch_size {
            self.cursor = self.indices.len();
            return Ok(None);
        }
        let selected = &self.indices[self.cursor..end];
        self.cursor = end;
        let tensors = selected.iter()
            .map(|&index| self.dataset.samples[index].tensor())
            .collect::<Vec<_>>();
        let (data, shape) = stack_context_tensors(&tensors)?;
        Ok(Some(ContextUnsupervisedBatch {
            samples: self.context.input(data, &shape)?,
        }))
    }
}

impl BatchLoader for ContextUnsupervisedDataLoader<'_> {
    type Batch = ContextUnsupervisedBatch;

    fn begin_epoch(&mut self, _epoch: usize, runtime: &TrainingRuntime) -> MlResult<()> {
        ContextUnsupervisedDataLoader::begin_epoch(self, runtime);
        Ok(())
    }

    fn next_batch(&mut self) -> MlResult<Option<Self::Batch>> {
        ContextUnsupervisedDataLoader::next_batch(self)
    }

    fn batch_count(&self) -> Option<usize> {
        Some(ContextUnsupervisedDataLoader::batch_count(self))
    }
}

pub struct ContextUnsupervisedTrainer {
    context: ExecutionContext,
    core: TrainerCore,
    nan_check_interval: usize,
    max_grad_norm: Option<f32>,
}

impl ContextUnsupervisedTrainer {
    pub fn from_trainer(context: &ExecutionContext, trainer: Trainer) -> Self {
        let trainer: crate::trainer::UnsupervisedTrainer = trainer.into();
        let nan_check_interval = trainer.core.config.nan_check_interval;
        Self {
            context: context.clone(),
            core: trainer.core,
            nan_check_interval,
            max_grad_norm: None,
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

    pub fn fit<M: ContextUnsupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        dataset: &ContextUnsupervisedDataset<'_>,
        schedule: EpochSchedule,
    ) -> MlResult<TrainResult> {
        self.validate(model, optimizer, dataset)?;
        let started = Instant::now();
        self.core.notify_train_start(&TrainStartContext {
            paradigm: "unsupervised",
            total_units: schedule.epochs,
        });
        let mut previous = f32::INFINITY;
        let mut final_loss = f32::INFINITY;
        let mut final_metrics = super::super::MetricValues::new();
        for epoch_index in 0..schedule.epochs {
            self.core.begin_epoch(epoch_index);
            let epoch = EpochContext {
                paradigm: "unsupervised",
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
                    paradigm: "unsupervised",
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
            paradigm: "unsupervised",
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

    pub fn fit_loader<M: ContextUnsupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        loader: &mut ContextUnsupervisedDataLoader<'_>,
        schedule: EpochSchedule,
    ) -> MlResult<TrainResult> {
        self.validate(model, optimizer, &loader.dataset)?;
        let started = Instant::now();
        self.core.notify_train_start(&TrainStartContext {
            paradigm: "unsupervised",
            total_units: schedule.epochs,
        });
        let mut previous = f32::INFINITY;
        let mut final_loss = f32::INFINITY;
        let mut final_metrics = super::super::MetricValues::new();
        for epoch_index in 0..schedule.epochs {
            self.core.begin_epoch(epoch_index);
            loader.begin_epoch(&self.core.runtime);
            let epoch = EpochContext {
                paradigm: "unsupervised",
                epoch: epoch_index + 1,
                total_epochs: schedule.epochs,
                total_batches: Some(ContextUnsupervisedDataLoader::batch_count(loader)),
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
                    paradigm: "unsupervised",
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
            paradigm: "unsupervised",
            units_completed: schedule.epochs,
            interrupted: false,
        });
        Ok(TrainResult::epochs(
            StopReason::Completed, schedule.epochs, final_loss, started.elapsed(),
        ).with_metrics(final_metrics))
    }

    fn run_epoch<M: ContextUnsupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        dataset: &ContextUnsupervisedDataset<'_>,
        epoch: &EpochContext,
    ) -> MlResult<ContextEpochOutcome> {
        for hook in self.core.hooks.borrow_mut().iter_mut() { hook.reset()?; }
        let started = Instant::now();
        let mut weighted_loss = 0.0;
        let mut samples = 0usize;
        let mut diagnostics = ContextEpochDiagnostics::default();
        let mut order = (0..dataset.len()).collect::<Vec<_>>();
        self.core.runtime.shuffle(&mut order);
        for (batch_index, sample_index) in order.into_iter().enumerate() {
            let sample = dataset.samples[sample_index];
            let batch = BatchStartContext {
                paradigm: "unsupervised",
                epoch: epoch.epoch,
                batch: batch_index + 1,
                total_epochs: epoch.total_epochs,
                total_batches: epoch.total_batches,
                episode: None,
            };
            let outcome = self.run_batch(model, optimizer, sample, &batch)?;
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

    fn run_loader_epoch<M: ContextUnsupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        loader: &mut ContextUnsupervisedDataLoader<'_>,
        epoch: &EpochContext,
    ) -> MlResult<ContextEpochOutcome> {
        for hook in self.core.hooks.borrow_mut().iter_mut() { hook.reset()?; }
        let started = Instant::now();
        let mut weighted_loss = 0.0;
        let mut samples = 0usize;
        let mut diagnostics = ContextEpochDiagnostics::default();
        let mut batch_index = 0usize;
        while let Some(batch) = loader.next_batch()? {
            let context = BatchStartContext {
                paradigm: "unsupervised",
                epoch: epoch.epoch,
                batch: batch_index + 1,
                total_epochs: epoch.total_epochs,
                total_batches: epoch.total_batches,
                episode: None,
            };
            let outcome = self.run_batch(model, optimizer, &batch.samples, &context)?;
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

    fn run_batch<M: ContextUnsupervisedModel>(
        &self,
        model: &mut M,
        optimizer: &mut dyn ContextOptimizer,
        sample: &ContextVariable,
        batch_context: &BatchStartContext,
    ) -> MlResult<UnsupervisedBatchOutcome> {
        validate_training_parameters(&self.context, model, optimizer)?;
        let scope = self.context.begin_training_scope()?;
        let result = (|| {
            let track_timing = self.core.config.metrics.fw_bw_timing;
            let forward_started = track_timing.then(Instant::now);
            let (prediction, loss) = model.forward_loss(sample)?;
            let forward = forward_started.map(|started| started.elapsed());
            if prediction.tensor().context_id() != self.context.id()
                || loss.tensor().context_id() != self.context.id()
            {
                return Err(ContextError::Mismatch.into());
            }
            let samples = sample.tensor().shape()?.first().copied().unwrap_or(1).max(1);
            let value = loss.tensor().item()?;
            if !value.is_finite() {
                return Err(MlError::StringError("non-finite context unsupervised loss".into()));
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
                let hook_context = BatchContext {
                    batch_idx: batch_context.batch - 1,
                    pred: Some(&prediction),
                    target: None,
                    loss: value,
                    n_tokens: None,
                    lambda: None,
                    lr: optimizer.lr(),
                };
                for hook in self.core.hooks.borrow_mut().iter_mut() { hook.update(&hook_context)?; }
            }
            optimizer.step()?;
            optimizer.zero_grad()?;

            Ok(UnsupervisedBatchOutcome {
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
        dataset: &ContextUnsupervisedDataset<'_>,
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

struct UnsupervisedBatchOutcome {
    batch: ContextBatchOutcome,
    samples: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::Reduction;
    use crate::nn::{ContextLayer, ContextLinear};

    struct ToyUnsupervisedModel {
        context: ExecutionContext,
        linear: ContextLinear,
    }

    impl ContextTrainableModel for ToyUnsupervisedModel {
        fn context_id(&self) -> ContextId { self.context.id() }
        fn parameters(&self) -> Vec<&ContextParameter> { self.linear.parameters() }
    }

    impl ContextUnsupervisedModel for ToyUnsupervisedModel {
        fn forward_loss(
            &mut self,
            sample: &ContextVariable,
        ) -> MlResult<(ContextVariable, ContextVariable)> {
            let prediction = self.linear.apply(sample)?;
            let loss = self.context.mse_loss_variable(
                &prediction,
                sample.tensor(),
                Reduction::Mean,
            )?;
            Ok((prediction, loss))
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

    fn model(context: &ExecutionContext) -> MlResult<ToyUnsupervisedModel> {
        let model = ToyUnsupervisedModel {
            context: context.clone(),
            linear: ContextLinear::new(context, 1, 1, "ar.linear")?,
        };
        context.replace_parameter(
            model.linear.weight().variable(),
            GlobalTensor::from_vec(vec![0.0], &[1, 1])?,
        )?;
        context.replace_parameter(
            model.linear.bias().variable(),
            GlobalTensor::from_vec(vec![0.0], &[1])?,
        )?;
        Ok(model)
    }

    #[test]
    fn presets_reuse_legacy_hook_policy() {
        let context = ExecutionContext::new();
        assert_eq!(ContextUnsupervisedTrainer::silent(&context).core.hook_count(), 0);
        assert_eq!(ContextUnsupervisedTrainer::default(&context).core.hook_count(), 0);
    }

    #[test]
    fn sample_weighted_loss_matches_legacy_contract() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = model(&context)?;
        let samples = [
            context.input(vec![1.0], &[1, 1])?,
            context.input(vec![2.0, 2.0, 2.0], &[3, 1])?,
        ];
        let refs: Vec<_> = samples.iter().collect();
        let dataset = ContextUnsupervisedDataset::new(&context, &refs)?;
        let mut optimizer = NoOpOptimizer { parameters: model.parameters().into_iter().cloned().collect(), context_id: context.id() };
        let result = ContextUnsupervisedTrainer::silent(&context)
            .fit(&mut model, &mut optimizer, &dataset, EpochSchedule::new(1)?)?;
        assert_eq!(result.final_loss, 3.25);
        assert_eq!(context.graph_stats()?.graph_nodes, 0);
        Ok(())
    }

    #[test]
    fn loader_stacks_shuffles_and_honors_drop_last() -> MlResult<()> {
        let context = ExecutionContext::new();
        let samples = [
            context.input(vec![1.0, 10.0], &[2])?,
            context.input(vec![2.0, 20.0], &[2])?,
            context.input(vec![3.0, 30.0], &[2])?,
        ];
        let refs: Vec<_> = samples.iter().collect();
        let mut loader = ContextUnsupervisedDataLoader::new(
            &context,
            ContextUnsupervisedDataset::new(&context, &refs)?,
        )?.batch_size(2)?.shuffle(false);
        assert_eq!(loader.batch_count(), 2);
        loader.begin_epoch(&TrainingRuntime::new(3));
        let first = loader.next_batch()?.expect("first batch");
        assert_eq!(first.samples.tensor().shape()?, vec![2, 2]);
        assert_eq!(first.samples.tensor().to_vec()?, vec![1.0, 10.0, 2.0, 20.0]);

        let dropped = ContextUnsupervisedDataLoader::new(
            &context,
            ContextUnsupervisedDataset::new(&context, &refs)?,
        )?.batch_size(2)?.drop_last(true)?;
        assert_eq!(dropped.batch_count(), 1);
        Ok(())
    }

    #[test]
    fn loader_fit_preserves_sample_weighting_and_cleans_graph() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = model(&context)?;
        let samples = [
            context.input(vec![1.0], &[1, 1])?,
            context.input(vec![2.0], &[1, 1])?,
            context.input(vec![3.0], &[1, 1])?,
        ];
        let refs: Vec<_> = samples.iter().collect();
        let mut loader = ContextUnsupervisedDataLoader::new(
            &context,
            ContextUnsupervisedDataset::new(&context, &refs)?,
        )?.batch_size(2)?.shuffle(false);
        let mut optimizer = NoOpOptimizer { parameters: model.parameters().into_iter().cloned().collect(), context_id: context.id() };
        let result = ContextUnsupervisedTrainer::silent(&context).fit_loader(
            &mut model,
            &mut optimizer,
            &mut loader,
            EpochSchedule::new(1)?,
        )?;
        assert!((result.final_loss - 14.0 / 3.0).abs() < 1e-6);
        assert_eq!(context.graph_stats()?.graph_nodes, 0);
        Ok(())
    }
}
