use super::service::{StepData, TrainingService, validate_parameters};
use super::*;
use crate::{RequiresGrad, optimizer::Optimizer};
use std::time::Instant;
pub struct StepResult {
    pub next_observation: TensorBuffer,
    pub reward: f32,
    pub done: bool,
}

pub trait Environment {
    fn reset(&mut self) -> MlResult<TensorBuffer>;
    fn step(&mut self, action: usize) -> MlResult<StepResult>;
    fn num_actions(&self) -> usize;
    fn observation_shape(&self) -> Vec<usize>;
}

pub trait RLModel: TrainableModel {
    fn policy_logits(&mut self, observation: &Variable) -> MlResult<Variable>;
    fn predict_policy_raw(&mut self, observation: &Tensor) -> MlResult<TensorBuffer> {
        let context = observation.execution_context()?;
        if self.context_id() != context.id() {
            return Err(ContextError::Mismatch.into());
        }
        context.no_grad(|| {
            let output = self.policy_logits(&observation.as_variable()?)?;
            context.validate(output.tensor())?;
            output.tensor().snapshot()
        })
    }
}

pub struct RLTrainer {
    service: TrainingService,
    gamma: f32,
    use_baseline: bool,
}
impl RLTrainer {
    pub fn new(ctx: &ExecutionContext) -> Self {
        Self::silent(ctx)
    }
    pub fn from_trainer(ctx: &ExecutionContext, trainer: Trainer) -> Self {
        Self {
            service: TrainingService::new(ctx, trainer),
            gamma: 0.99,
            use_baseline: true,
        }
    }
    pub fn silent(ctx: &ExecutionContext) -> Self {
        Self::from_trainer(ctx, Trainer::silent())
    }
    pub fn minimal(ctx: &ExecutionContext) -> Self {
        Self::from_trainer(ctx, Trainer::minimal())
    }
    pub fn default(ctx: &ExecutionContext) -> Self {
        Self::from_trainer(ctx, Trainer::default())
    }
    pub fn verbose(ctx: &ExecutionContext) -> Self {
        Self::from_trainer(ctx, Trainer::verbose())
    }
    pub fn with_gamma(mut self, gamma: f32) -> MlResult<Self> {
        if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            return Err(MlError::StringError("invalid gamma".into()));
        }
        self.gamma = gamma;
        Ok(self)
    }
    pub fn with_baseline(mut self, value: bool) -> Self {
        self.use_baseline = value;
        self
    }
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.service.core.config.seed = seed;
        self.service.core.runtime.reseed(seed);
        self
    }
    pub fn with_hook(self, hook: Box<dyn MetricHook>) -> Self {
        self.service.core.add_hook(hook);
        self
    }
    pub fn with_observer(self, observer: Box<dyn TrainingObserver>) -> Self {
        self.service.core.add_observer(observer);
        self
    }
    pub fn check_finite_gradients(mut self, enabled: bool) -> Self {
        self.service.core.config.nan_check_interval = if enabled { 1 } else { usize::MAX };
        self
    }
    pub fn with_max_grad_norm(mut self, max: f32) -> MlResult<Self> {
        if !max.is_finite() || max <= 0.0 {
            return Err(MlError::StringError("invalid max_grad_norm".into()));
        }
        self.service.max_grad_norm = Some(max);
        Ok(self)
    }
    pub fn fit<M: RLModel, E: Environment>(
        &self,
        model: &mut M,
        environment: &mut E,
        optimizer: &mut dyn Optimizer,
        schedule: EpisodeSchedule,
    ) -> MlResult<TrainResult> {
        self.fit_inner(model, environment, optimizer, schedule, None)
    }
    pub fn fit_checkpointed<M: RLModel + CheckpointableModel, E: Environment>(
        &self,
        model: &mut M,
        environment: &mut E,
        optimizer: &mut dyn Optimizer,
        schedule: EpisodeSchedule,
    ) -> MlResult<TrainResult> {
        self.fit_inner(
            model,
            environment,
            optimizer,
            schedule,
            Some(|model, path| model.save_checkpoint(path)),
        )
    }
    pub fn resume<M: RLModel, E: Environment>(
        &self,
        _model: &mut M,
        _environment: &mut E,
        _optimizer: &mut dyn Optimizer,
        _path: impl AsRef<std::path::Path>,
    ) -> MlResult<TrainResult> {
        Err(MlError::UnsupportedCapability {
            module: "trainer",
            capability: "optimizer snapshot and complete resume (P2)",
            operation: "resume",
        })
    }
    fn fit_inner<M: RLModel, E: Environment>(
        &self,
        model: &mut M,
        environment: &mut E,
        optimizer: &mut dyn Optimizer,
        schedule: EpisodeSchedule,
        save: Option<fn(&M, &std::path::Path) -> MlResult<()>>,
    ) -> MlResult<TrainResult> {
        validate_parameters(&self.service.context, model, optimizer)?;
        self.service.context.with_training_scope(|| Ok(()))?;
        if self.service.core.config.checkpoint_dir.is_some() && save.is_none() {
            return Err(MlError::UnsupportedCapability {
                module: "reinforcement",
                capability: "checkpointing requires fit_checkpointed",
                operation: "fit",
            });
        }
        let action_count = environment.num_actions();
        if action_count == 0 {
            return Err(MlError::StringError("environment has no actions".into()));
        }
        let started = Instant::now();
        let mut final_loss = 0.0;
        let mut metrics = MetricValues::new();
        let mut completed = 0;
        let mut interrupted = false;
        let progress =
            progress::EpochProgress::new(schedule.episodes, self.service.core.config.show_progress);
        self.service.core.notify_train_start(&TrainStartContext {
            paradigm: "reinforcement",
            total_units: schedule.episodes,
        });
        for episode in 0..schedule.episodes {
            self.service.core.begin_epoch(episode);
            for hook in self.service.core.hooks.borrow_mut().iter_mut() {
                hook.reset()?;
            }
            let epoch = EpochContext {
                paradigm: "reinforcement",
                epoch: episode + 1,
                total_epochs: schedule.episodes,
                total_batches: Some(1),
            };
            self.service.core.notify_epoch_start(&epoch);
            let batch = BatchStartContext {
                paradigm: "reinforcement",
                epoch: episode + 1,
                batch: 1,
                total_epochs: schedule.episodes,
                total_batches: Some(1),
                episode: Some(episode + 1),
            };
            let start = Instant::now();
            let result = self.service.context.with_training_scope(|| {
                let max_steps = schedule.max_steps_per_episode;
                let mut observation = environment.reset()?;
                if observation.shape() != environment.observation_shape() {
                    return Err(MlError::StringError(
                        "environment reset observation shape mismatch".into(),
                    ));
                }
                let mut trajectory = Vec::new();
                for _ in 0..max_steps {
                    let observation_tensor = self
                        .service
                        .context
                        .tensor(observation.data().to_vec(), observation.shape())?;
                    let logits = self
                        .service
                        .context
                        .no_grad(|| model.predict_policy_raw(&observation_tensor))?;
                    if logits.shape().last().copied() != Some(action_count)
                        || logits.data().len() != action_count
                        || logits.data().iter().any(|value| !value.is_finite())
                    {
                        return Err(MlError::StringError(
                            "policy action dimension mismatch".into(),
                        ));
                    }
                    let action = sample_categorical(logits.data(), self.service.core.random_f32());
                    let step = environment.step(action)?;
                    if step.next_observation.shape() != environment.observation_shape() {
                        return Err(MlError::StringError(
                            "environment step observation shape mismatch".into(),
                        ));
                    }
                    trajectory.push((observation, action, step.reward));
                    observation = step.next_observation;
                    if step.done {
                        break;
                    }
                }

                let steps = trajectory.len();
                let episode_return = trajectory.iter().map(|entry| entry.2).sum();
                let rewards = trajectory.iter().map(|entry| entry.2).collect::<Vec<_>>();
                let advantages = discounted_advantages(&rewards, self.gamma, self.use_baseline);

                let mut accumulated: Option<Variable> = None;
                for (index, (observation, action, _)) in trajectory.iter().enumerate() {
                    let input = self
                        .service
                        .context
                        .input(observation.data().to_vec(), observation.shape())?;
                    let logits = model.policy_logits(&input)?;
                    let logits_shape = logits.tensor().shape()?;
                    let logits_len = logits.tensor().to_vec()?.len();
                    if logits_shape.last().copied() != Some(action_count)
                        || logits_len != action_count
                    {
                        return Err(MlError::StringError(
                            "policy action dimension mismatch".into(),
                        ));
                    }
                    let mut target = vec![0.0; action_count];
                    target[*action] = 1.0;
                    let target = self.service.context.tensor(target, &logits_shape)?;
                    let negative_log_probability =
                        logits.softmax_cross_entropy(&target, crate::loss::Reduction::Mean)?;
                    let advantage = self.service.context.variable(
                        vec![advantages[index]],
                        &[],
                        RequiresGrad::No,
                    )?;
                    let weighted = negative_log_probability.mul(advantage.tensor())?;
                    accumulated = Some(match accumulated {
                        Some(current) => current.add(weighted.tensor())?,
                        None => weighted,
                    });
                }

                let loss = accumulated.ok_or(DataError::NoBatches)?;
                let outcome = self.service.finish_step(
                    model,
                    optimizer,
                    StepData {
                        loss,
                        prediction: None,
                        target: None,
                        weight: steps,
                        tokens: None,
                        lambda: None,
                    },
                    &batch,
                    start.elapsed(),
                )?;
                Ok((outcome, episode_return))
            });
            let (outcome, episode_return) = match result {
                Ok(v) => v,
                Err(e) => {
                    self.service.core.notify_train_error(&e.to_string());
                    return Err(e);
                }
            };

            if let Some(snapshot) = outcome.snapshot {
                self.service.core.deliver_graph_snapshot(&batch, snapshot);
            }
            final_loss = outcome.loss;
            metrics = outcome.metrics;
            metrics.insert("return".into(), episode_return);
            for hook in self.service.core.hooks.borrow().iter() {
                metrics.insert(hook.name().into(), hook.compute());
            }
            self.service.core.notify_batch_end(&BatchEndContext {
                batch,
                loss: final_loss,
            });
            self.service.core.notify_epoch_end(&epoch);
            completed = episode + 1;
            progress.inc();
            if checkpoint::interrupted() {
                interrupted = true;
                break;
            }
        }
        let mut result =
            TrainResult::episodes(completed, final_loss, started.elapsed()).with_metrics(metrics);
        if interrupted {
            result.stop_reason = StopReason::Interrupted;
            if let (Some(directory), Some(save)) = (&self.service.core.config.checkpoint_dir, save)
            {
                result.checkpoint = Some(checkpoint::save_model(
                    directory,
                    "reinforcement",
                    completed,
                    EpochSchedule::new(schedule.episodes)?,
                    final_loss,
                    optimizer.lr(),
                    self.service.core.config.seed,
                    |path| save(model, path),
                )?);
            }
            progress.finish_interrupted();
        } else {
            progress.finish_completed();
        }
        self.service.core.notify_train_end(&TrainEndContext {
            paradigm: "reinforcement",
            units_completed: completed,
            interrupted,
        });
        Ok(result)
    }
}
fn discounted_advantages(rewards: &[f32], gamma: f32, use_baseline: bool) -> Vec<f32> {
    let mut returns = vec![0.0; rewards.len()];
    let mut running = 0.0;
    for index in (0..rewards.len()).rev() {
        running = rewards[index] + gamma * running;
        returns[index] = running;
    }
    if use_baseline && returns.len() > 1 {
        let baseline = returns.iter().sum::<f32>() / returns.len() as f32;
        returns.iter_mut().for_each(|value| *value -= baseline);
    }
    returns
}

fn sample_categorical(logits: &[f32], sample: f32) -> usize {
    if logits.is_empty() {
        return 0;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponents = logits
        .iter()
        .map(|value| (value - max).exp())
        .collect::<Vec<_>>();
    let sum = exponents.iter().sum::<f32>();
    if !sum.is_finite() || sum == 0.0 {
        return ((sample * logits.len() as f32) as usize).min(logits.len() - 1);
    }
    let sample = sample.clamp(0.0, 1.0 - f32::EPSILON);
    let mut cumulative = 0.0;
    for (index, exponent) in exponents.iter().enumerate() {
        cumulative += exponent / sum;
        if sample < cumulative {
            return index;
        }
    }
    logits.len() - 1
}
