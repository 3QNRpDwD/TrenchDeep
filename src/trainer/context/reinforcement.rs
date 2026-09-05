//! 강화학습 아키텍처용 트레이너 네임스페이스.
//!
//! 지도·비지도·반지도 트레이너가 **고정 데이터셋** 위에서 학습하는 반면,
//! RL 트레이너는 **환경과의 상호작용을 통해 데이터(trajectory) 를 스스로 생성** 한다.
//!
//! ## 구성 요소
//!
//! - [`Environment`] : RL 환경 추상화 (reset / step).
//! - [`RLModel`]     : 정책 네트워크가 구현하는 인터페이스.
//! - [`RLTrainer`]   : REINFORCE (policy gradient) 학습 루프.
//!
//! ## 채택 알고리즘: REINFORCE (Monte-Carlo Policy Gradient)
//!
//! ```text
//! 1. π_θ 로 한 에피소드 rollout → (s₀,a₀,r₀), ..., (s_T,a_T,r_T)
//! 2. return G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
//! 3. advantage A_t = G_t - b(s)   (baseline = mean return)
//! 4. loss = -Σ_t A_t · log π_θ(a_t|s_t)
//! 5. ∇θ ← backward, optimizer.step
//! ```
//!
//! `-log π(a|s) = SoftmaxCrossEntropy(logits, one_hot(a))` 항등식을 이용해
//! 기존 `SoftmaxCrossEntropyLoss` 로 policy gradient 를 재활용한다.

use super::*;

use crate::trainer::EpisodeSchedule;
use crate::RequiresGrad;

pub struct ContextStepResult {
    pub next_observation: GlobalTensor<f32>,
    pub reward: f32,
    pub done: bool,
}

pub trait ContextEnvironment {
    fn reset(&mut self) -> MlResult<GlobalTensor<f32>>;
    fn step(&mut self, action: usize) -> MlResult<ContextStepResult>;
    fn num_actions(&self) -> usize;
    fn observation_shape(&self) -> Vec<usize>;
}

pub trait ContextRLModel: ContextTrainableModel {
    fn policy_logits(&mut self, observation: &ContextVariable) -> MlResult<ContextVariable>;
    fn predict_policy_raw(&mut self, observation: &ContextTensor) -> MlResult<GlobalTensor<f32>>;
}

pub struct ContextRLTrainer {
    context: ExecutionContext,
    core: TrainerCore,
    gamma: f32,
    use_baseline: bool,
    nan_check_interval: usize,
    max_grad_norm: Option<f32>,
}

impl ContextRLTrainer {
    pub fn from_trainer(context: &ExecutionContext, trainer: Trainer) -> Self {
        let trainer: crate::trainer::RLTrainer = trainer.into();
        let nan_check_interval = trainer.core.config.nan_check_interval;
        Self {
            context: context.clone(),
            core: trainer.core,
            gamma: trainer.gamma,
            use_baseline: trainer.use_baseline,
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
    pub fn with_gamma(mut self, gamma: f32) -> MlResult<Self> {
        if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            return Err(MlError::StringError("gamma must be finite and in [0, 1]".into()));
        }
        self.gamma = gamma;
        Ok(self)
    }
    pub fn with_baseline(mut self, use_baseline: bool) -> Self {
        self.use_baseline = use_baseline;
        self
    }
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.core.config.seed = seed;
        self.core.runtime.reseed(seed);
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

    pub fn fit<M: ContextRLModel, E: ContextEnvironment>(
        &self,
        model: &mut M,
        environment: &mut E,
        optimizer: &mut dyn ContextOptimizer,
        schedule: EpisodeSchedule,
    ) -> MlResult<TrainResult> {
        self.validate(model, optimizer)?;
        let action_count = environment.num_actions();
        if action_count == 0 {
            return Err(MlError::StringError("environment must expose at least one action".into()));
        }
        let started = Instant::now();
        self.core.notify_train_start(&TrainStartContext {
            paradigm: "reinforcement",
            total_units: schedule.episodes,
        });
        let mut final_loss = 0.0;
        let mut final_metrics = super::super::MetricValues::new();
        for episode_index in 0..schedule.episodes {
            let epoch = EpochContext {
                paradigm: "reinforcement",
                epoch: episode_index + 1,
                total_epochs: schedule.episodes,
                total_batches: Some(1),
            };
            let batch = BatchStartContext {
                paradigm: "reinforcement",
                epoch: episode_index + 1,
                batch: 1,
                total_epochs: schedule.episodes,
                total_batches: Some(1),
                episode: Some(episode_index + 1),
            };
            self.core.notify_epoch_start(&epoch);
            let episode_started = Instant::now();
            let outcome = match self.run_episode(
                model,
                environment,
                optimizer,
                action_count,
                schedule.max_steps_per_episode,
                episode_index,
            ) {
                Ok(outcome) => outcome,
                Err(error) => {
                    self.core.notify_train_error(&error.to_string());
                    return Err(error);
                }
            };
            final_loss = outcome.loss;
            final_metrics.insert("episode_return".into(), outcome.episode_return);
            final_metrics.insert("episode_steps".into(), outcome.steps as f32);
            final_metrics.insert("episode_duration_secs".into(), episode_started.elapsed().as_secs_f32());
            final_metrics.insert("loss".into(), outcome.loss);
            self.core.notify_batch_end(&BatchEndContext { batch, loss: outcome.loss });
            self.core.notify_epoch_end(&epoch);
        }
        self.core.notify_train_end(&TrainEndContext {
            paradigm: "reinforcement",
            units_completed: schedule.episodes,
            interrupted: false,
        });
        Ok(TrainResult::episodes(schedule.episodes, final_loss, started.elapsed())
            .with_metrics(final_metrics))
    }

    fn run_episode<M: ContextRLModel, E: ContextEnvironment>(
        &self,
        model: &mut M,
        environment: &mut E,
        optimizer: &mut dyn ContextOptimizer,
        action_count: usize,
        max_steps: usize,
        episode_index: usize,
    ) -> MlResult<ContextEpisodeOutcome> {
        validate_training_parameters(&self.context, model, optimizer)?;
        let scope = self.context.begin_training_scope()?;
        let result = (|| {
            let mut observation = environment.reset()?;
            if observation.shape != environment.observation_shape() {
                return Err(MlError::StringError("environment reset observation shape mismatch".into()));
            }
            let mut trajectory = Vec::new();
            for _ in 0..max_steps {
                let observation_tensor = self.context.tensor(
                    observation.data.clone(), &observation.shape,
                )?;
                let logits = self
                    .context
                    .no_grad(|| model.predict_policy_raw(&observation_tensor))?;
                if logits.shape.last().copied() != Some(action_count)
                    || logits.data.len() != action_count
                {
                    return Err(MlError::StringError("policy action dimension mismatch".into()));
                }
                let action = sample_categorical(&logits.data, self.core.random_f32());
                let step = environment.step(action)?;
                if step.next_observation.shape != environment.observation_shape() {
                    return Err(MlError::StringError("environment step observation shape mismatch".into()));
                }
                trajectory.push((observation, action, step.reward));
                observation = step.next_observation;
                if step.done { break; }
            }

            let steps = trajectory.len();
            let episode_return = trajectory.iter().map(|entry| entry.2).sum();
            let rewards = trajectory.iter().map(|entry| entry.2).collect::<Vec<_>>();
            let advantages = discounted_advantages(&rewards, self.gamma, self.use_baseline);

            let mut accumulated: Option<ContextVariable> = None;
            for (index, (observation, action, _)) in trajectory.iter().enumerate() {
                let input = self.context.input(observation.data.clone(), &observation.shape)?;
                let logits = model.policy_logits(&input)?;
                let logits_shape = logits.tensor().shape()?;
                let logits_len = logits.tensor().to_vec()?.len();
                if logits_shape.last().copied() != Some(action_count)
                    || logits_len != action_count
                {
                    return Err(MlError::StringError("policy action dimension mismatch".into()));
                }
                let mut target = vec![0.0; action_count];
                target[*action] = 1.0;
                let target = self.context.tensor(target, &logits_shape)?;
                let negative_log_probability = self.context.softmax_cross_entropy_variable(
                    &logits,
                    &target,
                    crate::loss::Reduction::Mean,
                )?;
                let advantage = self.context.variable(
                    vec![advantages[index]], &[], RequiresGrad::No,
                )?;
                let weighted = self.context.mul_variable(&negative_log_probability, &advantage)?;
                accumulated = Some(match accumulated {
                    Some(current) => self.context.add_variable(&current, &weighted)?,
                    None => weighted,
                });
            }

            let loss = if let Some(loss) = accumulated {
                let value = loss.tensor().item()?;
                if !value.is_finite() {
                    return Err(MlError::StringError("non-finite reinforcement loss".into()));
                }
                loss.backward()?;
                let parameters = model.parameters();
                if self.nan_check_interval != usize::MAX
                    && (episode_index + 1) % self.nan_check_interval == 0
                    && context_has_invalid_grad(&parameters)?
                {
                    return Err(MlError::StringError("non-finite context gradient".into()));
                }
                if let Some(max_norm) = self.max_grad_norm {
                    clip_context_grad_norm(&self.context, &parameters, max_norm)?;
                }
                optimizer.step()?;
                optimizer.zero_grad()?;
                value
            } else {
                0.0
            };
            Ok(ContextEpisodeOutcome { loss, episode_return, steps })
        })();
        scope.finish(result)
    }

    fn validate<M: ContextTrainableModel + ?Sized>(
        &self,
        model: &M,
        optimizer: &dyn ContextOptimizer,
    ) -> MlResult<()> {
        if model.context_id() != self.context.id()
            || optimizer.context_id() != self.context.id()
            || model.parameters().iter().any(|parameter| parameter.context_id() != self.context.id())
        {
            return Err(ContextError::Mismatch.into());
        }
        Ok(())
    }
}

struct ContextEpisodeOutcome {
    loss: f32,
    episode_return: f32,
    steps: usize,
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
    if logits.is_empty() { return 0; }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponents = logits.iter().map(|value| (value - max).exp()).collect::<Vec<_>>();
    let sum = exponents.iter().sum::<f32>();
    if !sum.is_finite() || sum == 0.0 {
        return ((sample * logits.len() as f32) as usize).min(logits.len() - 1);
    }
    let sample = sample.clamp(0.0, 1.0 - f32::EPSILON);
    let mut cumulative = 0.0;
    for (index, exponent) in exponents.iter().enumerate() {
        cumulative += exponent / sum;
        if sample < cumulative { return index; }
    }
    logits.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{ContextLayer, ContextLinear};
    use crate::optimizer::{ContextOptimizer, ContextSGD};

    struct OneStepBandit;

    impl ContextEnvironment for OneStepBandit {
        fn reset(&mut self) -> MlResult<GlobalTensor<f32>> {
            GlobalTensor::from_vec(vec![1.0], &[1, 1])
        }
        fn step(&mut self, action: usize) -> MlResult<ContextStepResult> {
            let _ = action;
            Ok(ContextStepResult {
                next_observation: GlobalTensor::from_vec(vec![1.0], &[1, 1])?,
                reward: 1.0,
                done: true,
            })
        }
        fn num_actions(&self) -> usize { 2 }
        fn observation_shape(&self) -> Vec<usize> { vec![1, 1] }
    }

    struct LinearPolicy {
        context: ExecutionContext,
        linear: ContextLinear,
    }

    impl ContextTrainableModel for LinearPolicy {
        fn context_id(&self) -> ContextId { self.context.id() }
        fn parameters(&self) -> Vec<&ContextParameter> { self.linear.parameters() }
    }

    impl ContextRLModel for LinearPolicy {
        fn policy_logits(&mut self, observation: &ContextVariable) -> MlResult<ContextVariable> {
            self.linear.apply(observation)
        }
        fn predict_policy_raw(&mut self, observation: &ContextTensor) -> MlResult<GlobalTensor<f32>> {
            let output = self.linear.predict(observation)?;
            GlobalTensor::from_vec(output.to_vec()?, &output.shape()?)
        }
    }

    fn policy(context: &ExecutionContext) -> MlResult<LinearPolicy> {
        let policy = LinearPolicy {
            context: context.clone(),
            linear: ContextLinear::new(context, 1, 2, "policy")?,
        };
        context.replace_parameter(
            policy.linear.weight().variable(),
            GlobalTensor::from_vec(vec![0.0, 0.0], &[1, 2])?,
        )?;
        context.replace_parameter(
            policy.linear.bias().variable(),
            GlobalTensor::from_vec(vec![-1.0, 1.0], &[2])?,
        )?;
        Ok(policy)
    }

    #[test]
    fn categorical_sampling_preserves_legacy_edge_cases() {
        let selected = (0..200)
            .filter(|index| {
                sample_categorical(&[-10.0, 10.0], (*index as f32 + 0.5) / 200.0) == 1
            })
            .count();
        assert!(selected > 190);
        assert_eq!(sample_categorical(&[5.0], 0.5), 0);
        assert!(sample_categorical(&[f32::NAN, 1.0], 0.5) < 2);
    }

    #[test]
    fn discounted_advantages_match_reinforce_and_center_the_baseline() {
        assert_eq!(discounted_advantages(&[1.0, 3.0], 0.5, false), vec![2.5, 3.0]);
        assert_eq!(discounted_advantages(&[1.0, 3.0], 0.5, true), vec![-0.25, 0.25]);
        assert!(discounted_advantages(&[], 0.9, true).is_empty());
    }

    #[test]
    fn one_step_bandit_runs_rollout_and_policy_update() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut policy = policy(&context)?;
        let original_bias = policy.linear.bias().tensor().to_vec()?;
        let mut environment = OneStepBandit;
        let mut optimizer = ContextSGD::new(&context, 0.1)?;
        optimizer.register_all(&policy.parameters())?;
        let result = ContextRLTrainer::silent(&context).with_seed(7).fit(
            &mut policy,
            &mut environment,
            &mut optimizer,
            EpisodeSchedule::new(3, 1)?,
        )?;
        assert_eq!(result.units_completed, 3);
        assert_eq!(result.metrics["episode_return"], 1.0);
        assert_eq!(result.metrics["episode_steps"], 1.0);
        assert!(result.final_loss.is_finite());
        assert_ne!(policy.linear.bias().tensor().to_vec()?, original_bias);
        assert_eq!(context.graph_stats()?.graph_nodes, 0);
        assert!(policy.parameters().iter().all(|parameter| parameter.grad().unwrap().is_none()));
        Ok(())
    }

    #[test]
    fn trainer_rejects_foreign_optimizer() -> MlResult<()> {
        let context = ExecutionContext::new();
        let foreign = ExecutionContext::new();
        let mut policy = policy(&context)?;
        let mut environment = OneStepBandit;
        let mut optimizer = ContextSGD::new(&foreign, 0.1)?;
        assert!(matches!(
            ContextRLTrainer::silent(&context).fit(
                &mut policy,
                &mut environment,
                &mut optimizer,
                EpisodeSchedule::new(1, 1)?,
            ),
            Err(MlError::ContextError(ContextError::Mismatch))
        ));
        Ok(())
    }
}
