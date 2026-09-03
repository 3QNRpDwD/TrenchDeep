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

// ────────────────────────────────────────────────────────────────────────────
// Environment — RL 환경 추상화
// ────────────────────────────────────────────────────────────────────────────

/// 환경 스텝의 결과.
pub struct StepResult {
    /// 다음 관측치.
    pub next_observation: crate::tensor::Tensor,
    /// 이 스텝에서 받은 보상.
    pub reward: f32,
    /// 에피소드 종료 여부.
    pub done: bool,
}

/// RL 환경이 구현해야 하는 인터페이스.
///
/// Gym 스타일의 최소 API 를 따른다. Observation 은 `Tensor` 로 고정하여
/// dyn-compatible 을 유지한다.
///
/// # 구현 예시 (multi-armed bandit)
/// ```ignore
/// impl Environment for TwoArmedBandit {
///     fn reset(&mut self) -> MlResult<Tensor> {
///         Ok(Tensor::from_vec(vec![1.0], &[1, 1])?)
///     }
///     fn step(&mut self, action: usize) -> MlResult<StepResult> {
///         let reward = if action == 0 { 0.2 } else { 0.8 };
///         Ok(StepResult {
///             next_observation: Tensor::from_vec(vec![1.0], &[1, 1])?,
///             reward,
///             done: true,
///         })
///     }
///     fn num_actions(&self) -> usize { 2 }
///     fn observation_shape(&self) -> Vec<usize> { vec![1, 1] }
/// }
/// ```
pub trait Environment {
    /// 환경을 초기 상태로 리셋하고 초기 관측치를 반환.
    fn reset(&mut self) -> MlResult<crate::tensor::Tensor>;

    /// 이산 행동 `action` 을 실행하고 `(next_obs, reward, done)` 을 반환.
    fn step(&mut self, action: usize) -> MlResult<StepResult>;

    /// 이산 행동 공간의 크기.
    fn num_actions(&self) -> usize;

    /// 관측치 텐서의 형태 (행동 샘플링 시 shape 참고용).
    fn observation_shape(&self) -> Vec<usize>;
}

// ────────────────────────────────────────────────────────────────────────────
// RLModel — 정책 네트워크 인터페이스
// ────────────────────────────────────────────────────────────────────────────

/// REINFORCE 트레이너가 학습할 정책 모델의 인터페이스.
///
/// 트레이너는 두 경로로 모델을 호출한다:
/// - **롤아웃(rollout)** : `predict_policy_raw` 로 no-grad 순전파 → 행동 샘플링.
/// - **학습(update)**    : `policy_logits` 로 그래프를 만들고 REINFORCE 손실로 역전파.
///
/// # 구현 예시 (선형 정책)
/// ```ignore
/// impl RLModel for LinearPolicy {
///     fn policy_logits(&mut self, obs: &Variable) -> MlResult<Variable> {
///         let pre = Matmul::new()?.apply(&[obs, &self.w])?;
///         Ok(&pre + &self.b)
///     }
///     fn predict_policy_raw(&mut self, obs: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
///         let pre = Matmul::new()?.forward(&[obs, self.w.tensor()])?.remove(0);
///         Ok(Add::new()?.forward(&[&pre, self.b.tensor()])?.remove(0))
///     }
///     fn params(&self) -> Vec<&dyn Parameter> { vec![&self.w, &self.b] }
/// }
/// ```
#[cfg(feature = "enableBackward")]
pub trait RLModel: TrainableModel {
    /// 정책 로짓 `[batch, n_actions]` 을 반환. 학습용 그래프 경로.
    fn policy_logits(
        &mut self,
        obs: &crate::nn::Variable,
    ) -> MlResult<crate::nn::Variable>;

    /// No-grad 정책 순전파. 행동 샘플링 시 softmax 적용 전 로짓을 반환한다.
    fn predict_policy_raw(
        &mut self,
        obs: &dyn TensorBase,
    ) -> MlResult<crate::tensor::GlobalTensor<f32>>;

}

// ────────────────────────────────────────────────────────────────────────────
// RLTrainer
// ────────────────────────────────────────────────────────────────────────────

/// REINFORCE 알고리즘 전용 트레이너.
///
/// # 학습 하이퍼파라미터
///
/// - `gamma`        : 할인율. 기본 `0.99`.
/// - `use_baseline` : 에피소드 내 return 평균을 baseline 으로 차감할지 여부.
///                    기본 `true`. 분산 감소 및 수렴 안정화에 도움.
pub struct RLTrainer {
    pub(crate) core:         TrainerCore,
    pub(crate) gamma:        f32,
    pub(crate) use_baseline: bool,
}

impl From<Trainer> for RLTrainer {
    fn from(t: Trainer) -> Self {
        Self { core: t.core, gamma: 0.99, use_baseline: true }
    }
}

impl RLTrainer {
    /// 지정 `LogConfig` 로 생성.
    pub fn from_config(config: LogConfig) -> Self {
        Self { core: TrainerCore::new(config), gamma: 0.99, use_baseline: true }
    }

    /// 기존 `TrainerCore` 주입.
    pub fn from_core(core: TrainerCore) -> Self {
        Self { core, gamma: 0.99, use_baseline: true }
    }

    /// 할인율을 교체.
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// baseline(return 평균 차감) 을 켜거나 끈다.
    pub fn with_baseline(mut self, use_baseline: bool) -> Self {
        self.use_baseline = use_baseline;
        self
    }

    #[inline]
    pub(crate) fn config(&self) -> &LogConfig {
        self.core.config()
    }

    // ── 프리셋 ────────────────────────────────────────────────────────────
    pub fn silent()  -> Self { Trainer::silent().into() }
    pub fn minimal() -> Self { Trainer::minimal().into() }
    pub fn default() -> Self { Trainer::default().into() }
    pub fn verbose() -> Self { Trainer::verbose().into() }

    // ── 메트릭 훅 ─────────────────────────────────────────────────────────

    /// 커스텀 메트릭 훅을 장착한다. 체이닝 스타일로 여러 번 호출 가능.
    pub fn with_hook(self, hook: Box<dyn MetricHook>) -> Self {
        self.core.add_hook(hook);
        self
    }

    pub fn with_observer(self, observer: Box<dyn TrainingObserver>) -> Self {
        self.core.add_observer(observer);
        self
    }

    // ── 학습 루프 ─────────────────────────────────────────────────────────

    /// REINFORCE 로 정책을 학습시킨다.
    ///
    /// # 파라미터
    /// - `model`                 : `RLModel` 구현체.
    /// - `env`                   : `Environment` 구현체.
    /// - `optimizer`             : 옵티마이저 (`register` 완료 필요).
    /// - `num_episodes`          : 총 에피소드 수 (= 에폭 수).
    /// - `max_steps_per_episode` : 에피소드당 최대 스텝 (무한 루프 방지).
    #[cfg(feature = "enableBackward")]
    pub fn fit<M: RLModel, E: Environment>(
        &self,
        model:                 &mut M,
        env:                   &mut E,
        optimizer:             &mut dyn crate::optimizer::Optimizer,
        schedule:              EpisodeSchedule,
    ) -> MlResult<TrainResult> {
        use std::time::Instant;
        use tracing::{info, error};
        use crate::tensor::{AutogradFunction, ComputationGraph, Tensor};
        use crate::nn::Variable;
        use crate::tensor::operators::{Mul, Function};
        use crate::loss::SoftmaxCrossEntropyLoss;

        let num_episodes = schedule.episodes;
        let max_steps_per_episode = schedule.max_steps_per_episode;

        let cfg            = self.config();
        let n_actions      = env.num_actions();
        let training_start = Instant::now();
        self.core.trace_model(
            "reinforcement",
            &*model,
            num_episodes,
            Some(max_steps_per_episode),
        );
        let progress       = EpochProgress::new(num_episodes, cfg.show_progress);
        self.core.notify_train_start(&TrainStartContext { paradigm: "reinforcement", total_units: num_episodes });

        let mut last_loss         = f32::INFINITY;
        let mut last_episode_ret  = 0.0f32;
        let mut episodes_done     = 0usize;
        let mut summary_logs      = Vec::new();
        let mut final_metrics     = MetricValues::new();

        // SCE 인스턴스는 루프 외부에서 한 번 생성해 재사용.
        let mut sce = SoftmaxCrossEntropyLoss::new()?;

        for episode in 0..num_episodes {
            let batch_context = BatchStartContext {
                paradigm: "reinforcement",
                epoch: episode + 1,
                batch: 1,
                total_epochs: num_episodes,
                total_batches: Some(1),
                episode: Some(episode + 1),
            };
            let epoch_context = EpochContext {
                paradigm: "reinforcement",
                epoch: episode + 1,
                total_epochs: num_episodes,
                total_batches: Some(1),
            };
            self.core.notify_epoch_start(&epoch_context);
            #[cfg(feature = "enableVisualization")]
            let capture = self.core.begin_graph_capture(&batch_context);
            #[cfg(feature = "enableVisualization")]
            let capture = match capture {
                Ok(capture) => capture,
                Err(error) => {
                    progress.abandon("Error while starting computation graph capture");
                    self.core.notify_train_error(&error.to_string());
                    return Err(error);
                }
            };
            #[cfg(feature = "debugging")]
            let episode_span = tracing::debug_span!(
                target: "trench_deep::trainer::debug",
                "trainer_episode",
                episode = episode + 1,
                total_episodes = num_episodes,
                max_steps = max_steps_per_episode,
            );
            #[cfg(feature = "debugging")]
            let _episode_guard = episode_span.enter();
            #[cfg(feature = "debugging")]
            tracing::debug!(
                target: "trench_deep::trainer::debug",
                "episode execution started"
            );

            let episode_start = Instant::now();
            ComputationGraph::reset_graph();

            // ── 1. 롤아웃 (no-grad) ─────────────────────────────────────────
            let mut obs = env.reset()?;
            let mut trajectory: Vec<(Tensor, usize, f32)> = Vec::new();

            for _step in 0..max_steps_per_episode {
                let logits_t = model.predict_policy_raw(&obs)?;
                let action   = sample_categorical(logits_t.data.as_slice(), self.core.random_f32());
                let result   = env.step(action)?;

                trajectory.push((obs.clone(), action, result.reward));
                obs = result.next_observation;

                if result.done { break; }
            }

            // ── 2. Return 계산 G_t = r_t + γ G_{t+1} ────────────────────────
            let t_len = trajectory.len();
            let mut returns = vec![0.0f32; t_len];
            let mut running = 0.0f32;
            for t in (0..t_len).rev() {
                running    = trajectory[t].2 + self.gamma * running;
                returns[t] = running;
            }
            let episode_return: f32 = trajectory.iter().map(|(_, _, r)| *r).sum();

            // baseline: return 평균 차감으로 advantage 계산 (분산 감소)
            let advantages: Vec<f32> = if self.use_baseline && t_len > 1 {
                let mean = returns.iter().sum::<f32>() / t_len as f32;
                returns.iter().map(|g| g - mean).collect()
            } else {
                returns.clone()
            };

            // ── 3. 손실 그래프 구축 + 역전파 ────────────────────────────────
            let mut loss_accum: Option<Variable> = None;

            for (t, (obs_t, action, _r)) in trajectory.iter().enumerate() {
                let a = advantages[t];
                // 모든 sample 의 advantage 가 0 이면 gradient 가 0. 스킵 가능하지만 정확성 유지.
                let obs_var = Variable::new(obs_t.clone());
                let logits  = model.policy_logits(&obs_var)?;

                // target = one_hot(action_t)
                let mut target_data = vec![0.0f32; n_actions];
                target_data[*action] = 1.0;
                // obs shape 의 leading batch dim 을 target 에도 맞춘다 (기본 [1, n_actions]).
                let target = Variable::new(
                    Tensor::from_vec(target_data, &[1, n_actions])?
                );

                // -log π(a|s) = SoftmaxCE(logits, one_hot(a))
                let neg_log_pi = sce.apply_with_label(&[&logits, &target], "rl_step")?;

                // weighted = A_t · (-log π(a|s))
                let adv_var = Variable::new(Tensor::from_vec(vec![a], &[1, 1])?);
                let weighted = Mul::new()?.apply(&[&neg_log_pi, &adv_var])?;

                loss_accum = Some(match loss_accum {
                    None      => weighted,
                    Some(acc) => &acc + &weighted,
                });
            }

            let step_loss = if let Some(loss_var) = loss_accum {
                let scalar = loss_var.tensor().data()[0];
                loss_var.backward()?;

                // NaN/Inf 검사
                if cfg.nan_check_interval != usize::MAX
                    && (episode + 1) % cfg.nan_check_interval == 0
                {
                    let params = model.params();
                    if has_invalid_grad(&params) {
                        progress.abandon("Error: NaN/Inf Gradient");
                        error!(
                            "NaN/Inf gradient at episode {}. step_loss: {:.6}",
                            episode + 1, scalar
                        );
                        self.core.notify_train_error("Numerical instability during RL training");
                        return Err(MlError::StringError(
                            "Numerical instability during RL training".into()
                        ));
                    }
                }

                #[cfg(feature = "enableVisualization")]
                let pending_snapshot = match capture.finish() {
                    Ok(snapshot) => snapshot,
                    Err(error) => {
                        progress.abandon("Error while capturing computation graph");
                        self.core.notify_train_error(&error.to_string());
                        return Err(error);
                    }
                };
                if let Err(error) = optimizer.step() {
                    progress.abandon("Error during optimizer step");
                    self.core.notify_train_error(&error.to_string());
                    return Err(error.into());
                }
                if let Err(error) = optimizer.zero_grad() {
                    progress.abandon("Error while clearing gradients");
                    self.core.notify_train_error(&error.to_string());
                    return Err(error.into());
                }
                #[cfg(feature = "enableVisualization")]
                pending_snapshot.commit(&self.core);
                scalar
            } else {
                0.0
            };

            episodes_done    = episode + 1;
            last_loss        = step_loss;
            last_episode_ret = episode_return;

            let ep_dur = episode_start.elapsed();

            let should_log_epoch = cfg.epoch_log_interval != usize::MAX
                && ((episode + 1) % cfg.epoch_log_interval == 0
                    || episode + 1 == num_episodes);
            if should_log_epoch {
                let mut parts = Vec::with_capacity(4);
                parts.push(format!("L: {:+.4}", step_loss));
                if cfg.metrics.paradigm {
                    parts.push(format!("EP-R: {:+.4}", episode_return));
                    parts.push(format!("T: {}", t_len));
                }
                if cfg.metrics.grad_norm {
                    let params = model.params();
                    parts.push(format!("GN: {:.2e}", grad_norm(&params)));
                }
                parts.push(format!("{:.2?}", ep_dur));
                let msg = parts.join(" | ");
                progress.set_msg(&msg);
                summary_logs.push(format!(
                    "Episode {}/{} | {}",
                    episode + 1,
                    num_episodes,
                    msg,
                ));
            }
            progress.inc();

            final_metrics.insert("episode_return".into(), episode_return);
            final_metrics.insert("episode_steps".into(), t_len as f32);
            final_metrics.insert("episode_duration_secs".into(), ep_dur.as_secs_f32());
            final_metrics.insert("loss".into(), step_loss);
            for observer in self.core.observers.borrow_mut().iter_mut() {
                observer.on_batch_end(&BatchEndContext { batch: batch_context.clone(), loss: step_loss });
            }
            self.core.notify_epoch_end(&epoch_context);

            #[cfg(feature = "debugging")]
            tracing::debug!(
                target: "trench_deep::trainer::debug",
                loss = step_loss,
                episode_return,
                steps = t_len,
                elapsed = ?ep_dur,
                "episode execution completed"
            );

            if episode == num_episodes - 1 {
                progress.finish_completed();
            }
        }

        let total_duration = training_start.elapsed();
        self.core.notify_train_end(&TrainEndContext { paradigm: "reinforcement", units_completed: episodes_done, interrupted: false });
        for summary in &summary_logs {
            info!("{}", summary);
        }
        info!(
            "RL training finished. Episodes: {}, Last return: {:.4}, Final loss: {:.6}, Duration: {:.2?}",
            episodes_done, last_episode_ret, last_loss, total_duration
        );

        Ok(TrainResult::episodes(episodes_done, last_loss, total_duration)
            .with_metrics(final_metrics))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 유틸 — categorical 샘플링
// ────────────────────────────────────────────────────────────────────────────

/// 로짓 벡터에 softmax 를 적용하고 카테고리컬 샘플을 뽑는다.
/// numerically-stable: max subtraction 후 exp.
fn sample_categorical(logits: &[f32], sample: f32) -> usize {
    if logits.is_empty() {
        return 0;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if !sum.is_finite() || sum == 0.0 {
        // 비정상 상황: 균등 분포로 fallback.
        return ((sample * logits.len() as f32) as usize).min(logits.len() - 1);
    }
    let r = sample.clamp(0.0, 1.0 - f32::EPSILON);
    let mut cum = 0.0;
    for (i, &e) in exps.iter().enumerate() {
        cum += e / sum;
        if r < cum { return i; }
    }
    logits.len() - 1
}

// ────────────────────────────────────────────────────────────────────────────
// 테스트
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_categorical_respects_distribution() {
        // logits = [-10, 10] → 거의 확실하게 action=1
        let mut count_one = 0;
        for i in 0..200 {
            if sample_categorical(&[-10.0, 10.0], (i as f32 + 0.5) / 200.0) == 1 {
                count_one += 1;
            }
        }
        assert!(count_one > 190, "action 1 이 거의 항상 선택되어야 함: {}/200", count_one);
    }

    #[test]
    fn sample_categorical_handles_edge_cases() {
        // 동일 로짓 → 어떤 값도 반환 가능하지만 panic 하면 안 됨
        for i in 0..50 {
            let a = sample_categorical(&[0.0, 0.0, 0.0], i as f32 / 50.0);
            assert!(a < 3);
        }
        // 단일 원소
        assert_eq!(sample_categorical(&[5.0], 0.5), 0);
        // 비정상 로짓
        assert!(sample_categorical(&[f32::NAN, 1.0], 0.5) < 2);
    }
}
