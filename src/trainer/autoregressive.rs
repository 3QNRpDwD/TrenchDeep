//! 자기회귀(Autoregressive) 아키텍처용 트레이너 네임스페이스.
//!
//! 언어모델(LM), 음성 합성, 코드 생성 등 **이전 토큰들로부터 다음 토큰을 예측**
//! 하는 모델을 학습한다. 학습 시에는 teacher forcing 으로 전체 시퀀스를 한 번에
//! 처리하며, 손실은 **토큰 단위 음 로그우도(NLL)** 의 평균이다.
//!
//! ## 지도/비지도 트레이너와의 차이
//!
//! | 항목              | Supervised            | Unsupervised     | **Autoregressive**        |
//! |------------------|-----------------------|------------------|---------------------------|
//! | `forward_loss`   | `(x, t)`              | `(x)`            | `(x)` — t 는 shift-by-one |
//! | 대표 메트릭      | Classification Acc.   | (없음)           | **Perplexity = exp(NLL)** |
//! | 타깃 생성        | 사용자가 직접 제공    | 모델 내부 생성   | 입력을 shift 해서 생성    |
//!
//! AR 트레이너는 입력 인자 측면에서 `UnsupervisedTrainer` 와 동일하지만,
//! **퍼플렉서티 추적** 과 **장기적으로는 토큰-레벨 메트릭 훅** 을 전용으로 장착한다.
//! 현재(Phase 1) 단계에서는 공통 학습 루프를 `UnsupervisedTrainer` 와 나란히 두어
//! 의도적 중복을 유지하고, Phase 2 에서 `TrainerCore::run_epoch` 로 통합한다.

use super::*;

// ────────────────────────────────────────────────────────────────────────────
// AutoregressiveModel — AR 학습 대상 모델의 인터페이스
// ────────────────────────────────────────────────────────────────────────────

/// 자기회귀 트레이너가 학습할 수 있는 모델이 구현해야 하는 인터페이스.
///
/// `forward_loss` 는 시퀀스 입력 `x` **하나만** 받는다. 모델 내부에서
/// `x[:, :-1]` 을 입력으로, `x[:, 1:]` 을 타깃으로 사용해 teacher forcing
/// 손실을 계산한다(구체적인 shift 규약은 모델의 자유).
///
/// # 구현 예시 (토이 Bigram LM)
/// ```ignore
/// impl AutoregressiveModel for BigramLM {
///     fn forward_loss(&mut self, x: &Variable) -> MlResult<(Variable, Variable, usize)> {
///         // 내부에서 shift + SoftmaxCrossEntropy 적용
///         let (logits, loss, n_tokens) = self.teacher_forced_loss(x)?;
///         Ok((logits, loss, n_tokens))
///     }
///     fn params(&self) -> Vec<&dyn Parameter> { vec![&self.embed] }
///     fn predict_raw(&mut self, x: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
///         self.forward_logits_no_grad(x)
///     }
/// }
/// ```
#[cfg(feature = "enableBackward")]
pub trait AutoregressiveModel: TrainableModel {
    /// 순전파와 손실 계산을 수행한다.
    ///
    /// 반환값 `(logits, loss, n_tokens)`:
    /// - `logits`    : 순방향 예측 텐서. 훅이 관찰용으로 받아가지만 루프 자체는 쓰지 않는다.
    /// - `loss`      : 스칼라 평균 NLL Variable (backward 가능).
    /// - `n_tokens`  : 이번 배치에서 **실제로 손실에 기여한 타깃 토큰 수** (padding 제외).
    ///                 퍼플렉서티를 정확히 계산하기 위해 필요.
    fn forward_loss(
        &mut self,
        x: &crate::nn::Variable,
    ) -> MlResult<(crate::nn::Variable, crate::nn::Variable, usize)>;

    /// No-grad 순전파. 초기 손실 표시 및 평가에 사용한다.
    fn predict_raw(
        &mut self,
        x: &dyn TensorBase,
    ) -> MlResult<crate::tensor::GlobalTensor<f32>>;

}

// ────────────────────────────────────────────────────────────────────────────
// AutoregressiveTrainer
// ────────────────────────────────────────────────────────────────────────────

/// 자기회귀 학습 전용 트레이너.
///
/// `UnsupervisedTrainer` 와 동일한 로그·체크포인트·메트릭 훅 구조를 공유하면서,
/// 에폭 요약에 **퍼플렉서티(PPL)** 를 추가로 출력한다.
///
/// # 생성
///
/// ```ignore
/// // `Trainer` 프리셋을 그대로 가져와 변환:
/// let trainer: AutoregressiveTrainer = Trainer::default().into();
///
/// // 또는 전용 프리셋:
/// let trainer = AutoregressiveTrainer::default();
/// ```
pub struct AutoregressiveTrainer {
    pub(crate) core: TrainerCore,
}

impl From<Trainer> for AutoregressiveTrainer {
    fn from(t: Trainer) -> Self {
        let this = Self { core: t.core };
        // default/verbose처럼 패러다임 메트릭이 활성화된 경우에만 PPL을 장착한다.
        if this.core.config().metrics.paradigm {
            this.with_hook(Box::new(Perplexity::new()))
        } else {
            this
        }
    }
}

impl AutoregressiveTrainer {
    /// 지정된 `LogConfig` 로 트레이너를 생성한다.
    pub fn from_config(config: LogConfig) -> Self {
        Self { core: TrainerCore::new(config) }
    }

    /// 기존 `TrainerCore` 를 그대로 주입한다. 훅이 미리 장착된 상태 재사용에 유용.
    pub fn from_core(core: TrainerCore) -> Self {
        Self { core }
    }

    /// 내부 로그 설정 접근자.
    #[inline]
    pub(crate) fn config(&self) -> &LogConfig {
        self.core.config()
    }

    // ── 프리셋 (Trainer 와 동등) ─────────────────────────────────────────

    /// 최대 성능 모드. 로그·NaN 검사 비활성.
    pub fn silent()  -> Self { Trainer::silent().into() }
    /// 핵심 메트릭만 출력.
    pub fn minimal() -> Self { Trainer::minimal().into() }
    /// 기본 모드. 핵심 메트릭과 PPL 포함.
    pub fn default() -> Self { Trainer::default().into() }
    /// 상세 진단 모드. FW/BW, GradNorm, Update Ratio 포함.
    pub fn verbose() -> Self { Trainer::verbose().into() }

    // ── 메트릭 훅 ─────────────────────────────────────────────────────────

    /// 커스텀 메트릭 훅을 장착한다. 체이닝 스타일로 여러 번 호출 가능.
    pub fn with_hook(self, hook: Box<dyn MetricHook>) -> Self {
        self.core.add_hook(hook);
        self
    }

    /// Perplexity 훅을 자동 장착한다. 토큰 수로 가중 평균한 PPL 을 에폭 요약에 포함.
    pub fn with_perplexity(self) -> Self {
        self.with_hook(Box::new(Perplexity::new()))
    }

    // ── 학습 루프 ─────────────────────────────────────────────────────────

    /// 자기회귀 모델을 학습시킨다.
    ///
    /// # 파라미터
    /// - `model`     : `AutoregressiveModel` 구현체
    /// - `optimizer` : 옵티마이저 (`register` 완료 필요)
    /// - `x_set`     : 학습 시퀀스 배치 슬라이스. 각 요소 shape 는 모델이 기대하는 형태 (e.g. `[B, L]`).
    /// - `epochs`    : 최대 에폭 수
    /// - `tolerance` : 연속 두 에폭의 평균 손실 차이가 이 값 미만이면 조기 종료.
    ///                 `0` 이하는 조기 종료 비활성.
    #[cfg(feature = "enableBackward")]
    pub fn fit<M: AutoregressiveModel>(
        &self,
        model:     &mut M,
        optimizer: &mut dyn crate::optimizer::Optimizer,
        dataset:   AutoregressiveDataset<'_>,
        schedule:  EpochSchedule,
    ) -> MlResult<TrainResult> {
        self.fit_inner(model, optimizer, dataset.sequences, schedule.epochs,
            schedule.convergence, 0, f32::INFINITY, None)
    }
    #[cfg(feature = "enableBackward")]
    pub fn fit_checkpointed<M: AutoregressiveModel + CheckpointableModel>(&self, model: &mut M,
        optimizer: &mut dyn crate::optimizer::Optimizer, dataset: AutoregressiveDataset<'_>, schedule: EpochSchedule)
        -> MlResult<TrainResult> {
        self.fit_inner(model, optimizer, dataset.sequences, schedule.epochs, schedule.convergence,
            0, f32::INFINITY, Some(|m, p| m.save_checkpoint(p)))
    }

    /// 체크포인트에서 학습을 재개한다.
    #[cfg(feature = "enableBackward")]
    pub fn resume<M: AutoregressiveModel + CheckpointableModel>(
        &self,
        model:           &mut M,
        optimizer:       &mut dyn crate::optimizer::Optimizer,
        dataset:         AutoregressiveDataset<'_>,
        checkpoint_path: &str,
    ) -> MlResult<TrainResult> {
        use tracing::info;

        let ckpt = CheckpointManager::load_into(checkpoint_path, ParadigmTag::Autoregressive, model, optimizer)?;

        info!(
            "Resuming autoregressive from checkpoint: epoch {}/{}, loss: {:.6}, lr: {:.2e}",
            ckpt.epochs_done, ckpt.total_epochs, ckpt.last_loss, ckpt.optimizer_lr
        );

        self.fit_inner(
            model,
            optimizer,
            dataset.sequences,
            ckpt.total_epochs,
            Convergence::from_tolerance(ckpt.tolerance),
            ckpt.epochs_done,
            ckpt.last_loss,
            Some(|m, p| m.save_checkpoint(p)),
        )
    }

    #[cfg(feature = "enableBackward")]
    fn fit_inner<M: AutoregressiveModel>(
        &self,
        model:       &mut M,
        optimizer:   &mut dyn crate::optimizer::Optimizer,
        x_set:       &[&crate::nn::Variable],
        epochs:      usize,
        convergence: Convergence,
        start_epoch: usize,
        init_loss:   f32,
        save_model:  Option<fn(&M, &std::path::Path) -> MlResult<()>>,
    ) -> MlResult<TrainResult> {
        use std::time::Instant;
        use tracing::info;
        use checkpoint::{interrupt_flag, clear_interrupt};

        let cfg              = self.config();
        let training_start   = Instant::now();
        let remaining_epochs = epochs.saturating_sub(start_epoch);
        self.core.trace_model(
            "autoregressive",
            &*model,
            remaining_epochs,
            x_set.len(),
        );
        let progress         = EpochProgress::new(remaining_epochs, cfg.show_progress);

        let interrupt = if cfg.checkpoint_dir.is_some() {
            let flag = interrupt_flag();
            clear_interrupt(&flag);
            Some(flag)
        } else {
            None
        };

        let mut last_loss   = init_loss;
        let mut epochs_done = start_epoch;
        let mut converged   = false;
        let mut interrupted = false;
        let mut saved_checkpoint = None;
        let mut summary_logs = Vec::new();
        let mut final_metrics = MetricValues::new();

        for epoch in start_epoch..epochs {
            self.core.begin_epoch(epoch);
            let mut xs: Vec<&crate::nn::Variable> = x_set.iter().copied().collect();
            self.core.shuffle(&mut xs);

            let outcome = {
                let mut step = AutoregressiveEpochStep {
                    model: &mut *model,
                    optimizer: &mut *optimizer,
                    xs,
                    last_y: None,
                    last_n_tokens: None,
                };
                self.core.run_epoch(
                    &mut step,
                    epoch - start_epoch,
                    remaining_epochs,
                    &progress,
                    interrupt.as_deref(),
                )?
            };

            epochs_done = epoch + 1;
            let avg_loss          = outcome.avg_loss;
            let batch_interrupted = outcome.interrupted;
            summary_logs.extend(outcome.batch_summaries.iter().cloned());
            final_metrics = outcome.metrics.clone();

            let should_log_epoch =
                cfg.epoch_log_interval != usize::MAX
                && (epoch + 1) % cfg.epoch_log_interval == 0;

            if should_log_epoch {
                let loss_change = last_loss.is_finite().then(|| avg_loss - last_loss);
                let extras_str  = outcome.summary_extras.join(" | ");
                let loss_change = loss_change
                    .map(|value| format!("{value:+.6}"))
                    .unwrap_or_else(|| "N/A".to_string());
                let msg = format!(
                    "AL: {:.6} | LC: {} | {}",
                    avg_loss, loss_change, extras_str
                );
                progress.set_msg(&msg);
                summary_logs.push(format!("Epoch {}/{} | {}", epoch + 1, epochs, msg));
                progress.inc();
            } else {
                progress.inc();
            }

            // ── 인터럽트 체크포인트 저장 ───────────────────────────────────
            if batch_interrupted {
                if let (Some(ckpt_dir), Some(save)) = (&cfg.checkpoint_dir, save_model) {
                    saved_checkpoint = Some(checkpoint::save_interrupt_checkpoint(
                        std::path::Path::new(ckpt_dir),
                        epochs_done,
                        epochs,
                        avg_loss,
                        convergence.tolerance(),
                        optimizer.lr(),
                        ParadigmTag::Autoregressive,
                        cfg.seed,
                        |p| save(&*model, p),
                        &progress,
                    )?);
                } else if cfg.checkpoint_dir.is_some() {
                    return Err(MlError::StringError("checkpointing requires fit_checkpointed()".into()));
                } else {
                    progress.abandon("Interrupted — no checkpoint_dir configured");
                }

                interrupted = true;
                last_loss = avg_loss;
                break;
            }

            // ── 수렴 판정 ────────────────────────────────────────────────
            if convergence.should_stop(last_loss, avg_loss) {
                progress.finish_converged();
                info!("Loss converged at epoch {}. Early stopping.", epoch + 1);
                converged = true;
                last_loss = avg_loss;
                break;
            }
            last_loss = avg_loss;

            if epoch == epochs - 1 {
                progress.finish_completed();
            }
        }

        if let Some(ref flag) = interrupt {
            clear_interrupt(flag);
        }

        let total_duration = training_start.elapsed();
        for summary in &summary_logs {
            info!("{}", summary);
        }
        if !interrupted {
            info!(
                "Autoregressive training finished. Epochs: {}/{}, Final loss: {:.6}, Duration: {:.2?}",
                epochs_done, epochs, last_loss, total_duration
            );
        }

        let reason = if interrupted { StopReason::Interrupted }
            else if converged { StopReason::Converged } else { StopReason::Completed };
        Ok(TrainResult::epochs(reason, epochs_done, last_loss, total_duration)
            .with_metrics(final_metrics)
            .with_checkpoint(saved_checkpoint))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AutoregressiveEpochStep — `run_epoch` 어댑터
// ────────────────────────────────────────────────────────────────────────────
//
// 퍼플렉서티 누적(토큰 수 가중 NLL)이 핵심. 배치/에폭 로그에 `PPL: {..}` 를
// 덧붙여 원본 포맷을 보존한다.

#[cfg(feature = "enableBackward")]
struct AutoregressiveEpochStep<'a, M: AutoregressiveModel> {
    model:         &'a mut M,
    optimizer:     &'a mut dyn crate::optimizer::Optimizer,
    xs:            Vec<&'a crate::nn::Variable>,
    // Perplexity 누적/에폭 요약 PPL 문자열은 Perplexity 훅(From<Trainer> 에서
    // 자동 장착) 이 담당한다. 배치 레벨 live PPL 도 훅으로 일원화된 경로로 넘어가
    // 이 구조체는 순수 루프 상태만 보관한다.
    last_y:        Option<crate::nn::Variable>,
    last_n_tokens: Option<usize>,
}

#[cfg(feature = "enableBackward")]
impl<'a, M: AutoregressiveModel> EpochStep for AutoregressiveEpochStep<'a, M> {
    fn n_batches(&self) -> usize { self.xs.len() }

    fn reset_epoch_state(&mut self) {
        self.last_y = None;
        self.last_n_tokens = None;
    }

    fn forward_backward(
        &mut self,
        batch_idx: usize,
        cfg:       &LogConfig,
    ) -> MlResult<StepOutput> {
        use std::time::Instant;
        use crate::tensor::ComputationGraph;

        ComputationGraph::reset_graph();
        let x = self.xs[batch_idx];

        let fw_start = if cfg.metrics.fw_bw_timing { Some(Instant::now()) } else { None };
        let (y, loss_var, n_tokens) = self.model.forward_loss(x)?;
        let fw_dur = fw_start.map(|s| s.elapsed());

        let loss = loss_var.tensor().data()[0];
        self.last_y = Some(y);
        self.last_n_tokens = Some(n_tokens);

        let bw_start = if cfg.metrics.fw_bw_timing { Some(Instant::now()) } else { None };
        loss_var.backward()?;
        let bw_dur = bw_start.map(|s| s.elapsed());

        let has_nan = (batch_idx + 1) % cfg.nan_check_interval == 0
            && has_invalid_grad(&self.model.params());

        let should_log_batch = cfg.batch_log_interval != usize::MAX
            && (batch_idx + 1) % cfg.batch_log_interval == 0;
        let (gn, ur) = if should_log_batch {
            let params = self.model.params();
            let gn = cfg.metrics.grad_norm.then(|| grad_norm(&params));
            let ur = cfg.metrics.update_ratio.then(|| update_ratio(&params, self.optimizer.lr()));
            (gn, ur)
        } else {
            (None, None)
        };

        let observations = BatchObservations { pred: self.last_y.clone(), target: None,
            n_tokens: Some(n_tokens), lambda: None };
        Ok(StepOutput { loss, loss_weight: n_tokens.max(1), observations, diagnostics: StepDiagnostics {
            has_nan,
            fw_dur,
            bw_dur,
            grad_norm:    gn,
            update_ratio: ur,
            extra_msg:    Vec::new(),
        }})
    }

    fn optimizer_step(&mut self) -> MlResult<()> {
        self.optimizer.step()?;
        self.optimizer.zero_grad()?;
        Ok(())
    }

    fn current_lr(&self) -> f32 { self.optimizer.lr() }

    // 에폭 요약의 "PPL: ..." 는 Perplexity 훅이 자동으로 붙인다.
}

// ────────────────────────────────────────────────────────────────────────────
// Tests — 프리셋의 자동 훅 장착 계약
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_preset_auto_attaches_perplexity_hook() {
        let trainer = AutoregressiveTrainer::default();
        assert_eq!(trainer.core.hook_count(), 1,
            "default() 는 Perplexity 훅을 자동 장착해야 함");
    }

    #[test]
    fn verbose_preset_auto_attaches_perplexity_hook() {
        let trainer = AutoregressiveTrainer::verbose();
        assert_eq!(trainer.core.hook_count(), 1,
            "verbose() 는 Perplexity 훅을 자동 장착해야 함");
    }

    #[test]
    fn minimal_preset_has_no_perplexity_hook() {
        let trainer = AutoregressiveTrainer::minimal();
        assert_eq!(trainer.core.hook_count(), 0,
            "minimal() 은 핵심 메트릭만 표시하므로 PPL 훅이 없어야 함");
    }

    #[test]
    fn silent_preset_has_no_auto_hooks() {
        let trainer = AutoregressiveTrainer::silent();
        assert_eq!(trainer.core.hook_count(), 0,
            "silent() 은 epoch_log_interval=MAX → 훅 없음 (최대 성능 모드)");
    }

    #[test]
    fn from_config_raw_path_has_no_auto_hooks() {
        // epoch_log_interval 이 유효해도 from_config 경로는 자동 장착 안 함.
        let cfg = LogConfig {
            batch_log_interval: 1,
            batch_summary_interval: 100,
            epoch_log_interval: 1,
            nan_check_interval: 1,
            metrics:            Metrics::default(),
            show_progress:      false,
            checkpoint_dir:     None,
            seed:               0,
        };
        let trainer = AutoregressiveTrainer::from_config(cfg);
        assert_eq!(trainer.core.hook_count(), 0);
    }

    #[test]
    fn with_perplexity_shortcut_adds_single_hook_from_silent() {
        let trainer = AutoregressiveTrainer::silent().with_perplexity();
        assert_eq!(trainer.core.hook_count(), 1,
            "silent() + with_perplexity() 는 훅 1개가 되어야 함");
    }
}
