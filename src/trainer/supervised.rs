//! 지도학습 아키텍처용 트레이너 네임스페이스.
//!
//! 이 모듈은 4-분법(지도·비지도·반지도·강화학습) 중 **지도학습** 을 담당한다.
//! `SupervisedTrainer` 는 자체 `fit_inner` 루프를 가지며 `(x, t)` 쌍과
//! 정확도 메트릭을 처리한다. `TrainerCore` 인프라(로그·훅·진행률)는
//! 다른 패러다임과 공유한다.
//!
//! # 이름
//!
//! - `SupervisedModel`   — 지도학습 가능한 모델이 구현하는 트레잇.
//! - `SupervisedTrainer` — `(x, t)` 지도학습 루프를 실행하는 트레이너.

use super::*;

// ────────────────────────────────────────────────────────────────────────────
// SupervisedModel — 지도학습 대상 모델의 인터페이스
// ────────────────────────────────────────────────────────────────────────────

/// 지도학습 트레이너가 학습할 수 있는 모델이 구현해야 하는 인터페이스.
///
/// `(x, t)` 쌍을 입력으로 받아 손실을 계산하는 **지도학습** 루프 전용이다.
/// 비지도·반지도·강화학습 모델은 각 아키텍처별 모듈의 대응 트레잇을 사용한다.
///
/// # 구현 예시
/// ```ignore
/// impl SupervisedModel for SoftmaxRegression {
///     fn forward_loss(&mut self, x: &Variable, t: &Variable) -> MlResult<(Variable, Variable)> {
///         let y = self.apply(x)?;
///         let loss = self.loss_fn.apply_with_label(&[&y, t], "loss")?;
///         Ok((y, loss))
///     }
///     fn params(&self) -> Vec<&dyn Parameter> { vec![&self.w1, &self.b1] }
///     fn predict_raw(&mut self, x: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
///         self.predict(x)
///     }
/// }
/// ```
#[cfg(feature = "enableBackward")]
pub trait SupervisedModel: TrainableModel {
    /// 순전파와 손실 계산을 수행하여 `(예측값, 손실값)` 을 반환한다.
    ///
    /// - `x`: 입력 배치
    /// - `t`: 정답(타깃) 배치
    fn forward_loss(
        &mut self,
        x: &crate::nn::Variable,
        t: &crate::nn::Variable,
    ) -> MlResult<(crate::nn::Variable, crate::nn::Variable)>;

    /// No-grad 순전파. 초기 손실 표시 및 평가에 사용한다.
    fn predict_raw(
        &mut self,
        x: &dyn TensorBase,
    ) -> MlResult<crate::tensor::GlobalTensor<f32>>;

}

// ────────────────────────────────────────────────────────────────────────────
// SupervisedTrainer
// ────────────────────────────────────────────────────────────────────────────

/// 지도학습 전용 트레이너.
///
/// `(x, t)` 쌍을 받아 `forward_loss(x, t)` → backprop → optimizer.step 루프를
/// 실행하며, `ClassificationAccuracy` 를 기본 에폭 요약에 포함한다.
/// `TrainerCore` 를 래핑하므로 로그·훅·체크포인트 인프라는 다른 패러다임과 공유한다.
///
/// # 생성
/// ```ignore
/// // 프리셋
/// let trainer = SupervisedTrainer::default();
///
/// // 범용 빌더를 거쳐 변환 (다른 패러다임과 동일 패턴)
/// let trainer: SupervisedTrainer = Trainer::builder()
///     .log_every_n_batches(50)
///     .show_progress(true)
///     .build()
///     .into();
/// ```
pub struct SupervisedTrainer {
    pub(crate) core: TrainerCore,
}

impl From<Trainer> for SupervisedTrainer {
    fn from(t: Trainer) -> Self {
        let this = Self { core: t.core };
        // 기본 패러다임 메트릭 또는 명시적 accuracy가 켜진 경로에는
        // ClassificationAccuracy 훅을 자동 장착해 에폭 요약에 "AC: ..." 를 출력한다. 원시 경로(from_config) 는
        // 명시적으로 `.with_hook(...)` 을 호출해야 한다.
        if this.core.config().metrics.paradigm || this.core.config().metrics.accuracy {
            this.with_hook(Box::new(ClassificationAccuracy::new()))
        } else {
            this
        }
    }
}

impl SupervisedTrainer {
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

    /// 최대 성능 모드. 모든 로그·NaN 검사가 비활성화.
    pub fn silent()  -> Self { Trainer::silent().into() }
    /// 핵심 메트릭 모드. 배치 손실과 에폭 평균 손실을 표시.
    pub fn minimal() -> Self { Trainer::minimal().into() }
    /// 기본 모드. 핵심 메트릭과 Accuracy 포함.
    pub fn default() -> Self { Trainer::default().into() }
    /// 상세 진단 모드. FW/BW, GradNorm, Update Ratio 포함.
    pub fn verbose() -> Self { Trainer::verbose().into() }

    /// 커스텀 빌더(= `Trainer::builder()`). `.build().into()` 로 변환한다.
    pub fn builder() -> TrainerBuilder { Trainer::builder() }

    // ── 메트릭 훅 ─────────────────────────────────────────────────────────

    /// 커스텀 메트릭 훅을 장착한다. 체이닝 스타일로 여러 번 호출 가능.
    pub fn with_hook(self, hook: Box<dyn MetricHook>) -> Self {
        self.core.add_hook(hook);
        self
    }

    // ── 학습 루프 ─────────────────────────────────────────────────────────

    /// 모델을 학습시킨다.
    ///
    /// # 파라미터
    /// - `model`: `SupervisedModel`을 구현한 모델
    /// - `optimizer`: 파라미터 갱신을 담당하는 옵티마이저 (사전에 `register` 완료 필요)
    /// - `x_set`, `t_set`: 학습 입력·정답 슬라이스 (같은 길이여야 함)
    /// - `epochs`: 최대 에폭 수
    /// - `tolerance`: 연속 두 에폭의 평균 손실 차이가 이 값 미만이면 조기 종료
    ///
    /// # 인터럽트 처리
    /// `checkpoint_dir`이 설정된 경우, 학습 중 Ctrl+C를 누르면:
    /// 1. 현재 배치 완료 후 일시 중지
    /// 2. 사용자에게 종료 확인을 묻는다
    /// 3. 확인 시 모델 가중치와 학습 상태를 체크포인트로 저장하고 종료
    /// 4. 거부 시 학습을 계속한다
    #[cfg(feature = "enableBackward")]
    pub fn fit<M: SupervisedModel>(
        &self,
        model:     &mut M,
        optimizer: &mut dyn crate::optimizer::Optimizer,
        dataset:   SupervisedDataset<'_>,
        schedule:  EpochSchedule,
    ) -> MlResult<TrainResult> {
        self.fit_inner(model, optimizer, dataset.inputs, dataset.targets, schedule.epochs,
            schedule.convergence, 0, f32::INFINITY, None)
    }

    #[cfg(feature = "enableBackward")]
    pub fn fit_checkpointed<M: SupervisedModel + CheckpointableModel>(&self, model: &mut M,
        optimizer: &mut dyn crate::optimizer::Optimizer, dataset: SupervisedDataset<'_>, schedule: EpochSchedule)
        -> MlResult<TrainResult> {
        self.fit_inner(model, optimizer, dataset.inputs, dataset.targets, schedule.epochs,
            schedule.convergence, 0, f32::INFINITY, Some(|m, p| m.save_checkpoint(p)))
    }

    /// 체크포인트에서 학습을 재개한다.
    ///
    /// 체크포인트 파일에서 학습 상태를 복원하고, 모델 가중치를 로드한 뒤
    /// 중단된 에폭의 다음 에폭부터 학습을 이어간다.
    #[cfg(feature = "enableBackward")]
    pub fn resume<M: SupervisedModel + CheckpointableModel>(
        &self,
        model:           &mut M,
        optimizer:       &mut dyn crate::optimizer::Optimizer,
        dataset:         SupervisedDataset<'_>,
        checkpoint_path: &str,
    ) -> MlResult<TrainResult> {
        use tracing::info;

        let ckpt = CheckpointManager::load_into(checkpoint_path, ParadigmTag::Supervised, model, optimizer)?;

        info!(
            "Resuming from checkpoint: epoch {}/{}, loss: {:.6}, lr: {:.2e}",
            ckpt.epochs_done, ckpt.total_epochs, ckpt.last_loss, ckpt.optimizer_lr
        );

        self.fit_inner(
            model,
            optimizer,
            dataset.inputs,
            dataset.targets,
            ckpt.total_epochs,
            Convergence::from_tolerance(ckpt.tolerance),
            ckpt.epochs_done,
            ckpt.last_loss,
            Some(|m, p| m.save_checkpoint(p)),
        )
    }

    /// `fit`과 `resume`의 공통 내부 학습 루프.
    #[cfg(feature = "enableBackward")]
    fn fit_inner<M: SupervisedModel>(
        &self,
        model:       &mut M,
        optimizer:   &mut dyn crate::optimizer::Optimizer,
        x_set:       &[&crate::nn::Variable],
        t_set:       &[&crate::nn::Variable],
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
            "supervised",
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

        let mut last_loss    = init_loss;
        let mut epochs_done  = start_epoch;
        let mut converged    = false;
        let mut interrupted  = false;
        let mut saved_checkpoint = None;
        let mut summary_logs = Vec::new();
        let mut final_metrics = MetricValues::new();

        for epoch in start_epoch..epochs {
            self.core.begin_epoch(epoch);
            let mut pairs: Vec<(&crate::nn::Variable, &crate::nn::Variable)> =
                x_set.iter().zip(t_set.iter()).map(|(x, t)| (*x, *t)).collect();
            self.core.shuffle(&mut pairs);

            let outcome = {
                let mut step = SupervisedEpochStep {
                    model: &mut *model,
                    optimizer: &mut *optimizer,
                    pairs,
                    last_y: None,
                    last_t: None,
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

            if batch_interrupted {
                if let (Some(ckpt_dir), Some(save)) = (&cfg.checkpoint_dir, save_model) {
                    saved_checkpoint = Some(checkpoint::save_interrupt_checkpoint(
                        std::path::Path::new(ckpt_dir),
                        epochs_done,
                        epochs,
                        avg_loss,
                        convergence.tolerance(),
                        optimizer.lr(),
                        ParadigmTag::Supervised,
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
                "Training finished. Epochs: {}/{}, Final loss: {:.6}, Duration: {:.2?}",
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
// SupervisedEpochStep — `run_epoch` 어댑터
// ────────────────────────────────────────────────────────────────────────────
//
// `SupervisedTrainer::fit_inner` 가 매 에폭마다 스택에 조립해
// `TrainerCore::run_epoch` 로 넘기는 `EpochStep` 구현. 지도학습 전용 배치
// 시맨틱(`forward_loss(x, t)` → accuracy 누적) 만 담당하고, 루프/NaN/로그/
// 인터럽트 같은 공통 로직은 `run_epoch` 에 위임한다.

#[cfg(feature = "enableBackward")]
struct SupervisedEpochStep<'a, M: SupervisedModel> {
    model:     &'a mut M,
    optimizer: &'a mut dyn crate::optimizer::Optimizer,
    pairs:     Vec<(&'a crate::nn::Variable, &'a crate::nn::Variable)>,
    // MetricHook 가 접근할 수 있도록 이번 배치의 예측/타깃을 stash.
    // ClassificationAccuracy 누적은 훅 경로(`run_epoch` 의 `hook.update`) 에 일임한다.
    last_y:    Option<crate::nn::Variable>,
    last_t:    Option<&'a crate::nn::Variable>,
}

#[cfg(feature = "enableBackward")]
impl<'a, M: SupervisedModel> EpochStep for SupervisedEpochStep<'a, M> {
    fn n_batches(&self) -> usize { self.pairs.len() }

    fn reset_epoch_state(&mut self) {
        self.last_y = None;
        self.last_t = None;
    }

    fn forward_backward(
        &mut self,
        batch_idx: usize,
        cfg:       &LogConfig,
    ) -> MlResult<StepOutput> {
        use std::time::Instant;
        use crate::tensor::ComputationGraph;

        ComputationGraph::reset_graph();
        let (x, t) = self.pairs[batch_idx];

        let fw_start = if cfg.metrics.fw_bw_timing { Some(Instant::now()) } else { None };
        let (y, loss_var) = self.model.forward_loss(x, t)?;
        let fw_dur = fw_start.map(|s| s.elapsed());

        let loss = loss_var.tensor().data()[0];

        self.last_y = Some(y);
        self.last_t = Some(t);

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

        let observations = BatchObservations { pred: self.last_y.clone(), target: Some(t.clone()), n_tokens: None, lambda: None };
        let loss_weight = t.tensor().shape().first().copied().unwrap_or(1).max(1);
        Ok(StepOutput { loss, loss_weight, observations, diagnostics: StepDiagnostics {
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

    // 에폭 요약의 "AC: ..." 는 ClassificationAccuracy 훅이 자동으로 붙인다.
    // 훅이 장착되지 않은 원시 경로에서는 에폭 요약에서 AC 가 빠지며, 이는 문서화된 계약이다.
}

// ────────────────────────────────────────────────────────────────────────────
// Tests — 프리셋의 자동 훅 장착 계약
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_preset_auto_attaches_accuracy_hook() {
        let trainer = SupervisedTrainer::default();
        assert_eq!(trainer.core.hook_count(), 1,
            "default() 는 ClassificationAccuracy 훅을 자동 장착해야 함");
    }

    #[test]
    fn verbose_preset_auto_attaches_accuracy_hook() {
        let trainer = SupervisedTrainer::verbose();
        assert_eq!(trainer.core.hook_count(), 1,
            "verbose() 는 ClassificationAccuracy 훅을 자동 장착해야 함");
    }

    #[test]
    fn minimal_preset_has_no_paradigm_hook() {
        let trainer = SupervisedTrainer::minimal();
        assert_eq!(trainer.core.hook_count(), 0,
            "minimal() 은 핵심 메트릭만 표시하므로 accuracy 훅이 없어야 함");
    }

    #[test]
    fn silent_preset_has_no_auto_hooks() {
        let trainer = SupervisedTrainer::silent();
        assert_eq!(trainer.core.hook_count(), 0,
            "silent() 은 metrics.accuracy=false → 훅 없음");
    }

    #[test]
    fn builder_without_accuracy_metric_skips_auto_hook() {
        let trainer: SupervisedTrainer = Trainer::builder()
            .metrics(Metrics::none().grad_norm())
            .build()
            .into();
        assert_eq!(trainer.core.hook_count(), 0,
            "metrics.accuracy=false 빌더 경로는 자동 훅이 없어야 함");
    }

    #[test]
    fn builder_with_accuracy_metric_auto_attaches_hook() {
        let trainer: SupervisedTrainer = Trainer::builder()
            .metrics(Metrics::none().accuracy())
            .build()
            .into();
        assert_eq!(trainer.core.hook_count(), 1,
            "metrics.accuracy=true 빌더 경로는 자동 훅이 장착되어야 함");
    }

    #[test]
    fn from_config_raw_path_has_no_auto_hooks() {
        // metrics.accuracy=true 여도 from_config 경로는 자동 훅을 달지 않는다 (계약).
        let cfg = LogConfig {
            batch_log_interval: usize::MAX,
            batch_summary_interval: usize::MAX,
            epoch_log_interval: usize::MAX,
            nan_check_interval: usize::MAX,
            metrics:            Metrics::all(),
            show_progress:      false,
            checkpoint_dir:     None,
            seed:               0,
        };
        let trainer = SupervisedTrainer::from_config(cfg);
        assert_eq!(trainer.core.hook_count(), 0,
            "from_config 원시 경로는 훅 자동 장착 대상이 아니다");
    }
}
