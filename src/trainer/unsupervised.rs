//! 비지도학습 아키텍처용 트레이너 네임스페이스.
//!
//! 지도학습 트레이너가 `(x, t)` 쌍을 가정하는 반면, 비지도 트레이너는
//! 입력 `x` 만 받고 모델 내부에서 손실을 자기생성(self-supervised)한다.
//!
//! ## 대상 모델
//!
//! | 패러다임       | forward_loss 입력 | 손실 자기생성 방식               |
//! |----------------|-------------------|----------------------------------|
//! | Autoencoder    | x                 | 재구성 오차 `‖x − x̂‖²`           |
//! | VAE            | x                 | ELBO = recon + KL                |
//! | DDPM/Diffusion | x₀                | `‖ε − ε_θ(x_t, t)‖²` (내부 노이즈) |
//! | Contrastive    | x                 | 증강쌍 간 InfoNCE                |
//!
//! ## 재사용 전략
//!
//! `Trainer` 의 학습 루프와 대부분이 동일하므로 `TrainerCore` 인프라를 공유한다.
//! `UnsupervisedTrainer` 는 자체 `fit_inner` 를 가지며 정확도 메트릭과
//! 타깃 관련 처리를 제거한다. P3 단계에서는 의도적으로 중복을 허용하고,
//! 이후 공통 루프를 `TrainerCore::run_epoch` 로 끌어올릴 여지를 남겨둔다.

use super::*;

// ────────────────────────────────────────────────────────────────────────────
// UnsupervisedModel — 비지도학습 대상 모델의 인터페이스
// ────────────────────────────────────────────────────────────────────────────

/// 비지도학습 트레이너가 학습할 수 있는 모델이 구현해야 하는 인터페이스.
///
/// `forward_loss` 는 입력 `x` **하나만** 받으며, 손실에 필요한 타깃은
/// 모델 내부에서 생성한다 (예: DDPM 은 랜덤 노이즈, AE 는 입력 자체).
///
/// # 구현 예시 (DDPM)
/// ```ignore
/// impl UnsupervisedModel for Diffusion {
///     fn forward_loss(&mut self, x: &Variable) -> MlResult<(Variable, Variable)> {
///         // 내부에서 랜덤 노이즈/타임스텝을 샘플링하고 MSE loss 를 계산
///         self.forward_loss_diffusion(x)
///     }
///     fn params(&self) -> Vec<&dyn Parameter> { self.unet.params() }
///     fn predict_raw(&mut self, x: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
///         self.unet.predict(x)
///     }
/// }
/// ```
#[cfg(feature = "enableBackward")]
pub trait UnsupervisedModel: TrainableModel {
    /// 순전파와 손실 계산을 수행하여 `(예측값, 손실값)` 을 반환한다.
    ///
    /// 지도학습과 달리 타깃 `t` 인자는 없다. 필요한 타깃은 모델 내부에서
    /// 스토캐스틱하게 생성되거나 (`Diffusion::noise`) 입력 자체로부터 유도된다
    /// (`Autoencoder::reconstruct`).
    ///
    /// - `x`     : 학습 입력 배치
    /// - 반환 `y`: 모델 출력. 훅이 관찰할 수 있도록 노출하지만, 루프 자체는 사용하지 않음.
    /// - 반환 `loss`: 스칼라 손실 Variable.
    fn forward_loss(
        &mut self,
        x: &crate::nn::Variable,
    ) -> MlResult<(crate::nn::Variable, crate::nn::Variable)>;

    /// No-grad 순전파. 초기 손실 표시 및 평가에 사용한다.
    fn predict_raw(
        &mut self,
        x: &dyn TensorBase,
    ) -> MlResult<crate::tensor::GlobalTensor<f32>>;

}

// ────────────────────────────────────────────────────────────────────────────
// UnsupervisedTrainer
// ────────────────────────────────────────────────────────────────────────────

/// 비지도학습 전용 트레이너.
///
/// `Trainer` 와 동일한 로그·체크포인트·메트릭 훅 구조를 공유하면서,
/// 학습 루프는 `(x)` 단일 입력에 맞춰 단순화되어 있다.
///
/// # 생성
///
/// ```ignore
/// // `Trainer` 프리셋을 그대로 가져와 변환:
/// let trainer: UnsupervisedTrainer = Trainer::default().into();
///
/// // 또는 빌더 사용:
/// let trainer = Trainer::builder()
///     .log_every_n_batches(50)
///     .show_progress(true)
///     .build()
///     .into();   // Trainer → UnsupervisedTrainer
/// ```
pub struct UnsupervisedTrainer {
    pub(crate) core: TrainerCore,
}

impl From<Trainer> for UnsupervisedTrainer {
    fn from(t: Trainer) -> Self {
        Self { core: t.core }
    }
}

impl UnsupervisedTrainer {
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
    /// 에폭 평균 손실만 출력.
    pub fn minimal() -> Self { Trainer::minimal().into() }
    /// 기본 모드. FW/BW 타이밍, GradNorm, progress bar 포함.
    pub fn default() -> Self { Trainer::default().into() }
    /// 디버그 모드. 모든 메트릭 활성.
    pub fn verbose() -> Self { Trainer::verbose().into() }

    // ── 메트릭 훅 ─────────────────────────────────────────────────────────

    /// 커스텀 메트릭 훅을 장착한다. 체이닝 스타일로 여러 번 호출 가능.
    pub fn with_hook(self, hook: Box<dyn MetricHook>) -> Self {
        self.core.add_hook(hook);
        self
    }

    // ── 학습 루프 ─────────────────────────────────────────────────────────

    /// 비지도 모델을 학습시킨다.
    ///
    /// # 파라미터
    /// - `model`     : `UnsupervisedModel` 구현체
    /// - `optimizer` : 옵티마이저 (`register` 완료 필요)
    /// - `x_set`     : 학습 입력 배치 슬라이스
    /// - `epochs`    : 최대 에폭 수
    /// - `tolerance` : 연속 두 에폭의 평균 손실 차이가 이 값 미만이면 조기 종료.
    ///                 `0` 이하는 조기 종료 비활성.
    #[cfg(feature = "enableBackward")]
    pub fn fit<M: UnsupervisedModel>(
        &self,
        model:     &mut M,
        optimizer: &mut dyn crate::optimizer::Optimizer,
        dataset:   UnsupervisedDataset<'_>,
        schedule:  EpochSchedule,
    ) -> MlResult<TrainResult> {
        self.fit_inner(model, optimizer, dataset.samples, schedule.epochs,
            schedule.convergence, 0, f32::INFINITY, None)
    }
    #[cfg(feature = "enableBackward")]
    pub fn fit_checkpointed<M: UnsupervisedModel + CheckpointableModel>(&self, model: &mut M,
        optimizer: &mut dyn crate::optimizer::Optimizer, dataset: UnsupervisedDataset<'_>, schedule: EpochSchedule)
        -> MlResult<TrainResult> {
        self.fit_inner(model, optimizer, dataset.samples, schedule.epochs, schedule.convergence,
            0, f32::INFINITY, Some(|m, p| m.save_checkpoint(p)))
    }

    /// 체크포인트에서 학습을 재개한다.
    #[cfg(feature = "enableBackward")]
    pub fn resume<M: UnsupervisedModel + CheckpointableModel>(
        &self,
        model:           &mut M,
        optimizer:       &mut dyn crate::optimizer::Optimizer,
        dataset:         UnsupervisedDataset<'_>,
        checkpoint_path: &str,
    ) -> MlResult<TrainResult> {
        use tracing::info;

        let ckpt = CheckpointManager::load_into(checkpoint_path, ParadigmTag::Unsupervised, model, optimizer)?;

        info!(
            "Resuming unsupervised from checkpoint: epoch {}/{}, loss: {:.6}, lr: {:.2e}",
            ckpt.epochs_done, ckpt.total_epochs, ckpt.last_loss, ckpt.optimizer_lr
        );

        self.fit_inner(
            model,
            optimizer,
            dataset.samples,
            ckpt.total_epochs,
            Convergence::from_tolerance(ckpt.tolerance),
            ckpt.epochs_done,
            ckpt.last_loss,
            Some(|m, p| m.save_checkpoint(p)),
        )
    }

    #[cfg(feature = "enableBackward")]
    fn fit_inner<M: UnsupervisedModel>(
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

        for epoch in start_epoch..epochs {
            self.core.begin_epoch(epoch);
            // 셔플
            let mut xs: Vec<&crate::nn::Variable> = x_set.iter().copied().collect();
            self.core.shuffle(&mut xs);

            let outcome = {
                let mut step = UnsupervisedEpochStep {
                    model: &mut *model,
                    optimizer: &mut *optimizer,
                    xs,
                    last_y: None,
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

            let should_log_epoch =
                cfg.epoch_log_interval != usize::MAX
                && (epoch + 1) % cfg.epoch_log_interval == 0;

            if should_log_epoch {
                let loss_change = avg_loss - last_loss;
                let extras_str  = outcome.summary_extras.join(" | ");
                let msg = format!("AL: {:.6} | LC: {:+.6} | {}", avg_loss, loss_change, extras_str);
                progress.set_msg(&msg);
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
                        ParadigmTag::Unsupervised,
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
        if !interrupted {
            info!(
                "Unsupervised training finished. Epochs: {}/{}, Final loss: {:.6}, Duration: {:.2?}",
                epochs_done, epochs, last_loss, total_duration
            );
        }

        let reason = if interrupted { StopReason::Interrupted }
            else if converged { StopReason::Converged } else { StopReason::Completed };
        Ok(TrainResult::epochs(reason, epochs_done, last_loss, total_duration)
            .with_checkpoint(saved_checkpoint))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// UnsupervisedEpochStep — `run_epoch` 어댑터
// ────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "enableBackward")]
struct UnsupervisedEpochStep<'a, M: UnsupervisedModel> {
    model:     &'a mut M,
    optimizer: &'a mut dyn crate::optimizer::Optimizer,
    xs:        Vec<&'a crate::nn::Variable>,
    last_y:    Option<crate::nn::Variable>,
}

#[cfg(feature = "enableBackward")]
impl<'a, M: UnsupervisedModel> EpochStep for UnsupervisedEpochStep<'a, M> {
    fn n_batches(&self) -> usize { self.xs.len() }

    fn reset_epoch_state(&mut self) {
        self.last_y = None;
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
        let (y, loss_var) = self.model.forward_loss(x)?;
        let fw_dur = fw_start.map(|s| s.elapsed());

        let loss = loss_var.tensor().data()[0];
        self.last_y = Some(y);

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

        let observations = BatchObservations { pred: self.last_y.clone(), target: None, n_tokens: None, lambda: None };
        let loss_weight = x.tensor().shape().first().copied().unwrap_or(1).max(1);
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

}
