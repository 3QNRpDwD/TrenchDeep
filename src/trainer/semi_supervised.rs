//! 반지도학습 아키텍처용 트레이너 네임스페이스.
//!
//! 반지도 학습은 **소량의 labeled 데이터** 와 **대량의 unlabeled 데이터** 를
//! 함께 활용한다. 대표적인 전략:
//!
//! | 방법              | 아이디어                                                       |
//! |-------------------|---------------------------------------------------------------|
//! | Pi-model          | 동일 입력의 두 stochastic forward pass 간 출력이 일치하도록 제약 |
//! | Mean Teacher      | student 모델이 EMA teacher 의 출력에 수렴하도록 일관성 손실      |
//! | FixMatch          | 약증강 pseudo-label 로 강증강 출력에 지도                       |
//! | VAT               | 입력에 adversarial perturbation 을 준 출력과의 일관성            |
//! | Self-training     | confidence 가 높은 unlabeled 예측을 pseudo-label 로 재귀 학습    |
//!
//! P4 에서는 가장 기본적인 패턴만 추상화하고, 모델이 손실 결합 방식을 직접
//! 결정한다. 이렇게 하면 위 변종들을 모두 동일 트레이너로 구동할 수 있다.

use super::*;

// ────────────────────────────────────────────────────────────────────────────
// SemiSupervisedModel — 반지도 학습 대상 모델의 인터페이스
// ────────────────────────────────────────────────────────────────────────────

/// 반지도학습 트레이너가 학습할 수 있는 모델이 구현해야 하는 인터페이스.
///
/// 트레이너는 매 배치마다 labeled 쌍 `(x_l, t_l)` 과 unlabeled 입력 `x_u`,
/// 그리고 현재 에폭에서 결정된 **일관성 가중치 `lambda`** 를 전달한다.
/// 모델은 이를 사용해 자신만의 손실 결합 전략을 구현한다:
///
/// ```text
/// loss_total = loss_supervised(x_l, t_l) + lambda · loss_consistency(x_u)
/// ```
///
/// 이 분담 방식은 Pi-model, Mean Teacher, VAT, FixMatch 등 대부분의
/// consistency-based 반지도 기법을 단일 트레이너 인터페이스로 포용한다.
///
/// # 구현 예시 (Pi-model 스타일)
/// ```ignore
/// impl SemiSupervisedModel for ToyClassifier {
///     fn forward_loss(
///         &mut self,
///         x_l: &Variable, t_l: &Variable,
///         x_u: &Variable, lambda: f32,
///     ) -> MlResult<(Variable, Variable)> {
///         let y_l   = self.apply(x_l)?;
///         let l_sup = self.ce.apply_with_label(&[&y_l, t_l], "sup")?;
///
///         // Pi-model: 동일 입력에 서로 다른 stochastic noise 를 주어 두 번 순전파
///         let x_u1  = add_noise(x_u, 0.1)?;
///         let x_u2  = add_noise(x_u, 0.1)?;
///         let y_u1  = self.apply(&x_u1)?;
///         let y_u2  = self.apply(&x_u2)?;
///         let l_con = self.mse.apply_with_label(&[&y_u1, &y_u2], "con")?;
///
///         // total = sup + lambda · consistency
///         let total = combine(&l_sup, &l_con, lambda)?;
///         Ok((y_l, total))
///     }
///     // ...
/// }
/// ```
#[cfg(feature = "enableBackward")]
pub trait SemiSupervisedModel: TrainableModel {
    /// 지도 + 비지도(일관성) 손실을 합쳐 `(예측값, 총 손실)` 을 반환.
    ///
    /// - `x_labeled`, `t_labeled` : labeled 배치와 그 타깃.
    /// - `x_unlabeled`            : unlabeled 배치. 일관성 손실 계산에 사용.
    /// - `lambda`                 : 일관성 가중치. 트레이너가 에폭마다 결정.
    fn forward_loss(
        &mut self,
        x_labeled:   &crate::nn::Variable,
        t_labeled:   &crate::nn::Variable,
        x_unlabeled: &crate::nn::Variable,
        lambda:      f32,
    ) -> MlResult<(crate::nn::Variable, crate::nn::Variable)>;

    /// No-grad 순전파. 평가·추론용.
    fn predict_raw(
        &mut self,
        x: &dyn TensorBase,
    ) -> MlResult<crate::tensor::GlobalTensor<f32>>;

}

// ────────────────────────────────────────────────────────────────────────────
// ConsistencyRamp — 일관성 가중치 스케줄
// ────────────────────────────────────────────────────────────────────────────

/// 에폭에 따른 일관성 손실 가중치 `lambda` 의 스케줄.
///
/// Pi-model 원 논문(Laine & Aila, 2017) 에서는 학습 초반에는 지도 손실이
/// 충분히 잡혀야 하므로 lambda 를 0 에서 시작해 점진적으로 최대값까지 램프업한다.
#[derive(Debug, Clone, Copy)]
pub enum ConsistencyRamp {
    /// 고정 가중치. 전 에폭 동안 `w` 를 사용.
    Constant(f32),
    /// `0 → max_weight` 로 `ramp_epochs` 동안 시그모이드 램프업.
    /// Pi-model 논문의 기본 스케줄.
    Sigmoid { max_weight: f32, ramp_epochs: usize },
}

impl ConsistencyRamp {
    /// 주어진 에폭에서의 가중치 계산.
    pub fn value(&self, epoch: usize) -> f32 {
        match *self {
            ConsistencyRamp::Constant(w) => w,
            ConsistencyRamp::Sigmoid { max_weight, ramp_epochs } => {
                if ramp_epochs == 0 {
                    return max_weight;
                }
                let e = epoch.min(ramp_epochs) as f32;
                let r = ramp_epochs as f32;
                // Pi-model 의 exp(-5·(1 - t)²) 스케줄
                let phase = 1.0 - e / r;
                max_weight * (-5.0 * phase * phase).exp()
            }
        }
    }
}

impl Default for ConsistencyRamp {
    /// 기본 스케줄: 30 에폭 동안 0 → 1.0 으로 시그모이드 램프업.
    fn default() -> Self {
        ConsistencyRamp::Sigmoid { max_weight: 1.0, ramp_epochs: 30 }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// SemiSupervisedTrainer
// ────────────────────────────────────────────────────────────────────────────

/// 반지도학습 전용 트레이너.
///
/// 매 배치마다 labeled 쌍과 unlabeled 입력을 함께 모델에 전달하고,
/// 모델이 반환한 `(pred, total_loss)` 로 역전파한다.
/// 일관성 가중치 `lambda` 는 `ConsistencyRamp` 로 에폭별 결정된다.
///
/// # 배치 페어링
///
/// labeled 와 unlabeled 의 크기가 다른 경우 **긴 쪽에 맞춰 순환 반복** 한다
/// (labeled 가 짧으면 labeled 를 wraparound 사용). 이는 반지도 학습에서
/// 일반적인 패턴이며, labeled ≪ unlabeled 인 실전 환경을 가정한다.
///
/// # 생성
///
/// ```ignore
/// let trainer: SemiSupervisedTrainer = Trainer::default().into();
/// ```
pub struct SemiSupervisedTrainer {
    pub(crate) core: TrainerCore,
    pub(crate) ramp: ConsistencyRamp,
}

impl From<Trainer> for SemiSupervisedTrainer {
    fn from(t: Trainer) -> Self {
        let this = Self { core: t.core, ramp: ConsistencyRamp::default() };
        if this.core.config().metrics.paradigm || this.core.config().metrics.accuracy {
            this.with_hook(Box::new(ClassificationAccuracy::new()))
        } else {
            this
        }
    }
}

impl SemiSupervisedTrainer {
    /// 지정 `LogConfig` 와 기본 램프 스케줄로 생성.
    pub fn from_config(config: LogConfig) -> Self {
        Self {
            core: TrainerCore::new(config),
            ramp: ConsistencyRamp::default(),
        }
    }

    /// 기존 `TrainerCore` 를 주입.
    pub fn from_core(core: TrainerCore) -> Self {
        Self { core, ramp: ConsistencyRamp::default() }
    }

    /// 일관성 가중치 스케줄을 교체.
    pub fn with_ramp(mut self, ramp: ConsistencyRamp) -> Self {
        self.ramp = ramp;
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

    // ── 학습 루프 ─────────────────────────────────────────────────────────

    /// 반지도 모델을 학습시킨다.
    ///
    /// # 파라미터
    /// - `model`        : `SemiSupervisedModel` 구현체
    /// - `optimizer`    : 옵티마이저 (`register` 완료 필요)
    /// - `x_labeled`    : labeled 입력 배치 슬라이스
    /// - `t_labeled`    : labeled 타깃 배치 슬라이스 (같은 길이)
    /// - `x_unlabeled`  : unlabeled 입력 배치 슬라이스
    /// - `epochs`       : 최대 에폭 수
    /// - `tolerance`    : 조기 종료 임계값 (0 이하 = 비활성)
    #[cfg(feature = "enableBackward")]
    pub fn fit<M: SemiSupervisedModel>(
        &self,
        model:       &mut M,
        optimizer:   &mut dyn crate::optimizer::Optimizer,
        dataset:     SemiSupervisedDataset<'_>,
        schedule:    EpochSchedule,
    ) -> MlResult<TrainResult> {
        self.fit_inner(model, optimizer, dataset.labeled_inputs, dataset.labeled_targets,
            dataset.unlabeled_inputs, schedule.epochs, schedule.convergence, 0, f32::INFINITY, None)
    }
    #[cfg(feature = "enableBackward")]
    pub fn fit_checkpointed<M: SemiSupervisedModel + CheckpointableModel>(&self, model: &mut M,
        optimizer: &mut dyn crate::optimizer::Optimizer, dataset: SemiSupervisedDataset<'_>, schedule: EpochSchedule)
        -> MlResult<TrainResult> {
        self.fit_inner(model, optimizer, dataset.labeled_inputs, dataset.labeled_targets,
            dataset.unlabeled_inputs, schedule.epochs, schedule.convergence, 0, f32::INFINITY,
            Some(|m, p| m.save_checkpoint(p)))
    }

    /// 체크포인트에서 학습을 재개한다.
    #[cfg(feature = "enableBackward")]
    pub fn resume<M: SemiSupervisedModel + CheckpointableModel>(
        &self,
        model:           &mut M,
        optimizer:       &mut dyn crate::optimizer::Optimizer,
        dataset:         SemiSupervisedDataset<'_>,
        checkpoint_path: &str,
    ) -> MlResult<TrainResult> {
        use tracing::info;

        let ckpt = CheckpointManager::load_into(checkpoint_path, ParadigmTag::SemiSupervised, model, optimizer)?;

        info!(
            "Resuming semi-supervised from checkpoint: epoch {}/{}, loss: {:.6}, lr: {:.2e}",
            ckpt.epochs_done, ckpt.total_epochs, ckpt.last_loss, ckpt.optimizer_lr
        );

        self.fit_inner(
            model, optimizer, dataset.labeled_inputs, dataset.labeled_targets, dataset.unlabeled_inputs,
            ckpt.total_epochs,
            Convergence::from_tolerance(ckpt.tolerance),
            ckpt.epochs_done,
            ckpt.last_loss,
            Some(|m, p| m.save_checkpoint(p)),
        )
    }

    #[cfg(feature = "enableBackward")]
    fn fit_inner<M: SemiSupervisedModel>(
        &self,
        model:       &mut M,
        optimizer:   &mut dyn crate::optimizer::Optimizer,
        x_labeled:   &[&crate::nn::Variable],
        t_labeled:   &[&crate::nn::Variable],
        x_unlabeled: &[&crate::nn::Variable],
        epochs:      usize,
        convergence: Convergence,
        start_epoch: usize,
        init_loss:   f32,
        save_model:  Option<fn(&M, &std::path::Path) -> MlResult<()>>,
    ) -> MlResult<TrainResult> {
        use std::time::Instant;
        use tracing::info;
        use checkpoint::{interrupt_flag, clear_interrupt};

        if x_labeled.len() != t_labeled.len() {
            return Err(MlError::StringError(format!(
                "x_labeled ({}) 와 t_labeled ({}) 길이가 일치해야 합니다.",
                x_labeled.len(), t_labeled.len()
            )));
        }
        if x_labeled.is_empty() {
            return Err(MlError::StringError(
                "labeled 데이터가 비어 있습니다.".into()
            ));
        }
        if x_unlabeled.is_empty() {
            return Err(MlError::StringError(
                "unlabeled 데이터가 비어 있습니다. 반지도학습은 unlabeled 배치가 필수입니다.".into()
            ));
        }

        let cfg              = self.config();
        let n_l              = x_labeled.len();
        let n_u              = x_unlabeled.len();
        let training_start   = Instant::now();
        let remaining_epochs = epochs.saturating_sub(start_epoch);
        self.core.trace_model(
            "semi_supervised",
            &*model,
            remaining_epochs,
            n_l.max(n_u),
        );
        let progress         = EpochProgress::new(remaining_epochs, cfg.show_progress);

        // 인터럽트 핸들러 — 다른 트레이너와 동일 규약.
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
            // 라벨·언라벨 각각 셔플 후, 긴 쪽에 맞춰 순환.
            let mut labeled_idx:   Vec<usize> = (0..n_l).collect();
            let mut unlabeled_idx: Vec<usize> = (0..n_u).collect();
            self.core.shuffle(&mut labeled_idx);
            self.core.shuffle(&mut unlabeled_idx);

            let lambda = self.ramp.value(epoch);

            let outcome = {
                let mut step = SemiSupervisedEpochStep {
                    model: &mut *model,
                    optimizer: &mut *optimizer,
                    x_labeled, t_labeled, x_unlabeled,
                    labeled_idx, unlabeled_idx,
                    lambda,
                    show_paradigm: cfg.metrics.paradigm,
                    last_y: None,
                    last_t: None,
                };
                self.core.run_epoch(
                    &mut step,
                    epoch - start_epoch, remaining_epochs,
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
            }
            progress.inc();

            if batch_interrupted {
                if let (Some(ckpt_dir), Some(save)) = (&cfg.checkpoint_dir, save_model) {
                    saved_checkpoint = Some(checkpoint::save_interrupt_checkpoint(
                        std::path::Path::new(ckpt_dir),
                        epochs_done,
                        epochs,
                        avg_loss,
                        convergence.tolerance(),
                        optimizer.lr(),
                        ParadigmTag::SemiSupervised,
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
                "Semi-supervised training finished. Epochs: {}/{}, Final loss: {:.6}, Duration: {:.2?}",
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
// SemiSupervisedEpochStep — `run_epoch` 어댑터
// ────────────────────────────────────────────────────────────────────────────
//
// 에폭 길이는 `max(n_l, n_u)` 이며, 짧은 쪽은 wraparound 로 순환 사용.
// `lambda` 는 에폭 당 고정이므로 step 생성 시점에 주입한다.

#[cfg(feature = "enableBackward")]
struct SemiSupervisedEpochStep<'a, M: SemiSupervisedModel> {
    model:         &'a mut M,
    optimizer:     &'a mut dyn crate::optimizer::Optimizer,
    x_labeled:     &'a [&'a crate::nn::Variable],
    t_labeled:     &'a [&'a crate::nn::Variable],
    x_unlabeled:   &'a [&'a crate::nn::Variable],
    labeled_idx:   Vec<usize>,
    unlabeled_idx: Vec<usize>,
    lambda:        f32,
    show_paradigm: bool,
    last_y:        Option<crate::nn::Variable>,
    last_t:        Option<&'a crate::nn::Variable>,
}

#[cfg(feature = "enableBackward")]
impl<'a, M: SemiSupervisedModel> EpochStep for SemiSupervisedEpochStep<'a, M> {
    fn n_batches(&self) -> usize { self.labeled_idx.len().max(self.unlabeled_idx.len()) }

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

        let n_l = self.labeled_idx.len();
        let n_u = self.unlabeled_idx.len();
        let li  = self.labeled_idx[batch_idx % n_l];
        let ui  = self.unlabeled_idx[batch_idx % n_u];
        let x_l = self.x_labeled[li];
        let t_l = self.t_labeled[li];
        let x_u = self.x_unlabeled[ui];

        let fw_start = if cfg.metrics.fw_bw_timing { Some(Instant::now()) } else { None };
        let (y, loss_var) = self.model.forward_loss(x_l, t_l, x_u, self.lambda)?;
        let fw_dur = fw_start.map(|s| s.elapsed());

        let loss = loss_var.tensor().data()[0];
        self.last_y = Some(y);
        self.last_t = Some(t_l);

        let bw_start = if cfg.metrics.fw_bw_timing { Some(Instant::now()) } else { None };
        loss_var.backward()?;
        let bw_dur = bw_start.map(|s| s.elapsed());

        let has_nan = (batch_idx + 1) % cfg.nan_check_interval == 0
            && has_invalid_grad(&self.model.params());

        let should_log_batch = cfg.batch_log_interval != usize::MAX
            && (batch_idx + 1) % cfg.batch_log_interval == 0;
        let (gn, ur, extra_msg) = if should_log_batch {
            let params = self.model.params();
            let gn = cfg.metrics.grad_norm.then(|| grad_norm(&params));
            let ur = cfg.metrics.update_ratio.then(|| update_ratio(&params, self.optimizer.lr()));
            let extra_msg = cfg.metrics.paradigm
                .then(|| format!("λ: {:.3}", self.lambda))
                .into_iter()
                .collect();
            (gn, ur, extra_msg)
        } else {
            (None, None, Vec::new())
        };

        let observations = BatchObservations { pred: self.last_y.clone(), target: Some(t_l.clone()),
            n_tokens: None, lambda: Some(self.lambda) };
        let loss_weight = t_l.tensor().shape().first().copied().unwrap_or(1).max(1);
        Ok(StepOutput { loss, loss_weight, observations, diagnostics: StepDiagnostics {
            has_nan,
            fw_dur,
            bw_dur,
            grad_norm:    gn,
            update_ratio: ur,
            extra_msg,
        }})
    }

    fn optimizer_step(&mut self) -> MlResult<()> {
        self.optimizer.step()?;
        self.optimizer.zero_grad()?;
        Ok(())
    }

    fn current_lr(&self) -> f32 { self.optimizer.lr() }

    fn format_epoch_extras(&self, _avg_loss: f32) -> Vec<String> {
        self.show_paradigm
            .then(|| format!("λ: {:.3}", self.lambda))
            .into_iter()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consistency_ramp_constant() {
        let r = ConsistencyRamp::Constant(0.5);
        assert_eq!(r.value(0),   0.5);
        assert_eq!(r.value(100), 0.5);
    }

    #[test]
    fn consistency_ramp_sigmoid_monotonic() {
        let r = ConsistencyRamp::Sigmoid { max_weight: 1.0, ramp_epochs: 30 };
        let v0  = r.value(0);
        let v15 = r.value(15);
        let v30 = r.value(30);
        let v40 = r.value(40);
        assert!(v0  < v15, "0 < 15 실패: {} < {}", v0, v15);
        assert!(v15 < v30, "15 < 30 실패: {} < {}", v15, v30);
        assert!((v30 - 1.0).abs() < 1e-5, "램프 끝에서 max_weight 수렴 실패: {}", v30);
        assert!((v40 - 1.0).abs() < 1e-5, "램프 초과 영역에서 max_weight 유지 실패: {}", v40);
    }

    #[test]
    fn consistency_ramp_sigmoid_zero_length() {
        let r = ConsistencyRamp::Sigmoid { max_weight: 0.7, ramp_epochs: 0 };
        assert_eq!(r.value(0), 0.7);
    }
}
