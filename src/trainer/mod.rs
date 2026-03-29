pub mod config;
pub mod metrics;
pub mod checkpoint;
pub(crate) mod progress;

pub use config::{Metrics, TrainerBuilder};
pub use metrics::{
    grad_norm, weight_norm, update_ratio, has_invalid_grad,
    argmax, ClassificationAccuracy,
};
pub use checkpoint::TrainingCheckpoint;

use config::LogConfig;
use progress::EpochProgress;

// ── trainer 하위 모듈 공통 import ────────────────────────────────────────────
pub(crate) use crate::{MlError, MlResult};
pub(crate) use crate::nn::Parameter;
pub(crate) use crate::tensor::TensorBase;
// ────────────────────────────────────────────────────────────────────────────
// TrainableModel trait
// ────────────────────────────────────────────────────────────────────────────

/// Trainer 가 학습할 수 있는 모델이 구현해야 하는 인터페이스.
///
/// `Model` trait과 달리 학습 루프의  연산만 노출:
/// - `forward_loss`: 순전파 + 손실 계산을 한 번에 수행
/// - `params`: NaN 검사·노름 계산을 위한 파라미터 목록
/// - `predict_raw`: no-grad 순전파 (평가·초기 손실 표시용)
///
/// # 구현 예시
/// ```no_run
///
/// impl TrainableModel for SoftmaxRegression {
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
pub trait TrainableModel {
    /// 순전파와 손실 계산을 수행하여 `(예측값, 손실값)`을 반환한다.
    fn forward_loss(
        &mut self,
        x: &crate::nn::Variable,
        t: &crate::nn::Variable,
    ) -> MlResult<(crate::nn::Variable, crate::nn::Variable)>;

    /// 학습 파라미터 목록을 반환한다.
    /// Trainer 가 NaN 검사, 노름 계산에 사용한다.
    fn params(&self) -> Vec<&dyn Parameter>;

    /// No-grad 순전파. 초기 손실 표시 및 평가에 사용한다.
    fn predict_raw(
        &mut self,
        x: &dyn TensorBase,
    ) -> MlResult<crate::tensor::GlobalTensor<f32>>;

    /// 모델 가중치를 파일로 저장한다.
    ///
    /// 체크포인트 기능을 사용하려면 이 메서드를 구현해야 한다.
    /// 기본 구현은 미구현 에러를 반환한다.
    ///
    /// # 구현 예시 (Sequential 기반 모델)
    /// ```no_run
    /// fn save_model(&self, path: &str) -> MlResult<()> {
    ///     self.net.save(path)
    /// }
    /// ```
    fn save_model(&self, path: &str) -> MlResult<()> {
        let _ = path;
        Err(MlError::StringError(
            "save_model이 구현되지 않았습니다. 체크포인트 기능을 사용하려면 구현하세요.".into()
        ))
    }

    /// 파일에서 모델 가중치를 로드한다.
    ///
    /// 체크포인트에서 학습을 재개하려면 이 메서드를 구현해야 한다.
    /// 기본 구현은 미구현 에러를 반환한다.
    ///
    /// # 구현 예시 (Sequential 기반 모델)
    /// ```no_run
    /// fn load_model(&mut self, path: &str) -> MlResult<()> {
    ///     self.net.load(path)
    /// }
    /// ```
    fn load_model(&mut self, path: &str) -> MlResult<()> {
        let _ = path;
        Err(MlError::StringError(
            "load_model이 구현되지 않았습니다. 체크포인트 기능을 사용하려면 구현하세요.".into()
        ))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// TrainResult
// ────────────────────────────────────────────────────────────────────────────

/// `Trainer::fit()` 완료 후 반환되는 학습 결과.
pub struct TrainResult {
    /// 수렴 조건(tolerance)을 만족하여 조기 종료되었는지 여부.
    pub converged:      bool,
    /// 사용자 인터럽트(Ctrl+C)로 중단되었는지 여부.
    /// `true`이면 체크포인트가 저장되었으며 `resume()`으로 재개 가능.
    pub interrupted:    bool,
    /// 실제로 학습한 에폭 수.
    pub epochs_trained: usize,
    /// 마지막 에폭의 평균 손실.
    pub final_loss:     f32,
    /// 전체 학습 소요 시간.
    pub total_duration: std::time::Duration,
}

// ────────────────────────────────────────────────────────────────────────────
// Trainer
// ────────────────────────────────────────────────────────────────────────────

/// 학습 루프, 로그, 메트릭 계산을 담당하는 범용 트레이너.
///
/// # 프리셋 (권장 진입점)
/// ```no_run
/// trench_deep::trainer::Trainer::silent();   // 최대 성능, 로그 없음
/// trench_deep::trainer::Trainer::minimal();  // 에폭 손실만
/// trench_deep::trainer::Trainer::default();  // 기본 (GradNorm, Accuracy, FW/BW 타이밍)
/// trench_deep::trainer::Trainer::verbose();  // 전체 메트릭
/// ```
///
/// # 커스텀 빌더
/// ```no_run
/// let trainer = trench_deep::trainer::Trainer::builder()
///     .log_every_n_batches(50)
///     .metrics(trench_deep::trainer::Metrics::none().grad_norm().accuracy())
///     .show_progress(true)
///     .build();
/// ```
pub struct Trainer {
    pub(crate) config: LogConfig,
}

impl Trainer {
    /// 커스텀 빌더를 반환.
    pub fn builder() -> TrainerBuilder {
        TrainerBuilder::new()
    }

    // ── 프리셋 ────────────────────────────────────────────────────────────

    /// 최대 성능 모드. 모든 로그·NaN 검사가 비활성화.
    ///
    /// # 주의
    /// NaN 검사가 꺼져 있으므로 발산이 발생해도 감지되지 않음.
    /// 완전히 검증된 모델과 학습률 조합에서만 사용.
    pub fn silent() -> Self {
        Self::builder()
            .log_every_n_batches(0)
            .log_every_n_epochs(0)
            .nan_check(false)
            .metrics(Metrics::none())
            .show_progress(false)
            .build()
    }

    /// 최소 로그 모드. 에폭 평균 손실만 출력한다. NaN 검사는 유지.
    pub fn minimal() -> Self {
        Self::builder()
            .log_every_n_batches(0)
            .nan_check(true)
            .metrics(Metrics::none().accuracy())
            .show_progress(false)
            .build()
    }

    /// 기본 모드. FW/BW 타이밍, GradNorm, Accuracy, progress bar 포함.
    pub fn default() -> Self {
        Self::builder()
            .log_every_n_batches(1)
            .nan_check(true)
            .metrics(Metrics::none().grad_norm().accuracy().fw_bw_timing())
            .show_progress(true)
            .build()
    }

    /// 전체 디버그 모드. 모든 메트릭 활성화.
    pub fn verbose() -> Self {
        Self::builder()
            .log_every_n_batches(1)
            .nan_check(true)
            .metrics(Metrics::all())
            .show_progress(true)
            .build()
    }

    // ── 학습 루프 ─────────────────────────────────────────────────────────

    /// 모델을 학습시킨다.
    ///
    /// # 파라미터
    /// - `model`: `TrainableModel`을 구현한 모델
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
    ///
    /// # 반환
    /// `TrainResult` — 수렴 여부, 인터럽트 여부, 학습한 에폭 수, 최종 손실, 소요 시간.
    #[cfg(feature = "enableBackward")]
    pub fn fit<M: TrainableModel>(
        &self,
        model:     &mut M,
        optimizer: &mut dyn crate::optimizer::Optimizer,
        x_set:     &[&crate::nn::Variable],
        t_set:     &[&crate::nn::Variable],
        epochs:    usize,
        tolerance: f32,
    ) -> MlResult<TrainResult> {
        self.fit_inner(model, optimizer, x_set, t_set, epochs, tolerance, 0, f32::INFINITY)
    }

    /// 체크포인트에서 학습을 재개한다.
    ///
    /// 체크포인트 파일에서 학습 상태를 복원하고, 모델 가중치를 로드한 뒤
    /// 중단된 에폭의 다음 에폭부터 학습을 이어간다.
    ///
    /// # 파라미터
    /// - `model`: 체크포인트 저장 시와 동일한 구조의 모델 (`save_model`/`load_model` 구현 필요)
    /// - `optimizer`: 옵티마이저 (사전에 `register` 완료 필요)
    /// - `x_set`, `t_set`: 학습 데이터
    /// - `checkpoint_path`: 체크포인트 메타데이터 파일 경로 (예: `checkpoints/checkpoint.json`)
    ///
    /// # 예시
    /// ```no_run
    /// // 첫 번째 학습 (중간에 Ctrl+C로 중단됨)
    /// let result = trainer.fit(&mut model, &mut opt, &x, &t, 100, 1e-6)?;
    ///
    /// // 다음 실행에서 재개
    /// if TrainingCheckpoint::exists("checkpoints/checkpoint.json") {
    ///     let result = trainer.resume(
    ///         &mut model, &mut opt, &x, &t,
    ///         "checkpoints/checkpoint.json",
    ///     )?;
    /// }
    /// ```
    #[cfg(feature = "enableBackward")]
    pub fn resume<M: TrainableModel>(
        &self,
        model:           &mut M,
        optimizer:       &mut dyn crate::optimizer::Optimizer,
        x_set:           &[&crate::nn::Variable],
        t_set:           &[&crate::nn::Variable],
        checkpoint_path: &str,
    ) -> MlResult<TrainResult> {
        use tracing::info;

        let ckpt = TrainingCheckpoint::load(checkpoint_path)?;

        // 모델 가중치 복원
        model.load_model(&ckpt.model_path)?;

        // 옵티마이저 학습률 복원
        optimizer.set_lr(ckpt.optimizer_lr);

        info!(
            "Resuming from checkpoint: epoch {}/{}, loss: {:.6}, lr: {:.2e}",
            ckpt.epochs_done, ckpt.total_epochs, ckpt.last_loss, ckpt.optimizer_lr
        );

        self.fit_inner(
            model,
            optimizer,
            x_set,
            t_set,
            ckpt.total_epochs,
            ckpt.tolerance,
            ckpt.epochs_done,
            ckpt.last_loss,
        )
    }

    /// `fit`과 `resume`의 공통 내부 학습 루프.
    #[cfg(feature = "enableBackward")]
    fn fit_inner<M: TrainableModel>(
        &self,
        model:       &mut M,
        optimizer:   &mut dyn crate::optimizer::Optimizer,
        x_set:       &[&crate::nn::Variable],
        t_set:       &[&crate::nn::Variable],
        epochs:      usize,
        tolerance:   f32,
        start_epoch: usize,
        init_loss:   f32,
    ) -> MlResult<TrainResult> {
        use std::time::Instant;
        use rand::{rng, seq::SliceRandom};
        use tracing::{info, warn, error};
        use crate::tensor::ComputationGraph;
        use checkpoint::{interrupt_flag, is_interrupted, clear_interrupt, confirm_interrupt};

        let cfg              = &self.config;
        let n_samples        = x_set.len();
        let training_start   = Instant::now();
        let remaining_epochs = epochs.saturating_sub(start_epoch);
        let progress         = EpochProgress::new(remaining_epochs, cfg.show_progress);

        // 인터럽트 핸들러 설정 (checkpoint_dir이 있을 때만)
        let interrupt = if cfg.checkpoint_dir.is_some() {
            let flag = interrupt_flag();
            clear_interrupt(&flag);   // 이전 실행의 잔여 플래그 초기화
            Some(flag)
        } else {
            None
        };

        let mut last_loss    = init_loss;
        let mut epochs_done  = start_epoch;
        let mut converged    = false;
        let mut interrupted  = false;

        for epoch in start_epoch..epochs {
            let mut total_loss       = 0.0f32;
            let mut total_loss_count = 0usize;
            let mut accuracy         = ClassificationAccuracy::new();
            let epoch_start          = Instant::now();

            let batch_bar = progress.start_batch_bar(
                epoch - start_epoch, remaining_epochs, n_samples,
            );

            // 셔플
            let mut rng_inst = rng();
            let mut pairs: Vec<_> = x_set.iter().zip(t_set.iter()).collect();
            pairs.shuffle(&mut rng_inst);

            let mut batch_interrupted = false;

            for (batch_idx, (x, t)) in pairs.into_iter().enumerate() {
                ComputationGraph::reset_graph();

                // ── 순전파 ──────────────────────────────────────────────
                let fw_start = if cfg.metrics.fw_bw_timing { Some(Instant::now()) } else { None };
                let (y, loss_var) = model.forward_loss(x, t)?;
                let fw_dur = fw_start.map(|s| s.elapsed());

                // 정확도 누적 (항상 수행 — argmax는 O(C)로 비용 낮음)
                if cfg.metrics.accuracy {
                    accuracy.update(y.tensor(), t.tensor());
                }
                total_loss       += loss_var.tensor().data()[0];
                total_loss_count += 1;

                // ── 역전파 ──────────────────────────────────────────────
                let bw_start = if cfg.metrics.fw_bw_timing { Some(Instant::now()) } else { None };
                loss_var.backward()?;
                let bw_dur = bw_start.map(|s| s.elapsed());

                // ── NaN/Inf 검사 ─────────────────────────────────────────
                if (batch_idx + 1) % cfg.nan_check_interval == 0 {
                    let params = model.params();
                    if has_invalid_grad(&params) {
                        progress.abandon("Error: NaN/Inf Gradient");
                        batch_bar.abandon("NaN/Inf Gradient");
                        error!(
                            "NaN/Inf gradient at epoch {}, batch {}. total_loss so far: {:.6}",
                            epoch + 1, batch_idx + 1, total_loss
                        );
                        return Err(MlError::StringError(
                            "Numerical instability during training".to_string()
                        ));
                    }
                }

                // ── 메트릭 계산 + 배치 로그 (interval마다만) ─────────────
                let should_log_batch =
                    cfg.batch_log_interval != usize::MAX
                    && (batch_idx + 1) % cfg.batch_log_interval == 0;

                if should_log_batch {
                    let params    = model.params();
                    let gn        = if cfg.metrics.grad_norm    { grad_norm(&params)                    } else { 0.0 };
                    let ur        = if cfg.metrics.update_ratio { update_ratio(&params, optimizer.lr()) } else { 0.0 };

                    let mut parts = Vec::with_capacity(4);
                    if let (Some(fw), Some(bw)) = (fw_dur, bw_dur) {
                        parts.push(format!("FW: {:>7.2?} | BW: {:>7.2?}", fw, bw));
                    }
                    if cfg.metrics.grad_norm    { parts.push(format!("GN: {:.2e}", gn)); }
                    if cfg.metrics.update_ratio { parts.push(format!("UR: {:.2e}", ur)); }
                    batch_bar.set_msg(&parts.join(" | "));
                }

                batch_bar.inc();

                // ── 파라미터 갱신 ────────────────────────────────────────
                optimizer.step()?;
                optimizer.zero_grad()?;

                // ── 인터럽트 감지 (배치 완료 후) ─────────────────────────
                if let Some(ref flag) = interrupt {
                    if is_interrupted(flag) {
                        let should_stop = progress.suspend(|| {
                            let result = confirm_interrupt();
                            if !result {
                                clear_interrupt(flag);
                            }
                            result
                        });

                        if should_stop {
                            batch_bar.abandon("Interrupted");
                            batch_interrupted = true;
                            break;
                        }
                    }
                }
            }

            if !batch_interrupted {
                batch_bar.finish();
            }

            // ── 에폭 요약 로그 ───────────────────────────────────────────
            epochs_done = epoch + 1;

            let avg_loss       = if total_loss_count > 0 { total_loss / total_loss_count as f32 } else { 0.0 };
            let epoch_accuracy = accuracy.compute();
            let epoch_dur      = epoch_start.elapsed();

            let should_log_epoch =
                cfg.epoch_log_interval != usize::MAX
                && (epoch + 1) % cfg.epoch_log_interval == 0;

            if should_log_epoch {
                let loss_change = avg_loss - last_loss;
                let msg = if cfg.metrics.accuracy {
                    format!(
                        "AL: {:.6} | LC: {:+.6} | AC: {:>6.2}% | {:.2?}",
                        avg_loss, loss_change, epoch_accuracy, epoch_dur
                    )
                } else {
                    format!(
                        "AL: {:.6} | LC: {:+.6} | {:.2?}",
                        avg_loss, loss_change, epoch_dur
                    )
                };
                progress.set_msg(&msg);
                progress.inc();
            } else {
                progress.inc();
            }

            // ── 인터럽트 시 체크포인트 저장 후 종료 ────────────────────────
            if batch_interrupted {
                if let Some(ref ckpt_dir) = cfg.checkpoint_dir {
                    let current_loss = avg_loss;
                    // last_loss가 INFINITY이면 현재 loss 사용
                    let loss_for_ckpt = if last_loss.is_infinite() { current_loss } else { current_loss };

                    let model_path   = format!("{}/model_weights.tdw", ckpt_dir);
                    let ckpt_path    = format!("{}/checkpoint.json", ckpt_dir);

                    match model.save_model(&model_path) {
                        Ok(()) => {
                            let ckpt = TrainingCheckpoint {
                                epochs_done,
                                total_epochs: epochs,
                                last_loss: loss_for_ckpt,
                                tolerance,
                                optimizer_lr: optimizer.lr(),
                                model_path: model_path.clone(),
                                timestamp: format!("{:?}", std::time::SystemTime::now()),
                            };
                            match ckpt.save(&ckpt_path) {
                                Ok(()) => {
                                    progress.finish_interrupted();
                                    info!(
                                        "Checkpoint saved: epoch {}/{}, loss: {:.6}\n  Model:      {}\n  Checkpoint: {}",
                                        epochs_done, epochs, loss_for_ckpt, model_path, ckpt_path
                                    );
                                    eprintln!(
                                        "\n✓ 체크포인트 저장 완료 (epoch {}/{})\n  모델:       {}\n  체크포인트: {}\n  resume()으로 학습을 재개할 수 있습니다.",
                                        epochs_done, epochs, model_path, ckpt_path
                                    );
                                }
                                Err(e) => {
                                    progress.abandon("Checkpoint save failed");
                                    error!("체크포인트 메타데이터 저장 실패: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            progress.abandon("Model save failed");
                            warn!(
                                "모델 저장 실패: {}. save_model()이 구현되어 있는지 확인하세요.",
                                e
                            );
                        }
                    }
                } else {
                    progress.abandon("Interrupted — no checkpoint_dir configured");
                }

                interrupted = true;
                last_loss = avg_loss;
                break;
            }

            // ── 수렴 판정 ────────────────────────────────────────────────
            if (last_loss - avg_loss).abs() < tolerance {
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

        // 인터럽트 플래그 정리
        if let Some(ref flag) = interrupt {
            clear_interrupt(flag);
        }

        let total_duration = training_start.elapsed();
        if !interrupted {
            info!(
                "Training finished. Epochs: {}/{}, Final loss: {:.6}, Duration: {:.2?}",
                epochs_done, epochs, last_loss, total_duration
            );
        }

        Ok(TrainResult {
            converged,
            interrupted,
            epochs_trained: epochs_done,
            final_loss: last_loss,
            total_duration,
        })
    }
}
