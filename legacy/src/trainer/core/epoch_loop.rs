//! 에폭 공용 루프 (`TrainerCore::run_epoch`).
//!
//! Phase 2 에서 4 개 트레이너(Supervised / Unsupervised / SemiSupervised /
//! Autoregressive) 의 `fit_inner` 에 거의 동일하게 반복되던 배치 루프 —
//! progress bar 생성, NaN 검사, 배치 로그 포매팅, optimizer.step, 인터럽트
//! 감지 — 을 하나로 끌어올린다. RL 은 에피소드 시맨틱이라 범위 외.
//!
//! 패러다임별 차이점은 [`EpochStep`] 트레잇 구현체에 캡슐화된다.
//! 호출부의 `fit_inner` 는 `EpochStep` 구현을 임시로 조립해 `run_epoch` 에
//! 넘기기만 하면 된다.

use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use super::metric_hook::BatchContext;
use super::*;
use crate::trainer::checkpoint::{clear_interrupt, confirm_interrupt, is_interrupted};
use crate::trainer::data::BatchLoader;
use crate::trainer::progress::{BatchProgress, EpochProgress};

// ────────────────────────────────────────────────────────────────────────────
// StepOutput — 한 배치의 forward/backward 결과 요약
// ────────────────────────────────────────────────────────────────────────────

/// 배치 한 번의 실행 결과. `EpochStep::forward_backward` 반환값.
///
/// `optimizer.step()` 호출 **전** 의 관측치만 담는다. optimizer step 이후
/// 갱신된 param/grad 는 다음 배치의 `StepOutput` 에 반영된다.
#[derive(Default)]
pub struct BatchObservations {
    pub pred: Option<crate::nn::Variable>,
    pub target: Option<crate::nn::Variable>,
    pub n_tokens: Option<usize>,
    pub lambda: Option<f32>,
}

pub struct StepDiagnostics {
    /// 이번 배치의 스칼라 손실값 (fan-in 된 평균, 모델 정의에 따름).
    /// 그래디언트에 NaN/Inf 가 포함되어 있는지. `true` 이면 run_epoch 가
    /// 즉시 에폭을 중단하고 에러를 반환한다.
    pub has_nan: bool,
    /// forward 소요 시간. `cfg.metrics.fw_bw_timing = false` 이면 `None`.
    pub fw_dur: Option<Duration>,
    /// backward 소요 시간. 동일.
    pub bw_dur: Option<Duration>,
    /// 전체 파라미터의 grad L2 노름. `cfg.metrics.grad_norm = false` 이면 `None`.
    pub grad_norm: Option<f32>,
    /// Update ratio = ||lr·g|| / ||W||. `cfg.metrics.update_ratio = false` 이면 `None`.
    pub update_ratio: Option<f32>,
    /// 패러다임별 추가 배치 로그 조각. 예: SemiSup 의 `"λ: 0.321"`, AR 의 `"PPL: 12.34"`.
    pub extra_msg: Vec<String>,
}

impl StepDiagnostics {
    /// 배치 로그 메시지를 조립한다. FW/BW 타이밍 → GN → UR → paradigm extras 순.
    pub fn build_batch_msg(&self) -> String {
        let mut parts: Vec<String> = Vec::with_capacity(4 + self.extra_msg.len());
        if let (Some(fw), Some(bw)) = (self.fw_dur, self.bw_dur) {
            parts.push(format!("FW: {:>7.2?} | BW: {:>7.2?}", fw, bw));
        }
        if let Some(gn) = self.grad_norm {
            parts.push(format!("GN: {:.2e}", gn));
        }
        if let Some(ur) = self.update_ratio {
            parts.push(format!("UR: {:.2e}", ur));
        }
        parts.extend(self.extra_msg.iter().cloned());
        parts.join(" | ")
    }
}

pub struct StepOutput {
    pub loss: f32,
    pub loss_weight: usize,
    pub observations: BatchObservations,
    pub diagnostics: StepDiagnostics,
}

// ────────────────────────────────────────────────────────────────────────────
// EpochStep — 패러다임이 구현하는 배치 훅
// ────────────────────────────────────────────────────────────────────────────

/// 한 에폭 동안 `TrainerCore::run_epoch` 가 호출하는 콜백 세트.
///
/// 구현체는 `&mut model`, `&mut optimizer`, 사전 셔플된 데이터 인덱스 등을
/// 내부에 보관한다. 보통 `fit_inner` 안에서 스택 로컬 구조체로 만들어져
/// `run_epoch` 에 `&mut` 로 전달된다.
pub trait EpochStep {
    type Batch;

    /// forward → backward → 선택적 메트릭 계산까지 수행. **optimizer.step 는 호출하지 않는다.**
    ///
    /// 구현체가 해야 할 일:
    /// 1. `ComputationGraph::reset_graph()` 호출
    /// 2. forward_loss
    /// 3. backward (loss_var.backward())
    /// 4. (`cfg.nan_check_interval` 주기에 맞춰) `has_invalid_grad` 실행, 결과를 diagnostics에 담음
    /// 5. (`cfg.metrics.grad_norm/update_ratio` 플래그) 메트릭 계산
    /// 6. 패러다임별 `extra_msg` 조립
    fn forward_backward(
        &mut self,
        batch_idx: usize,
        batch: Self::Batch,
        cfg: &LogConfig,
    ) -> MlResult<StepOutput>;

    /// optimizer.step + zero_grad. run_epoch 가 NaN 이 없을 때만 호출한다.
    fn optimizer_step(&mut self) -> MlResult<()>;

    /// 현재 옵티마이저 학습률. 훅에 전달할 `BatchContext.lr` 채우기에 사용.
    fn current_lr(&self) -> f32;

    /// 에폭 요약 로그에 덧붙일 조각. 예: Supervised 의 `"AC: 87.50%"`, AR 의 `"PPL: 12.34"`.
    fn format_epoch_extras(&self, _avg_loss: f32) -> Vec<String> {
        Vec::new()
    }

    /// 에폭 시작 시 내부 누적 상태를 초기화. accuracy/perplexity 등.
    fn reset_epoch_state(&mut self) {}
}

// ────────────────────────────────────────────────────────────────────────────
// EpochOutcome — run_epoch 반환값
// ────────────────────────────────────────────────────────────────────────────

/// `run_epoch` 한 번의 결과.
pub struct EpochOutcome {
    /// 배치 평균 손실 (total / n_batches_processed). 배치가 0 건이면 0.
    pub avg_loss: f32,
    /// 이번 에폭이 사용자 인터럽트로 도중 중단되었는지 여부.
    pub interrupted: bool,
    /// 에폭 전체 소요 시간.
    pub epoch_dur: Duration,
    /// 에폭 요약 로그 라인의 뒷부분 (패러다임 extras + duration). epoch_log_interval
    /// 판정은 run_epoch 외부의 호출자가 담당한다.
    pub summary_extras: Vec<String>,
    /// Progress bar 종료 뒤 `tracing`으로 발행할 배치 요약.
    pub batch_summaries: Vec<String>,
    /// 마지막 에폭에서 계산된 수치 메트릭.
    pub metrics: crate::trainer::MetricValues,
    /// 실제로 처리한 배치 수. 길이를 미리 알 수 없는 loader에서도 정확하다.
    pub processed_batches: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// TrainerCore::run_epoch 구현
// ────────────────────────────────────────────────────────────────────────────

impl TrainerCore {
    /// 패러다임-불문 에폭 루프.
    ///
    /// # 파라미터
    /// - `step`            : 패러다임별 `EpochStep` 구현
    /// - `epoch_display_idx`: 배치 바 타이틀의 에폭 진행률 계산용 (0-indexed)
    /// - `total_epochs_display`: 에폭 진행률의 전체 에폭 수
    /// - `progress`        : 에폭 바 (배치 바는 여기서 생성)
    /// - `interrupt`       : Ctrl+C 플래그. `None` 이면 인터럽트 감지 비활성.
    ///
    /// # 반환
    /// - `Ok(EpochOutcome)` : 정상 완료 또는 사용자 인터럽트 확인.
    /// - `Err(...)`        : NaN/Inf 발산 감지 시.
    pub(crate) fn run_epoch<S, L>(
        &self,
        step: &mut S,
        loader: &mut L,
        paradigm: &'static str,
        epoch_number: usize,
        total_epochs: usize,
        epoch_display_idx: usize,
        total_epochs_display: usize,
        progress: &EpochProgress,
        interrupt: Option<&AtomicBool>,
    ) -> MlResult<EpochOutcome>
    where
        S: EpochStep,
        L: BatchLoader<Batch = S::Batch>,
    {
        let cfg = self.config();
        let n_batches = loader.batch_count();

        #[cfg(feature = "debugging")]
        let epoch_span = tracing::debug_span!(
            target: "trench_deep::trainer::debug",
            "trainer_epoch",
            epoch = epoch_number,
            total_epochs,
            batches = ?n_batches,
        );
        #[cfg(feature = "debugging")]
        let _epoch_guard = epoch_span.enter();
        #[cfg(feature = "debugging")]
        tracing::debug!(
            target: "trench_deep::trainer::debug",
            "epoch execution started"
        );

        let batch_bar =
            progress.start_batch_bar(epoch_display_idx, total_epochs_display, n_batches);
        let epoch_start = Instant::now();

        step.reset_epoch_state();
        let epoch_context = EpochContext {
            paradigm,
            epoch: epoch_number,
            total_epochs,
            total_batches: n_batches,
        };
        self.notify_epoch_start(&epoch_context);

        // 훅 활성 여부는 에폭 진입 시 한 번만 확인. 빈 목록이면 zero-overhead.
        let hooks_active = !self.hooks.borrow().is_empty();
        if hooks_active {
            for hook in self.hooks.borrow_mut().iter_mut() {
                if let Err(error) = hook.reset() {
                    progress.abandon("Error while resetting training hook");
                    batch_bar.abandon("Hook error");
                    self.notify_train_error(&error.to_string());
                    return Err(error);
                }
            }
        }

        let mut total_loss = 0.0f32;
        let mut total_weight = 0usize;
        let mut interrupted = false;
        let mut batch_summaries = Vec::new();
        let mut grad_norm_sum = 0.0f32;
        let mut grad_norm_count = 0usize;
        let mut update_ratio_sum = 0.0f32;
        let mut update_ratio_count = 0usize;
        let mut fw_secs = 0.0f32;
        let mut fw_count = 0usize;
        let mut bw_secs = 0.0f32;
        let mut bw_count = 0usize;

        let mut batch_idx = 0usize;
        loop {
            let batch = match loader.next_batch() {
                Ok(Some(batch)) => batch,
                Ok(None) => break,
                Err(error) => {
                    progress.abandon("Error while loading batch");
                    batch_bar.abandon("Loader error");
                    self.notify_train_error(&error.to_string());
                    return Err(error);
                }
            };
            let batch_context = BatchStartContext {
                paradigm,
                epoch: epoch_number,
                batch: batch_idx + 1,
                total_epochs,
                total_batches: n_batches,
                episode: None,
            };
            #[cfg(feature = "enableVisualization")]
            let capture = self.begin_graph_capture(&batch_context);
            #[cfg(feature = "enableVisualization")]
            let capture = match capture {
                Ok(capture) => capture,
                Err(error) => {
                    progress.abandon("Error while starting computation graph capture");
                    batch_bar.abandon("Capture error");
                    self.notify_train_error(&error.to_string());
                    return Err(error);
                }
            };
            #[cfg(feature = "debugging")]
            let batch_span = tracing::debug_span!(
                target: "trench_deep::trainer::debug",
                "trainer_batch",
                batch = batch_idx + 1,
                total_batches = ?n_batches,
            );
            #[cfg(feature = "debugging")]
            let _batch_guard = batch_span.enter();
            #[cfg(feature = "debugging")]
            tracing::trace!(
                target: "trench_deep::trainer::debug",
                "batch execution started"
            );

            let info = match step.forward_backward(batch_idx, batch, cfg) {
                Ok(i) => i,
                Err(e) => {
                    progress.abandon("Error during forward/backward");
                    batch_bar.abandon("Error");
                    self.notify_train_error(&e.to_string());
                    return Err(e);
                }
            };

            #[cfg(feature = "debugging")]
            tracing::trace!(
                target: "trench_deep::trainer::debug",
                loss = info.loss,
                loss_weight = info.loss_weight,
                diagnostics = %info.diagnostics.build_batch_msg(),
                "forward/backward completed"
            );

            if info.diagnostics.has_nan {
                progress.abandon("Error: NaN/Inf Gradient");
                batch_bar.abandon("NaN/Inf Gradient");
                tracing::error!(
                    "NaN/Inf gradient at epoch {}, batch {}. total_loss so far: {:.6}",
                    epoch_number,
                    batch_idx + 1,
                    total_loss
                );
                self.notify_train_error("Numerical instability during training");
                return Err(MlError::StringError(
                    "Numerical instability during training".to_string(),
                ));
            }

            let weight = info.loss_weight.max(1);
            total_loss += info.loss * weight as f32;
            total_weight += weight;

            if let Some(value) = info.diagnostics.grad_norm {
                grad_norm_sum += value;
                grad_norm_count += 1;
            }
            if let Some(value) = info.diagnostics.update_ratio {
                update_ratio_sum += value;
                update_ratio_count += 1;
            }
            if let Some(value) = info.diagnostics.fw_dur {
                fw_secs += value.as_secs_f32();
                fw_count += 1;
            }
            if let Some(value) = info.diagnostics.bw_dur {
                bw_secs += value.as_secs_f32();
                bw_count += 1;
            }

            // 훅 업데이트 — forward_backward 가 스태시해둔 last_* 에서 참조를 꺼내
            // BatchContext 를 조립해 전달한다. NaN 검출 이후에 호출하므로 학습을
            // 망가뜨린 배치는 누적되지 않는다.
            if hooks_active {
                let ctx = BatchContext {
                    batch_idx,
                    pred: info
                        .observations
                        .pred
                        .as_ref()
                        .map(|v| v.tensor() as &dyn TensorBase),
                    target: info
                        .observations
                        .target
                        .as_ref()
                        .map(|v| v.tensor() as &dyn TensorBase),
                    loss: info.loss,
                    n_tokens: info.observations.n_tokens,
                    lambda: info.observations.lambda,
                    lr: step.current_lr(),
                };
                for hook in self.hooks.borrow_mut().iter_mut() {
                    if let Err(error) = hook.update(&ctx) {
                        progress.abandon("Error while updating training hook");
                        batch_bar.abandon("Hook error");
                        self.notify_train_error(&error.to_string());
                        return Err(error);
                    }
                }
            }

            let should_log_batch = cfg.batch_log_interval != usize::MAX
                && (batch_idx + 1) % cfg.batch_log_interval == 0;
            if should_log_batch {
                let diagnostics = info.diagnostics.build_batch_msg();
                let message = if diagnostics.is_empty() {
                    format!("L: {:.6}", info.loss)
                } else {
                    format!("L: {:.6} | {}", info.loss, diagnostics)
                };
                batch_bar.set_msg(&message);
                let summary_interval = cfg.batch_summary_interval;
                let should_summarize = summary_interval != usize::MAX
                    && ((batch_idx + 1) % summary_interval == 0
                        || n_batches.is_some_and(|total| batch_idx + 1 == total));
                if should_summarize {
                    let batch_progress = n_batches
                        .map(|total| format!("Batch {:>3}%", (batch_idx + 1) * 100 / total.max(1)))
                        .unwrap_or_else(|| format!("Batch {}", batch_idx + 1));
                    batch_summaries.push(format!("Epoch {}/{} | {} | {}",
                        epoch_number, total_epochs, batch_progress, message));
                }
            }

            batch_bar.inc();
            #[cfg(feature = "enableVisualization")]
            let pending_snapshot = match capture.finish() {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    progress.abandon("Error while capturing computation graph");
                    batch_bar.abandon("Capture error");
                    self.notify_train_error(&error.to_string());
                    return Err(error);
                }
            };
            if let Err(error) = step.optimizer_step() {
                progress.abandon("Error during optimizer step");
                batch_bar.abandon("Optimizer error");
                self.notify_train_error(&error.to_string());
                return Err(error);
            }
            #[cfg(feature = "enableVisualization")]
            pending_snapshot.commit(self);
            let batch_end = BatchEndContext { batch: batch_context, loss: info.loss };
            for observer in self.observers.borrow_mut().iter_mut() {
                observer.on_batch_end(&batch_end);
            }
            batch_idx += 1;

            if let Some(flag) = interrupt {
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
                        interrupted = true;
                        break;
                    }
                }
            }
        }

        if !interrupted {
            batch_bar.finish();
        }

        let avg_loss = if total_weight > 0 {
            total_loss / total_weight as f32
        } else {
            0.0
        };
        let epoch_dur = epoch_start.elapsed();
        self.notify_epoch_end(&epoch_context);

        let mut summary_extras = step.format_epoch_extras(avg_loss);

        // 훅의 포맷 문자열을 에폭 extras 에 덧붙인다. 이렇게 하면 사용자 훅이
        // 에폭 요약 라인에 자동으로 등장한다.
        if hooks_active {
            for hook in self.hooks.borrow().iter() {
                summary_extras.push(hook.format());
            }
        }

        summary_extras.push(format!("{:.2?}", epoch_dur));

        let mut metrics = crate::trainer::MetricValues::new();
        metrics.insert("avg_loss".into(), avg_loss);
        metrics.insert("epoch_duration_secs".into(), epoch_dur.as_secs_f32());
        if grad_norm_count > 0 {
            metrics.insert("grad_norm".into(), grad_norm_sum / grad_norm_count as f32);
        }
        if update_ratio_count > 0 {
            metrics.insert(
                "update_ratio".into(),
                update_ratio_sum / update_ratio_count as f32,
            );
        }
        if fw_count > 0 {
            metrics.insert("forward_secs".into(), fw_secs / fw_count as f32);
        }
        if bw_count > 0 {
            metrics.insert("backward_secs".into(), bw_secs / bw_count as f32);
        }
        if hooks_active {
            for hook in self.hooks.borrow().iter() {
                metrics.insert(hook.name().to_string(), hook.compute());
            }
        }

        #[cfg(feature = "debugging")]
        tracing::debug!(
            target: "trench_deep::trainer::debug",
            avg_loss,
            elapsed = ?epoch_dur,
            interrupted,
            "epoch execution completed"
        );

        Ok(EpochOutcome {
            avg_loss,
            interrupted,
            epoch_dur,
            summary_extras,
            batch_summaries,
            metrics,
            processed_batches: batch_idx,
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 내부 공용 유틸: 배치 바의 abandon 메시지 재사용을 위한 헬퍼
// ────────────────────────────────────────────────────────────────────────────
//
// BatchProgress 는 crate-내부 타입이므로 trait 외부 노출 없이 사용 가능.
// 직접 임포트로만 사용하며 이 모듈에서 별도 재노출은 하지 않는다.
#[allow(unused_imports)]
use BatchProgress as _BatchProgressImportWitness;

#[cfg(test)]
mod tests {
    use super::*;

    struct UnknownLengthLoader { next: usize }
    impl BatchLoader for UnknownLengthLoader {
        type Batch = usize;
        fn begin_epoch(&mut self, _epoch: usize, _runtime: &TrainingRuntime) -> MlResult<()> { self.next = 0; Ok(()) }
        fn next_batch(&mut self) -> MlResult<Option<Self::Batch>> {
            if self.next == 3 { return Ok(None); }
            let value = self.next;
            self.next += 1;
            Ok(Some(value))
        }
        fn batch_count(&self) -> Option<usize> { None }
    }

    struct TestStep;
    impl EpochStep for TestStep {
        type Batch = usize;
        fn forward_backward(&mut self, batch: usize, _value: usize, _cfg: &LogConfig) -> MlResult<StepOutput> {
            Ok(StepOutput {
                loss: batch as f32 + 1.0,
                loss_weight: 1,
                observations: BatchObservations::default(),
                diagnostics: StepDiagnostics {
                    has_nan: false, fw_dur: None, bw_dur: None, grad_norm: None,
                    update_ratio: None, extra_msg: Vec::new(),
                },
            })
        }
        fn optimizer_step(&mut self) -> MlResult<()> { Ok(()) }
        fn current_lr(&self) -> f32 { 0.0 }
    }

    #[test]
    fn unknown_batch_count_uses_processed_count_without_fake_percentage() -> MlResult<()> {
        let config = LogConfig {
            batch_log_interval: 1,
            batch_summary_interval: 2,
            epoch_log_interval: 1,
            nan_check_interval: usize::MAX,
            metrics: Metrics::none(),
            show_progress: false,
            checkpoint_dir: None,
            seed: 0,
        };
        let core = TrainerCore::new(config);
        let mut loader = UnknownLengthLoader { next: 0 };
        let mut step = TestStep;
        let outcome = core.run_epoch(
            &mut step, &mut loader, "test", 1, 1, 0, 1, &EpochProgress::new(1, false), None,
        )?;
        assert_eq!(outcome.processed_batches, 3);
        assert!(outcome.batch_summaries.iter().any(|line| line.contains("Batch 2")));
        assert!(outcome.batch_summaries.iter().all(|line| !line.contains("Batch  %")));
        Ok(())
    }

    #[cfg(feature = "enableVisualization")]
    #[test]
    fn trainer_captures_only_the_requested_batch() -> MlResult<()> {
        use crate::{
            nn::{Parameter, Variable},
            tensor::{AutogradFunction, Tensor, TensorBase, operators::{Add, Function}},
            visualization::{CaptureProfile, GraphSnapshot},
        };
        use std::{cell::RefCell, rc::Rc};

        struct Collector(Rc<RefCell<Vec<GraphSnapshot>>>);
        impl TrainingObserver for Collector {
            fn capture_profile(&self, context: &BatchStartContext) -> Option<CaptureProfile> {
                (context.epoch == 1 && context.batch == 2).then_some(CaptureProfile::Analysis)
            }
            fn on_graph_snapshot(&mut self, snapshot: GraphSnapshot) { self.0.borrow_mut().push(snapshot); }
        }

        struct GraphStep;
        impl EpochStep for GraphStep {
            type Batch = usize;
            fn forward_backward(&mut self, _batch: usize, value: usize, _cfg: &LogConfig) -> MlResult<StepOutput> {
                let x = Variable::new(Tensor::from_vec(vec![value as f32], &[1])?);
                x.retain_grad();
                let y = Variable::new(Tensor::from_vec(vec![1.0], &[1])?);
                let output = Add::new()?.apply(&[&x, &y])?;
                output.backward()?;
                Ok(StepOutput {
                    loss: value as f32,
                    loss_weight: 1,
                    observations: BatchObservations::default(),
                    diagnostics: StepDiagnostics { has_nan: false, fw_dur: None, bw_dur: None, grad_norm: None, update_ratio: None, extra_msg: Vec::new() },
                })
            }
            fn optimizer_step(&mut self) -> MlResult<()> { Ok(()) }
            fn current_lr(&self) -> f32 { 0.0 }
        }

        let config = LogConfig {
            batch_log_interval: usize::MAX, batch_summary_interval: usize::MAX,
            epoch_log_interval: usize::MAX, nan_check_interval: usize::MAX,
            metrics: Metrics::none(), show_progress: false, checkpoint_dir: None, seed: 0,
        };
        let core = TrainerCore::new(config);
        let snapshots = Rc::new(RefCell::new(Vec::new()));
        core.add_observer(Box::new(Collector(snapshots.clone())));
        let mut loader = UnknownLengthLoader { next: 0 };
        let mut step = GraphStep;
        core.run_epoch(&mut step, &mut loader, "test", 1, 1, 0, 1, &EpochProgress::new(1, false), None)?;
        let snapshots = snapshots.borrow();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].context.batch, Some(2));
        Ok(())
    }

    #[cfg(feature = "enableVisualization")]
    #[test]
    fn optimizer_failure_discards_pending_snapshot_and_restores_capture() {
        use crate::{
            nn::{Parameter, Variable},
            tensor::{AutogradFunction, Tensor, operators::{Add, Function}},
            visualization::{CaptureProfile, GraphSnapshot},
        };
        use std::{cell::RefCell, rc::Rc};

        struct Collector(Rc<RefCell<Vec<GraphSnapshot>>>);
        impl TrainingObserver for Collector {
            fn capture_profile(&self, _context: &BatchStartContext) -> Option<CaptureProfile> {
                Some(CaptureProfile::Analysis)
            }
            fn on_graph_snapshot(&mut self, snapshot: GraphSnapshot) {
                self.0.borrow_mut().push(snapshot);
            }
        }

        struct FailingOptimizerStep;
        impl EpochStep for FailingOptimizerStep {
            type Batch = usize;
            fn forward_backward(&mut self, _batch: usize, _value: usize, _cfg: &LogConfig) -> MlResult<StepOutput> {
                let x = Variable::new(Tensor::from_vec(vec![1.0], &[1])?);
                let y = Variable::new(Tensor::from_vec(vec![2.0], &[1])?);
                Add::new()?.apply(&[&x, &y])?.backward()?;
                Ok(StepOutput {
                    loss: 1.0,
                    loss_weight: 1,
                    observations: BatchObservations::default(),
                    diagnostics: StepDiagnostics { has_nan: false, fw_dur: None, bw_dur: None, grad_norm: None, update_ratio: None, extra_msg: Vec::new() },
                })
            }
            fn optimizer_step(&mut self) -> MlResult<()> { Err("optimizer failed".into()) }
            fn current_lr(&self) -> f32 { 0.0 }
        }

        let config = LogConfig {
            batch_log_interval: usize::MAX, batch_summary_interval: usize::MAX,
            epoch_log_interval: usize::MAX, nan_check_interval: usize::MAX,
            metrics: Metrics::none(), show_progress: false, checkpoint_dir: None, seed: 0,
        };
        let core = TrainerCore::new(config);
        let snapshots = Rc::new(RefCell::new(Vec::new()));
        core.add_observer(Box::new(Collector(snapshots.clone())));
        let mut loader = UnknownLengthLoader { next: 0 };
        let result = core.run_epoch(
            &mut FailingOptimizerStep, &mut loader, "test", 1, 1, 0, 1,
            &EpochProgress::new(1, false), None,
        );
        assert!(result.is_err());
        assert!(snapshots.borrow().is_empty());
        assert!(!crate::visualization::recording::is_active());
    }
}
