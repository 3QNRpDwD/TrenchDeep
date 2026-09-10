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
    use crate::legacy::{
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
    use crate::legacy::{
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
    assert!(!crate::legacy::visualization::recording::is_active());
}
