#![cfg(all(
    feature = "builtinStorage",
    feature = "builtinKernels",
    feature = "enableBackward"
))]

use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::Mutex,
};
use trench_deep::{
    optimizer::{Optimizer, SGD},
    runtime::prepared::*,
    trainer::*,
    *,
};

// The interrupt flag is process-wide; serialize this binary's training tests.
static TRAINING: Mutex<()> = Mutex::new(());

struct Scheduled {
    ctx: ExecutionContext,
    weight: Parameter,
    captures: Cell<usize>,
    variants: bool,
    fail_input: bool,
    steps: Vec<(usize, usize, f32)>,
}
impl TrainableModel for Scheduled {
    fn context_id(&self) -> ContextId {
        self.ctx.id()
    }
    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight]
    }
}
struct ScheduledStrategy;
impl TrainingStrategy<Scheduled> for ScheduledStrategy {
    type Batch = SemiSupervisedBatch;
    const PARADIGM: ParadigmTag = ParadigmTag::SemiSupervised;
    fn forward_batch<L: trench_deep::loss::Loss + ?Sized>(
        &self,
        loss: &L,
        model: &mut Scheduled,
        batch: &Self::Batch,
        step: &TrainingStepContext,
    ) -> MlResult<TrainingOutput> {
        let batch = self.execution_batch(model, batch, step)?;
        let output = self.forward_inputs(loss, model, &batch.inputs)?;
        Ok(TrainingOutput {
            loss: output.loss,
            prediction: output.prediction,
            target: batch.target,
            weight: batch.weight,
            tokens: batch.tokens,
            lambda: batch.lambda,
        })
    }
}
impl PreparedTrainingStrategy<Scheduled> for ScheduledStrategy {
    fn execution_batch(
        &self,
        model: &mut Scheduled,
        batch: &Self::Batch,
        step: &TrainingStepContext,
    ) -> MlResult<PreparedBatch> {
        if model.fail_input {
            let _partial_graph = model.weight.square()?;
            return Err(MlError::StringError("input preparation failed".into()));
        }
        let lambda = step.lambda.expect("semi-supervised lambda");
        model.steps.push((step.epoch, step.batch, lambda));
        let variant = if model.variants && step.epoch % 2 == 1 {
            "alternate"
        } else {
            "base"
        };
        let inputs = ExecutionInputs::new(variant)
            .with("x", batch.labeled_inputs.tensor().clone())?
            .with("target", batch.labeled_targets.clone())?
            .with("lambda", model.ctx.tensor(vec![lambda], &[])?)?;
        let mut result = PreparedBatch::new(inputs, batch.labeled_inputs.tensor().shape()?[0]);
        result.target = Some(batch.labeled_targets.clone());
        result.lambda = Some(lambda);
        Ok(result)
    }
    fn forward_inputs<L: trench_deep::loss::Loss + ?Sized>(
        &self,
        loss: &L,
        model: &Scheduled,
        inputs: &ExecutionInputs,
    ) -> MlResult<ModelOutput> {
        model.captures.set(model.captures.get() + 1);
        let prediction = inputs.get("x")?.as_variable()?.mul(model.weight.tensor())?;
        let loss = loss.compute(&model.ctx, &prediction, inputs.get("target")?)?;
        let loss = loss.mul(inputs.get("lambda")?)?;
        Ok(ModelOutput::new(loss, Some(prediction)))
    }
}
impl CheckpointableModel for Scheduled {
    fn save_checkpoint(&self, path: &std::path::Path) -> MlResult<()> {
        // Saving happens after graph/gradient cleanup.
        assert_eq!(self.ctx.graph_stats()?.graph_nodes, 0);
        assert!(self.weight.grad()?.is_none());
        std::fs::write(path, format!("{:?}", self.weight.tensor().to_vec()?))
            .map_err(|e| MlError::StringError(e.to_string()))
    }
}
fn model(ctx: &ExecutionContext) -> MlResult<Scheduled> {
    Ok(Scheduled {
        ctx: ctx.clone(),
        weight: ctx.parameter(vec![2.0], &[])?,
        captures: Cell::new(0),
        variants: false,
        fail_input: false,
        steps: Vec::new(),
    })
}
struct Data {
    batch: SemiSupervisedBatch,
    emitted: bool,
}
impl BatchLoader for Data {
    type Batch = SemiSupervisedBatch;
    fn begin_epoch(&mut self, _: usize, _: &TrainingRuntime) -> MlResult<()> {
        self.emitted = false;
        Ok(())
    }
    fn next_batch(&mut self) -> MlResult<Option<Self::Batch>> {
        if self.emitted {
            return Ok(None);
        }
        self.emitted = true;
        Ok(Some(SemiSupervisedBatch {
            labeled_inputs: self.batch.labeled_inputs.clone(),
            labeled_targets: self.batch.labeled_targets.clone(),
            unlabeled_inputs: self.batch.unlabeled_inputs.clone(),
        }))
    }
    fn batch_count(&self) -> Option<usize> {
        Some(1)
    }
}
fn dataset(ctx: &ExecutionContext) -> MlResult<Data> {
    let x = ctx.input(vec![1.0], &[1, 1])?;
    let target = ctx.tensor(vec![0.0], &[1, 1])?;
    Ok(Data {
        batch: SemiSupervisedBatch {
            labeled_inputs: x.clone(),
            labeled_targets: target,
            unlabeled_inputs: x,
        },
        emitted: false,
    })
}
#[derive(Default, Debug, PartialEq)]
struct Trace {
    events: Vec<String>,
    losses: Vec<f32>,
}
struct Observer {
    ctx: ExecutionContext,
    trace: Rc<RefCell<Trace>>,
    stop: bool,
}
impl TrainingObserver for Observer {
    fn on_train_start(&mut self, c: &TrainStartContext) {
        assert_eq!(c.paradigm, ParadigmTag::SemiSupervised);
        self.trace.borrow_mut().events.push("start".into());
    }
    fn on_epoch_start(&mut self, c: &EpochContext) {
        self.trace
            .borrow_mut()
            .events
            .push(format!("epoch {}", c.epoch));
    }
    fn on_batch_end(&mut self, c: &BatchEndContext) {
        assert_eq!(self.ctx.graph_stats().unwrap().graph_nodes, 0);
        let mut trace = self.trace.borrow_mut();
        trace
            .events
            .push(format!("batch {}:{}", c.batch.epoch, c.batch.batch));
        trace.losses.push(c.loss);
        if self.stop {
            checkpoint::request_interrupt();
        }
    }
    fn on_epoch_end(&mut self, c: &EpochContext) {
        self.trace
            .borrow_mut()
            .events
            .push(format!("end epoch {}", c.epoch));
    }
    fn on_train_end(&mut self, _: &TrainEndContext) {
        self.trace.borrow_mut().events.push("end".into());
    }
}
struct LambdaHook {
    value: f32,
}
impl MetricHook for LambdaHook {
    fn update(&mut self, c: &BatchContext<'_>) -> MlResult<()> {
        self.value = c.lambda.expect("hook lambda");
        Ok(())
    }
    fn compute(&self) -> f32 {
        self.value
    }
    fn reset(&mut self) -> MlResult<()> {
        self.value = 0.0;
        Ok(())
    }
    fn name(&self) -> &str {
        "hook_lambda"
    }
}

#[test]
fn modes_preserve_configuration_hooks_observers_and_scheduled_gradients() -> MlResult<()> {
    let _lock = TRAINING.lock().unwrap();
    let ramp = ConsistencyRamp::Sigmoid {
        max_weight: 1.0,
        ramp_epochs: 2,
    };
    let mut runs = Vec::new();
    for prepared in [false, true] {
        let ctx = ExecutionContext::new();
        let mut model = model(&ctx)?;
        let mut optimizer = SGD::new(&ctx, 0.1)?;
        optimizer.register_all(&model.parameters())?;
        let mut data = dataset(&ctx)?;
        let trace = Rc::new(RefCell::new(Trace::default()));
        let trainer = Trainer::from_strategy(&ctx, ScheduledStrategy)
            .show_progress(false)
            .metrics(Metrics::all())
            .seed(7)
            .with_seed(42)
            .with_ramp(ramp)
            .check_finite_gradients(true)
            .with_max_grad_norm(0.05)?
            .with_hook(Box::new(LambdaHook { value: 0.0 }))
            .with_observer(Box::new(Observer {
                ctx: ctx.clone(),
                trace: trace.clone(),
                stop: false,
            }));
        let schedule = EpochSchedule::new(3)?;
        let result = if prepared {
            trainer.prepared().fit(
                &mut model,
                &trench_deep::loss::MseLoss::new(),
                &mut optimizer,
                &mut data,
                schedule,
            )?
        } else {
            trainer.fit(
                &mut model,
                &trench_deep::loss::MseLoss::new(),
                &mut optimizer,
                &mut data,
                schedule,
            )?
        };
        assert_eq!(
            model.steps,
            (0..3).map(|e| (e, 0, ramp.value(e))).collect::<Vec<_>>()
        );
        assert_eq!(result.metrics["lambda"], 1.0);
        assert_eq!(result.metrics["hook_lambda"], 1.0);
        assert_eq!(model.captures.get(), if prepared { 1 } else { 3 });
        let weight = model.weight.tensor().item()?;
        let mut expected = 2.0f32;
        for epoch in 0..3 {
            // d(lambda * w^2)/dw = 2 * lambda * w, clipped to 0.05.
            expected -= 0.1 * (2.0 * ramp.value(epoch) * expected).min(0.05);
        }
        assert!((weight - expected).abs() < 1e-6);
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
        assert!(model.weight.grad()?.is_none());
        runs.push((
            weight,
            result.final_loss,
            result.metrics["grad_norm"],
            trace,
        ));
    }
    assert_eq!(runs[0].0, runs[1].0);
    assert_eq!(runs[0].1, runs[1].1);
    assert_eq!(runs[0].2, runs[1].2);
    assert_eq!(*runs[0].3.borrow(), *runs[1].3.borrow());
    assert_eq!(runs[0].3.borrow().events.len(), 11);
    Ok(())
}

#[test]
fn prepared_variants_are_cached_within_fit_and_errors_release_scope() -> MlResult<()> {
    let _lock = TRAINING.lock().unwrap();
    let ctx = ExecutionContext::new();
    let mut model = model(&ctx)?;
    model.variants = true;
    let mut optimizer = SGD::new(&ctx, 0.01)?;
    optimizer.register_all(&model.parameters())?;
    let mut data = dataset(&ctx)?;
    // Configuration methods also work after selecting the prepared type.
    let trainer = Trainer::from_strategy(&ctx, ScheduledStrategy)
        .silent()
        .prepared()
        .with_seed(8)
        .with_ramp(ConsistencyRamp::Constant(1.0))
        .check_finite_gradients(true)
        .with_max_grad_norm(1.0)?
        .with_hook(Box::new(LambdaHook { value: 0.0 }));
    let before = ctx.graph_stats()?;
    for expected in [2, 4] {
        trainer.fit(
            &mut model,
            &trench_deep::loss::MseLoss::new(),
            &mut optimizer,
            &mut data,
            EpochSchedule::new(3)?,
        )?;
        assert_eq!(model.captures.get(), expected);
        assert_eq!(ctx.graph_stats()?, before);
    }
    let weights = model.weight.tensor().to_vec()?;
    model.fail_input = true;
    assert!(
        trainer
            .fit(
                &mut model,
                &trench_deep::loss::MseLoss::new(),
                &mut optimizer,
                &mut data,
                EpochSchedule::new(1)?
            )
            .is_err()
    );
    assert_eq!(model.weight.tensor().to_vec()?, weights);
    assert_eq!(ctx.graph_stats()?, before);
    assert!(model.weight.grad()?.is_none());
    model.fail_input = false;
    trainer.fit(
        &mut model,
        &trench_deep::loss::MseLoss::new(),
        &mut optimizer,
        &mut data,
        EpochSchedule::new(1)?,
    )?;

    let other = ExecutionContext::new();
    assert!(matches!(
        Trainer::from_strategy(&other, ScheduledStrategy)
            .silent()
            .prepared()
            .fit(
                &mut model,
                &trench_deep::loss::MseLoss::new(),
                &mut optimizer,
                &mut data,
                EpochSchedule::new(1)?
            ),
        Err(MlError::ContextError(ContextError::Mismatch))
    ));
    let mut empty_optimizer = SGD::new(&ctx, 0.1)?;
    assert!(matches!(
        trainer.fit(
            &mut model,
            &trench_deep::loss::MseLoss::new(),
            &mut empty_optimizer,
            &mut data,
            EpochSchedule::new(1)?
        ),
        Err(MlError::OptimError(OptimError::ParameterSetMismatch { .. }))
    ));
    Ok(())
}

#[test]
fn both_modes_require_checkpointed_fit_and_save_after_cleanup()
-> Result<(), Box<dyn std::error::Error>> {
    let _lock = TRAINING.lock().unwrap();
    struct ResetInterrupt;
    impl Drop for ResetInterrupt {
        fn drop(&mut self) {
            checkpoint::clear_interrupt();
        }
    }
    let _reset = ResetInterrupt;
    for prepared in [false, true] {
        checkpoint::clear_interrupt();
        let ctx = ExecutionContext::new();
        let mut model = model(&ctx)?;
        let mut optimizer = SGD::new(&ctx, 0.1)?;
        optimizer.register_all(&model.parameters())?;
        let mut data = dataset(&ctx)?;
        let dir = std::env::temp_dir().join(format!(
            "trench-unified-{}-{}",
            std::process::id(),
            prepared
        ));
        let trainer = Trainer::from_strategy(&ctx, ScheduledStrategy)
            .show_progress(false)
            .checkpoint_dir(dir.to_str().unwrap())
            .with_seed(123)
            .with_observer(Box::new(Observer {
                ctx: ctx.clone(),
                trace: Rc::default(),
                stop: true,
            }));
        let schedule = EpochSchedule::new(3)?;
        let result = if prepared {
            let trainer = trainer.prepared();
            assert!(matches!(
                trainer.fit(
                    &mut model,
                    &trench_deep::loss::MseLoss::new(),
                    &mut optimizer,
                    &mut data,
                    schedule
                ),
                Err(MlError::UnsupportedCapability { .. })
            ));
            trainer.fit_checkpointed(
                &mut model,
                &trench_deep::loss::MseLoss::new(),
                &mut optimizer,
                &mut data,
                schedule,
            )?
        } else {
            assert!(matches!(
                trainer.fit(
                    &mut model,
                    &trench_deep::loss::MseLoss::new(),
                    &mut optimizer,
                    &mut data,
                    schedule
                ),
                Err(MlError::UnsupportedCapability { .. })
            ));
            trainer.fit_checkpointed(
                &mut model,
                &trench_deep::loss::MseLoss::new(),
                &mut optimizer,
                &mut data,
                schedule,
            )?
        };
        assert_eq!(result.stop_reason, StopReason::Interrupted);
        assert_eq!(model.steps.len(), 1);
        let paths = result.checkpoint.unwrap();
        assert!(paths.model.is_file());
        let metadata = checkpoint::TrainingCheckpoint::load(&paths.metadata)?;
        assert_eq!(metadata.rng_seed, 123);
        assert_eq!(
            metadata.paradigm,
            Some(checkpoint::ParadigmTag::SemiSupervised)
        );
        std::fs::remove_file(paths.model)?;
        std::fs::remove_file(paths.metadata)?;
        std::fs::remove_dir(dir)?;
    }
    Ok(())
}
