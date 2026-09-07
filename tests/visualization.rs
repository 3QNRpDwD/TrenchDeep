#![cfg(all(
    feature = "enableVisualization",
    feature = "builtinStorage",
    feature = "builtinKernels"
))]
use std::{cell::RefCell, rc::Rc};
use trench_deep::visualization::*;
use trench_deep::*;
use trench_deep::{
    nn::LinearRegression,
    optimizer::{Optimizer, SGD},
    trainer::*,
};

struct CaptureOrder {
    ctx: ExecutionContext,
    parameters: Vec<Parameter>,
    events: Rc<RefCell<Vec<&'static str>>>,
}
impl CaptureOrder {
    fn cleaned(&self) {
        assert_eq!(self.ctx.graph_stats().unwrap().graph_nodes, 0);
        assert!(self.parameters.iter().all(|p| p.grad().unwrap().is_none()));
    }
}
impl TrainingObserver for CaptureOrder {
    fn capture_profile(&self, _: &BatchStartContext) -> Option<CaptureProfile> {
        Some(CaptureProfile::Analysis)
    }
    fn on_graph_snapshot(&mut self, snapshot: GraphSnapshot) {
        self.cleaned();
        assert!(snapshot.nodes.iter().any(|n| n.gradient_stats.is_some()));
        self.events.borrow_mut().push("snapshot");
    }
    fn on_batch_end(&mut self, _: &BatchEndContext) {
        self.cleaned();
        self.events.borrow_mut().push("batch");
    }
}
struct FailHook;
impl MetricHook for FailHook {
    fn update(&mut self, _: &BatchContext<'_>) -> MlResult<()> {
        Err(TensorError::InvalidOperation {
            op: "test_hook",
            reason: "injected failure".into(),
        }
        .into())
    }
    fn compute(&self) -> f32 {
        0.0
    }
    fn reset(&mut self) -> MlResult<()> {
        Ok(())
    }
    fn name(&self) -> &str {
        "failure"
    }
}

#[test]
fn capture_is_published_after_cleanup_and_suppressed_when_hook_fails() -> MlResult<()> {
    for fail in [false, true] {
        let ctx = ExecutionContext::new();
        let mut model = LinearRegression::new(&ctx, 1, 1)?;
        let mut optimizer = SGD::new(&ctx, 0.01)?;
        optimizer.register_all(&model.parameters())?;
        let input = ctx.input(vec![1.0], &[1, 1])?;
        let target = ctx.tensor(vec![2.0], &[1, 1])?;
        let inputs = [&input];
        let targets = [&target];
        let dataset = SupervisedDataset::new(&ctx, &inputs, &targets)?;
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut trainer = SupervisedTrainer::silent(&ctx).with_observer(Box::new(CaptureOrder {
            ctx: ctx.clone(),
            parameters: model.parameters().into_iter().cloned().collect(),
            events: events.clone(),
        }));
        if fail {
            trainer = trainer.with_hook(Box::new(FailHook));
        }
        let result = trainer.fit(&mut model, &mut optimizer, &dataset, EpochSchedule::new(1)?);
        if fail {
            assert!(matches!(
                result,
                Err(MlError::TensorError(TensorError::InvalidOperation {
                    op: "test_hook",
                    ..
                }))
            ));
            assert!(events.borrow().is_empty());
        } else {
            result?;
            assert_eq!(*events.borrow(), vec!["snapshot", "batch"]);
        }
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
        for p in model.parameters() {
            assert!(p.grad()?.is_none());
        }
    }
    Ok(())
}

#[test]
fn snapshot_owns_intermediate_gradients_after_graph_cleanup() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let p = ctx.parameter(vec![2.0], &[])?;
    let intermediate = p.square()?;
    let loss = intermediate.square()?;
    let snapshot = ctx.backward_snapshot(
        &loss,
        BackwardOptions::default(),
        CaptureProfile::Analysis,
        CaptureContext::default(),
    )?;
    assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    assert_eq!(snapshot.nodes.len(), 3);
    assert_eq!(snapshot.edges.len(), 2);
    assert!(
        snapshot
            .nodes
            .iter()
            .all(|node| node.gradient_stats.is_some())
    );
    assert!(intermediate.grad()?.is_none());
    assert_eq!(p.grad()?.map(|g| g.data()[0]), Some(32.0));
    drop(loss);
    drop(intermediate);
    drop(p);
    drop(ctx);
    assert!(DotEncoder::encode(&snapshot).contains("digraph"));
    Ok(())
}

#[test]
fn independent_contexts_capture_without_shared_sessions() -> MlResult<()> {
    let a = ExecutionContext::new();
    let b = ExecutionContext::new();
    let p = a.parameter(vec![2.0], &[])?;
    let q = b.parameter(vec![3.0], &[])?;
    let x = p.square()?;
    let y = q.square()?;
    let snapshot = a.graph_snapshot(CaptureProfile::Structure, CaptureContext::default())?;
    assert_eq!(snapshot.nodes.len(), 2);
    assert!(
        snapshot
            .nodes
            .iter()
            .all(|n| n.value_stats.is_none() && n.gradient_stats.is_none())
    );
    x.backward()?;
    assert_eq!(b.graph_stats()?.graph_nodes, 1);
    y.backward()?;
    Ok(())
}
