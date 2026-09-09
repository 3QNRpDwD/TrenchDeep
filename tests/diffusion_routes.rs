#![cfg(all(
    feature = "legacyBenchmark",
    feature = "builtinStorage",
    feature = "builtinKernels"
))]
//! Direct product-model training: no legacy reference model or replay wrapper.
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};
use trench_deep::{
    nn::{Diffusion, DiffusionScheduler, Unet},
    optimizer::{Adam, Optimizer},
    trainer::*,
    *,
};

#[derive(Default)]
struct Trace {
    losses: Vec<f32>,
    weights: Vec<BTreeMap<String, Vec<f32>>>,
}
struct Observer {
    ctx: ExecutionContext,
    named: BTreeMap<String, Parameter>,
    tensors: usize,
    trace: Rc<RefCell<Trace>>,
}
impl TrainingObserver for Observer {
    fn on_batch_end(&mut self, event: &BatchEndContext) {
        self.trace.borrow_mut().losses.push(event.loss);
    }
    fn on_epoch_end(&mut self, _: &EpochContext) {
        assert_eq!(self.ctx.graph_stats().unwrap().graph_nodes, 0);
        assert_eq!(self.ctx.graph_stats().unwrap().tensors, self.tensors);
        assert!(self.named.values().all(|p| p.grad().unwrap().is_none()));
        self.trace.borrow_mut().weights.push(
            self.named
                .iter()
                .map(|(name, p)| (name.clone(), p.tensor().to_vec().unwrap()))
                .collect(),
        );
    }
}

fn run(route: ExecutionRoute) -> MlResult<(BTreeMap<String, Vec<f32>>, Trace)> {
    let ctx = ExecutionContext::builder()
        .initialization_seed(7)
        .route(route)
        .build()?;
    assert_eq!(ctx.route(), route);
    let mut model = Diffusion::new(
        &ctx,
        Unet::new(&ctx, 1, 8, &[1, 2], 4, &[])?,
        DiffusionScheduler::linear(10, 1e-4, 0.02)?,
        11,
    )?;
    let named: BTreeMap<_, _> = model
        .unet
        .named_parameters()
        .into_iter()
        .map(|(name, p)| (name, p.clone()))
        .collect();
    let initial = named
        .iter()
        .map(|(name, p)| Ok((name.clone(), p.tensor().to_vec()?)))
        .collect::<MlResult<BTreeMap<_, _>>>()?;
    let mut optimizer = Adam::new(&ctx, 1e-3, 0.9, 0.999, 1e-8)?;
    optimizer.register_all(&model.parameters())?;
    let dataset = DatasetBuilder::from_source(MemorySource::new(vec![
        ctx.tensor(vec![0.5; 64], &[1, 8, 8])?,
        ctx.tensor(vec![0.5; 64], &[1, 8, 8])?,
    ]))
    .map(|input| Ok(UnsupervisedSample::new(input)))
    .build()?;
    let mut loader = DataLoader::builder(dataset)
        .collator(UnsupervisedStackCollator::new())
        .batch_size(2)
        .shuffle(false)
        .build()?;
    let trace = Rc::new(RefCell::new(Trace::default()));
    let observer = Observer {
        ctx: ctx.clone(),
        named,
        tensors: ctx.graph_stats()?.tensors,
        trace: trace.clone(),
    };
    // The exact same concrete Diffusion and Adam are passed to the public Trainer
    // on both routes. Diffusion::forward_loss owns timestep/noise generation.
    let result = Trainer::builder()
        .metrics(Metrics::all())
        .show_progress(false)
        .build()
        .unsupervised(&ctx)
        .with_observer(Box::new(observer))
        .fit(
            &mut model,
            &mut optimizer,
            &mut loader,
            EpochSchedule::new(3)?.with_tolerance(1e-10),
        )?;
    assert_eq!(result.units_completed, 3);
    for key in [
        "avg_loss",
        "grad_norm",
        "update_ratio",
        "forward_secs",
        "backward_secs",
    ] {
        assert!(
            result.metrics.get(key).is_some_and(|v| v.is_finite()),
            "{key}"
        );
    }
    let trace = std::mem::take(&mut *trace.borrow_mut());
    assert_eq!(trace.losses.len(), 3);
    assert_eq!(trace.weights.len(), 3);
    assert_eq!(result.final_loss, trace.losses[2]);
    assert!(
        initial
            .iter()
            .any(|(name, values)| values != &trace.weights[2][name]),
        "Adam must update weights"
    );
    Ok((initial, trace))
}

#[test]
fn product_diffusion_and_common_trainer_switch_execution_routes() -> MlResult<()> {
    let (initial, p1) = run(ExecutionRoute::P1)?;
    let (other_initial, legacy) = run(ExecutionRoute::Legacy)?;
    // Both executions use the same product RNG implementation and seeds. Check
    // initial weights explicitly rather than assuming seed equality is enough.
    assert_eq!(initial, other_initial);
    let close = |a: f32, b: f32| {
        assert!(
            a.is_finite()
                && b.is_finite()
                && (a - b).abs() <= 1e-3f32.max(1e-3 * a.abs().max(b.abs())),
            "{a} != {b}"
        )
    };
    for (a, b) in p1.losses.iter().zip(&legacy.losses) {
        close(*a, *b);
    }
    for (a, b) in p1.weights.iter().zip(&legacy.weights) {
        assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
        for (name, values) in a {
            assert_eq!(values.len(), b[name].len());
            for (x, y) in values.iter().zip(&b[name]) {
                close(*x, *y);
            }
        }
    }
    Ok(())
}
