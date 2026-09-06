mod support;
use support::*;
use trench_deep::{
    contracts::*,
    nn::LinearRegression,
    optimizer::{Optimizer, SGD},
    trainer::*,
    *,
};

fn reference() -> ExecutionContext {
    ExecutionContextBuilder::empty()
        .storage(SlotStore::default())
        .autograd(Tape::default())
        .operations(ReferenceOps)
        .build()
}
fn train(ctx: &ExecutionContext) -> MlResult<Vec<f32>> {
    let mut model = LinearRegression::new(ctx, 1, 1)?;
    ctx.replace_parameter(
        model.layer().weight().variable(),
        TensorBuffer::from_vec(vec![0.25], &[1, 1])?,
    )?;
    ctx.replace_parameter(
        model.layer().bias().variable(),
        TensorBuffer::from_vec(vec![0.0], &[1])?,
    )?;
    let input = ctx.input(vec![1.0, 2.0], &[2, 1])?;
    let target = ctx.tensor(vec![2.0, 4.0], &[2, 1])?;
    let x = [&input];
    let y = [&target];
    let data = SupervisedDataset::new(ctx, &x, &y)?;
    let mut optimizer = SGD::new(ctx, 0.05)?;
    optimizer.register_all(&model.parameters())?;
    SupervisedTrainer::silent(ctx).fit(
        &mut model,
        &mut optimizer,
        &data,
        EpochSchedule::new(3)?,
    )?;
    assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    model
        .parameters()
        .into_iter()
        .map(|p| p.tensor().item())
        .collect()
}
#[test]
fn all_providers_can_be_replaced_without_default_features() -> MlResult<()> {
    assert!(train(&reference())?.iter().all(|x| x.is_finite()));
    Ok(())
}
#[cfg(all(
    feature = "builtinStorage",
    feature = "builtinKernels",
    feature = "enableBackward"
))]
#[test]
fn independent_storage_graph_and_compute_preserve_the_same_model_and_trainer() -> MlResult<()> {
    let expected = train(&ExecutionContext::new())?;
    for ctx in [
        reference(),
        ExecutionContext::builder()
            .storage(SlotStore::default())
            .build(),
        ExecutionContext::builder()
            .autograd(Tape::default())
            .build(),
        ExecutionContext::builder().operations(ReferenceOps).build(),
    ] {
        let actual = train(&ctx)?;
        for (a, b) in actual.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
    }
    Ok(())
}
#[test]
fn missing_providers_report_the_required_module() -> MlResult<()> {
    let empty = ExecutionContextBuilder::empty().build();
    assert!(matches!(
        empty.scalar(1.0),
        Err(MlError::DependencyUnavailable {
            module: "storage",
            ..
        })
    ));
    let no_ops = ExecutionContextBuilder::empty()
        .storage(SlotStore::default())
        .build();
    let x = no_ops.scalar(1.0)?;
    assert!(matches!(
        x.square(),
        Err(MlError::DependencyUnavailable {
            module: "operations",
            ..
        })
    ));
    let no_grad = ExecutionContextBuilder::empty()
        .storage(SlotStore::default())
        .operations(ReferenceOps)
        .build();
    let p = no_grad.parameter(vec![2.0], &[])?;
    assert!(matches!(
        p.square(),
        Err(MlError::DependencyUnavailable {
            module: "autograd",
            ..
        })
    ));
    assert_eq!(no_grad.no_grad(|| p.square())?.tensor().item()?, 4.0);
    assert!(matches!(
        p.backward(),
        Err(MlError::DependencyUnavailable {
            module: "autograd",
            ..
        })
    ));
    Ok(())
}
#[test]
fn unsupported_operation_does_not_disable_other_operations() -> MlResult<()> {
    let ctx = reference();
    let x = ctx.scalar(2.0)?;
    assert!(matches!(
        x.sin(),
        Err(MlError::UnsupportedCapability { .. })
    ));
    assert_eq!(x.square()?.item()?, 4.0);
    Ok(())
}
#[test]
fn alternate_store_preserves_detach_alias_and_handle_lifetime() -> MlResult<()> {
    let ctx = reference();
    let p = ctx.parameter(vec![2.0], &[])?;
    let alias = p.detach()?;
    ctx.replace_parameter(p.variable(), TensorBuffer::from_vec(vec![5.0], &[])?)?;
    drop(p);
    assert_eq!(alias.tensor().item()?, 5.0);
    assert_eq!(ctx.graph_stats()?.tensors, 1);
    drop(alias);
    assert_eq!(ctx.graph_stats()?.tensors, 0);
    Ok(())
}

struct FailingEpochLoader {
    parameter: Parameter,
}
impl BatchLoader for FailingEpochLoader {
    type Batch = SupervisedBatch;
    fn begin_epoch(&mut self, _: usize, _: &TrainingRuntime) -> MlResult<()> {
        let _graph = self.parameter.square()?;
        Err(MlError::UnsupportedCapability {
            module: "fixture loader",
            capability: "epoch initialization",
            operation: "begin_epoch",
        })
    }
    fn next_batch(&mut self) -> MlResult<Option<Self::Batch>> {
        Ok(None)
    }
    fn batch_count(&self) -> Option<usize> {
        None
    }
}

#[test]
fn failed_epoch_initialization_cleans_graph_and_allows_the_next_fit() -> MlResult<()> {
    let ctx = reference();
    let mut model = LinearRegression::new(&ctx, 1, 1)?;
    let mut optimizer = SGD::new(&ctx, 0.05)?;
    optimizer.register_all(&model.parameters())?;
    let loader = FailingEpochLoader {
        parameter: model.parameters()[0].clone(),
    };
    let error = SupervisedTrainer::silent(&ctx).fit(
        &mut model,
        &mut optimizer,
        loader,
        EpochSchedule::new(1)?,
    );
    assert!(matches!(
        error,
        Err(MlError::UnsupportedCapability {
            module: "fixture loader",
            ..
        })
    ));
    assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    for p in model.parameters() {
        assert!(p.grad()?.is_none());
    }
    assert!(train(&ctx)?.iter().all(|x| x.is_finite()));
    Ok(())
}

#[test]
fn data_order_and_action_randomness_are_independent() {
    let a = TrainingRuntime::new(12);
    let b = TrainingRuntime::new(12);
    for _ in 0..50 {
        a.random_f32();
    }
    let mut x = (0..100).collect::<Vec<_>>();
    let mut y = x.clone();
    a.shuffle(&mut x);
    b.shuffle(&mut y);
    assert_eq!(x, y);
    a.reseed(37);
    b.reseed(37);
    a.shuffle(&mut x);
    for _ in 0..50 {
        assert_eq!(a.random_f32(), b.random_f32());
    }
}

#[test]
fn shared_parameters_register_once() -> MlResult<()> {
    let ctx = reference();
    let p = ctx.parameter(vec![2.0], &[])?;
    let mut optimizer = SGD::new(&ctx, 0.1)?;
    optimizer.register_all(&[&p, &p])?;
    assert_eq!(optimizer.registered_param_count(), 1);
    p.square()?.backward()?;
    optimizer.step()?;
    assert!((p.tensor().item()? - 1.6).abs() < 1e-6);
    Ok(())
}

#[test]
fn model_noise_and_trainer_seed_do_not_change_initialization() -> MlResult<()> {
    let make = || {
        ExecutionContextBuilder::empty()
            .storage(SlotStore::default())
            .autograd(Tape::default())
            .operations(ReferenceOps)
            .initialization_seed(19)
            .model_seed(77)
            .build()
    };
    let a = make();
    let b = make();
    a.model_uniform(100, 0.2)?;
    let left = LinearRegression::new(&a, 2, 1)?;
    let right = LinearRegression::new(&b, 2, 1)?;
    for (p, q) in left.parameters().iter().zip(right.parameters()) {
        assert_eq!(p.tensor().to_vec()?, q.tensor().to_vec()?);
    }
    let before = left.parameters()[0].tensor().to_vec()?;
    let _trainer = SupervisedTrainer::silent(&a).with_seed(999);
    assert_eq!(left.parameters()[0].tensor().to_vec()?, before);
    Ok(())
}

#[cfg(not(feature = "enableVisualization"))]
#[test]
fn capture_api_reports_missing_feature_without_consuming_graph() -> MlResult<()> {
    use trench_deep::visualization::*;
    let ctx = reference();
    let p = ctx.parameter(vec![2.0], &[])?;
    let loss = p.square()?;
    assert!(matches!(
        ctx.backward_snapshot(
            &loss,
            BackwardOptions::default(),
            CaptureProfile::Analysis,
            CaptureContext::default()
        ),
        Err(MlError::DependencyUnavailable {
            module: "visualization",
            ..
        })
    ));
    assert_eq!(ctx.graph_stats()?.graph_nodes, 1);
    loss.backward()?;
    Ok(())
}
