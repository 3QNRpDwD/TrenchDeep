use trench_deep::*;

#[test]
fn default_and_explicit_p1_preserve_provider_composition() -> MlResult<()> {
    assert_eq!(ExecutionContext::new().route(), ExecutionRoute::P1);
    let ctx = ExecutionContextBuilder::empty()
        .route(ExecutionRoute::P1)
        .build()?;
    assert_eq!(ctx.route(), ExecutionRoute::P1);
    assert!(matches!(
        ctx.scalar(1.0),
        Err(MlError::DependencyUnavailable {
            module: "storage",
            ..
        })
    ));
    Ok(())
}

#[test]
fn legacy_selection_never_silently_runs_p1() {
    let result = ExecutionContext::builder()
        .route(ExecutionRoute::Legacy)
        .build();
    #[cfg(not(feature = "legacyBenchmark"))]
    assert!(matches!(
        result,
        Err(MlError::DependencyUnavailable {
            module: "legacy execution",
            ..
        })
    ));
    #[cfg(feature = "legacyBenchmark")]
    assert_eq!(result.unwrap().route(), ExecutionRoute::Legacy);
}

#[cfg(feature = "legacyBenchmark")]
#[test]
fn legacy_public_scope_preserves_owned_outputs_and_rejects_unsupported_requests() -> MlResult<()> {
    let ctx = ExecutionContext::builder()
        .route(ExecutionRoute::Legacy)
        .build()?;
    assert!(
        ExecutionContext::builder()
            .route(ExecutionRoute::Legacy)
            .build()
            .is_err()
    );
    let p = ctx.parameter(vec![2.0], &[])?;
    let baseline = ctx.graph_stats()?.tensors;
    for _ in 0..16 {
        let output = ctx.with_training_scope(|| {
            assert!(ctx.with_training_scope(|| Ok(())).is_err());
            let loss = p.mul(p.tensor())?;
            loss.backward()?;
            assert_eq!(p.grad()?.unwrap().data(), &[4.0]);
            Ok(loss)
        })?;
        assert_eq!(output.tensor().item()?, 4.0);
        assert_eq!(ctx.graph_stats()?.tensors, baseline + 1);
        assert!(p.grad()?.is_none());
        drop(output);
        assert_eq!(ctx.graph_stats()?.tensors, baseline);
    }
    let failure: MlResult<()> = ctx.with_training_scope(|| {
        let _loss = p.mul(p.tensor())?;
        p.tensor().detach()?;
        Ok(())
    });
    assert!(matches!(
        failure,
        Err(MlError::UnsupportedCapability { .. })
    ));
    assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    assert_eq!(ctx.graph_stats()?.tensors, baseline);
    assert!(matches!(
        p.tensor().detach(),
        Err(MlError::UnsupportedCapability { .. })
    ));
    let foreign = ExecutionContext::new();
    #[cfg(feature = "builtinStorage")]
    assert!(matches!(
        p.add(&foreign.scalar(1.0)?),
        Err(MlError::ContextError(ContextError::Mismatch))
    ));
    let _ = foreign;
    Ok(())
}

#[cfg(feature = "legacyBenchmark")]
#[test]
fn legacy_rejects_explicit_provider_overrides() {
    assert!(matches!(
        ExecutionContext::builder()
            .without_operations()
            .route(ExecutionRoute::Legacy)
            .build(),
        Err(MlError::UnsupportedCapability { .. })
    ));
}

#[cfg(all(
    feature = "legacyBenchmark",
    feature = "builtinStorage",
    feature = "builtinKernels"
))]
#[test]
fn same_public_trainer_runs_both_routes() -> MlResult<()> {
    use trench_deep::{
        optimizer::{Optimizer, SGD},
        trainer::*,
    };
    struct Model {
        ctx: ExecutionContext,
        weight: Parameter,
    }
    impl TrainableModel for Model {
        fn context_id(&self) -> ContextId {
            self.ctx.id()
        }
        fn parameters(&self) -> Vec<&Parameter> {
            vec![&self.weight]
        }
    }
    impl UnsupervisedModel for Model {
        fn forward_loss(&mut self, input: &Variable) -> MlResult<(Variable, Variable)> {
            let prediction = input.mul(self.weight.tensor())?;
            let loss = prediction.mul(prediction.tensor())?;
            Ok((prediction, loss))
        }
    }
    fn run(route: ExecutionRoute) -> MlResult<Vec<f32>> {
        let ctx = ExecutionContext::builder().route(route).build()?;
        let mut model = Model {
            weight: ctx.parameter(vec![2.0], &[])?,
            ctx: ctx.clone(),
        };
        let input = ctx.input(vec![0.5], &[])?;
        let inputs = [&input];
        let dataset = UnsupervisedDataset::new(&ctx, &inputs)?;
        let baseline = ctx.graph_stats()?.tensors;
        let mut optimizer = SGD::new(&ctx, 0.1)?;
        optimizer.register_all(&model.parameters())?;
        UnsupervisedTrainer::silent(&ctx).fit(
            &mut model,
            &mut optimizer,
            &dataset,
            EpochSchedule::new(3)?,
        )?;
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
        assert_eq!(ctx.graph_stats()?.tensors, baseline);
        assert!(model.weight.grad()?.is_none());
        ctx.no_grad(|| {
            let prediction = input.mul(model.weight.tensor())?;
            assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
            assert!(prediction.tensor().item()?.is_finite());
            Ok(())
        })?;
        model.weight.tensor().to_vec()
    }
    assert_eq!(run(ExecutionRoute::P1)?, run(ExecutionRoute::Legacy)?);
    Ok(())
}
