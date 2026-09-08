mod support;
use support::{ReferenceOps, SlotStore, Tape};
use trench_deep::{
    contracts::*,
    optimizer::{Optimizer, SGD},
    trainer::*,
    *,
};

fn repeated_batches_release_tensors(ctx: ExecutionContext) -> MlResult<()> {
    let parameter = ctx.parameter(vec![2.0], &[])?;
    let baseline = ctx.graph_stats()?;
    for batch in 0..64 {
        let fail = batch % 3 == 1;
        let result = ctx.with_training_scope(|| {
            let input = ctx.scalar(0.5)?;
            // Fan-out/fan-in exercises multiple graph references to one value.
            let branch = parameter.mul(&input)?;
            let loss = branch.mul(branch.tensor())?;
            assert!(ctx.graph_stats()?.tensors > baseline.tensors);
            if fail {
                return Err(MlError::UnsupportedCapability {
                    module: "batch failure fixture",
                    capability: "forward",
                    operation: "batch",
                });
            }
            loss.backward()?;
            let gradient = parameter
                .grad()?
                .expect("successful backward must produce a gradient");
            assert_eq!(gradient.data(), &[1.0]);
            // A returned tensor must survive scope cleanup until its owner drops it.
            Ok(loss)
        });
        if fail {
            assert!(matches!(
                result,
                Err(MlError::UnsupportedCapability {
                    module: "batch failure fixture",
                    ..
                })
            ));
        } else {
            let loss = result?;
            assert_eq!(loss.tensor().item()?, 1.0);
            let held = ctx.graph_stats()?;
            assert_eq!(held.graph_nodes, 0);
            assert_eq!(held.tensors, baseline.tensors + 1);
            drop(loss);
        }
        assert!(parameter.grad()?.is_none());
        assert_eq!(
            ctx.graph_stats()?,
            baseline,
            "batch {batch} leaked runtime state"
        );
    }
    Ok(())
}

#[test]
fn repeated_batches_release_external_provider_tensors() -> MlResult<()> {
    repeated_batches_release_tensors(
        ExecutionContextBuilder::empty()
            .storage(SlotStore::default())
            .autograd(Tape::default())
            .operations(ReferenceOps)
            .build(),
    )
}

#[test]
#[cfg(all(
    feature = "builtinStorage",
    feature = "builtinKernels",
    feature = "enableBackward"
))]
fn repeated_batches_release_builtin_provider_tensors() -> MlResult<()> {
    repeated_batches_release_tensors(ExecutionContext::new())
}

#[test]
fn public_training_scope_supports_external_training_without_builtin_implementations() -> MlResult<()>
{
    let ctx = ExecutionContextBuilder::empty()
        .storage(SlotStore::default())
        .autograd(Tape::default())
        .operations(ReferenceOps)
        .build();
    let p = ctx.parameter(vec![2.0], &[])?;
    let result = ctx.with_training_scope(|| {
        assert!(matches!(
            ctx.with_training_scope(|| Ok(())),
            Err(MlError::ContextError(ContextError::ActiveGraphConflict))
        ));
        let loss = p.square()?;
        loss.backward()?;
        assert!(p.grad()?.is_some());
        loss.tensor().item()
    })?;
    assert_eq!(result, 4.0);
    assert!(p.grad()?.is_none());
    assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    let result: MlResult<()> = ctx.with_training_scope(|| {
        let _loss = p.square()?;
        Err(TensorError::InvalidOperation {
            op: "external_trainer",
            reason: "failure".into(),
        }
        .into())
    });
    assert!(result.is_err());
    assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    ctx.with_training_scope(|| Ok(()))
}

#[derive(Debug, Default)]
struct FaultTape {
    inner: Tape,
    failed: bool,
}
impl AutogradEngine for FaultTape {
    fn record(&mut self, r: GradientRecord) -> MlResult<()> {
        self.inner.record(r)
    }
    fn get(&self, id: TensorId) -> Option<GradientRecord> {
        self.inner.get(id)
    }
    fn nodes(&self) -> Vec<TensorId> {
        self.inner.nodes()
    }
    fn order(&self, id: TensorId) -> MlResult<Vec<TensorId>> {
        self.inner.order(id)
    }
    fn remove(&mut self, id: TensorId) -> MlResult<Option<GradientRecord>> {
        if !self.failed {
            self.failed = true;
            return Err(MlError::UnsupportedCapability {
                module: "fault tape",
                capability: "remove",
                operation: "cleanup",
            });
        }
        self.inner.remove(id)
    }
}
struct BrokenModel {
    ctx: ExecutionContext,
    p: Parameter,
}
impl TrainableModel for BrokenModel {
    fn context_id(&self) -> ContextId {
        self.ctx.id()
    }
    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.p]
    }
}
impl SupervisedModel for BrokenModel {
    fn forward_loss(&mut self, _: &Variable, _: &Tensor) -> MlResult<(Variable, Variable)> {
        let _graph = self.p.square()?;
        Err(MlError::UnsupportedCapability {
            module: "broken model",
            capability: "forward",
            operation: "forward_loss",
        })
    }
}
#[test]
fn primary_and_cleanup_errors_are_both_preserved() -> MlResult<()> {
    let ctx = ExecutionContextBuilder::empty()
        .storage(SlotStore::default())
        .autograd(FaultTape::default())
        .operations(ReferenceOps)
        .build();
    let mut model = BrokenModel {
        ctx: ctx.clone(),
        p: ctx.parameter(vec![2.0], &[])?,
    };
    let mut optimizer = SGD::new(&ctx, 0.01)?;
    optimizer.register_all(&model.parameters())?;
    let x = ctx.input(vec![1.0], &[1, 1])?;
    let y = ctx.scalar(1.0)?;
    let xs = [&x];
    let ys = [&y];
    let data = SupervisedDataset::new(&ctx, &xs, &ys)?;
    let error = SupervisedTrainer::silent(&ctx).fit(
        &mut model,
        &mut optimizer,
        &data,
        EpochSchedule::new(1)?,
    );
    match error {
        Err(MlError::CleanupError { primary, cleanup }) => {
            assert!(matches!(
                *primary,
                MlError::UnsupportedCapability {
                    module: "broken model",
                    ..
                }
            ));
            assert!(matches!(
                *cleanup,
                MlError::UnsupportedCapability {
                    module: "fault tape",
                    ..
                }
            ));
        }
        other => panic!("unexpected result: {other:?}"),
    }
    // Drop retries cleanup without replacing the already reported primary error.
    assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    assert!(model.p.grad()?.is_none());
    Ok(())
}

struct MissingDataset;
impl Dataset for MissingDataset {
    type Sample = ();
    fn len(&self) -> usize {
        1
    }
    fn get(&self, _: usize) -> Option<&()> {
        None
    }
}
#[test]
fn a_dataset_that_cannot_supply_its_sample_returns_an_error() -> MlResult<()> {
    let mut loader = DataLoader::builder(MissingDataset)
        .collator(|_: &[&()]| Ok(()))
        .build()?;
    loader.begin_epoch(0, &TrainingRuntime::new(1))?;
    assert!(matches!(
        loader.next_batch(),
        Err(MlError::DataError(DataError::MissingSample { index: 0 }))
    ));
    Ok(())
}
