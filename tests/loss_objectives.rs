#![cfg(all(
    feature = "builtinStorage",
    feature = "builtinKernels",
    feature = "enableBackward"
))]
use std::cell::Cell;
use trench_deep::{
    contracts::{LossKind, Operation},
    loss::*,
    nn::{BigramLm, PiClassifier},
    optimizer::{Optimizer, SGD},
    runtime::prepared::*,
    trainer::*,
    *,
};

fn wrapper(kind: LossKind, reduction: Reduction) -> Box<dyn Loss> {
    match kind {
        LossKind::Mse => Box::new(MseLoss::new().with_reduction(reduction)),
        LossKind::Mae => Box::new(MaeLoss::new().with_reduction(reduction)),
        LossKind::Huber { delta } => Box::new(HuberLoss::new(delta).with_reduction(reduction)),
        LossKind::BinaryCrossEntropy => {
            Box::new(BinaryCrossEntropyLoss::new().with_reduction(reduction))
        }
        LossKind::CrossEntropy => Box::new(CrossEntropyLoss::new().with_reduction(reduction)),
        LossKind::SoftmaxCrossEntropy => {
            Box::new(SoftmaxCrossEntropyLoss::new().with_reduction(reduction))
        }
    }
}
fn kinds() -> [LossKind; 6] {
    [
        LossKind::Mse,
        LossKind::Mae,
        LossKind::Huber { delta: 0.3 },
        LossKind::BinaryCrossEntropy,
        LossKind::CrossEntropy,
        LossKind::SoftmaxCrossEntropy,
    ]
}
fn evaluate(kind: LossKind, reduction: Reduction, wrapped: bool) -> MlResult<(Vec<f32>, Vec<f32>)> {
    let ctx = ExecutionContext::new();
    let prediction = ctx.parameter(vec![0.25, 0.75, 0.6, 0.4], &[2, 2])?;
    // A tracked target must still be excluded from the loss gradient.
    let target = ctx.parameter(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
    ctx.with_training_scope(|| {
        let loss = if wrapped {
            wrapper(kind, reduction).compute(&ctx, prediction.variable(), target.tensor())?
        } else {
            ctx.execute(
                &Operation::Loss { kind, reduction },
                &[prediction.tensor(), target.tensor()],
            )?
            .remove(0)
            .as_variable()?
        };
        let values = loss.tensor().to_vec()?;
        let root = if matches!(reduction, Reduction::None) {
            loss.sum()?
        } else {
            loss
        };
        root.backward()?;
        assert!(target.grad()?.is_none());
        Ok((values, prediction.grad()?.unwrap().data().to_vec()))
    })
}

#[test]
fn six_losses_preserve_values_gradients_reductions_and_errors() -> MlResult<()> {
    for kind in kinds() {
        for reduction in [Reduction::Mean, Reduction::Sum, Reduction::None] {
            assert_eq!(
                evaluate(kind, reduction, false)?,
                evaluate(kind, reduction, true)?
            );
        }
        let ctx = ExecutionContext::new();
        let prediction = ctx.input(vec![0.25, 0.75], &[1, 2])?;
        let bad_target = ctx.tensor(vec![0.0, 1.0, 0.0], &[1, 3])?;
        let a = wrapper(kind, Reduction::Mean)
            .compute(&ctx, &prediction, &bad_target)
            .unwrap_err();
        let b = ctx
            .execute(
                &Operation::Loss {
                    kind,
                    reduction: Reduction::Mean,
                },
                &[prediction.tensor(), &bad_target],
            )
            .unwrap_err();
        assert_eq!(a.to_string(), b.to_string());
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    }
    let ctx = ExecutionContext::new();
    let x = ctx.input(vec![0.5], &[])?;
    let y = ctx.tensor(vec![0.0], &[])?;
    for delta in [0.0, -1.0, f32::NAN] {
        assert_eq!(
            HuberLoss::new(delta)
                .with_reduction(Reduction::Mean)
                .compute(&ctx, &x, &y)
                .unwrap_err()
                .to_string(),
            ctx.huber_loss(x.tensor(), &y, delta, Reduction::Mean)
                .unwrap_err()
                .to_string(),
        );
    }
    let other = ExecutionContext::new();
    assert!(matches!(
        MseLoss::new().compute(&other, &x, &y),
        Err(MlError::ContextError(ContextError::Mismatch))
    ));
    Ok(())
}

struct Linear {
    ctx: ExecutionContext,
    weight: Parameter,
    calls: Cell<usize>,
}
impl TrainableModel for Linear {
    fn context_id(&self) -> ContextId {
        self.ctx.id()
    }
    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight]
    }
}
impl ForwardModel for Linear {
    fn forward(&self, x: &Variable) -> MlResult<Variable> {
        self.calls.set(self.calls.get() + 1);
        x.matmul(self.weight.tensor())
    }
}
fn linear(ctx: &ExecutionContext) -> MlResult<Linear> {
    Ok(Linear {
        ctx: ctx.clone(),
        weight: ctx.parameter(vec![0.5], &[1, 1])?,
        calls: Cell::new(0),
    })
}
struct DoubleMse;
impl Loss for DoubleMse {
    fn compute(&self, ctx: &ExecutionContext, p: &Variable, t: &Tensor) -> MlResult<Variable> {
        MseLoss::new().compute(ctx, p, t)?.mul(&ctx.scalar(2.0)?)
    }
}
fn train_loss<L: Loss>(model: &mut Linear, loss: L, prepared: bool) -> MlResult<f32> {
    let ctx = model.ctx.clone();
    ctx.replace_parameter(
        model.weight.variable(),
        TensorBuffer::from_vec(vec![0.5], &[1, 1])?,
    )?;
    let x = ctx.input(vec![1.0, 2.0], &[2, 1])?;
    let target = ctx.tensor(vec![0.0, 0.0], &[2, 1])?;
    let xs = [&x];
    let ys = [&target];
    let data = SupervisedDataset::new(&ctx, &xs, &ys)?;
    let mut optimizer = SGD::new(&ctx, 0.1)?;
    optimizer.register_all(&model.parameters())?;

    let trainer = Trainer::supervised(&ctx).silent();
    if prepared {
        trainer
            .prepared()
            .fit(model, &loss, &mut optimizer, &data, EpochSchedule::new(1)?)?;
    } else {
        trainer.fit(model, &loss, &mut optimizer, &data, EpochSchedule::new(1)?)?;
    }
    Ok(model.weight.tensor().item()?)
}

#[test]
fn same_model_accepts_replacement_and_custom_losses_without_stale_capture() -> MlResult<()> {
    for prepared in [false, true] {
        let ctx = ExecutionContext::new();
        let mut model = linear(&ctx)?;
        assert!((train_loss(&mut model, MseLoss::new(), prepared)? - 0.25).abs() < 1e-6);
        assert!((train_loss(&mut model, MaeLoss::new(), prepared)? - 0.35).abs() < 1e-6);
        assert!(train_loss(&mut model, DoubleMse, prepared)?.abs() < 1e-6);
        assert_eq!(model.calls.get(), 3);
        assert!(
            train_loss(
                &mut model,
                MseLoss::new().with_reduction(Reduction::None),
                prepared
            )
            .is_err()
        );
        assert_eq!(model.weight.tensor().item()?, 0.5);
        assert!(model.weight.grad()?.is_none());
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    }
    Ok(())
}

#[test]
fn extracted_objective_preserves_prepared_graph_roots_and_arena() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let mut model = linear(&ctx)?;
    let inputs = ExecutionInputs::new("supervised")
        .with("x", ctx.tensor(vec![1.0, 2.0], &[2, 1])?)?
        .with("target", ctx.tensor(vec![0.0, 0.0], &[2, 1])?)?;
    let reference = ctx
        .prepare_forward_with_roots(
            &inputs,
            &model.parameters(),
            PreparedMode::Training,
            &[0],
            |inputs| {
                let prediction = model.forward(&inputs.get("x")?.as_variable()?)?;
                let loss = prediction.mse_loss(inputs.get("target")?, Reduction::Mean)?;
                Ok(vec![loss.tensor().clone(), prediction.tensor().clone()])
            },
        )?
        .into_executor(&ctx)?;
    let actual =
        ctx.prepare_training_for_loss(&mut model, &Supervised, &MseLoss::new(), &inputs)?;
    assert_eq!(
        actual.executor().plan().node_count(),
        reference.plan().node_count()
    );
    assert_eq!(
        actual.executor().plan().slot_count(),
        reference.plan().slot_count()
    );
    assert_eq!(
        actual.executor().plan().backward_plan_stats(),
        reference.plan().backward_plan_stats()
    );
    assert_eq!(actual.executor().arena_bytes(), reference.arena_bytes());
    Ok(())
}

#[test]
fn semi_supervised_consistency_differentiates_both_views() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let model = PiClassifier::new(&ctx, 1, 1)?;
    // Linear weight = 1 and bias = 0; inspect gradients on the supplied views.
    for p in model.parameters() {
        let value = if p.tensor().shape()?.len() == 2 {
            1.0
        } else {
            0.0
        };
        ctx.replace_parameter(
            p.variable(),
            TensorBuffer::from_vec(vec![value; p.tensor().numel()?], &p.tensor().shape()?)?,
        )?;
    }
    let labeled = ctx.input(vec![0.0], &[1, 1])?;
    let target = ctx.tensor(vec![0.0], &[1, 1])?;
    let first = ctx.variable(vec![1.0], &[1, 1], RequiresGrad::Yes)?;
    let second = ctx.variable(vec![3.0], &[1, 1], RequiresGrad::Yes)?;
    let loss_fn = MseLoss::new();
    let objective = SemiSupervised::new(0.0)?;
    ctx.with_training_scope(|| {
        let (_, loss) = objective.forward_loss_with_augmentations(
            &loss_fn, &model, &labeled, &target, &first, &second, 0.5,
        )?;
        loss.backward()?;
        assert_eq!(first.grad()?.unwrap().data(), &[-2.0]);
        assert_eq!(second.grad()?.unwrap().data(), &[2.0]);
        Ok(())
    })
}

#[test]
fn built_in_semi_supervised_modes_preserve_rng_and_scheduled_updates() -> MlResult<()> {
    let mut runs = Vec::new();
    for prepared in [false, true] {
        let ctx = ExecutionContext::builder()
            .initialization_seed(8)
            .model_seed(9)
            .build();
        let mut model = PiClassifier::new(&ctx, 2, 2)?;
        let x = ctx.input(vec![0.2, 0.8], &[1, 2])?;
        let target = ctx.tensor(vec![0.0, 1.0], &[1, 2])?;
        let xs = [&x];
        let ys = [&target];
        let data = SemiSupervisedDataset::new(&ctx, &xs, &ys, &xs)?;
        let loss_fn = SoftmaxCrossEntropyLoss::new();
        let mut optimizer = SGD::new(&ctx, 0.01)?;
        optimizer.register_all(&model.parameters())?;
        let trainer = Trainer::semi_supervised(&ctx)
            .with_noise_scale(0.2)?
            .show_progress(false)
            .with_ramp(ConsistencyRamp::Sigmoid {
                max_weight: 1.0,
                ramp_epochs: 2,
            });
        let result = if prepared {
            trainer.prepared().fit(
                &mut model,
                &loss_fn,
                &mut optimizer,
                &data,
                EpochSchedule::new(3)?,
            )?
        } else {
            trainer.fit(
                &mut model,
                &loss_fn,
                &mut optimizer,
                &data,
                EpochSchedule::new(3)?,
            )?
        };
        assert_eq!(result.metrics["lambda"], 1.0);
        let weights = model
            .parameters()
            .iter()
            .map(|p| p.tensor().to_vec())
            .collect::<MlResult<Vec<_>>>()?;
        runs.push((result.final_loss, weights, ctx.model_uniform(4, 1.0)?));
    }
    assert_eq!(runs[0], runs[1]);
    Ok(())
}

#[test]
fn autoregressive_objective_preserves_tokens_and_modes() -> MlResult<()> {
    let mut runs = Vec::new();
    for prepared in [false, true] {
        let ctx = ExecutionContext::builder().initialization_seed(8).build();
        let mut model = BigramLm::new(&ctx, 2)?;
        let sequence = ctx.input(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0], &[3, 2])?;
        let loss_fn = SoftmaxCrossEntropyLoss::new();
        let objective = Autoregressive;
        ctx.with_training_scope(|| {
            let output = objective.forward_batch(
                &loss_fn,
                &mut model,
                &AutoregressiveBatch {
                    sequences: sequence.clone(),
                },
                &TrainingStepContext::default(),
            )?;
            assert_eq!(output.weight, 2);
            assert_eq!(output.tokens, Some(2));
            assert!(output.target.is_none());
            Ok(())
        })?;
        let xs = [&sequence];
        let data = AutoregressiveDataset::new(&ctx, &xs)?;
        let mut optimizer = SGD::new(&ctx, 0.01)?;
        optimizer.register_all(&model.parameters())?;
        let trainer = Trainer::autoregressive(&ctx).show_progress(false);
        let result = if prepared {
            trainer.prepared().fit(
                &mut model,
                &loss_fn,
                &mut optimizer,
                &data,
                EpochSchedule::new(2)?,
            )?
        } else {
            trainer.fit(
                &mut model,
                &loss_fn,
                &mut optimizer,
                &data,
                EpochSchedule::new(2)?,
            )?
        };
        runs.push((
            result.final_loss,
            result.metrics["perplexity"],
            model.weight().tensor().to_vec()?,
        ));
    }
    assert_eq!(runs[0], runs[1]);
    Ok(())
}

#[test]
fn loss_setter_changes_are_observed_by_the_next_fit() -> MlResult<()> {
    for prepared in [false, true] {
        let ctx = ExecutionContext::new();
        let mut model = linear(&ctx)?;
        let x = ctx.input(vec![2.0], &[1, 1])?;
        let target = ctx.tensor(vec![0.0], &[1, 1])?;
        let xs = [&x];
        let ys = [&target];
        let data = SupervisedDataset::new(&ctx, &xs, &ys)?;
        let mut optimizer = SGD::new(&ctx, 0.1)?;
        optimizer.register_all(&model.parameters())?;
        let trainer = Trainer::supervised(&ctx).silent();
        let mut loss = HuberLoss::new(0.25);
        macro_rules! check {
            ($trainer:expr) => {{
                let trainer = $trainer;
                for (delta, expected) in [(0.25, 0.45), (2.0, 0.3)] {
                    ctx.replace_parameter(
                        model.weight.variable(),
                        TensorBuffer::from_vec(vec![0.5], &[1, 1])?,
                    )?;
                    loss.set_delta(delta);
                    trainer.fit(
                        &mut model,
                        &loss,
                        &mut optimizer,
                        &data,
                        EpochSchedule::new(1)?,
                    )?;
                    assert!((model.weight.tensor().item()? - expected).abs() < 1e-6);
                }
            }};
        }
        if prepared {
            check!(trainer.prepared());
        } else {
            check!(trainer);
        }
        assert_eq!(model.calls.get(), 2);
    }
    Ok(())
}
