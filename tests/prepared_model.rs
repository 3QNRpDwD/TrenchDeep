#![cfg(all(
    feature = "builtinStorage",
    feature = "builtinKernels",
    feature = "enableBackward"
))]
use std::cell::Cell;
use trench_deep::{
    contracts::Operation,
    optimizer::{Adam, Optimizer},
    runtime::prepared::*,
    trainer::*,
    *,
};
struct Linear {
    ctx: ExecutionContext,
    weight: Parameter,
    descriptions: Cell<usize>,
}
impl TrainableModel for Linear {
    fn context_id(&self) -> ContextId {
        self.ctx.id()
    }
    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight]
    }
}

fn make() -> MlResult<(ExecutionContext, Linear, Adam)> {
    let ctx = ExecutionContext::new();
    let model = Linear {
        ctx: ctx.clone(),
        weight: ctx.parameter(vec![0.4, 0.7], &[2, 1])?,
        descriptions: Cell::new(0),
    };
    let mut adam = Adam::new(&ctx, 0.001, 0.9, 0.999, 1e-8)?;
    adam.register_all(&model.parameters())?;
    Ok((ctx, model, adam))
}
#[test]
fn loss_only_model_keeps_prediction_and_matches_default_gradients() -> MlResult<()> {
    let (ctx, mut model, _) = make()?;
    let batch = SupervisedBatch {
        inputs: ctx.tensor(vec![1.0, 2.0], &[1, 2])?.as_variable()?,
        targets: ctx.tensor(vec![0.0], &[1, 1])?,
    };
    let batch = trench_deep::trainer::Supervised.execution_batch(
        &mut model,
        &batch,
        &TrainingStepContext::default(),
    )?;
    let mut full = ctx.prepare_training(
        &mut model,
        &trench_deep::trainer::Supervised,
        &loss(),
        &batch.inputs,
    )?;
    let mut selected = ctx.prepare_training_for_loss(
        &mut model,
        &trench_deep::trainer::Supervised,
        &loss(),
        &batch.inputs,
    )?;
    assert_eq!(full.executor().plan().backward_plan_stats().roots, 2);
    assert_eq!(selected.executor().plan().backward_plan_stats().roots, 1);
    let read = |output: ModelOutput| {
        let prediction = output.prediction.unwrap();
        prediction.retain_grad()?;
        let values = prediction.tensor().to_vec()?;
        output.loss.backward()?;
        Ok((
            values,
            model.weight.grad()?.unwrap(),
            prediction.grad()?.unwrap(),
        ))
    };
    let a = full.run(&model, &batch.inputs, read)?;
    let b = selected.run(&model, &batch.inputs, read)?;
    assert_eq!(a.0, b.0);
    assert_eq!(a.1.data(), b.1.data());
    assert_eq!(a.2.data(), b.2.data());
    Ok(())
}
#[test]
fn common_model_api_reuses_plan_and_recovers_from_callback_errors() -> MlResult<()> {
    let (ctx, mut model, mut adam) = make()?;
    let batch = SupervisedBatch {
        inputs: ctx.tensor(vec![1.0, 2.0], &[1, 2])?.as_variable()?,
        targets: ctx.tensor(vec![0.0], &[1, 1])?,
    };
    let batch = trench_deep::trainer::Supervised.execution_batch(
        &mut model,
        &batch,
        &TrainingStepContext::default(),
    )?;
    let before = ctx.graph_stats()?;
    let mut prepared = ctx.prepare_training(
        &mut model,
        &trench_deep::trainer::Supervised,
        &loss(),
        &batch.inputs,
    )?;
    assert_eq!(model.descriptions.get(), 1);
    let bytes = prepared.executor().arena_bytes();
    let failed: MlResult<()> = prepared.run(&model, &batch.inputs, |output| {
        output.loss.backward()?;
        Err(ContextError::BorrowConflict.into())
    });
    assert!(failed.is_err());
    assert!(model.weight.grad()?.is_none());
    for _ in 0..100 {
        prepared.run(&model, &batch.inputs, |output| {
            output.loss.backward()?;
            adam.step()
        })?;
    }
    assert_eq!(model.descriptions.get(), 1);
    assert_eq!(prepared.executor().arena_bytes(), bytes);
    assert_eq!(ctx.graph_stats()?, before);
    assert!(model.weight.grad()?.is_none());
    Ok(())
}
#[test]
fn common_trainer_prepares_once_per_shape_and_reuses_across_epochs() -> MlResult<()> {
    let (ctx, mut model, mut adam) = make()?;
    let x = ctx.tensor(vec![1.0, 2.0], &[1, 2])?.as_variable()?;
    let y = ctx
        .tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?
        .as_variable()?;
    let t = ctx.tensor(vec![0.0], &[1, 1])?;
    let u = ctx.tensor(vec![0.0, 0.0], &[2, 1])?;
    let inputs = [&x, &y];
    let targets = [&t, &u];
    let dataset = SupervisedDataset::new(&ctx, &inputs, &targets)?;
    let before = ctx.graph_stats()?;
    let result = Trainer::supervised(&ctx).silent().prepared().fit(
        &mut model,
        &loss(),
        &mut adam,
        &dataset,
        EpochSchedule::new(3)?.with_tolerance(0.0),
    )?;
    assert_eq!(result.units_completed, 3);
    assert_eq!(model.descriptions.get(), 2);
    assert_eq!(ctx.graph_stats()?, before);
    assert!(model.weight.grad()?.is_none());
    Ok(())
}

#[test]
fn eager_loader_operations_remain_in_the_batch_graph() -> MlResult<()> {
    struct Loader {
        ctx: ExecutionContext,
        weight: Parameter,
        emitted: bool,
    }
    impl BatchLoader for Loader {
        type Batch = SupervisedBatch;
        fn begin_epoch(&mut self, _: usize, _: &TrainingRuntime) -> MlResult<()> {
            self.emitted = false;
            Ok(())
        }
        fn next_batch(&mut self) -> MlResult<Option<Self::Batch>> {
            if self.emitted {
                return Ok(None);
            }
            self.emitted = true;
            let inputs = self
                .ctx
                .execute(&Operation::Transpose(vec![1, 0]), &[self.weight.tensor()])?
                .remove(0)
                .as_variable()?;
            Ok(Some(SupervisedBatch {
                inputs,
                targets: self.ctx.tensor(vec![0.0], &[1, 1])?,
            }))
        }
        fn batch_count(&self) -> Option<usize> {
            Some(1)
        }
    }
    let (ctx, mut model, _) = make()?;
    let mut optimizer = trench_deep::optimizer::SGD::new(&ctx, 0.1)?;
    optimizer.register_all(&model.parameters())?;
    let loader = Loader {
        ctx: ctx.clone(),
        weight: model.weight.clone(),
        emitted: false,
    };
    Trainer::supervised(&ctx).silent().fit(
        &mut model,
        &loss(),
        &mut optimizer,
        loader,
        EpochSchedule::new(1)?,
    )?;
    let values = model.weight.tensor().to_vec()?;
    assert!((values[0] - 0.296).abs() < 1e-6 && (values[1] - 0.518).abs() < 1e-6);
    Ok(())
}

impl trench_deep::trainer::ForwardModel for Linear {
    fn forward(&self, input: &trench_deep::Variable) -> MlResult<trench_deep::Variable> {
        self.descriptions.set(self.descriptions.get() + 1);
        input.matmul(self.weight.tensor())
    }
}

fn loss() -> trench_deep::loss::MseLoss {
    trench_deep::loss::MseLoss::new()
}
