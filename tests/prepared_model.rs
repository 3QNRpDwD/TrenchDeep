#![cfg(all(
    feature = "builtinStorage",
    feature = "builtinKernels",
    feature = "enableBackward"
))]
use std::cell::Cell;
use trench_deep::{
    contracts::{LossKind, Operation},
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
impl PreparedModel for Linear {
    type Batch = SupervisedBatch;
    const PARADIGM: &'static str = "supervised";
    fn execution_batch(&mut self, batch: &Self::Batch) -> MlResult<PreparedBatch> {
        let inputs = ExecutionInputs::new("linear")
            .with("x", batch.inputs.tensor().clone())?
            .with("target", batch.targets.clone())?;
        let mut result = PreparedBatch::new(inputs, batch.inputs.tensor().shape()?[0]);
        result.target = Some(batch.targets.clone());
        Ok(result)
    }
    fn forward_inputs(&self, inputs: &ExecutionInputs) -> MlResult<ModelOutput> {
        self.descriptions.set(self.descriptions.get() + 1);
        let prediction = self
            .ctx
            .execute(
                &Operation::Matmul,
                &[inputs.get("x")?, self.weight.tensor()],
            )?
            .remove(0);
        let loss = self
            .ctx
            .execute(
                &Operation::Loss {
                    kind: LossKind::Mse,
                    reduction: Reduction::Mean,
                },
                &[&prediction, inputs.get("target")?],
            )?
            .remove(0);
        Ok(ModelOutput::new(
            loss.as_variable()?,
            Some(prediction.as_variable()?),
        ))
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
    let batch = model.execution_batch(&batch)?;
    let mut full = ctx.prepare_model(&model, &batch.inputs)?;
    let mut selected = ctx.prepare_model_for_loss(&model, &batch.inputs)?;
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
    let batch = model.execution_batch(&batch)?;
    let before = ctx.graph_stats()?;
    let mut prepared = ctx.prepare_model(&model, &batch.inputs)?;
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
    let result = Trainer::silent().prepared(&ctx).fit(
        &mut model,
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

impl SupervisedModel for Linear {
    fn forward_loss(
        &mut self,
        input: &Variable,
        target: &Tensor,
    ) -> MlResult<(Variable, Variable)> {
        let inputs = ExecutionInputs::new("linear")
            .with("x", input.tensor().clone())?
            .with("target", target.clone())?;
        let output = self.forward_inputs(&inputs)?;
        Ok((output.prediction.unwrap(), output.loss))
    }
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
    Trainer::silent().supervised(&ctx).fit(
        &mut model,
        &mut optimizer,
        loader,
        EpochSchedule::new(1)?,
    )?;
    let values = model.weight.tensor().to_vec()?;
    assert!((values[0] - 0.296).abs() < 1e-6 && (values[1] - 0.518).abs() < 1e-6);
    Ok(())
}
