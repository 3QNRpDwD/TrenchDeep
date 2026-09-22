#![cfg(all(
    feature = "builtinStorage",
    feature = "builtinKernels",
    feature = "enableBackward"
))]
use trench_deep::{contracts::Operation, runtime::prepared::*, *};

#[test]
fn reshape_views_share_forward_storage_but_not_gradients() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let w = ctx.parameter(vec![1., 2., 3., 4., 5., 6.], &[2, 3])?;
    let mut p = PreparedProgram::new();
    let weight = p.parameter(&[2, 3])?;
    let square = p.operation(Operation::Square, &[weight])?;
    let view = p.operation(Operation::Reshape(vec![3, 2]), &[square])?;
    let flat = p.operation(Operation::Reshape(vec![6]), &[view])?;
    let left = p.operation(Operation::Sum, &[square])?;
    let right = p.operation(Operation::Sum, &[flat])?;
    let loss = p.operation(Operation::Add, &[left, right])?;
    let plan = ctx.prepare(
        &p,
        &[&w],
        &[loss, square, view, flat],
        PreparedMode::Training,
    )?;
    let before = plan.buffer_plan().layout.unreused_bytes;
    let mut executor = plan.into_executor(&ctx)?;
    let buffers = executor.plan().buffer_plan();
    buffers.validate()?;
    assert_eq!(buffers.tensor_aliases[2], 1);
    assert_eq!(buffers.tensor_aliases[3], 1);
    assert_eq!(buffers.aliases[2], 2);
    assert_eq!(buffers.aliases[3], 3);
    assert_eq!(before - buffers.layout.unreused_bytes, 2 * 6 * 4);
    let held = executor.with_run(&[], &[&w], |out| {
        let a = out[1].as_variable()?;
        let b = out[2].as_variable()?;
        a.retain_grad()?;
        b.retain_grad()?;
        out[0].as_variable()?.backward()?;
        assert_eq!(a.grad()?.unwrap().data(), &[2.; 6]);
        assert_eq!(b.grad()?.unwrap().data(), &[1.; 6]);
        assert_eq!(w.grad()?.unwrap().data(), &[4., 8., 12., 16., 20., 24.]);
        assert_eq!(out[2].shape()?, vec![3, 2]);
        Ok(out[2].clone())
    })?;
    assert!(
        executor
            .with_run(&[], &[&w], |_| -> MlResult<()> {
                Err(MlError::StringError("callback failure".into()))
            })
            .is_err()
    );
    ctx.replace_parameter(w.variable(), TensorBuffer::from_vec(vec![2.; 6], &[2, 3])?)?;
    for _ in 0..100 {
        executor.with_run(&[], &[&w], |out| {
            let seed = ctx.tensor(vec![3.; 6], &[3, 2])?;
            out[2].as_variable()?.backward_with_grad(&seed)?;
            assert_eq!(w.grad()?.unwrap().data(), &[12.; 6]);
            assert_eq!(out[3].to_vec()?, vec![4.; 6]);
            Ok(())
        })?;
        assert_eq!(held.to_vec()?, vec![1., 4., 9., 16., 25., 36.]);
        assert!(w.grad()?.is_none());
    }
    Ok(())
}

#[test]
fn reshape_feed_views_preserve_values_until_late_consumers() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let mut p = PreparedProgram::new();
    let x = p.input(&[2, 3], false)?;
    let v = p.operation(Operation::Reshape(vec![6]), &[x])?;
    let neg = p.operation(Operation::Neg, &[x])?;
    let square = p.operation(Operation::Square, &[neg])?;
    let summed = p.operation(Operation::Sum, &[square])?;
    let mut e = ctx
        .prepare(&p, &[], &[v, summed], PreparedMode::Inference)?
        .into_executor(&ctx)?;
    for value in [2., 3.] {
        let x = ctx.tensor(vec![value; 6], &[2, 3])?;
        e.with_run(&[&x], &[], |out| {
            assert_eq!(out[0].to_vec()?, vec![value; 6]);
            assert_eq!(out[1].item()?, 6. * value * value);
            Ok(())
        })?;
    }
    Ok(())
}

#[test]
fn named_forward_is_shared_by_linear_model_and_checks_signature() -> MlResult<()> {
    use trench_deep::{nn::LinearRegression, trainer::TrainableModel};
    let ctx = ExecutionContext::new();
    let model = LinearRegression::new(&ctx, 2, 1)?;
    let inputs =
        ExecutionInputs::new("predict").with("input", ctx.tensor(vec![1.0, 2.0], &[1, 2])?)?;
    let plan = ctx.prepare_forward(&inputs, &model.parameters(), PreparedMode::Inference, |i| {
        Ok(vec![model.predict(i.get("input")?)?])
    })?;
    plan.buffer_plan().validate()?;
    for values in [vec![1.0, 2.0], vec![3.0, 4.0]] {
        let inputs = ExecutionInputs::new("predict").with("input", ctx.tensor(values, &[1, 2])?)?;
        let actual = plan.with_inputs(&ctx, &inputs, &model.parameters(), |out| out[0].to_vec())?;
        assert_eq!(actual, model.predict(inputs.get("input")?)?.to_vec()?);
    }
    let changed = ExecutionInputs::new("other").with("input", inputs.get("input")?.clone())?;
    assert!(
        plan.with_inputs(&ctx, &changed, &model.parameters(), |_| Ok(()))
            .is_err()
    );
    assert!(
        plan.with_run(&ctx, &[inputs.get("input")?], &model.parameters(), |_| Ok(
            ()
        ))
        .is_err()
    );
    let renamed = ExecutionInputs::new("predict").with("wrong", inputs.get("input")?.clone())?;
    assert!(
        plan.with_inputs(&ctx, &renamed, &model.parameters(), |_| Ok(()))
            .is_err()
    );
    Ok(())
}

#[test]
fn lifetimes_preserve_backward_inputs_aliases_and_exported_gradients() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let w = ctx.parameter(vec![2.0; 4], &[4])?;
    let mut p = PreparedProgram::new();
    let x = p.input(&[4], false)?;
    let a = p.parameter(&[4])?;
    let b = p.parameter(&[4])?;
    let product = p.operation(Operation::Mul, &[x, a])?;
    let sum = p.operation(Operation::Add, &[product, b])?;
    let activated = p.operation(Operation::Tanh, &[sum])?;
    let loss = p.operation(Operation::Sum, &[activated])?;
    let plan = ctx.prepare(&p, &[&w, &w], &[product, loss], PreparedMode::Training)?;
    let buffers = plan.buffer_plan();
    assert_eq!(buffers.aliases[1], buffers.aliases[2]);
    buffers.validate()?;
    assert_eq!(
        buffers
            .layout
            .lifetimes
            .iter()
            .filter(|l| l.role == BufferRole::Parameter)
            .count(),
        1
    );
    let input = buffers
        .layout
        .lifetimes
        .iter()
        .find(|l| l.value == BufferValue::Tensor(0))
        .unwrap();
    assert!(
        input.last > plan.node_count() + 1,
        "Mul needs original input in backward"
    );
    assert!(
        buffers
            .layout
            .lifetimes
            .iter()
            .any(|l| l.role == BufferRole::Saved && l.last > plan.node_count() + 1)
    );
    assert!(
        buffers
            .layout
            .copies
            .iter()
            .any(|c| c.reason == CopyReason::ExportedOutput)
    );
    // Root layouts must fit within the common lifetimes; root choice is late.
    for root in &buffers.roots {
        for life in &root.lifetimes {
            let common = buffers
                .layout
                .lifetimes
                .iter()
                .find(|l| l.value == life.value)
                .unwrap();
            assert!(common.first <= life.first && common.last >= life.last);
        }
    }
    let mut arena = buffers.allocate_arena()?;
    assert_eq!(arena.bytes(), buffers.capacity_bytes);
    for step in 0..100 {
        arena.buffer_mut(0)?.fill(step as f32);
    }
    assert_eq!(arena.buffer(0)?[0], 99.0);
    assert!(arena.buffer_mut(usize::MAX).is_err());
    assert!(!plan.uses_static_buffers());
    Ok(())
}

#[test]
fn out_of_place_reuse_never_overlaps_inputs_and_outputs() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let mut p = PreparedProgram::new();
    let x = p.input(&[8], false)?;
    let mut y = x;
    for _ in 0..10 {
        y = p.operation(Operation::Neg, &[y])?;
    }
    let plan = ctx.prepare(&p, &[], &[y], PreparedMode::Inference)?;
    let layout = &plan.buffer_plan().layout;
    assert!(layout.arena_bytes < layout.unreused_bytes);
    assert_eq!(layout.buffers.len(), 2);
    for pair in layout.lifetimes.windows(2) {
        assert_ne!(pair[0].buffer, pair[1].buffer);
    }
    let mut broken = plan.buffer_plan().clone();
    broken.layout.lifetimes[1].buffer = broken.layout.lifetimes[0].buffer;
    assert!(broken.validate().is_err());
    Ok(())
}

#[test]
fn common_preparation_trains_linear_with_changing_inputs_and_adam() -> MlResult<()> {
    use trench_deep::{
        nn::LinearRegression,
        optimizer::{Adam, Optimizer},
        trainer::TrainableModel,
    };
    let ctx = ExecutionContext::builder().initialization_seed(7).build();
    let eager = ExecutionContext::builder().initialization_seed(7).build();
    let mut model = LinearRegression::new(&ctx, 2, 1)?;
    let mut other = LinearRegression::new(&eager, 2, 1)?;
    let parameters = model.parameters().into_iter().cloned().collect::<Vec<_>>();
    let parameters = parameters.iter().collect::<Vec<_>>();
    let make = |ctx: &ExecutionContext, v| {
        ExecutionInputs::new("train")
            .with("image", ctx.tensor(vec![v, 0.5], &[1, 2])?)?
            .with("target", ctx.tensor(vec![1.0], &[1, 1])?)
    };
    let plan = ctx.prepare_forward(
        &make(&ctx, 0.0)?,
        &parameters,
        PreparedMode::Training,
        |inputs| {
            let (prediction, loss) =
                model.forward_loss(&inputs.get("image")?.as_variable()?, inputs.get("target")?)?;
            Ok(vec![prediction.tensor().clone(), loss.tensor().clone()])
        },
    )?;
    let mut adam = Adam::new(&ctx, 0.001, 0.9, 0.999, 1e-8)?;
    let mut eadam = Adam::new(&eager, 0.001, 0.9, 0.999, 1e-8)?;
    adam.register_all(&parameters)?;
    eadam.register_all(&other.parameters())?;
    for value in [0.2, 0.7, 1.5] {
        let result = plan.with_inputs(&ctx, &make(&ctx, value)?, &parameters, |out| {
            out[1].as_variable()?.backward()?;
            let result = (
                out[0].to_vec()?,
                out[1].item()?,
                parameters
                    .iter()
                    .map(|p| p.grad())
                    .collect::<MlResult<Vec<_>>>()?,
            );
            adam.step()?;
            Ok(result)
        })?;
        let inputs = make(&eager, value)?;
        let expected = eager.with_training_scope(|| {
            let (prediction, loss) =
                other.forward_loss(&inputs.get("image")?.as_variable()?, inputs.get("target")?)?;
            loss.backward()?;
            let result = (
                prediction.tensor().to_vec()?,
                loss.tensor().item()?,
                other
                    .parameters()
                    .iter()
                    .map(|p| p.grad())
                    .collect::<MlResult<Vec<_>>>()?,
            );
            eadam.step()?;
            Ok(result)
        })?;
        assert_eq!(result, expected);
        for (a, b) in parameters.iter().zip(other.parameters()) {
            assert_eq!(a.tensor().to_vec()?, b.tensor().to_vec()?);
        }
    }
    Ok(())
}
