#![cfg(all(
    feature = "builtinStorage",
    feature = "builtinKernels",
    feature = "enableBackward"
))]
use trench_deep::{
    contracts::{LossKind, Operation},
    optimizer::{Adam, Optimizer},
    runtime::prepared::{PreparedMode, PreparedProgram},
    *,
};

fn program() -> MlResult<(PreparedProgram, Vec<runtime::prepared::TensorSlotId>)> {
    let mut p = PreparedProgram::new();
    let x = p.input(&[1, 2], false)?;
    let target = p.input(&[1, 1], false)?;
    let weight = p.parameter(&[2, 1])?;
    let bias = p.parameter(&[1])?;
    let dot = p.operation(Operation::Matmul, &[x, weight])?;
    let linear = p.operation(Operation::Add, &[dot, bias])?;
    let prediction = p.operation(Operation::Relu, &[linear])?;
    let loss = p.operation(
        Operation::Loss {
            kind: LossKind::Mse,
            reduction: Reduction::Mean,
        },
        &[prediction, target],
    )?;
    Ok((p, vec![prediction, loss]))
}
#[test]
fn linear_plan_reuses_topology_with_new_feeds_and_adam_values() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let eager = ExecutionContext::new();
    let w = ctx.parameter(vec![0.4, 0.7], &[2, 1])?;
    let b = ctx.parameter(vec![0.2], &[1])?;
    let ew = eager.parameter(vec![0.4, 0.7], &[2, 1])?;
    let eb = eager.parameter(vec![0.2], &[1])?;
    let mut adam = Adam::new(&ctx, 0.001, 0.9, 0.999, 1e-8)?;
    adam.register_all(&[&w, &b])?;
    let mut other_adam = Adam::new(&eager, 0.001, 0.9, 0.999, 1e-8)?;
    other_adam.register_all(&[&ew, &eb])?;
    let (program, outputs) = program()?;
    let baseline = ctx.graph_stats()?;
    let plan = ctx.prepare(&program, &[&w, &b], &outputs, PreparedMode::Training)?;
    assert_eq!(ctx.graph_stats()?, baseline);
    assert_eq!(plan.node_count(), 4);
    assert!(!plan.uses_static_buffers());
    let mut held = None;
    let mut held_values = Vec::new();
    for step in 0..10 {
        let values = vec![0.5 + step as f32 * 0.1, -0.1];
        let x = ctx.tensor(values.clone(), &[1, 2])?;
        let target = ctx.tensor(vec![0.3], &[1, 1])?;
        let ex = eager.tensor(values, &[1, 2])?;
        let et = eager.tensor(vec![0.3], &[1, 1])?;
        let actual = plan.with_run(&ctx, &[&x, &target], &[&w, &b], |out| {
            out[1].as_variable()?.backward()?;
            assert!(out[1].as_variable()?.backward().is_err());
            let result = (
                out[0].to_vec()?,
                out[1].to_vec()?,
                w.grad()?.unwrap(),
                b.grad()?.unwrap(),
            );
            adam.step()?;
            if held.is_none() {
                held = Some(out[0].clone());
                held_values = out[0].to_vec()?;
            }
            Ok(result)
        })?;
        let expected = eager.with_training_scope(|| {
            let dot = eager
                .execute(&Operation::Matmul, &[&ex, ew.tensor()])?
                .remove(0);
            let linear = eager
                .execute(&Operation::Add, &[&dot, eb.tensor()])?
                .remove(0);
            let pred = eager.execute(&Operation::Relu, &[&linear])?.remove(0);
            let loss = eager
                .execute(
                    &Operation::Loss {
                        kind: LossKind::Mse,
                        reduction: Reduction::Mean,
                    },
                    &[&pred, &et],
                )?
                .remove(0);
            loss.as_variable()?.backward()?;
            let result = (
                pred.to_vec()?,
                loss.to_vec()?,
                ew.grad()?.unwrap(),
                eb.grad()?.unwrap(),
            );
            other_adam.step()?;
            Ok(result)
        })?;
        assert_eq!(actual, expected);
        assert_eq!(w.tensor().to_vec()?, ew.tensor().to_vec()?);
        assert_eq!(b.tensor().to_vec()?, eb.tensor().to_vec()?);
        assert_eq!(held.as_ref().unwrap().to_vec()?, held_values);
        assert!(w.grad()?.is_none());
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    }
    drop(held);
    assert_eq!(ctx.graph_stats()?, baseline);
    Ok(())
}
#[test]
fn binding_errors_are_atomic_and_callback_errors_clean_up() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let w = ctx.parameter(vec![0.4, 0.7], &[2, 1])?;
    let b = ctx.parameter(vec![0.2], &[1])?;
    let other = ctx.parameter(vec![0.4, 0.7], &[2, 1])?;
    let x = ctx.tensor(vec![1.0, 2.0], &[1, 2])?;
    let bad = ctx.tensor(vec![1.0; 4], &[2, 2])?;
    let tracked = ctx.parameter(vec![1.0, 2.0], &[1, 2])?;
    let target = ctx.tensor(vec![0.0], &[1, 1])?;
    let (p, out) = program()?;
    let plan = ctx.prepare(&p, &[&w, &b], &out, PreparedMode::Training)?;
    let baseline = ctx.graph_stats()?;
    assert!(
        plan.with_run(&ctx, &[&bad, &target], &[&w, &b], |_| Ok(()))
            .is_err()
    );
    assert!(
        plan.with_run(&ctx, &[tracked.tensor(), &target], &[&w, &b], |_| Ok(()))
            .is_err()
    );
    assert!(
        plan.with_run(&ctx, &[&x, &target], &[&other, &b], |_| Ok(()))
            .is_err()
    );
    assert_eq!(ctx.graph_stats()?, baseline);
    let failure: MlResult<()> = plan.with_run(&ctx, &[&x, &target], &[&w, &b], |out| {
        out[1].as_variable()?.backward()?;
        assert!(
            plan.with_run(&ctx, &[&x, &target], &[&w, &b], |_| Ok(()))
                .is_err()
        );
        Err(ContextError::BorrowConflict.into())
    });
    assert!(failure.is_err());
    assert_eq!(ctx.graph_stats()?, baseline);
    assert!(w.grad()?.is_none());
    plan.with_run(&ctx, &[&x, &target], &[&w, &b], |out| {
        out[1].as_variable()?.backward()
    })?;
    assert_eq!(ctx.graph_stats()?, baseline);
    let inference = ctx.prepare(&p, &[&w, &b], &out, PreparedMode::Inference)?;
    inference.with_run(&ctx, &[&x, &target], &[&w, &b], |out| {
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
        assert!(!out[1].as_variable()?.requires_grad()?);
        Ok(())
    })?;
    assert_eq!(ctx.graph_stats()?, baseline);
    Ok(())
}
#[test]
fn preparation_rejects_invalid_shapes_and_does_not_consume_rng() -> MlResult<()> {
    let ctx = ExecutionContext::builder()
        .initialization_seed(7)
        .model_seed(9)
        .build();
    let reference = ExecutionContext::builder()
        .initialization_seed(7)
        .model_seed(9)
        .build();
    let w = ctx.parameter(vec![1.0; 2], &[2, 1])?;
    let b = ctx.parameter(vec![0.0], &[1])?;
    let (mut p, outputs) = program()?;
    let baseline = ctx.graph_stats()?;
    assert!(p.input(&[usize::MAX, 2], false).is_err());
    assert!(p.input(&[0, 2], false).is_err());
    let foreign = PreparedProgram::new().input(&[1, 1], false)?;
    assert!(p.operation(Operation::Square, &[foreign]).is_err());
    assert!(
        p.operation(Operation::Transpose(vec![0, 0]), &[outputs[0]])
            .is_err()
    );
    assert!(
        p.operation(Operation::Reshape(vec![3]), &[outputs[0]])
            .is_err()
    );
    assert!(p.operation(Operation::Matmul, &[outputs[0]]).is_err());
    assert!(
        ctx.prepare(&p, &[&w], &outputs, PreparedMode::Training)
            .is_err()
    );
    assert!(
        ctx.prepare(&p, &[&w, &b], &[], PreparedMode::Training)
            .is_err()
    );
    let plan = ctx.prepare(&p, &[&w, &b], &outputs, PreparedMode::Training)?;
    assert_eq!(plan.node_count(), 4);
    assert_eq!(ctx.graph_stats()?, baseline);
    assert_eq!(
        ctx.initialization_uniform(20, 1.0)?,
        reference.initialization_uniform(20, 1.0)?
    );
    assert_eq!(
        ctx.model_uniform(20, 1.0)?,
        reference.model_uniform(20, 1.0)?
    );
    assert_eq!(w.tensor().to_vec()?, vec![1.0; 2]);
    Ok(())
}

#[test]
fn shared_parameter_binding_and_nonunit_seed_are_preserved() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let w = ctx.parameter(vec![2.0], &[])?;
    let other = ctx.parameter(vec![2.0], &[])?;
    let x = ctx.tensor(vec![3.0], &[])?;
    let seed = ctx.tensor(vec![2.5], &[])?;
    let mut p = PreparedProgram::new();
    let a = p.parameter(&[])?;
    let b = p.parameter(&[])?;
    let feed = p.input(&[], false)?;
    let sum = p.operation(Operation::Add, &[a, b])?;
    let product = p.operation(Operation::Mul, &[sum, feed])?;
    let plan = ctx.prepare(&p, &[&w, &w], &[product], PreparedMode::Training)?;
    assert!(
        plan.with_run(&ctx, &[&x], &[&w, &other], |_| Ok(()))
            .is_err()
    );
    plan.with_run(&ctx, &[&x], &[&w, &w], |out| {
        assert_eq!(out[0].item()?, 12.0);
        assert!(ctx.execute(&Operation::Mul, &[&out[0], &seed]).is_err());
        out[0].as_variable()?.backward_with_grad(&seed)?;
        assert_eq!(w.grad()?.unwrap().data(), &[15.0]);
        Ok(())
    })?;
    assert!(w.grad()?.is_none());
    Ok(())
}

#[cfg(feature = "legacyBenchmark")]
#[test]
fn legacy_route_does_not_accept_prepared_policy() -> MlResult<()> {
    let ctx = ExecutionContext::builder()
        .route(ExecutionRoute::Legacy)
        .build()?;
    let mut p = PreparedProgram::new();
    let x = p.input(&[], false)?;
    let y = p.operation(Operation::Square, &[x])?;
    assert!(ctx.prepare(&p, &[], &[y], PreparedMode::Inference).is_err());
    Ok(())
}

#[derive(Debug)]
struct NoDynamicGraph;
impl contracts::AutogradEngine for NoDynamicGraph {
    fn record(&mut self, _: contracts::GradientRecord) -> MlResult<()> {
        panic!("dynamic record")
    }
    fn get(&self, _: TensorId) -> Option<contracts::GradientRecord> {
        panic!("dynamic get")
    }
    fn remove(&mut self, _: TensorId) -> MlResult<Option<contracts::GradientRecord>> {
        panic!("dynamic remove")
    }
    fn nodes(&self) -> Vec<TensorId> {
        Vec::new()
    }
    fn order(&self, _: TensorId) -> MlResult<Vec<TensorId>> {
        panic!("dynamic order")
    }
}

#[test]
fn prepared_backward_never_registers_or_traverses_dynamic_graph() -> MlResult<()> {
    let ctx = ExecutionContext::builder().autograd(NoDynamicGraph).build();
    let w = ctx.parameter(vec![2.0], &[])?;
    let mut p = PreparedProgram::new();
    let leaf = p.parameter(&[])?;
    let square = p.operation(Operation::Square, &[leaf])?;
    let sum = p.operation(Operation::Add, &[square, square])?;
    let plan = ctx.prepare(&p, &[&w], &[square, sum], PreparedMode::Training)?;
    assert_eq!(plan.backward_plan_stats().maximum_fan_in, 2);
    let seed = ctx.tensor(vec![2.5], &[])?;
    let wrong = ctx.tensor(vec![1.0, 1.0], &[2])?;
    for _ in 0..3 {
        plan.with_run(&ctx, &[], &[&w], |out| {
            assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
            assert!(
                ctx.replace_parameter(w.variable(), TensorBuffer::from_vec(vec![9.0], &[])?)
                    .is_err()
            );
            let retained = out[0].as_variable()?;
            retained.retain_grad()?;
            let root = out[1].as_variable()?;
            assert!(root.backward_with_grad(&wrong).is_err());
            root.backward_with_grad(&seed)?;
            assert_eq!(w.grad()?.unwrap().data(), &[20.0]);
            assert_eq!(retained.grad()?.unwrap().data(), &[5.0]);
            assert!(root.backward().is_err());
            assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
            Ok(())
        })?;
        assert!(w.grad()?.is_none());
    }

    Ok(())
}

#[derive(Debug)]
struct ForwardOnly;
impl contracts::OperationProvider for ForwardOnly {
    fn supports_prepared_replay(&self) -> bool {
        true
    }
    fn execute(
        &self,
        op: &Operation,
        inputs: &[TensorView<'_>],
    ) -> MlResult<contracts::OperationOutput> {
        contracts::OperationProvider::execute(&backend::CpuBackend::default(), op, inputs)
    }
}
#[test]
fn training_requires_explicit_backward_capability() -> MlResult<()> {
    let ctx = ExecutionContext::builder().operations(ForwardOnly).build();
    let w = ctx.parameter(vec![2.0], &[])?;
    let mut p = PreparedProgram::new();
    let a = p.parameter(&[])?;
    let b = p.operation(Operation::Square, &[a])?;
    assert!(
        ctx.prepare(&p, &[&w], &[b], PreparedMode::Training)
            .is_err()
    );
    let plan = ctx.prepare(&p, &[&w], &[b], PreparedMode::Inference)?;
    plan.with_run(&ctx, &[], &[&w], |out| {
        assert_eq!(out[0].item()?, 4.0);
        assert!(!out[0].as_variable()?.requires_grad()?);
        Ok(())
    })
}

#[test]
fn prepared_panic_cleanup_allows_next_run() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let w = ctx.parameter(vec![2.0], &[])?;
    let mut p = PreparedProgram::new();
    let a = p.parameter(&[])?;
    let b = p.operation(Operation::Square, &[a])?;
    let plan = ctx.prepare(&p, &[&w], &[b], PreparedMode::Training)?;
    let mut held = None;
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: MlResult<()> = plan.with_run(&ctx, &[], &[&w], |out| {
                held = Some(out[0].clone());
                panic!("callback panic");
            });
        }))
        .is_err()
    );
    assert!(held.unwrap().as_variable()?.backward().is_err());
    assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    plan.with_run(&ctx, &[], &[&w], |out| {
        out[0].as_variable()?.backward()?;
        assert_eq!(w.grad()?.unwrap().data(), &[4.0]);
        Ok(())
    })
}
