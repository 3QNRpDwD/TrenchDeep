#![cfg(all(
    feature = "legacyBenchmark",
    feature = "builtinStorage",
    feature = "builtinKernels"
))]
use trench_deep::{contracts::Operation, *};

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-5 * (1.0 + b.abs()),
            "{a} != {b}"
        );
    }
}

// A weighted dot product exercises every output gradient without depending on
// the legacy Sum backward, which has a different scalar broadcast contract.
fn compare_operation(
    op: Operation,
    inputs: Vec<(Vec<f32>, Vec<usize>)>,
    backward: bool,
) -> MlResult<()> {
    let run = |route| -> MlResult<_> {
        let ctx = ExecutionContext::builder().route(route).build()?;
        let parameters = inputs
            .iter()
            .map(|(data, shape)| ctx.parameter(data.clone(), shape))
            .collect::<MlResult<Vec<_>>>()?;
        let baseline = ctx.graph_stats()?;
        let result = ctx.with_training_scope(|| {
            let refs = parameters.iter().map(|p| p.tensor()).collect::<Vec<_>>();
            let compute = || ctx.execute(&op, &refs);
            let outputs = if backward {
                compute()?
            } else {
                ctx.no_grad(compute)?
            };
            let values = outputs
                .iter()
                .map(|t| t.snapshot())
                .collect::<MlResult<Vec<_>>>()?;
            if backward {
                let n = values[0].data().len();
                let flat = outputs[0].as_variable()?.reshape(&[n])?;
                let weights = ctx.tensor((0..n).map(|i| 0.5 + i as f32 * 0.07).collect(), &[n])?;
                flat.matmul(&weights)?.backward()?;
            }
            let gradients = parameters
                .iter()
                .map(|p| p.grad())
                .collect::<MlResult<Vec<_>>>()?;
            Ok((values, gradients))
        })?;
        assert_eq!(ctx.graph_stats()?, baseline);
        Ok(result)
    };
    let p1 = run(ExecutionRoute::P1)?;
    let legacy = run(ExecutionRoute::Legacy)?;
    assert_eq!(p1.0.len(), legacy.0.len(), "{op:?}");
    for (a, b) in p1.0.iter().zip(&legacy.0) {
        assert_eq!(a.shape(), b.shape(), "{op:?}");
        close(a.data(), b.data());
    }
    for (a, b) in p1.1.iter().zip(&legacy.1) {
        assert_eq!(a.is_some(), b.is_some(), "{op:?}");
        if let (Some(a), Some(b)) = (a, b) {
            assert_eq!(a.shape(), b.shape());
            close(a.data(), b.data());
        }
    }
    Ok(())
}

#[test]
fn recomputed_activation_probabilities_match_backward() -> MlResult<()> {
    compare_operation(Operation::Tanh, vec![(vec![-2.0,-0.5,0.0,0.5,1.0,2.0], vec![2,3])], true)?;
    for axis in 0..3 {
        compare_operation(Operation::Softmax { axis }, vec![((0..12).map(|i| 1000.0 + (i as f32 - 6.0) * 0.3).collect(), vec![2,3,2])], true)?;
    }
    compare_operation(Operation::Softmax { axis: 0 }, vec![(vec![0.0,0.0],vec![2])], true)?;
    Ok(())
}

#[test]
fn spatial_and_saved_tensor_operations_match_gradients() -> MlResult<()> {
    let image = (
        (0..32).map(|i| (i as f32 - 12.0) * 0.1).collect::<Vec<_>>(),
        vec![1, 2, 4, 4],
    );
    for op in [
        Operation::MaxPool2d {
            kernel: (2, 2),
            stride: (1, 2),
        },
        Operation::AvgPool2d {
            kernel: (2, 3),
            stride: (2, 1),
        },
        Operation::NearestUpsample2d { scale: (2, 3) },
    ] {
        for backward in [false, true] {
            compare_operation(op.clone(), vec![image.clone()], backward)?;
        }
    }
    for backward in [false, true] {
        compare_operation(
            Operation::GroupNorm {
                groups: 2,
                epsilon: 1e-5,
            },
            vec![
                image.clone(),
                (vec![0.7, 1.2], vec![2]),
                (vec![0.1, -0.2], vec![2]),
            ],
            backward,
        )?;
        compare_operation(
            Operation::Conv2d {
                stride: (2, 1),
                padding: (1, 0),
            },
            vec![
                image.clone(),
                (
                    (0..24).map(|i| (i as f32 - 10.0) * 0.03).collect(),
                    vec![3, 2, 2, 2],
                ),
                (vec![0.1, -0.2, 0.3], vec![3]),
            ],
            backward,
        )?;
    }
    Ok(())
}

#[test]
fn remaining_forwarding_categories_and_loss_targets() -> MlResult<()> {
    use trench_deep::contracts::{LossKind, Reduction};
    for op in [
        Operation::Pow(2.5),
        Operation::ApproxSin { threshold: 0.0001 },
        Operation::ApproxCos { threshold: 0.0001 },
    ] {
        compare_operation(op, vec![(vec![0.2, 0.7, 1.4], vec![3])], true)?;
    }
    compare_operation(
        Operation::Concat { axis: 1 },
        vec![
            (vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            (vec![5.0, 6.0], vec![2, 1]),
        ],
        true,
    )?;
    compare_operation(
        Operation::Matmul,
        vec![
            (vec![0.2; 12], vec![2, 2, 3]),
            (vec![0.7; 12], vec![2, 3, 2]),
        ],
        true,
    )?;
    for op in [
        Operation::Tanh,
        Operation::Softmax { axis: 1 },
        Operation::Sum,
        Operation::TopK { k: 2, sorted: true },
        Operation::Matmax {
            axis: Some(-1),
            keepdim: true,
        },
    ] {
        compare_operation(
            op,
            vec![(vec![0.2, 0.7, 1.4, -0.3, 1.1, 0.4], vec![2, 3])],
            false,
        )?;
    }
    compare_operation(
        Operation::Div,
        vec![(vec![1.0, 2.0], vec![2]), (vec![2.0, 4.0], vec![2])],
        false,
    )?;
    for kind in [
        LossKind::Mse,
        LossKind::Mae,
        LossKind::Huber { delta: 1.0 },
        LossKind::BinaryCrossEntropy,
        LossKind::CrossEntropy,
        LossKind::SoftmaxCrossEntropy,
    ] {
        compare_operation(
            Operation::Loss {
                kind,
                reduction: Reduction::Mean,
            },
            vec![
                (vec![0.2, 0.8, 0.6, 0.4], vec![2, 2]),
                (vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]),
            ],
            true,
        )?;
    }
    Ok(())
}

#[test]
fn expanded_validation_rejects_before_recording_and_shared_loss_target_is_constant() -> MlResult<()>
{
    use trench_deep::contracts::{LossKind, Reduction};
    let ctx = ExecutionContext::builder()
        .route(ExecutionRoute::Legacy)
        .build()?;
    let x = ctx.parameter(vec![0.2; 8], &[1, 2, 2, 2])?;
    let channel = ctx.parameter(vec![1.0; 2], &[2])?;
    let baseline = ctx.graph_stats()?;
    for op in [
        Operation::MaxPool2d {
            kernel: (3, 3),
            stride: (1, 1),
        },
        Operation::AvgPool2d {
            kernel: (1, 1),
            stride: (0, 1),
        },
        Operation::NearestUpsample2d { scale: (0, 1) },
        Operation::TopK { k: 0, sorted: true },
        Operation::Softmax { axis: 4 },
        Operation::ApproxSin {
            threshold: f32::NAN,
        },
        Operation::Sigmoid,
        Operation::Sum,
        Operation::Matmax {
            axis: Some(1),
            keepdim: false,
        },
    ] {
        assert!(ctx.execute(&op, &[x.tensor()]).is_err(), "{op:?}");
        assert_eq!(ctx.graph_stats()?, baseline);
    }
    assert!(
        ctx.execute(
            &Operation::GroupNorm {
                groups: 0,
                epsilon: 1e-5
            },
            &[x.tensor(), channel.tensor(), channel.tensor()]
        )
        .is_err()
    );
    assert_eq!(ctx.graph_stats()?, baseline);
    ctx.with_training_scope(|| {
        // A shared target must not add the target derivative to the prediction
        // derivative. Cross entropy has a nonzero derivative at equality.
        let loss = ctx
            .execute(
                &Operation::Loss {
                    kind: LossKind::CrossEntropy,
                    reduction: Reduction::Mean,
                },
                &[channel.tensor(), channel.tensor()],
            )?
            .remove(0);
        loss.as_variable()?.backward()?;
        close(channel.grad()?.unwrap().data(), &[-1.0, -1.0]);
        Ok(())
    })?;
    assert_eq!(ctx.graph_stats()?, baseline);
    Ok(())
}
fn run(
    route: ExecutionRoute,
    op: &Operation,
    values: &[f32],
    shape: &[usize],
    backward: bool,
) -> MlResult<(Vec<f32>, Vec<f32>)> {
    let ctx = ExecutionContext::builder().route(route).build()?;
    let x = ctx.parameter(values.to_vec(), shape)?;
    ctx.with_training_scope(|| {
        let compute = || -> MlResult<Tensor> { Ok(ctx.execute(op, &[x.tensor()])?.remove(0)) };
        let result = if backward {
            compute()?
        } else {
            ctx.no_grad(compute)?
        };
        let output = result.to_vec()?;
        let gradient = if backward {
            result.as_variable()?.backward()?;
            x.grad()?.unwrap().data().to_vec()
        } else {
            assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
            assert!(x.grad()?.is_none());
            Vec::new()
        };
        Ok((output, gradient))
    })
}

#[test]
fn small_unary_categories_match_original_forward_and_backward() -> MlResult<()> {
    for operation in [
        Operation::Neg,
        Operation::Square,
        Operation::Exp,
        Operation::Sin,
        Operation::Cos,
        Operation::Relu,
        Operation::Silu,
    ] {
        // Legacy scalar backward seeds its original graph. Each element is checked
        // independently, avoiding any dependence on the not-yet-supported Sum.
        for value in [-2.0, -0.3, 0.0, 0.7, 2.0] {
            let (a, da) = run(ExecutionRoute::P1, &operation, &[value], &[], true)?;
            let (b, db) = run(ExecutionRoute::Legacy, &operation, &[value], &[], true)?;
            close(&a, &b);
            close(&da, &db);
        }
        let values = [-2.0, -0.3, 0.0, 0.7, 2.0, 1.0];
        let (a, _) = run(ExecutionRoute::P1, &operation, &values, &[2, 3], false)?;
        let (b, _) = run(ExecutionRoute::Legacy, &operation, &values, &[2, 3], false)?;
        close(&a, &b);
    }
    Ok(())
}

#[test]
fn inference_only_operators_reject_tracking_before_creating_graphs() -> MlResult<()> {
    for operation in [Operation::Abs, Operation::Log, Operation::Sqrt] {
        let (a, _) = run(
            ExecutionRoute::P1,
            &operation,
            &[0.25, 1.0, 4.0],
            &[3],
            false,
        )?;
        let (b, _) = run(
            ExecutionRoute::Legacy,
            &operation,
            &[0.25, 1.0, 4.0],
            &[3],
            false,
        )?;
        close(&a, &b);
        let ctx = ExecutionContext::builder()
            .route(ExecutionRoute::Legacy)
            .build()?;
        let x = ctx.parameter(vec![1.0], &[])?;
        let before = ctx.graph_stats()?;
        assert!(matches!(
            ctx.execute(&operation, &[x.tensor()]),
            Err(MlError::UnsupportedCapability { .. })
        ));
        assert_eq!(ctx.graph_stats()?, before);
    }
    Ok(())
}

#[test]
fn subtraction_gradients_and_rejected_broadcast() -> MlResult<()> {
    let ctx = ExecutionContext::builder()
        .route(ExecutionRoute::Legacy)
        .build()?;
    let x = ctx.parameter(vec![3.0], &[])?;
    let y = ctx.parameter(vec![2.0], &[])?;
    ctx.with_training_scope(|| {
        let loss = x.sub(y.tensor())?;
        assert_eq!(loss.tensor().item()?, 1.0);
        loss.backward()?;
        assert_eq!(x.grad()?.unwrap().data(), &[1.0]);
        assert_eq!(y.grad()?.unwrap().data(), &[-1.0]);
        Ok(())
    })?;
    let vector = ctx.tensor(vec![1.0, 2.0], &[2])?;
    assert!(matches!(
        x.sub(&vector),
        Err(MlError::UnsupportedCapability { .. })
    ));
    assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    Ok(())
}

#[test]
fn shape_and_matrix_categories_match_both_routes() -> MlResult<()> {
    fn run(
        route: ExecutionRoute,
        a_shape: &[usize],
        b_shape: &[usize],
        transpose: bool,
    ) -> MlResult<(Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let ctx = ExecutionContext::builder().route(route).build()?;
        let a = ctx.parameter(
            (0..a_shape.iter().product())
                .map(|i| 0.1 + i as f32 * 0.2)
                .collect(),
            a_shape,
        )?;
        let b = ctx.parameter(
            (0..b_shape.iter().product())
                .map(|i| -0.3 + i as f32 * 0.1)
                .collect(),
            b_shape,
        )?;
        let baseline = ctx.graph_stats()?.tensors;
        let result = ctx.with_training_scope(|| {
            let input = if transpose {
                a.transpose(&[1, 0])?
            } else {
                a.variable().clone()
            };
            let output = input.matmul(b.tensor())?;
            let shape = output.tensor().shape()?;
            let values = output.tensor().to_vec()?;
            let flat = output.reshape(&[values.len()])?;
            let weights = ctx.tensor(
                (0..values.len()).map(|i| 1.0 + i as f32 * 0.25).collect(),
                &[values.len()],
            )?;
            let loss = flat.matmul(&weights)?;
            loss.backward()?;
            Ok((
                shape,
                values,
                a.grad()?.unwrap().data().to_vec(),
                b.grad()?.unwrap().data().to_vec(),
            ))
        })?;
        assert_eq!(ctx.graph_stats()?.tensors, baseline);
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
        Ok(result)
    }
    for (a, b, transpose) in [
        (vec![3], vec![3], false),
        (vec![2, 3], vec![3], false),
        (vec![3], vec![3, 2], false),
        (vec![2, 3], vec![3, 2], false),
        (vec![3, 2], vec![3, 2], true),
    ] {
        let p1 = run(ExecutionRoute::P1, &a, &b, transpose)?;
        let legacy = run(ExecutionRoute::Legacy, &a, &b, transpose)?;
        assert_eq!(p1.0, legacy.0);
        close(&p1.1, &legacy.1);
        close(&p1.2, &legacy.2);
        close(&p1.3, &legacy.3);
    }
    Ok(())
}

#[test]
fn shape_validation_and_no_grad_leave_no_native_graph() -> MlResult<()> {
    let ctx = ExecutionContext::builder()
        .route(ExecutionRoute::Legacy)
        .build()?;
    let x = ctx.parameter((0..24).map(|i| i as f32).collect(), &[2, 3, 4])?;
    let before = ctx.graph_stats()?;
    assert!(x.reshape(&[5]).is_err());
    assert!(x.transpose(&[0, 0, 2]).is_err());
    assert!(matches!(
        x.transpose(&[1, 2, 0]),
        Err(MlError::UnsupportedCapability { .. })
    ));
    assert!(x.matmul(x.tensor()).is_err());
    assert_eq!(ctx.graph_stats()?, before);
    ctx.no_grad(|| {
        let transposed = x.transpose(&[0, 2, 1])?;
        assert_eq!(transposed.tensor().shape()?, vec![2, 4, 3]);
        let restored = transposed.transpose(&[0, 2, 1])?;
        assert_eq!(restored.tensor().to_vec()?, x.tensor().to_vec()?);
        let identity = x.transpose(&[0, 1, 2])?;
        assert_eq!(identity.tensor().to_vec()?, x.tensor().to_vec()?);
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
        Ok(())
    })?;
    assert_eq!(ctx.graph_stats()?, before);
    Ok(())
}
