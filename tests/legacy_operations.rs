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

// A weighted dot product exercises every output gradient independently of Sum.
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
fn division_and_sum_match_native_gradients() -> MlResult<()> {
    for backward in [false, true] {
        for shape in [vec![], vec![1], vec![2, 3], vec![1, 2, 3]] {
            let n = shape.iter().product::<usize>();
            let lhs = (0..n).map(|i| i as f32 - 2.0).collect::<Vec<_>>();
            let rhs = (0..n).map(|i| if i % 2 == 0 { -0.5 } else { 2.0 }).collect::<Vec<_>>();
            compare_operation(Operation::Div, vec![(lhs.clone(), shape.clone()), (rhs, shape.clone())], backward)?;
            compare_operation(Operation::Sum, vec![(lhs, shape)], backward)?;
        }
    }
    let ctx = ExecutionContext::builder().route(ExecutionRoute::Legacy).build()?;
    let x = ctx.parameter(vec![2.0, -3.0], &[2])?;
    let scalar = ctx.tensor(vec![2.0], &[])?;
    let baseline = ctx.graph_stats()?;
    for _ in 0..8 {
        ctx.with_training_scope(|| {
            // Shared numerator/denominator gradients must accumulate to zero.
            x.div(x.tensor())?.sum()?.backward()?;
            close(x.grad()?.unwrap().data(), &[0.0, 0.0]);
            Ok(())
        })?;
        assert_eq!(ctx.graph_stats()?, baseline);
    }
    assert!(matches!(x.div(&scalar), Err(MlError::UnsupportedCapability { .. })));
    assert_eq!(ctx.graph_stats()?, baseline);
    Ok(())
}

#[test]
fn sigmoid_matches_forward_backward_and_no_grad() -> MlResult<()> {
    for backward in [false, true] {
        for (data, shape) in [
            (vec![-100.0, -20.0, -2.0, -0.5, 0.0, 0.5, 2.0, 20.0, 100.0], vec![3, 3]),
            (vec![0.0], vec![]),
        ] {
            compare_operation(Operation::Sigmoid, vec![(data, shape)], backward)?;
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
fn abs_log_sqrt_match_weighted_gradients_and_no_grad() -> MlResult<()> {
    for operation in [Operation::Abs, Operation::Log, Operation::Sqrt] {
        let values = if matches!(operation, Operation::Abs) {
            vec![-4.0, -0.25, -0.0, 0.0, 0.25, 4.0]
        } else {
            vec![0.0001, 0.25, 1.0, 4.0, 9.0, 100.0]
        };
        for backward in [false, true] {
            for shape in [vec![6], vec![2, 3], vec![1, 2, 3]] {
                compare_operation(operation.clone(), vec![(values.clone(), shape)], backward)?;
            }
        }
        // Independent analytic expectations also cover scalar native backward.
        for &x in &values {
            let (expected, derivative) = match operation {
                Operation::Abs => (x.abs(), if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }),
                Operation::Log => (x.ln(), 1.0 / x),
                Operation::Sqrt => (x.sqrt(), 0.5 / x.sqrt()),
                _ => unreachable!(),
            };
            for route in [ExecutionRoute::P1, ExecutionRoute::Legacy] {
                let (y, gradient) = run(route, &operation, &[x], &[], true)?;
                close(&y, &[expected]);
                close(&gradient, &[derivative]);
            }
        }
    }
    Ok(())
}

#[test]
fn log_sqrt_preserve_domain_boundary_behavior() -> MlResult<()> {
    for operation in [Operation::Log, Operation::Sqrt] {
        for x in [0.0, -1.0] {
            let (a, da) = run(ExecutionRoute::P1, &operation, &[x], &[], true)?;
            let (b, db) = run(ExecutionRoute::Legacy, &operation, &[x], &[], true)?;
            // Existing forward contracts differ outside Log's real domain:
            // P1 clamps nonpositive inputs to -Inf; native calls f32::ln.
            if matches!(operation, Operation::Log) && x < 0.0 {
                assert_eq!(a, vec![f32::NEG_INFINITY]);
                assert!(b[0].is_nan());
            } else {
                assert!((a[0].is_nan() && b[0].is_nan()) || a == b);
            }
            for (left, right) in da.iter().zip(&db) {
                assert!((left.is_nan() && right.is_nan()) || left == right,
                    "{operation:?}({x}): {left} != {right}");
            }
        }
    }
    Ok(())
}

#[test]
fn subtraction_broadcast_gradients_and_invalid_shapes() -> MlResult<()> {
    for (a, b) in [
        (vec![], vec![2, 3]), (vec![2, 3], vec![]),
        (vec![2, 3], vec![3]), (vec![3], vec![2, 3]),
        (vec![2, 1, 3], vec![1, 4, 1]),
        (vec![1, 4, 1], vec![2, 1, 3]),
        (vec![2, 3, 2, 2], vec![1, 3, 1, 1]),
    ] {
        for backward in [false, true] {
            let lhs = (0..a.iter().product()).map(|i| i as f32 * 0.2 - 1.0).collect();
            let rhs = (0..b.iter().product()).map(|i| i as f32 * -0.3 + 0.7).collect();
            compare_operation(Operation::Sub, vec![(lhs, a.clone()), (rhs, b.clone())], backward)?;
        }
    }
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
    let matrix = ctx.parameter(vec![1.0; 6], &[2, 3])?;
    let bias = ctx.parameter(vec![0.5; 3], &[3])?;
    let bad = ctx.tensor(vec![1.0; 2], &[2])?;
    let empty = ctx.tensor(vec![], &[0, 3])?;
    let baseline = ctx.graph_stats()?;
    for _ in 0..4 {
        ctx.with_training_scope(|| {
            matrix.sub(bias.tensor())?.sum()?.backward()?;
            close(matrix.grad()?.unwrap().data(), &[1.0; 6]);
            close(bias.grad()?.unwrap().data(), &[-2.0; 3]);
            Ok(())
        })?;
        ctx.with_training_scope(|| {
            matrix.sub(matrix.tensor())?.sum()?.backward()?;
            close(matrix.grad()?.unwrap().data(), &[0.0; 6]);
            Ok(())
        })?;
        assert!(matrix.sub(&bad).is_err());
        assert!(x.sub(&empty).is_err());
        assert!(empty.as_variable()?.sub(bias.tensor()).is_err());
        assert_eq!(ctx.graph_stats()?, baseline);
    }
    assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    Ok(())
}

#[test]
fn arbitrary_transpose_permutations_match_gradients_and_cleanup() -> MlResult<()> {
    fn permutations(values: &mut [usize], start: usize, out: &mut Vec<Vec<usize>>) {
        if start == values.len() { out.push(values.to_vec()); return; }
        for i in start..values.len() {
            values.swap(start, i);
            permutations(values, start + 1, out);
            values.swap(start, i);
        }
    }
    for shape in [vec![], vec![3], vec![2, 3, 4], vec![2, 1, 3, 2]] {
        let mut axes = (0..shape.len()).collect::<Vec<_>>();
        let mut orders = Vec::new();
        permutations(&mut axes, 0, &mut orders);
        let data = (0..shape.iter().product()).map(|i| i as f32 * 0.17 - 1.0).collect::<Vec<_>>();
        for order in orders {
            for backward in [false, true] {
                compare_operation(Operation::Transpose(order.clone()), vec![(data.clone(), shape.clone())], backward)?;
            }
        }
    }
    for route in [ExecutionRoute::P1, ExecutionRoute::Legacy] {
        let ctx = ExecutionContext::builder().route(route).build()?;
        let x = ctx.parameter((0..24).map(|i| i as f32).collect(), &[2, 3, 4])?;
        let baseline = ctx.graph_stats()?;
        for _ in 0..4 {
            let output = ctx.with_training_scope(|| {
                let output = ctx.execute(&Operation::Transpose(vec![1, 2, 0]), &[x.tensor()])?.remove(0);
                output.as_variable()?.sum()?.backward()?;
                close(x.grad()?.unwrap().data(), &[1.0; 24]);
                Ok(output)
            })?;
            assert_eq!(output.shape()?, vec![3, 4, 2]);
            assert_eq!(output.to_vec()?.len(), 24);
            assert!(x.grad()?.is_none());
            drop(output);
            assert_eq!(ctx.graph_stats()?, baseline);
            for axes in [vec![1, 1, 0], vec![1, 2], vec![1, 2, 3]] {
                assert!(ctx.execute(&Operation::Transpose(axes), &[x.tensor()]).is_err());
                assert_eq!(ctx.graph_stats()?, baseline);
            }
        }
    }
    Ok(())
}

#[test]
fn single_input_concat_preserves_values_gradients_and_owned_output() -> MlResult<()> {
    for shape in [vec![3], vec![2, 3], vec![2, 1, 3]] {
        let values = (0..shape.iter().product()).map(|i| i as f32 * 0.3 - 0.7).collect::<Vec<_>>();
        for axis in 0..shape.len() {
            for backward in [false, true] {
                compare_operation(Operation::Concat { axis }, vec![(values.clone(), shape.clone())], backward)?;
            }
        }
    }
    for route in [ExecutionRoute::P1, ExecutionRoute::Legacy] {
        let ctx = ExecutionContext::builder().route(route).build()?;
        let x = ctx.parameter(vec![1.0, 2.0, 3.0], &[3])?;
        let baseline = ctx.graph_stats()?;
        for _ in 0..4 {
            let output = ctx.with_training_scope(|| {
                let output = ctx.execute(&Operation::Concat { axis: 0 }, &[x.tensor()])?.remove(0);
                // Concat produces its own handle; it must not shortcut to an
                // alias of the leaf and lose the native graph connection.
                assert_ne!(output.id(), x.tensor().id());
                output.as_variable()?.sum()?.backward()?;
                close(x.grad()?.unwrap().data(), &[1.0; 3]);
                Ok(output)
            })?;
            close(&output.to_vec()?, &[1.0, 2.0, 3.0]);
            assert!(x.grad()?.is_none());
            assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
            drop(output);
            assert_eq!(ctx.graph_stats()?, baseline);
        }
        assert!(ctx.execute(&Operation::Concat { axis: 0 }, &[]).is_err());
        assert!(ctx.execute(&Operation::Concat { axis: 1 }, &[x.tensor()]).is_err());
        assert_eq!(ctx.graph_stats()?, baseline);
    }
    Ok(())
}

#[test]
fn native_sub_assignment_and_backward_shape_contract() -> MlResult<()> {
    use trench_deep::legacy::{tensor::{Tensor as NativeTensor, TensorBase, operators::{Function, Sub}}, MlError as NativeError};
    let op = Sub::new().map_err(|e| MlError::StringError(e.to_string()))?;
    let lhs = NativeTensor::from_vec(vec![2.0, 4.0], &[2, 1]).unwrap();
    let rhs = NativeTensor::from_vec(vec![0.5, 1.0, 1.5], &[3]).unwrap();
    let out = op.assign_forward(&[&lhs, &rhs], lhs.id()).unwrap().remove(0);
    assert_eq!(out.id(), lhs.id());
    assert_eq!(out.shape(), &[2, 3]);
    close(out.data(), &[1.5, 1.0, 0.5, 3.5, 3.0, 2.5]);
    let wrong = NativeTensor::from_vec(vec![1.0], &[]).unwrap();
    assert!(matches!(op.backward(&[&out, &rhs], &wrong), Err(NativeError::TensorError(_))));
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
    assert!(x.transpose(&[1, 2, 3]).is_err());
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
