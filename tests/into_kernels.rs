#![cfg(all(
    feature = "builtinStorage",
    feature = "builtinKernels",
    feature = "enableBackward"
))]
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
use trench_deep::{contracts::*, runtime::prepared::*, *};
struct Counter;
thread_local! {static COUNT:Cell<Option<usize>>=const {Cell::new(None)};}
fn tick() {
    let _ = COUNT.try_with(|c| {
        if let Some(n) = c.get() {
            c.set(Some(n + 1));
        }
    });
}
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        tick();
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        tick();
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        tick();
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOC: Counter = Counter;
fn no_alloc(f: impl FnOnce() -> MlResult<()>) -> MlResult<()> {
    COUNT.with(|c| c.set(Some(0)));
    let result = f();
    let allocations = COUNT.with(|c| c.replace(None).unwrap());
    assert_eq!(allocations, 0, "kernel allocated during execution");
    result
}

#[test]
fn prepared_metadata_allocations_do_not_grow_with_node_count() -> MlResult<()> {
    let mut counts = Vec::new();
    for depth in [2, 128] {
        let ctx = ExecutionContext::new();
        let w = ctx.parameter(vec![2.; 4], &[4])?;
        let mut p = PreparedProgram::new();
        let mut value = p.parameter(&[4])?;
        for _ in 0..depth {
            value = p.operation(Operation::Neg, &[value])?;
        }
        let loss = p.operation(Operation::Sum, &[value])?;
        let mut e = ctx
            .prepare(&p, &[&w], &[loss], PreparedMode::Training)?
            .into_executor(&ctx)?;
        for iteration in 0..3 {
            COUNT.with(|c| c.set(Some(0)));
            let result = e.with_run(&[], &[&w], |out| out[0].as_variable()?.backward());
            let count = COUNT.with(|c| c.replace(None).unwrap());
            result?;
            if iteration == 2 {
                counts.push(count);
            }
        }
    }
    assert_eq!(
        counts[0], counts[1],
        "per-node metadata allocated: {counts:?}"
    );
    Ok(())
}

#[test]
fn metadata_supports_large_arity_and_repeated_shared_destinations() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let w = ctx.parameter(vec![2., 3.], &[2])?;
    let mut p = PreparedProgram::new();
    let x = p.parameter(&[2])?;
    let y = p.operation(Operation::Concat { axis: 0 }, &vec![x; 129])?;
    let loss = p.operation(Operation::Sum, &[y])?;
    let mut e = ctx
        .prepare(&p, &[&w], &[loss], PreparedMode::Training)?
        .into_executor(&ctx)?;
    for _ in 0..3 {
        e.with_run(&[], &[&w], |out| {
            assert_eq!(out[0].item()?, 645.);
            out[0].as_variable()?.backward()?;
            assert_eq!(w.grad()?.unwrap().data(), &[129., 129.]);
            Ok(())
        })?;
    }
    Ok(())
}

#[test]
fn builtin_optimizer_steps_borrow_gradients_and_update_without_allocating() -> MlResult<()> {
    use trench_deep::optimizer::*;
    for kind in 0..6 {
        let ctx = ExecutionContext::new();
        let w = ctx.parameter(vec![0.5, -1.0, 2.0], &[3])?;
        let alias = w.variable().detach()?;
        let mut optimizer: Box<dyn Optimizer> = match kind {
            0 => Box::new(SGD::new(&ctx, 0.01)?),
            1 => Box::new(Momentum::new(&ctx, 0.01, 0.9)?),
            2 => Box::new(AdaGrad::new(&ctx, 0.01, 1e-8)?),
            3 => Box::new(RMSProp::new(&ctx, 0.01, 0.9, 1e-8)?),
            4 => Box::new(Adam::new(&ctx, 0.01, 0.9, 0.999, 1e-8)?),
            _ => Box::new(AdamW::new(&ctx, 0.01, 0.9, 0.999, 1e-8, 0.1)?),
        };
        optimizer.register(&w)?;
        for _ in 0..5 {
            ctx.with_training_scope(|| {
                w.variable().square()?.sum()?.backward()?;
                let snapshot = w.grad()?.unwrap();
                no_alloc(|| optimizer.step())?;
                assert_eq!(w.grad()?.unwrap(), snapshot);
                assert_eq!(alias.tensor().to_vec()?, w.tensor().to_vec()?);
                Ok(())
            })?;
        }
        no_alloc(|| optimizer.step())?; // Missing gradient is a no-op for weights.
    }
    Ok(())
}
fn equal(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (&x, &y) in a.iter().zip(b) {
        assert!(
            x == y
                || (x.is_nan() && y.is_nan())
                || (x - y).abs() <= 1e-6 * (1.0 + x.abs().max(y.abs())),
            "{x} != {y}"
        );
    }
}

#[test]
fn convolution_rows_and_tiles_match_independent_scalar_order() -> MlResult<()> {
    let provider = backend::CpuBackend::default();
    let same = |a: &[f32], b: &[f32]| {
        assert_eq!(a.len(), b.len());
        for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "element {i}: {a:?} != {b:?}"
            );
        }
    };
    for (xs, ws, stride, padding) in [
        ([2, 3, 5, 7], [4, 3, 3, 2], (1, 1), (1, 2)),
        ([2, 3, 7, 8], [4, 3, 2, 3], (2, 3), (2, 1)),
        ([1, 2, 2, 2], [3, 2, 1, 1], (1, 2), (3, 1)),
        ([1, 2, 1, 1], [3, 2, 3, 3], (1, 1), (4, 4)),
        ([0, 2, 4, 5], [3, 2, 3, 3], (1, 1), (1, 1)),
        ([1, 0, 3, 4], [2, 0, 3, 3], (1, 1), (1, 1)),
        ([1, 2, 0, 3], [3, 2, 3, 3], (1, 1), (2, 1)),
        ([1, 2, 3, 3], [0, 2, 1, 1], (1, 1), (0, 0)),
        ([1, 1, 2, 3], [2, 1, 0, 2], (1, 1), (1, 1)),
    ] {
        let [n, ci, h, w] = xs;
        let [co, _, kh, kw] = ws;
        let oh = (h + 2 * padding.0 - kh) / stride.0 + 1;
        let ow = (w + 2 * padding.1 - kw) / stride.1 + 1;
        let shape = [n, co, oh, ow];
        let bs = [co];
        for exceptional in [false, true] {
            let make = |len, shift| {
                (0..len)
                    .map(|i| {
                        if exceptional {
                            [
                                f32::INFINITY,
                                f32::NEG_INFINITY,
                                f32::NAN,
                                -0.0,
                                0.0,
                                0.75,
                                -2.0,
                            ][(i + shift) % 7]
                        } else {
                            ((i + shift) as f32 * 0.17).sin()
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let x = make(n * ci * h * w, 0);
            let weight = make(co * ci * kh * kw, 3);
            let bias = vec![-0.0; co];
            let g = make(n * co * oh * ow, 5);
            let mut reference = vec![0.0; g.len()];
            let mut dx = vec![0.0; x.len()];
            let mut dw = vec![0.0; weight.len()];
            let mut db = vec![0.0; co];
            // Independent original output-major scalar definition. Do not call
            // either shared CPU kernel here: eager now uses the optimized core.
            for b in 0..n {
                for oc in 0..co {
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let out = ((b * co + oc) * oh + oy) * ow + ox;
                            reference[out] = bias[oc];
                            db[oc] += g[out];
                            for ic in 0..ci {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let py = oy * stride.0 + ky;
                                        let px = ox * stride.1 + kx;
                                        if py >= padding.0
                                            && px >= padding.1
                                            && py - padding.0 < h
                                            && px - padding.1 < w
                                        {
                                            let xi = ((b * ci + ic) * h + py - padding.0) * w + px
                                                - padding.1;
                                            let wi = ((oc * ci + ic) * kh + ky) * kw + kx;
                                            reference[out] += x[xi] * weight[wi];
                                            dx[xi] += g[out] * weight[wi];
                                            dw[wi] += g[out] * x[xi];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let inputs = [
                TensorView::new(&x, &xs)?,
                TensorView::new(&weight, &ws)?,
                TensorView::new(&bias, &bs)?,
            ];
            let op = Operation::Conv2d { stride, padding };
            let eager = provider.execute(&op, &inputs)?;
            same(eager.outputs[0].data(), &reference);
            let seed = TensorView::new(&g, &shape)?;
            let eager_grad = eager.backward.unwrap().backward(&inputs, &[], seed)?;
            for (actual, expected) in eager_grad.iter().zip([&dx, &dw, &db]) {
                same(actual.as_ref().unwrap().data(), expected);
            }
            if xs.contains(&0) || ws.contains(&0) {
                assert!(provider.prepare_into(&op, &[&xs, &ws, &bs], true).is_err());
                continue;
            }
            let kernel = provider.prepare_into(&op, &[&xs, &ws, &bs], true)?.unwrap();
            let mut workspace = vec![f32::NAN; kernel.spec().workspace_elements];
            let mut output = vec![1.0; g.len()];
            no_alloc(|| kernel.execute_into(&inputs, &mut output, &mut [], &mut workspace))?;
            same(&output, &reference);
            for mask in 0..8 {
                let expected = [&dx, &dw, &db];
                let mut data = expected.map(|v| vec![9.0; v.len() + 2]);
                let mut destinations = data
                    .iter_mut()
                    .enumerate()
                    .map(|(i, v)| {
                        let n = v.len();
                        (mask & (1 << i) != 0).then_some(&mut v[1..n - 1])
                    })
                    .collect::<Vec<_>>();
                no_alloc(|| {
                    kernel.backward_many_into(
                        &[Some(inputs[0]), Some(inputs[1]), None],
                        &[],
                        seed,
                        &mut destinations,
                        &[
                            GradientWrite::Assign,
                            GradientWrite::Add,
                            GradientWrite::Assign,
                        ],
                        &mut workspace,
                    )
                })?;
                drop(destinations);
                for i in 0..3 {
                    assert_eq!(data[i][0], 9.0);
                    assert_eq!(data[i][data[i].len() - 1], 9.0);
                    let expected = expected[i]
                        .iter()
                        .map(|&v| {
                            if mask & (1 << i) == 0 {
                                9.0
                            } else if i == 1 {
                                9.0 + v
                            } else {
                                v
                            }
                        })
                        .collect::<Vec<_>>();
                    same(&data[i][1..data[i].len() - 1], &expected);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn fused_vjps_preserve_exact_reductions_masks_and_zero_allocations() -> MlResult<()> {
    let provider = backend::CpuBackend::default();
    let cases = [
        (
            Operation::Conv2d {
                stride: (2, 1),
                padding: (1, 1),
            },
            vec![vec![2, 2, 5, 4], vec![3, 2, 3, 3], vec![3]],
        ),
        (Operation::Matmul, vec![vec![2, 1, 3, 5], vec![1, 4, 5, 2]]),
        (
            Operation::Matmul,
            vec![vec![2, 1, 17, 131], vec![1, 2, 131, 33]],
        ),
        (
            Operation::GroupNorm {
                groups: 2,
                epsilon: 1e-5,
            },
            vec![vec![2, 4, 2, 3], vec![4], vec![4]],
        ),
        (
            Operation::GroupNorm {
                groups: 1,
                epsilon: 1e-5,
            },
            vec![vec![1, 2, 1, 1], vec![2], vec![2]],
        ),
    ];
    for (op, shapes) in cases {
        let data = shapes
            .iter()
            .enumerate()
            .map(|(j, s)| {
                (0..s.iter().product())
                    .map(|i| ((i + j * 11) as f32 * 0.17).sin())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let inputs = data
            .iter()
            .zip(&shapes)
            .map(|(d, s)| TensorView::new(d, s))
            .collect::<MlResult<Vec<_>>>()?;
        let kernel = provider
            .prepare_into(
                &op,
                &shapes.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                true,
            )?
            .unwrap();
        let reference = provider.execute(&op, &inputs)?;
        let saved = reference
            .saved
            .iter()
            .map(TensorBuffer::view)
            .collect::<Vec<_>>();
        let gradient = TensorBuffer::from_vec(
            (0..reference.outputs[0].numel())
                .map(|i| (i as f32 * 0.11).cos())
                .collect(),
            &kernel.spec().output,
        )?;
        let expected =
            reference
                .backward
                .as_ref()
                .unwrap()
                .backward(&inputs, &saved, gradient.view())?;
        let optional = inputs
            .iter()
            .zip(&kernel.spec().backward_inputs)
            .map(|(&v, d)| (*d == BackwardInput::Values).then_some(v))
            .collect::<Vec<_>>();
        let mut workspace = vec![0.0; kernel.spec().workspace_elements];
        for mask in 0..1usize << inputs.len() {
            for phase in 0..2 {
                let writes = (0..inputs.len())
                    .map(|i| {
                        if (i + phase) % 2 == 0 {
                            GradientWrite::Assign
                        } else {
                            GradientWrite::Add
                        }
                    })
                    .collect::<Vec<_>>();
                let mut buffers = data
                    .iter()
                    .map(|d| vec![13.25; d.len() + 2])
                    .collect::<Vec<_>>();
                let mut destinations = buffers
                    .iter_mut()
                    .enumerate()
                    .map(|(i, d)| {
                        let n = d.len();
                        (mask & (1 << i) != 0).then_some(&mut d[1..n - 1])
                    })
                    .collect::<Vec<_>>();
                no_alloc(|| {
                    kernel.backward_many_into(
                        &optional,
                        &saved,
                        gradient.view(),
                        &mut destinations,
                        &writes,
                        &mut workspace,
                    )
                })?;
                drop(destinations);
                for (i, d) in buffers.iter().enumerate() {
                    assert_eq!(d[0], 13.25);
                    assert_eq!(d[d.len() - 1], 13.25);
                    let expected = expected[i]
                        .as_ref()
                        .unwrap()
                        .data()
                        .iter()
                        .map(|&v| {
                            if mask & (1 << i) == 0 {
                                13.25
                            } else if writes[i] == GradientWrite::Assign {
                                v
                            } else {
                                13.25 + v
                            }
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(&d[1..d.len() - 1], expected, "{op:?}, mask {mask}");
                }
            }
        }
        // A malformed later destination must not partially update earlier ones.
        let mut buffers = data.iter().map(|d| vec![77.0; d.len()]).collect::<Vec<_>>();
        buffers.last_mut().unwrap().push(77.0);
        let mut destinations = buffers
            .iter_mut()
            .map(|d| Some(d.as_mut_slice()))
            .collect::<Vec<_>>();
        assert!(
            kernel
                .backward_many_into(
                    &optional,
                    &saved,
                    gradient.view(),
                    &mut destinations,
                    &vec![GradientWrite::Assign; inputs.len()],
                    &mut workspace
                )
                .is_err()
        );
        drop(destinations);
        assert!(buffers.iter().flatten().all(|&v| v == 77.0));
    }
    Ok(())
}

#[test]
fn fused_matmul_aliases_accumulate_in_input_order_across_runs() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let w = ctx.parameter(vec![0.1, -0.7, 1.3, 2.1], &[2, 2])?;
    let seed = ctx.tensor(vec![0.3, -0.9, 0.2, 1.1], &[2, 2])?;
    let expected = ctx.with_training_scope(|| {
        let product = w.variable().matmul(w.tensor())?;
        product.backward_with_grad(&seed)?;
        Ok(w.grad()?.unwrap().data().to_vec())
    })?;
    for separate_slots in [false, true] {
        let mut program = PreparedProgram::new();
        let a = program.parameter(&[2, 2])?;
        let b = if separate_slots {
            program.parameter(&[2, 2])?
        } else {
            a
        };
        let product = program.operation(Operation::Matmul, &[a, b])?;
        let parameters = if separate_slots {
            vec![&w, &w]
        } else {
            vec![&w]
        };
        let mut executor = ctx
            .prepare(&program, &parameters, &[product], PreparedMode::Training)?
            .into_executor(&ctx)?;
        for _ in 0..3 {
            executor.with_run(&[], &parameters, |out| {
                out[0].as_variable()?.backward_with_grad(&seed)?;
                assert_eq!(w.grad()?.unwrap().data(), expected);
                Ok(())
            })?;
            assert!(w.grad()?.is_none());
        }
    }
    Ok(())
}
#[test]
fn elementwise_into_matches_eager_vjp_with_zero_allocations() -> MlResult<()> {
    let provider = backend::CpuBackend::default();
    for op in [
        Operation::Add,
        Operation::Sub,
        Operation::Mul,
        Operation::Div,
        Operation::Neg,
        Operation::Square,
        Operation::Exp,
        Operation::Log,
        Operation::Sqrt,
        Operation::Tanh,
        Operation::Sigmoid,
        Operation::Silu,
        Operation::Relu,
        Operation::Sin,
        Operation::Cos,
        Operation::Abs,
    ] {
        let binary = op.input_count() == Some(2);
        let shapes: Vec<&[usize]> = if binary {
            vec![&[2, 1, 3], &[1, 4, 1]]
        } else {
            vec![&[6]]
        };
        let values = if binary {
            vec![vec![0.2, 0.4, 0.8, 1.2, 1.7, 2.1], vec![0.3, 0.7, 1.1, 1.9]]
        } else {
            vec![vec![0.2, 0.4, 0.8, 1.2, 1.7, 2.1]]
        };
        let inputs = values
            .iter()
            .zip(&shapes)
            .map(|(v, s)| TensorView::new(v, s))
            .collect::<MlResult<Vec<_>>>()?;
        let reference = provider.execute(&op, &inputs)?;
        let kernel = provider.prepare_into(&op, &shapes, true)?.unwrap();
        let spec = kernel.spec();
        let n = reference.outputs[0].numel();
        let mut output = vec![12345.0; n + 2];
        let mut saved = spec
            .saved
            .iter()
            .map(|s| vec![0.0; s.iter().product()])
            .collect::<Vec<_>>();
        let mut saved_mut = saved.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>();
        let mut workspace = vec![0.0; spec.workspace_elements];
        no_alloc(|| {
            kernel.execute_into(
                &inputs,
                &mut output[1..n + 1],
                &mut saved_mut,
                &mut workspace,
            )
        })?;
        assert_eq!(output[0], 12345.0);
        assert_eq!(output[n + 1], 12345.0);
        equal(&output[1..n + 1], reference.outputs[0].data());
        drop(saved_mut);
        let saved_views = saved
            .iter()
            .zip(&spec.saved)
            .map(|(v, s)| TensorView::new(v, s))
            .collect::<MlResult<Vec<_>>>()?;
        let grad = TensorBuffer::from_vec(
            (0..n).map(|i| 0.3 + i as f32 * 0.13).collect(),
            &spec.output,
        )?;
        let expected = reference.backward.as_ref().unwrap().backward(
            &inputs,
            &reference
                .saved
                .iter()
                .map(TensorBuffer::view)
                .collect::<Vec<_>>(),
            grad.view(),
        )?;
        let optional = inputs
            .iter()
            .zip(&spec.backward_inputs)
            .map(|(v, d)| {
                if *d == BackwardInput::Shape {
                    None
                } else {
                    Some(*v)
                }
            })
            .collect::<Vec<_>>();
        for (index, expected) in expected.iter().enumerate() {
            let expected = expected.as_ref().unwrap();
            let mut destination = vec![12345.0; expected.numel() + 2];
            let len = destination.len();
            no_alloc(|| {
                kernel.backward_into(
                    index,
                    &optional,
                    &saved_views,
                    grad.view(),
                    &mut destination[1..len - 1],
                    GradientWrite::Assign,
                    &mut workspace,
                )
            })?;
            equal(&destination[1..len - 1], expected.data());
            no_alloc(|| {
                kernel.backward_into(
                    index,
                    &optional,
                    &saved_views,
                    grad.view(),
                    &mut destination[1..len - 1],
                    GradientWrite::Add,
                    &mut workspace,
                )
            })?;
            equal(
                &destination[1..len - 1],
                &expected.data().iter().map(|x| x + x).collect::<Vec<_>>(),
            );
            assert_eq!(destination[0], 12345.0);
            assert_eq!(destination[len - 1], 12345.0);
        }
        for (v, original) in inputs.iter().zip(&values) {
            assert_eq!(v.data(), original);
        }
    }
    Ok(())
}
#[test]
fn invalid_into_bindings_do_not_write_destinations() -> MlResult<()> {
    let provider = backend::CpuBackend::default();
    let kernel = provider
        .prepare_into(&Operation::Exp, &[&[2]], true)?
        .unwrap();
    let x = [0.2, 0.4];
    let input = TensorView::new(&x, &[2])?;
    let mut out = [19.0; 2];
    let mut saved = [23.0; 2];
    assert!(
        kernel
            .execute_into(&[input], &mut out, &mut [], &mut [])
            .is_err()
    );
    assert_eq!(out, [19.0; 2]);
    kernel.execute_into(&[input], &mut out, &mut [&mut saved], &mut [])?;
    let mut dst = [31.0; 2];
    let wrong = TensorView::new(&x, &[1, 2])?;
    assert!(
        kernel
            .backward_into(
                0,
                &[None],
                &[TensorView::new(&saved, &[2])?],
                wrong,
                &mut dst,
                GradientWrite::Assign,
                &mut []
            )
            .is_err()
    );
    assert_eq!(dst, [31.0; 2]);
    assert!(
        provider
            .prepare_into(&Operation::Add, &[&[usize::MAX], &[1]], false)
            .is_err()
    );
    assert!(
        provider
            .prepare_into(&Operation::Pow(2.0), &[&[2, 2]], false)?
            .is_none()
    );
    Ok(())
}
#[test]
fn inference_executor_reuses_arena_and_preserves_held_exports() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let w = ctx.parameter(vec![2.0; 3], &[3])?;
    let sample = ExecutionInputs::new("forward").with("x", ctx.tensor(vec![1.0; 6], &[2, 3])?)?;
    let plan = ctx.prepare_forward(&sample, &[&w], PreparedMode::Inference, |i| {
        let y = i.get("x")?.mul(w.tensor())?.tanh()?.square()?;
        Ok(vec![y])
    })?;
    let mut executor = plan.into_executor(&ctx)?;
    assert!(executor.uses_static_buffers());
    let bytes = executor.arena_bytes();
    let mut held = None;
    for step in 0..100 {
        ctx.replace_parameter(
            w.variable(),
            TensorBuffer::from_vec(vec![1.0 + step as f32 * 0.01; 3], &[3])?,
        )?;
        let x = ctx.tensor(vec![step as f32 * 0.01; 6], &[2, 3])?;
        let expected = ctx.no_grad(|| x.mul(w.tensor())?.tanh()?.square()?.to_vec())?;
        let inputs = ExecutionInputs::new("forward").with("x", x)?;
        executor.with_inputs(&inputs, &[&w], |out| {
            equal(&out[0].to_vec()?, &expected);
            assert!(!out[0].as_variable()?.requires_grad()?);
            if held.is_none() {
                held = Some(out[0].clone());
            }
            Ok(())
        })?;
        assert_eq!(executor.arena_bytes(), bytes);
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    }
    assert_eq!(held.unwrap().to_vec()?, vec![0.0; 6]);
    let bad = ExecutionInputs::new("wrong").with("x", sample.get("x")?.clone())?;
    assert!(executor.with_inputs(&bad, &[&w], |_| Ok(())).is_err());
    assert!(
        executor
            .with_inputs(&sample, &[&w], |_| Err::<(), _>(MlError::StringError(
                "observer".into()
            )))
            .is_err()
    );
    executor.with_inputs(&sample, &[&w], |_| Ok(()))?;
    Ok(())
}
#[test]
fn into_executor_rejects_unsupported_kernels() -> MlResult<()> {
    #[derive(Debug)]
    struct ReplayOnly;
    impl OperationProvider for ReplayOnly {
        fn supports_prepared_replay(&self) -> bool {
            true
        }
        fn execute(&self, _: &Operation, _: &[TensorView<'_>]) -> MlResult<OperationOutput> {
            panic!("must not fall back")
        }
    }
    let ctx = ExecutionContext::builder().operations(ReplayOnly).build();
    let mut p = PreparedProgram::new();
    let x = p.input(&[2, 2], false)?;
    let y = p.operation(Operation::Softmax { axis: 1 }, &[x])?;
    assert!(
        ctx.prepare(&p, &[], &[y], PreparedMode::Inference)?
            .into_executor(&ctx)
            .is_err()
    );
    Ok(())
}

#[test]
fn unary_edges_preserve_eager_numerical_contract() -> MlResult<()> {
    let provider = backend::CpuBackend::default();
    let values = [
        f32::NEG_INFINITY,
        -100.0,
        -88.1,
        -88.0,
        -1.0,
        -0.0,
        0.0,
        0.1,
        88.0,
        88.1,
        100.0,
        f32::INFINITY,
        f32::NAN,
    ];
    let shape = [values.len()];
    let view = TensorView::new(&values, &shape)?;
    for op in [
        Operation::Neg,
        Operation::Square,
        Operation::Exp,
        Operation::Log,
        Operation::Sqrt,
        Operation::Tanh,
        Operation::Sigmoid,
        Operation::Silu,
        Operation::Relu,
        Operation::Sin,
        Operation::Cos,
        Operation::Abs,
    ] {
        let expected = provider.execute(&op, &[view])?;
        let kernel = provider.prepare_into(&op, &[&shape], true)?.unwrap();
        let mut output = vec![0.0; values.len()];
        let mut saved = kernel
            .spec()
            .saved
            .iter()
            .map(|_| vec![0.0; values.len()])
            .collect::<Vec<_>>();
        let mut saved_mut = saved.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>();
        no_alloc(|| kernel.execute_into(&[view], &mut output, &mut saved_mut, &mut []))?;
        for (&actual, &expected) in output.iter().zip(expected.outputs[0].data()) {
            assert!(
                actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan()),
                "{op:?}: {actual} != {expected}"
            );
        }
        drop(saved_mut);
        let saved_views = saved
            .iter()
            .map(|s| TensorView::new(s, &shape))
            .collect::<MlResult<Vec<_>>>()?;
        let g = vec![2.5; values.len()];
        let grad = TensorView::new(&g, &shape)?;
        let expected_grad = expected.backward.as_ref().unwrap().backward(
            &[view],
            &expected
                .saved
                .iter()
                .map(TensorBuffer::view)
                .collect::<Vec<_>>(),
            grad,
        )?;
        let input = if kernel.spec().backward_inputs[0] == BackwardInput::Shape {
            None
        } else {
            Some(view)
        };
        let mut dst = vec![0.0; values.len()];
        no_alloc(|| {
            kernel.backward_into(
                0,
                &[input],
                &saved_views,
                grad,
                &mut dst,
                GradientWrite::Assign,
                &mut [],
            )
        })?;
        equal(&dst, expected_grad[0].as_ref().unwrap().data());
    }
    Ok(())
}

#[test]
fn structural_into_matches_forward_and_vjp_without_allocating() -> MlResult<()> {
    let provider = backend::CpuBackend::default();
    let cases = vec![
        (Operation::Reshape(vec![3, 2]), vec![vec![2, 3]]),
        (Operation::Reshape(vec![]), vec![vec![1]]),
        (Operation::Transpose(vec![2, 0, 1]), vec![vec![2, 3, 4]]),
        (Operation::Transpose(vec![]), vec![vec![]]),
        (
            Operation::Concat { axis: 1 },
            vec![vec![2, 1, 3], vec![2, 4, 3], vec![2, 2, 3]],
        ),
        (Operation::Concat { axis: 0 }, vec![vec![1, 2], vec![3, 2]]),
        (
            Operation::Concat { axis: 2 },
            vec![vec![2, 3, 1], vec![2, 3, 2]],
        ),
        (Operation::Sum, vec![vec![17]]),
        (Operation::Sum, vec![vec![]]),
    ];
    for (op, shapes) in cases {
        let data = shapes
            .iter()
            .enumerate()
            .map(|(j, s)| {
                (0..s.iter().product())
                    .map(|i| i as f32 * 0.3 - j as f32 * 2.1)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let inputs = data
            .iter()
            .zip(&shapes)
            .map(|(d, s)| TensorView::new(d, s))
            .collect::<MlResult<Vec<_>>>()?;
        let shape_refs = shapes.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let kernel = provider.prepare_into(&op, &shape_refs, true)?.unwrap();
        let expected = provider.execute(&op, &inputs)?;
        let n = expected.outputs[0].numel();
        let mut out = vec![8765.0; n + 2];
        no_alloc(|| kernel.execute_into(&inputs, &mut out[1..n + 1], &mut [], &mut []))?;
        assert_eq!(&out[1..n + 1], expected.outputs[0].data());
        assert_eq!(out[0], 8765.0);
        assert_eq!(out[n + 1], 8765.0);
        let seed = TensorBuffer::from_vec(
            (0..n).map(|i| 0.4 + i as f32 * 0.7).collect(),
            &kernel.spec().output,
        )?;
        let grads = expected
            .backward
            .unwrap()
            .backward(&inputs, &[], seed.view())?;
        let shape_only = vec![None; inputs.len()];
        for (i, g) in grads.iter().enumerate() {
            let g = g.as_ref().unwrap();
            let len = g.numel();
            let mut dst = vec![8765.0; len + 2];
            no_alloc(|| {
                kernel.backward_into(
                    i,
                    &shape_only,
                    &[],
                    seed.view(),
                    &mut dst[1..len + 1],
                    GradientWrite::Assign,
                    &mut [],
                )
            })?;
            assert_eq!(&dst[1..len + 1], g.data());
            no_alloc(|| {
                kernel.backward_into(
                    i,
                    &shape_only,
                    &[],
                    seed.view(),
                    &mut dst[1..len + 1],
                    GradientWrite::Add,
                    &mut [],
                )
            })?;
            assert_eq!(
                &dst[1..len + 1],
                g.data().iter().map(|x| x + x).collect::<Vec<_>>()
            );
            assert_eq!(dst[0], 8765.0);
            assert_eq!(dst[len + 1], 8765.0);
        }
        for (v, d) in inputs.iter().zip(&data) {
            assert_eq!(v.data(), d);
        }
    }
    Ok(())
}

#[test]
fn structural_validation_and_sum_grouping_are_preserved() -> MlResult<()> {
    let provider = backend::CpuBackend::default();
    for (op, shapes) in [
        (Operation::Reshape(vec![3]), vec![vec![2]]),
        (Operation::Transpose(vec![0, 0]), vec![vec![2, 3]]),
        (Operation::Transpose(vec![2, 0]), vec![vec![2, 3]]),
        (Operation::Concat { axis: 2 }, vec![vec![2, 3]]),
        (Operation::Concat { axis: 0 }, vec![vec![2, 3], vec![1, 4]]),
        (Operation::Concat { axis: 0 }, vec![]),
    ] {
        assert!(
            provider
                .prepare_into(
                    &op,
                    &shapes.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    true
                )
                .is_err()
        );
    }
    let values = [
        1e20, 1.0, -1e20, 3.0, 4.0, 5.0, 6.0, 7.0, 1e20, -1e20, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
        0.25,
    ];
    let view = TensorView::new(&values, &[17])?;
    let kernel = provider
        .prepare_into(&Operation::Sum, &[&[17]], true)?
        .unwrap();
    let expected = provider.execute(&Operation::Sum, &[view])?;
    let mut out = [0.0];
    kernel.execute_into(&[view], &mut out, &mut [], &mut [])?;
    assert_eq!(out[0].to_bits(), expected.outputs[0].data()[0].to_bits());
    let mut sentinel = [123.0; 2];
    assert!(
        kernel
            .execute_into(&[view], &mut sentinel, &mut [], &mut [])
            .is_err()
    );
    assert_eq!(sentinel, [123.0; 2]);
    let mut grad = [321.0; 17];
    let wrong = TensorView::new(&[1.0], &[1])?;
    assert!(
        kernel
            .backward_into(
                0,
                &[None],
                &[],
                wrong,
                &mut grad,
                GradientWrite::Assign,
                &mut []
            )
            .is_err()
    );
    assert_eq!(grad, [321.0; 17]);
    Ok(())
}

#[test]
fn structural_arena_pipeline_reuses_buffers_and_matches_eager() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let sample = ExecutionInputs::new("pipeline").with("x", ctx.tensor(vec![0.0; 6], &[2, 3])?)?;
    let forward = |inputs: &ExecutionInputs| -> MlResult<Vec<Tensor>> {
        let x = inputs.get("x")?;
        let a = ctx
            .execute(&Operation::Reshape(vec![3, 2]), &[x])?
            .remove(0);
        let b = ctx
            .execute(&Operation::Transpose(vec![1, 0]), &[&a])?
            .remove(0);
        let c = ctx
            .execute(&Operation::Concat { axis: 0 }, &[&b, &b])?
            .remove(0);
        let d = ctx.execute(&Operation::Sum, &[&c])?.remove(0);
        Ok(vec![c, d])
    };
    let plan = ctx.prepare_forward(&sample, &[], PreparedMode::Inference, forward)?;
    let mut executor = plan.into_executor(&ctx)?;
    let bytes = executor.arena_bytes();
    let mut held = None;
    for step in 0..100 {
        let inputs = ExecutionInputs::new("pipeline").with(
            "x",
            ctx.tensor((0..6).map(|i| (i + step) as f32).collect(), &[2, 3])?,
        )?;
        let expected = ctx.no_grad(|| forward(&inputs))?;
        executor.with_inputs(&inputs, &[], |out| {
            for (a, b) in out.iter().zip(&expected) {
                assert_eq!(a.to_vec()?, b.to_vec()?);
            }
            if held.is_none() {
                held = Some((out[0].clone(), out[0].to_vec()?));
            }
            Ok(())
        })?;
        assert_eq!(executor.arena_bytes(), bytes);
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    }
    let (held, values) = held.unwrap();
    assert_eq!(held.to_vec()?, values);
    Ok(())
}

#[test]
fn matmul_into_broadcast_vjp_matches_eager_without_allocating() -> MlResult<()> {
    let provider = backend::CpuBackend::default();
    for (left, right) in [
        (vec![2, 3], vec![3, 4]),
        (vec![2, 1, 3, 5], vec![1, 4, 5, 2]),
        (vec![3, 5], vec![2, 4, 5, 2]),
        (vec![2, 4, 3, 5], vec![5, 2]),
        (vec![33, 35], vec![35, 34]),
        (vec![17, 131], vec![131, 137]),
        (vec![2, 1, 17, 131], vec![1, 2, 131, 33]),
    ] {
        let a = (0..left.iter().product())
            .map(|i| (i as f32 * 0.17).sin())
            .collect::<Vec<_>>();
        let b = (0..right.iter().product())
            .map(|i| (i as f32 * 0.13).cos())
            .collect::<Vec<_>>();
        let inputs = [TensorView::new(&a, &left)?, TensorView::new(&b, &right)?];
        let kernel = provider
            .prepare_into(&Operation::Matmul, &[&left, &right], true)?
            .unwrap();
        let expected = provider.execute(&Operation::Matmul, &inputs)?;
        let n = expected.outputs[0].numel();
        let mut dst = vec![12345.0; n + 2];
        let mut workspace = vec![0.0; kernel.spec().workspace_elements];
        no_alloc(|| kernel.execute_into(&inputs, &mut dst[1..n + 1], &mut [], &mut workspace))?;
        assert_eq!(&dst[1..n + 1], expected.outputs[0].data());
        assert_eq!(dst[0], 12345.0);
        assert_eq!(dst[n + 1], 12345.0);
        let seed = TensorBuffer::from_vec(
            (0..n).map(|i| 0.3 + (i as f32 * 0.11).sin()).collect(),
            &kernel.spec().output,
        )?;
        let expected_grads = expected
            .backward
            .unwrap()
            .backward(&inputs, &[], seed.view())?;
        for index in 0..2 {
            let expected = expected_grads[index].as_ref().unwrap();
            let n = expected.numel();
            let mut grad = vec![12345.0; n + 2];
            no_alloc(|| {
                kernel.backward_into(
                    index,
                    &[Some(inputs[0]), Some(inputs[1])],
                    &[],
                    seed.view(),
                    &mut grad[1..n + 1],
                    GradientWrite::Assign,
                    &mut workspace,
                )
            })?;
            assert_eq!(&grad[1..n + 1], expected.data());
            no_alloc(|| {
                kernel.backward_into(
                    index,
                    &[Some(inputs[0]), Some(inputs[1])],
                    &[],
                    seed.view(),
                    &mut grad[1..n + 1],
                    GradientWrite::Add,
                    &mut workspace,
                )
            })?;
            assert_eq!(
                &grad[1..n + 1],
                expected.data().iter().map(|x| x + x).collect::<Vec<_>>()
            );
            assert_eq!(grad[0], 12345.0);
            assert_eq!(grad[n + 1], 12345.0);
        }
    }
    Ok(())
}
#[test]
fn matmul_into_validation_and_independent_gradient_check() -> MlResult<()> {
    let provider = backend::CpuBackend::default();
    for (a, b) in [
        (vec![3], vec![3, 2]),
        (vec![2, 3], vec![4, 2]),
        (vec![2, 2, 3], vec![3, 3, 2]),
        (vec![usize::MAX, 2], vec![2, 1]),
    ] {
        assert!(
            provider
                .prepare_into(&Operation::Matmul, &[&a, &b], true)
                .is_err()
        );
    }
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0];
    let inputs = [TensorView::new(&a, &[2, 2])?, TensorView::new(&b, &[2, 1])?];
    let kernel = provider
        .prepare_into(&Operation::Matmul, &[&[2, 2], &[2, 1]], true)?
        .unwrap();
    let mut out = [123.0; 2];
    assert!(
        kernel
            .execute_into(&inputs, &mut out, &mut [], &mut [])
            .is_err()
    );
    assert_eq!(out, [123.0; 2]);
    let mut work = vec![0.0; kernel.spec().workspace_elements];
    kernel.execute_into(&inputs, &mut out, &mut [], &mut work)?;
    assert_eq!(out, [17.0, 39.0]);
    let seed = TensorView::new(&[2.0, 3.0], &[2, 1])?;
    let mut da = [0.0; 4];
    let mut db = [0.0; 2];
    kernel.backward_into(
        0,
        &[Some(inputs[0]), Some(inputs[1])],
        &[],
        seed,
        &mut da,
        GradientWrite::Assign,
        &mut work,
    )?;
    kernel.backward_into(
        1,
        &[Some(inputs[0]), Some(inputs[1])],
        &[],
        seed,
        &mut db,
        GradientWrite::Assign,
        &mut work,
    )?;
    assert_eq!(da, [10.0, 12.0, 15.0, 18.0]);
    assert_eq!(db, [11.0, 16.0]);
    assert!(
        kernel
            .backward_into(
                1,
                &[None, Some(inputs[1])],
                &[],
                seed,
                &mut db,
                GradientWrite::Assign,
                &mut work
            )
            .is_err()
    );
    assert_eq!(db, [11.0, 16.0]);
    Ok(())
}
#[test]
fn linear_model_uses_matmul_arena_for_repeated_inference() -> MlResult<()> {
    use trench_deep::{nn::LinearRegression, trainer::TrainableModel};
    let ctx = ExecutionContext::new();
    let model = LinearRegression::new(&ctx, 3, 2)?;
    let sample = ExecutionInputs::new("predict").with("x", ctx.tensor(vec![1.0; 6], &[2, 3])?)?;
    let plan = ctx.prepare_forward(&sample, &model.parameters(), PreparedMode::Inference, |i| {
        Ok(vec![model.predict(i.get("x")?)?])
    })?;
    let mut executor = plan.into_executor(&ctx)?;
    let bytes = executor.arena_bytes();
    let mut held = None;
    for step in 0..100 {
        let inputs = ExecutionInputs::new("predict")
            .with("x", ctx.tensor(vec![step as f32 * 0.01; 6], &[2, 3])?)?;
        let expected = ctx.no_grad(|| model.predict(inputs.get("x")?)?.to_vec())?;
        executor.with_inputs(&inputs, &model.parameters(), |out| {
            assert_eq!(out[0].to_vec()?, expected);
            if held.is_none() {
                held = Some((out[0].clone(), expected));
            }
            Ok(())
        })?;
        assert_eq!(executor.arena_bytes(), bytes);
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    }
    let (held, values) = held.unwrap();
    assert_eq!(held.to_vec()?, values);
    Ok(())
}
#[test]
fn into_div_zero_preserves_existing_cpu_forward_contract() -> MlResult<()> {
    let p = backend::CpuBackend::default();
    let a = [-1.0, 0.0, 1.0, f32::NAN];
    let b = [0.0, -0.0, 0.0, 0.0];
    let views = [TensorView::new(&a, &[4])?, TensorView::new(&b, &[4])?];
    let expected = p.execute(&Operation::Div, &views)?;
    let k = p
        .prepare_into(&Operation::Div, &[&[4], &[4]], false)?
        .unwrap();
    let mut out = [0.0; 4];
    no_alloc(|| k.execute_into(&views, &mut out, &mut [], &mut []))?;
    assert_eq!(out, expected.outputs[0].data());
    Ok(())
}

#[test]
fn conv_into_matches_all_gradients_without_allocations() -> MlResult<()> {
    let provider = backend::CpuBackend::default();
    for (xs, ws, stride, padding) in [
        ([2, 2, 5, 6], [3, 2, 3, 2], (1, 1), (0, 0)),
        ([2, 2, 4, 5], [3, 2, 2, 3], (2, 1), (1, 2)),
        ([1, 1, 2, 2], [2, 1, 1, 1], (1, 2), (3, 1)),
        ([1, 1, 3, 3], [1, 1, 3, 3], (1, 1), (0, 0)),
    ] {
        let op = Operation::Conv2d { stride, padding };
        let bs = [ws[0]];
        let data = [
            (0..xs.iter().product())
                .map(|i| (i as f32 * 0.13).sin())
                .collect::<Vec<_>>(),
            (0..ws.iter().product())
                .map(|i| (i as f32 * 0.17).cos())
                .collect(),
            (0..bs[0]).map(|i| i as f32 * 0.3).collect(),
        ];
        let inputs = [
            TensorView::new(&data[0], &xs)?,
            TensorView::new(&data[1], &ws)?,
            TensorView::new(&data[2], &bs)?,
        ];
        let k = provider.prepare_into(&op, &[&xs, &ws, &bs], true)?.unwrap();
        let expected = provider.execute(&op, &inputs)?;
        let n = expected.outputs[0].numel();
        let mut out = vec![12345.0; n + 2];
        let mut work = vec![0.0; k.spec().workspace_elements];
        no_alloc(|| k.execute_into(&inputs, &mut out[1..n + 1], &mut [], &mut work))?;
        assert_eq!(&out[1..n + 1], expected.outputs[0].data());
        assert_eq!(out[0], 12345.0);
        assert_eq!(out[n + 1], 12345.0);
        let seed = TensorBuffer::from_vec(
            (0..n).map(|i| 0.2 + (i as f32 * 0.19).cos()).collect(),
            &k.spec().output,
        )?;
        let grads = expected
            .backward
            .unwrap()
            .backward(&inputs, &[], seed.view())?;
        for i in 0..3 {
            let expected = grads[i].as_ref().unwrap();
            let n = expected.numel();
            let mut dst = vec![12345.0; n + 2];
            no_alloc(|| {
                k.backward_into(
                    i,
                    &[Some(inputs[0]), Some(inputs[1]), None],
                    &[],
                    seed.view(),
                    &mut dst[1..n + 1],
                    GradientWrite::Assign,
                    &mut work,
                )
            })?;
            assert_eq!(&dst[1..n + 1], expected.data());
            no_alloc(|| {
                k.backward_into(
                    i,
                    &[Some(inputs[0]), Some(inputs[1]), None],
                    &[],
                    seed.view(),
                    &mut dst[1..n + 1],
                    GradientWrite::Add,
                    &mut work,
                )
            })?;
            assert_eq!(
                &dst[1..n + 1],
                expected.data().iter().map(|x| x + x).collect::<Vec<_>>()
            );
            assert_eq!(dst[0], 12345.0);
            assert_eq!(dst[n + 1], 12345.0);
        }
        for (v, d) in inputs.iter().zip(&data) {
            assert_eq!(v.data(), d);
        }
    }
    Ok(())
}
#[test]
fn conv_into_independent_values_and_invalid_bindings() -> MlResult<()> {
    let p = backend::CpuBackend::default();
    let op = Operation::Conv2d {
        stride: (1, 1),
        padding: (0, 0),
    };
    let shapes: [&[usize]; 3] = [&[1, 1, 2, 2], &[1, 1, 1, 1], &[1]];
    let inputs = [
        TensorView::new(&[1.0, 2.0, 3.0, 4.0], shapes[0])?,
        TensorView::new(&[2.0], shapes[1])?,
        TensorView::new(&[1.0], shapes[2])?,
    ];
    let k = p.prepare_into(&op, &shapes, true)?.unwrap();
    let mut out = [77.0; 4];
    let mut work = vec![0.0; k.spec().workspace_elements];
    assert!(k.execute_into(&inputs, &mut out, &mut [], &mut []).is_err());
    assert_eq!(out, [77.0; 4]);
    k.execute_into(&inputs, &mut out, &mut [], &mut work)?;
    assert_eq!(out, [3.0, 5.0, 7.0, 9.0]);
    let seed = TensorView::new(&[1.0, 2.0, 3.0, 4.0], shapes[0])?;
    for (i, expected) in [vec![2.0, 4.0, 6.0, 8.0], vec![30.0], vec![10.0]]
        .into_iter()
        .enumerate()
    {
        let mut dst = vec![0.0; expected.len()];
        k.backward_into(
            i,
            &[Some(inputs[0]), Some(inputs[1]), None],
            &[],
            seed,
            &mut dst,
            GradientWrite::Assign,
            &mut work,
        )?;
        assert_eq!(dst, expected);
        assert!(
            k.backward_into(
                i,
                &[None, Some(inputs[1]), None],
                &[],
                seed,
                &mut dst,
                GradientWrite::Assign,
                &mut work
            )
            .is_err()
        );
        assert_eq!(dst, expected);
    }
    for (stride, padding) in [((0, 1), (0, 0)), ((1, 1), (usize::MAX, 0))] {
        assert!(
            p.prepare_into(&Operation::Conv2d { stride, padding }, &shapes, true)
                .is_err()
        );
    }
    assert!(
        p.prepare_into(&op, &[&[1, 2, 2, 2], shapes[1], shapes[2]], true)
            .is_err()
    );
    assert!(
        p.prepare_into(&op, &[shapes[0], &[1, 1, 3, 3], shapes[2]], true)
            .is_err()
    );
    assert!(
        p.prepare_into(&op, &[shapes[0], shapes[1], &[2]], true)
            .is_err()
    );
    Ok(())
}
#[test]
fn conv_arena_pipeline_reuses_current_parameters_and_exports() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let w = ctx.parameter(vec![0.2; 9], &[1, 1, 3, 3])?;
    let b = ctx.parameter(vec![0.1], &[1])?;
    let sample =
        ExecutionInputs::new("conv").with("x", ctx.tensor(vec![1.0; 16], &[1, 1, 4, 4])?)?;
    let forward = |i: &ExecutionInputs| -> MlResult<Vec<Tensor>> {
        let out = ctx
            .execute(
                &Operation::Conv2d {
                    stride: (1, 1),
                    padding: (1, 1),
                },
                &[i.get("x")?, w.tensor(), b.tensor()],
            )?
            .remove(0)
            .tanh()?;
        Ok(vec![out])
    };
    let mut executor = ctx
        .prepare_forward(&sample, &[&w, &b], PreparedMode::Inference, forward)?
        .into_executor(&ctx)?;
    let bytes = executor.arena_bytes();
    let mut held = None;
    for step in 0..100 {
        ctx.replace_parameter(
            w.variable(),
            TensorBuffer::from_vec(vec![0.2 + step as f32 * 0.001; 9], &[1, 1, 3, 3])?,
        )?;
        let inputs = ExecutionInputs::new("conv").with(
            "x",
            ctx.tensor(vec![step as f32 * 0.01; 16], &[1, 1, 4, 4])?,
        )?;
        let expected = ctx.no_grad(|| forward(&inputs))?;
        executor.with_inputs(&inputs, &[&w, &b], |out| {
            assert_eq!(out[0].to_vec()?, expected[0].to_vec()?);
            if held.is_none() {
                held = Some((out[0].clone(), out[0].to_vec()?));
            }
            Ok(())
        })?;
        assert_eq!(executor.arena_bytes(), bytes);
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    }
    let (held, values) = held.unwrap();
    assert_eq!(held.to_vec()?, values);
    Ok(())
}

#[test]
fn norm_softmax_upsample_and_losses_match_eager_without_allocating() -> MlResult<()> {
    let p = backend::CpuBackend::default();
    let mut cases = vec![
        (Operation::Softmax { axis: 0 }, vec![vec![2, 3, 4]]),
        (Operation::Softmax { axis: 1 }, vec![vec![2, 3, 4]]),
        (Operation::Softmax { axis: 2 }, vec![vec![2, 3, 4]]),
        (
            Operation::NearestUpsample2d { scale: (2, 3) },
            vec![vec![2, 3, 2, 4]],
        ),
        (
            Operation::GroupNorm {
                groups: 1,
                epsilon: 1e-5,
            },
            vec![vec![2, 4, 2, 3], vec![4], vec![4]],
        ),
        (
            Operation::GroupNorm {
                groups: 2,
                epsilon: 1e-5,
            },
            vec![vec![2, 4, 2, 3], vec![4], vec![4]],
        ),
        (
            Operation::GroupNorm {
                groups: 4,
                epsilon: 1e-5,
            },
            vec![vec![1, 4, 1, 1], vec![4], vec![4]],
        ),
    ];
    for kind in [LossKind::Mse, LossKind::Mae, LossKind::BinaryCrossEntropy] {
        cases.push((
            Operation::Loss {
                kind,
                reduction: Reduction::Mean,
            },
            vec![vec![2, 3], vec![2, 3]],
        ));
    }
    for (op, shapes) in cases {
        let values = shapes
            .iter()
            .enumerate()
            .map(|(j, s)| {
                (0..s.iter().product())
                    .map(|i| ((i + 3 * j) % 11) as f32 / 10.0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let inputs = values
            .iter()
            .zip(&shapes)
            .map(|(v, s)| TensorView::new(v, s))
            .collect::<MlResult<Vec<_>>>()?;
        let k = p
            .prepare_into(
                &op,
                &shapes.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                true,
            )?
            .unwrap();
        let spec = k.spec();
        let expected = p.execute(&op, &inputs)?;
        let n = expected.outputs[0].numel();
        let mut out = vec![999.0; n + 2];
        let mut saved = spec
            .saved
            .iter()
            .map(|s| vec![0.0; s.iter().product()])
            .collect::<Vec<_>>();
        let mut saved_mut = saved.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>();
        let mut work = vec![0.0; spec.workspace_elements];
        no_alloc(|| k.execute_into(&inputs, &mut out[1..n + 1], &mut saved_mut, &mut work))?;
        drop(saved_mut);
        assert_eq!(&out[1..n + 1], expected.outputs[0].data());
        assert_eq!(out[0], 999.0);
        assert_eq!(out[n + 1], 999.0);
        for (a, b) in saved.iter().zip(&expected.saved) {
            assert_eq!(a, b.data());
        }
        let saved_views = saved
            .iter()
            .zip(&spec.saved)
            .map(|(v, s)| TensorView::new(v, s))
            .collect::<MlResult<Vec<_>>>()?;
        let seed = TensorBuffer::from_vec(
            (0..n).map(|i| 0.3 + (i % 7) as f32 * 0.2).collect(),
            &spec.output,
        )?;
        let grads = expected.backward.unwrap().backward(
            &inputs,
            &expected
                .saved
                .iter()
                .map(TensorBuffer::view)
                .collect::<Vec<_>>(),
            seed.view(),
        )?;
        let optional = inputs
            .iter()
            .zip(&spec.backward_inputs)
            .map(|(v, d)| {
                if *d == BackwardInput::Values {
                    Some(*v)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        for (i, g) in grads.iter().enumerate() {
            if let Some(g) = g {
                let n = g.numel();
                let mut dst = vec![999.0; n + 2];
                no_alloc(|| {
                    k.backward_into(
                        i,
                        &optional,
                        &saved_views,
                        seed.view(),
                        &mut dst[1..n + 1],
                        GradientWrite::Assign,
                        &mut work,
                    )
                })?;
                assert_eq!(&dst[1..n + 1], g.data());
                no_alloc(|| {
                    k.backward_into(
                        i,
                        &optional,
                        &saved_views,
                        seed.view(),
                        &mut dst[1..n + 1],
                        GradientWrite::Add,
                        &mut work,
                    )
                })?;
                assert_eq!(
                    &dst[1..n + 1],
                    g.data().iter().map(|v| v + v).collect::<Vec<_>>()
                );
                assert_eq!(dst[0], 999.0);
                assert_eq!(dst[n + 1], 999.0);
            } else {
                let mut dst = vec![123.0; values[i].len()];
                assert!(
                    k.backward_into(
                        i,
                        &optional,
                        &saved_views,
                        seed.view(),
                        &mut dst,
                        GradientWrite::Assign,
                        &mut work
                    )
                    .is_err()
                );
                assert!(dst.iter().all(|&v| v == 123.0));
            }
        }
        let inference = p
            .prepare_into(
                &op,
                &shapes.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                false,
            )?
            .unwrap();
        no_alloc(|| inference.execute_into(&inputs, &mut out[1..n + 1], &mut [], &mut []))?;
        assert_eq!(&out[1..n + 1], expected.outputs[0].data());
    }
    Ok(())
}
#[test]
fn saved_and_loss_validation_precedes_destination_writes() -> MlResult<()> {
    let p = backend::CpuBackend::default();
    let op = Operation::GroupNorm {
        groups: 2,
        epsilon: 1e-5,
    };
    let shapes: [&[usize]; 3] = [&[1, 2, 1, 2], &[2], &[2]];
    let inputs = [
        TensorView::new(&[1.0, 2.0, 3.0, 4.0], shapes[0])?,
        TensorView::new(&[1.0, 1.0], shapes[1])?,
        TensorView::new(&[0.0, 0.0], shapes[2])?,
    ];
    let k = p.prepare_into(&op, &shapes, true)?.unwrap();
    let mut dst = [123.0; 4];
    let mut work = [0.0; 4];
    assert!(
        k.execute_into(&inputs, &mut dst, &mut [], &mut work)
            .is_err()
    );
    assert_eq!(dst, [123.0; 4]);
    assert!(
        k.backward_into(
            0,
            &[None, Some(inputs[1]), None],
            &[],
            inputs[0],
            &mut dst,
            GradientWrite::Assign,
            &mut work
        )
        .is_err()
    );
    assert_eq!(dst, [123.0; 4]);
    for op in [
        Operation::GroupNorm {
            groups: 3,
            epsilon: 1e-5,
        },
        Operation::GroupNorm {
            groups: 2,
            epsilon: 0.0,
        },
    ] {
        assert!(p.prepare_into(&op, &shapes, false).is_err());
    }
    assert!(
        p.prepare_into(&Operation::Softmax { axis: 2 }, &[&[2, 3]], false)
            .is_err()
    );
    assert!(
        p.prepare_into(
            &Operation::NearestUpsample2d { scale: (0, 1) },
            &[&[1, 1, 2, 2]],
            false
        )
        .is_err()
    );
    let loss = p
        .prepare_into(
            &Operation::Loss {
                kind: LossKind::BinaryCrossEntropy,
                reduction: Reduction::Mean,
            },
            &[&[2], &[2]],
            true,
        )?
        .unwrap();
    for target in [[-0.1, 1.0], [f32::NAN, 0.0]] {
        let inputs = [
            TensorView::new(&[0.1, 0.9], &[2])?,
            TensorView::new(&target, &[2])?,
        ];
        let mut out = [123.0];
        assert!(
            loss.execute_into(&inputs, &mut out, &mut [], &mut work)
                .is_err()
        );
        assert_eq!(out, [123.0]);
        let mut grad = [456.0; 2];
        assert!(
            loss.backward_into(
                0,
                &[Some(inputs[0]), Some(inputs[1])],
                &[],
                TensorView::new(&[1.0], &[])?,
                &mut grad,
                GradientWrite::Assign,
                &mut work
            )
            .is_err()
        );
        assert_eq!(grad, [456.0; 2]);
    }
    Ok(())
}
