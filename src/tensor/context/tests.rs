    use super::*;

    #[test]
    fn contexts_are_isolated_and_mismatch_is_rejected() -> MlResult<()> {
        let a = ExecutionContext::new();
        let b = ExecutionContext::new();
        let x = a.tensor(vec![1.0], &[1])?;
        let y = b.tensor(vec![2.0], &[1])?;
        assert!(matches!(
            a.add(&x, &y),
            Err(crate::MlError::ContextError(ContextError::Mismatch))
        ));
        Ok(())
    }

    #[test]
    fn dropped_context_is_reported() -> MlResult<()> {
        let tensor = {
            let ctx = ExecutionContext::new();
            ctx.tensor(vec![1.0], &[1])?
        };
        assert!(matches!(
            tensor.to_vec(),
            Err(crate::MlError::ContextError(ContextError::Dropped))
        ));
        Ok(())
    }

    #[test]
    fn no_grad_depth_is_restored_on_error() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let _: Result<(), _> = ctx.no_grad(|| Err(crate::MlError::StringError("stop".into())));
        assert_eq!(ctx.graph_stats()?.no_grad_depth, 0);
        Ok(())
    }

    #[test]
    fn backward_accumulates_fan_in_and_consumes_graph() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![3.0], &[])?;
        let xx = ctx.mul_variable(&x, &x)?;
        let y = ctx.add_variable(&xx, &x)?;
        y.backward()?;
        assert_eq!(x.grad()?.expect("leaf gradient").data, vec![7.0]);
        assert!(matches!(
            y.backward(),
            Err(crate::MlError::AutogradError(
                AutogradError::GraphAlreadyFreed(_)
            ))
        ));
        Ok(())
    }

    #[test]
    fn vector_output_requires_explicit_cotangent() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![2.0, 3.0], &[2])?;
        let y = ctx.mul_variable(&x, &x)?;
        assert!(matches!(
            y.backward(),
            Err(crate::MlError::AutogradError(
                AutogradError::OutputNotScalar(_)
            ))
        ));
        let seed = ctx.tensor(vec![1.0, 2.0], &[2])?;
        y.backward_with_grad(&seed)?;
        assert_eq!(x.grad()?.expect("leaf gradient").data, vec![4.0, 12.0]);
        Ok(())
    }

    #[test]
    fn fallible_operator_overloads_use_the_owning_context() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![2.0], &[])?;
        let y = (&x * &x)?;
        let z = (&y + &x)?;
        z.backward()?;
        assert_eq!(x.grad()?.expect("leaf gradient").data, vec![5.0]);
        Ok(())
    }

    #[test]
    fn no_grad_skips_graph_registration() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![2.0], &[])?;
        let y = ctx.no_grad(|| ctx.mul_variable(&x, &x))?;
        assert!(!y.requires_grad());
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
        Ok(())
    }

    #[test]
    fn unary_sum_chain_has_correct_gradient() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![0.5, 1.5], &[2])?;
        let squared = ctx.square_variable(&x)?;
        let exponentiated = ctx.variable_from(ctx.exp(squared.tensor())?)?;
        let loss = ctx.sum_variable(&exponentiated)?;
        loss.backward()?;
        let gradient = x.grad()?.expect("leaf gradient").data;
        assert!((gradient[0] - 2.0 * 0.5 * (0.25f32).exp()).abs() < 1e-5);
        assert!((gradient[1] - 2.0 * 1.5 * (2.25f32).exp()).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn matmul_gradient_matches_known_result() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let w = ctx.parameter(vec![2.0, 0.0, 1.0, 3.0], &[2, 2])?;
        let y = ctx.matmul_variable(&x, &w)?;
        let loss = ctx.sum_variable(&y)?;
        loss.backward()?;
        assert_eq!(
            x.grad()?.expect("x gradient").data,
            vec![2.0, 4.0, 2.0, 4.0]
        );
        assert_eq!(
            w.grad()?.expect("w gradient").data,
            vec![4.0, 4.0, 6.0, 6.0]
        );
        Ok(())
    }

    fn finite_difference_check(
        x0: f32,
        expected: impl Fn(f32) -> f32,
        build: impl Fn(&ExecutionContext, &ContextVariable) -> MlResult<ContextVariable>,
    ) -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![x0], &[])?;
        build(&ctx, &x)?.backward()?;
        let analytic = x.grad()?.expect("leaf gradient").data[0];
        let epsilon = 1e-3;
        let numeric = (expected(x0 + epsilon) - expected(x0 - epsilon)) / (2.0 * epsilon);
        let absolute = (analytic - numeric).abs();
        let relative = absolute / numeric.abs().max(1e-6);
        assert!(
            absolute <= 1e-3 || relative <= 1e-3,
            "analytic={analytic}, numeric={numeric}, abs={absolute}, rel={relative}"
        );
        Ok(())
    }

    #[test]
    fn unary_gradients_match_central_finite_difference() -> MlResult<()> {
        finite_difference_check(1.3, f32::sqrt, |ctx, x| ctx.sqrt_variable(x))?;
        finite_difference_check(1.3, |x| x.powf(2.7), |ctx, x| ctx.powf_variable(x, 2.7))?;
        finite_difference_check(0.7, f32::sin, |ctx, x| ctx.sin_variable(x))?;
        finite_difference_check(0.7, f32::cos, |ctx, x| ctx.cos_variable(x))?;
        finite_difference_check(0.7, f32::tanh, |ctx, x| ctx.tanh_variable(x))?;
        finite_difference_check(
            0.7,
            |x| 1.0 / (1.0 + (-x).exp()),
            |ctx, x| ctx.sigmoid_variable(x),
        )?;
        finite_difference_check(
            0.7,
            |x| x / (1.0 + (-x).exp()),
            |ctx, x| ctx.silu_variable(x),
        )?;
        finite_difference_check(0.7, |x| x.max(0.0), |ctx, x| ctx.relu_variable(x))?;
        Ok(())
    }

    #[test]
    fn approximate_trigonometry_matches_reference_and_finite_difference() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let values = vec![-std::f32::consts::PI, -1.5, 0.0, 1.5, std::f32::consts::PI];
        let input = ctx.tensor(values.clone(), &[5])?;
        let sin = ctx.approx_sin(&input, 1e-8)?.to_vec()?;
        let cos = ctx.approx_cos(&input, 1e-8)?.to_vec()?;
        for ((value, approximate_sin), approximate_cos) in
            values.iter().zip(sin.iter()).zip(cos.iter())
        {
            assert!((approximate_sin - value.sin()).abs() <= 1e-4);
            assert!((approximate_cos - value.cos()).abs() <= 1e-4);
        }
        finite_difference_check(
            0.7,
            approx_sin_value,
            |ctx, x| ctx.approx_sin_variable(x, 1e-8),
        )?;
        finite_difference_check(
            0.7,
            approx_cos_value,
            |ctx, x| ctx.approx_cos_variable(x, 1e-8),
        )?;
        assert!(ctx.approx_sin(&input, 0.0).is_err());
        assert!(ctx.approx_cos(&input, f32::NAN).is_err());
        Ok(())
    }

    #[test]
    fn elementwise_losses_implement_all_reductions() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let prediction = ctx.tensor(vec![1.0, 3.0], &[2])?;
        let target = ctx.tensor(vec![2.0, 1.0], &[2])?;
        assert_eq!(
            ctx.mse_loss(&prediction, &target, Reduction::None)?
                .to_vec()?,
            vec![1.0, 4.0]
        );
        assert_eq!(
            ctx.mse_loss(&prediction, &target, Reduction::Sum)?.item()?,
            5.0
        );
        assert_eq!(
            ctx.mse_loss(&prediction, &target, Reduction::Mean)?
                .item()?,
            2.5
        );
        assert_eq!(
            ctx.mae_loss(&prediction, &target, Reduction::None)?
                .to_vec()?,
            vec![1.0, 2.0]
        );
        assert_eq!(
            ctx.huber_loss(&prediction, &target, 1.0, Reduction::None)?
                .to_vec()?,
            vec![0.5, 1.5]
        );
        let probabilities = ctx.tensor(vec![0.25, 0.8], &[2])?;
        let binary_target = ctx.tensor(vec![0.0, 1.0], &[2])?;
        let bce = ctx
            .binary_cross_entropy(&probabilities, &binary_target, Reduction::None)?
            .to_vec()?;
        assert!((bce[0] - -(0.75_f32).ln()).abs() <= 1e-6);
        assert!((bce[1] - -(0.8_f32).ln()).abs() <= 1e-6);
        Ok(())
    }

    #[test]
    fn categorical_losses_reduce_rows_and_use_stable_log_sum_exp() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let probabilities = ctx.tensor(vec![0.8, 0.1, 0.1, 0.2, 0.3, 0.5], &[2, 3])?;
        let target = ctx.tensor(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0], &[2, 3])?;
        let none = ctx.cross_entropy(&probabilities, &target, Reduction::None)?;
        assert_eq!(none.shape()?, vec![2]);
        let rows = none.to_vec()?;
        assert!((rows[0] + 0.8_f32.ln()).abs() <= 1e-6);
        assert!((rows[1] + 0.5_f32.ln()).abs() <= 1e-6);
        let logits = ctx.tensor(
            vec![1001.0, 1000.0, 999.0, -999.0, -1000.0, -1001.0],
            &[2, 3],
        )?;
        let fused = ctx.softmax_cross_entropy(&logits, &target, Reduction::None)?;
        assert_eq!(fused.shape()?, vec![2]);
        assert!(fused.to_vec()?.iter().all(|value| value.is_finite()));
        let mean = ctx.softmax_cross_entropy(&logits, &target, Reduction::Mean)?;
        assert_eq!(mean.shape()?, Vec::<usize>::new());
        Ok(())
    }

    fn finite_difference_loss(
        initial: &[f32],
        target_data: &[f32],
        shape: &[usize],
        build: impl Fn(&ExecutionContext, &ContextVariable, &ContextTensor) -> MlResult<ContextVariable>,
        evaluate: impl Fn(&ExecutionContext, &ContextTensor, &ContextTensor) -> MlResult<ContextTensor>,
    ) -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let prediction = ctx.parameter(initial.to_vec(), shape)?;
        let target = ctx.tensor(target_data.to_vec(), shape)?;
        build(&ctx, &prediction, &target)?.backward()?;
        let analytic = prediction.grad()?.expect("prediction gradient").data;
        let epsilon = 1e-3;
        for index in 0..initial.len() {
            let mut plus = initial.to_vec();
            let mut minus = initial.to_vec();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let plus_tensor = ctx.tensor(plus, shape)?;
            let minus_tensor = ctx.tensor(minus, shape)?;
            let numeric = (evaluate(&ctx, &plus_tensor, &target)?.item()?
                - evaluate(&ctx, &minus_tensor, &target)?.item()?)
                / (2.0 * epsilon);
            let absolute = (analytic[index] - numeric).abs();
            let relative = absolute / numeric.abs().max(1e-6);
            assert!(
                absolute <= 1e-3 || relative <= 1e-3,
                "index={index}, analytic={}, numeric={numeric}",
                analytic[index]
            );
        }
        Ok(())
    }

    #[test]
    fn all_context_losses_match_prediction_finite_differences() -> MlResult<()> {
        let prediction = [0.2, 0.7, 0.6, 0.4];
        let target = [0.0, 1.0, 1.0, 0.0];
        finite_difference_loss(
            &prediction,
            &target,
            &[2, 2],
            |ctx, p, t| ctx.mse_loss_variable(p, t, Reduction::Mean),
            |ctx, p, t| ctx.mse_loss(p, t, Reduction::Mean),
        )?;
        finite_difference_loss(
            &prediction,
            &target,
            &[2, 2],
            |ctx, p, t| ctx.mae_loss_variable(p, t, Reduction::Mean),
            |ctx, p, t| ctx.mae_loss(p, t, Reduction::Mean),
        )?;
        finite_difference_loss(
            &prediction,
            &target,
            &[2, 2],
            |ctx, p, t| ctx.huber_loss_variable(p, t, 0.5, Reduction::Mean),
            |ctx, p, t| ctx.huber_loss(p, t, 0.5, Reduction::Mean),
        )?;
        finite_difference_loss(
            &prediction,
            &target,
            &[2, 2],
            |ctx, p, t| ctx.binary_cross_entropy_variable(p, t, Reduction::Mean),
            |ctx, p, t| ctx.binary_cross_entropy(p, t, Reduction::Mean),
        )?;
        finite_difference_loss(
            &prediction,
            &target,
            &[2, 2],
            |ctx, p, t| ctx.cross_entropy_variable(p, t, Reduction::Mean),
            |ctx, p, t| ctx.cross_entropy(p, t, Reduction::Mean),
        )?;
        let logits = [1.2, -0.3, 0.5, 0.8];
        finite_difference_loss(
            &logits,
            &target,
            &[2, 2],
            |ctx, p, t| ctx.softmax_cross_entropy_variable(p, t, Reduction::Mean),
            |ctx, p, t| ctx.softmax_cross_entropy(p, t, Reduction::Mean),
        )
    }

    #[test]
    fn loss_targets_are_not_differentiated_and_invalid_inputs_are_rejected() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let prediction = ctx.parameter(vec![0.4, 0.6], &[2])?;
        let target = ctx.parameter(vec![0.0, 1.0], &[2])?;
        ctx.mse_loss_variable(&prediction, target.tensor(), Reduction::Mean)?
            .backward()?;
        assert!(prediction.grad()?.is_some());
        assert!(target.grad()?.is_none());
        let wrong_shape = ctx.tensor(vec![1.0], &[1])?;
        assert!(
            ctx.mse_loss(prediction.tensor(), &wrong_shape, Reduction::Mean)
                .is_err()
        );
        assert!(
            ctx.huber_loss(prediction.tensor(), target.tensor(), 0.0, Reduction::Mean)
                .is_err()
        );
        let invalid_target = ctx.tensor(vec![0.2, 0.2], &[2])?;
        assert!(
            ctx.softmax_cross_entropy(prediction.tensor(), &invalid_target, Reduction::Mean)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn multidimensional_broadcast_reduces_gradients() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let rows = ctx.parameter(vec![1.0, 2.0], &[2, 1])?;
        let columns = ctx.parameter(vec![10.0, 20.0, 30.0], &[3])?;
        let product = ctx.mul_variable(&rows, &columns)?;
        assert_eq!(product.tensor().shape()?, vec![2, 3]);
        assert_eq!(
            product.tensor().to_vec()?,
            vec![10.0, 20.0, 30.0, 20.0, 40.0, 60.0]
        );
        ctx.sum_variable(&product)?.backward()?;
        assert_eq!(rows.grad()?.expect("row gradient").data, vec![60.0, 60.0]);
        assert_eq!(
            columns.grad()?.expect("column gradient").data,
            vec![3.0, 3.0, 3.0]
        );
        Ok(())
    }

    #[test]
    fn scalar_broadcast_and_incompatible_shapes_are_handled() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let values = ctx.parameter(vec![1.0, 2.0, 3.0], &[3])?;
        let scalar = ctx.parameter(vec![2.0], &[])?;
        let quotient = ctx.div_variable(&values, &scalar)?;
        ctx.sum_variable(&quotient)?.backward()?;
        assert_eq!(values.grad()?.expect("value gradient").data, vec![0.5; 3]);
        assert_eq!(scalar.grad()?.expect("scalar gradient").data, vec![-1.5]);

        let incompatible = ctx.tensor(vec![1.0; 4], &[2, 2])?;
        assert!(matches!(
            ctx.add(values.tensor(), &incompatible),
            Err(crate::MlError::TensorError(TensorError::InvalidOperation {
                op: "add",
                ..
            }))
        ));
        Ok(())
    }

    #[test]
    fn transpose_and_reshape_reverse_the_layout_in_backward() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input = ctx.parameter(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let transposed = ctx.transpose_variable(&input, &[1, 0])?;
        assert_eq!(
            transposed.tensor().to_vec()?,
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
        let flattened = ctx.reshape_variable(&transposed, &[6])?;
        let cotangent = ctx.tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6])?;
        flattened.backward_with_grad(&cotangent)?;
        assert_eq!(
            input.grad()?.expect("input gradient").data,
            vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
        );
        Ok(())
    }

    #[test]
    fn concat_backward_splits_each_outer_block() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let left = ctx.parameter(vec![10.0, 20.0], &[2, 1])?;
        let right = ctx.parameter(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let joined = ctx.concat_variables(&[&left, &right], 1)?;
        assert_eq!(ctx.graph_stats()?.dynamic_backward_nodes, 1);
        assert_eq!(
            joined.tensor().to_vec()?,
            vec![10.0, 1.0, 2.0, 20.0, 3.0, 4.0]
        );
        let cotangent = ctx.tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        joined.backward_with_grad(&cotangent)?;
        assert_eq!(left.grad()?.expect("left gradient").data, vec![1.0, 4.0]);
        assert_eq!(
            right.grad()?.expect("right gradient").data,
            vec![2.0, 3.0, 5.0, 6.0]
        );
        Ok(())
    }

    #[test]
    fn matmul_supports_vector_contracts() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![1.0, 2.0, 3.0], &[3])?;
        let w = ctx.parameter(vec![4.0, 5.0, 6.0], &[3])?;
        let dot = ctx.matmul_variable(&x, &w)?;
        assert_eq!(dot.tensor().shape()?, Vec::<usize>::new());
        assert_eq!(dot.tensor().item()?, 32.0);
        dot.backward()?;
        assert_eq!(x.grad()?.expect("x gradient").data, vec![4.0, 5.0, 6.0]);
        assert_eq!(w.grad()?.expect("w gradient").data, vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn batched_matmul_broadcasts_and_reduces_batch_gradient() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let lhs = ctx.parameter(
            vec![1.0; 4].into_iter().chain(vec![2.0; 4]).collect(),
            &[2, 2, 2],
        )?;
        let rhs = ctx.parameter(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let output = ctx.matmul_variable(&lhs, &rhs)?;
        assert_eq!(ctx.graph_stats()?.dynamic_backward_nodes, 1);
        assert_eq!(output.tensor().shape()?, vec![2, 2, 2]);
        ctx.sum_variable(&output)?.backward()?;
        assert_eq!(
            lhs.grad()?.expect("lhs gradient").data,
            vec![3.0, 7.0, 3.0, 7.0, 3.0, 7.0, 3.0, 7.0]
        );
        assert_eq!(
            rhs.grad()?.expect("rhs gradient").data,
            vec![6.0, 6.0, 6.0, 6.0]
        );
        Ok(())
    }

    #[test]
    fn matrix_vector_matmul_has_expected_gradient() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let matrix = ctx.parameter(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let vector = ctx.parameter(vec![2.0, 3.0, 4.0], &[3])?;
        let output = ctx.matmul_variable(&matrix, &vector)?;
        assert_eq!(output.tensor().to_vec()?, vec![20.0, 47.0]);
        ctx.sum_variable(&output)?.backward()?;
        assert_eq!(
            matrix.grad()?.expect("matrix gradient").data,
            vec![2.0, 3.0, 4.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            vector.grad()?.expect("vector gradient").data,
            vec![5.0, 7.0, 9.0]
        );
        Ok(())
    }

    #[test]
    fn non_leaf_gradients_require_explicit_retention() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![2.0], &[])?;
        let hidden = ctx.square_variable(&x)?;
        let output = ctx.square_variable(&hidden)?;
        output.backward()?;
        assert!(hidden.grad()?.is_none());
        assert_eq!(x.grad()?.expect("leaf gradient").data, vec![32.0]);

        let hidden = ctx.square_variable(&x)?;
        hidden.retain_grad()?;
        let output = ctx.square_variable(&hidden)?;
        output.backward()?;
        assert_eq!(hidden.grad()?.expect("retained gradient").data, vec![8.0]);
        Ok(())
    }

    #[test]
    fn detach_creates_an_untracked_leaf_in_the_same_context() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![3.0], &[])?;
        let connected = ctx.square_variable(&x)?;
        let detached = connected.detach()?;
        assert_eq!(detached.tensor().context_id(), x.tensor().context_id());
        assert_ne!(detached.tensor().node_id(), connected.tensor().node_id());
        assert!(!detached.requires_grad());
        assert_eq!(detached.tensor().item()?, 9.0);
        assert_eq!(ctx.graph_stats()?.graph_nodes, 1);
        Ok(())
    }

    #[test]
    fn graph_nodes_own_dynamic_backward_ops_and_separate_saved_tensors() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![1.0], &[])?;
        let sum = ctx.add_variable(&x, &x)?;
        let product = ctx.mul_variable(&sum, &x)?;
        let _exponential = ctx.exp_variable(&product)?;
        let stats = ctx.graph_stats()?;
        assert_eq!(stats.graph_nodes, 3);
        assert_eq!(stats.dynamic_backward_nodes, 3);
        assert_eq!(stats.saved_tensor_references, 1);
        Ok(())
    }

    #[test]
    fn abs_gradient_matches_finite_difference_away_from_zero() -> MlResult<()> {
        finite_difference_check(-0.7, f32::abs, |ctx, x| ctx.abs_variable(x))
    }

    #[test]
    fn softmax_axis_vjp_matches_finite_difference() -> MlResult<()> {
        let input_data = vec![0.2, -0.4, 1.1, 2.0, 0.3, -0.5];
        let cotangent_data = vec![1.0, -2.0, 0.5, 0.3, 0.7, -1.0];
        let ctx = ExecutionContext::new();
        let input = ctx.parameter(input_data.clone(), &[2, 3])?;
        let output = ctx.softmax_variable(&input, 1)?;
        let probabilities = output.tensor().to_vec()?;
        assert!((probabilities[..3].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((probabilities[3..].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let cotangent = ctx.tensor(cotangent_data.clone(), &[2, 3])?;
        output.backward_with_grad(&cotangent)?;
        let analytic = input.grad()?.expect("softmax input gradient").data;

        let objective = |values: &[f32]| -> f32 {
            values
                .chunks_exact(3)
                .zip(cotangent_data.chunks_exact(3))
                .map(|(row, weights)| {
                    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<_> = row.iter().map(|x| (x - maximum).exp()).collect();
                    let normalizer: f32 = exps.iter().sum();
                    exps.iter()
                        .zip(weights)
                        .map(|(value, weight)| value / normalizer * weight)
                        .sum::<f32>()
                })
                .sum()
        };
        let epsilon = 1e-3;
        for index in 0..input_data.len() {
            let mut plus = input_data.clone();
            let mut minus = input_data.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numeric = (objective(&plus) - objective(&minus)) / (2.0 * epsilon);
            let error = (analytic[index] - numeric).abs();
            assert!(
                error <= 1e-3,
                "index={index}, analytic={}, numeric={numeric}",
                analytic[index]
            );
        }
        Ok(())
    }

    #[test]
    fn conv2d_forward_and_all_gradients_match_finite_difference() -> MlResult<()> {
        let input_data = vec![0.2, -0.4, 0.7, 1.1, -0.3, 0.5, 0.9, -0.8, 0.6];
        let weight_data = vec![0.4, -0.2, 0.3, 0.8];
        let bias_data = vec![0.15];
        let ctx = ExecutionContext::new();
        let input = ctx.parameter(input_data.clone(), &[1, 1, 3, 3])?;
        let weight = ctx.parameter(weight_data.clone(), &[1, 1, 2, 2])?;
        let bias = ctx.parameter(bias_data.clone(), &[1])?;
        let output = ctx.conv2d_variable(&input, &weight, &bias, (1, 1), (0, 0))?;
        assert_eq!(output.tensor().shape()?, vec![1, 1, 2, 2]);
        ctx.sum_variable(&output)?.backward()?;
        let analytic_input = input.grad()?.expect("input gradient").data;
        let analytic_weight = weight.grad()?.expect("weight gradient").data;
        let analytic_bias = bias.grad()?.expect("bias gradient").data;
        let objective = |x: &[f32], w: &[f32], b: &[f32]| -> MlResult<f32> {
            Ok(conv2d_forward_data(
                &GlobalTensor::from_vec(x.to_vec(), &[1, 1, 3, 3])?,
                &GlobalTensor::from_vec(w.to_vec(), &[1, 1, 2, 2])?,
                &GlobalTensor::from_vec(b.to_vec(), &[1])?,
                (1, 1),
                (0, 0),
            )?
            .data
            .iter()
            .sum())
        };
        let epsilon = 1e-3;
        for index in 0..input_data.len() {
            let (mut plus, mut minus) = (input_data.clone(), input_data.clone());
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numeric = (objective(&plus, &weight_data, &bias_data)?
                - objective(&minus, &weight_data, &bias_data)?)
                / (2.0 * epsilon);
            assert!((analytic_input[index] - numeric).abs() <= 1e-3);
        }
        for index in 0..weight_data.len() {
            let (mut plus, mut minus) = (weight_data.clone(), weight_data.clone());
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numeric = (objective(&input_data, &plus, &bias_data)?
                - objective(&input_data, &minus, &bias_data)?)
                / (2.0 * epsilon);
            assert!((analytic_weight[index] - numeric).abs() <= 1e-3);
        }
        let numeric_bias = (objective(&input_data, &weight_data, &[bias_data[0] + epsilon])?
            - objective(&input_data, &weight_data, &[bias_data[0] - epsilon])?)
            / (2.0 * epsilon);
        assert!((analytic_bias[0] - numeric_bias).abs() <= 1e-3);
        Ok(())
    }

    #[test]
    fn conv2d_supports_batch_channels_stride_and_padding() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input = ctx.tensor(vec![1.0; 2 * 2 * 4 * 4], &[2, 2, 4, 4])?;
        let weight = ctx.tensor(vec![1.0; 3 * 2 * 3 * 3], &[3, 2, 3, 3])?;
        let bias = ctx.tensor(vec![1.0, 2.0, 3.0], &[3])?;
        let output = ctx.conv2d(&input, &weight, &bias, (2, 2), (1, 1))?;
        assert_eq!(output.shape()?, vec![2, 3, 2, 2]);
        let data = output.to_vec()?;
        assert_eq!(&data[..4], &[9.0, 13.0, 13.0, 19.0]);
        Ok(())
    }

    #[test]
    fn max_pool2d_uses_saved_mask_and_releases_it_with_the_graph() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input_data = vec![1.0, 4.0, 2.0, 3.0, 8.0, 5.0, 0.0, 6.0, 7.0];
        let input = ctx.parameter(input_data.clone(), &[1, 1, 3, 3])?;
        let output = ctx.max_pool2d_variable(&input, (2, 2), (1, 1))?;
        assert_eq!(output.tensor().to_vec()?, vec![8.0, 8.0, 8.0, 8.0]);
        let stats = ctx.graph_stats()?;
        assert_eq!(stats.saved_tensor_references, 1);
        assert_eq!(stats.tensors, 3);
        ctx.sum_variable(&output)?.backward()?;
        assert_eq!(
            input.grad()?.expect("max pool input gradient").data,
            vec![0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(ctx.graph_stats()?.saved_tensor_references, 0);
        assert_eq!(ctx.graph_stats()?.tensors, 2);

        let objective = |values: &[f32]| -> MlResult<f32> {
            Ok(max_pool2d_forward_data(
                &GlobalTensor::from_vec(values.to_vec(), &[1, 1, 3, 3])?,
                (2, 2),
                (1, 1),
            )?
            .0
            .data
            .iter()
            .sum())
        };
        let epsilon = 1e-3;
        for index in 0..input_data.len() {
            let (mut plus, mut minus) = (input_data.clone(), input_data.clone());
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numeric = (objective(&plus)? - objective(&minus)?) / (2.0 * epsilon);
            let analytic = if index == 4 { 4.0 } else { 0.0 };
            assert!((analytic - numeric).abs() <= 1e-3);
        }
        Ok(())
    }

    #[test]
    fn avg_pool2d_accumulates_overlapping_window_gradients() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input = ctx.parameter((1..=9).map(|value| value as f32).collect(), &[1, 1, 3, 3])?;
        let output = ctx.avg_pool2d_variable(&input, (2, 2), (1, 1))?;
        assert_eq!(output.tensor().to_vec()?, vec![3.0, 4.0, 6.0, 7.0]);
        ctx.sum_variable(&output)?.backward()?;
        assert_eq!(
            input.grad()?.expect("average pool input gradient").data,
            vec![0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25]
        );
        Ok(())
    }

    #[test]
    fn clear_graph_releases_owned_max_pool_mask() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input = ctx.parameter(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
        let output = ctx.max_pool2d_variable(&input, (2, 2), (2, 2))?;
        assert_eq!(ctx.graph_stats()?.tensors, 3);
        ctx.clear_graph()?;
        let stats = ctx.graph_stats()?;
        assert_eq!(stats.tensors, 2);
        assert_eq!(stats.graph_nodes, 0);
        assert_eq!(stats.saved_tensor_references, 0);
        assert_eq!(output.tensor().item()?, 4.0);
        Ok(())
    }

    #[test]
    fn nearest_upsample2d_supports_asymmetric_scale_and_vjp() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = ctx.parameter(input_data.clone(), &[1, 1, 2, 2])?;
        let output = ctx.nearest_upsample2d_variable(&input, (2, 3))?;
        assert_eq!(output.tensor().shape()?, vec![1, 1, 4, 6]);
        assert_eq!(
            output.tensor().to_vec()?,
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0,
                4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
            ]
        );
        let cotangent = ctx.tensor((1..=24).map(|value| value as f32).collect(), &[1, 1, 4, 6])?;
        output.backward_with_grad(&cotangent)?;
        let analytic = input.grad()?.expect("upsample input gradient").data;
        assert_eq!(analytic, vec![30.0, 48.0, 102.0, 120.0]);
        let cotangent_data: Vec<_> = (1..=24).map(|value| value as f32).collect();
        let objective = |values: &[f32]| -> MlResult<f32> {
            let output = nearest_upsample2d_forward_data(
                &GlobalTensor::from_vec(values.to_vec(), &[1, 1, 2, 2])?,
                (2, 3),
            )?;
            Ok(output
                .data
                .iter()
                .zip(&cotangent_data)
                .map(|(x, g)| x * g)
                .sum())
        };
        let epsilon = 1e-3;
        for index in 0..input_data.len() {
            let (mut plus, mut minus) = (input_data.clone(), input_data.clone());
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numeric = (objective(&plus)? - objective(&minus)?) / (2.0 * epsilon);
            let absolute_error = (analytic[index] - numeric).abs();
            let relative_error =
                absolute_error / analytic[index].abs().max(numeric.abs()).max(1e-12);
            assert!(absolute_error <= 1e-3 || relative_error <= 1e-3);
        }
        Ok(())
    }

    #[test]
    fn nearest_upsample2d_rejects_zero_scale() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input = ctx.tensor(vec![1.0], &[1, 1, 1, 1])?;
        assert!(ctx.nearest_upsample2d(&input, (0, 2)).is_err());
        Ok(())
    }

    #[test]
    fn group_norm_all_gradients_match_finite_difference_and_release_saved() -> MlResult<()> {
        let input_data = vec![0.2, -0.7, 1.1, 0.4, -0.3, 0.8, 1.5, -1.2];
        let gamma_data = vec![1.3, -0.6, 0.8, 1.1];
        let beta_data = vec![0.1, -0.2, 0.3, -0.4];
        let cotangent_data = vec![0.5, -0.4, 0.7, 0.2, -0.8, 0.3, 0.6, -0.1];
        let ctx = ExecutionContext::new();
        let input = ctx.parameter(input_data.clone(), &[1, 4, 1, 2])?;
        let gamma = ctx.parameter(gamma_data.clone(), &[4])?;
        let beta = ctx.parameter(beta_data.clone(), &[4])?;
        let output = ctx.group_norm_variable(&input, &gamma, &beta, 2, 1e-3)?;
        let stats = ctx.graph_stats()?;
        assert_eq!(stats.saved_tensor_references, 3);
        assert_eq!(stats.tensors, 7);
        let cotangent = ctx.tensor(cotangent_data.clone(), &[1, 4, 1, 2])?;
        output.backward_with_grad(&cotangent)?;
        let analytic_input = input.grad()?.expect("group norm input gradient").data;
        let analytic_gamma = gamma.grad()?.expect("group norm gamma gradient").data;
        let analytic_beta = beta.grad()?.expect("group norm beta gradient").data;
        let stats = ctx.graph_stats()?;
        assert_eq!(stats.saved_tensor_references, 0);
        assert_eq!(stats.tensors, 5);

        let objective = |x: &[f32], scale: &[f32], shift: &[f32]| -> MlResult<f32> {
            let (output, _) = group_norm_forward_data(
                &GlobalTensor::from_vec(x.to_vec(), &[1, 4, 1, 2])?,
                &GlobalTensor::from_vec(scale.to_vec(), &[4])?,
                &GlobalTensor::from_vec(shift.to_vec(), &[4])?,
                2,
                1e-3,
            )?;
            Ok(output
                .data
                .iter()
                .zip(&cotangent_data)
                .map(|(y, g)| y * g)
                .sum())
        };
        let assert_gradient = |analytic: f32, numeric: f32| {
            let absolute_error = (analytic - numeric).abs();
            let relative_error = absolute_error / analytic.abs().max(numeric.abs()).max(1e-12);
            assert!(
                absolute_error <= 1e-3 || relative_error <= 1e-3,
                "analytic={analytic}, numeric={numeric}, absolute={absolute_error}, relative={relative_error}"
            );
        };
        let epsilon = 1e-3;
        for index in 0..input_data.len() {
            let (mut plus, mut minus) = (input_data.clone(), input_data.clone());
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numeric = (objective(&plus, &gamma_data, &beta_data)?
                - objective(&minus, &gamma_data, &beta_data)?)
                / (2.0 * epsilon);
            assert_gradient(analytic_input[index], numeric);
        }
        for index in 0..gamma_data.len() {
            let (mut plus, mut minus) = (gamma_data.clone(), gamma_data.clone());
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numeric = (objective(&input_data, &plus, &beta_data)?
                - objective(&input_data, &minus, &beta_data)?)
                / (2.0 * epsilon);
            assert_gradient(analytic_gamma[index], numeric);
        }
        for index in 0..beta_data.len() {
            let (mut plus, mut minus) = (beta_data.clone(), beta_data.clone());
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numeric = (objective(&input_data, &gamma_data, &plus)?
                - objective(&input_data, &gamma_data, &minus)?)
                / (2.0 * epsilon);
            assert_gradient(analytic_beta[index], numeric);
        }
        Ok(())
    }

    #[test]
    fn group_norm_validates_groups_and_parameter_shapes() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input = ctx.tensor(vec![1.0; 8], &[1, 4, 1, 2])?;
        let gamma = ctx.tensor(vec![1.0; 4], &[4])?;
        let beta = ctx.tensor(vec![0.0; 4], &[4])?;
        assert!(ctx.group_norm(&input, &gamma, &beta, 0, 1e-5).is_err());
        assert!(ctx.group_norm(&input, &gamma, &beta, 3, 1e-5).is_err());
        let wrong_gamma = ctx.tensor(vec![1.0; 3], &[3])?;
        assert!(
            ctx.group_norm(&input, &wrong_gamma, &beta, 2, 1e-5)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn topk_preserves_tie_order_and_supports_unsorted_output() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input = ctx.tensor(vec![4.0, 2.0, 4.0, 3.0, 1.0, 5.0], &[2, 3])?;
        let sorted = ctx.topk(&input, 2, true)?;
        assert_eq!(sorted.values.shape()?, vec![2, 2]);
        assert_eq!(sorted.values.to_vec()?, vec![4.0, 4.0, 5.0, 3.0]);
        assert_eq!(sorted.indices.to_vec()?, vec![0.0, 2.0, 2.0, 0.0]);
        let unsorted = ctx.topk(&input, 2, false)?;
        assert_eq!(unsorted.values.to_vec()?, vec![4.0, 4.0, 3.0, 5.0]);
        assert_eq!(unsorted.indices.to_vec()?, vec![0.0, 2.0, 0.0, 2.0]);
        Ok(())
    }

    #[test]
    fn matmax_supports_global_axis_and_negative_axis_results() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input = ctx.tensor(vec![1.0, 6.0, 3.0, 4.0, 5.0, 2.0], &[2, 3])?;
        let global = ctx.matmax(&input, None, false)?;
        assert_eq!(global.values.shape()?, Vec::<usize>::new());
        assert_eq!(global.values.item()?, 6.0);
        assert_eq!(global.indices.item()?, 1.0);
        let rows = ctx.matmax(&input, Some(-1), true)?;
        assert_eq!(rows.values.shape()?, vec![2, 1]);
        assert_eq!(rows.values.to_vec()?, vec![6.0, 5.0]);
        assert_eq!(rows.indices.to_vec()?, vec![1.0, 1.0]);
        let columns = ctx.matmax(&input, Some(0), false)?;
        assert_eq!(columns.values.shape()?, vec![3]);
        assert_eq!(columns.values.to_vec()?, vec![4.0, 6.0, 3.0]);
        assert_eq!(columns.indices.to_vec()?, vec![1.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn forward_only_reductions_reject_tracked_inputs_except_in_no_grad() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input = ctx.parameter(vec![1.0, 3.0, 2.0], &[3])?;
        assert!(matches!(
            ctx.topk(input.tensor(), 1, true),
            Err(crate::MlError::AutogradError(
                AutogradError::BackwardNotSupported(_)
            ))
        ));
        assert!(matches!(
            ctx.matmax(input.tensor(), None, false),
            Err(crate::MlError::AutogradError(
                AutogradError::BackwardNotSupported(_)
            ))
        ));
        ctx.no_grad(|| {
            assert_eq!(ctx.topk(input.tensor(), 1, true)?.values.item()?, 3.0);
            assert_eq!(ctx.matmax(input.tensor(), None, false)?.values.item()?, 3.0);
            Ok(())
        })
    }

    #[test]
    fn forward_only_reductions_validate_arguments() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let scalar = ctx.scalar(1.0)?;
        assert!(ctx.topk(&scalar, 1, true).is_err());
        let input = ctx.tensor(vec![1.0, 2.0], &[2])?;
        assert!(ctx.topk(&input, 0, true).is_err());
        assert!(ctx.topk(&input, 3, true).is_err());
        assert!(ctx.matmax(&input, Some(-2), false).is_err());
        Ok(())
    }

    #[test]
    fn untracked_tensor_storage_follows_external_handle_lifetime() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let tensor = ctx.tensor(vec![1.0, 2.0], &[2])?;
        let clone = tensor.clone();
        assert_eq!(ctx.graph_stats()?.tensors, 1);
        drop(tensor);
        assert_eq!(ctx.graph_stats()?.tensors, 1);
        drop(clone);
        assert_eq!(ctx.graph_stats()?.tensors, 0);
        Ok(())
    }

    #[test]
    fn graph_pin_keeps_dropped_input_alive_until_backward_consumes_graph() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input = ctx.parameter(vec![3.0], &[])?;
        let output = ctx.square_variable(&input)?;
        drop(input);
        assert_eq!(ctx.graph_stats()?.tensors, 2);
        output.backward()?;
        assert_eq!(ctx.graph_stats()?.tensors, 1);
        Ok(())
    }

    #[test]
    fn handle_drop_during_context_borrow_is_collected_on_next_entry() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let tensor = ctx.tensor(vec![1.0], &[])?;
        let state_borrow = ctx.runtime.state.borrow();
        drop(tensor);
        assert!(ctx.runtime.gc_pending.get());
        drop(state_borrow);
        assert_eq!(ctx.graph_stats()?.tensors, 0);
        assert!(!ctx.runtime.gc_pending.get());
        Ok(())
    }

    #[test]
    fn no_grad_temporary_loop_has_stable_storage() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input = ctx.parameter(vec![2.0; 16], &[4, 4])?;
        for _ in 0..128 {
            ctx.no_grad(|| {
                let output = ctx.square(input.tensor())?;
                assert_eq!(output.shape()?, vec![4, 4]);
                Ok(())
            })?;
        }
        assert_eq!(ctx.graph_stats()?.tensors, 1);
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
        Ok(())
    }

    #[test]
    fn detach_uses_a_new_node_with_the_same_buffer() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input = ctx.parameter(vec![1.0, 2.0], &[2])?;
        let detached = input.detach()?;
        assert_ne!(input.tensor().node_id(), detached.tensor().node_id());
        {
            let state = ctx.runtime.state.borrow();
            let input_buffer = &state.tensors[&input.tensor().node_id()].buffer;
            let detached_buffer = &state.tensors[&detached.tensor().node_id()].buffer;
            assert!(Rc::ptr_eq(input_buffer, detached_buffer));
            input_buffer.borrow_mut().data[0] = 9.0;
        }
        assert_eq!(detached.tensor().to_vec()?, vec![9.0, 2.0]);
        Ok(())
    }
