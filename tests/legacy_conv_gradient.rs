#![cfg(all(
    feature = "legacyBenchmark",
    feature = "builtinStorage",
    feature = "builtinKernels"
))]
use old::tensor::{
    TensorBase,
    operators::{Conv2dOp, Function},
};
use trench_deep::{legacy as old, *};

#[test]
fn corrected_legacy_conv_input_gradient_matches_context_and_finite_difference()
-> Result<(), Box<dyn std::error::Error>> {
    let ctx = ExecutionContext::new();
    let x = ctx.parameter(vec![0.5, 0.7], &[1, 2, 1, 1])?;
    let w = ctx.tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2, 1, 1])?;
    let b = ctx.tensor(vec![0.0, 0.0], &[2])?;
    let y = x.variable().conv2d(&w, &b, (1, 1), (0, 0))?;
    y.sum()?.backward()?;
    assert_eq!(x.grad()?.ok_or("gradient")?.data(), &[4.0, 6.0]);
    let ox = old::tensor::GlobalTensor::from_vec(vec![0.5, 0.7], &[1, 2, 1, 1])?;
    let ow = old::tensor::GlobalTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2, 1, 1])?;
    let ob = old::tensor::GlobalTensor::from_vec(vec![0.0, 0.0], &[2])?;
    let one = old::tensor::GlobalTensor::from_vec(vec![1.0], &[1, 1])?;
    let zero = old::tensor::GlobalTensor::from_vec(vec![0.0], &[1, 1])?;
    let grad = old::tensor::GlobalTensor::from_vec(vec![1.0, 1.0], &[1, 2, 1, 1])?;
    let op = Conv2dOp::new()?;
    let inputs: [&dyn TensorBase; 7] = [&ox, &ow, &ob, &one, &one, &zero, &zero];
    let original = op.forward(&inputs)?;
    assert_eq!(y.tensor().to_vec()?, original[0].data());
    let gradients = op.backward(&inputs, &grad)?;
    assert_eq!(gradients[0].data(), &[4.0, 6.0]);
    // Independent finite difference of the unchanged baseline forward sum.
    for index in 0..2 {
        let evaluate = |delta: f32| -> old::MlResult<f32> {
            let mut values = vec![0.5, 0.7];
            values[index] += delta;
            let input = old::tensor::GlobalTensor::from_vec(values, &[1, 2, 1, 1])?;
            Ok(
                op.forward(&[&input, &ow, &ob, &one, &one, &zero, &zero])?[0]
                    .data()
                    .iter()
                    .sum(),
            )
        };
        let derivative = (evaluate(0.001)? - evaluate(-0.001)?) / 0.002;
        assert!((derivative - [4.0, 6.0][index]).abs() < 0.001);
    }
    Ok(())
}
