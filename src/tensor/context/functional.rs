use super::*;

pub(super) fn topk_forward_data(
    input: TensorView<'_>,
    k: usize,
    sorted: bool,
) -> MlResult<(Vec<f32>, Vec<f32>, Vec<usize>)> {
    let rank = input.shape.len();
    if rank == 0 {
        return Err(TensorError::InvalidOperation {
            op: "topk",
            reason: "input must have at least one dimension".into(),
        }.into());
    }
    let width = input.shape[rank - 1];
    if k == 0 || k > width {
        return Err(TensorError::InvalidOperation {
            op: "topk",
            reason: format!("k must be in 1..={width}, got {k}"),
        }.into());
    }
    let compare = |(left_index, left): &(usize, f32), (right_index, right): &(usize, f32)| {
        right.total_cmp(left).then(left_index.cmp(right_index))
    };
    let mut values = Vec::with_capacity(input.data.len() / width * k);
    let mut indices = Vec::with_capacity(values.capacity());
    for row in input.data.chunks_exact(width) {
        let mut pairs: Vec<_> = row.iter().copied().enumerate().collect();
        if k < width {
            pairs.select_nth_unstable_by(k, compare);
            pairs.truncate(k);
        }
        if sorted {
            pairs.sort_unstable_by(compare);
        } else {
            pairs.sort_unstable_by_key(|(index, _)| *index);
        }
        values.extend(pairs.iter().map(|(_, value)| *value));
        indices.extend(pairs.iter().map(|(index, _)| *index as f32));
    }
    let mut shape = input.shape.to_vec();
    shape[rank - 1] = k;
    Ok((values, indices, shape))
}

pub(super) fn matmax_forward_data(
    input: TensorView<'_>,
    axis: Option<isize>,
    keepdim: bool,
) -> MlResult<(Vec<f32>, Vec<f32>, Vec<usize>)> {
    if input.data.is_empty() {
        return Err(TensorError::EmptyTensor.into());
    }
    let Some(requested_axis) = axis else {
        let (index, maximum) = input.data.iter().copied().enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right).then(right_index.cmp(left_index))
            })
            .ok_or(TensorError::EmptyTensor)?;
        return Ok((vec![maximum], vec![index as f32], Vec::new()));
    };
    let rank = input.shape.len();
    let normalized = if requested_axis < 0 {
        requested_axis.checked_add(rank as isize)
    } else {
        Some(requested_axis)
    };
    let axis = normalized
        .filter(|axis| *axis >= 0 && (*axis as usize) < rank)
        .ok_or_else(|| TensorError::InvalidOperation {
            op: "matmax",
            reason: format!("axis {requested_axis} is invalid for shape {:?}", input.shape),
        })? as usize;
    let width = input.shape[axis];
    if width == 0 {
        return Err(TensorError::EmptyTensor.into());
    }
    let inner: usize = input.shape[axis + 1..].iter().product();
    let outer: usize = input.shape[..axis].iter().product();
    let mut values = Vec::with_capacity(input.data.len() / width);
    let mut indices = Vec::with_capacity(values.capacity());
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let mut maximum = input.data[(outer_index * width) * inner + inner_index];
            let mut maximum_index = 0;
            for axis_index in 1..width {
                let value = input.data[(outer_index * width + axis_index) * inner + inner_index];
                if value > maximum {
                    maximum = value;
                    maximum_index = axis_index;
                }
            }
            values.push(maximum);
            indices.push(maximum_index as f32);
        }
    }
    let mut shape = input.shape.to_vec();
    if keepdim { shape[axis] = 1; } else { shape.remove(axis); }
    Ok((values, indices, shape))
}

pub(super) fn validate_approx_threshold(operation: &'static str, threshold: f32) -> MlResult<()> {
    if !threshold.is_finite() || threshold <= 0.0 {
        return Err(TensorError::InvalidOperation {
            op: operation,
            reason: format!("threshold must be finite and positive, got {threshold}"),
        }
        .into());
    }
    Ok(())
}

const CONTEXT_LOSS_EPSILON: f32 = 1e-7;

pub(super) fn validate_loss_pair(
    kind: ContextLossKind,
    prediction: TensorView<'_>,
    target: TensorView<'_>,
) -> MlResult<()> {
    if prediction.shape != target.shape {
        return Err(LossError::InvalidShape {
            expected: prediction.shape.to_vec(),
            got: target.shape.to_vec(),
        }
        .into());
    }
    if prediction.data.is_empty() {
        return Err(LossError::InvalidOperation {
            op: kind.name(),
            reason: "loss input cannot be empty".into(),
        }
        .into());
    }
    if prediction.data.iter().any(|value| !value.is_finite())
        || target.data.iter().any(|value| !value.is_finite())
    {
        return Err(LossError::InvalidOperation {
            op: kind.name(),
            reason: "prediction and target values must be finite".into(),
        }
        .into());
    }
    match kind {
        ContextLossKind::BinaryCrossEntropy => {
            if target.data.iter().any(|value| !(0.0..=1.0).contains(value)) {
                return Err(LossError::InvalidOperation {
                    op: kind.name(),
                    reason: "binary targets must be in [0, 1]".into(),
                }
                .into());
            }
        }
        ContextLossKind::CrossEntropy | ContextLossKind::SoftmaxCrossEntropy => {
            let classes =
                prediction
                    .shape
                    .last()
                    .copied()
                    .ok_or_else(|| LossError::InvalidOperation {
                        op: kind.name(),
                        reason: "categorical loss requires a class axis".into(),
                    })?;
            if classes == 0 {
                return Err(LossError::InvalidOperation {
                    op: kind.name(),
                    reason: "class axis cannot be empty".into(),
                }
                .into());
            }
            for row in target.data.chunks_exact(classes) {
                let ones = row.iter().filter(|value| (**value - 1.0).abs() <= 1e-6).count();
                let binary = row.iter().all(|value| value.abs() <= 1e-6 || (*value - 1.0).abs() <= 1e-6);
                if !binary || ones != 1 {
                    return Err(LossError::InvalidOperation {
                        op: kind.name(),
                        reason: "each categorical target row must be one-hot encoded".into(),
                    }
                    .into());
                }
            }
            if matches!(kind, ContextLossKind::CrossEntropy)
                && prediction
                    .data
                    .iter()
                    .any(|value| *value < 0.0 || *value > 1.0)
            {
                return Err(LossError::InvalidOperation {
                    op: kind.name(),
                    reason: "probabilities must be in [0, 1]".into(),
                }
                .into());
            }
        }
        _ => {}
    }
    Ok(())
}

pub(super) fn reduce_loss_values(
    values: Vec<f32>,
    none_shape: &[usize],
    reduction: Reduction,
) -> MlResult<GlobalTensor<f32>> {
    match reduction {
        Reduction::None => GlobalTensor::from_vec(values, none_shape),
        Reduction::Sum => GlobalTensor::from_vec(vec![values.iter().sum()], &[]),
        Reduction::Mean => {
            let count = values.len();
            if count == 0 {
                return Err(LossError::InvalidOperation {
                    op: "loss_reduction",
                    reason: "mean reduction requires at least one loss element".into(),
                }
                .into());
            }
            GlobalTensor::from_vec(vec![values.iter().sum::<f32>() / count as f32], &[])
        }
    }
}

pub(super) fn loss_forward(
    kind: ContextLossKind,
    reduction: Reduction,
    prediction: TensorView<'_>,
    target: TensorView<'_>,
    save_softmax: bool,
) -> MlResult<(GlobalTensor<f32>, Option<GlobalTensor<f32>>)> {
    validate_loss_pair(kind, prediction, target)?;
    let mut saved = None;
    let output = match kind {
        ContextLossKind::Mse => reduce_loss_values(
            prediction
                .data
                .iter()
                .zip(target.data)
                .map(|(p, t)| (p - t).powi(2))
                .collect(),
            prediction.shape,
            reduction,
        ),
        ContextLossKind::Mae => reduce_loss_values(
            prediction
                .data
                .iter()
                .zip(target.data)
                .map(|(p, t)| (p - t).abs())
                .collect(),
            prediction.shape,
            reduction,
        ),
        ContextLossKind::Huber { delta } => reduce_loss_values(
            prediction
                .data
                .iter()
                .zip(target.data)
                .map(|(p, t)| {
                    let difference = (p - t).abs();
                    if difference <= delta {
                        0.5 * difference * difference
                    } else {
                        delta * (difference - 0.5 * delta)
                    }
                })
                .collect(),
            prediction.shape,
            reduction,
        ),
        ContextLossKind::BinaryCrossEntropy => reduce_loss_values(
            prediction
                .data
                .iter()
                .zip(target.data)
                .map(|(p, t)| {
                    let probability = p.clamp(CONTEXT_LOSS_EPSILON, 1.0 - CONTEXT_LOSS_EPSILON);
                    -(t * probability.ln() + (1.0 - t) * (1.0 - probability).ln())
                })
                .collect(),
            prediction.shape,
            reduction,
        ),
        ContextLossKind::CrossEntropy | ContextLossKind::SoftmaxCrossEntropy => {
            let classes = prediction.shape[prediction.shape.len() - 1];
            let mut losses = Vec::with_capacity(prediction.data.len() / classes);
            let mut probabilities = (save_softmax
                && matches!(kind, ContextLossKind::SoftmaxCrossEntropy))
                .then(|| Vec::with_capacity(prediction.data.len()));
            for (prediction_row, target_row) in prediction
                .data
                .chunks_exact(classes)
                .zip(target.data.chunks_exact(classes))
            {
                let loss = if matches!(kind, ContextLossKind::CrossEntropy) {
                    -prediction_row
                        .iter()
                        .zip(target_row)
                        .map(|(p, t)| t * p.max(CONTEXT_LOSS_EPSILON).ln())
                        .sum::<f32>()
                } else {
                    let maximum = prediction_row
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let denominator = prediction_row
                        .iter()
                        .map(|value| (value - maximum).exp())
                        .sum::<f32>();
                    let log_sum_exp = maximum + denominator.ln();
                    if let Some(probabilities) = &mut probabilities {
                        probabilities.extend(
                            prediction_row
                                .iter()
                                .map(|value| (value - maximum).exp() / denominator),
                        );
                    }
                    log_sum_exp
                        - prediction_row
                            .iter()
                            .zip(target_row)
                            .map(|(logit, t)| logit * t)
                            .sum::<f32>()
                };
                losses.push(loss);
            }
            let mut shape = prediction.shape.to_vec();
            shape.pop();
            if let Some(probabilities) = probabilities {
                saved = Some(GlobalTensor::from_vec(probabilities, prediction.shape)?);
            }
            reduce_loss_values(losses, &shape, reduction)
        }
    }?;
    Ok((output, saved))
}

pub(super) fn loss_backward(
    kind: &ContextLossKind,
    reduction: Reduction,
    prediction: TensorView<'_>,
    target: TensorView<'_>,
    output_gradient: TensorView<'_>,
    saved: Option<TensorView<'_>>,
) -> MlResult<GlobalTensor<f32>> {
    validate_loss_pair(*kind, prediction, target)?;
    let categorical = matches!(
        kind,
        ContextLossKind::CrossEntropy | ContextLossKind::SoftmaxCrossEntropy
    );
    let classes = if categorical {
        prediction.shape[prediction.shape.len() - 1]
    } else {
        1
    };
    let loss_count = if categorical {
        prediction.data.len() / classes
    } else {
        prediction.data.len()
    };
    let scale_for = |element: usize| match reduction {
        Reduction::None => output_gradient.data[element / classes],
        Reduction::Sum => output_gradient.data[0],
        Reduction::Mean => output_gradient.data[0] / loss_count as f32,
    };
    let mut gradient = vec![0.0; prediction.data.len()];
    match kind {
        ContextLossKind::Mse => {
            for index in 0..gradient.len() {
                gradient[index] =
                    scale_for(index) * 2.0 * (prediction.data[index] - target.data[index]);
            }
        }
        ContextLossKind::Mae => {
            for index in 0..gradient.len() {
                gradient[index] =
                    scale_for(index) * (prediction.data[index] - target.data[index]).signum();
            }
        }
        ContextLossKind::Huber { delta } => {
            for index in 0..gradient.len() {
                let difference = prediction.data[index] - target.data[index];
                let derivative = if difference.abs() <= *delta {
                    difference
                } else {
                    *delta * difference.signum()
                };
                gradient[index] = scale_for(index) * derivative;
            }
        }
        ContextLossKind::BinaryCrossEntropy => {
            for index in 0..gradient.len() {
                let p =
                    prediction.data[index].clamp(CONTEXT_LOSS_EPSILON, 1.0 - CONTEXT_LOSS_EPSILON);
                gradient[index] = scale_for(index) * (p - target.data[index]) / (p * (1.0 - p));
            }
        }
        ContextLossKind::CrossEntropy => {
            for index in 0..gradient.len() {
                let p = prediction.data[index].max(CONTEXT_LOSS_EPSILON);
                gradient[index] = scale_for(index) * -target.data[index] / p;
            }
        }
        ContextLossKind::SoftmaxCrossEntropy => {
            let probabilities = saved.ok_or(AutogradError::BackwardArityMismatch {
                expected: 1,
                got: 0,
            })?;
            for row in 0..loss_count {
                let start = row * classes;
                for class in 0..classes {
                    gradient[start + class] =
                        scale_for(start + class) * (probabilities.data[start + class] - target.data[start + class]);
                }
            }
        }
    }
    GlobalTensor::from_vec(gradient, prediction.shape)
}

pub(super) fn approx_sin_value(x: f32) -> f32 {
    let mut result = x;
    let mut sign = -1.0;
    let mut power = 3_u32;
    let mut x_power = x * x * x;
    let mut factorial = 6.0;
    while power <= 15 {
        let term = sign * x_power / factorial;
        result += term;
        sign = -sign;
        x_power *= x * x;
        factorial *= (power + 1) as f32 * (power + 2) as f32;
        power += 2;
    }
    result
}

pub(super) fn approx_cos_value(x: f32) -> f32 {
    let x_squared = x * x;
    let mut result = 1.0;
    let mut sign = -1.0;
    let mut power = 2_u32;
    let mut x_power = x_squared;
    let mut factorial = 2.0;
    while power <= 14 {
        let term = sign * x_power / factorial;
        result += term;
        sign = -sign;
        x_power *= x_squared;
        factorial *= (power + 1) as f32 * (power + 2) as f32;
        power += 2;
    }
    result
}

pub(super) fn approx_sin_derivative(x: f32) -> f32 {
    approx_cos_value(x)
}

pub(super) fn approx_cos_derivative(x: f32) -> f32 {
    let x_squared = x * x;
    let mut result = -x;
    let mut sign = 1.0;
    let mut power = 3_u32;
    let mut x_power = x * x_squared;
    let mut factorial = 6.0;
    while power <= 13 {
        result += sign * x_power / factorial;
        sign = -sign;
        x_power *= x_squared;
        factorial *= (power + 1) as f32 * (power + 2) as f32;
        power += 2;
    }
    result
}

