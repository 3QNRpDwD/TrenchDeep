use super::*;

pub(super) fn group_norm_spec(
    input: &[usize],
    gamma: &[usize],
    beta: &[usize],
    groups: usize,
    epsilon: f32,
) -> MlResult<(usize, usize, usize, usize, usize, usize)> {
    let channels = input.get(1).copied().unwrap_or_default();
    let group_size = if groups == 0 { 0 } else { channels / groups };
    let elements_per_group = input
        .get(2)
        .and_then(|height| height.checked_mul(*input.get(3)?))
        .and_then(|spatial| spatial.checked_mul(group_size));
    if input.len() != 4
        || groups == 0
        || channels == 0
        || channels % groups != 0
        || gamma != [channels]
        || beta != [channels]
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || elements_per_group.is_none_or(|elements| elements == 0)
    {
        return Err(TensorError::InvalidOperation {
            op: "group_norm",
            reason: format!(
                "expected input [N,C,H,W], gamma/beta [C], C divisible by non-zero groups, and finite positive epsilon; got input={input:?}, gamma={gamma:?}, beta={beta:?}, groups={groups}, epsilon={epsilon}"
            ),
        }
        .into());
    }
    Ok((
        input[0],
        channels,
        input[2],
        input[3],
        group_size,
        elements_per_group.unwrap_or_default(),
    ))
}

pub(super) fn group_norm_forward_data(
    input: &GlobalTensor<f32>,
    gamma: &GlobalTensor<f32>,
    beta: &GlobalTensor<f32>,
    groups: usize,
    epsilon: f32,
) -> MlResult<(GlobalTensor<f32>, Vec<GlobalTensor<f32>>)> {
    let (n, c, h, w, channels_per_group, elements_per_group) =
        group_norm_spec(&input.shape, &gamma.shape, &beta.shape, groups, epsilon)?;
    let mut output = vec![0.0; input.data.len()];
    let mut normalized = vec![0.0; input.data.len()];
    let mut means = vec![0.0; n * groups];
    let mut variances = vec![0.0; n * groups];
    for batch in 0..n {
        for group in 0..groups {
            let channel_start = group * channels_per_group;
            let statistic_index = batch * groups + group;
            let mut sum = 0.0;
            for channel in channel_start..channel_start + channels_per_group {
                let base = (batch * c + channel) * h * w;
                sum += input.data[base..base + h * w].iter().sum::<f32>();
            }
            let mean = sum / elements_per_group as f32;
            means[statistic_index] = mean;
            let mut squared_deviation = 0.0;
            for channel in channel_start..channel_start + channels_per_group {
                let base = (batch * c + channel) * h * w;
                squared_deviation += input.data[base..base + h * w]
                    .iter()
                    .map(|value| (value - mean) * (value - mean))
                    .sum::<f32>();
            }
            let variance = squared_deviation / elements_per_group as f32;
            variances[statistic_index] = variance;
            let inverse_std = 1.0 / (variance + epsilon).sqrt();
            for channel in channel_start..channel_start + channels_per_group {
                let base = (batch * c + channel) * h * w;
                for offset in 0..h * w {
                    let index = base + offset;
                    normalized[index] = (input.data[index] - mean) * inverse_std;
                    output[index] = gamma.data[channel] * normalized[index] + beta.data[channel];
                }
            }
        }
    }
    Ok((
        GlobalTensor::from_vec(output, &input.shape)?,
        vec![
            GlobalTensor::from_vec(normalized, &input.shape)?,
            GlobalTensor::from_vec(means, &[n, groups])?,
            GlobalTensor::from_vec(variances, &[n, groups])?,
        ],
    ))
}

pub(super) fn group_norm_backward_data(
    input: &GlobalTensor<f32>,
    gamma: &GlobalTensor<f32>,
    saved: &[TensorView<'_>],
    grad: &GlobalTensor<f32>,
    groups: usize,
    epsilon: f32,
) -> MlResult<(GlobalTensor<f32>, GlobalTensor<f32>, GlobalTensor<f32>)> {
    let beta_shape = [gamma.shape.first().copied().unwrap_or_default()];
    let (n, c, h, w, channels_per_group, elements_per_group) =
        group_norm_spec(&input.shape, &gamma.shape, &beta_shape, groups, epsilon)?;
    if saved.len() != 3 {
        return Err(AutogradError::BackwardArityMismatch {
            expected: 3,
            got: saved.len(),
        }
        .into());
    }
    let expected_statistics = [n, groups];
    if grad.shape != input.shape
        || saved[0].shape != input.shape
        || saved[1].shape != expected_statistics
        || saved[2].shape != expected_statistics
    {
        return Err(TensorError::InvalidOperation {
            op: "group_norm_backward",
            reason: "gradient or saved tensor shape does not match the forward contract".into(),
        }
        .into());
    }
    let normalized = saved[0].data;
    let variances = saved[2].data;
    let mut dx = vec![0.0; input.data.len()];
    let mut dgamma = vec![0.0; c];
    let mut dbeta = vec![0.0; c];
    for batch in 0..n {
        for channel in 0..c {
            let base = (batch * c + channel) * h * w;
            for offset in 0..h * w {
                let index = base + offset;
                dgamma[channel] += grad.data[index] * normalized[index];
                dbeta[channel] += grad.data[index];
            }
        }
        for group in 0..groups {
            let channel_start = group * channels_per_group;
            let mut sum_scaled_grad = 0.0;
            let mut sum_scaled_grad_normalized = 0.0;
            for channel in channel_start..channel_start + channels_per_group {
                let base = (batch * c + channel) * h * w;
                for offset in 0..h * w {
                    let index = base + offset;
                    let scaled_grad = grad.data[index] * gamma.data[channel];
                    sum_scaled_grad += scaled_grad;
                    sum_scaled_grad_normalized += scaled_grad * normalized[index];
                }
            }
            let inverse_std = 1.0 / (variances[batch * groups + group] + epsilon).sqrt();
            let count = elements_per_group as f32;
            for channel in channel_start..channel_start + channels_per_group {
                let base = (batch * c + channel) * h * w;
                for offset in 0..h * w {
                    let index = base + offset;
                    let scaled_grad = grad.data[index] * gamma.data[channel];
                    dx[index] = inverse_std / count
                        * (count * scaled_grad
                            - sum_scaled_grad
                            - normalized[index] * sum_scaled_grad_normalized);
                }
            }
        }
    }
    Ok((
        GlobalTensor::from_vec(dx, &input.shape)?,
        GlobalTensor::from_vec(dgamma, &gamma.shape)?,
        GlobalTensor::from_vec(dbeta, &beta_shape)?,
    ))
}

pub(super) fn nearest_upsample2d_spec(input: &[usize], scale: (usize, usize)) -> MlResult<(usize, usize)> {
    let output_height = input.get(2).and_then(|height| height.checked_mul(scale.0));
    let output_width = input.get(3).and_then(|width| width.checked_mul(scale.1));
    if input.len() != 4
        || scale.0 == 0
        || scale.1 == 0
        || output_height.is_none()
        || output_width.is_none()
    {
        return Err(TensorError::InvalidOperation {
            op: "nearest_upsample2d",
            reason: format!(
                "expected input [N,C,H,W] with non-zero scales and representable output dimensions; got {input:?}, scale={scale:?}"
            ),
        }
        .into());
    }
    Ok((
        output_height.unwrap_or_default(),
        output_width.unwrap_or_default(),
    ))
}

pub(super) fn nearest_upsample2d_forward_data(
    input: &GlobalTensor<f32>,
    scale: (usize, usize),
) -> MlResult<GlobalTensor<f32>> {
    let (oh, ow) = nearest_upsample2d_spec(&input.shape, scale)?;
    let (n, c, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let mut output = vec![0.0; n * c * oh * ow];
    for batch in 0..n {
        for channel in 0..c {
            for y in 0..oh {
                for x in 0..ow {
                    let input_index = ((batch * c + channel) * h + y / scale.0) * w + x / scale.1;
                    let output_index = ((batch * c + channel) * oh + y) * ow + x;
                    output[output_index] = input.data[input_index];
                }
            }
        }
    }
    GlobalTensor::from_vec(output, &[n, c, oh, ow])
}

pub(super) fn nearest_upsample2d_backward_data(
    input: &GlobalTensor<f32>,
    grad: &GlobalTensor<f32>,
    scale: (usize, usize),
) -> MlResult<GlobalTensor<f32>> {
    let (oh, ow) = nearest_upsample2d_spec(&input.shape, scale)?;
    let expected = vec![input.shape[0], input.shape[1], oh, ow];
    if grad.shape != expected {
        return Err(AutogradError::GradientShapeMismatch {
            expected,
            got: grad.shape.clone(),
        }
        .into());
    }
    let (n, c, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let mut dx = vec![0.0; input.data.len()];
    for batch in 0..n {
        for channel in 0..c {
            for y in 0..oh {
                for x in 0..ow {
                    let input_index = ((batch * c + channel) * h + y / scale.0) * w + x / scale.1;
                    let output_index = ((batch * c + channel) * oh + y) * ow + x;
                    dx[input_index] += grad.data[output_index];
                }
            }
        }
    }
    GlobalTensor::from_vec(dx, &input.shape)
}

pub(super) fn pool2d_spec(
    input: &[usize],
    kernel: (usize, usize),
    stride: (usize, usize),
    op: &'static str,
) -> MlResult<(usize, usize)> {
    if input.len() != 4
        || kernel.0 == 0
        || kernel.1 == 0
        || stride.0 == 0
        || stride.1 == 0
        || input.get(2).is_none_or(|height| *height < kernel.0)
        || input.get(3).is_none_or(|width| *width < kernel.1)
    {
        return Err(TensorError::InvalidOperation {
            op,
            reason: format!(
                "expected input [N,C,H,W] with non-zero kernel/stride fitting the input; got {input:?}, kernel={kernel:?}, stride={stride:?}"
            ),
        }
        .into());
    }
    Ok((
        (input[2] - kernel.0) / stride.0 + 1,
        (input[3] - kernel.1) / stride.1 + 1,
    ))
}

pub(super) fn max_pool2d_forward_data(
    input: &GlobalTensor<f32>,
    kernel: (usize, usize),
    stride: (usize, usize),
) -> MlResult<(GlobalTensor<f32>, GlobalTensor<f32>)> {
    let (oh, ow) = pool2d_spec(&input.shape, kernel, stride, "max_pool2d")?;
    let (n, c, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let mut output = vec![f32::NEG_INFINITY; n * c * oh * ow];
    let mut mask = vec![0.0; output.len()];
    for batch in 0..n {
        for channel in 0..c {
            for y in 0..oh {
                for x in 0..ow {
                    let output_index = ((batch * c + channel) * oh + y) * ow + x;
                    for ky in 0..kernel.0 {
                        for kx in 0..kernel.1 {
                            let input_index = ((batch * c + channel) * h + y * stride.0 + ky) * w
                                + x * stride.1
                                + kx;
                            if input.data[input_index] > output[output_index] {
                                output[output_index] = input.data[input_index];
                                mask[output_index] = input_index as f32;
                            }
                        }
                    }
                }
            }
        }
    }
    let shape = [n, c, oh, ow];
    Ok((
        GlobalTensor::from_vec(output, &shape)?,
        GlobalTensor::from_vec(mask, &shape)?,
    ))
}

pub(super) fn max_pool2d_backward_data(
    input: &GlobalTensor<f32>,
    mask: &TensorView<'_>,
    grad: &GlobalTensor<f32>,
    kernel: (usize, usize),
    stride: (usize, usize),
) -> MlResult<GlobalTensor<f32>> {
    let (oh, ow) = pool2d_spec(&input.shape, kernel, stride, "max_pool2d")?;
    let expected = vec![input.shape[0], input.shape[1], oh, ow];
    if grad.shape != expected || mask.shape != expected {
        return Err(AutogradError::GradientShapeMismatch {
            expected,
            got: grad.shape.clone(),
        }
        .into());
    }
    let mut dx = vec![0.0; input.data.len()];
    for (&upstream, &saved_index) in grad.data.iter().zip(mask.data) {
        let index = saved_index as usize;
        if !saved_index.is_finite() || saved_index < 0.0 || index >= dx.len() {
            return Err(TensorError::InvalidOperation {
                op: "max_pool2d_backward",
                reason: "saved maximum index is invalid".into(),
            }
            .into());
        }
        dx[index] += upstream;
    }
    GlobalTensor::from_vec(dx, &input.shape)
}

pub(super) fn avg_pool2d_forward_data(
    input: &GlobalTensor<f32>,
    kernel: (usize, usize),
    stride: (usize, usize),
) -> MlResult<GlobalTensor<f32>> {
    let (oh, ow) = pool2d_spec(&input.shape, kernel, stride, "avg_pool2d")?;
    let (n, c, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let mut output = vec![0.0; n * c * oh * ow];
    let area = (kernel.0 * kernel.1) as f32;
    for batch in 0..n {
        for channel in 0..c {
            for y in 0..oh {
                for x in 0..ow {
                    let output_index = ((batch * c + channel) * oh + y) * ow + x;
                    for ky in 0..kernel.0 {
                        for kx in 0..kernel.1 {
                            let input_index = ((batch * c + channel) * h + y * stride.0 + ky) * w
                                + x * stride.1
                                + kx;
                            output[output_index] += input.data[input_index] / area;
                        }
                    }
                }
            }
        }
    }
    GlobalTensor::from_vec(output, &[n, c, oh, ow])
}

pub(super) fn avg_pool2d_backward_data(
    input: &GlobalTensor<f32>,
    grad: &GlobalTensor<f32>,
    kernel: (usize, usize),
    stride: (usize, usize),
) -> MlResult<GlobalTensor<f32>> {
    let (oh, ow) = pool2d_spec(&input.shape, kernel, stride, "avg_pool2d")?;
    let expected = vec![input.shape[0], input.shape[1], oh, ow];
    if grad.shape != expected {
        return Err(AutogradError::GradientShapeMismatch {
            expected,
            got: grad.shape.clone(),
        }
        .into());
    }
    let (n, c, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let mut dx = vec![0.0; input.data.len()];
    let area = (kernel.0 * kernel.1) as f32;
    for batch in 0..n {
        for channel in 0..c {
            for y in 0..oh {
                for x in 0..ow {
                    let upstream = grad.data[((batch * c + channel) * oh + y) * ow + x] / area;
                    for ky in 0..kernel.0 {
                        for kx in 0..kernel.1 {
                            let input_index = ((batch * c + channel) * h + y * stride.0 + ky) * w
                                + x * stride.1
                                + kx;
                            dx[input_index] += upstream;
                        }
                    }
                }
            }
        }
    }
    GlobalTensor::from_vec(dx, &input.shape)
}

pub(super) fn conv2d_spec(
    input: &[usize],
    weight: &[usize],
    bias: &[usize],
    stride: (usize, usize),
    padding: (usize, usize),
) -> MlResult<(usize, usize)> {
    let padded_height = padding
        .0
        .checked_mul(2)
        .and_then(|padding| input.get(2)?.checked_add(padding));
    let padded_width = padding
        .1
        .checked_mul(2)
        .and_then(|padding| input.get(3)?.checked_add(padding));
    if input.len() != 4
        || weight.len() != 4
        || bias.len() != 1
        || input[1] != weight[1]
        || bias[0] != weight[0]
        || stride.0 == 0
        || stride.1 == 0
        || padded_height.is_none_or(|height| height < weight[2])
        || padded_width.is_none_or(|width| width < weight[3])
    {
        return Err(TensorError::InvalidOperation {
            op: "conv2d",
            reason: format!("expected input [N,C,H,W], weight [O,C,kH,kW], bias [O]; got {input:?}, {weight:?}, {bias:?}"),
        }.into());
    }
    Ok((
        (padded_height.unwrap_or_default() - weight[2]) / stride.0 + 1,
        (padded_width.unwrap_or_default() - weight[3]) / stride.1 + 1,
    ))
}

pub(super) fn conv2d_forward_data(
    input: &GlobalTensor<f32>,
    weight: &GlobalTensor<f32>,
    bias: &GlobalTensor<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> MlResult<GlobalTensor<f32>> {
    let (oh, ow) = conv2d_spec(&input.shape, &weight.shape, &bias.shape, stride, padding)?;
    let (n, ci, h, w, co, kh, kw) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
        weight.shape[0],
        weight.shape[2],
        weight.shape[3],
    );
    let mut output = vec![0.0; n * co * oh * ow];
    for b in 0..n {
        for oc in 0..co {
            for y in 0..oh {
                for x in 0..ow {
                    let mut sum = bias.data[oc];
                    for ic in 0..ci {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = y * stride.0 + ky;
                                let ix = x * stride.1 + kx;
                                if iy >= padding.0 && ix >= padding.1 {
                                    let sy = iy - padding.0;
                                    let sx = ix - padding.1;
                                    if sy < h && sx < w {
                                        sum += input.data[((b * ci + ic) * h + sy) * w + sx]
                                            * weight.data[((oc * ci + ic) * kh + ky) * kw + kx];
                                    }
                                }
                            }
                        }
                    }
                    output[((b * co + oc) * oh + y) * ow + x] = sum;
                }
            }
        }
    }
    GlobalTensor::from_vec(output, &[n, co, oh, ow])
}

pub(super) fn conv2d_backward_data(
    input: &GlobalTensor<f32>,
    weight: &GlobalTensor<f32>,
    grad: &GlobalTensor<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> MlResult<(GlobalTensor<f32>, GlobalTensor<f32>, GlobalTensor<f32>)> {
    let bias_shape = [weight.shape[0]];
    let (oh, ow) = conv2d_spec(&input.shape, &weight.shape, &bias_shape, stride, padding)?;
    let expected = vec![input.shape[0], weight.shape[0], oh, ow];
    if grad.shape != expected {
        return Err(AutogradError::GradientShapeMismatch {
            expected,
            got: grad.shape.clone(),
        }
        .into());
    }
    let (n, ci, h, w, co, kh, kw) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
        weight.shape[0],
        weight.shape[2],
        weight.shape[3],
    );
    let mut dx = vec![0.0; input.data.len()];
    let mut dw = vec![0.0; weight.data.len()];
    let mut db = vec![0.0; co];
    for b in 0..n {
        for oc in 0..co {
            for y in 0..oh {
                for x in 0..ow {
                    let upstream = grad.data[((b * co + oc) * oh + y) * ow + x];
                    db[oc] += upstream;
                    for ic in 0..ci {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = y * stride.0 + ky;
                                let ix = x * stride.1 + kx;
                                if iy >= padding.0 && ix >= padding.1 {
                                    let sy = iy - padding.0;
                                    let sx = ix - padding.1;
                                    if sy < h && sx < w {
                                        let input_index = ((b * ci + ic) * h + sy) * w + sx;
                                        let weight_index = ((oc * ci + ic) * kh + ky) * kw + kx;
                                        dx[input_index] += upstream * weight.data[weight_index];
                                        dw[weight_index] += upstream * input.data[input_index];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((
        GlobalTensor::from_vec(dx, &input.shape)?,
        GlobalTensor::from_vec(dw, &weight.shape)?,
        GlobalTensor::from_vec(db, &bias_shape)?,
    ))
}

pub(super) struct MatmulSpec {
    pub(super) left_batch: Vec<usize>,
    pub(super) right_batch: Vec<usize>,
    pub(super) batch_shape: Vec<usize>,
    pub(super) output_shape: Vec<usize>,
    pub(super) m: usize,
    pub(super) k: usize,
    pub(super) n: usize,
}

impl MatmulSpec {
    pub(super) fn new(left: &[usize], right: &[usize]) -> MlResult<Self> {
        if left.is_empty() || right.is_empty() {
            return Err(TensorError::MatrixMultiplicationError {
                left_shape: left.to_vec(),
                right_shape: right.to_vec(),
            }
            .into());
        }
        let left_vector = left.len() == 1;
        let right_vector = right.len() == 1;
        let (m, k) = if left_vector {
            (1, left[0])
        } else {
            (left[left.len() - 2], left[left.len() - 1])
        };
        let (right_k, n) = if right_vector {
            (right[0], 1)
        } else {
            (right[right.len() - 2], right[right.len() - 1])
        };
        let left_batch = if left_vector {
            vec![]
        } else {
            left[..left.len() - 2].to_vec()
        };
        let right_batch = if right_vector {
            vec![]
        } else {
            right[..right.len() - 2].to_vec()
        };
        let batch_shape = broadcast_shape(&left_batch, &right_batch).ok_or_else(|| {
            TensorError::MatrixMultiplicationError {
                left_shape: left.to_vec(),
                right_shape: right.to_vec(),
            }
        })?;
        if k != right_k {
            return Err(TensorError::MatrixMultiplicationError {
                left_shape: left.to_vec(),
                right_shape: right.to_vec(),
            }
            .into());
        }
        let mut output_shape = batch_shape.clone();
        if !left_vector {
            output_shape.push(m);
        }
        if !right_vector {
            output_shape.push(n);
        }
        Ok(Self {
            left_batch,
            right_batch,
            batch_shape,
            output_shape,
            m,
            k,
            n,
        })
    }
}

pub(super) fn validate_permutation(shape: &[usize], axes: &[usize]) -> MlResult<()> {
    if axes.len() != shape.len() {
        return Err(TensorError::InvalidOperation {
            op: "transpose",
            reason: "axis count must equal rank".into(),
        }
        .into());
    }
    let mut seen = vec![false; shape.len()];
    for &axis in axes {
        if axis >= shape.len() || seen[axis] {
            return Err(TensorError::InvalidAxis {
                axis,
                shape: shape.to_vec(),
            }
            .into());
        }
        seen[axis] = true;
    }
    Ok(())
}

pub(super) fn permute_data(data: &[f32], input_shape: &[usize], axes: &[usize]) -> Vec<f32> {
    let output_shape: Vec<_> = axes.iter().map(|&axis| input_shape[axis]).collect();
    let mut output = vec![0.0; data.len()];
    for output_flat in 0..output.len() {
        let mut remainder = output_flat;
        let mut input_coordinates = vec![0; input_shape.len()];
        for output_axis in (0..output_shape.len()).rev() {
            let coordinate = remainder % output_shape[output_axis];
            remainder /= output_shape[output_axis];
            input_coordinates[axes[output_axis]] = coordinate;
        }
        let input_flat = input_coordinates
            .iter()
            .zip(input_shape)
            .fold(0, |flat, (&coordinate, &dim)| flat * dim + coordinate);
        output[output_flat] = data[input_flat];
    }
    output
}

pub(super) fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Option<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    let mut output = vec![1; rank];
    for offset in 0..rank {
        let left = lhs
            .len()
            .checked_sub(offset + 1)
            .map(|i| lhs[i])
            .unwrap_or(1);
        let right = rhs
            .len()
            .checked_sub(offset + 1)
            .map(|i| rhs[i])
            .unwrap_or(1);
        if left != right && left != 1 && right != 1 {
            return None;
        }
        output[rank - offset - 1] = left.max(right);
    }
    Some(output)
}

pub(super) fn broadcast_offset(flat: usize, output_shape: &[usize], input_shape: &[usize]) -> usize {
    let rank_delta = output_shape.len() - input_shape.len();
    let mut remainder = flat;
    let mut coordinates = vec![0; output_shape.len()];
    for axis in (0..output_shape.len()).rev() {
        coordinates[axis] = remainder % output_shape[axis];
        remainder /= output_shape[axis];
    }
    let mut input_offset = 0;
    for (axis, &dim) in input_shape.iter().enumerate() {
        let coordinate = if dim == 1 {
            0
        } else {
            coordinates[axis + rank_delta]
        };
        input_offset = input_offset * dim + coordinate;
    }
    input_offset
}

pub(super) fn broadcast_data(input: &GlobalTensor<f32>, output_shape: &[usize]) -> MlResult<Vec<f32>> {
    if broadcast_shape(&input.shape, output_shape).as_deref() != Some(output_shape) {
        return Err(TensorError::InvalidOperation {
            op: "broadcast",
            reason: format!("cannot broadcast {:?} to {:?}", input.shape, output_shape),
        }
        .into());
    }
    let length: usize = output_shape.iter().product();
    Ok((0..length)
        .map(|flat| input.data[broadcast_offset(flat, output_shape, &input.shape)])
        .collect())
}

pub(super) fn reduce_to_shape(
    input: &GlobalTensor<f32>,
    target_shape: &[usize],
) -> MlResult<GlobalTensor<f32>> {
    if broadcast_shape(target_shape, &input.shape).as_deref() != Some(input.shape.as_slice()) {
        return Err(AutogradError::GradientShapeMismatch {
            expected: target_shape.to_vec(),
            got: input.shape.clone(),
        }
        .into());
    }
    let target_length: usize = target_shape.iter().product();
    let mut data = vec![0.0; target_length];
    for (flat, value) in input.data.iter().copied().enumerate() {
        data[broadcast_offset(flat, &input.shape, target_shape)] += value;
    }
    GlobalTensor::from_vec(data, target_shape)
}

pub(super) fn tensor_map(tensor: &GlobalTensor<f32>, f: impl Fn(f32) -> f32) -> MlResult<GlobalTensor<f32>> {
    GlobalTensor::from_vec(tensor.data.iter().copied().map(f).collect(), &tensor.shape)
}

pub(super) fn tensor_zip(
    lhs: &GlobalTensor<f32>,
    rhs: &GlobalTensor<f32>,
    f: impl Fn(f32, f32) -> f32,
) -> MlResult<GlobalTensor<f32>> {
    if lhs.shape != rhs.shape {
        return Err(AutogradError::GradientShapeMismatch {
            expected: rhs.shape.clone(),
            got: lhs.shape.clone(),
        }
        .into());
    }
    GlobalTensor::from_vec(
        lhs.data
            .iter()
            .copied()
            .zip(rhs.data.iter().copied())
            .map(|(a, b)| f(a, b))
            .collect(),
        &lhs.shape,
    )
}

