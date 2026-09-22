use super::*;
pub(super) fn prepared_backward(
    op: &Operation,
    shapes: &[&[usize]],
) -> MlResult<Option<PreparedBackward>> {
    use BackwardInput::{Shape, Values};
    if shapes.is_empty() || op.input_count().is_some_and(|n| n != shapes.len()) {
        return Err(TensorError::InvalidOperation {
            op: "prepare backward",
            reason: "invalid arity".into(),
        }
        .into());
    }
    let mut saved_shapes = Vec::new();
    let mut inputs = vec![Values; shapes.len()];
    let mut differentiable = vec![true; shapes.len()];
    if matches!(op, Operation::Concat {axis} if shapes.iter().any(|s| *axis >= s.len()))
        || matches!(op, Operation::GroupNorm {..} if shapes[0].len() != 4)
    {
        return Err(TensorError::InvalidOperation {
            op: "prepare backward",
            reason: "invalid input rank or axis".into(),
        }
        .into());
    }
    let backward = match op {
        Operation::Add => BuiltinBackward::Add,
        Operation::Sub => BuiltinBackward::Sub,
        Operation::Mul => BuiltinBackward::Mul,
        Operation::Div => BuiltinBackward::Div,
        Operation::Neg => BuiltinBackward::Neg,
        Operation::Square => BuiltinBackward::Square,
        Operation::Exp => BuiltinBackward::Exp,
        Operation::Log => BuiltinBackward::Log,
        Operation::Sqrt => BuiltinBackward::Sqrt,
        Operation::Relu => BuiltinBackward::Relu,
        Operation::Tanh => BuiltinBackward::Tanh,
        Operation::Sigmoid => BuiltinBackward::Sigmoid,
        Operation::Silu => BuiltinBackward::Silu,
        Operation::Sin => BuiltinBackward::Sin,
        Operation::Cos => BuiltinBackward::Cos,
        Operation::Abs => BuiltinBackward::Abs,
        Operation::Reshape(_) => BuiltinBackward::Reshape,
        Operation::Transpose(axes) => BuiltinBackward::Transpose(axes.clone()),
        Operation::Concat { axis } => BuiltinBackward::Concat {
            axis: *axis,
            sizes: shapes.iter().map(|s| s[*axis]).collect(),
        },
        Operation::Sum => BuiltinBackward::Sum,
        Operation::Matmul => BuiltinBackward::Matmul,
        Operation::Softmax { axis } => BuiltinBackward::Softmax { axis: *axis },
        Operation::Conv2d { stride, padding } => {
            inputs[2] = Shape;
            BuiltinBackward::Conv2d {
                stride: *stride,
                padding: *padding,
            }
        }
        Operation::GroupNorm { groups, epsilon } => {
            inputs = vec![Shape, Values, Shape];
            saved_shapes = vec![
                shapes[0].to_vec(),
                vec![shapes[0][0], *groups],
                vec![shapes[0][0], *groups],
            ];
            BuiltinBackward::GroupNorm {
                groups: *groups,
                epsilon: *epsilon,
            }
        }
        Operation::NearestUpsample2d { scale } => {
            BuiltinBackward::NearestUpsample2d { scale: *scale }
        }
        Operation::Loss { kind, reduction } => {
            differentiable[1] = false;
            if matches!(kind, LossKind::SoftmaxCrossEntropy) {
                saved_shapes.push(shapes[0].to_vec());
            }
            BuiltinBackward::Loss {
                kind: *kind,
                reduction: *reduction,
            }
        }
        _ => return Ok(None),
    };
    if matches!(
        op,
        Operation::Add
            | Operation::Sub
            | Operation::Neg
            | Operation::Sum
            | Operation::Reshape(_)
            | Operation::Transpose(_)
            | Operation::Concat { .. }
            | Operation::NearestUpsample2d { .. }
    ) {
        inputs.fill(Shape);
    }
    if matches!(
        op,
        Operation::Exp
            | Operation::Sqrt
            | Operation::Tanh
            | Operation::Sigmoid
            | Operation::Softmax { .. }
    ) {
        inputs.fill(Shape);
        saved_shapes.push(shapes[0].to_vec());
    }
    Ok(Some(PreparedBackward {
        operation: std::rc::Rc::from(into_node_backward(backward)),
        inputs,
        differentiable,
        saved_shapes,
    }))
}
