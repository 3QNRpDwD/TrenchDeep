//! First into-kernel family: broadcast elementwise and unary activations.
use super::*;
use std::rc::Rc;
#[derive(Debug)]
struct ElementwiseInto {
    op: Operation,
    spec: IntoKernelSpec,
    output_size: usize,
    input_sizes: Vec<usize>,
}
pub(super) fn invalid_into() -> crate::MlError {
    TensorError::InvalidOperation {
        op: "into kernel",
        reason: "shape, arity, destination or workspace mismatch".into(),
    }
    .into()
}
pub(super) fn size(shape: &[usize]) -> MlResult<usize> {
    if shape.contains(&0) {
        return Err(invalid_into());
    }
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|&n| n <= isize::MAX as usize / 4)
        .ok_or_else(invalid_into)
}
pub(super) fn prepare(
    op: &Operation,
    shapes: &[&[usize]],
    training: bool,
) -> MlResult<Option<Rc<dyn IntoKernel>>> {
    if !matches!(
        op,
        Operation::Add
            | Operation::Sub
            | Operation::Mul
            | Operation::Div
            | Operation::Neg
            | Operation::Square
            | Operation::Exp
            | Operation::Log
            | Operation::Sqrt
            | Operation::Tanh
            | Operation::Sigmoid
            | Operation::Silu
            | Operation::Relu
            | Operation::Sin
            | Operation::Cos
            | Operation::Abs
    ) {
        return Ok(None);
    }
    if op.input_count() != Some(shapes.len()) {
        return Err(invalid_into());
    }
    let input_sizes = shapes
        .iter()
        .map(|s| size(s))
        .collect::<MlResult<Vec<_>>>()?;
    let output = if shapes.len() == 2 {
        broadcast_shape(shapes[0], shapes[1]).ok_or_else(invalid_into)?
    } else {
        shapes[0].to_vec()
    };
    let output_size = size(&output)?;
    let vjp = prepared_backward(op, shapes)?.expect("supported elementwise VJP");
    // Broadcast VJP reduction must be completed before adding it to an existing
    // shared destination. Workspace holds one reduced input contribution.
    let workspace_elements = if training && shapes.len() == 2 {
        *input_sizes.iter().max().unwrap()
    } else {
        0
    };
    let spec = IntoKernelSpec {
        inputs: shapes.iter().map(|s| s.to_vec()).collect(),
        output,
        saved: if training {
            vjp.saved_shapes
        } else {
            Vec::new()
        },
        backward_inputs: vjp.inputs,
        workspace_elements,
        training,
    };
    Ok(Some(Rc::new(ElementwiseInto {
        op: op.clone(),
        spec,
        output_size,
        input_sizes,
    })))
}
impl ElementwiseInto {
    fn value(&self, x: f32, y: f32) -> f32 {
        match self.op {
            Operation::Add => x + y,
            Operation::Sub => x - y,
            Operation::Mul => x * y,
            Operation::Div => {
                if y == 0.0 {
                    f32::INFINITY
                } else {
                    x / y
                }
            }
            Operation::Neg => -x,
            Operation::Square => x * x,
            Operation::Exp => {
                if x > 88.0 {
                    f32::INFINITY
                } else if x < -88.0 {
                    0.0
                } else {
                    x.exp()
                }
            }
            Operation::Log => x.ln(),
            Operation::Sqrt => x.sqrt(),
            Operation::Tanh => x.tanh(),
            Operation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Operation::Silu => x / (1.0 + (-x).exp()),
            Operation::Relu => x.max(0.0),
            Operation::Sin => x.sin(),
            Operation::Cos => x.cos(),
            Operation::Abs => x.abs(),
            _ => unreachable!(),
        }
    }
}
impl IntoKernel for ElementwiseInto {
    fn spec(&self) -> &IntoKernelSpec {
        &self.spec
    }
    fn execute_into(
        &self,
        inputs: &[TensorView<'_>],
        output: &mut [f32],
        saved: &mut [&mut [f32]],
        workspace: &mut [f32],
    ) -> MlResult<()> {
        if inputs.len() != self.spec.inputs.len()
            || inputs
                .iter()
                .zip(&self.spec.inputs)
                .any(|(v, s)| v.shape() != s)
            || output.len() != self.output_size
            || saved.len() != self.spec.saved.len()
            || saved.iter().any(|s| s.len() != self.output_size)
            || workspace.len() < self.spec.workspace_elements
        {
            return Err(invalid_into());
        }
        for (i, dst) in output.iter_mut().enumerate() {
            let x = inputs[0].data()[offset(i, &self.spec.output, &self.spec.inputs[0])];
            let y = if inputs.len() == 2 {
                inputs[1].data()[offset(i, &self.spec.output, &self.spec.inputs[1])]
            } else {
                0.0
            };
            *dst = self.value(x, y);
        }
        if let Some(saved) = saved.first_mut() {
            saved.copy_from_slice(output);
        }
        Ok(())
    }
    fn backward_into(
        &self,
        input: usize,
        inputs: &[Option<TensorView<'_>>],
        saved: &[TensorView<'_>],
        gradient: TensorView<'_>,
        destination: &mut [f32],
        write: GradientWrite,
        workspace: &mut [f32],
    ) -> MlResult<()> {
        if !self.spec.training
            || input >= self.input_sizes.len()
            || inputs.len() != self.input_sizes.len()
            || destination.len() != self.input_sizes[input]
            || gradient.shape() != self.spec.output
            || saved.len() != self.spec.saved.len()
            || saved
                .iter()
                .zip(&self.spec.saved)
                .any(|(v, s)| v.shape() != s)
            || workspace.len() < self.spec.workspace_elements
        {
            return Err(invalid_into());
        }
        for ((value, shape), dependency) in inputs
            .iter()
            .zip(&self.spec.inputs)
            .zip(&self.spec.backward_inputs)
        {
            if value.is_some_and(|v| v.shape() != shape)
                || (value.is_none() && *dependency == BackwardInput::Values)
            {
                return Err(invalid_into());
            }
        }
        let binary = inputs.len() == 2;
        let target = if binary {
            &mut workspace[..destination.len()]
        } else {
            &mut *destination
        };
        if binary {
            target.fill(0.0);
        }
        for (i, &g) in gradient.data().iter().enumerate() {
            let x = inputs[0]
                .map(|v| v.data()[offset(i, &self.spec.output, &self.spec.inputs[0])])
                .unwrap_or(0.0);
            let y = if binary {
                inputs[1]
                    .map(|v| v.data()[offset(i, &self.spec.output, &self.spec.inputs[1])])
                    .unwrap_or(0.0)
            } else {
                saved.first().map(|v| v.data()[i]).unwrap_or(0.0)
            };
            let value = match self.op {
                Operation::Add => g,
                Operation::Sub => {
                    if input == 0 {
                        g
                    } else {
                        -g
                    }
                }
                Operation::Mul => g * if input == 0 { y } else { x },
                Operation::Div => {
                    if input == 0 {
                        g / y
                    } else {
                        -g * x / (y * y)
                    }
                }
                Operation::Neg => -g,
                Operation::Square => 2.0 * g * x,
                Operation::Exp => g * y,
                Operation::Log => g / x,
                Operation::Sqrt => g * 0.5 / y,
                Operation::Tanh => g * (1.0 - y * y),
                Operation::Sigmoid => g * y * (1.0 - y),
                Operation::Silu => {
                    let s = 1.0 / (1.0 + (-x).exp());
                    g * s * (1.0 + x * (1.0 - s))
                }
                Operation::Relu => {
                    if x > 0.0 {
                        g
                    } else {
                        0.0
                    }
                }
                Operation::Sin => g * x.cos(),
                Operation::Cos => -g * x.sin(),
                Operation::Abs => {
                    if x > 0.0 {
                        g
                    } else if x < 0.0 {
                        -g
                    } else {
                        0.0
                    }
                }
                _ => unreachable!(),
            };
            let offset = if binary {
                offset(i, &self.spec.output, &self.spec.inputs[input])
            } else {
                i
            };
            if binary || write == GradientWrite::Add {
                target[offset] += value;
            } else {
                target[offset] = value;
            }
        }
        if binary {
            for (dst, &value) in destination.iter_mut().zip(workspace.iter()) {
                if write == GradientWrite::Add {
                    *dst += value;
                } else {
                    *dst = value;
                }
            }
        }
        Ok(())
    }
}

// Direct contiguous/broadcast addressing; no expanded tensor or coordinate Vec.
pub(super) fn offset(mut flat: usize, output: &[usize], input: &[usize]) -> usize {
    let mut result = 0;
    let mut stride = 1;
    let delta = output.len() - input.len();
    for axis in (0..output.len()).rev() {
        let coordinate = flat % output[axis];
        flat /= output[axis];
        if axis >= delta {
            let width = input[axis - delta];
            if width != 1 {
                result += coordinate * stride;
            }
            stride *= width;
        }
    }
    result
}

// Preserve the reduction-then-add order used by eager and shared gradients.
pub(super) fn publish(contribution: &[f32], destination: &mut [f32], write: GradientWrite) {
    match write {
        GradientWrite::Assign => destination.copy_from_slice(contribution),
        GradientWrite::Add => {
            for (dst, &v) in destination.iter_mut().zip(contribution) {
                *dst += v;
            }
        }
    }
}
