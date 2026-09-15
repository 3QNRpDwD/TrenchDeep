//! Out-of-place structural kernels. All indexing metadata is fixed at prepare.
use super::into::{invalid_into, size};
use super::*;
use std::rc::Rc;
#[derive(Debug)]
struct StructuralInto {
    op: Operation,
    spec: IntoKernelSpec,
    sizes: Vec<usize>,
    output_size: usize,
    strides: Vec<usize>,
    offsets: Vec<usize>,
    inner: usize,
}
pub(super) fn prepare(
    op: &Operation,
    shapes: &[&[usize]],
    training: bool,
) -> MlResult<Option<Rc<dyn IntoKernel>>> {
    if !matches!(
        op,
        Operation::Reshape(_) | Operation::Transpose(_) | Operation::Concat { .. } | Operation::Sum
    ) {
        return Ok(None);
    }
    if shapes.is_empty() || op.input_count().is_some_and(|n| n != shapes.len()) {
        return Err(invalid_into());
    }
    let sizes = shapes
        .iter()
        .map(|s| size(s))
        .collect::<MlResult<Vec<_>>>()?;
    let x = shapes[0];
    let mut strides = vec![1; x.len()];
    for i in (0..x.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * x[i + 1];
    }
    let mut offsets = Vec::new();
    let mut inner = 1;
    let output = match op {
        Operation::Reshape(shape) => {
            if size(shape)? != sizes[0] {
                return Err(invalid_into());
            }
            shape.clone()
        }
        Operation::Transpose(axes) => {
            if axes.len() != x.len()
                || axes
                    .iter()
                    .enumerate()
                    .any(|(i, &a)| a >= x.len() || axes[..i].contains(&a))
            {
                return Err(invalid_into());
            }
            axes.iter().map(|&a| x[a]).collect()
        }
        Operation::Concat { axis } => {
            if *axis >= x.len()
                || shapes.iter().any(|s| {
                    s.len() != x.len()
                        || s.iter()
                            .zip(x)
                            .enumerate()
                            .any(|(i, (a, b))| i != *axis && a != b)
                })
            {
                return Err(invalid_into());
            }
            let mut output = x.to_vec();
            let mut total = 0usize;
            for shape in shapes {
                offsets.push(total);
                total = total.checked_add(shape[*axis]).ok_or_else(invalid_into)?;
            }
            output[*axis] = total;
            inner = x[*axis + 1..].iter().product();
            output
        }
        Operation::Sum => Vec::new(),
        _ => unreachable!(),
    };
    let output_size = size(&output)?;
    let spec = IntoKernelSpec {
        inputs: shapes.iter().map(|s| s.to_vec()).collect(),
        output,
        saved: Vec::new(),
        backward_inputs: vec![BackwardInput::Shape; shapes.len()],
        workspace_elements: 0,
        training,
    };
    Ok(Some(Rc::new(StructuralInto {
        op: op.clone(),
        spec,
        sizes,
        output_size,
        strides,
        offsets,
        inner,
    })))
}
impl StructuralInto {
    fn transpose_source(&self, mut flat: usize) -> usize {
        let Operation::Transpose(axes) = &self.op else {
            unreachable!()
        };
        let mut source = 0;
        for axis in (0..axes.len()).rev() {
            source += (flat % self.spec.output[axis]) * self.strides[axes[axis]];
            flat /= self.spec.output[axis];
        }
        source
    }
    fn concat_destination(&self, input: usize, flat: usize) -> usize {
        let Operation::Concat { axis } = &self.op else {
            unreachable!()
        };
        let block = self.spec.inputs[input][*axis] * self.inner;
        (flat / block) * self.spec.output[*axis] * self.inner
            + self.offsets[input] * self.inner
            + flat % block
    }
}
impl IntoKernel for StructuralInto {
    fn spec(&self) -> &IntoKernelSpec {
        &self.spec
    }
    fn execute_into(
        &self,
        inputs: &[TensorView<'_>],
        output: &mut [f32],
        saved: &mut [&mut [f32]],
        _workspace: &mut [f32],
    ) -> MlResult<()> {
        if inputs.len() != self.sizes.len()
            || inputs
                .iter()
                .zip(&self.spec.inputs)
                .any(|(v, s)| v.shape() != s)
            || output.len() != self.output_size
            || !saved.is_empty()
        {
            return Err(invalid_into());
        }
        match &self.op {
            Operation::Reshape(_) => output.copy_from_slice(inputs[0].data()),
            Operation::Transpose(_) => {
                for (i, dst) in output.iter_mut().enumerate() {
                    *dst = inputs[0].data()[self.transpose_source(i)];
                }
            }
            Operation::Concat { .. } => {
                for (index, input) in inputs.iter().enumerate() {
                    for (i, &value) in input.data().iter().enumerate() {
                        output[self.concat_destination(index, i)] = value;
                    }
                }
            }
            Operation::Sum => {
                // Preserve CpuCompute::sum's eight-element grouping exactly.
                let mut sum = 0.0;
                let mut chunks = inputs[0].data().chunks_exact(8);
                for x in &mut chunks {
                    sum += x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7];
                }
                for &x in chunks.remainder() {
                    sum += x;
                }
                output[0] = sum;
            }
            _ => unreachable!(),
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
        _workspace: &mut [f32],
    ) -> MlResult<()> {
        if !self.spec.training
            || input >= self.sizes.len()
            || inputs.len() != self.sizes.len()
            || inputs
                .iter()
                .zip(&self.spec.inputs)
                .any(|(v, s)| v.is_some_and(|v| v.shape() != s))
            || !saved.is_empty()
            || gradient.shape() != self.spec.output
            || destination.len() != self.sizes[input]
        {
            return Err(invalid_into());
        }
        let mut put = |i: usize, v: f32| {
            if write == GradientWrite::Add {
                destination[i] += v
            } else {
                destination[i] = v
            }
        };
        match &self.op {
            Operation::Reshape(_) => {
                for (i, &g) in gradient.data().iter().enumerate() {
                    put(i, g);
                }
            }
            Operation::Transpose(_) => {
                for (i, &g) in gradient.data().iter().enumerate() {
                    put(self.transpose_source(i), g);
                }
            }
            Operation::Concat { .. } => {
                for i in 0..self.sizes[input] {
                    put(i, gradient.data()[self.concat_destination(input, i)]);
                }
            }
            Operation::Sum => {
                for i in 0..self.sizes[0] {
                    put(i, gradient.data()[0]);
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    }
}
