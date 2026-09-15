//! Fixed-destination rank>=2 matmul with direct broadcast-batch addressing.
use super::into::{invalid_into, offset, size};
use super::*;
use std::rc::Rc;
#[derive(Debug)]
struct MatmulInto {
    spec: IntoKernelSpec,
    matrix: MatmulSpec,
    sizes: [usize; 2],
    batches: usize,
    output_size: usize,
}
pub(super) fn prepare(
    op: &Operation,
    shapes: &[&[usize]],
    training: bool,
) -> MlResult<Option<Rc<dyn IntoKernel>>> {
    if !matches!(op, Operation::Matmul) {
        return Ok(None);
    }
    if shapes.len() != 2 || shapes.iter().any(|s| s.len() < 2) {
        return Err(invalid_into());
    }
    let sizes = [size(shapes[0])?, size(shapes[1])?];
    let matrix = MatmulSpec::new(shapes[0], shapes[1])?;
    let output_size = size(&matrix.output_shape)?;
    let batches = size(&matrix.batch_shape)?;
    let spec = IntoKernelSpec {
        inputs: shapes.iter().map(|s| s.to_vec()).collect(),
        output: matrix.output_shape.clone(),
        saved: Vec::new(),
        backward_inputs: vec![BackwardInput::Values; 2],
        workspace_elements: if training {
            sizes[0].checked_add(sizes[1]).ok_or_else(invalid_into)?
        } else {
            0
        },
        training,
    };
    Ok(Some(Rc::new(MatmulInto {
        spec,
        matrix,
        sizes,
        batches,
        output_size,
    })))
}
impl IntoKernel for MatmulInto {
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
        if inputs.len() != 2
            || inputs
                .iter()
                .zip(&self.spec.inputs)
                .any(|(v, s)| v.shape() != s)
            || output.len() != self.output_size
            || !saved.is_empty()
            || workspace.len() < self.spec.workspace_elements
        {
            return Err(invalid_into());
        }
        output.fill(0.0);
        let s = &self.matrix;
        let a = inputs[0].data();
        let b = inputs[1].data();
        for batch in 0..self.batches {
            let ab = offset(batch, &s.batch_shape, &s.left_batch) * s.m * s.k;
            let bb = offset(batch, &s.batch_shape, &s.right_batch) * s.k * s.n;
            let cb = batch * s.m * s.n;
            // Same 32-wide reduction order as CpuCompute, reading B directly.
            for i0 in (0..s.m).step_by(32) {
                for p0 in (0..s.k).step_by(32) {
                    for j0 in (0..s.n).step_by(32) {
                        for i in i0..(i0 + 32).min(s.m) {
                            for p in p0..(p0 + 32).min(s.k) {
                                let av = a[ab + i * s.k + p];
                                for j in j0..(j0 + 32).min(s.n) {
                                    output[cb + i * s.n + j] += av * b[bb + p * s.n + j];
                                }
                            }
                        }
                    }
                }
            }
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
        if input >= 2 {
            return Err(invalid_into());
        }
        let mut destinations = [const { None }; 2];
        destinations[input] = Some(destination);
        self.backward_many_into(
            inputs,
            saved,
            gradient,
            &mut destinations,
            &[write; 2],
            workspace,
        )
    }

    fn backward_many_into(
        &self,
        inputs: &[Option<TensorView<'_>>],
        saved: &[TensorView<'_>],
        gradient: TensorView<'_>,
        destinations: &mut [Option<&mut [f32]>],
        writes: &[GradientWrite],
        workspace: &mut [f32],
    ) -> MlResult<()> {
        if !self.spec.training
            || destinations.len() != 2
            || writes.len() != 2
            || inputs.len() != 2
            || inputs
                .iter()
                .zip(&self.spec.inputs)
                .any(|(v, s)| v.is_none_or(|v| v.shape() != s))
            || !saved.is_empty()
            || gradient.shape() != self.spec.output
            || destinations
                .iter()
                .zip(self.sizes)
                .any(|(d, n)| d.as_ref().is_some_and(|d| d.len() != n))
            || workspace.len() < self.spec.workspace_elements
        {
            return Err(invalid_into());
        }
        let left = destinations[0].is_some();
        let right = destinations[1].is_some();
        let (dl, rest) = workspace.split_at_mut(self.sizes[0]);
        let dr = &mut rest[..self.sizes[1]];
        if left {
            dl.fill(0.0);
        }
        if right {
            dr.fill(0.0);
        }
        let s = &self.matrix;
        let a = inputs[0].unwrap().data();
        let b = inputs[1].unwrap().data();
        let g = gradient.data();
        if left || right {
            for batch in 0..self.batches {
                let ab = offset(batch, &s.batch_shape, &s.left_batch) * s.m * s.k;
                let bb = offset(batch, &s.batch_shape, &s.right_batch) * s.k * s.n;
                for i in 0..s.m {
                    for p in 0..s.k {
                        for j in 0..s.n {
                            let upstream = g[(batch * s.m + i) * s.n + j];
                            if left {
                                dl[ab + i * s.k + p] += upstream * b[bb + p * s.n + j];
                            }
                            if right {
                                dr[bb + p * s.n + j] += a[ab + i * s.k + p] * upstream;
                            }
                        }
                    }
                }
            }
        }
        for ((destination, contribution), &write) in
            destinations.iter_mut().zip([dl, dr]).zip(writes)
        {
            if let Some(destination) = destination {
                super::into::publish(contribution, destination, write);
            }
        }
        Ok(())
    }
}
