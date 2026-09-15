//! Softmax, nearest upsample, and the currently prepared mean loss contracts.
use super::into::{invalid_into, size};
use super::*;
use std::rc::Rc;
#[derive(Debug)]
struct AuxInto {
    op: Operation,
    spec: IntoKernelSpec,
    input_size: usize,
    output_size: usize,
    outer: usize,
    width: usize,
    inner: usize,
}
pub(super) fn prepare(
    op: &Operation,
    shapes: &[&[usize]],
    training: bool,
) -> MlResult<Option<Rc<dyn IntoKernel>>> {
    if !matches!(
        op,
        Operation::Softmax { .. }
            | Operation::NearestUpsample2d { .. }
            | Operation::Loss {
                kind: LossKind::Mse | LossKind::Mae | LossKind::BinaryCrossEntropy,
                reduction: Reduction::Mean
            }
    ) {
        return Ok(None);
    }
    if op.input_count() != Some(shapes.len()) {
        return Err(invalid_into());
    }
    for shape in shapes {
        size(shape)?;
    }
    let input_size = size(shapes[0])?;
    let mut outer = 1;
    let mut width = 1;
    let mut inner = 1;
    let output = match op {
        Operation::Softmax { axis } => {
            if *axis >= shapes[0].len() {
                return Err(invalid_into());
            }
            outer = shapes[0][..*axis].iter().product();
            width = shapes[0][*axis];
            inner = shapes[0][*axis + 1..].iter().product();
            shapes[0].to_vec()
        }
        Operation::NearestUpsample2d { scale } => {
            let (oh, ow) = nearest_upsample2d_spec(shapes[0], *scale)?;
            vec![shapes[0][0], shapes[0][1], oh, ow]
        }
        Operation::Loss { .. } => {
            if shapes[0] != shapes[1] {
                return Err(invalid_into());
            }
            vec![]
        }
        _ => unreachable!(),
    };
    let output_size = size(&output)?;
    let saved = if training && matches!(op, Operation::Softmax { .. }) {
        vec![output.clone()]
    } else {
        vec![]
    };
    let spec = IntoKernelSpec {
        inputs: shapes.iter().map(|s| s.to_vec()).collect(),
        output,
        saved,
        backward_inputs: vec![
            if matches!(op, Operation::Loss { .. }) {
                BackwardInput::Values
            } else {
                BackwardInput::Shape
            };
            shapes.len()
        ],
        workspace_elements: if training { input_size } else { 0 },
        training,
    };
    Ok(Some(Rc::new(AuxInto {
        op: op.clone(),
        spec,
        input_size,
        output_size,
        outer,
        width,
        inner,
    })))
}
impl AuxInto {
    fn source(&self, i: usize) -> usize {
        let Operation::NearestUpsample2d { scale } = &self.op else {
            unreachable!()
        };
        let oh = self.spec.output[2];
        let ow = self.spec.output[3];
        let h = self.spec.inputs[0][2];
        let w = self.spec.inputs[0][3];
        (i / (oh * ow)) * h * w + ((i / ow) % oh / scale.0) * w + (i % ow) / scale.1
    }
}
impl IntoKernel for AuxInto {
    fn spec(&self) -> &IntoKernelSpec {
        &self.spec
    }
    fn execute_into(
        &self,
        inputs: &[TensorView<'_>],
        out: &mut [f32],
        saved: &mut [&mut [f32]],
        work: &mut [f32],
    ) -> MlResult<()> {
        if inputs.len() != self.spec.inputs.len()
            || inputs
                .iter()
                .zip(&self.spec.inputs)
                .any(|(v, s)| v.shape() != s)
            || out.len() != self.output_size
            || saved.len() != self.spec.saved.len()
            || saved.iter().any(|s| s.len() != self.output_size)
            || work.len() < self.spec.workspace_elements
        {
            return Err(invalid_into());
        }
        let x = inputs[0].data();
        match self.op {
            Operation::Softmax { .. } => {
                for o in 0..self.outer {
                    for i in 0..self.inner {
                        let max = (0..self.width)
                            .map(|j| x[(o * self.width + j) * self.inner + i])
                            .fold(f32::NEG_INFINITY, f32::max);
                        let sum: f32 = (0..self.width)
                            .map(|j| (x[(o * self.width + j) * self.inner + i] - max).exp())
                            .sum();
                        for j in 0..self.width {
                            let idx = (o * self.width + j) * self.inner + i;
                            out[idx] = (x[idx] - max).exp() / sum;
                        }
                    }
                }
                if let Some(saved) = saved.first_mut() {
                    saved.copy_from_slice(out);
                }
            }
            Operation::NearestUpsample2d { .. } => {
                for (i, v) in out.iter_mut().enumerate() {
                    *v = x[self.source(i)];
                }
            }
            Operation::Loss { kind, .. } => {
                validate_loss_pair(kind, inputs[0], inputs[1])?;
                let loss: f32 = x
                    .iter()
                    .zip(inputs[1].data())
                    .map(|(&p, &t)| match kind {
                        LossKind::Mse => (p - t).powi(2),
                        LossKind::Mae => (p - t).abs(),
                        LossKind::BinaryCrossEntropy => {
                            let p = p.clamp(1e-7, 1.0 - 1e-7);
                            -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
                        }
                        _ => unreachable!(),
                    })
                    .sum();
                out[0] = loss / self.input_size as f32;
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
        dst: &mut [f32],
        write: GradientWrite,
        work: &mut [f32],
    ) -> MlResult<()> {
        // The target of a loss is never differentiable.
        if !self.spec.training
            || input != 0
            || inputs.len() != self.spec.inputs.len()
            || inputs
                .iter()
                .zip(&self.spec.inputs)
                .zip(&self.spec.backward_inputs)
                .any(|((v, s), d)| {
                    v.is_some_and(|v| v.shape() != s)
                        || (*d == BackwardInput::Values && v.is_none())
                })
            || saved.len() != self.spec.saved.len()
            || saved
                .iter()
                .zip(&self.spec.saved)
                .any(|(v, s)| v.shape() != s)
            || gradient.shape() != self.spec.output
            || dst.len() != self.input_size
            || work.len() < self.spec.workspace_elements
        {
            return Err(invalid_into());
        }
        if let Operation::Loss { kind, .. } = self.op {
            validate_loss_pair(kind, inputs[0].unwrap(), inputs[1].unwrap())?;
        }
        let tmp = &mut work[..self.input_size];
        let g = gradient.data();
        match self.op {
            Operation::Softmax { .. } => {
                let y = saved[0].data();
                for o in 0..self.outer {
                    for i in 0..self.inner {
                        let dot: f32 = (0..self.width)
                            .map(|j| {
                                let idx = (o * self.width + j) * self.inner + i;
                                g[idx] * y[idx]
                            })
                            .sum();
                        for j in 0..self.width {
                            let idx = (o * self.width + j) * self.inner + i;
                            tmp[idx] = y[idx] * (g[idx] - dot);
                        }
                    }
                }
            }
            Operation::NearestUpsample2d { .. } => {
                tmp.fill(0.0);
                for (i, &g) in g.iter().enumerate() {
                    tmp[self.source(i)] += g;
                }
            }
            Operation::Loss { kind, .. } => {
                let scale = g[0] / self.input_size as f32;
                for ((v, &p), &t) in tmp
                    .iter_mut()
                    .zip(inputs[0].unwrap().data())
                    .zip(inputs[1].unwrap().data())
                {
                    *v = match kind {
                        LossKind::Mse => scale * 2.0 * (p - t),
                        LossKind::Mae => {
                            let d = p - t;
                            scale * if d == 0.0 { 0.0 } else { d.signum() }
                        }
                        LossKind::BinaryCrossEntropy => {
                            let p = p.clamp(1e-7, 1.0 - 1e-7);
                            scale * (p - t) / (p * (1.0 - p))
                        }
                        _ => unreachable!(),
                    };
                }
            }
            _ => unreachable!(),
        }
        for (d, &v) in dst.iter_mut().zip(tmp.iter()) {
            if write == GradientWrite::Assign {
                *d = v
            } else {
                *d += v
            }
        }
        Ok(())
    }
}
