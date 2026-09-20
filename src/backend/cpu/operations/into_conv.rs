//! Spatial-row forward/dx and bounded patch tiles for dw. Reduction order is
//! preserved per destination; padding terms are skipped, never multiplied by zero.
use super::into::{invalid_into, size};
use super::*;
use std::rc::Rc;
const PATCH_TILE: usize = 16;
#[derive(Debug)]
struct Span {
    begin: usize,
    end: usize,
    input: usize,
}
fn spans(kernel: usize, input: usize, output: usize, stride: usize, padding: usize) -> Vec<Span> {
    (0..kernel)
        .map(|k| {
            let begin = padding.saturating_sub(k).div_ceil(stride).min(output);
            let end = (input + padding)
                .saturating_sub(k)
                .div_ceil(stride)
                .min(output);
            Span {
                begin,
                end,
                input: if begin < end {
                    begin * stride + k - padding
                } else {
                    0
                },
            }
        })
        .collect()
}
#[derive(Debug)]
struct PatchRegion {
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
}
#[derive(Debug)]
struct ConvInto {
    spec: IntoKernelSpec,
    sizes: [usize; 3],
    output_size: usize,
    n: usize,
    ci: usize,
    h: usize,
    w: usize,
    co: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    rows: Vec<Span>,
    columns: Vec<Span>,
    patches: Vec<PatchRegion>,
    spatial: usize,
    patch_width: usize,
}
pub(super) fn prepare(
    op: &Operation,
    shapes: &[&[usize]],
    training: bool,
) -> MlResult<Option<Rc<dyn IntoKernel>>> {
    Ok(build(op, shapes, training, false)?.map(|kernel| Rc::new(kernel) as Rc<dyn IntoKernel>))
}
pub(super) fn forward_data(
    input: &TensorBuffer,
    weight: &TensorBuffer,
    bias: &TensorBuffer,
    stride: (usize, usize),
    padding: (usize, usize),
) -> MlResult<TensorBuffer> {
    let kernel = build(
        &Operation::Conv2d { stride, padding },
        &[input.shape(), weight.shape(), bias.shape()],
        false,
        true,
    )?
    .unwrap();
    let mut output = vec![0.0; kernel.output_size];
    kernel.forward_rows(input.data(), weight.data(), bias.data(), &mut output);
    TensorBuffer::from_vec(output, &kernel.spec.output)
}
pub(super) fn backward_data(
    input: &TensorBuffer,
    weight: &TensorBuffer,
    grad: &TensorBuffer,
    stride: (usize, usize),
    padding: (usize, usize),
) -> MlResult<(TensorBuffer, TensorBuffer, TensorBuffer)> {
    let bias_shape = [weight.shape()[0]];
    let kernel = build(
        &Operation::Conv2d { stride, padding },
        &[input.shape(), weight.shape(), &bias_shape],
        true,
        true,
    )?
    .unwrap();
    if grad.shape() != kernel.spec.output {
        return Err(AutogradError::GradientShapeMismatch {
            expected: kernel.spec.output.clone(),
            got: grad.shape().to_vec(),
        }
        .into());
    }
    let mut dx = vec![0.0; kernel.sizes[0]];
    let mut dw = vec![0.0; kernel.sizes[1]];
    let mut db = vec![0.0; kernel.sizes[2]];
    let mut tile = vec![0.0; kernel.patch_width * kernel.spatial.min(PATCH_TILE)];
    kernel.input_gradient(grad.data(), weight.data(), &mut dx);
    kernel.weight_gradient(input.data(), grad.data(), &mut dw, &mut tile);
    for (i, &g) in grad.data().iter().enumerate() {
        db[(i / kernel.spatial) % kernel.co] += g;
    }
    Ok((
        TensorBuffer::from_vec(dx, input.shape())?,
        TensorBuffer::from_vec(dw, weight.shape())?,
        TensorBuffer::from_vec(db, &bias_shape)?,
    ))
}
fn build(
    op: &Operation,
    shapes: &[&[usize]],
    training: bool,
    allow_empty: bool,
) -> MlResult<Option<ConvInto>> {
    let Operation::Conv2d { stride, padding } = op else {
        return Ok(None);
    };
    if shapes.len() != 3 {
        return Err(invalid_into());
    }
    // Eager historically accepts empty dimensions; prepared keeps its nonempty contract.
    let count = |shape: &[usize]| {
        if allow_empty {
            shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .filter(|&n| n <= isize::MAX as usize / 4)
                .ok_or_else(invalid_into)
        } else {
            size(shape)
        }
    };
    let sizes = [count(shapes[0])?, count(shapes[1])?, count(shapes[2])?];
    let (oh, ow) = conv2d_spec(shapes[0], shapes[1], shapes[2], *stride, *padding)?;
    let (n, ci, h, w, co, kh, kw) = (
        shapes[0][0],
        shapes[0][1],
        shapes[0][2],
        shapes[0][3],
        shapes[1][0],
        shapes[1][2],
        shapes[1][3],
    );
    let output = vec![n, co, oh, ow];
    let output_size = count(&output)?;
    let spatial = oh.checked_mul(ow).ok_or_else(invalid_into)?;
    let patch_width = count(&[ci, kh, kw])?;
    let tile_elements = patch_width
        .checked_mul(spatial.min(PATCH_TILE))
        .ok_or_else(invalid_into)?;
    let spec = IntoKernelSpec {
        inputs: shapes.iter().map(|s| s.to_vec()).collect(),
        output,
        saved: Vec::new(),
        backward_inputs: vec![
            BackwardInput::Values,
            BackwardInput::Values,
            BackwardInput::Shape,
        ],
        workspace_elements: if training {
            sizes
                .iter()
                .try_fold(tile_elements, |a, &b| a.checked_add(b))
                .ok_or_else(invalid_into)?
        } else {
            0
        },
        training,
    };
    Ok(Some(ConvInto {
        spec,
        sizes,
        output_size,
        n,
        ci,
        h,
        w,
        co,
        kh,
        kw,
        oh,
        ow,
        stride: *stride,
        padding: *padding,
        rows: spans(kh, h, oh, stride.0, padding.0),
        columns: spans(kw, w, ow, stride.1, padding.1),
        patches: if training {
            (0..spatial)
                .map(|p| {
                    let y = (p / ow) * stride.0;
                    let x = (p % ow) * stride.1;
                    PatchRegion {
                        top: padding.0.saturating_sub(y).min(kh),
                        bottom: (h + padding.0).saturating_sub(y).min(kh),
                        left: padding.1.saturating_sub(x).min(kw),
                        right: (w + padding.1).saturating_sub(x).min(kw),
                    }
                })
                .collect()
        } else {
            Vec::new()
        },
        spatial,
        patch_width,
    }))
}
impl ConvInto {
    fn forward_rows(&self, x: &[f32], weights: &[f32], bias: &[f32], output: &mut [f32]) {
        for b in 0..self.n {
            for oc in 0..self.co {
                let out = &mut output
                    [(b * self.co + oc) * self.spatial..(b * self.co + oc + 1) * self.spatial];
                out.fill(bias[oc]);
                // Every output retains the original ic/ky/kx reduction order.
                for ic in 0..self.ci {
                    for (ky, rows) in self.rows.iter().enumerate() {
                        for (kx, cols) in self.columns.iter().enumerate() {
                            if cols.begin == cols.end {
                                continue;
                            }
                            let weight =
                                weights[((oc * self.ci + ic) * self.kh + ky) * self.kw + kx];
                            for y in rows.begin..rows.end {
                                let iy = rows.input + (y - rows.begin) * self.stride.0;
                                let start =
                                    ((b * self.ci + ic) * self.h + iy) * self.w + cols.input;
                                let dst =
                                    &mut out[y * self.ow + cols.begin..y * self.ow + cols.end];
                                if self.stride.1 == 1 {
                                    let source = &x[start..start + dst.len()];
                                    for (d, &v) in dst.iter_mut().zip(source) {
                                        *d += v * weight;
                                    }
                                } else {
                                    for (i, d) in dst.iter_mut().enumerate() {
                                        *d += x[start + i * self.stride.1] * weight;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    fn input_gradient(&self, g: &[f32], weights: &[f32], dx: &mut [f32]) {
        for b in 0..self.n {
            for oc in 0..self.co {
                let grad =
                    &g[(b * self.co + oc) * self.spatial..(b * self.co + oc + 1) * self.spatial];
                for ic in 0..self.ci {
                    // For a fixed input element, decreasing ky/kx visits its
                    // contributing output y/x in ascending order, exactly as eager.
                    for (ky, rows) in self.rows.iter().enumerate().rev() {
                        for (kx, cols) in self.columns.iter().enumerate().rev() {
                            if cols.begin == cols.end {
                                continue;
                            }
                            let weight =
                                weights[((oc * self.ci + ic) * self.kh + ky) * self.kw + kx];
                            for y in rows.begin..rows.end {
                                let iy = rows.input + (y - rows.begin) * self.stride.0;
                                let start =
                                    ((b * self.ci + ic) * self.h + iy) * self.w + cols.input;
                                let source =
                                    &grad[y * self.ow + cols.begin..y * self.ow + cols.end];
                                if self.stride.1 == 1 {
                                    let dst = &mut dx[start..start + source.len()];
                                    for (d, &v) in dst.iter_mut().zip(source) {
                                        *d += v * weight;
                                    }
                                } else {
                                    for (i, &v) in source.iter().enumerate() {
                                        dx[start + i * self.stride.1] += v * weight;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    fn weight_gradient(&self, x: &[f32], g: &[f32], dw: &mut [f32], tile: &mut [f32]) {
        if self.patch_width == 0 || self.co == 0 {
            return;
        }
        for b in 0..self.n {
            for first in (0..self.spatial).step_by(PATCH_TILE) {
                let count = PATCH_TILE.min(self.spatial - first);
                // Pack once for all output channels. Padding remains unread,
                // preserving skip semantics even for NaN/Inf and signed zeros.
                for local in 0..count {
                    let p = first + local;
                    let r = &self.patches[p];
                    if r.left == r.right {
                        continue;
                    }
                    let y = (p / self.ow) * self.stride.0;
                    let x0 = (p % self.ow) * self.stride.1;
                    let patch = &mut tile[local * self.patch_width..(local + 1) * self.patch_width];
                    for ic in 0..self.ci {
                        for ky in r.top..r.bottom {
                            let source = ((b * self.ci + ic) * self.h + y + ky - self.padding.0)
                                * self.w
                                + x0
                                + r.left
                                - self.padding.1;
                            let start = (ic * self.kh + ky) * self.kw + r.left;
                            patch[start..start + r.right - r.left]
                                .copy_from_slice(&x[source..source + r.right - r.left]);
                        }
                    }
                }
                for oc in 0..self.co {
                    let dst = &mut dw[oc * self.patch_width..(oc + 1) * self.patch_width];
                    for local in 0..count {
                        let p = first + local;
                        let r = &self.patches[p];
                        let upstream = g[(b * self.co + oc) * self.spatial + p];
                        let patch = &tile[local * self.patch_width..(local + 1) * self.patch_width];
                        // Independent weights form SIMD lanes; each weight still
                        // reduces in the original batch/y/x order.
                        if r.top == 0 && r.bottom == self.kh && r.left == 0 && r.right == self.kw {
                            for (d, &v) in dst.iter_mut().zip(patch) {
                                *d += upstream * v;
                            }
                        } else {
                            for ic in 0..self.ci {
                                for ky in r.top..r.bottom {
                                    let start = (ic * self.kh + ky) * self.kw;
                                    for (d, &v) in dst[start + r.left..start + r.right]
                                        .iter_mut()
                                        .zip(&patch[start + r.left..start + r.right])
                                    {
                                        *d += upstream * v;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
impl IntoKernel for ConvInto {
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
        if inputs.len() != 3
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
        self.forward_rows(inputs[0].data(), inputs[1].data(), inputs[2].data(), output);
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
        if input >= 3 {
            return Err(invalid_into());
        }
        let mut destinations = [const { None }; 3];
        destinations[input] = Some(destination);
        self.backward_many_into(
            inputs,
            saved,
            gradient,
            &mut destinations,
            &[write; 3],
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
            || destinations.len() != 3
            || writes.len() != 3
            || inputs.len() != 3
            || inputs
                .iter()
                .zip(&self.spec.inputs)
                .enumerate()
                .any(|(i, (v, s))| v.is_some_and(|v| v.shape() != s) || (i < 2 && v.is_none()))
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
        let active = [
            destinations[0].is_some(),
            destinations[1].is_some(),
            destinations[2].is_some(),
        ];
        let (dx, rest) = workspace.split_at_mut(self.sizes[0]);
        let (dw, rest) = rest.split_at_mut(self.sizes[1]);
        let (db, tile) = rest.split_at_mut(self.sizes[2]);
        if active[0] {
            dx.fill(0.0);
        }
        if active[1] {
            dw.fill(0.0);
        }
        if active[2] {
            db.fill(0.0);
        }
        let g = gradient.data();
        let x = inputs[0].unwrap().data();
        let w = inputs[1].unwrap().data();
        if active[0] {
            self.input_gradient(g, w, dx);
        }
        if active[1] {
            self.weight_gradient(x, g, dw, tile);
        }
        if active[2] {
            for (i, &upstream) in g.iter().enumerate() {
                db[(i / (self.oh * self.ow)) % self.co] += upstream;
            }
        }
        for ((destination, contribution), &write) in
            destinations.iter_mut().zip([dx, dw, db]).zip(writes)
        {
            if let Some(destination) = destination {
                super::into::publish(contribution, destination, write);
            }
        }
        Ok(())
    }
}
