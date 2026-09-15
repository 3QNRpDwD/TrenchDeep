//! Direct NCHW convolution into fixed destinations, without im2col or padding copies.
use super::into::{invalid_into, size};
use super::*;
use std::rc::Rc;
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
}
pub(super) fn prepare(
    op: &Operation,
    shapes: &[&[usize]],
    training: bool,
) -> MlResult<Option<Rc<dyn IntoKernel>>> {
    let Operation::Conv2d { stride, padding } = op else {
        return Ok(None);
    };
    if shapes.len() != 3 {
        return Err(invalid_into());
    }
    let sizes = [size(shapes[0])?, size(shapes[1])?, size(shapes[2])?];
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
    let output_size = size(&output)?;
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
                .try_fold(0usize, |a, &b| a.checked_add(b))
                .ok_or_else(invalid_into)?
        } else {
            0
        },
        training,
    };
    Ok(Some(Rc::new(ConvInto {
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
    })))
}
impl ConvInto {
    fn products(&self, mut visit: impl FnMut(usize, usize, usize)) {
        for b in 0..self.n {
            for oc in 0..self.co {
                for y in 0..self.oh {
                    for x in 0..self.ow {
                        let out = ((b * self.co + oc) * self.oh + y) * self.ow + x;
                        for ic in 0..self.ci {
                            for ky in 0..self.kh {
                                for kx in 0..self.kw {
                                    let iy = y * self.stride.0 + ky;
                                    let ix = x * self.stride.1 + kx;
                                    if iy >= self.padding.0 && ix >= self.padding.1 {
                                        let sy = iy - self.padding.0;
                                        let sx = ix - self.padding.1;
                                        if sy < self.h && sx < self.w {
                                            visit(
                                                out,
                                                ((b * self.ci + ic) * self.h + sy) * self.w + sx,
                                                ((oc * self.ci + ic) * self.kh + ky) * self.kw + kx,
                                            );
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
        for (i, dst) in output.iter_mut().enumerate() {
            *dst = inputs[2].data()[(i / (self.oh * self.ow)) % self.co];
        }
        self.products(|out, x, w| output[out] += inputs[0].data()[x] * inputs[1].data()[w]);
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
        let (dw, db) = rest.split_at_mut(self.sizes[1]);
        let db = &mut db[..self.sizes[2]];
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
        if active[0] || active[1] {
            self.products(|out, xi, wi| {
                let upstream = g[out];
                if active[0] {
                    dx[xi] += upstream * w[wi];
                }
                if active[1] {
                    dw[wi] += upstream * x[xi];
                }
            });
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
