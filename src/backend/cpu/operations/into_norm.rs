//! GroupNorm with prepared normalized/statistic destinations and workspace.
use super::into::{invalid_into, size};
use super::*;
use std::rc::Rc;
#[derive(Debug)]
struct NormInto {
    spec: IntoKernelSpec,
    n: usize,
    c: usize,
    spatial: usize,
    groups: usize,
    channels: usize,
    count: usize,
    elements: usize,
    epsilon: f32,
}
pub(super) fn prepare(
    op: &Operation,
    shapes: &[&[usize]],
    training: bool,
) -> MlResult<Option<Rc<dyn IntoKernel>>> {
    let Operation::GroupNorm { groups, epsilon } = op else {
        return Ok(None);
    };
    if shapes.len() != 3 {
        return Err(invalid_into());
    }
    for shape in shapes {
        size(shape)?;
    }
    let (n, c, h, w, channels, count) =
        group_norm_spec(shapes[0], shapes[1], shapes[2], *groups, *epsilon)?;
    let elements = size(shapes[0])?;
    let saved = if training {
        vec![shapes[0].to_vec(), vec![n, *groups], vec![n, *groups]]
    } else {
        vec![]
    };
    let spec = IntoKernelSpec {
        inputs: shapes.iter().map(|s| s.to_vec()).collect(),
        output: shapes[0].to_vec(),
        saved,
        backward_inputs: vec![
            BackwardInput::Shape,
            BackwardInput::Values,
            BackwardInput::Shape,
        ],
        workspace_elements: if training {
            elements.max(c.checked_mul(2).ok_or_else(invalid_into)?)
        } else {
            0
        },
        training,
    };
    Ok(Some(Rc::new(NormInto {
        spec,
        n,
        c,
        spatial: h * w,
        groups: *groups,
        channels,
        count,
        elements,
        epsilon: *epsilon,
    })))
}
impl IntoKernel for NormInto {
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
        if inputs.len() != 3
            || inputs
                .iter()
                .zip(&self.spec.inputs)
                .any(|(v, s)| v.shape() != s)
            || out.len() != self.elements
            || saved.len() != self.spec.saved.len()
            || saved
                .iter()
                .zip(&self.spec.saved)
                .any(|(v, s)| v.len() != s.iter().product::<usize>())
            || work.len() < self.spec.workspace_elements
        {
            return Err(invalid_into());
        }
        let x = inputs[0].data();
        let gamma = inputs[1].data();
        let beta = inputs[2].data();
        for b in 0..self.n {
            for group in 0..self.groups {
                let start = group * self.channels;
                let mut sum = 0.0;
                for c in start..start + self.channels {
                    let base = (b * self.c + c) * self.spatial;
                    sum += x[base..base + self.spatial].iter().sum::<f32>();
                }
                let mean = sum / self.count as f32;
                let mut deviation = 0.0;
                for c in start..start + self.channels {
                    let base = (b * self.c + c) * self.spatial;
                    deviation += x[base..base + self.spatial]
                        .iter()
                        .map(|v| (v - mean) * (v - mean))
                        .sum::<f32>();
                }
                let variance = deviation / self.count as f32;
                let inv = 1.0 / (variance + self.epsilon).sqrt();
                if self.spec.training {
                    saved[1][b * self.groups + group] = mean;
                    saved[2][b * self.groups + group] = variance;
                }
                for c in start..start + self.channels {
                    let base = (b * self.c + c) * self.spatial;
                    for i in base..base + self.spatial {
                        let norm = (x[i] - mean) * inv;
                        if self.spec.training {
                            saved[0][i] = norm;
                        }
                        out[i] = gamma[c] * norm + beta[c];
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
        dst: &mut [f32],
        write: GradientWrite,
        work: &mut [f32],
    ) -> MlResult<()> {
        if !self.spec.training
            || input >= 3
            || inputs.len() != 3
            || inputs
                .iter()
                .zip(&self.spec.inputs)
                .enumerate()
                .any(|(i, (v, s))| v.is_some_and(|v| v.shape() != s) || (i == 1 && v.is_none()))
            || saved.len() != 3
            || saved
                .iter()
                .zip(&self.spec.saved)
                .any(|(v, s)| v.shape() != s)
            || gradient.shape() != self.spec.output
            || dst.len() != if input == 0 { self.elements } else { self.c }
            || work.len() < self.spec.workspace_elements
        {
            return Err(invalid_into());
        }
        let tmp = &mut work[..dst.len()];
        let norm = saved[0].data();
        let variance = saved[2].data();
        let g = gradient.data();
        if input != 0 {
            tmp.fill(0.0);
            for (i, &g) in g.iter().enumerate() {
                let c = (i / self.spatial) % self.c;
                tmp[c] += if input == 1 { g * norm[i] } else { g };
            }
        } else {
            let gamma = inputs[1].unwrap().data();
            for b in 0..self.n {
                for group in 0..self.groups {
                    let start = group * self.channels;
                    let mut sum = 0.0;
                    let mut sum_norm = 0.0;
                    for c in start..start + self.channels {
                        let base = (b * self.c + c) * self.spatial;
                        for i in base..base + self.spatial {
                            let scaled = g[i] * gamma[c];
                            sum += scaled;
                            sum_norm += scaled * norm[i];
                        }
                    }
                    let inv = 1.0 / (variance[b * self.groups + group] + self.epsilon).sqrt();
                    let count = self.count as f32;
                    for c in start..start + self.channels {
                        let base = (b * self.c + c) * self.spatial;
                        for i in base..base + self.spatial {
                            let scaled = g[i] * gamma[c];
                            tmp[i] = inv / count * (count * scaled - sum - norm[i] * sum_norm);
                        }
                    }
                }
            }
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
                .any(|(i, (v, s))| v.is_some_and(|v| v.shape() != s) || (i == 1 && v.is_none()))
            || saved.len() != 3
            || saved
                .iter()
                .zip(&self.spec.saved)
                .any(|(v, s)| v.shape() != s)
            || gradient.shape() != self.spec.output
            || destinations.iter().enumerate().any(|(i, d)| {
                d.as_ref()
                    .is_some_and(|d| d.len() != if i == 0 { self.elements } else { self.c })
            })
            || workspace.len() < self.spec.workspace_elements
        {
            return Err(invalid_into());
        }
        if let Some(dx) = destinations[0].as_deref_mut() {
            self.backward_into(0, inputs, saved, gradient, dx, writes[0], workspace)?;
        }
        let gamma = destinations[1].is_some();
        let beta = destinations[2].is_some();
        if gamma || beta {
            let (dg, rest) = workspace.split_at_mut(self.c);
            let db = &mut rest[..self.c];
            if gamma {
                dg.fill(0.0);
            }
            if beta {
                db.fill(0.0);
            }
            let norm = saved[0].data();
            for (i, &g) in gradient.data().iter().enumerate() {
                let c = (i / self.spatial) % self.c;
                if gamma {
                    dg[c] += g * norm[i];
                }
                if beta {
                    db[c] += g;
                }
            }
            for ((destination, contribution), &write) in
                destinations[1..].iter_mut().zip([dg, db]).zip(&writes[1..])
            {
                if let Some(destination) = destination {
                    super::into::publish(contribution, destination, write);
                }
            }
        }
        Ok(())
    }
}
