//! Fixed-destination VJPs follow the same compiled order as allocating replay.
use super::{backward::PreparedRun, into_executor::IntoStorage, invalid, prepare::numel};
use crate::{contracts::GradientWrite, *};
pub(super) fn execute(
    ctx: &ExecutionContext,
    run: &mut PreparedRun,
    slot: usize,
    seed: TensorBuffer,
    observe: impl FnOnce(&super::super::State) -> MlResult<()>,
) -> MlResult<()> {
    let root = run.plan.roots.iter().find(|r| r.slot == slot).unwrap();
    let mut storage = run
        .into
        .as_ref()
        .unwrap()
        .try_borrow_mut()
        .map_err(|_| ContextError::BorrowConflict)?;
    let IntoStorage {
        arena,
        workspace,
        kernels,
        backward_io,
        gradients,
        received,
        metadata,
        ..
    } = &mut *storage;
    received.fill(0);
    let seed_slot = run.plan.canonical[slot];
    let buffer = gradients[seed_slot].ok_or_else(|| invalid("missing root gradient placement"))?;
    arena.buffer_mut(buffer)?[..seed.data().len()].copy_from_slice(seed.data());
    received[seed_slot] = 1;
    run.consumed = true;
    for &index in root.order.iter().rev() {
        let node = &run.plan.nodes[index];
        if received[node.output] != root.fan_in[node.output] {
            return Err(invalid("incomplete prepared gradient fan-in"));
        }
        let kernel = &kernels[index];
        let io = backward_io[index]
            .as_ref()
            .ok_or_else(|| invalid("missing backward IO plan"))?;
        for group in &io.groups {
            metadata.writes.with(
                node.inputs.len(),
                |_| Ok(GradientWrite::Assign),
                |writes| {
                    for (&input, &target) in group.inputs.iter().zip(&group.targets) {
                        if received[target] != 0 {
                            writes[input] = GradientWrite::Add;
                        }
                    }
                    group.io.run(
                        arena,
                        &mut metadata.io_inputs,
                        &mut metadata.io_outputs,
                        |values, destinations| {
                            metadata.inputs.with(
                                io.input_positions.len(),
                                |i| {
                                    let shape = &kernel.spec().inputs[i];
                                    io.input_positions[i]
                                        .map(|p| {
                                            TensorView::new(&values[p][..numel(shape)?], shape)
                                        })
                                        .transpose()
                                },
                                |inputs| {
                                    metadata.saved.with(
                                        kernel.spec().saved.len(),
                                        |i| {
                                            let shape = &kernel.spec().saved[i];
                                            TensorView::new(
                                                &values[io.saved_start + i][..numel(shape)?],
                                                shape,
                                            )
                                        },
                                        |saved| {
                                            let shape = &run.plan.shapes[node.output];
                                            let gradient = TensorView::new(
                                                &values[values.len() - 1][..numel(shape)?],
                                                shape,
                                            )?;
                                            metadata.outputs.with(
                                                node.inputs.len(),
                                                |_| Ok(None),
                                                |outputs| {
                                                    for ((&input, &target), destination) in group
                                                        .inputs
                                                        .iter()
                                                        .zip(&group.targets)
                                                        .zip(destinations.iter_mut())
                                                    {
                                                        outputs[input] = Some(
                                                            &mut destination[..numel(
                                                                &run.plan.shapes[target],
                                                            )?],
                                                        );
                                                    }
                                                    kernel.backward_many_into(
                                                        inputs, saved, gradient, outputs, writes,
                                                        workspace,
                                                    )
                                                },
                                            )
                                        },
                                    )
                                },
                            )
                        },
                    )
                },
            )?;
            for &target in &group.targets {
                received[target] += 1;
            }
        }
    }
    if *received != root.fan_in {
        return Err(invalid("incomplete prepared gradient fan-in"));
    }
    let mut state = ctx
        .inner
        .state
        .try_borrow_mut()
        .map_err(|_| ContextError::BorrowConflict)?;
    // Gradient publication is an ownership boundary, like public output copies.
    // Numeric intermediates and accumulation remain in reusable arena storage.
    for (slot, id) in run.ids.iter().enumerate() {
        let Some(id) = id else {
            continue;
        };
        if run.plan.canonical[slot] != slot
            || received[slot] == 0
            || (!state.leaves.contains(id) && !state.retained.contains(id))
        {
            continue;
        }
        let shape = &run.plan.shapes[slot];
        let values = &arena.buffer(gradients[slot].unwrap())?[..numel(shape)?];
        if let Some(existing) = state.gradients.get_mut(id) {
            for (a, b) in existing.data.iter_mut().zip(values) {
                *a += b;
            }
        } else {
            state
                .gradients
                .insert(*id, TensorBuffer::from_vec(values.to_vec(), shape)?);
        }
    }
    if let Err(error) = observe(&state) {
        state.gradients.clear();
        return Err(error);
    }
    state.consumed.extend(run.exports.iter().map(|(id, _)| *id));
    run.completed = true;
    Ok(())
}
