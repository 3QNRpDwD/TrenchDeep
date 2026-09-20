//! Explicit into execution for inference and training;
//! unsupported nodes fail before arena allocation, with no allocating fallback.
use super::{BufferArena, BufferValue, ExecutionInputs, buffers::ArenaIo, invalid, plan::*};
use crate::{contracts::IntoKernel, *};
use std::{cell::RefCell, rc::Rc};
#[derive(Debug)]
pub struct PreparedExecutor {
    plan: PreparedPlan,
    context: ExecutionContext,
    kernels: Vec<Rc<dyn IntoKernel>>,
    storage: Rc<RefCell<IntoStorage>>,
    placements: Vec<usize>,
    sizes: Vec<usize>,
    forward_io: Vec<ArenaIo>,
}
#[derive(Debug)]
pub(super) struct IntoStorage {
    pub arena: BufferArena,
    pub workspace: Vec<f32>,
    pub kernels: Vec<Rc<dyn IntoKernel>>,
    pub gradients: Vec<Option<usize>>,
    pub received: Vec<usize>,
    pub backward_io: Vec<Option<BackwardIo>>,
}
#[derive(Debug)]
pub(super) struct BackwardIo {
    pub input_positions: Vec<Option<usize>>,
    pub saved_start: usize,
    pub groups: Vec<GradientGroup>,
}
#[derive(Debug)]
pub(super) struct GradientGroup {
    pub inputs: Vec<usize>,
    pub targets: Vec<usize>,
    pub io: ArenaIo,
}
impl PreparedPlan {
    /// Compile the optional into capability and allocate reusable storage.
    /// Unsupported operations fail explicitly.
    pub fn into_executor(self, ctx: &ExecutionContext) -> MlResult<PreparedExecutor> {
        ctx.deny_preparation("into executor preparation")?;
        ctx.reject_prepared_extension()?;
        if ctx.id() != self.context {
            return Err(ContextError::Mismatch.into());
        }
        let provider = ctx
            .inner
            .operations
            .as_ref()
            .ok_or_else(|| invalid("missing into provider"))?;
        let mut kernels = Vec::new();
        for (index, node) in self.program.instructions.iter().enumerate() {
            let training = self
                .backward
                .roots
                .iter()
                .any(|root| root.order.contains(&index));
            let shapes = node
                .inputs
                .iter()
                .map(|s| self.program.slots[s.0].shape.as_slice())
                .collect::<Vec<_>>();
            let kernel = provider
                .prepare_into(&node.operation, &shapes, training)?
                .ok_or(MlError::UnsupportedCapability {
                    module: "prepared into",
                    capability: "fixed-destination kernel",
                    operation: node.operation.name(),
                })?;
            let spec = kernel.spec();
            if spec.training != training
                || spec.inputs.iter().map(Vec::as_slice).collect::<Vec<_>>() != shapes
                || spec.output != self.program.slots[node.output.0].shape
                || (if training {
                    spec.saved
                        != self.backward.nodes[index]
                            .vjp
                            .as_ref()
                            .unwrap()
                            .saved_shapes
                        || spec.backward_inputs
                            != self.backward.nodes[index].vjp.as_ref().unwrap().inputs
                } else {
                    !spec.saved.is_empty()
                })
                || spec.workspace_elements > isize::MAX as usize / 4
            {
                return Err(invalid("invalid into contract"));
            }
            kernels.push(kernel);
        }
        let sizes = self
            .program
            .slots
            .iter()
            .map(|s| super::prepare::numel(&s.shape))
            .collect::<MlResult<Vec<_>>>()?;
        let mut placements = vec![0; sizes.len()];
        for life in &self.buffers.layout.lifetimes {
            if let BufferValue::Tensor(slot) = life.value {
                placements[slot] = life.buffer;
            }
        }
        for slot in 0..placements.len() {
            placements[slot] = placements[self.buffers.aliases[slot]];
        }
        let reads: Vec<Vec<usize>> = self
            .program
            .instructions
            .iter()
            .map(|node| node.inputs.iter().map(|s| placements[s.0]).collect())
            .collect();
        let elements = kernels
            .iter()
            .map(|k| k.spec().workspace_elements)
            .max()
            .unwrap_or(0);
        let mut workspace = Vec::new();
        workspace
            .try_reserve_exact(elements)
            .map_err(|_| invalid("workspace allocation failed"))?;
        workspace.resize(elements, 0.0);
        let arena = self.buffers.allocate_validated_arena()?;
        let mut saved = kernels
            .iter()
            .map(|k| vec![usize::MAX; k.spec().saved.len()])
            .collect::<Vec<_>>();
        let mut gradients = vec![None; sizes.len()];
        for life in &self.buffers.layout.lifetimes {
            match life.value {
                BufferValue::Saved { node, index } => saved[node][index] = life.buffer,
                BufferValue::Gradient(slot) => gradients[slot] = Some(life.buffer),
                _ => {}
            }
        }
        if saved.iter().flatten().any(|&s| s == usize::MAX) {
            return Err(invalid("missing saved placement"));
        }
        let buffer_count = self.buffers.capacity_elements.len();
        let forward_io = self
            .program
            .instructions
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let mut writes = vec![placements[node.output.0]];
                writes.extend(&saved[i]);
                ArenaIo::new(&reads[i], &writes, buffer_count)
            })
            .collect::<MlResult<Vec<_>>>()?;
        let backward_io = self
            .backward
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| {
                let Some(vjp) = &node.vjp else {
                    return Ok(None);
                };
                let mut reads = Vec::new();
                let input_positions = node
                    .inputs
                    .iter()
                    .zip(&vjp.inputs)
                    .map(|(&slot, dependency)| {
                        if *dependency == contracts::BackwardInput::Values {
                            let position = reads.len();
                            reads.push(placements[slot]);
                            Some(position)
                        } else {
                            None
                        }
                    })
                    .collect();
                let saved_start = reads.len();
                reads.extend(&saved[index]);
                let Some(cotangent) = gradients[node.output] else {
                    return Ok(None);
                };
                reads.push(cotangent);
                let inputs = node
                    .inputs
                    .iter()
                    .zip(&vjp.differentiable)
                    .enumerate()
                    .filter_map(|(i, (&s, &enabled))| {
                        (enabled && self.backward.tracked[s]).then_some(i)
                    })
                    .collect::<Vec<_>>();
                let destinations = inputs
                    .iter()
                    .map(|&i| self.backward.canonical[node.inputs[i]])
                    .collect::<Vec<_>>();
                let aliases = destinations
                    .iter()
                    .enumerate()
                    .any(|(i, t)| destinations[..i].contains(t));
                let groups = if aliases {
                    inputs.into_iter().map(|i| vec![i]).collect()
                } else {
                    vec![inputs]
                };
                let groups = groups
                    .into_iter()
                    .map(|inputs: Vec<usize>| {
                        let targets = inputs
                            .iter()
                            .map(|&i| self.backward.canonical[node.inputs[i]])
                            .collect::<Vec<_>>();
                        let writes = targets
                            .iter()
                            .map(|&t| {
                                gradients[t].ok_or_else(|| invalid("missing gradient placement"))
                            })
                            .collect::<MlResult<Vec<_>>>()?;
                        Ok(GradientGroup {
                            inputs,
                            targets,
                            io: ArenaIo::new(&reads, &writes, buffer_count)?,
                        })
                    })
                    .collect::<MlResult<Vec<_>>>()?;
                Ok(Some(BackwardIo {
                    input_positions,
                    saved_start,
                    groups,
                }))
            })
            .collect::<MlResult<Vec<_>>>()?;
        let storage = Rc::new(RefCell::new(IntoStorage {
            arena,
            workspace,
            kernels: kernels.clone(),
            gradients,
            received: vec![0; sizes.len()],
            backward_io,
        }));
        Ok(PreparedExecutor {
            plan: self,
            context: ctx.clone(),
            kernels,
            storage,
            placements,
            sizes,
            forward_io,
        })
    }
}
impl PreparedExecutor {
    pub(super) fn context_id(&self) -> ContextId {
        self.context.id()
    }
    pub fn plan(&self) -> &PreparedPlan {
        &self.plan
    }
    pub fn arena_bytes(&self) -> usize {
        self.storage.borrow().arena.bytes()
    }
    pub fn workspace_bytes(&self) -> usize {
        self.storage.borrow().workspace.len() * 4
    }
    pub fn uses_static_buffers(&self) -> bool {
        true
    }
    pub fn with_inputs<T>(
        &mut self,
        inputs: &ExecutionInputs,
        parameters: &[&Parameter],
        callback: impl FnOnce(&[Tensor]) -> MlResult<T>,
    ) -> MlResult<T> {
        if self.plan.input_signature.as_ref() != Some(&inputs.signature()) {
            return Err(invalid("input names/order or topology variant changed"));
        }
        self.run_bound(&inputs.bindings(), parameters, callback)
    }
    pub fn with_run<T>(
        &mut self,
        inputs: &[&Tensor],
        parameters: &[&Parameter],
        callback: impl FnOnce(&[Tensor]) -> MlResult<T>,
    ) -> MlResult<T> {
        if self.plan.input_signature.is_some() {
            return Err(invalid("named plans require with_inputs"));
        }
        self.run_bound(inputs, parameters, callback)
    }
    fn run_bound<T>(
        &mut self,
        inputs: &[&Tensor],
        parameters: &[&Parameter],
        callback: impl FnOnce(&[Tensor]) -> MlResult<T>,
    ) -> MlResult<T> {
        self.context.deny_preparation("into execution")?;
        self.context.reject_prepared_extension()?;
        self.plan
            .validate_bindings(&self.context, inputs, parameters)?;
        let ctx = self.context.clone();
        if self.plan.mode == PreparedMode::Training {
            ctx.with_training_scope(|| self.execute_bound(inputs, parameters, callback))
        } else {
            self.execute_bound(inputs, parameters, callback)
        }
    }
    fn execute_bound<T>(
        &mut self,
        inputs: &[&Tensor],
        parameters: &[&Parameter],
        callback: impl FnOnce(&[Tensor]) -> MlResult<T>,
    ) -> MlResult<T> {
        let mut storage = self.storage.borrow_mut();
        let IntoStorage {
            arena, workspace, ..
        } = &mut *storage;
        // All external values are recopied every run: scratch may reuse their
        // slots after last use, and parameters may have changed since last run.
        for (tensor, slot) in inputs.iter().copied().zip(&self.plan.program.feeds).chain(
            parameters
                .iter()
                .map(|p| p.tensor())
                .zip(&self.plan.program.parameters),
        ) {
            let dst = &mut arena.buffer_mut(self.placements[slot.0])?[..self.sizes[slot.0]];
            tensor.with_view(|v| dst.copy_from_slice(v.data()))?;
        }
        for (slot, spec) in self.plan.program.slots.iter().enumerate() {
            if let Source::Constant(value) = &spec.source {
                arena.buffer_mut(self.placements[slot])?[..self.sizes[slot]]
                    .copy_from_slice(value.data());
            }
        }
        for (index, node) in self.plan.program.instructions.iter().enumerate() {
            let kernel = &self.kernels[index];
            self.forward_io[index].run(arena, |values, destinations| {
                let (destination, saved) = destinations.split_first_mut().unwrap();
                let mut saved = saved
                    .iter_mut()
                    .zip(&kernel.spec().saved)
                    .map(|(data, shape)| Ok(&mut data[..super::prepare::numel(shape)?]))
                    .collect::<MlResult<Vec<_>>>()?;
                let views = values
                    .iter()
                    .zip(&kernel.spec().inputs)
                    .map(|(data, shape)| {
                        let size = super::prepare::numel(shape)?;
                        TensorView::new(&data[..size], shape)
                    })
                    .collect::<MlResult<Vec<_>>>()?;
                kernel.execute_into(
                    &views,
                    &mut destination[..self.sizes[node.output.0]],
                    &mut saved,
                    workspace,
                )
            })?;
        }
        // Public outputs are independent copies and excluded from the no-
        // intermediate-data-allocation claim. Repeated exports share a handle.
        let mut exported: Vec<Option<Tensor>> = vec![None; self.sizes.len()];
        let mut outputs = Vec::new();
        for slot in &self.plan.outputs {
            if exported[slot.0].is_none() {
                exported[slot.0] = Some(self.context.tensor(
                    arena.buffer(self.placements[slot.0])?[..self.sizes[slot.0]].to_vec(),
                    &self.plan.program.slots[slot.0].shape,
                )?);
            }
            outputs.push(exported[slot.0].as_ref().unwrap().clone());
        }
        drop(storage);
        if self.plan.mode == PreparedMode::Inference {
            return self.context.no_grad(|| callback(&outputs));
        }
        let mut ids = vec![None; self.sizes.len()];
        for (tensor, slot) in inputs.iter().copied().zip(&self.plan.program.feeds).chain(
            parameters
                .iter()
                .map(|p| p.tensor())
                .zip(&self.plan.program.parameters),
        ) {
            ids[slot.0] = Some(tensor.id());
        }
        let mut exports = Vec::new();
        for (slot, tensor) in self.plan.outputs.iter().zip(&outputs) {
            if ids[slot.0].is_none() {
                ids[slot.0] = Some(tensor.id());
                exports.push((tensor.id(), slot.0));
                if self.plan.backward.tracked[slot.0] {
                    self.context
                        .inner
                        .state
                        .borrow_mut()
                        .tracked
                        .insert(tensor.id());
                }
            }
        }
        *self.context.inner.prepared_run.borrow_mut() = Some(super::backward::PreparedRun {
            plan: self.plan.backward.clone(),
            values: Vec::new(),
            saved: Vec::new(),
            into: Some(self.storage.clone()),
            ids,
            exports,
            consumed: false,
            completed: false,
        });
        let _cleanup = super::executor::Cleanup(&self.context);
        callback(&outputs)
    }
}
