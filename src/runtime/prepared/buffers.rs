//! Conservative, out-of-place buffer lifetimes for each selectable backward root.
//! This describes an into-kernel arena; allocating compatibility kernels do not
//! use these placements yet. Endpoints are inclusive: no reuse within an event.
use super::{backward::BackwardPlan, invalid, plan::*, prepare::numel};
use crate::{MlResult, Parameter};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferValue {
    Tensor(usize),
    Saved { node: usize, index: usize },
    Gradient(usize),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferRole {
    Feed,
    Parameter,
    Constant,
    Temporary,
    Saved,
    Gradient,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyReason {
    InputOwnership,
    ParameterSnapshot,
    ConstantInitialization,
    BackwardPreservation,
    ExportedOutput,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyRequirement {
    pub value: BufferValue,
    pub reason: CopyReason,
    pub bytes: usize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferLifetime {
    pub value: BufferValue,
    pub role: BufferRole,
    pub shape: Vec<usize>,
    pub first: usize,
    pub last: usize,
    pub buffer: usize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferSlot {
    pub elements: usize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootBufferPlan {
    /// None is forward-only. Some(slot) selects a declared backward output.
    pub root: Option<usize>,
    pub lifetimes: Vec<BufferLifetime>,
    pub buffers: Vec<BufferSlot>,
    pub copies: Vec<CopyRequirement>,
    pub arena_bytes: usize,
    pub unreused_bytes: usize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferPlan {
    /// Common placements valid before the callback selects its backward root.
    pub layout: RootBufferPlan,
    /// Parameter slots sharing a ParameterId have the same canonical value.
    pub aliases: Vec<usize>,
    /// Forward-only and each supported backward root have independent layouts.
    pub roots: Vec<RootBufferPlan>,
    /// Capacity envelope for an arena reusable across any root layout.
    pub capacity_elements: Vec<usize>,
    pub capacity_bytes: usize,
}
fn bytes(elements: usize) -> MlResult<usize> {
    elements
        .checked_mul(4)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| invalid("buffer byte size overflow"))
}
fn total(mut sizes: impl Iterator<Item = usize>) -> MlResult<usize> {
    let elements = sizes
        .try_fold(0usize, |a, b| a.checked_add(b))
        .ok_or_else(|| invalid("arena size overflow"))?;
    bytes(elements)
}
impl BufferPlan {
    /// Independently check placement bounds and overlapping live allocations.
    pub fn validate(&self) -> MlResult<()> {
        if self
            .aliases
            .iter()
            .any(|&a| a >= self.aliases.len() || self.aliases[a] != a)
        {
            return Err(invalid("invalid canonical alias"));
        }
        for root in &self.roots {
            for life in &root.lifetimes {
                if !self.layout.lifetimes.iter().any(|common| {
                    common.value == life.value
                        && common.shape == life.shape
                        && common.first <= life.first
                        && common.last >= life.last
                }) {
                    return Err(invalid("common layout does not preserve a root lifetime"));
                }
            }
        }
        for root in self.roots.iter().chain(std::iter::once(&self.layout)) {
            for (i, a) in root.lifetimes.iter().enumerate() {
                if a.first > a.last
                    || a.buffer >= root.buffers.len()
                    || numel(&a.shape)? > root.buffers[a.buffer].elements
                {
                    return Err(invalid("invalid buffer placement"));
                }
                for b in &root.lifetimes[..i] {
                    if a.buffer == b.buffer && a.first <= b.last && b.first <= a.last {
                        return Err(invalid("overlapping buffer lifetimes"));
                    }
                }
            }
            if total(root.buffers.iter().map(|b| b.elements))? != root.arena_bytes {
                return Err(invalid("invalid arena capacity"));
            }
            if root.buffers.iter().enumerate().any(|(i, b)| {
                self.capacity_elements
                    .get(i)
                    .is_none_or(|&n| n < b.elements)
            }) {
                return Err(invalid("invalid root capacity envelope"));
            }
        }
        if total(self.capacity_elements.iter().copied())? != self.capacity_bytes {
            return Err(invalid("invalid capacity bytes"));
        }
        Ok(())
    }
}
pub(super) fn compile(
    program: &PreparedProgram,
    parameters: &[&Parameter],
    outputs: &[TensorSlotId],
    backward: &BackwardPlan,
) -> MlResult<BufferPlan> {
    let mut aliases: Vec<_> = (0..program.slots.len()).collect();
    let mut parameter_aliases = HashMap::new();
    for (slot, p) in program.parameters.iter().zip(parameters) {
        aliases[slot.0] = *parameter_aliases.entry(p.id()).or_insert(slot.0);
    }
    let mut roots = Vec::new();
    // Include a no-backward run even for a training plan.
    for root in std::iter::once(None).chain(backward.roots.iter().map(Some)) {
        let n = program.instructions.len();
        let export = n.checked_add(1).ok_or_else(|| invalid("event overflow"))?;
        let finish = export
            .checked_add(root.map_or(0, |r| r.order.len()))
            .and_then(|x| x.checked_add(1))
            .ok_or_else(|| invalid("event overflow"))?;
        let mut lifetimes = Vec::new();
        let mut copies = Vec::new();
        for (slot, spec) in program.slots.iter().enumerate() {
            if aliases[slot] != slot {
                continue;
            }
            let role = match spec.source {
                Source::Feed { .. } => BufferRole::Feed,
                Source::Parameter => BufferRole::Parameter,
                Source::Constant(_) => BufferRole::Constant,
                Source::Node => BufferRole::Temporary,
            };
            lifetimes.push(BufferLifetime {
                value: BufferValue::Tensor(slot),
                role,
                shape: spec.shape.clone(),
                first: 0,
                last: 0,
                buffer: 0,
            });
            let reason = match role {
                BufferRole::Feed => Some(CopyReason::InputOwnership),
                BufferRole::Parameter => Some(CopyReason::ParameterSnapshot),
                BufferRole::Constant => Some(CopyReason::ConstantInitialization),
                _ => None,
            };
            if let Some(reason) = reason {
                copies.push(CopyRequirement {
                    value: BufferValue::Tensor(slot),
                    reason,
                    bytes: bytes(numel(&spec.shape)?)?,
                });
            }
        }
        let mut positions = vec![0; program.slots.len()];
        for (i, life) in lifetimes.iter().enumerate() {
            if let BufferValue::Tensor(slot) = life.value {
                positions[slot] = i;
            }
        }
        for slot in 0..positions.len() {
            positions[slot] = positions[aliases[slot]];
        }
        for (i, node) in program.instructions.iter().enumerate() {
            let event = i + 1;
            lifetimes[positions[node.output.0]].first = event;
            lifetimes[positions[node.output.0]].last = event;
            for input in &node.inputs {
                lifetimes[positions[input.0]].last = event;
            }
        }
        for output in outputs {
            lifetimes[positions[output.0]].last = export;
            if !copies.iter().any(|c| {
                c.reason == CopyReason::ExportedOutput && c.value == BufferValue::Tensor(output.0)
            }) {
                copies.push(CopyRequirement {
                    value: BufferValue::Tensor(output.0),
                    reason: CopyReason::ExportedOutput,
                    bytes: bytes(numel(&program.slots[output.0].shape)?)?,
                });
            }
        }
        if let Some(root) = root {
            let mut gradients: HashMap<usize, (usize, usize)> = HashMap::new();
            gradients.insert(root.slot, (export, export));
            for (step, &index) in root.order.iter().rev().enumerate() {
                let event = export + step + 1;
                let node = &backward.nodes[index];
                let vjp = node.vjp.as_ref().unwrap();
                gradients
                    .get_mut(&node.output)
                    .expect("derivative reachable")
                    .1 = event;
                for (&input, dependency) in node.inputs.iter().zip(&vjp.inputs) {
                    if *dependency == crate::contracts::BackwardInput::Values {
                        lifetimes[positions[input]].last = event;
                    }
                }
                for (saved, shape) in vjp.saved_shapes.iter().enumerate() {
                    let value = BufferValue::Saved {
                        node: index,
                        index: saved,
                    };
                    lifetimes.push(BufferLifetime {
                        value,
                        role: BufferRole::Saved,
                        shape: shape.clone(),
                        first: index + 1,
                        last: event,
                        buffer: 0,
                    });
                    // A separate preserved value, not necessarily a memcpy: into
                    // kernels may compute directly into this destination.
                    copies.push(CopyRequirement {
                        value,
                        reason: CopyReason::BackwardPreservation,
                        bytes: bytes(numel(shape)?)?,
                    });
                }
                for (&input, &enabled) in node.inputs.iter().zip(&vjp.differentiable) {
                    if enabled && backward.tracked[input] {
                        let entry = gradients.entry(aliases[input]).or_insert((event, event));
                        entry.1 = event;
                    }
                }
            }
            let mut gradients: Vec<_> = gradients.into_iter().collect();
            gradients.sort_by_key(|(slot, _)| *slot);
            for (slot, (first, mut last)) in gradients {
                // Leaf gradients serve the optimizer; any exported intermediate
                // can request retain_grad in the callback, so preserve it too.
                if !matches!(program.slots[slot].source, Source::Node)
                    || outputs.iter().any(|s| s.0 == slot)
                {
                    last = finish;
                }
                lifetimes.push(BufferLifetime {
                    value: BufferValue::Gradient(slot),
                    role: BufferRole::Gradient,
                    shape: program.slots[slot].shape.clone(),
                    first,
                    last,
                    buffer: 0,
                });
            }
        }
        let (buffers, arena_bytes, unreused_bytes) = place(&mut lifetimes)?;
        roots.push(RootBufferPlan {
            root: root.map(|r| r.slot),
            lifetimes,
            buffers,
            copies,
            arena_bytes,
            unreused_bytes,
        });
    }
    let mut merged: Vec<BufferLifetime> = Vec::new();
    let mut copies = Vec::new();
    for root in &roots {
        for life in &root.lifetimes {
            if let Some(existing) = merged.iter_mut().find(|l| l.value == life.value) {
                existing.first = existing.first.min(life.first);
                existing.last = existing.last.max(life.last);
            } else {
                merged.push(life.clone());
            }
        }
        for copy in &root.copies {
            if !copies.contains(copy) {
                copies.push(copy.clone());
            }
        }
    }
    let (buffers, arena_bytes, unreused_bytes) = place(&mut merged)?;
    let layout = RootBufferPlan {
        root: None,
        lifetimes: merged,
        buffers,
        copies,
        arena_bytes,
        unreused_bytes,
    };
    let mut capacity_elements = vec![
        0;
        roots
            .iter()
            .chain(std::iter::once(&layout))
            .map(|r| r.buffers.len())
            .max()
            .unwrap_or(0)
    ];
    for root in roots.iter().chain(std::iter::once(&layout)) {
        for (i, b) in root.buffers.iter().enumerate() {
            capacity_elements[i] = capacity_elements[i].max(b.elements);
        }
    }
    let plan = BufferPlan {
        layout,
        aliases,
        roots,
        capacity_bytes: total(capacity_elements.iter().copied())?,
        capacity_elements,
    };
    plan.validate()?;
    Ok(plan)
}
fn place(lifetimes: &mut [BufferLifetime]) -> MlResult<(Vec<BufferSlot>, usize, usize)> {
    lifetimes.sort_by_key(|l| l.first);
    let unreused_bytes = total(
        lifetimes
            .iter()
            .map(|l| numel(&l.shape).expect("validated shape")),
    )?;
    let mut buffers: Vec<BufferSlot> = Vec::new();
    let mut ends: Vec<usize> = Vec::new();
    for life in lifetimes {
        let elements = numel(&life.shape)?;
        let available = buffers
            .iter()
            .enumerate()
            .filter(|(i, b)| ends[*i] < life.first && b.elements >= elements)
            .min_by_key(|(_, b)| b.elements)
            .map(|(i, _)| i);
        let index = available.unwrap_or_else(|| {
            buffers.push(BufferSlot { elements });
            ends.push(0);
            buffers.len() - 1
        });
        ends[index] = life.last;
        life.buffer = index;
    }
    let arena_bytes = total(buffers.iter().map(|b| b.elements))?;
    Ok((buffers, arena_bytes, unreused_bytes))
}

/// Owned, fixed-capacity storage. No resizing or raw pointers are exposed.
/// Into-kernel integration is separate from planning/allocation.
#[derive(Debug)]
pub struct BufferArena {
    buffers: Vec<Box<[f32]>>,
}
impl BufferPlan {
    pub fn allocate_arena(&self) -> MlResult<BufferArena> {
        self.validate()?;
        let mut buffers = Vec::new();
        buffers
            .try_reserve_exact(self.capacity_elements.len())
            .map_err(|_| invalid("arena allocation failed"))?;
        for &elements in &self.capacity_elements {
            let mut values = Vec::new();
            values
                .try_reserve_exact(elements)
                .map_err(|_| invalid("arena allocation failed"))?;
            values.resize(elements, 0.0);
            buffers.push(values.into_boxed_slice());
        }
        Ok(BufferArena { buffers })
    }
}
impl BufferArena {
    pub fn buffer(&self, id: usize) -> MlResult<&[f32]> {
        self.buffers
            .get(id)
            .map(|b| b.as_ref())
            .ok_or_else(|| invalid("unknown arena buffer"))
    }
    pub fn buffer_mut(&mut self, id: usize) -> MlResult<&mut [f32]> {
        self.buffers
            .get_mut(id)
            .map(|b| b.as_mut())
            .ok_or_else(|| invalid("unknown arena buffer"))
    }
    pub fn bytes(&self) -> usize {
        self.buffers.iter().map(|b| b.len() * 4).sum()
    }
}
