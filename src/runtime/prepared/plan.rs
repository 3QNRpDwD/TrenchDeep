use super::{invalid, prepare::infer};
use crate::{ContextId, MlResult, ParameterId, contracts::Operation};
use std::sync::atomic::{AtomicU64, Ordering};
static PROGRAM_IDS: AtomicU64 = AtomicU64::new(1);

/// A symbolic slot, unrelated to a runtime TensorId or allocation address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorSlotId(pub(super) usize, pub(super) u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedMode {
    Training,
    Inference,
}
#[derive(Debug, Clone)]
pub(super) enum Source {
    Feed { requires_grad: bool },
    Parameter,
    Node,
    Constant(crate::TensorBuffer),
}
#[derive(Debug, Clone)]
pub(super) struct Slot {
    pub shape: Vec<usize>,
    pub source: Source,
}
#[derive(Debug, Clone)]
pub(super) struct Instruction {
    pub operation: Operation,
    pub inputs: Vec<TensorSlotId>,
    pub output: TensorSlotId,
}

/// Shape-only operation description. All slots are contiguous f32 tensors.
/// Runtime values (including target/noise/timestep) must be feeds, never captured.
#[derive(Debug, Clone)]
pub struct PreparedProgram {
    pub(super) id: u64,
    pub(super) slots: Vec<Slot>,
    pub(super) feeds: Vec<TensorSlotId>,
    pub(super) parameters: Vec<TensorSlotId>,
    pub(super) instructions: Vec<Instruction>,
}
impl Default for PreparedProgram {
    fn default() -> Self {
        Self {
            id: PROGRAM_IDS.fetch_add(1, Ordering::Relaxed),
            slots: Vec::new(),
            feeds: Vec::new(),
            parameters: Vec::new(),
            instructions: Vec::new(),
        }
    }
}
impl PreparedProgram {
    pub fn constant(&mut self, value: crate::TensorBuffer) -> MlResult<TensorSlotId> {
        let shape = value.shape().to_vec();
        self.slot(&shape, Source::Constant(value))
    }
    pub fn new() -> Self {
        Self::default()
    }
    fn slot(&mut self, shape: &[usize], source: Source) -> MlResult<TensorSlotId> {
        super::prepare::numel(shape)?;
        let id = TensorSlotId(self.slots.len(), self.id);
        self.slots.push(Slot {
            shape: shape.to_vec(),
            source,
        });
        Ok(id)
    }
    pub fn input(&mut self, shape: &[usize], requires_grad: bool) -> MlResult<TensorSlotId> {
        let slot = self.slot(shape, Source::Feed { requires_grad })?;
        self.feeds.push(slot);
        Ok(slot)
    }
    pub fn parameter(&mut self, shape: &[usize]) -> MlResult<TensorSlotId> {
        let slot = self.slot(shape, Source::Parameter)?;
        self.parameters.push(slot);
        Ok(slot)
    }
    /// Append one supported builtin. References must already exist, so the
    /// description is a DAG in forward order. Failed validation changes nothing.
    pub fn operation(
        &mut self,
        operation: Operation,
        inputs: &[TensorSlotId],
    ) -> MlResult<TensorSlotId> {
        if inputs.iter().any(|id| id.1 != self.id) {
            return Err(invalid("slot belongs to another program"));
        }
        let shapes = inputs
            .iter()
            .map(|id| {
                self.slots
                    .get(id.0)
                    .map(|slot| slot.shape.as_slice())
                    .ok_or_else(|| invalid("unknown input slot"))
            })
            .collect::<MlResult<Vec<_>>>()?;
        let shape = infer(&operation, &shapes)?;
        let output = self.slot(&shape, Source::Node)?;
        self.instructions.push(Instruction {
            operation,
            inputs: inputs.to_vec(),
            output,
        });
        Ok(output)
    }
}

/// Reusable immutable topology and binding signature. Contains no Tensor,
/// Variable, Parameter handle, context owner, or RNG state. Only explicitly
/// declared immutable constants carry values; feeds are never captured.
#[derive(Debug)]
pub struct PreparedPlan {
    pub(super) buffers: super::buffers::BufferPlan,
    pub(super) input_signature: Option<(String, Vec<String>)>,
    pub(super) backward: std::rc::Rc<super::backward::BackwardPlan>,
    pub(super) context: ContextId,
    pub(super) mode: PreparedMode,
    pub(super) program: PreparedProgram,
    pub(super) outputs: Vec<TensorSlotId>,
    pub(super) parameter_ids: Vec<ParameterId>,
}
impl PreparedPlan {
    pub fn buffer_plan(&self) -> &super::buffers::BufferPlan {
        &self.buffers
    }
    pub fn backward_plan_stats(&self) -> super::backward::BackwardPlanStats {
        self.backward.stats()
    }
    pub fn mode(&self) -> PreparedMode {
        self.mode
    }
    pub fn node_count(&self) -> usize {
        self.program.instructions.len()
    }
    pub fn slot_count(&self) -> usize {
        self.program.slots.len()
    }
    /// Prepared kernels and gradients still allocate; no static arena is used.
    pub fn uses_static_buffers(&self) -> bool {
        false
    }
}
