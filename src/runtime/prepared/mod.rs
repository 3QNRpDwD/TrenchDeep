//! Explicit P1 preparation, initially an **allocating replay** implementation.
//!
//! Preparation validates descriptions without executing kernels or consuming RNG.
//! The common forward adapter uses temporary shape placeholders for the concrete Tensor
//! API and retains only explicitly declared configuration constants in the plan.
//! Training reuses compiled VJPs, backward order and accumulation slots without
//! registering dynamic graph nodes. Kernels and cotangents still allocate.
//! BufferPlan describes safe reuse and can allocate a fixed-capacity BufferArena.
//! Allocating replay does not use that arena yet; into-kernel integration is S4.
//! There is no implicit cache or default-mode change.
pub(crate) mod backward;
mod buffers;
pub use buffers::{
    BufferArena, BufferLifetime, BufferPlan, BufferRole, BufferSlot, BufferValue, CopyReason,
    CopyRequirement, RootBufferPlan,
};
mod executor;
mod inputs;
pub use inputs::ExecutionInputs;
mod plan;
mod prepare;
pub(crate) mod recording;
pub use backward::BackwardPlanStats;
pub use plan::{PreparedMode, PreparedPlan, PreparedProgram, TensorSlotId};

fn invalid(reason: impl Into<String>) -> crate::MlError {
    crate::TensorError::InvalidOperation {
        op: "prepared replay",
        reason: reason.into(),
    }
    .into()
}
