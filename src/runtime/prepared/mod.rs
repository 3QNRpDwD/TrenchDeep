//! Explicit P1 preparation with reusable forward and backward arenas.
//! Shape recording does not execute kernels or consume RNG. Into execution uses
//! compiled VJPs and fixed destinations; public outputs and published gradients
//! cross ownership boundaries by copying. Allocating replay remains available.
//! The model facade is shared with Trainer; unsupported capabilities are errors.
//! The default eager route is retained until the performance gate passes.
pub(crate) mod backward;
mod buffers;
pub use buffers::{
    BufferArena, BufferLifetime, BufferPlan, BufferRole, BufferSlot, BufferValue, CopyReason,
    CopyRequirement, RootBufferPlan,
};
mod executor;
mod into_backward;
mod into_executor;
pub use into_executor::PreparedExecutor;
mod model;
pub use model::{ModelOutput, PreparedBatch, PreparedModel, PreparedModelExecutor};
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
