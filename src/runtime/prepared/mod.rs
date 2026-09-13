//! Explicit P1 preparation, initially an **allocating replay** implementation.
//!
//! Preparation validates immutable descriptions without executing tensors or RNG.
//! Each run still uses eager kernels and records its own backward graph. There is
//! no buffer arena, cached backward order, implicit cache, or default-mode change.
//! S2/S3/S4 will replace these costs; callers must not treat S0 as static memory.
mod executor;
mod plan;
mod prepare;
pub use plan::{PreparedMode, PreparedPlan, PreparedProgram, TensorSlotId};

fn invalid(reason: impl Into<String>) -> crate::MlError {
    crate::TensorError::InvalidOperation {
        op: "prepared replay",
        reason: reason.into(),
    }
    .into()
}
