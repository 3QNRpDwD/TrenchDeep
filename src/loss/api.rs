//! Interchangeable losses backed by the existing context operations.
use crate::{ExecutionContext, MlResult, Tensor, Variable};
pub use crate::{LossError, Reduction};

/// Numeric loss computation. Targets follow the context operation's gradient rules.
/// Keep host sampling outside this method so it can also be captured for prepared execution.
/// Structural configuration must stay fixed while a prepared fit is running.
/// Use `ctx.constant_tensor(...)` for configuration-only constants during capture;
/// ordinary `ctx.tensor`/`ctx.scalar` values are not implicitly captured.
pub trait Loss {
    fn compute(
        &self,
        ctx: &ExecutionContext,
        prediction: &Variable,
        target: &Tensor,
    ) -> MlResult<Variable>;
}

macro_rules! context_loss {
    ($name:ident, $method:ident) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $name {
            reduction: Reduction,
        }
        impl $name {
            /// Uses mean reduction. Explicit reduction configuration remains available.
            pub fn new() -> Self {
                Self {
                    reduction: Reduction::Mean,
                }
            }
            pub fn with_reduction(mut self, reduction: Reduction) -> Self {
                self.reduction = reduction;
                self
            }
            /// Change reduction between fits; reprepare manually owned executors.
            pub fn set_reduction(&mut self, reduction: Reduction) {
                self.reduction = reduction;
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
        impl Loss for $name {
            fn compute(
                &self,
                ctx: &ExecutionContext,
                prediction: &Variable,
                target: &Tensor,
            ) -> MlResult<Variable> {
                ctx.$method(prediction.tensor(), target, self.reduction)?
                    .as_variable()
            }
        }
    };
}
context_loss!(MseLoss, mse_loss);
context_loss!(MaeLoss, mae_loss);
context_loss!(BinaryCrossEntropyLoss, binary_cross_entropy);
context_loss!(CrossEntropyLoss, cross_entropy);
context_loss!(SoftmaxCrossEntropyLoss, softmax_cross_entropy);

#[derive(Debug, Clone, Copy)]
pub struct HuberLoss {
    delta: f32,
    reduction: Reduction,
}
impl HuberLoss {
    pub fn new(delta: f32) -> Self {
        Self {
            delta,
            reduction: Reduction::Mean,
        }
    }
    pub fn with_reduction(mut self, reduction: Reduction) -> Self {
        self.reduction = reduction;
        self
    }
    pub fn set_reduction(&mut self, reduction: Reduction) {
        self.reduction = reduction;
    }
    /// Change delta between fits. Validation remains in the context loss operation.
    pub fn set_delta(&mut self, delta: f32) {
        self.delta = delta;
    }
}
impl Loss for HuberLoss {
    fn compute(
        &self,
        ctx: &ExecutionContext,
        prediction: &Variable,
        target: &Tensor,
    ) -> MlResult<Variable> {
        ctx.huber_loss(prediction.tensor(), target, self.delta, self.reduction)?
            .as_variable()
    }
}
