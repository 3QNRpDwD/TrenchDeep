//! Interchangeable losses backed by the existing context operations.
pub use crate::{LossError, Reduction};
use crate::{ExecutionContext, MlResult, Tensor, Variable};

/// Numeric loss computation. Targets follow the context operation's gradient rules.
/// Keep host sampling outside this method so it can also be captured for prepared execution.
/// Structural configuration must stay fixed while a prepared fit is running.
pub trait Loss {
    fn compute(&self, ctx: &ExecutionContext, prediction: &Variable, target: &Tensor)
    -> MlResult<Variable>;
}

macro_rules! context_loss {
    ($name:ident, $method:ident) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $name { reduction: Reduction }
        impl $name {
            pub fn new(reduction: Reduction) -> Self { Self { reduction } }
        }
        impl Loss for $name {
            fn compute(&self, ctx: &ExecutionContext, prediction: &Variable, target: &Tensor) -> MlResult<Variable> {
                ctx.$method(prediction.tensor(), target, self.reduction)?.as_variable()
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
pub struct HuberLoss { delta: f32, reduction: Reduction }
impl HuberLoss {
    pub fn new(delta: f32, reduction: Reduction) -> Self { Self { delta, reduction } }
}
impl Loss for HuberLoss {
    fn compute(&self, ctx: &ExecutionContext, prediction: &Variable, target: &Tensor) -> MlResult<Variable> {
        ctx.huber_loss(prediction.tensor(), target, self.delta, self.reduction)?.as_variable()
    }
}
