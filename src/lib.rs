//! Explicit, composable tensor and training runtime.
pub mod backend;
pub mod contracts;
pub mod runtime;
pub use contracts::{
    AutogradError, BackwardOp, ContextError, ContextId, CustomOp, DataError, LossError, MlError,
    MlResult, OpOutput, OptimError, ParameterId, Reduction, TensorBuffer, TensorError, TensorId,
    TensorView,
};
pub use runtime::{
    BackwardOptions, ExecutionContext, ExecutionContextBuilder, GraphStats, MaxResult, Parameter,
    RequiresGrad, Tensor, TopKResult, Variable,
    ExecutionRoute, RoutedContextBuilder,
};
pub mod tensor {
    pub use crate::{Tensor, TensorBuffer, TensorView};
}
pub mod loss {
    pub use crate::{LossError, Reduction};
}
#[cfg(feature = "legacyBenchmark")]
pub mod legacy;
pub mod nn;
pub mod optimizer;
pub mod trainer;

pub mod visualization;
