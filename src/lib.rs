//! Explicit, composable tensor and training runtime.
pub mod contracts;
pub mod backend;
pub mod runtime;
pub use contracts::{AutogradError, ContextError, DataError, LossError, OptimError, MlError, MlResult,
    TensorError, ContextId, TensorId, ParameterId, TensorBuffer, TensorView, Reduction,
    CustomOp, OpOutput, BackwardOp};
pub use runtime::{Tensor, Variable, Parameter, ExecutionContext, ExecutionContextBuilder,
    BackwardOptions, RequiresGrad, GraphStats, TopKResult, MaxResult};
pub mod tensor { pub use crate::{Tensor, TensorBuffer, TensorView}; }
pub mod loss { pub use crate::{LossError, Reduction}; }
#[cfg(feature="legacyBenchmark")]
pub use trench_deep_legacy as legacy;
pub mod nn;
pub mod optimizer;
pub mod trainer;
