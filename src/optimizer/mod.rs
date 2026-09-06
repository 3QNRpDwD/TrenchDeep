//! Optimizers consume stable parameters, never graph IDs or storage internals.
mod algorithms;
pub use crate::OptimError;
pub use algorithms::*;
