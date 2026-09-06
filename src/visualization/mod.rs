//! Owned snapshots of an explicit execution context.
mod dot;
mod error;
mod graphviz;
mod snapshot;
pub(crate) mod statistics;
mod writer;
pub use dot::{DotEncoder, DotProfile};
pub use error::VisualizationError;
pub use graphviz::{GraphvizFailure, GraphvizFailureKind};
pub use snapshot::*;
pub use writer::{
    FileSnapshotWriter, FileSnapshotWriterBuilder, SnapshotWriter, WriteReport, WriteWarning,
};
