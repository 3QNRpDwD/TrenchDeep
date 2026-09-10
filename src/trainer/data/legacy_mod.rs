//! Eager dataset construction and batching infrastructure.
//!
//! `DatasetBuilder` owns decoding and record-to-sample transforms. `DataLoader`
//! owns batching and epoch ordering. The small borrowed dataset wrappers in
//! [`crate::legacy::trainer::api`] remain the pre-batched convenience path.

#[path = "legacy.rs"]
mod legacy;
#[path = "legacy_loader.rs"]
mod loader;
#[path = "legacy_source.rs"]
mod source;

pub use loader::{
    AutoregressiveBatch, AutoregressiveSample, AutoregressiveStackCollator, BatchLoader, Collator,
    DataLoader, DataLoaderBuilder, Dataset, InMemoryDataset, IntoBatchLoader, MissingCollator,
    SemiSupervisedBatch, SemiSupervisedDataLoader, SemiSupervisedDataLoaderBuilder,
    SupervisedBatch, SupervisedSample, SupervisedStackCollator, UnsupervisedBatch,
    UnsupervisedSample, UnsupervisedStackCollator,
};
pub use source::{
    CsvRecord, CsvSource, DatasetBuilder, JsonLinesSource, JsonRecord, LocatedRecord, MemorySource,
    RecordSource, Transform,
};

use std::path::PathBuf;

/// Errors raised while decoding, transforming, and batching training data.
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("data I/O failed for {path}: {message}")]
    Io { path: PathBuf, message: String },
    #[error("data decode failed for {path} at line/row {line}: {message}")]
    Decode {
        path: PathBuf,
        line: usize,
        message: String,
    },
    #[error("data transform failed{location}: {message}")]
    Transform { location: String, message: String },
    #[error("dataset must not be empty")]
    EmptyDataset,
    #[error("batch_size must be greater than zero")]
    InvalidBatchSize,
    #[error("data loader would produce zero batches; disable drop_last or reduce batch_size")]
    NoBatches,
    #[error("cannot collate an empty batch")]
    EmptyBatch,
    #[error("batch collation failed: {message}")]
    Collate { message: String },
    #[error("shape mismatch at sample {sample_index}: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        sample_index: usize,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
}

#[cfg(test)]
#[path = "../../tests/trainer/data/legacy_mod_tests.rs"]
mod tests;
