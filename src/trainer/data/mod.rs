//! Eager dataset construction and batching infrastructure.
//!
//! `DatasetBuilder` owns decoding and record-to-sample transforms. `DataLoader`
//! owns batching and epoch ordering. The small borrowed dataset wrappers in
//! [`crate::trainer::api`] remain the pre-batched convenience path.


mod loader;
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

pub use crate::DataError;
mod prebatched;
pub use prebatched::*;
