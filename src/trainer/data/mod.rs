//! Eager dataset construction and batching infrastructure.
//!
//! `DatasetBuilder` owns decoding and record-to-sample transforms. `DataLoader`
//! owns batching and epoch ordering. The small borrowed dataset wrappers in
//! [`crate::trainer::api`] remain the pre-batched convenience path.

mod legacy;
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
mod tests {
    use std::{fs, path::PathBuf};

    use super::*;
    use crate::{
        MlResult,
        nn::Parameter,
        tensor::{Tensor, TensorBase},
        trainer::TrainingRuntime,
    };

    fn tensor(value: f32, shape: &[usize]) -> Tensor {
        Tensor::from_vec(vec![value; shape.iter().product()], shape).unwrap()
    }

    #[test]
    fn memory_dataset_and_loader_stack_samples() -> MlResult<()> {
        let dataset = DatasetBuilder::from_source(MemorySource::new(vec![1.0, 2.0, 3.0]))
            .map(|value| {
                Ok(SupervisedSample::new(
                    tensor(value, &[2]),
                    tensor(value, &[1]),
                ))
            })
            .build()?;
        let mut loader = DataLoader::builder(dataset)
            .collator(SupervisedStackCollator::new())
            .batch_size(2)
            .shuffle(false)
            .build()?;
        loader.begin_epoch(0, &TrainingRuntime::new(0))?;
        let first = loader.next_batch()?.unwrap();
        let second = loader.next_batch()?.unwrap();
        assert_eq!(first.inputs.tensor().shape(), &[2, 2]);
        assert_eq!(first.targets.tensor().shape(), &[2, 1]);
        assert_eq!(second.inputs.tensor().shape(), &[1, 2]);
        assert!(loader.next_batch()?.is_none());
        Ok(())
    }

    #[test]
    fn drop_last_and_invalid_batch_configuration_are_validated() -> MlResult<()> {
        let make_dataset = || {
            InMemoryDataset::new(vec![
                UnsupervisedSample::new(tensor(1.0, &[1])),
                UnsupervisedSample::new(tensor(2.0, &[1])),
                UnsupervisedSample::new(tensor(3.0, &[1])),
            ])
            .unwrap()
        };
        let loader = DataLoader::builder(make_dataset())
            .collator(UnsupervisedStackCollator::new())
            .batch_size(2)
            .drop_last(true)
            .build()?;
        assert_eq!(loader.batch_count(), Some(1));
        assert!(
            DataLoader::builder(make_dataset())
                .collator(UnsupervisedStackCollator::new())
                .batch_size(0)
                .build()
                .is_err()
        );
        assert!(
            DataLoader::builder(make_dataset())
                .collator(UnsupervisedStackCollator::new())
                .batch_size(4)
                .drop_last(true)
                .build()
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn stack_collator_reports_sample_shape() {
        let samples = [
            UnsupervisedSample::new(tensor(1.0, &[2])),
            UnsupervisedSample::new(tensor(1.0, &[3])),
        ];
        let refs = samples.iter().collect::<Vec<_>>();
        let error = match UnsupervisedStackCollator::new().collate(&refs) {
            Ok(_) => panic!("shape mismatch must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("sample 1"));
        assert!(error.to_string().contains("expected [2], got [3]"));
    }

    #[test]
    fn custom_collator_errors_are_classified() -> MlResult<()> {
        let dataset = InMemoryDataset::new(vec![UnsupervisedSample::new(tensor(1.0, &[1]))])?;
        let mut loader = DataLoader::builder(dataset)
            .collator(|_: &[&UnsupervisedSample]| -> MlResult<UnsupervisedBatch> {
                Err("custom failure".into())
            })
            .shuffle(false)
            .build()?;
        loader.begin_epoch(0, &TrainingRuntime::new(0))?;
        let error = loader.next_batch().unwrap_err();
        assert!(error.to_string().contains("batch collation failed: custom failure"));
        Ok(())
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("data_loader_{}_{}", std::process::id(), name))
    }

    #[test]
    fn csv_and_jsonl_sources_preserve_locations() -> MlResult<()> {
        let csv_path = temp_path("records.csv");
        fs::create_dir_all(csv_path.parent().unwrap()).unwrap();
        fs::write(&csv_path, "x,label\n1.5,2\n").unwrap();
        let csv = DatasetBuilder::from_source(CsvSource::new(&csv_path))
            .map(|row: CsvRecord| {
                let x = row.get("x").unwrap().parse::<f32>().unwrap();
                Ok(SupervisedSample::new(tensor(x, &[1]), tensor(2.0, &[1])))
            })
            .build()?;
        assert_eq!(csv.len(), 1);

        let json_path = temp_path("records.jsonl");
        fs::write(&json_path, "\n{\"value\": 1}\n{\"value\": 2}\n").unwrap();
        let json = JsonLinesSource::new(&json_path).read()?;
        assert_eq!(json.len(), 2);
        assert_eq!(json[1]["value"], 2);

        fs::write(&json_path, "\n{\"value\": 1}\n{broken}\n").unwrap();
        let error = JsonLinesSource::new(&json_path).read().unwrap_err();
        assert!(error.to_string().contains("line/row 3"));
        let _ = fs::remove_file(csv_path);
        let _ = fs::remove_file(json_path);
        Ok(())
    }

    #[test]
    fn transform_error_contains_file_and_row() {
        let path = temp_path("transform.csv");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "x\nnot-a-number\n").unwrap();
        let result = DatasetBuilder::from_source(CsvSource::new(&path))
            .map(|row: CsvRecord| -> MlResult<UnsupervisedSample> {
                let value = row
                    .get("x")
                    .unwrap()
                    .parse::<f32>()
                    .map_err(|error| error.to_string())?;
                Ok(UnsupervisedSample::new(tensor(value, &[1])))
            })
            .build();
        let error = match result {
            Ok(_) => panic!("invalid transform must fail"),
            Err(error) => error,
        };
        let message = error.to_string();
        assert!(message.contains(path.to_string_lossy().as_ref()));
        assert!(message.contains(":2"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn semi_supervised_loader_wraps_shorter_side() -> MlResult<()> {
        let labeled = InMemoryDataset::new(vec![SupervisedSample::new(
            tensor(1.0, &[1]),
            tensor(1.0, &[1]),
        )])?;
        let unlabeled = InMemoryDataset::new(vec![
            UnsupervisedSample::new(tensor(1.0, &[1])),
            UnsupervisedSample::new(tensor(2.0, &[1])),
            UnsupervisedSample::new(tensor(3.0, &[1])),
        ])?;
        let mut loader = SemiSupervisedDataLoader::builder(labeled, unlabeled)
            .labeled_collator(SupervisedStackCollator::new())
            .unlabeled_collator(UnsupervisedStackCollator::new())
            .shuffle(false)
            .build()?;
        assert_eq!(loader.batch_count(), Some(3));
        loader.begin_epoch(0, &TrainingRuntime::new(0))?;
        for _ in 0..3 {
            assert!(loader.next_batch()?.is_some());
        }
        assert!(loader.next_batch()?.is_none());
        Ok(())
    }

    #[test]
    fn trainer_runtime_seed_controls_shuffle_order() -> MlResult<()> {
        fn make_loader() -> MlResult<impl BatchLoader<Batch = UnsupervisedBatch>> {
            let samples = (0..16)
                .map(|value| UnsupervisedSample::new(tensor(value as f32, &[1])))
                .collect();
            let dataset = InMemoryDataset::new(samples)?;
            DataLoader::builder(dataset)
                .collator(UnsupervisedStackCollator::new())
                .shuffle(true)
                .build()
        }
        fn order(
            loader: &mut impl BatchLoader<Batch = UnsupervisedBatch>,
            runtime: &TrainingRuntime,
        ) -> MlResult<Vec<f32>> {
            loader.begin_epoch(0, runtime)?;
            let mut values = Vec::new();
            while let Some(batch) = loader.next_batch()? {
                values.push(batch.samples.tensor().data()[0]);
            }
            Ok(values)
        }

        let mut a = make_loader()?;
        let mut b = make_loader()?;
        let mut c = make_loader()?;
        let left = order(&mut a, &TrainingRuntime::new(7))?;
        let right = order(&mut b, &TrainingRuntime::new(7))?;
        let other = order(&mut c, &TrainingRuntime::new(8))?;
        assert_eq!(left, right);
        assert_ne!(left, other);
        Ok(())
    }
}
