# Dataset and DataLoader

TrenchDeep separates record decoding, sample conversion, batching, and training:

```text
RecordSource -> DatasetBuilder -> InMemoryDataset -> DataLoaderBuilder -> Trainer::fit
```

`DatasetBuilder` eagerly reads records and applies a user transform. `DataLoader`
creates model-facing `Variable` batches, resets its order at every epoch, and uses
the seed configured on `TrainerBuilder` when shuffling.

## Small pre-batched data

The borrowed dataset wrappers remain the shortest path when each `Variable` is
already one complete batch:

```rust,ignore
let result = Trainer::silent().supervised().fit(
    &mut model,
    &mut optimizer,
    SupervisedDataset::new(&inputs, &targets)?,
    EpochSchedule::new(20)?,
)?;
```

`UnsupervisedDataset`, `SemiSupervisedDataset`, and `AutoregressiveDataset` have
the same pre-batched role.

## CSV to supervised batches

```rust,ignore
let dataset = DatasetBuilder::from_source(CsvSource::new("train.csv"))
    .map(|row: CsvRecord| {
        let x1 = row.get("x1").ok_or("missing x1")?.parse::<f32>()
            .map_err(|error| error.to_string())?;
        let x2 = row.get("x2").ok_or("missing x2")?.parse::<f32>()
            .map_err(|error| error.to_string())?;
        let label = row.get("label").ok_or("missing label")?.parse::<f32>()
            .map_err(|error| error.to_string())?;
        Ok(SupervisedSample::new(
            Tensor::from_vec(vec![x1, x2], &[2])?,
            Tensor::from_vec(vec![label], &[1])?,
        ))
    })
    .build()?;

let mut loader = DataLoader::builder(dataset)
    .collator(SupervisedStackCollator::new())
    .batch_size(32)
    .shuffle(true)
    .drop_last(false)
    .build()?;

trainer.fit(&mut model, &mut optimizer, &mut loader, schedule)?;
```

The same `IntoBatchLoader` contract is used by `fit_checkpointed` and `resume`.
At an epoch-boundary resume the loader is reset with `begin_epoch`; an in-epoch
cursor is intentionally not checkpointed.

`JsonLinesSource` decodes one `serde_json::Value` per non-empty line. Decode and
transform failures identify the source path and physical row or line.

## Unsupervised and autoregressive data

Use `UnsupervisedSample` with `UnsupervisedStackCollator` for equal-shaped
unsupervised samples. Autoregressive representations vary between models, so an
autoregressive collator is always explicit. `AutoregressiveStackCollator` is
available for equal-shaped sequences; padding, truncation, masking, and token
encoding belong in a user collator.

```rust,ignore
let mut loader = DataLoader::builder(sequence_dataset)
    .collator(my_padding_collator)
    .batch_size(16)
    .build()?;

trainer.fit(&mut model, &mut optimizer, &mut loader, schedule)?;
```

## Custom sources and collators

A source only decodes records. A collator only converts references to samples
into one model batch, so application-specific storage and padding remain outside
the trainer:

```rust,ignore
struct ApplicationSource(Vec<MyRecord>);

impl RecordSource for ApplicationSource {
    type Record = MyRecord;

    fn read(self) -> MlResult<Vec<Self::Record>> {
        Ok(self.0)
    }
}

struct PaddedSequenceCollator { pad_id: f32 }

impl Collator<AutoregressiveSample> for PaddedSequenceCollator {
    type Batch = AutoregressiveBatch;

    fn collate(
        &mut self,
        samples: &[&AutoregressiveSample],
    ) -> MlResult<Self::Batch> {
        // Pad or truncate `samples`, build a Tensor, then wrap it for the model.
        let tensor = pad_sequences(samples, self.pad_id)?;
        Ok(AutoregressiveBatch { sequences: Variable::new(tensor) })
    }
}
```

## Semi-supervised data

Semi-supervised training combines a dataset of `SupervisedSample` with a dataset
of `UnsupervisedSample`:

```rust,ignore
let mut loader = SemiSupervisedDataLoader::builder(labeled, unlabeled)
    .labeled_collator(SupervisedStackCollator::new())
    .unlabeled_collator(UnsupervisedStackCollator::new())
    .labeled_batch_size(16)
    .unlabeled_batch_size(48)
    .shuffle(true)
    .build()?;
```

The two sides are shuffled independently. The shorter side cycles until the
longer side finishes, preserving the previous trainer behavior.

Reinforcement learning intentionally keeps its `Environment + EpisodeSchedule`
input contract because trajectories are generated online rather than read from a
fixed dataset.
