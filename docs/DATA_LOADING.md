# Dataset and DataLoader

All tensors belong to an explicit `ExecutionContext`. Keep that context alive while
loading and training. Sources, datasets and collators are independent contracts;
there is no second Context-specific loader hierarchy.

```rust
use trench_deep::{ExecutionContext, MlResult};
use trench_deep::nn::LinearRegression;
use trench_deep::optimizer::{Optimizer, SGD};
use trench_deep::trainer::*;

fn train() -> MlResult<()> {
    let ctx = ExecutionContext::builder().initialization_seed(7).build();
    let dataset = InMemoryDataset::new(vec![
        SupervisedSample::new(ctx.tensor(vec![0.0, 1.0], &[2])?, ctx.tensor(vec![1.0], &[1])?),
        SupervisedSample::new(ctx.tensor(vec![1.0, 0.0], &[2])?, ctx.tensor(vec![2.0], &[1])?),
    ])?;
    let mut loader = DataLoader::builder(dataset)
        .collator(SupervisedStackCollator::new())
        .batch_size(2).shuffle(true).drop_last(false).build()?;
    let mut model = LinearRegression::new(&ctx, 2, 1)?;
    let mut optimizer = SGD::new(&ctx, 0.01)?;
    optimizer.register_all(&model.parameters())?;
    SupervisedTrainer::silent(&ctx).with_seed(13)
        .fit(&mut model, &mut optimizer, &mut loader, EpochSchedule::new(5)?)?;
    Ok(())
}
```

Run training with `enableBackward`, or inject an autograd provider. Stack collators
preserve the sample context, validate shapes and context identity, and return owned
batch handles. Missing samples produce `DataError::MissingSample`; source/transform
and dependency errors propagate through the common training scope.

For eager ingestion, use `DatasetBuilder::from_source(source).map(transform).build()`.
`MemorySource` accepts records already in memory. `CsvSource` exposes decoded
`CsvRecord` fields by name or index and supports header/delimiter settings.
`JsonLinesSource` decodes one JSON value per line. Transforms construct sample
handles using the same context as the model. Location-aware source failures retain
path and line/row information. There is no asynchronous or streaming source promise.

`Dataset`, `Collator`, `BatchLoader`, and `IntoBatchLoader` are public extension
points. `BatchLoader::begin_epoch` receives `TrainingRuntime`; its `shuffle` method
uses the data-order stream, independent of RL action draws. A custom loader does
not import CPU kernels, graph representations or tensor registries.

For already batched input, `SupervisedDataset::new(&ctx, inputs, targets)`,
`UnsupervisedDataset`, `SemiSupervisedDataset`, and `AutoregressiveDataset` implement
`IntoBatchLoader` through the same training service. Prebatched inputs are shuffled
with the trainer seed. Constructors validate context identity; malformed manually
constructed public dataset values return errors when loading begins.

The generic loader validates batch size and `drop_last` when building, regardless
of setter order. Semi-supervised loaders support separate labeled/unlabeled batch
sizes. Training loss is weighted by samples, labeled samples, or target tokens as
appropriate. Bigram accepts `[length, vocabulary]` and `[batch, length, vocabulary]`,
shifts each sequence independently, and rejects requested padding explicitly.

See [P1 API](p1/P1_API.md) for providers and [P1 status](p1/P1_STATUS.md) for verified cases.
