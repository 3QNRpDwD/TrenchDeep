# TrenchDeep
This framework is a project that I am independently designing/developing to study about deep learning. 
If you have any better ideas for the project, please contact us here at 2QRNpDwD@gmail.com .

Training data can be supplied either as small pre-batched in-memory datasets or
through the eager `DatasetBuilder -> DataLoader` pipeline with memory, CSV, and
JSON Lines sources. See [Dataset and DataLoader](docs/DATA_LOADING.md).

Computation graph visualization is captured only for explicitly selected training batches or a
manual capture scope. See [Selective Training Graph Visualization](docs/TRAINING_VISUALIZATION.md).

The current public API uses an explicit `ExecutionContext`, `Tensor`, `Variable`,
and stable `Parameter`. See [the P1 public API](docs/P1_API.md) and
[implementation status and retained comparisons](docs/P1_STATUS.md). These documents
describe the current API alongside the data loading and visualization guides.
