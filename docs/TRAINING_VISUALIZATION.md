# Selective Training Graph Visualization

The `enableVisualization` feature compiles visualization support, but it no longer records every
operation automatically. Recording is enabled only for an explicit capture scope or for batches
selected by `GraphVisualizationObserver`.

## Trainer capture

```rust,ignore
use trench_deep::{
    trainer::{CaptureSelector, GraphVisualizationObserver, Trainer},
    visualization::{CaptureProfile, FileSnapshotWriter},
};

let writer = FileSnapshotWriter::builder("graph")
    .render_svg(true)
    .build()?;

let observer = GraphVisualizationObserver::builder()
    .writer(Box::new(writer))
    .selectors([
        CaptureSelector::FirstBatch,
        CaptureSelector::EpochBatch { epoch: 10, batch: 25 },
    ])
    .profile(CaptureProfile::Analysis)
    .build()?;

let trainer = Trainer::default()
    .with_observer(Box::new(observer))
    .unsupervised();
```

Coordinates are one-based. With no selectors, the first successfully trained batch is selected.
DOT and JSON files are always written after progress rendering has finished. SVG is optional and
requires Graphviz. A requested point that is not reached produces a warning without failing the
training result.

`CaptureProfile::Analysis` records shapes, roles, operation connectivity, tensor sizes, and summary
statistics for values and retained gradients. Raw tensor values are never written.

## Direct capture

Code that visualizes a graph without a trainer must now open an explicit scope:

```rust,ignore
use trench_deep::visualization::{
    CaptureProfile, DotProfile, FileSnapshotWriter, SnapshotWriter, VisualizationCapture,
};

let capture = VisualizationCapture::builder(CaptureProfile::Analysis)
    .context(Default::default())
    .begin()?;
// Build the forward graph and call backward while this scope is active.
let snapshot = capture.finish()?;
let mut writer = FileSnapshotWriter::builder("graph")
    .render_svg(true)
    .dot_profile(DotProfile::Auto)
    .build()?;
let report = writer.write(&snapshot, "manual-capture")?;
```

Dropping the scope without calling `finish` disables capture and discards its partial visualization
state. A completed `GraphSnapshot` owns its data and remains valid after the computation graph is
reset.

## Large graph layout

`FileSnapshotWriter` uses `DotProfile::Auto` by default. Graphs with at least 180 nodes are rendered
as a compact top-to-bottom overview. Parameters, scalar constants, and backward-only saved tensors
are omitted from DOT/SVG, leaving the model's operation path and tensor shapes visible. Labels use
compact multiline formatting, while JSON still contains every captured node, edge, and statistic.

Use `.dot_profile(DotProfile::Overview)` to force this layout for a small graph, or
`.dot_profile(DotProfile::Detailed)` to render every node including saved tensors.
