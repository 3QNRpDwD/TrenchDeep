# Explicit graph visualization

Graph capture belongs to an `ExecutionContext`; no global capture session or global
computation graph is used by production. Snapshot and request types are always
available. Without `enableVisualization`, a capture request returns
`DependencyUnavailable` and leaves the graph available for an ordinary backward.

```rust
use trench_deep::{BackwardOptions, ExecutionContext};
use trench_deep::visualization::*;

fn capture() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = ExecutionContext::new();
    let p = ctx.parameter(vec![2.0], &[])?;
    let loss = p.square()?.square()?;
    let snapshot = ctx.backward_snapshot(
        &loss, BackwardOptions::default(), CaptureProfile::Analysis,
        CaptureContext { paradigm: Some("example".into()), ..CaptureContext::default() },
    )?;
    let mut writer = FileSnapshotWriter::builder("captures").render_svg(false).build()?;
    writer.write(&snapshot, "example")?;
    Ok(())
}
```

`graph_snapshot` captures the current graph without running backward.
`backward_snapshot` captures values and available gradients before normal graph and
intermediate-gradient cleanup; it does not make gradients permanently retained.
The returned snapshot owns all data and survives handle/context destruction.
`Structure` omits statistics; `Analysis` includes value and gradient statistics.

Training observers can request a `CaptureProfile` for a `BatchStartContext`.
The common training service collects the requested snapshot during backward and
delivers it only after scope cleanup succeeds, before the successful batch event.
A failed step does not publish its pending success snapshot. RL uses episode
coordinates and the same step capture boundary. Independent contexts can capture
without contending for a global session.

`GraphVisualizationObserver::builder()` accepts a `SnapshotWriter`, capture profile
and selectors: `FirstBatch`, `EpochBatch { epoch, batch }`, or `Episode { episode }`.
Coordinates are one-based; invalid selectors fail at construction. Matching
snapshots are buffered and written by the observer after training. `FileSnapshotWriter`
writes DOT and JSON and can optionally invoke Graphviz for SVG. Graphviz failure is
reported while retaining the DOT/JSON artifacts. `DotEncoder` also works directly
with an owned snapshot and supports Auto, Overview and Detailed profiles.

The snapshot schema and artifact encoders are retained. Overview can hide saved
and parameter nodes, but keeps scalar input nodes. This is graph representation
and collected tensor statistics, not an allocator peak-memory profiler.

Enable `debugging` and a `tracing` subscriber to observe tensor/custom operation
and backward spans. Trainer batch summaries are emitted after progress cleanup;
ordinary training without requested capture does not collect snapshot statistics.
