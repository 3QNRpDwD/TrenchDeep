#![cfg(all(
    feature = "enableVisualization",
    feature = "builtinStorage",
    feature = "builtinKernels"
))]
use trench_deep::visualization::*;
use trench_deep::*;

#[test]
fn snapshot_owns_intermediate_gradients_after_graph_cleanup() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let p = ctx.parameter(vec![2.0], &[])?;
    let intermediate = p.square()?;
    let loss = intermediate.square()?;
    let snapshot = ctx.backward_snapshot(
        &loss,
        BackwardOptions::default(),
        CaptureProfile::Analysis,
        CaptureContext::default(),
    )?;
    assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    assert_eq!(snapshot.nodes.len(), 3);
    assert_eq!(snapshot.edges.len(), 2);
    assert!(
        snapshot
            .nodes
            .iter()
            .all(|node| node.gradient_stats.is_some())
    );
    assert!(intermediate.grad()?.is_none());
    assert_eq!(p.grad()?.map(|g| g.data()[0]), Some(32.0));
    drop(loss);
    drop(intermediate);
    drop(p);
    drop(ctx);
    assert!(DotEncoder::encode(&snapshot).contains("digraph"));
    Ok(())
}

#[test]
fn independent_contexts_capture_without_shared_sessions() -> MlResult<()> {
    let a = ExecutionContext::new();
    let b = ExecutionContext::new();
    let p = a.parameter(vec![2.0], &[])?;
    let q = b.parameter(vec![3.0], &[])?;
    let x = p.square()?;
    let y = q.square()?;
    let snapshot = a.graph_snapshot(CaptureProfile::Structure, CaptureContext::default())?;
    assert_eq!(snapshot.nodes.len(), 2);
    assert!(
        snapshot
            .nodes
            .iter()
            .all(|n| n.value_stats.is_none() && n.gradient_stats.is_none())
    );
    x.backward()?;
    assert_eq!(b.graph_stats()?.graph_nodes, 1);
    y.backward()?;
    Ok(())
}
