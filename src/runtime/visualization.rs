use super::*;
use crate::visualization::statistics::tensor_statistics;
use crate::visualization::*;
use std::collections::BTreeMap;

impl State {
    fn graph_snapshot(
        &self,
        profile: CaptureProfile,
        context: CaptureContext,
    ) -> MlResult<GraphSnapshot> {
        let engine = self.engine()?;
        let mut ids = self.tracked.clone();
        let mut operations = HashMap::new();
        let mut edges = Vec::new();
        for id in engine.nodes() {
            let record = engine.get(id).ok_or(AutogradError::NodeNotFound(id))?;
            ids.insert(id);
            operations.insert(id, record.backward.name().to_owned());
            for input in record.inputs {
                ids.insert(input);
                edges.push(GraphEdgeSnapshot {
                    from: input.0,
                    to: id.0,
                    kind: GraphEdgeKind::Data,
                });
            }
        }
        let mut nodes = Vec::new();
        for id in ids {
            let value = self.snapshot(id)?;
            let leaf = self.leaves.contains(&id);
            nodes.push(GraphNodeSnapshot {
                id: id.0,
                label: format!("tensor_{}", id.0),
                role: if self.parameters.contains(&id) {
                    NodeRole::Weight
                } else if !operations.contains_key(&id) {
                    NodeRole::Input
                } else {
                    NodeRole::Variable
                },
                operation: operations.remove(&id),
                shape: value.shape().to_vec(),
                dtype: "f32",
                elements: value.data().len(),
                estimated_bytes: value.data().len() * size_of::<f32>(),
                is_parameter: self.parameters.contains(&id),
                is_leaf: leaf,
                requires_grad: self.tracked.contains(&id),
                retain_grad: self.retained.contains(&id),
                value_stats: if profile == CaptureProfile::Analysis {
                    tensor_statistics(value.data())
                } else {
                    None
                },
                gradient_stats: if profile == CaptureProfile::Analysis {
                    self.gradients
                        .get(&id)
                        .and_then(|g| tensor_statistics(g.data()))
                } else {
                    None
                },
                attributes: BTreeMap::new(),
            });
        }
        nodes.sort_by_key(|n| n.id);
        edges.sort_by_key(|e| (e.from, e.to));
        Ok(GraphSnapshot {
            schema_version: GRAPH_SNAPSHOT_SCHEMA_VERSION,
            profile,
            context,
            nodes,
            edges,
            attributes: BTreeMap::new(),
        })
    }
}

impl ExecutionContext {
    pub fn graph_snapshot(
        &self,
        profile: CaptureProfile,
        context: CaptureContext,
    ) -> MlResult<GraphSnapshot> {
        self.inner
            .state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?
            .graph_snapshot(profile, context)
    }

    /// Capture gradients before normal graph and intermediate-gradient cleanup.
    pub fn backward_snapshot(
        &self,
        output: &Variable,
        options: BackwardOptions<'_>,
        profile: CaptureProfile,
        context: CaptureContext,
    ) -> MlResult<GraphSnapshot> {
        let mut snapshot = None;
        self.backward_observed(output, options, |state| {
            snapshot = Some(state.graph_snapshot(profile, context)?);
            Ok(())
        })?;
        snapshot.ok_or_else(|| MlError::UnsupportedCapability {
            module: "visualization",
            capability: "backward snapshot",
            operation: "capture",
        })
    }
}
