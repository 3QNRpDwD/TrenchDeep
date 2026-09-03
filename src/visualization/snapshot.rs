use super::{VisualizationError, recording::SessionState, statistics::tensor_statistics};
use crate::tensor::{COMPUTATION_GRAPH, TensorBase};
use serde::Serialize;
use std::collections::BTreeMap;

pub const GRAPH_SNAPSHOT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureProfile {
    Structure,
    Analysis,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct CaptureContext {
    pub paradigm: Option<String>,
    pub epoch: Option<usize>,
    pub batch: Option<usize>,
    pub episode: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeRole {
    Variable,
    Function,
    Saved,
    Input,
    Weight,
    Bias,
    Loss,
    Activation,
    Output,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorStatistics {
    pub min: Option<f32>,
    pub max: Option<f32>,
    pub mean: Option<f32>,
    pub std_dev: Option<f32>,
    pub l1_norm: Option<f32>,
    pub l2_norm: Option<f32>,
    pub zeros: usize,
    pub nan: usize,
    pub positive_infinity: usize,
    pub negative_infinity: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum GraphAttributeValue {
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Integers(Vec<i64>),
    Floats(Vec<f64>),
    Strings(Vec<String>),
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphNodeSnapshot {
    pub id: u64,
    pub label: String,
    pub role: NodeRole,
    pub operation: Option<String>,
    pub shape: Vec<usize>,
    pub dtype: &'static str,
    pub elements: usize,
    pub estimated_bytes: usize,
    pub is_parameter: bool,
    pub is_leaf: bool,
    pub requires_grad: bool,
    pub retain_grad: bool,
    pub value_stats: Option<TensorStatistics>,
    pub gradient_stats: Option<TensorStatistics>,
    pub attributes: BTreeMap<String, GraphAttributeValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphEdgeKind {
    Data,
    Gradient,
    Control,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphEdgeSnapshot {
    pub from: u64,
    pub to: u64,
    pub kind: GraphEdgeKind,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphSnapshot {
    pub schema_version: u32,
    pub profile: CaptureProfile,
    pub context: CaptureContext,
    pub nodes: Vec<GraphNodeSnapshot>,
    pub edges: Vec<GraphEdgeSnapshot>,
    pub attributes: BTreeMap<String, GraphAttributeValue>,
}

pub(crate) fn build_snapshot(session: &SessionState) -> Result<GraphSnapshot, VisualizationError> {
    let graph = COMPUTATION_GRAPH.with(|graph| {
        let graph = graph
            .lock()
            .map_err(|_| VisualizationError::GraphLockPoisoned)?;
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        graph.visit_nodes(|node| {
            let metadata = session.metadata.get(&node.id);
            let role = metadata
                .map(|metadata| metadata.role.clone())
                .unwrap_or_else(|| {
                    if node.is_leaf {
                        NodeRole::Input
                    } else {
                        NodeRole::Variable
                    }
                });
            let data = node.tensor.data();
            let gradient = node.grad.data();
            nodes.push(GraphNodeSnapshot {
                id: node.id.as_raw(),
                label: metadata
                    .map(|metadata| metadata.label.clone())
                    .unwrap_or_else(|| format!("node_{}", node.id.as_raw())),
                is_parameter: matches!(role, NodeRole::Weight | NodeRole::Bias),
                role,
                operation: node.operation.map(str::to_owned),
                shape: node.tensor.shape().to_vec(),
                dtype: "f32",
                elements: data.len(),
                estimated_bytes: data.len() * std::mem::size_of::<f32>(),
                is_leaf: node.is_leaf,
                requires_grad: node.requires_grad,
                retain_grad: node.requires_grad,
                value_stats: (session.profile == CaptureProfile::Analysis)
                    .then(|| tensor_statistics(data))
                    .flatten(),
                gradient_stats: (session.profile == CaptureProfile::Analysis)
                    .then(|| tensor_statistics(gradient))
                    .flatten(),
                attributes: BTreeMap::new(),
            });
            edges.extend(node.inputs.iter().map(|input| GraphEdgeSnapshot {
                from: input.as_raw(),
                to: node.id.as_raw(),
                kind: GraphEdgeKind::Data,
            }));
        });
        Ok::<_, VisualizationError>((nodes, edges))
    })?;
    let (mut nodes, mut edges) = graph;
    nodes.sort_by_key(|node| node.id);
    edges.sort_by_key(|edge| (edge.from, edge.to));
    Ok(GraphSnapshot {
        schema_version: GRAPH_SNAPSHOT_SCHEMA_VERSION,
        profile: session.profile,
        context: session.context.clone(),
        nodes,
        edges,
        attributes: BTreeMap::new(),
    })
}
