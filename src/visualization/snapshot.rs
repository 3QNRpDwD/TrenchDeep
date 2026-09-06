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
