use super::{GraphEdgeKind, GraphSnapshot, NodeRole};
use std::collections::HashSet;

const LARGE_GRAPH_THRESHOLD: usize = 180;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DotProfile {
    Auto,
    Overview,
    Detailed,
}

pub struct DotEncoder;

impl DotEncoder {
    pub fn encode(snapshot: &GraphSnapshot) -> String {
        Self::encode_with_profile(snapshot, DotProfile::Auto)
    }

    pub fn encode_with_profile(snapshot: &GraphSnapshot, profile: DotProfile) -> String {
        let overview = match profile {
            DotProfile::Auto => snapshot.nodes.len() >= LARGE_GRAPH_THRESHOLD,
            DotProfile::Overview => true,
            DotProfile::Detailed => false,
        };
        let hidden: HashSet<u64> = if overview {
            snapshot
                .nodes
                .iter()
                .filter(|node| {
                    matches!(
                        node.role,
                        NodeRole::Saved | NodeRole::Weight | NodeRole::Bias
                    ) || (node.role == NodeRole::Input && node.elements <= 1)
                })
                .map(|node| node.id)
                .collect()
        } else {
            HashSet::new()
        };
        let mut dot = String::from("digraph ComputationGraph {\n");
        if overview {
            dot.push_str("  graph [bgcolor=\"#FAFBFC\", rankdir=TB, splines=polyline, concentrate=true, newrank=true, nodesep=0.12, ranksep=0.35, pad=0.2, ratio=compress, size=\"16,100\"];\n");
            dot.push_str("  node [fontname=\"Segoe UI\", fontsize=9, style=\"rounded,filled\", fontcolor=white, margin=\"0.06,0.04\", height=0.25];\n");
            dot.push_str("  edge [color=\"#94A3B8\", penwidth=1.0, arrowsize=0.55];\n");
        } else {
            dot.push_str("  graph [bgcolor=\"#FAFBFC\", rankdir=LR, splines=curved, nodesep=0.25, ranksep=0.5, pad=0.2];\n");
            dot.push_str(
                "  node [fontname=\"Segoe UI\", fontsize=10, style=filled, fontcolor=white];\n",
            );
            dot.push_str("  edge [color=\"#6B7280\", penwidth=1.5, arrowsize=0.7];\n");
        }

        for node in snapshot
            .nodes
            .iter()
            .filter(|node| !hidden.contains(&node.id))
        {
            let (shape, color) = node_style(&node.role, overview);
            let label = node_label(node, overview);
            dot.push_str(&format!(
                "  \"node-{}\" [id=\"node-{}\", label=\"{}\", shape={}, fillcolor=\"{}\"];\n",
                node.id,
                node.id,
                escape(&label),
                shape,
                color,
            ));
        }
        for edge in &snapshot.edges {
            if hidden.contains(&edge.from) || hidden.contains(&edge.to) {
                continue;
            }
            let style = match edge.kind {
                GraphEdgeKind::Data => "solid",
                GraphEdgeKind::Gradient => "dashed",
                GraphEdgeKind::Control => "dotted",
            };
            dot.push_str(&format!(
                "  \"node-{}\" -> \"node-{}\" [style={}];\n",
                edge.from, edge.to, style,
            ));
        }
        if overview && !hidden.is_empty() {
            dot.push_str(&format!(
                "  graph [label=\"Overview: {} auxiliary nodes hidden (parameters, scalar constants, saved tensors); full data in JSON\", labelloc=b, labeljust=l, fontname=\"Segoe UI\", fontsize=9, fontcolor=\"#64748B\"];\n",
                hidden.len(),
            ));
        }
        dot.push_str("}\n");
        dot
    }
}

fn node_label(node: &super::GraphNodeSnapshot, overview: bool) -> String {
    if overview {
        match node.role {
            NodeRole::Weight => format!("W {}\n{:?}", node.label, node.shape),
            NodeRole::Bias => format!("B {}\n{:?}", node.label, node.shape),
            _ => match &node.operation {
                Some(operation) => format!("{}\n{}\n{:?}", node.label, operation, node.shape),
                None => format!("{}\n{:?}", node.label, node.shape),
            },
        }
    } else {
        let mut parts = vec![
            node.label.clone(),
            format!("{:?}", node.role),
            format!("{:?}", node.shape),
        ];
        if let Some(operation) = &node.operation {
            parts.insert(2, operation.clone());
        }
        parts.join("\n")
    }
}

fn node_style(role: &NodeRole, overview: bool) -> (&'static str, &'static str) {
    match role {
        NodeRole::Input => ("house", "#10B981"),
        NodeRole::Output => ("invhouse", "#EF4444"),
        NodeRole::Weight => (if overview { "box" } else { "diamond" }, "#8B5CF6"),
        NodeRole::Bias => (if overview { "box" } else { "circle" }, "#F97316"),
        NodeRole::Loss => ("octagon", "#EC4899"),
        NodeRole::Activation => ("doublecircle", "#06B6D4"),
        NodeRole::Function => ("hexagon", "#3B82F6"),
        NodeRole::Variable => ("ellipse", "#F59E0B"),
        NodeRole::Saved => ("note", "#64748B"),
    }
}

fn escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visualization::{
        CaptureContext, CaptureProfile, GRAPH_SNAPSHOT_SCHEMA_VERSION, GraphNodeSnapshot,
    };
    use std::collections::BTreeMap;

    fn node(id: u64, role: NodeRole) -> GraphNodeSnapshot {
        GraphNodeSnapshot {
            id,
            label: format!("n{id}"),
            role,
            operation: None,
            shape: vec![1],
            dtype: "f32",
            elements: 1,
            estimated_bytes: 4,
            is_parameter: false,
            is_leaf: true,
            requires_grad: false,
            retain_grad: false,
            value_stats: None,
            gradient_stats: None,
            attributes: BTreeMap::new(),
        }
    }

    #[test]
    fn overview_hides_saved_nodes_but_detailed_keeps_them() {
        let snapshot = GraphSnapshot {
            schema_version: GRAPH_SNAPSHOT_SCHEMA_VERSION,
            profile: CaptureProfile::Structure,
            context: CaptureContext::default(),
            nodes: vec![node(1, NodeRole::Input), node(2, NodeRole::Saved)],
            edges: Vec::new(),
            attributes: BTreeMap::new(),
        };
        let overview = DotEncoder::encode_with_profile(&snapshot, DotProfile::Overview);
        assert!(overview.contains("node-1"));
        assert!(!overview.contains("id=\"node-2\""));
        assert!(overview.contains("1 auxiliary nodes hidden"));
        let detailed = DotEncoder::encode_with_profile(&snapshot, DotProfile::Detailed);
        assert!(detailed.contains("id=\"node-2\""));
    }

    #[test]
    fn labels_use_graphviz_line_breaks_not_literal_backslash_n() {
        let mut snapshot_node = node(1, NodeRole::Variable);
        snapshot_node.operation = Some("Add".into());
        let snapshot = GraphSnapshot {
            schema_version: GRAPH_SNAPSHOT_SCHEMA_VERSION,
            profile: CaptureProfile::Structure,
            context: CaptureContext::default(),
            nodes: vec![snapshot_node],
            edges: Vec::new(),
            attributes: BTreeMap::new(),
        };
        let dot = DotEncoder::encode_with_profile(&snapshot, DotProfile::Detailed);
        assert!(dot.contains("n1\\nVariable\\nAdd\\n[1]"));
        assert!(!dot.contains("n1\\\\nVariable"));
    }
}
