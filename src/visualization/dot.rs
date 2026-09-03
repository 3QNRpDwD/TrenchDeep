use super::{GraphEdgeKind, GraphSnapshot, NodeRole};

pub struct DotEncoder;

impl DotEncoder {
    pub fn encode(snapshot: &GraphSnapshot) -> String {
        let mut dot = String::from(
            "digraph ComputationGraph {\n  bgcolor=\"#FAFBFC\";\n  rankdir=LR;\n  splines=curved;\n  node [fontname=\"Segoe UI\", fontsize=10, style=filled, fontcolor=white];\n  edge [color=\"#6B7280\", penwidth=1.5];\n",
        );
        for node in &snapshot.nodes {
            let (shape, color) = node_style(&node.role);
            let mut parts = vec![
                node.label.clone(),
                format!("{:?}", node.role),
                format!("{:?}", node.shape),
            ];
            if let Some(operation) = &node.operation {
                parts.insert(2, operation.clone());
            }
            let label = escape(&parts.join("\\n"));
            dot.push_str(&format!(
                "  \"node-{}\" [id=\"node-{}\", label=\"{}\", shape={}, fillcolor=\"{}\"];\n",
                node.id, node.id, label, shape, color,
            ));
        }
        for edge in &snapshot.edges {
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
        dot.push_str("}\n");
        dot
    }
}

fn node_style(role: &NodeRole) -> (&'static str, &'static str) {
    match role {
        NodeRole::Input => ("house", "#10B981"),
        NodeRole::Output => ("invhouse", "#EF4444"),
        NodeRole::Weight => ("diamond", "#8B5CF6"),
        NodeRole::Bias => ("circle", "#F97316"),
        NodeRole::Loss => ("octagon", "#EC4899"),
        NodeRole::Activation => ("doublecircle", "#06B6D4"),
        NodeRole::Function => ("hexagon", "#3B82F6"),
        NodeRole::Variable => ("ellipse", "#F59E0B"),
    }
}

fn escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}
