use super::{CaptureContext, CaptureProfile, NodeRole, capture};
use crate::tensor::{NodeId, Tensor, TensorBase};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(crate) struct RecordedMetadata {
    pub label: String,
    pub role: NodeRole,
}

#[derive(Debug, Clone)]
pub(crate) struct SessionState {
    pub profile: CaptureProfile,
    pub context: CaptureContext,
    pub metadata: HashMap<NodeId, RecordedMetadata>,
    pub(crate) label_counters: HashMap<String, usize>,
}

impl SessionState {
    pub(crate) fn new(profile: CaptureProfile, context: CaptureContext) -> Self {
        Self {
            profile,
            context,
            metadata: HashMap::new(),
            label_counters: HashMap::new(),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.metadata.clear();
        self.label_counters.clear();
    }
}

#[inline(always)]
pub(crate) fn is_active() -> bool {
    capture::is_active()
}

pub(crate) fn clear_temporary() {
    capture::clear_temporary();
}

pub(crate) fn record_node(
    id: NodeId,
    tensor: &Tensor,
    explicit_label: Option<&str>,
    explicit_role: Option<&NodeRole>,
    fallback_role: NodeRole,
) {
    if !is_active() {
        return;
    }
    capture::with_session_mut(|session| {
        if session.metadata.contains_key(&id) {
            return;
        }
        let role = explicit_role.cloned().unwrap_or(fallback_role);
        let base = explicit_label
            .map(str::to_owned)
            .unwrap_or_else(|| inferred_label(tensor.shape(), &role));
        let count = session.label_counters.entry(base.clone()).or_insert(0);
        *count += 1;
        let label = if *count == 1 {
            base
        } else {
            format!("{base}_{}", count)
        };
        session
            .metadata
            .insert(id, RecordedMetadata { label, role });
    });
}

fn inferred_label(shape: &[usize], role: &NodeRole) -> String {
    if !matches!(role, NodeRole::Variable | NodeRole::Input) {
        return format!("{:?}", role).to_lowercase();
    }
    match shape {
        [] | [1] | [1, 1] => "scalar".into(),
        [n] => format!("vector_{n}"),
        [rows, columns] => format!("matrix_{rows}x{columns}"),
        _ => format!("tensor_{}d", shape.len()),
    }
}

#[cfg(test)]
pub(crate) fn metadata_len() -> usize {
    let mut length = 0;
    capture::with_session_mut(|session| length = session.metadata.len());
    length
}
