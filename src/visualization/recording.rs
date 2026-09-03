use super::{CaptureContext, CaptureProfile, NodeRole, VisualizationError};
use crate::tensor::{NodeId, Tensor, TensorBase};
use std::{cell::RefCell, collections::HashMap};

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
    label_counters: HashMap<String, usize>,
}

#[derive(Debug)]
enum RecordingState {
    Disabled,
    Capturing(SessionState),
}

thread_local! {
    static STATE: RefCell<RecordingState> = const { RefCell::new(RecordingState::Disabled) };
}

pub(crate) fn begin(
    profile: CaptureProfile,
    context: CaptureContext,
) -> Result<(), VisualizationError> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if matches!(*state, RecordingState::Capturing(_)) {
            return Err(VisualizationError::CaptureAlreadyActive);
        }
        *state = RecordingState::Capturing(SessionState {
            profile,
            context,
            metadata: HashMap::new(),
            label_counters: HashMap::new(),
        });
        Ok(())
    })
}

#[inline(always)]
pub(crate) fn is_active() -> bool {
    STATE.with(|state| matches!(*state.borrow(), RecordingState::Capturing(_)))
}

pub(crate) fn session() -> Result<SessionState, VisualizationError> {
    STATE.with(|state| match &*state.borrow() {
        RecordingState::Capturing(session) => Ok(session.clone()),
        RecordingState::Disabled => Err(VisualizationError::CaptureNotActive),
    })
}

pub(crate) fn disable() {
    STATE.with(|state| *state.borrow_mut() = RecordingState::Disabled);
}

pub(crate) fn clear_temporary() {
    STATE.with(|state| {
        if let RecordingState::Capturing(session) = &mut *state.borrow_mut() {
            session.metadata.clear();
            session.label_counters.clear();
        }
    });
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
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let RecordingState::Capturing(session) = &mut *state else {
            return;
        };
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
    STATE.with(|state| match &*state.borrow() {
        RecordingState::Disabled => 0,
        RecordingState::Capturing(session) => session.metadata.len(),
    })
}
