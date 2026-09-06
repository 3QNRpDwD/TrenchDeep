use super::{CaptureContext, CaptureProfile, GraphSnapshot, VisualizationError, recording};
use std::cell::RefCell;

#[derive(Debug)]
enum CaptureState {
    Disabled,
    Capturing(recording::SessionState),
}

thread_local! {
    static STATE: RefCell<CaptureState> = const { RefCell::new(CaptureState::Disabled) };
}

pub struct VisualizationCaptureBuilder {
    profile: CaptureProfile,
    context: CaptureContext,
}

impl VisualizationCaptureBuilder {
    pub fn context(mut self, context: CaptureContext) -> Self {
        self.context = context;
        self
    }

    pub fn begin(self) -> Result<VisualizationCapture, VisualizationError> {
        begin_session(self.profile, self.context)?;
        let capture = VisualizationCapture { finished: false };
        crate::tensor::ComputationGraph::reset_graph();
        Ok(capture)
    }
}

pub struct VisualizationCapture {
    finished: bool,
}

impl VisualizationCapture {
    pub fn builder(profile: CaptureProfile) -> VisualizationCaptureBuilder {
        VisualizationCaptureBuilder {
            profile,
            context: CaptureContext::default(),
        }
    }

    pub fn finish(mut self) -> Result<GraphSnapshot, VisualizationError> {
        let session = session()?;
        let snapshot = super::snapshot::build_snapshot(&session);
        crate::tensor::ComputationGraph::reset_graph();
        disable();
        self.finished = true;
        snapshot
    }
}

impl Drop for VisualizationCapture {
    fn drop(&mut self) {
        if !self.finished {
            crate::tensor::ComputationGraph::reset_graph();
            disable();
        }
    }
}

fn begin_session(
    profile: CaptureProfile,
    context: CaptureContext,
) -> Result<(), VisualizationError> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if matches!(*state, CaptureState::Capturing(_)) {
            return Err(VisualizationError::CaptureAlreadyActive);
        }
        *state = CaptureState::Capturing(recording::SessionState::new(profile, context));
        Ok(())
    })
}

pub(crate) fn is_active() -> bool {
    STATE.with(|state| matches!(*state.borrow(), CaptureState::Capturing(_)))
}

fn session() -> Result<recording::SessionState, VisualizationError> {
    STATE.with(|state| match &*state.borrow() {
        CaptureState::Capturing(session) => Ok(session.clone()),
        CaptureState::Disabled => Err(VisualizationError::CaptureNotActive),
    })
}

fn disable() {
    STATE.with(|state| *state.borrow_mut() = CaptureState::Disabled);
}

pub(crate) fn clear_temporary() {
    with_session_mut(|session| session.clear());
}

pub(crate) fn with_session_mut(operation: impl FnOnce(&mut recording::SessionState)) {
    STATE.with(|state| {
        if let CaptureState::Capturing(session) = &mut *state.borrow_mut() {
            operation(session);
        }
    });
}
