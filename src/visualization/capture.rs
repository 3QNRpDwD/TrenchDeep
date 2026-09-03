use super::{CaptureContext, CaptureProfile, GraphSnapshot, VisualizationError, recording};

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
        recording::begin(self.profile, self.context)?;
        crate::tensor::ComputationGraph::reset_graph();
        Ok(VisualizationCapture { finished: false })
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
        let session = recording::session()?;
        let snapshot = super::snapshot::build_snapshot(&session);
        crate::tensor::ComputationGraph::reset_graph();
        recording::disable();
        self.finished = true;
        snapshot
    }
}

impl Drop for VisualizationCapture {
    fn drop(&mut self) {
        if !self.finished {
            crate::tensor::ComputationGraph::reset_graph();
            recording::disable();
        }
    }
}
