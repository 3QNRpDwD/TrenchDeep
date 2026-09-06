use super::*;
use crate::visualization::{CaptureContext, CaptureProfile, GraphSnapshot};

impl ExecutionContext {
    pub fn graph_snapshot(
        &self,
        _profile: CaptureProfile,
        _context: CaptureContext,
    ) -> MlResult<GraphSnapshot> {
        Err(missing(
            "visualization",
            "graph capture (enableVisualization)",
            "graph_snapshot",
        ))
    }
    pub fn backward_snapshot(
        &self,
        _output: &Variable,
        _options: BackwardOptions<'_>,
        _profile: CaptureProfile,
        _context: CaptureContext,
    ) -> MlResult<GraphSnapshot> {
        Err(missing(
            "visualization",
            "graph capture (enableVisualization)",
            "backward_snapshot",
        ))
    }
}
