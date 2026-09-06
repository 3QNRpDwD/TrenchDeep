//! Shared graph-capture lifecycle for every training paradigm.

use super::{BatchStartContext, MlResult, TrainerCore};
use crate::visualization::{CaptureContext, GraphSnapshot, VisualizationCapture};

pub(crate) struct TrainingGraphCapture {
    context: BatchStartContext,
    capture: Option<VisualizationCapture>,
}

pub(crate) struct PendingGraphSnapshot {
    context: BatchStartContext,
    snapshot: Option<GraphSnapshot>,
}

impl TrainerCore {
    pub(crate) fn begin_graph_capture(
        &self,
        context: &BatchStartContext,
    ) -> MlResult<TrainingGraphCapture> {
        let capture = self
            .requested_capture_profile(context)
            .map(|profile| {
                let capture_context = if let Some(episode) = context.episode {
                    CaptureContext {
                        paradigm: Some(context.paradigm.to_owned()),
                        epoch: None,
                        batch: None,
                        episode: Some(episode),
                    }
                } else {
                    CaptureContext {
                        paradigm: Some(context.paradigm.to_owned()),
                        epoch: Some(context.epoch),
                        batch: Some(context.batch),
                        episode: None,
                    }
                };
                VisualizationCapture::builder(profile)
                    .context(capture_context)
                    .begin()
            })
            .transpose()?;

        Ok(TrainingGraphCapture {
            context: context.clone(),
            capture,
        })
    }
}

impl TrainingGraphCapture {
    /// Finalizes after backward/validation, while gradients are still present.
    pub(crate) fn finish(mut self) -> MlResult<PendingGraphSnapshot> {
        let snapshot = self
            .capture
            .take()
            .map(VisualizationCapture::finish)
            .transpose()?;
        Ok(PendingGraphSnapshot {
            context: self.context,
            snapshot,
        })
    }
}

impl PendingGraphSnapshot {
    /// Commit only after the optimizer step succeeds.
    pub(crate) fn commit(self, core: &TrainerCore) {
        if let Some(snapshot) = self.snapshot {
            core.deliver_graph_snapshot(&self.context, snapshot);
        }
    }
}
