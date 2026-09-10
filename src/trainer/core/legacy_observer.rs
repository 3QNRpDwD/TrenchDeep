//! Training lifecycle observers and optional graph-capture integration.

#[derive(Debug, Clone)]
pub struct TrainStartContext {
    pub paradigm: &'static str,
    pub total_units: usize,
}

#[derive(Debug, Clone)]
pub struct EpochContext {
    pub paradigm: &'static str,
    pub epoch: usize,
    pub total_epochs: usize,
    pub total_batches: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct BatchStartContext {
    pub paradigm: &'static str,
    pub epoch: usize,
    pub batch: usize,
    pub total_epochs: usize,
    pub total_batches: Option<usize>,
    pub episode: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct BatchEndContext {
    pub batch: BatchStartContext,
    pub loss: f32,
}

#[derive(Debug, Clone)]
pub struct TrainEndContext {
    pub paradigm: &'static str,
    pub units_completed: usize,
    pub interrupted: bool,
}

pub trait TrainingObserver {
    fn on_train_start(&mut self, _context: &TrainStartContext) {}
    fn on_epoch_start(&mut self, _context: &EpochContext) {}
    fn on_batch_end(&mut self, _context: &BatchEndContext) {}
    fn on_epoch_end(&mut self, _context: &EpochContext) {}
    fn on_train_end(&mut self, _context: &TrainEndContext) {}
    fn on_train_error(&mut self, _message: &str) {}

    #[cfg(feature = "enableVisualization")]
    fn capture_profile(
        &self,
        _context: &BatchStartContext,
    ) -> Option<crate::legacy::visualization::CaptureProfile> {
        None
    }

    #[cfg(feature = "enableVisualization")]
    fn on_graph_snapshot(&mut self, _snapshot: crate::legacy::visualization::GraphSnapshot) {}
}

#[cfg(feature = "enableVisualization")]
mod graph_observer {
    use super::*;
    use crate::legacy::visualization::{CaptureProfile, GraphSnapshot, SnapshotWriter, VisualizationError};
    use std::collections::HashSet;

    #[non_exhaustive]
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub enum CaptureSelector {
        FirstBatch,
        EpochBatch { epoch: usize, batch: usize },
        Episode { episode: usize },
    }

    pub struct GraphVisualizationObserverBuilder {
        writer: Option<Box<dyn SnapshotWriter + Send>>,
        selectors: Vec<CaptureSelector>,
        profile: CaptureProfile,
    }

    impl GraphVisualizationObserverBuilder {
        pub fn selectors<I>(mut self, selectors: I) -> Self
        where
            I: IntoIterator<Item = CaptureSelector>,
        {
            self.selectors = selectors.into_iter().collect();
            self
        }

        pub fn profile(mut self, profile: CaptureProfile) -> Self {
            self.profile = profile;
            self
        }

        pub fn writer(mut self, writer: Box<dyn SnapshotWriter + Send>) -> Self {
            self.writer = Some(writer);
            self
        }

        pub fn build(mut self) -> Result<GraphVisualizationObserver, VisualizationError> {
            if self.selectors.is_empty() {
                self.selectors.push(CaptureSelector::FirstBatch);
            }
            for selector in &self.selectors {
                match selector {
                    CaptureSelector::EpochBatch { epoch: 0, .. }
                    | CaptureSelector::EpochBatch { batch: 0, .. }
                    | CaptureSelector::Episode { episode: 0 } => {
                        return Err(VisualizationError::InvalidCaptureCoordinate);
                    }
                    _ => {}
                }
            }
            let requested: HashSet<_> = self.selectors.into_iter().collect();
            Ok(GraphVisualizationObserver {
                writer: self.writer.ok_or(VisualizationError::MissingWriter)?,
                requested,
                captured: HashSet::new(),
                profile: self.profile,
                snapshots: Vec::new(),
            })
        }
    }

    pub struct GraphVisualizationObserver {
        writer: Box<dyn SnapshotWriter + Send>,
        requested: HashSet<CaptureSelector>,
        captured: HashSet<CaptureSelector>,
        profile: CaptureProfile,
        snapshots: Vec<GraphSnapshot>,
    }

    impl GraphVisualizationObserver {
        pub fn builder() -> GraphVisualizationObserverBuilder {
            GraphVisualizationObserverBuilder {
                writer: None,
                selectors: Vec::new(),
                profile: CaptureProfile::Analysis,
            }
        }

        fn matching_selector(&self, context: &BatchStartContext) -> Option<CaptureSelector> {
            if self.requested.contains(&CaptureSelector::FirstBatch)
                && !self.captured.contains(&CaptureSelector::FirstBatch)
            {
                return Some(CaptureSelector::FirstBatch);
            }
            if let Some(episode) = context.episode {
                let selector = CaptureSelector::Episode { episode };
                return (self.requested.contains(&selector) && !self.captured.contains(&selector))
                    .then_some(selector);
            }
            let selector = CaptureSelector::EpochBatch {
                epoch: context.epoch,
                batch: context.batch,
            };
            (self.requested.contains(&selector) && !self.captured.contains(&selector))
                .then_some(selector)
        }

        fn stem(snapshot: &GraphSnapshot) -> String {
            if let Some(episode) = snapshot.context.episode {
                format!("capture-episode-{episode:04}")
            } else {
                format!(
                    "capture-e{:04}-b{:04}",
                    snapshot.context.epoch.unwrap_or(1),
                    snapshot.context.batch.unwrap_or(1)
                )
            }
        }

        fn flush(&mut self) {
            for snapshot in self.snapshots.drain(..) {
                let stem = Self::stem(&snapshot);
                match self.writer.write(&snapshot, &stem) {
                    Ok(report) => {
                        tracing::info!(target: "trench_deep::trainer::visualization", files = ?report.artifacts, "computation graph capture saved");
                        for warning in report.warnings {
                            tracing::warn!(target: "trench_deep::trainer::visualization", artifact = ?warning.artifact, kind = ?warning.kind, message = %warning.message, "computation graph artifact was not generated");
                        }
                    }
                    Err(error) => {
                        tracing::warn!(target: "trench_deep::trainer::visualization", %error, %stem, "failed to save computation graph capture")
                    }
                }
            }
            for missing in self.requested.difference(&self.captured) {
                tracing::warn!(target: "trench_deep::trainer::visualization", selector = ?missing, "requested computation graph capture point was not reached");
            }
        }
    }

    impl TrainingObserver for GraphVisualizationObserver {
        fn on_train_start(&mut self, _context: &TrainStartContext) {
            self.captured.clear();
            self.snapshots.clear();
        }

        fn capture_profile(&self, context: &BatchStartContext) -> Option<CaptureProfile> {
            self.matching_selector(context).map(|_| self.profile)
        }

        fn on_graph_snapshot(&mut self, snapshot: GraphSnapshot) {
            let coordinate = if snapshot.context.episode.is_some() {
                CaptureSelector::Episode {
                    episode: snapshot.context.episode.unwrap(),
                }
            } else {
                CaptureSelector::EpochBatch {
                    epoch: snapshot.context.epoch.unwrap_or(1),
                    batch: snapshot.context.batch.unwrap_or(1),
                }
            };
            self.captured.insert(coordinate);
            if self.requested.contains(&CaptureSelector::FirstBatch) && self.snapshots.is_empty() {
                self.captured.insert(CaptureSelector::FirstBatch);
            }
            self.snapshots.push(snapshot);
        }

        fn on_train_end(&mut self, _context: &TrainEndContext) {
            self.flush();
        }

        fn on_train_error(&mut self, _message: &str) {
            self.flush();
        }
    }

    #[cfg(test)]
    mod tests {
        include!("../../tests/trainer/core/legacy_observer_tests.rs");
    }

    pub use CaptureSelector as PublicCaptureSelector;
    pub use GraphVisualizationObserver as PublicGraphVisualizationObserver;
    pub use GraphVisualizationObserverBuilder as PublicGraphVisualizationObserverBuilder;
}

#[cfg(feature = "enableVisualization")]
pub use graph_observer::{
    PublicCaptureSelector as CaptureSelector,
    PublicGraphVisualizationObserver as GraphVisualizationObserver,
    PublicGraphVisualizationObserverBuilder as GraphVisualizationObserverBuilder,
};
