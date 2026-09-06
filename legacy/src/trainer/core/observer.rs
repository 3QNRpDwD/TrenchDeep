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
    ) -> Option<crate::visualization::CaptureProfile> {
        None
    }

    #[cfg(feature = "enableVisualization")]
    fn on_graph_snapshot(&mut self, _snapshot: crate::visualization::GraphSnapshot) {}
}

#[cfg(feature = "enableVisualization")]
mod graph_observer {
    use super::*;
    use crate::visualization::{CaptureProfile, GraphSnapshot, SnapshotWriter, VisualizationError};
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
        use super::*;
        use crate::visualization::{
            CaptureContext, GRAPH_SNAPSHOT_SCHEMA_VERSION, VisualizationError, WriteReport,
        };
        use std::{
            collections::BTreeMap,
            sync::{Arc, Mutex},
        };

        struct NullWriter;

        impl SnapshotWriter for NullWriter {
            fn write(
                &mut self,
                _snapshot: &GraphSnapshot,
                _stem: &str,
            ) -> Result<WriteReport, VisualizationError> {
                Ok(WriteReport::default())
            }
        }

        fn builder() -> GraphVisualizationObserverBuilder {
            GraphVisualizationObserver::builder().writer(Box::new(NullWriter))
        }

        fn context(epoch: usize, batch: usize) -> BatchStartContext {
            BatchStartContext {
                paradigm: "test",
                epoch,
                batch,
                total_epochs: 20,
                total_batches: Some(100),
                episode: None,
            }
        }

        #[test]
        fn selectors_are_deduplicated_and_default_to_first_batch() {
            let default = builder().build().unwrap();
            assert_eq!(default.requested.len(), 1);
            assert!(default.requested.contains(&CaptureSelector::FirstBatch));

            let selected = builder()
                .selectors([
                    CaptureSelector::EpochBatch { epoch: 2, batch: 3 },
                    CaptureSelector::EpochBatch { epoch: 2, batch: 3 },
                ])
                .build()
                .unwrap();
            assert_eq!(selected.requested.len(), 1);
            assert!(selected.matching_selector(&context(2, 3)).is_some());
            assert!(selected.matching_selector(&context(1, 1)).is_none());
        }

        #[test]
        fn zero_based_coordinates_are_rejected() {
            assert!(
                builder()
                    .selectors([CaptureSelector::EpochBatch { epoch: 0, batch: 1 }])
                    .build()
                    .is_err()
            );
        }

        #[test]
        fn episode_selector_uses_one_based_episode_coordinate() {
            let observer = builder()
                .selectors([CaptureSelector::Episode { episode: 3 }])
                .build()
                .unwrap();
            let mut episode = context(3, 1);
            episode.episode = Some(3);
            assert_eq!(
                observer.matching_selector(&episode),
                Some(CaptureSelector::Episode { episode: 3 })
            );
        }

        struct RecordingWriter(Arc<Mutex<Vec<String>>>);

        impl SnapshotWriter for RecordingWriter {
            fn write(
                &mut self,
                _snapshot: &GraphSnapshot,
                stem: &str,
            ) -> Result<WriteReport, VisualizationError> {
                self.0.lock().unwrap().push(stem.to_owned());
                Ok(WriteReport::default())
            }
        }

        #[test]
        fn injected_writer_runs_only_at_train_end() {
            let writes = Arc::new(Mutex::new(Vec::new()));
            let mut observer = GraphVisualizationObserver::builder()
                .writer(Box::new(RecordingWriter(writes.clone())))
                .build()
                .unwrap();
            observer.on_graph_snapshot(GraphSnapshot {
                schema_version: GRAPH_SNAPSHOT_SCHEMA_VERSION,
                profile: CaptureProfile::Structure,
                context: CaptureContext {
                    paradigm: Some("test".into()),
                    epoch: Some(1),
                    batch: Some(1),
                    episode: None,
                },
                nodes: Vec::new(),
                edges: Vec::new(),
                attributes: BTreeMap::new(),
            });
            assert!(writes.lock().unwrap().is_empty());
            observer.on_train_end(&TrainEndContext {
                paradigm: "test",
                units_completed: 1,
                interrupted: false,
            });
            assert_eq!(writes.lock().unwrap().as_slice(), ["capture-e0001-b0001"]);
        }

        #[test]
        fn writer_is_required() {
            assert!(matches!(
                GraphVisualizationObserver::builder().build(),
                Err(VisualizationError::MissingWriter)
            ));
        }
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
