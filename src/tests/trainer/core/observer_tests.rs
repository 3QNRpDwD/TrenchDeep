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
