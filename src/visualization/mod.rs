//! Explicit, low-overhead computation graph capture and rendering.

mod capture;
mod dot;
mod error;
mod graphviz;
pub(crate) mod recording;
mod snapshot;
mod statistics;
mod writer;

pub use capture::{VisualizationCapture, VisualizationCaptureBuilder};
pub use dot::DotEncoder;
pub use error::VisualizationError;
pub use snapshot::{
    CaptureContext, CaptureProfile, GRAPH_SNAPSHOT_SCHEMA_VERSION, GraphAttributeValue,
    GraphEdgeKind, GraphEdgeSnapshot, GraphNodeSnapshot, GraphSnapshot, NodeRole, TensorStatistics,
};
pub use writer::{
    FileSnapshotWriter, FileSnapshotWriterBuilder, SnapshotWriter, WriteReport, WriteWarning,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::{Parameter, Variable},
        tensor::{
            AutogradFunction, ComputationGraph, Tensor, TensorBase,
            operators::{Add, Function},
        },
    };

    fn captured_add(profile: CaptureProfile, backward: bool) -> crate::MlResult<GraphSnapshot> {
        let capture = VisualizationCapture::builder(profile)
            .context(CaptureContext {
                paradigm: Some("test".into()),
                epoch: Some(1),
                batch: Some(1),
                episode: None,
            })
            .begin()?;
        let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2])?);
        x.retain_grad();
        let y = Variable::new(Tensor::from_vec(vec![3.0, 4.0], &[2])?);
        let output = Add::new()?.apply(&[&x, &y])?;
        if backward {
            output.backward()?;
        }
        Ok(capture.finish()?)
    }

    #[test]
    fn inactive_recording_skips_sidecar_and_statistics() -> crate::MlResult<()> {
        ComputationGraph::reset_graph();
        statistics::reset_collection_count();
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1])?);
        let y = Variable::new(Tensor::from_vec(vec![2.0], &[1])?);
        let _ = Add::new()?.apply(&[&x, &y])?;
        assert!(!recording::is_active());
        assert_eq!(recording::metadata_len(), 0);
        assert_eq!(statistics::collection_count(), 0);
        Ok(())
    }

    #[test]
    fn profiles_control_statistics_and_backward_stats_are_retained() -> crate::MlResult<()> {
        let structure = captured_add(CaptureProfile::Structure, true)?;
        assert!(
            structure
                .nodes
                .iter()
                .all(|node| node.value_stats.is_none())
        );
        assert!(
            structure
                .nodes
                .iter()
                .all(|node| node.gradient_stats.is_none())
        );

        let analysis = captured_add(CaptureProfile::Analysis, true)?;
        assert!(analysis.nodes.iter().any(|node| node.value_stats.is_some()));
        assert!(analysis.nodes.iter().any(|node| {
            node.is_leaf
                && node
                    .gradient_stats
                    .as_ref()
                    .is_some_and(|stats| stats.l1_norm == Some(2.0))
        }));
        Ok(())
    }

    #[test]
    fn capture_rejects_nesting_and_recovers_on_drop_error_and_panic() {
        let capture = VisualizationCapture::builder(CaptureProfile::Structure)
            .begin()
            .unwrap();
        assert!(matches!(
            VisualizationCapture::builder(CaptureProfile::Structure).begin(),
            Err(VisualizationError::CaptureAlreadyActive)
        ));
        drop(capture);
        assert!(!recording::is_active());

        fn early_error() -> Result<(), VisualizationError> {
            let _capture = VisualizationCapture::builder(CaptureProfile::Structure).begin()?;
            Err(VisualizationError::CaptureNotActive)
        }
        assert!(early_error().is_err());
        assert!(!recording::is_active());

        let _ = std::panic::catch_unwind(|| {
            let _capture = VisualizationCapture::builder(CaptureProfile::Structure)
                .begin()
                .unwrap();
            panic!("test unwind");
        });
        assert!(!recording::is_active());
    }

    #[test]
    fn snapshot_is_owned_and_schema_matches_dot_ids() -> crate::MlResult<()> {
        let snapshot = captured_add(CaptureProfile::Analysis, false)?;
        let node_count = snapshot.nodes.len();
        ComputationGraph::reset_graph();
        assert_eq!(snapshot.nodes.len(), node_count);
        assert_eq!(snapshot.schema_version, 1);

        let json = serde_json::to_value(&snapshot).map_err(VisualizationError::from)?;
        assert_eq!(json["schema_version"], 1);
        assert!(json["edges"].as_array().is_some_and(|edges| {
            edges.iter().all(|edge| {
                edge.get("from").is_some() && edge.get("to").is_some() && edge.get("kind").is_some()
            })
        }));
        let dot = DotEncoder::encode(&snapshot);
        for node in &snapshot.nodes {
            assert!(dot.contains(&format!("id=\"node-{}\"", node.id)));
        }
        assert_eq!(dot.matches(" [id=\"node-").count(), snapshot.nodes.len());
        Ok(())
    }

    #[test]
    fn file_writer_keeps_dot_and_json_when_graphviz_is_missing() -> crate::MlResult<()> {
        let snapshot = captured_add(CaptureProfile::Structure, false)?;
        let directory = std::env::temp_dir().join(format!(
            "trench-deep-visualization-writer-{}-{:?}",
            std::process::id(),
            std::thread::current().id(),
        ));
        let mut writer = FileSnapshotWriter::builder(&directory)
            .render_svg(true)
            .graphviz_program(directory.join("definitely-missing-dot"))
            .build()?;
        let report = writer.write(&snapshot, "capture-e0001-b0001")?;
        assert_eq!(report.artifacts.len(), 2);
        assert_eq!(report.warnings.len(), 1);
        assert!(directory.join("capture-e0001-b0001.dot").exists());
        assert!(directory.join("capture-e0001-b0001.json").exists());
        let _ = std::fs::remove_dir_all(directory);
        Ok(())
    }

    #[test]
    #[ignore = "manual capture-disabled micro-benchmark"]
    fn capture_disabled_micro_benchmark() -> crate::MlResult<()> {
        use std::time::Instant;

        ComputationGraph::reset_graph();
        statistics::reset_collection_count();
        let started = Instant::now();
        for _ in 0..10_000 {
            let x = Variable::new(Tensor::from_vec(vec![1.0], &[1])?);
            let y = Variable::new(Tensor::from_vec(vec![2.0], &[1])?);
            let _ = Add::new()?.apply(&[&x, &y])?;
            ComputationGraph::reset_graph();
        }
        assert_eq!(recording::metadata_len(), 0);
        assert_eq!(statistics::collection_count(), 0);
        eprintln!("capture-disabled 10k batches: {:?}", started.elapsed());
        Ok(())
    }
}
