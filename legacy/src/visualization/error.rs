use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum VisualizationError {
    #[error("a visualization capture is already active on this thread")]
    CaptureAlreadyActive,
    #[error("visualization capture is not active")]
    CaptureNotActive,
    #[error("GraphVisualizationObserver requires a SnapshotWriter")]
    MissingWriter,
    #[error("visualization capture coordinates are 1-based")]
    InvalidCaptureCoordinate,
    #[error("invalid visualization artifact stem: {0}")]
    InvalidArtifactStem(String),
    #[error("visualization I/O failed for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("visualization JSON serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("computation graph lock is poisoned")]
    GraphLockPoisoned,
}

impl VisualizationError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
