use thiserror::Error;
use std::path::PathBuf;
#[derive(Error, Debug, Clone)]
pub enum TensorError {
    #[error("Invalid shape: expected {:?}, got {:?}", expected, got)]
    InvalidShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("Invalid data length: expected {}, got {}", expected, got)]
    InvalidDataLength { expected: usize, got: usize },
    #[error("Invalid operation '{}': {}", op, reason)]
    InvalidOperation { op: &'static str, reason: String },
    #[error("Invalid axis {} for tensor with shape {:?}", axis, shape)]
    InvalidAxis { axis: usize, shape: Vec<usize> },
    #[error(
        "Invalid dimensions for matrix multiplication: left shape {:?}, right shape {:?}",
        left_shape,
        right_shape
    )]
    MatrixMultiplicationError {
        left_shape: Vec<usize>,
        right_shape: Vec<usize>,
    },
    #[error("InvalidInputCount: expected {:?}, got {:?}", expected, got)]
    InvalidInputCount { expected: i32, got: usize },
    #[error("Empty tensor")]
    EmptyTensor,
    #[error("Expected a scalar tensor, got shape {shape:?}")]
    NotScalar { shape: Vec<usize> },
    #[error("Invalid tensor index {indices:?} for shape {shape:?}")]
    InvalidIndex {
        indices: Vec<usize>,
        shape: Vec<usize>,
    },
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ContextError {
    #[error("The execution context has already been dropped")]
    Dropped,
    #[error("Tensors belong to different execution contexts")]
    Mismatch,
    #[error("Tensor node {0:?} is not present in this execution context")]
    UnknownTensor(super::TensorId),
    #[error("The execution context is already borrowed")]
    BorrowConflict,
    #[error("A training scope cannot start while another scope or graph is active")]
    ActiveGraphConflict,
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum AutogradError {
    #[error("backward() requires a scalar output, got shape {0:?}")]
    OutputNotScalar(Vec<usize>),
    #[error("Gradient shape mismatch: expected {expected:?}, got {got:?}")]
    GradientShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("Operator '{0}' does not support backward")]
    BackwardNotSupported(String),
    #[error("The computation graph for node {0:?} has already been freed")]
    GraphAlreadyFreed(super::TensorId),
    #[error("Computation node {0:?} was not found")]
    NodeNotFound(super::TensorId),
    #[error("A cycle was detected in the computation graph")]
    CycleDetected,
    #[error("Backward result arity mismatch: expected {expected}, got {got}")]
    BackwardArityMismatch { expected: usize, got: usize },
}

#[derive(Error, Debug)]
pub enum MlError {
    #[error("required module '{module}' is unavailable: capability '{capability}' requested by '{operation}'")]
    DependencyUnavailable { module: &'static str, capability: &'static str, operation: &'static str },
    #[error("module '{module}' does not support capability '{capability}' requested by '{operation}'")]
    UnsupportedCapability { module: &'static str, capability: &'static str, operation: &'static str },
    #[error("{primary}; cleanup also failed: {cleanup}")]
    CleanupError { primary: Box<MlError>, cleanup: Box<MlError> },
    #[error(transparent)]
    TensorError(#[from] TensorError),
    #[error(transparent)]
    LossError(#[from] LossError),
    #[error("{0}")]
    StringError(String),
    #[error(transparent)]
    OptimError(#[from] OptimError),
    #[error(transparent)]
    ContextError(#[from] ContextError),
    #[error(transparent)]
    AutogradError(#[from] AutogradError),
    #[error(transparent)]
    DataError(#[from] DataError),
}

impl From<String> for MlError {
    fn from(error: String) -> Self {
        MlError::StringError(error)
    }
}

impl From<&str> for MlError {
    fn from(error: &str) -> Self {
        MlError::StringError(error.to_string())
    }
}

pub type MlResult<T> = Result<T, MlError>;

#[derive(thiserror::Error, Debug)]
pub enum OptimError {
    #[error("Gradient Error: {0}")]
    GradientError(String),
    #[error("invalid optimizer hyperparameter '{name}': {reason}")]
    InvalidHyperparameter { name: &'static str, reason: String },
    #[error("parameter {0:?} is already registered")]
    DuplicateParameter(super::ParameterId),
    #[error("optimizer parameters do not match the model: missing {missing:?}, extra {extra:?}")]
    ParameterSetMismatch { missing: Vec<super::ParameterId>, extra: Vec<super::ParameterId> },
}

#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("data I/O failed for {path}: {message}")]
    Io { path: PathBuf, message: String },
    #[error("data decode failed for {path} at line/row {line}: {message}")]
    Decode {
        path: PathBuf,
        line: usize,
        message: String,
    },
    #[error("data transform failed{location}: {message}")]
    Transform { location: String, message: String },
    #[error("dataset must not be empty")]
    EmptyDataset,
    #[error("batch_size must be greater than zero")]
    InvalidBatchSize,
    #[error("data loader would produce zero batches; disable drop_last or reduce batch_size")]
    NoBatches,
    #[error("cannot collate an empty batch")]
    EmptyBatch,
    #[error("batch collation failed: {message}")]
    Collate { message: String },
    #[error("shape mismatch at sample {sample_index}: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        sample_index: usize,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum LossError {
    #[error("invalid loss shape: expected {expected:?}, got {got:?}")]
    InvalidShape { expected: Vec<usize>, got: Vec<usize> },
    #[error("invalid loss '{op}': {reason}")]
    InvalidOperation { op: &'static str, reason: String },
}
