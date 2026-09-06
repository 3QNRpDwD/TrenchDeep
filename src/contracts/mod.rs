//! Implementation-independent contracts for the single-threaded runtime.
//!
//! Providers own their implementation details. Buffers are validated at the
//! execution boundary; views must not escape the callback that lends them.
use std::fmt::Debug;
use std::rc::Rc;
mod errors;
pub use errors::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub(crate) u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParameterId(pub(crate) u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextId(pub(crate) u64);

#[derive(Debug, Clone, PartialEq)]
pub struct TensorBuffer {
    pub(crate) data: Vec<f32>,
    pub(crate) shape: Vec<usize>,
}
impl TensorBuffer {
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> MlResult<Self> {
        let expected = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| TensorError::InvalidOperation {
                op: "buffer",
                reason: "shape size overflow".into(),
            })?;
        if data.len() != expected {
            return Err(TensorError::InvalidDataLength {
                expected,
                got: data.len(),
            }
            .into());
        }
        Ok(Self {
            data,
            shape: shape.to_vec(),
        })
    }
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn numel(&self) -> usize {
        self.data.len()
    }
    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            data: &self.data,
            shape: &self.shape,
        }
    }
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    pub(crate) data: &'a [f32],
    pub(crate) shape: &'a [usize],
}
impl<'a> TensorView<'a> {
    pub fn new(data: &'a [f32], shape: &'a [usize]) -> MlResult<Self> {
        let size = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
        if size != Some(data.len()) {
            return Err(TensorError::InvalidOperation {
                op: "view",
                reason: "invalid shape or data length".into(),
            }
            .into());
        }
        Ok(Self { data, shape })
    }
    pub fn data(self) -> &'a [f32] {
        self.data
    }
    pub fn shape(self) -> &'a [usize] {
        self.shape
    }
    pub fn len(self) -> usize {
        self.data.len()
    }
    pub fn is_empty(self) -> bool {
        self.data.is_empty()
    }
    pub fn to_owned(self) -> MlResult<TensorBuffer> {
        TensorBuffer::from_vec(self.data.to_vec(), self.shape)
    }
}

/// Insertions and replacements must be atomic on error. Aliases share updates.
/// `remove` is idempotent. A view callback is invoked exactly once for a valid ID.
pub trait TensorStore: Debug {
    fn insert(&mut self, id: TensorId, buffer: TensorBuffer) -> MlResult<()>;
    fn alias(&mut self, id: TensorId, source: TensorId) -> MlResult<()>;
    fn with_view(
        &self,
        id: TensorId,
        visitor: &mut dyn FnMut(TensorView<'_>) -> MlResult<()>,
    ) -> MlResult<()>;
    fn replace(&mut self, id: TensorId, buffer: TensorBuffer) -> MlResult<()>;
    fn remove(&mut self, id: TensorId) -> MlResult<()>;
}

pub fn snapshot(store: &dyn TensorStore, id: TensorId) -> MlResult<TensorBuffer> {
    let mut result = None;
    store.with_view(id, &mut |view| {
        result = Some(view.to_owned()?);
        Ok(())
    })?;
    result.ok_or_else(|| crate::ContextError::UnknownTensor(id).into())
}

pub trait BackwardOp: Debug {
    fn name(&self) -> &'static str;
    fn input_count(&self) -> usize;
    fn backward(
        &self,
        inputs: &[TensorView<'_>],
        saved: &[TensorView<'_>],
        output_grad: TensorView<'_>,
    ) -> MlResult<Vec<Option<TensorBuffer>>>;
}

#[derive(Debug, Clone)]
pub struct GradientRecord {
    pub output: TensorId,
    pub inputs: Vec<TensorId>,
    pub saved: Vec<TensorId>,
    pub backward: Rc<dyn BackwardOp>,
}

/// Owns the graph representation and traversal policy. `order` returns each
/// reachable operation once, dependencies before consumers, rejecting cycles.
/// `record` must not change the graph on error; `remove` is idempotent.
pub trait AutogradEngine: Debug {
    fn record(&mut self, record: GradientRecord) -> MlResult<()>;
    fn get(&self, output: TensorId) -> Option<GradientRecord>;
    fn remove(&mut self, output: TensorId) -> MlResult<Option<GradientRecord>>;
    fn nodes(&self) -> Vec<TensorId>;
    fn order(&self, output: TensorId) -> MlResult<Vec<TensorId>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reduction {
    #[default]
    Mean,
    Sum,
    None,
}
#[derive(Debug, Clone, Copy)]
pub enum LossKind {
    Mse,
    Mae,
    Huber { delta: f32 },
    BinaryCrossEntropy,
    CrossEntropy,
    SoftmaxCrossEntropy,
}
impl LossKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Mse => "mse_loss",
            Self::Mae => "mae_loss",
            Self::Huber { .. } => "huber_loss",
            Self::BinaryCrossEntropy => "binary_cross_entropy",
            Self::CrossEntropy => "cross_entropy",
            Self::SoftmaxCrossEntropy => "softmax_cross_entropy",
        }
    }
}

#[derive(Debug, Clone)]
pub enum Operation {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Square,
    Exp,
    Log,
    Sqrt,
    Pow(f32),
    Sin,
    Cos,
    ApproxSin {
        threshold: f32,
    },
    ApproxCos {
        threshold: f32,
    },
    Tanh,
    Sigmoid,
    Silu,
    Relu,
    Abs,
    Softmax {
        axis: usize,
    },
    Reshape(Vec<usize>),
    Transpose(Vec<usize>),
    Concat {
        axis: usize,
    },
    Sum,
    Matmul,
    Conv2d {
        stride: (usize, usize),
        padding: (usize, usize),
    },
    MaxPool2d {
        kernel: (usize, usize),
        stride: (usize, usize),
    },
    AvgPool2d {
        kernel: (usize, usize),
        stride: (usize, usize),
    },
    NearestUpsample2d {
        scale: (usize, usize),
    },
    GroupNorm {
        groups: usize,
        epsilon: f32,
    },
    Loss {
        kind: LossKind,
        reduction: Reduction,
    },
    TopK {
        k: usize,
        sorted: bool,
    },
    Matmax {
        axis: Option<isize>,
        keepdim: bool,
    },
}
impl Operation {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Neg => "neg",
            Self::Square => "square",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sqrt => "sqrt",
            Self::Pow(_) => "pow",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::ApproxSin { .. } => "approx_sin",
            Self::ApproxCos { .. } => "approx_cos",
            Self::Tanh => "tanh",
            Self::Sigmoid => "sigmoid",
            Self::Silu => "silu",
            Self::Relu => "relu",
            Self::Abs => "abs",
            Self::Softmax { .. } => "softmax",
            Self::Reshape(_) => "reshape",
            Self::Transpose(_) => "transpose",
            Self::Concat { .. } => "concat",
            Self::Sum => "sum",
            Self::Matmul => "matmul",
            Self::Conv2d { .. } => "conv2d",
            Self::MaxPool2d { .. } => "max_pool2d",
            Self::AvgPool2d { .. } => "avg_pool2d",
            Self::NearestUpsample2d { .. } => "nearest_upsample2d",
            Self::GroupNorm { .. } => "group_norm",
            Self::Loss { kind, .. } => kind.name(),
            Self::TopK { .. } => "topk",
            Self::Matmax { .. } => "matmax",
        }
    }
    pub fn input_count(&self) -> Option<usize> {
        Some(match self {
            Self::Concat { .. } => return None,
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Matmul | Self::Loss { .. } => 2,
            Self::Conv2d { .. } | Self::GroupNorm { .. } => 3,
            _ => 1,
        })
    }
}

pub struct OperationOutput {
    pub outputs: Vec<TensorBuffer>,
    pub saved: Vec<TensorBuffer>,
    /// P1 differentiates only single-output operations.
    pub backward: Option<Box<dyn BackwardOp>>,
}
pub trait OperationProvider: Debug {
    fn execute(
        &self,
        operation: &Operation,
        inputs: &[TensorView<'_>],
    ) -> MlResult<OperationOutput>;
}
pub trait CustomOp {
    fn name(&self) -> &'static str;
    fn input_count(&self) -> usize;
    fn forward(&self, inputs: &[TensorView<'_>]) -> MlResult<OpOutput>;
}
pub struct OpOutput {
    pub output: TensorBuffer,
    pub saved: Vec<TensorBuffer>,
    pub backward: Option<Box<dyn BackwardOp>>,
}
