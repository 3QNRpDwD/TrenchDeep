//! Explicit, single-threaded execution context.
//!
//! This module is the migration target for the legacy thread-local tensor and
//! graph stores.  It deliberately exposes only fallible, borrow-safe access.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::loss::{LossError, Reduction};
use crate::{AutogradError, ContextError, MlResult, TensorError};

use super::{GlobalTensor, NodeId, TensorBase};

static NEXT_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextId(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequiresGrad {
    No,
    Yes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphStats {
    pub tensors: usize,
    pub graph_nodes: usize,
    pub dynamic_backward_nodes: usize,
    pub saved_tensor_references: usize,
    pub no_grad_depth: usize,
}

#[derive(Clone, Debug)]
pub struct TopKResult {
    pub values: ContextTensor,
    pub indices: ContextTensor,
}

#[derive(Clone, Debug)]
pub struct MaxResult {
    pub values: ContextTensor,
    pub indices: ContextTensor,
}

#[derive(Debug)]
struct ContextState {
    tensors: HashMap<NodeId, TensorStorageEntry>,
    next_node: u64,
    graph: HashMap<NodeId, GraphNode>,
    tracked: HashSet<NodeId>,
    leaves: HashSet<NodeId>,
    retained_gradients: HashSet<NodeId>,
    gradients: HashMap<NodeId, GlobalTensor<f32>>,
    consumed: HashSet<NodeId>,
    no_grad_depth: usize,
}

#[derive(Debug)]
struct TensorStorageEntry {
    buffer: Rc<RefCell<GlobalTensor<f32>>>,
    external_handle: Weak<ContextTensorHandle>,
    graph_pins: usize,
    internal_saved: bool,
}

#[derive(Debug)]
struct ContextRuntime {
    state: RefCell<ContextState>,
    gc_pending: Cell<bool>,
    training_active: Cell<bool>,
}

impl TensorStorageEntry {
    fn snapshot(&self) -> MlResult<GlobalTensor<f32>> {
        self.buffer
            .try_borrow()
            .map(|value| value.clone())
            .map_err(|_| ContextError::BorrowConflict.into())
    }
}

#[derive(Debug)]
struct GraphNode {
    inputs: Vec<NodeId>,
    saved: Vec<NodeId>,
    backward: Box<dyn BackwardOp>,
}

#[derive(Debug, Clone)]
enum BuiltinBackward {
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
    Reshape,
    Transpose(Vec<usize>),
    Concat {
        axis: usize,
        sizes: Vec<usize>,
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
        kind: ContextLossKind,
        reduction: Reduction,
    },
}

#[derive(Debug, Clone, Copy)]
enum ContextLossKind {
    Mse,
    Mae,
    Huber { delta: f32 },
    BinaryCrossEntropy,
    CrossEntropy,
    SoftmaxCrossEntropy,
}

impl ContextLossKind {
    fn name(self) -> &'static str {
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

#[derive(Debug, Clone, Copy)]
pub struct BackwardOptions<'a> {
    pub gradient: Option<&'a ContextTensor>,
    pub retain_graph: bool,
}

impl Default for BackwardOptions<'_> {
    fn default() -> Self {
        Self {
            gradient: None,
            retain_graph: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExecutionContext {
    id: ContextId,
    runtime: Rc<ContextRuntime>,
    _not_sync: Rc<Cell<()>>,
}

#[derive(Debug)]
struct ContextTensorHandle {
    context_id: ContextId,
    node_id: NodeId,
    runtime: Weak<ContextRuntime>,
}

/// Safe tensor handle used by the explicit context API during the P1 migration.
#[derive(Clone, Debug)]
pub struct ContextTensor(Rc<ContextTensorHandle>);

#[derive(Debug, Clone)]
pub struct ContextVariable {
    tensor: ContextTensor,
}

impl ContextState {
    fn pin(&mut self, id: NodeId) -> MlResult<()> {
        let entry = self
            .tensors
            .get_mut(&id)
            .ok_or(ContextError::UnknownTensor(id))?;
        entry.graph_pins =
            entry
                .graph_pins
                .checked_add(1)
                .ok_or_else(|| TensorError::InvalidOperation {
                    op: "graph_pin",
                    reason: "graph reference count overflow".into(),
                })?;
        Ok(())
    }

    fn unpin(&mut self, id: NodeId) {
        if let Some(entry) = self.tensors.get_mut(&id) {
            entry.graph_pins = entry.graph_pins.saturating_sub(1);
        }
    }

    fn collect_garbage(&mut self) {
        let garbage: Vec<_> = self
            .tensors
            .iter()
            .filter_map(|(&id, entry)| {
                let externally_dead =
                    entry.internal_saved || entry.external_handle.upgrade().is_none();
                (entry.graph_pins == 0 && externally_dead).then_some(id)
            })
            .collect();
        for id in garbage {
            self.tensors.remove(&id);
            self.tracked.remove(&id);
            self.leaves.remove(&id);
            self.retained_gradients.remove(&id);
            self.gradients.remove(&id);
            self.consumed.remove(&id);
        }
    }

    fn remove_graph_node(&mut self, output: NodeId) {
        if let Some(node) = self.graph.remove(&output) {
            self.unpin(output);
            for input in node.inputs {
                self.unpin(input);
            }
            for saved in node.saved {
                self.unpin(saved);
            }
        }
    }
}

impl Drop for ContextTensorHandle {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.upgrade() {
            runtime.gc_pending.set(true);
            if let Ok(mut state) = runtime.state.try_borrow_mut() {
                state.collect_garbage();
                runtime.gc_pending.set(false);
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct TensorView<'a> {
    data: &'a [f32],
    shape: &'a [usize],
}

pub trait BackwardOp: std::fmt::Debug {
    fn name(&self) -> &'static str;
    fn input_count(&self) -> usize;
    fn backward(
        &self,
        inputs: &[TensorView<'_>],
        saved: &[TensorView<'_>],
        output_grad: TensorView<'_>,
    ) -> MlResult<Vec<Option<GlobalTensor<f32>>>>;
}

#[derive(Debug)]
struct AddBackward;

impl BackwardOp for AddBackward {
    fn name(&self) -> &'static str {
        "add"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn backward(
        &self,
        inputs: &[TensorView<'_>],
        _saved: &[TensorView<'_>],
        grad: TensorView<'_>,
    ) -> MlResult<Vec<Option<GlobalTensor<f32>>>> {
        if inputs.len() != self.input_count() {
            return Err(AutogradError::BackwardArityMismatch {
                expected: self.input_count(),
                got: inputs.len(),
            }
            .into());
        }
        let grad = GlobalTensor::from_vec(grad.data.to_vec(), grad.shape)?;
        Ok(vec![
            Some(reduce_to_shape(&grad, inputs[0].shape)?),
            Some(reduce_to_shape(&grad, inputs[1].shape)?),
        ])
    }
}

#[derive(Debug)]
struct MulBackward;

impl BackwardOp for MulBackward {
    fn name(&self) -> &'static str {
        "mul"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn backward(
        &self,
        inputs: &[TensorView<'_>],
        _saved: &[TensorView<'_>],
        grad: TensorView<'_>,
    ) -> MlResult<Vec<Option<GlobalTensor<f32>>>> {
        if inputs.len() != self.input_count() {
            return Err(AutogradError::BackwardArityMismatch {
                expected: self.input_count(),
                got: inputs.len(),
            }
            .into());
        }
        let grad = GlobalTensor::from_vec(grad.data.to_vec(), grad.shape)?;
        let lhs = GlobalTensor::from_vec(inputs[0].data.to_vec(), inputs[0].shape)?;
        let rhs = GlobalTensor::from_vec(inputs[1].data.to_vec(), inputs[1].shape)?;
        let lhs_broadcast =
            GlobalTensor::from_vec(broadcast_data(&lhs, grad.shape.as_slice())?, &grad.shape)?;
        let rhs_broadcast =
            GlobalTensor::from_vec(broadcast_data(&rhs, grad.shape.as_slice())?, &grad.shape)?;
        Ok(vec![
            Some(reduce_to_shape(
                &tensor_zip(&grad, &rhs_broadcast, |g, x| g * x)?,
                &lhs.shape,
            )?),
            Some(reduce_to_shape(
                &tensor_zip(&grad, &lhs_broadcast, |g, x| g * x)?,
                &rhs.shape,
            )?),
        ])
    }
}

#[derive(Debug)]
struct ElementwiseBackward(BuiltinBackward);

impl BackwardOp for ElementwiseBackward {
    fn name(&self) -> &'static str {
        match &self.0 {
            BuiltinBackward::Sub => "sub",
            BuiltinBackward::Div => "div",
            BuiltinBackward::Neg => "neg",
            BuiltinBackward::Square => "square",
            BuiltinBackward::Exp => "exp",
            BuiltinBackward::Log => "log",
            BuiltinBackward::Sqrt => "sqrt",
            BuiltinBackward::Pow(_) => "pow",
            BuiltinBackward::Sin => "sin",
            BuiltinBackward::Cos => "cos",
            BuiltinBackward::ApproxSin { .. } => "approx_sin",
            BuiltinBackward::ApproxCos { .. } => "approx_cos",
            BuiltinBackward::Tanh => "tanh",
            BuiltinBackward::Sigmoid => "sigmoid",
            BuiltinBackward::Silu => "silu",
            BuiltinBackward::Relu => "relu",
            BuiltinBackward::Abs => "abs",
            BuiltinBackward::Softmax { .. } => "softmax",
            BuiltinBackward::Loss { kind, .. } => kind.name(),
            BuiltinBackward::Sum => "sum",
            BuiltinBackward::Reshape => "reshape",
            _ => "unsupported",
        }
    }

    fn input_count(&self) -> usize {
        match &self.0 {
            BuiltinBackward::Sub | BuiltinBackward::Div | BuiltinBackward::Loss { .. } => 2,
            _ => 1,
        }
    }

    fn backward(
        &self,
        inputs: &[TensorView<'_>],
        saved: &[TensorView<'_>],
        grad: TensorView<'_>,
    ) -> MlResult<Vec<Option<GlobalTensor<f32>>>> {
        if inputs.len() != self.input_count() {
            return Err(AutogradError::BackwardArityMismatch {
                expected: self.input_count(),
                got: inputs.len(),
            }
            .into());
        }
        if let BuiltinBackward::Loss { kind, reduction } = &self.0 {
            return Ok(vec![
                Some(loss_backward(
                    kind,
                    *reduction,
                    inputs[0],
                    inputs[1],
                    grad,
                    saved.first().copied(),
                )?),
                None,
            ]);
        }
        let values = inputs
            .iter()
            .map(|view| GlobalTensor::from_vec(view.data.to_vec(), view.shape))
            .collect::<MlResult<Vec<_>>>()?;
        let grad = GlobalTensor::from_vec(grad.data.to_vec(), grad.shape)?;
        let saved_output = || -> MlResult<GlobalTensor<f32>> {
            let view = saved
                .first()
                .ok_or_else(|| AutogradError::BackwardArityMismatch {
                    expected: 1,
                    got: 0,
                })?;
            GlobalTensor::from_vec(view.data.to_vec(), view.shape)
        };
        let results = match &self.0 {
            BuiltinBackward::Sub => vec![
                reduce_to_shape(&grad, &values[0].shape)?,
                reduce_to_shape(&tensor_map(&grad, |g| -g)?, &values[1].shape)?,
            ],
            BuiltinBackward::Div => {
                let rhs =
                    GlobalTensor::from_vec(broadcast_data(&values[1], &grad.shape)?, &grad.shape)?;
                let lhs =
                    GlobalTensor::from_vec(broadcast_data(&values[0], &grad.shape)?, &grad.shape)?;
                let left = tensor_zip(&grad, &rhs, |g, r| g / r)?;
                let right = GlobalTensor::from_vec(
                    grad.data
                        .iter()
                        .zip(&lhs.data)
                        .zip(&rhs.data)
                        .map(|((g, l), r)| -g * l / (r * r))
                        .collect(),
                    &grad.shape,
                )?;
                vec![
                    reduce_to_shape(&left, &values[0].shape)?,
                    reduce_to_shape(&right, &values[1].shape)?,
                ]
            }
            BuiltinBackward::Neg => vec![tensor_map(&grad, |g| -g)?],
            BuiltinBackward::Square => vec![tensor_zip(&grad, &values[0], |g, x| 2.0 * g * x)?],
            BuiltinBackward::Exp => vec![tensor_zip(&grad, &saved_output()?, |g, y| g * y)?],
            BuiltinBackward::Log => vec![tensor_zip(&grad, &values[0], |g, x| g / x)?],
            BuiltinBackward::Sqrt => vec![tensor_zip(&grad, &saved_output()?, |g, y| g * 0.5 / y)?],
            BuiltinBackward::Pow(exponent) => vec![tensor_zip(&grad, &values[0], |g, x| {
                g * *exponent * x.powf(*exponent - 1.0)
            })?],
            BuiltinBackward::Sin => vec![tensor_zip(&grad, &values[0], |g, x| g * x.cos())?],
            BuiltinBackward::Cos => vec![tensor_zip(&grad, &values[0], |g, x| -g * x.sin())?],
            BuiltinBackward::ApproxSin { threshold } => {
                vec![tensor_zip(&grad, &values[0], |g, x| {
                    let _ = threshold;
                    g * approx_sin_derivative(x)
                })?]
            }
            BuiltinBackward::ApproxCos { threshold } => {
                vec![tensor_zip(&grad, &values[0], |g, x| {
                    let _ = threshold;
                    g * approx_cos_derivative(x)
                })?]
            }
            BuiltinBackward::Tanh => vec![tensor_zip(&grad, &saved_output()?, |g, y| {
                g * (1.0 - y * y)
            })?],
            BuiltinBackward::Sigmoid => vec![tensor_zip(&grad, &saved_output()?, |g, y| {
                g * y * (1.0 - y)
            })?],
            BuiltinBackward::Silu => vec![tensor_zip(&grad, &values[0], |g, x| {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                g * sigmoid * (1.0 + x * (1.0 - sigmoid))
            })?],
            BuiltinBackward::Relu => vec![tensor_zip(&grad, &values[0], |g, x| {
                if x > 0.0 { g } else { 0.0 }
            })?],
            BuiltinBackward::Abs => vec![tensor_zip(&grad, &values[0], |g, x| {
                if x > 0.0 {
                    g
                } else if x < 0.0 {
                    -g
                } else {
                    0.0
                }
            })?],
            BuiltinBackward::Softmax { axis } => {
                let output = saved_output()?;
                let axis = *axis;
                let outer: usize = output.shape[..axis].iter().product();
                let width = output.shape[axis];
                let inner: usize = output.shape[axis + 1..].iter().product();
                let mut data = vec![0.0; output.data.len()];
                for outer_index in 0..outer {
                    for inner_index in 0..inner {
                        let dot: f32 = (0..width)
                            .map(|i| {
                                let index = (outer_index * width + i) * inner + inner_index;
                                grad.data[index] * output.data[index]
                            })
                            .sum();
                        for i in 0..width {
                            let index = (outer_index * width + i) * inner + inner_index;
                            data[index] = output.data[index] * (grad.data[index] - dot);
                        }
                    }
                }
                vec![GlobalTensor::from_vec(data, &output.shape)?]
            }
            BuiltinBackward::Sum => {
                let scalar = grad.data.first().copied().ok_or(TensorError::EmptyTensor)?;
                vec![GlobalTensor::from_vec(
                    vec![scalar; values[0].data.len()],
                    &values[0].shape,
                )?]
            }
            BuiltinBackward::Reshape => vec![GlobalTensor::from_vec(grad.data, &values[0].shape)?],
            BuiltinBackward::Loss { .. } => unreachable!("loss backward handled before snapshots"),
            _ => return Err(AutogradError::BackwardNotSupported(self.name().into()).into()),
        };
        Ok(results.into_iter().map(Some).collect())
    }
}

#[derive(Debug)]
struct StructuralBackward(BuiltinBackward);

impl BackwardOp for StructuralBackward {
    fn name(&self) -> &'static str {
        match &self.0 {
            BuiltinBackward::Transpose(_) => "transpose",
            BuiltinBackward::Concat { .. } => "concat",
            BuiltinBackward::Matmul => "matmul",
            BuiltinBackward::Conv2d { .. } => "conv2d",
            BuiltinBackward::MaxPool2d { .. } => "max_pool2d",
            BuiltinBackward::AvgPool2d { .. } => "avg_pool2d",
            BuiltinBackward::NearestUpsample2d { .. } => "nearest_upsample2d",
            BuiltinBackward::GroupNorm { .. } => "group_norm",
            _ => "unsupported",
        }
    }

    fn input_count(&self) -> usize {
        match &self.0 {
            BuiltinBackward::Concat { sizes, .. } => sizes.len(),
            BuiltinBackward::Matmul => 2,
            BuiltinBackward::Conv2d { .. } => 3,
            BuiltinBackward::MaxPool2d { .. } | BuiltinBackward::AvgPool2d { .. } => 1,
            BuiltinBackward::NearestUpsample2d { .. } => 1,
            BuiltinBackward::GroupNorm { .. } => 3,
            _ => 1,
        }
    }

    fn backward(
        &self,
        inputs: &[TensorView<'_>],
        saved: &[TensorView<'_>],
        grad: TensorView<'_>,
    ) -> MlResult<Vec<Option<GlobalTensor<f32>>>> {
        if inputs.len() != self.input_count() {
            return Err(AutogradError::BackwardArityMismatch {
                expected: self.input_count(),
                got: inputs.len(),
            }
            .into());
        }
        let values = inputs
            .iter()
            .map(|view| GlobalTensor::from_vec(view.data.to_vec(), view.shape))
            .collect::<MlResult<Vec<_>>>()?;
        let grad = GlobalTensor::from_vec(grad.data.to_vec(), grad.shape)?;
        let results = match &self.0 {
            BuiltinBackward::Transpose(axes) => {
                let mut inverse = vec![0; axes.len()];
                for (output_axis, &input_axis) in axes.iter().enumerate() {
                    inverse[input_axis] = output_axis;
                }
                vec![GlobalTensor::from_vec(
                    permute_data(&grad.data, &grad.shape, &inverse),
                    &values[0].shape,
                )?]
            }
            BuiltinBackward::Concat { axis, sizes } => {
                let outer: usize = grad.shape[..*axis].iter().product();
                let inner: usize = grad.shape[*axis + 1..].iter().product();
                let axis_width = grad.shape[*axis] * inner;
                let mut running = 0;
                let offsets: Vec<_> = sizes
                    .iter()
                    .map(|size| {
                        let offset = running;
                        running += size * inner;
                        offset
                    })
                    .collect();
                let mut split = Vec::with_capacity(values.len());
                for (index, value) in values.iter().enumerate() {
                    let chunk = sizes[index] * inner;
                    let mut data = Vec::with_capacity(value.data.len());
                    for outer_index in 0..outer {
                        let start = outer_index * axis_width + offsets[index];
                        data.extend_from_slice(&grad.data[start..start + chunk]);
                    }
                    split.push(GlobalTensor::from_vec(data, &value.shape)?);
                }
                split
            }
            BuiltinBackward::Matmul => {
                let (lhs, rhs) = (&values[0], &values[1]);
                let spec = MatmulSpec::new(&lhs.shape, &rhs.shape)?;
                let batch_count: usize = spec.batch_shape.iter().product();
                let mut dl = vec![0.0; lhs.data.len()];
                let mut dr = vec![0.0; rhs.data.len()];
                for batch in 0..batch_count {
                    let lb = broadcast_offset(batch, &spec.batch_shape, &spec.left_batch);
                    let rb = broadcast_offset(batch, &spec.batch_shape, &spec.right_batch);
                    for i in 0..spec.m {
                        for p in 0..spec.k {
                            for j in 0..spec.n {
                                let upstream = grad.data[(batch * spec.m + i) * spec.n + j];
                                dl[(lb * spec.m + i) * spec.k + p] +=
                                    upstream * rhs.data[(rb * spec.k + p) * spec.n + j];
                                dr[(rb * spec.k + p) * spec.n + j] +=
                                    lhs.data[(lb * spec.m + i) * spec.k + p] * upstream;
                            }
                        }
                    }
                }
                vec![
                    GlobalTensor::from_vec(dl, &lhs.shape)?,
                    GlobalTensor::from_vec(dr, &rhs.shape)?,
                ]
            }
            BuiltinBackward::Conv2d { stride, padding } => {
                let (dx, dw, db) =
                    conv2d_backward_data(&values[0], &values[1], &grad, *stride, *padding)?;
                vec![dx, dw, db]
            }
            BuiltinBackward::MaxPool2d { kernel, stride } => {
                let mask = saved.first().ok_or(AutogradError::BackwardArityMismatch {
                    expected: 1,
                    got: 0,
                })?;
                vec![max_pool2d_backward_data(
                    &values[0], mask, &grad, *kernel, *stride,
                )?]
            }
            BuiltinBackward::AvgPool2d { kernel, stride } => {
                vec![avg_pool2d_backward_data(
                    &values[0], &grad, *kernel, *stride,
                )?]
            }
            BuiltinBackward::NearestUpsample2d { scale } => {
                vec![nearest_upsample2d_backward_data(&values[0], &grad, *scale)?]
            }
            BuiltinBackward::GroupNorm { groups, epsilon } => {
                let (dx, dgamma, dbeta) = group_norm_backward_data(
                    &values[0], &values[1], saved, &grad, *groups, *epsilon,
                )?;
                vec![dx, dgamma, dbeta]
            }
            _ => return Err(AutogradError::BackwardNotSupported(self.name().into()).into()),
        };
        Ok(results.into_iter().map(Some).collect())
    }
}

impl<'a> TensorView<'a> {
    pub fn data(&self) -> &'a [f32] {
        self.data
    }
    pub fn shape(&self) -> &'a [usize] {
        self.shape
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

mod runtime;

fn into_node_backward(backward: BuiltinBackward) -> Box<dyn BackwardOp> {
    match backward {
        BuiltinBackward::Add => Box::new(AddBackward),
        BuiltinBackward::Mul => Box::new(MulBackward),
        other @ (BuiltinBackward::Sub
        | BuiltinBackward::Div
        | BuiltinBackward::Neg
        | BuiltinBackward::Square
        | BuiltinBackward::Exp
        | BuiltinBackward::Log
        | BuiltinBackward::Sqrt
        | BuiltinBackward::Pow(_)
        | BuiltinBackward::Sin
        | BuiltinBackward::Cos
        | BuiltinBackward::ApproxSin { .. }
        | BuiltinBackward::ApproxCos { .. }
        | BuiltinBackward::Tanh
        | BuiltinBackward::Sigmoid
        | BuiltinBackward::Silu
        | BuiltinBackward::Relu
        | BuiltinBackward::Abs
        | BuiltinBackward::Softmax { .. }
        | BuiltinBackward::Loss { .. }
        | BuiltinBackward::Sum
        | BuiltinBackward::Reshape) => Box::new(ElementwiseBackward(other)),
        other @ (BuiltinBackward::Transpose(_)
        | BuiltinBackward::Concat { .. }
        | BuiltinBackward::Matmul
        | BuiltinBackward::Conv2d { .. }
        | BuiltinBackward::MaxPool2d { .. }
        | BuiltinBackward::AvgPool2d { .. }
        | BuiltinBackward::NearestUpsample2d { .. }
        | BuiltinBackward::GroupNorm { .. }) => Box::new(StructuralBackward(other)),
    }
}

mod kernels;
use kernels::*;

mod handles;
mod receiver;
mod custom;
mod training;
pub use custom::{CustomOp, OpOutput};

mod functional;
use functional::*;

#[cfg(test)]
mod tests;
