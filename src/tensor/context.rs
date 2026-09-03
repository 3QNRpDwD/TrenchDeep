//! Explicit, single-threaded execution context.
//!
//! This module is the migration target for the legacy thread-local tensor and
//! graph stores.  It deliberately exposes only fallible, borrow-safe access.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicU64, Ordering};

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

#[derive(Debug)]
struct ContextState {
    tensors: HashMap<NodeId, GlobalTensor<f32>>,
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
struct GraphNode {
    inputs: Vec<NodeId>,
    saved: Vec<NodeId>,
    owned_saved: Vec<NodeId>,
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
    Tanh,
    Sigmoid,
    Silu,
    Relu,
    Abs,
    Softmax { axis: usize },
    Reshape,
    Transpose(Vec<usize>),
    Concat { axis: usize, sizes: Vec<usize> },
    Sum,
    Matmul,
    Conv2d { stride: (usize, usize), padding: (usize, usize) },
    MaxPool2d { kernel: (usize, usize), stride: (usize, usize) },
    AvgPool2d { kernel: (usize, usize), stride: (usize, usize) },
    NearestUpsample2d { scale: (usize, usize) },
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
    state: Rc<RefCell<ContextState>>,
    _not_sync: Rc<Cell<()>>,
}

#[derive(Debug)]
struct ContextTensorHandle {
    context_id: ContextId,
    node_id: NodeId,
    context: Weak<RefCell<ContextState>>,
}

/// Safe tensor handle used by the explicit context API during the P1 migration.
#[derive(Clone, Debug)]
pub struct ContextTensor(Rc<ContextTensorHandle>);

#[derive(Debug, Clone)]
pub struct ContextVariable {
    tensor: ContextTensor,
    requires_grad: bool,
}

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
    fn name(&self) -> &'static str { "add" }
    fn input_count(&self) -> usize { 2 }
    fn backward(&self, inputs: &[TensorView<'_>], _saved: &[TensorView<'_>], grad: TensorView<'_>) -> MlResult<Vec<Option<GlobalTensor<f32>>>> {
        if inputs.len() != self.input_count() {
            return Err(AutogradError::BackwardArityMismatch { expected: self.input_count(), got: inputs.len() }.into());
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
    fn name(&self) -> &'static str { "mul" }
    fn input_count(&self) -> usize { 2 }
    fn backward(&self, inputs: &[TensorView<'_>], _saved: &[TensorView<'_>], grad: TensorView<'_>) -> MlResult<Vec<Option<GlobalTensor<f32>>>> {
        if inputs.len() != self.input_count() {
            return Err(AutogradError::BackwardArityMismatch { expected: self.input_count(), got: inputs.len() }.into());
        }
        let grad = GlobalTensor::from_vec(grad.data.to_vec(), grad.shape)?;
        let lhs = GlobalTensor::from_vec(inputs[0].data.to_vec(), inputs[0].shape)?;
        let rhs = GlobalTensor::from_vec(inputs[1].data.to_vec(), inputs[1].shape)?;
        let lhs_broadcast = GlobalTensor::from_vec(broadcast_data(&lhs, grad.shape.as_slice())?, &grad.shape)?;
        let rhs_broadcast = GlobalTensor::from_vec(broadcast_data(&rhs, grad.shape.as_slice())?, &grad.shape)?;
        Ok(vec![
            Some(reduce_to_shape(&tensor_zip(&grad, &rhs_broadcast, |g, x| g * x)?, &lhs.shape)?),
            Some(reduce_to_shape(&tensor_zip(&grad, &lhs_broadcast, |g, x| g * x)?, &rhs.shape)?),
        ])
    }
}

#[derive(Debug)]
struct ElementwiseBackward(BuiltinBackward);

impl BackwardOp for ElementwiseBackward {
    fn name(&self) -> &'static str {
        match &self.0 {
            BuiltinBackward::Sub => "sub", BuiltinBackward::Div => "div",
            BuiltinBackward::Neg => "neg", BuiltinBackward::Square => "square",
            BuiltinBackward::Exp => "exp", BuiltinBackward::Log => "log",
            BuiltinBackward::Sqrt => "sqrt", BuiltinBackward::Pow(_) => "pow",
            BuiltinBackward::Sin => "sin", BuiltinBackward::Cos => "cos",
            BuiltinBackward::Tanh => "tanh", BuiltinBackward::Sigmoid => "sigmoid",
            BuiltinBackward::Silu => "silu", BuiltinBackward::Relu => "relu",
            BuiltinBackward::Abs => "abs", BuiltinBackward::Softmax { .. } => "softmax",
            BuiltinBackward::Sum => "sum", BuiltinBackward::Reshape => "reshape",
            _ => "unsupported",
        }
    }

    fn input_count(&self) -> usize {
        match &self.0 { BuiltinBackward::Sub | BuiltinBackward::Div => 2, _ => 1 }
    }

    fn backward(&self, inputs: &[TensorView<'_>], saved: &[TensorView<'_>], grad: TensorView<'_>) -> MlResult<Vec<Option<GlobalTensor<f32>>>> {
        if inputs.len() != self.input_count() {
            return Err(AutogradError::BackwardArityMismatch { expected: self.input_count(), got: inputs.len() }.into());
        }
        let values = inputs.iter().map(|view| GlobalTensor::from_vec(view.data.to_vec(), view.shape)).collect::<MlResult<Vec<_>>>()?;
        let grad = GlobalTensor::from_vec(grad.data.to_vec(), grad.shape)?;
        let saved_output = || -> MlResult<GlobalTensor<f32>> {
            let view = saved.first().ok_or_else(|| AutogradError::BackwardArityMismatch { expected: 1, got: 0 })?;
            GlobalTensor::from_vec(view.data.to_vec(), view.shape)
        };
        let results = match &self.0 {
            BuiltinBackward::Sub => vec![
                reduce_to_shape(&grad, &values[0].shape)?,
                reduce_to_shape(&tensor_map(&grad, |g| -g)?, &values[1].shape)?,
            ],
            BuiltinBackward::Div => {
                let rhs = GlobalTensor::from_vec(broadcast_data(&values[1], &grad.shape)?, &grad.shape)?;
                let lhs = GlobalTensor::from_vec(broadcast_data(&values[0], &grad.shape)?, &grad.shape)?;
                let left = tensor_zip(&grad, &rhs, |g, r| g / r)?;
                let right = GlobalTensor::from_vec(grad.data.iter().zip(&lhs.data).zip(&rhs.data)
                    .map(|((g, l), r)| -g * l / (r * r)).collect(), &grad.shape)?;
                vec![reduce_to_shape(&left, &values[0].shape)?, reduce_to_shape(&right, &values[1].shape)?]
            }
            BuiltinBackward::Neg => vec![tensor_map(&grad, |g| -g)?],
            BuiltinBackward::Square => vec![tensor_zip(&grad, &values[0], |g, x| 2.0 * g * x)?],
            BuiltinBackward::Exp => vec![tensor_zip(&grad, &saved_output()?, |g, y| g * y)?],
            BuiltinBackward::Log => vec![tensor_zip(&grad, &values[0], |g, x| g / x)?],
            BuiltinBackward::Sqrt => vec![tensor_zip(&grad, &saved_output()?, |g, y| g * 0.5 / y)?],
            BuiltinBackward::Pow(exponent) => vec![tensor_zip(&grad, &values[0], |g, x| g * *exponent * x.powf(*exponent - 1.0))?],
            BuiltinBackward::Sin => vec![tensor_zip(&grad, &values[0], |g, x| g * x.cos())?],
            BuiltinBackward::Cos => vec![tensor_zip(&grad, &values[0], |g, x| -g * x.sin())?],
            BuiltinBackward::Tanh => vec![tensor_zip(&grad, &saved_output()?, |g, y| g * (1.0 - y * y))?],
            BuiltinBackward::Sigmoid => vec![tensor_zip(&grad, &saved_output()?, |g, y| g * y * (1.0 - y))?],
            BuiltinBackward::Silu => vec![tensor_zip(&grad, &values[0], |g, x| {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                g * sigmoid * (1.0 + x * (1.0 - sigmoid))
            })?],
            BuiltinBackward::Relu => vec![tensor_zip(&grad, &values[0], |g, x| if x > 0.0 { g } else { 0.0 })?],
            BuiltinBackward::Abs => vec![tensor_zip(&grad, &values[0], |g, x| {
                if x > 0.0 { g } else if x < 0.0 { -g } else { 0.0 }
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
                        let dot: f32 = (0..width).map(|i| {
                            let index = (outer_index * width + i) * inner + inner_index;
                            grad.data[index] * output.data[index]
                        }).sum();
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
                vec![GlobalTensor::from_vec(vec![scalar; values[0].data.len()], &values[0].shape)?]
            }
            BuiltinBackward::Reshape => vec![GlobalTensor::from_vec(grad.data, &values[0].shape)?],
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
            _ => 1,
        }
    }

    fn backward(&self, inputs: &[TensorView<'_>], saved: &[TensorView<'_>], grad: TensorView<'_>) -> MlResult<Vec<Option<GlobalTensor<f32>>>> {
        if inputs.len() != self.input_count() {
            return Err(AutogradError::BackwardArityMismatch { expected: self.input_count(), got: inputs.len() }.into());
        }
        let values = inputs.iter().map(|view| GlobalTensor::from_vec(view.data.to_vec(), view.shape)).collect::<MlResult<Vec<_>>>()?;
        let grad = GlobalTensor::from_vec(grad.data.to_vec(), grad.shape)?;
        let results = match &self.0 {
            BuiltinBackward::Transpose(axes) => {
                let mut inverse = vec![0; axes.len()];
                for (output_axis, &input_axis) in axes.iter().enumerate() { inverse[input_axis] = output_axis; }
                vec![GlobalTensor::from_vec(permute_data(&grad.data, &grad.shape, &inverse), &values[0].shape)?]
            }
            BuiltinBackward::Concat { axis, sizes } => {
                let outer: usize = grad.shape[..*axis].iter().product();
                let inner: usize = grad.shape[*axis + 1..].iter().product();
                let axis_width = grad.shape[*axis] * inner;
                let mut running = 0;
                let offsets: Vec<_> = sizes.iter().map(|size| { let offset = running; running += size * inner; offset }).collect();
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
                    for i in 0..spec.m { for p in 0..spec.k { for j in 0..spec.n {
                        let upstream = grad.data[(batch * spec.m + i) * spec.n + j];
                        dl[(lb * spec.m + i) * spec.k + p] += upstream * rhs.data[(rb * spec.k + p) * spec.n + j];
                        dr[(rb * spec.k + p) * spec.n + j] += lhs.data[(lb * spec.m + i) * spec.k + p] * upstream;
                    }}}
                }
                vec![GlobalTensor::from_vec(dl, &lhs.shape)?, GlobalTensor::from_vec(dr, &rhs.shape)?]
            }
            BuiltinBackward::Conv2d { stride, padding } => {
                let (dx, dw, db) = conv2d_backward_data(&values[0], &values[1], &grad, *stride, *padding)?;
                vec![dx, dw, db]
            }
            BuiltinBackward::MaxPool2d { kernel, stride } => {
                let mask = saved.first().ok_or(AutogradError::BackwardArityMismatch {
                    expected: 1,
                    got: 0,
                })?;
                vec![max_pool2d_backward_data(&values[0], mask, &grad, *kernel, *stride)?]
            }
            BuiltinBackward::AvgPool2d { kernel, stride } => {
                vec![avg_pool2d_backward_data(&values[0], &grad, *kernel, *stride)?]
            }
            BuiltinBackward::NearestUpsample2d { scale } => {
                vec![nearest_upsample2d_backward_data(&values[0], &grad, *scale)?]
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

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self {
            id: ContextId(NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed)),
            state: Rc::new(RefCell::new(ContextState {
                tensors: HashMap::new(),
                next_node: 0,
                graph: HashMap::new(),
                tracked: HashSet::new(),
                leaves: HashSet::new(),
                retained_gradients: HashSet::new(),
                gradients: HashMap::new(),
                consumed: HashSet::new(),
                no_grad_depth: 0,
            })),
            _not_sync: Rc::new(Cell::new(())),
        }
    }

    pub fn id(&self) -> ContextId {
        self.id
    }

    pub fn tensor(&self, data: Vec<f32>, shape: &[usize]) -> MlResult<ContextTensor> {
        let tensor = GlobalTensor::from_vec(data, shape)?;
        self.insert(tensor)
    }

    pub fn scalar(&self, value: f32) -> MlResult<ContextTensor> {
        self.tensor(vec![value], &[])
    }

    pub fn variable(
        &self,
        data: Vec<f32>,
        shape: &[usize],
        requires_grad: RequiresGrad,
    ) -> MlResult<ContextVariable> {
        let tensor = self.tensor(data, shape)?;
        if requires_grad == RequiresGrad::Yes {
            let mut state = self.state
                .try_borrow_mut()
                .map_err(|_| ContextError::BorrowConflict)?;
            state.tracked.insert(tensor.node_id());
            state.leaves.insert(tensor.node_id());
        }
        Ok(ContextVariable {
            tensor,
            requires_grad: requires_grad == RequiresGrad::Yes,
        })
    }

    pub fn input(&self, data: Vec<f32>, shape: &[usize]) -> MlResult<ContextVariable> {
        self.variable(data, shape, RequiresGrad::No)
    }

    pub fn parameter(&self, data: Vec<f32>, shape: &[usize]) -> MlResult<ContextVariable> {
        self.variable(data, shape, RequiresGrad::Yes)
    }

    fn insert(&self, tensor: GlobalTensor<f32>) -> MlResult<ContextTensor> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        let id = NodeId::from_raw(state.next_node);
        state.next_node += 1;
        state.tensors.insert(id, tensor);
        Ok(ContextTensor(Rc::new(ContextTensorHandle {
            context_id: self.id,
            node_id: id,
            context: Rc::downgrade(&self.state),
        })))
    }

    fn validate(&self, tensor: &ContextTensor) -> MlResult<()> {
        if tensor.context_id() != self.id {
            return Err(ContextError::Mismatch.into());
        }
        if !self
            .state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?
            .tensors
            .contains_key(&tensor.node_id())
        {
            return Err(ContextError::UnknownTensor(tensor.node_id()).into());
        }
        Ok(())
    }

    pub fn with_tensor<R>(
        &self,
        tensor: &ContextTensor,
        f: impl FnOnce(TensorView<'_>) -> R,
    ) -> MlResult<R> {
        self.validate(tensor)?;
        let state = self
            .state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?;
        let value = state
            .tensors
            .get(&tensor.node_id())
            .ok_or(ContextError::UnknownTensor(tensor.node_id()))?;
        Ok(f(TensorView {
            data: &value.data,
            shape: &value.shape,
        }))
    }

    fn binary(
        &self,
        lhs: &ContextTensor,
        rhs: &ContextTensor,
        op: &'static str,
        backward: BuiltinBackward,
        f: impl Fn(f32, f32) -> f32,
    ) -> MlResult<ContextTensor> {
        self.validate(lhs)?;
        self.validate(rhs)?;
        let (left, right) = (lhs.snapshot()?, rhs.snapshot()?);
        let shape = broadcast_shape(&left.shape, &right.shape).ok_or_else(|| {
            TensorError::InvalidOperation {
                op,
                reason: format!("shapes {:?} and {:?} cannot be broadcast", left.shape, right.shape),
            }
        })?;
        let left_data = broadcast_data(&left, &shape)?;
        let right_data = broadcast_data(&right, &shape)?;
        let output = self.tensor(
            left_data
                .into_iter()
                .zip(right_data)
                .map(|(a, b)| f(a, b))
                .collect(),
            &shape,
        )?;
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        if state.no_grad_depth == 0
            && (state.tracked.contains(&lhs.node_id()) || state.tracked.contains(&rhs.node_id()))
        {
            state.tracked.insert(output.node_id());
            state.graph.insert(
                output.node_id(),
                GraphNode {
                    inputs: vec![lhs.node_id(), rhs.node_id()],
                    saved: Vec::new(),
                    owned_saved: Vec::new(),
                    backward: into_node_backward(backward),
                },
            );
        }
        Ok(output)
    }

    pub fn add(&self, lhs: &ContextTensor, rhs: &ContextTensor) -> MlResult<ContextTensor> {
        self.binary(lhs, rhs, "add", BuiltinBackward::Add, |a, b| a + b)
    }

    pub fn mul(&self, lhs: &ContextTensor, rhs: &ContextTensor) -> MlResult<ContextTensor> {
        self.binary(lhs, rhs, "mul", BuiltinBackward::Mul, |a, b| a * b)
    }

    pub fn sub(&self, lhs: &ContextTensor, rhs: &ContextTensor) -> MlResult<ContextTensor> {
        self.binary(lhs, rhs, "sub", BuiltinBackward::Sub, |a, b| a - b)
    }

    pub fn div(&self, lhs: &ContextTensor, rhs: &ContextTensor) -> MlResult<ContextTensor> {
        self.binary(lhs, rhs, "div", BuiltinBackward::Div, |a, b| a / b)
    }

    fn unary(
        &self,
        input: &ContextTensor,
        backward: BuiltinBackward,
        f: impl Fn(f32) -> f32,
    ) -> MlResult<ContextTensor> {
        self.validate(input)?;
        let value = input.snapshot()?;
        let shape = value.shape.clone();
        let output = self.tensor(value.data.into_iter().map(f).collect(), &shape)?;
        self.record(output.node_id(), vec![input.node_id()], backward)?;
        Ok(output)
    }

    fn record(
        &self,
        output: NodeId,
        inputs: Vec<NodeId>,
        backward: BuiltinBackward,
    ) -> MlResult<()> {
        self.record_with_saved(output, inputs, backward, Vec::new())
    }

    fn record_with_saved(
        &self,
        output: NodeId,
        inputs: Vec<NodeId>,
        backward: BuiltinBackward,
        saved_values: Vec<GlobalTensor<f32>>,
    ) -> MlResult<()> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        if state.no_grad_depth == 0 && inputs.iter().any(|id| state.tracked.contains(id)) {
            state.tracked.insert(output);
            let mut saved = match backward {
                BuiltinBackward::Exp
                | BuiltinBackward::Sqrt
                | BuiltinBackward::Tanh
                | BuiltinBackward::Sigmoid
                | BuiltinBackward::Softmax { .. } => vec![output],
                _ => Vec::new(),
            };
            let mut owned_saved = Vec::with_capacity(saved_values.len());
            for value in saved_values {
                let id = NodeId::from_raw(state.next_node);
                state.next_node += 1;
                state.tensors.insert(id, value);
                owned_saved.push(id);
            }
            saved.extend(owned_saved.iter().copied());
            state.graph.insert(output, GraphNode {
                inputs,
                saved,
                owned_saved,
                backward: into_node_backward(backward),
            });
        }
        Ok(())
    }

    pub fn neg(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.unary(input, BuiltinBackward::Neg, |x| -x)
    }

    pub fn square(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.unary(input, BuiltinBackward::Square, |x| x * x)
    }

    pub fn exp(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.unary(input, BuiltinBackward::Exp, f32::exp)
    }

    pub fn log(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.unary(input, BuiltinBackward::Log, f32::ln)
    }

    pub fn sqrt(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.unary(input, BuiltinBackward::Sqrt, f32::sqrt)
    }

    pub fn powf(&self, input: &ContextTensor, exponent: f32) -> MlResult<ContextTensor> {
        self.unary(input, BuiltinBackward::Pow(exponent), |x| x.powf(exponent))
    }

    pub fn sin(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.unary(input, BuiltinBackward::Sin, f32::sin)
    }

    pub fn cos(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.unary(input, BuiltinBackward::Cos, f32::cos)
    }

    pub fn tanh(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.unary(input, BuiltinBackward::Tanh, f32::tanh)
    }

    pub fn sigmoid(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.unary(input, BuiltinBackward::Sigmoid, |x| 1.0 / (1.0 + (-x).exp()))
    }

    pub fn silu(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.unary(input, BuiltinBackward::Silu, |x| x / (1.0 + (-x).exp()))
    }

    pub fn relu(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.unary(input, BuiltinBackward::Relu, |x| x.max(0.0))
    }

    pub fn abs(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.unary(input, BuiltinBackward::Abs, f32::abs)
    }

    pub fn softmax(&self, input: &ContextTensor, axis: usize) -> MlResult<ContextTensor> {
        self.validate(input)?;
        let value = input.snapshot()?;
        if axis >= value.shape.len() {
            return Err(TensorError::InvalidAxis { axis, shape: value.shape }.into());
        }
        let outer: usize = value.shape[..axis].iter().product();
        let width = value.shape[axis];
        let inner: usize = value.shape[axis + 1..].iter().product();
        let mut data = vec![0.0; value.data.len()];
        for outer_index in 0..outer {
            for inner_index in 0..inner {
                let maximum = (0..width).map(|i| value.data[(outer_index * width + i) * inner + inner_index])
                    .fold(f32::NEG_INFINITY, f32::max);
                let normalizer: f32 = (0..width).map(|i| {
                    (value.data[(outer_index * width + i) * inner + inner_index] - maximum).exp()
                }).sum();
                for i in 0..width {
                    let index = (outer_index * width + i) * inner + inner_index;
                    data[index] = (value.data[index] - maximum).exp() / normalizer;
                }
            }
        }
        let output = self.tensor(data, &value.shape)?;
        self.record(output.node_id(), vec![input.node_id()], BuiltinBackward::Softmax { axis })?;
        Ok(output)
    }

    pub fn sum(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.validate(input)?;
        let value = input.snapshot()?;
        let output = self.scalar(value.data.iter().sum())?;
        self.record(
            output.node_id(),
            vec![input.node_id()],
            BuiltinBackward::Sum,
        )?;
        Ok(output)
    }

    pub fn reshape(&self, input: &ContextTensor, shape: &[usize]) -> MlResult<ContextTensor> {
        self.validate(input)?;
        let value = input.snapshot()?;
        let requested_len = shape.iter().product::<usize>();
        if value.data.len() != requested_len {
            return Err(TensorError::InvalidDataLength {
                expected: value.data.len(), got: requested_len,
            }.into());
        }
        let output = self.tensor(value.data, shape)?;
        self.record(output.node_id(), vec![input.node_id()], BuiltinBackward::Reshape)?;
        Ok(output)
    }

    pub fn transpose(&self, input: &ContextTensor, axes: &[usize]) -> MlResult<ContextTensor> {
        self.validate(input)?;
        let value = input.snapshot()?;
        validate_permutation(&value.shape, axes)?;
        let output_shape: Vec<_> = axes.iter().map(|&axis| value.shape[axis]).collect();
        let data = permute_data(&value.data, &value.shape, axes);
        let output = self.tensor(data, &output_shape)?;
        self.record(output.node_id(), vec![input.node_id()], BuiltinBackward::Transpose(axes.to_vec()))?;
        Ok(output)
    }

    pub fn concat(&self, inputs: &[&ContextTensor], axis: usize) -> MlResult<ContextTensor> {
        if inputs.is_empty() {
            return Err(TensorError::InvalidInputCount { expected: 1, got: 0 }.into());
        }
        let values = inputs.iter().map(|input| {
            self.validate(input)?;
            input.snapshot()
        }).collect::<MlResult<Vec<_>>>()?;
        let rank = values[0].shape.len();
        if axis >= rank { return Err(TensorError::InvalidAxis { axis, shape: values[0].shape.clone() }.into()); }
        for value in &values[1..] {
            if value.shape.len() != rank || value.shape.iter().enumerate().any(|(i, dim)| i != axis && *dim != values[0].shape[i]) {
                return Err(TensorError::InvalidOperation { op: "concat", reason: "non-concatenated dimensions must match".into() }.into());
            }
        }
        let mut shape = values[0].shape.clone();
        shape[axis] = values.iter().map(|value| value.shape[axis]).sum();
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();
        let mut data = Vec::with_capacity(shape.iter().product());
        for outer_index in 0..outer {
            for value in &values {
                let chunk = value.shape[axis] * inner;
                let start = outer_index * chunk;
                data.extend_from_slice(&value.data[start..start + chunk]);
            }
        }
        let output = self.tensor(data, &shape)?;
        self.record(
            output.node_id(), inputs.iter().map(|input| input.node_id()).collect(),
            BuiltinBackward::Concat { axis, sizes: values.iter().map(|value| value.shape[axis]).collect() },
        )?;
        Ok(output)
    }

    pub fn matmul(&self, lhs: &ContextTensor, rhs: &ContextTensor) -> MlResult<ContextTensor> {
        self.validate(lhs)?;
        self.validate(rhs)?;
        let (left, right) = (lhs.snapshot()?, rhs.snapshot()?);
        let spec = MatmulSpec::new(&left.shape, &right.shape)?;
        let batch_count: usize = spec.batch_shape.iter().product();
        let mut data = vec![0.0; batch_count * spec.m * spec.n];
        for batch in 0..batch_count {
            let left_batch = broadcast_offset(batch, &spec.batch_shape, &spec.left_batch);
            let right_batch = broadcast_offset(batch, &spec.batch_shape, &spec.right_batch);
            for i in 0..spec.m {
                for j in 0..spec.n {
                    for p in 0..spec.k {
                        data[(batch * spec.m + i) * spec.n + j] +=
                            left.data[(left_batch * spec.m + i) * spec.k + p]
                                * right.data[(right_batch * spec.k + p) * spec.n + j];
                    }
                }
            }
        }
        let output = self.tensor(data, &spec.output_shape)?;
        self.record(
            output.node_id(),
            vec![lhs.node_id(), rhs.node_id()],
            BuiltinBackward::Matmul,
        )?;
        Ok(output)
    }

    pub fn conv2d(
        &self,
        input: &ContextTensor,
        weight: &ContextTensor,
        bias: &ContextTensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> MlResult<ContextTensor> {
        self.validate(input)?;
        self.validate(weight)?;
        self.validate(bias)?;
        let (input_value, weight_value, bias_value) =
            (input.snapshot()?, weight.snapshot()?, bias.snapshot()?);
        let output = conv2d_forward_data(&input_value, &weight_value, &bias_value, stride, padding)?;
        let result = self.tensor(output.data, &output.shape)?;
        self.record(
            result.node_id(),
            vec![input.node_id(), weight.node_id(), bias.node_id()],
            BuiltinBackward::Conv2d { stride, padding },
        )?;
        Ok(result)
    }

    pub fn max_pool2d(
        &self,
        input: &ContextTensor,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> MlResult<ContextTensor> {
        self.validate(input)?;
        let input_value = input.snapshot()?;
        let (output, mask) = max_pool2d_forward_data(&input_value, kernel, stride)?;
        let result = self.tensor(output.data, &output.shape)?;
        self.record_with_saved(
            result.node_id(),
            vec![input.node_id()],
            BuiltinBackward::MaxPool2d { kernel, stride },
            vec![mask],
        )?;
        Ok(result)
    }

    pub fn avg_pool2d(
        &self,
        input: &ContextTensor,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> MlResult<ContextTensor> {
        self.validate(input)?;
        let input_value = input.snapshot()?;
        let output = avg_pool2d_forward_data(&input_value, kernel, stride)?;
        let result = self.tensor(output.data, &output.shape)?;
        self.record(
            result.node_id(),
            vec![input.node_id()],
            BuiltinBackward::AvgPool2d { kernel, stride },
        )?;
        Ok(result)
    }

    pub fn nearest_upsample2d(
        &self,
        input: &ContextTensor,
        scale: (usize, usize),
    ) -> MlResult<ContextTensor> {
        self.validate(input)?;
        let input_value = input.snapshot()?;
        let output = nearest_upsample2d_forward_data(&input_value, scale)?;
        let result = self.tensor(output.data, &output.shape)?;
        self.record(
            result.node_id(),
            vec![input.node_id()],
            BuiltinBackward::NearestUpsample2d { scale },
        )?;
        Ok(result)
    }

    pub fn add_variable(
        &self,
        lhs: &ContextVariable,
        rhs: &ContextVariable,
    ) -> MlResult<ContextVariable> {
        let tensor = self.add(lhs.tensor(), rhs.tensor())?;
        Ok(ContextVariable {
            requires_grad: self.is_tracked(&tensor)?,
            tensor,
        })
    }

    pub fn mul_variable(
        &self,
        lhs: &ContextVariable,
        rhs: &ContextVariable,
    ) -> MlResult<ContextVariable> {
        let tensor = self.mul(lhs.tensor(), rhs.tensor())?;
        Ok(ContextVariable {
            requires_grad: self.is_tracked(&tensor)?,
            tensor,
        })
    }

    fn variable_from(&self, tensor: ContextTensor) -> MlResult<ContextVariable> {
        Ok(ContextVariable {
            requires_grad: self.is_tracked(&tensor)?,
            tensor,
        })
    }

    pub fn sub_variable(
        &self,
        lhs: &ContextVariable,
        rhs: &ContextVariable,
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.sub(lhs.tensor(), rhs.tensor())?)
    }

    pub fn div_variable(
        &self,
        lhs: &ContextVariable,
        rhs: &ContextVariable,
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.div(lhs.tensor(), rhs.tensor())?)
    }

    pub fn square_variable(&self, input: &ContextVariable) -> MlResult<ContextVariable> {
        self.variable_from(self.square(input.tensor())?)
    }

    pub fn neg_variable(&self, input: &ContextVariable) -> MlResult<ContextVariable> {
        self.variable_from(self.neg(input.tensor())?)
    }

    pub fn exp_variable(&self, input: &ContextVariable) -> MlResult<ContextVariable> {
        self.variable_from(self.exp(input.tensor())?)
    }

    pub fn log_variable(&self, input: &ContextVariable) -> MlResult<ContextVariable> {
        self.variable_from(self.log(input.tensor())?)
    }

    pub fn sqrt_variable(&self, input: &ContextVariable) -> MlResult<ContextVariable> {
        self.variable_from(self.sqrt(input.tensor())?)
    }

    pub fn powf_variable(&self, input: &ContextVariable, exponent: f32) -> MlResult<ContextVariable> {
        self.variable_from(self.powf(input.tensor(), exponent)?)
    }

    pub fn sin_variable(&self, input: &ContextVariable) -> MlResult<ContextVariable> {
        self.variable_from(self.sin(input.tensor())?)
    }

    pub fn cos_variable(&self, input: &ContextVariable) -> MlResult<ContextVariable> {
        self.variable_from(self.cos(input.tensor())?)
    }

    pub fn tanh_variable(&self, input: &ContextVariable) -> MlResult<ContextVariable> {
        self.variable_from(self.tanh(input.tensor())?)
    }

    pub fn sigmoid_variable(&self, input: &ContextVariable) -> MlResult<ContextVariable> {
        self.variable_from(self.sigmoid(input.tensor())?)
    }

    pub fn silu_variable(&self, input: &ContextVariable) -> MlResult<ContextVariable> {
        self.variable_from(self.silu(input.tensor())?)
    }

    pub fn relu_variable(&self, input: &ContextVariable) -> MlResult<ContextVariable> {
        self.variable_from(self.relu(input.tensor())?)
    }

    pub fn abs_variable(&self, input: &ContextVariable) -> MlResult<ContextVariable> {
        self.variable_from(self.abs(input.tensor())?)
    }

    pub fn softmax_variable(&self, input: &ContextVariable, axis: usize) -> MlResult<ContextVariable> {
        self.variable_from(self.softmax(input.tensor(), axis)?)
    }

    pub fn sum_variable(&self, input: &ContextVariable) -> MlResult<ContextVariable> {
        self.variable_from(self.sum(input.tensor())?)
    }

    pub fn reshape_variable(&self, input: &ContextVariable, shape: &[usize]) -> MlResult<ContextVariable> {
        self.variable_from(self.reshape(input.tensor(), shape)?)
    }

    pub fn transpose_variable(&self, input: &ContextVariable, axes: &[usize]) -> MlResult<ContextVariable> {
        self.variable_from(self.transpose(input.tensor(), axes)?)
    }

    pub fn concat_variables(&self, inputs: &[&ContextVariable], axis: usize) -> MlResult<ContextVariable> {
        let tensors: Vec<_> = inputs.iter().map(|input| input.tensor()).collect();
        self.variable_from(self.concat(&tensors, axis)?)
    }

    pub fn matmul_variable(
        &self,
        lhs: &ContextVariable,
        rhs: &ContextVariable,
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.matmul(lhs.tensor(), rhs.tensor())?)
    }

    pub fn conv2d_variable(
        &self,
        input: &ContextVariable,
        weight: &ContextVariable,
        bias: &ContextVariable,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.conv2d(input.tensor(), weight.tensor(), bias.tensor(), stride, padding)?)
    }

    pub fn max_pool2d_variable(
        &self,
        input: &ContextVariable,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.max_pool2d(input.tensor(), kernel, stride)?)
    }

    pub fn avg_pool2d_variable(
        &self,
        input: &ContextVariable,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.avg_pool2d(input.tensor(), kernel, stride)?)
    }

    pub fn nearest_upsample2d_variable(
        &self,
        input: &ContextVariable,
        scale: (usize, usize),
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.nearest_upsample2d(input.tensor(), scale)?)
    }

    fn is_tracked(&self, tensor: &ContextTensor) -> MlResult<bool> {
        self.validate(tensor)?;
        Ok(self
            .state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?
            .tracked
            .contains(&tensor.node_id()))
    }

    pub fn clear_graph(&self) -> MlResult<()> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        let owned_saved: Vec<_> = state
            .graph
            .values()
            .flat_map(|node| node.owned_saved.iter().copied())
            .collect();
        state.graph.clear();
        for id in owned_saved {
            state.tensors.remove(&id);
        }
        state.consumed.clear();
        Ok(())
    }

    pub fn clear_all(&self) -> MlResult<()> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        state.tensors.clear();
        state.graph.clear();
        state.tracked.clear();
        state.leaves.clear();
        state.retained_gradients.clear();
        state.gradients.clear();
        state.consumed.clear();
        Ok(())
    }

    pub fn no_grad<T>(&self, f: impl FnOnce() -> MlResult<T>) -> MlResult<T> {
        {
            let mut state = self
                .state
                .try_borrow_mut()
                .map_err(|_| ContextError::BorrowConflict)?;
            state.no_grad_depth += 1;
        }
        struct Reset<'a>(&'a ExecutionContext);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                if let Ok(mut state) = self.0.state.try_borrow_mut() {
                    state.no_grad_depth = state.no_grad_depth.saturating_sub(1);
                }
            }
        }
        let reset = Reset(self);
        let result = f();
        drop(reset);
        result
    }

    pub fn graph_stats(&self) -> MlResult<GraphStats> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?;
        Ok(GraphStats {
            tensors: state.tensors.len(),
            graph_nodes: state.graph.len(),
            dynamic_backward_nodes: state.graph.len(),
            saved_tensor_references: state.graph.values().map(|node| node.saved.len()).sum(),
            no_grad_depth: state.no_grad_depth,
        })
    }

    pub fn backward(&self, output: &ContextVariable, options: BackwardOptions<'_>) -> MlResult<()> {
        self.validate(output.tensor())?;
        if !output.requires_grad {
            return Err(AutogradError::NodeNotFound(output.tensor.node_id()).into());
        }
        let output_value = output.tensor.snapshot()?;
        let seed = if let Some(gradient) = options.gradient {
            self.validate(gradient)?;
            let gradient = gradient.snapshot()?;
            if gradient.shape != output_value.shape {
                return Err(AutogradError::GradientShapeMismatch {
                    expected: output_value.shape,
                    got: gradient.shape,
                }
                .into());
            }
            gradient
        } else {
            if output_value.data.len() != 1 {
                return Err(AutogradError::OutputNotScalar(output_value.shape).into());
            }
            GlobalTensor::from_vec(vec![1.0], &output_value.shape)?
        };

        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        if state.consumed.contains(&output.tensor.node_id()) {
            return Err(AutogradError::GraphAlreadyFreed(output.tensor.node_id()).into());
        }
        state.gradients.clear();
        state.gradients.insert(output.tensor.node_id(), seed);

        let mut reachable = Vec::new();
        let mut seen = HashSet::new();
        fn visit(
            id: NodeId,
            state: &ContextState,
            seen: &mut HashSet<NodeId>,
            out: &mut Vec<NodeId>,
        ) {
            if !seen.insert(id) {
                return;
            }
            if let Some(node) = state.graph.get(&id) {
                for input in &node.inputs {
                    visit(*input, state, seen, out);
                }
                out.push(id);
            }
        }
        visit(output.tensor.node_id(), &state, &mut seen, &mut reachable);

        for id in reachable.into_iter().rev() {
            let node = state
                .graph
                .get(&id)
                .ok_or(AutogradError::NodeNotFound(id))?;
            let input_ids = node.inputs.clone();
            let grad = state
                .gradients
                .get(&id)
                .cloned()
                .ok_or(AutogradError::NodeNotFound(id))?;
            let values = node
                .inputs
                .iter()
                .map(|input| {
                    state
                        .tensors
                        .get(input)
                        .cloned()
                        .ok_or(ContextError::UnknownTensor(*input))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let op = &node.backward;
            let saved = node.saved.iter().map(|saved| {
                state.tensors.get(saved).cloned().ok_or(ContextError::UnknownTensor(*saved))
            }).collect::<Result<Vec<_>, _>>()?;
            let input_views: Vec<_> = values.iter().map(|value| TensorView {
                data: &value.data,
                shape: &value.shape,
            }).collect();
            let saved_views: Vec<_> = saved.iter().map(|value| TensorView {
                data: &value.data,
                shape: &value.shape,
            }).collect();
            let results = op.backward(
                &input_views,
                &saved_views,
                TensorView { data: &grad.data, shape: &grad.shape },
            )?;
            if results.len() != op.input_count() {
                return Err(AutogradError::BackwardArityMismatch {
                    expected: op.input_count(),
                    got: results.len(),
                }.into());
            }
            let grads = results.into_iter().map(|result| result.ok_or_else(|| {
                AutogradError::BackwardNotSupported(format!(
                    "{} returned a non-differentiable input", op.name()
                )).into()
            })).collect::<MlResult<Vec<_>>>()?;
            if grads.len() != input_ids.len() {
                return Err(AutogradError::BackwardArityMismatch {
                    expected: input_ids.len(),
                    got: grads.len(),
                }
                .into());
            }
            for (input, incoming) in input_ids.into_iter().zip(grads) {
                if !state.tracked.contains(&input) {
                    continue;
                }
                if let Some(existing) = state.gradients.get_mut(&input) {
                    for (dst, src) in existing.data.iter_mut().zip(incoming.data) {
                        *dst += src;
                    }
                } else {
                    state.gradients.insert(input, incoming);
                }
            }
        }
        if !options.retain_graph {
            for id in seen {
                if let Some(node) = state.graph.remove(&id) {
                    for saved in node.owned_saved {
                        state.tensors.remove(&saved);
                    }
                }
            }
            state.consumed.insert(output.tensor.node_id());
        }
        let keep: HashSet<_> = state
            .leaves
            .union(&state.retained_gradients)
            .copied()
            .collect();
        state.gradients.retain(|id, _| keep.contains(id));
        Ok(())
    }
}

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
        | BuiltinBackward::Tanh
        | BuiltinBackward::Sigmoid
        | BuiltinBackward::Silu
        | BuiltinBackward::Relu
        | BuiltinBackward::Abs
        | BuiltinBackward::Softmax { .. }
        | BuiltinBackward::Sum
        | BuiltinBackward::Reshape) => Box::new(ElementwiseBackward(other)),
        other @ (BuiltinBackward::Transpose(_)
        | BuiltinBackward::Concat { .. }
        | BuiltinBackward::Matmul
        | BuiltinBackward::Conv2d { .. }
        | BuiltinBackward::MaxPool2d { .. }
        | BuiltinBackward::AvgPool2d { .. }
        | BuiltinBackward::NearestUpsample2d { .. }) => Box::new(StructuralBackward(other)),
    }
}

fn nearest_upsample2d_spec(
    input: &[usize],
    scale: (usize, usize),
) -> MlResult<(usize, usize)> {
    let output_height = input.get(2).and_then(|height| height.checked_mul(scale.0));
    let output_width = input.get(3).and_then(|width| width.checked_mul(scale.1));
    if input.len() != 4
        || scale.0 == 0
        || scale.1 == 0
        || output_height.is_none()
        || output_width.is_none()
    {
        return Err(TensorError::InvalidOperation {
            op: "nearest_upsample2d",
            reason: format!(
                "expected input [N,C,H,W] with non-zero scales and representable output dimensions; got {input:?}, scale={scale:?}"
            ),
        }
        .into());
    }
    Ok((
        output_height.unwrap_or_default(),
        output_width.unwrap_or_default(),
    ))
}

fn nearest_upsample2d_forward_data(
    input: &GlobalTensor<f32>,
    scale: (usize, usize),
) -> MlResult<GlobalTensor<f32>> {
    let (oh, ow) = nearest_upsample2d_spec(&input.shape, scale)?;
    let (n, c, h, w) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
    let mut output = vec![0.0; n * c * oh * ow];
    for batch in 0..n {
        for channel in 0..c {
            for y in 0..oh {
                for x in 0..ow {
                    let input_index = ((batch * c + channel) * h + y / scale.0) * w
                        + x / scale.1;
                    let output_index = ((batch * c + channel) * oh + y) * ow + x;
                    output[output_index] = input.data[input_index];
                }
            }
        }
    }
    GlobalTensor::from_vec(output, &[n, c, oh, ow])
}

fn nearest_upsample2d_backward_data(
    input: &GlobalTensor<f32>,
    grad: &GlobalTensor<f32>,
    scale: (usize, usize),
) -> MlResult<GlobalTensor<f32>> {
    let (oh, ow) = nearest_upsample2d_spec(&input.shape, scale)?;
    let expected = vec![input.shape[0], input.shape[1], oh, ow];
    if grad.shape != expected {
        return Err(AutogradError::GradientShapeMismatch {
            expected,
            got: grad.shape.clone(),
        }
        .into());
    }
    let (n, c, h, w) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
    let mut dx = vec![0.0; input.data.len()];
    for batch in 0..n {
        for channel in 0..c {
            for y in 0..oh {
                for x in 0..ow {
                    let input_index = ((batch * c + channel) * h + y / scale.0) * w
                        + x / scale.1;
                    let output_index = ((batch * c + channel) * oh + y) * ow + x;
                    dx[input_index] += grad.data[output_index];
                }
            }
        }
    }
    GlobalTensor::from_vec(dx, &input.shape)
}

fn pool2d_spec(
    input: &[usize],
    kernel: (usize, usize),
    stride: (usize, usize),
    op: &'static str,
) -> MlResult<(usize, usize)> {
    if input.len() != 4
        || kernel.0 == 0
        || kernel.1 == 0
        || stride.0 == 0
        || stride.1 == 0
        || input.get(2).is_none_or(|height| *height < kernel.0)
        || input.get(3).is_none_or(|width| *width < kernel.1)
    {
        return Err(TensorError::InvalidOperation {
            op,
            reason: format!(
                "expected input [N,C,H,W] with non-zero kernel/stride fitting the input; got {input:?}, kernel={kernel:?}, stride={stride:?}"
            ),
        }
        .into());
    }
    Ok((
        (input[2] - kernel.0) / stride.0 + 1,
        (input[3] - kernel.1) / stride.1 + 1,
    ))
}

fn max_pool2d_forward_data(
    input: &GlobalTensor<f32>,
    kernel: (usize, usize),
    stride: (usize, usize),
) -> MlResult<(GlobalTensor<f32>, GlobalTensor<f32>)> {
    let (oh, ow) = pool2d_spec(&input.shape, kernel, stride, "max_pool2d")?;
    let (n, c, h, w) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
    let mut output = vec![f32::NEG_INFINITY; n * c * oh * ow];
    let mut mask = vec![0.0; output.len()];
    for batch in 0..n {
        for channel in 0..c {
            for y in 0..oh {
                for x in 0..ow {
                    let output_index = ((batch * c + channel) * oh + y) * ow + x;
                    for ky in 0..kernel.0 {
                        for kx in 0..kernel.1 {
                            let input_index = ((batch * c + channel) * h + y * stride.0 + ky)
                                * w
                                + x * stride.1
                                + kx;
                            if input.data[input_index] > output[output_index] {
                                output[output_index] = input.data[input_index];
                                mask[output_index] = input_index as f32;
                            }
                        }
                    }
                }
            }
        }
    }
    let shape = [n, c, oh, ow];
    Ok((
        GlobalTensor::from_vec(output, &shape)?,
        GlobalTensor::from_vec(mask, &shape)?,
    ))
}

fn max_pool2d_backward_data(
    input: &GlobalTensor<f32>,
    mask: &TensorView<'_>,
    grad: &GlobalTensor<f32>,
    kernel: (usize, usize),
    stride: (usize, usize),
) -> MlResult<GlobalTensor<f32>> {
    let (oh, ow) = pool2d_spec(&input.shape, kernel, stride, "max_pool2d")?;
    let expected = vec![input.shape[0], input.shape[1], oh, ow];
    if grad.shape != expected || mask.shape != expected {
        return Err(AutogradError::GradientShapeMismatch {
            expected,
            got: grad.shape.clone(),
        }
        .into());
    }
    let mut dx = vec![0.0; input.data.len()];
    for (&upstream, &saved_index) in grad.data.iter().zip(mask.data) {
        let index = saved_index as usize;
        if !saved_index.is_finite() || saved_index < 0.0 || index >= dx.len() {
            return Err(TensorError::InvalidOperation {
                op: "max_pool2d_backward",
                reason: "saved maximum index is invalid".into(),
            }
            .into());
        }
        dx[index] += upstream;
    }
    GlobalTensor::from_vec(dx, &input.shape)
}

fn avg_pool2d_forward_data(
    input: &GlobalTensor<f32>,
    kernel: (usize, usize),
    stride: (usize, usize),
) -> MlResult<GlobalTensor<f32>> {
    let (oh, ow) = pool2d_spec(&input.shape, kernel, stride, "avg_pool2d")?;
    let (n, c, h, w) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
    let mut output = vec![0.0; n * c * oh * ow];
    let area = (kernel.0 * kernel.1) as f32;
    for batch in 0..n {
        for channel in 0..c {
            for y in 0..oh {
                for x in 0..ow {
                    let output_index = ((batch * c + channel) * oh + y) * ow + x;
                    for ky in 0..kernel.0 {
                        for kx in 0..kernel.1 {
                            let input_index = ((batch * c + channel) * h + y * stride.0 + ky)
                                * w
                                + x * stride.1
                                + kx;
                            output[output_index] += input.data[input_index] / area;
                        }
                    }
                }
            }
        }
    }
    GlobalTensor::from_vec(output, &[n, c, oh, ow])
}

fn avg_pool2d_backward_data(
    input: &GlobalTensor<f32>,
    grad: &GlobalTensor<f32>,
    kernel: (usize, usize),
    stride: (usize, usize),
) -> MlResult<GlobalTensor<f32>> {
    let (oh, ow) = pool2d_spec(&input.shape, kernel, stride, "avg_pool2d")?;
    let expected = vec![input.shape[0], input.shape[1], oh, ow];
    if grad.shape != expected {
        return Err(AutogradError::GradientShapeMismatch {
            expected,
            got: grad.shape.clone(),
        }
        .into());
    }
    let (n, c, h, w) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
    let mut dx = vec![0.0; input.data.len()];
    let area = (kernel.0 * kernel.1) as f32;
    for batch in 0..n {
        for channel in 0..c {
            for y in 0..oh {
                for x in 0..ow {
                    let upstream = grad.data[((batch * c + channel) * oh + y) * ow + x] / area;
                    for ky in 0..kernel.0 {
                        for kx in 0..kernel.1 {
                            let input_index = ((batch * c + channel) * h + y * stride.0 + ky)
                                * w
                                + x * stride.1
                                + kx;
                            dx[input_index] += upstream;
                        }
                    }
                }
            }
        }
    }
    GlobalTensor::from_vec(dx, &input.shape)
}

fn conv2d_spec(
    input: &[usize],
    weight: &[usize],
    bias: &[usize],
    stride: (usize, usize),
    padding: (usize, usize),
) -> MlResult<(usize, usize)> {
    let padded_height = padding
        .0
        .checked_mul(2)
        .and_then(|padding| input.get(2)?.checked_add(padding));
    let padded_width = padding
        .1
        .checked_mul(2)
        .and_then(|padding| input.get(3)?.checked_add(padding));
    if input.len() != 4
        || weight.len() != 4
        || bias.len() != 1
        || input[1] != weight[1]
        || bias[0] != weight[0]
        || stride.0 == 0
        || stride.1 == 0
        || padded_height.is_none_or(|height| height < weight[2])
        || padded_width.is_none_or(|width| width < weight[3])
    {
        return Err(TensorError::InvalidOperation {
            op: "conv2d",
            reason: format!("expected input [N,C,H,W], weight [O,C,kH,kW], bias [O]; got {input:?}, {weight:?}, {bias:?}"),
        }.into());
    }
    Ok((
        (padded_height.unwrap_or_default() - weight[2]) / stride.0 + 1,
        (padded_width.unwrap_or_default() - weight[3]) / stride.1 + 1,
    ))
}

fn conv2d_forward_data(input: &GlobalTensor<f32>, weight: &GlobalTensor<f32>, bias: &GlobalTensor<f32>, stride: (usize, usize), padding: (usize, usize)) -> MlResult<GlobalTensor<f32>> {
    let (oh, ow) = conv2d_spec(&input.shape, &weight.shape, &bias.shape, stride, padding)?;
    let (n, ci, h, w, co, kh, kw) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3], weight.shape[0], weight.shape[2], weight.shape[3]);
    let mut output = vec![0.0; n * co * oh * ow];
    for b in 0..n { for oc in 0..co { for y in 0..oh { for x in 0..ow {
        let mut sum = bias.data[oc];
        for ic in 0..ci { for ky in 0..kh { for kx in 0..kw {
            let iy = y * stride.0 + ky;
            let ix = x * stride.1 + kx;
            if iy >= padding.0 && ix >= padding.1 {
                let sy = iy - padding.0;
                let sx = ix - padding.1;
                if sy < h && sx < w {
                    sum += input.data[((b * ci + ic) * h + sy) * w + sx]
                        * weight.data[((oc * ci + ic) * kh + ky) * kw + kx];
                }
            }
        }}}
        output[((b * co + oc) * oh + y) * ow + x] = sum;
    }}}}
    GlobalTensor::from_vec(output, &[n, co, oh, ow])
}

fn conv2d_backward_data(input: &GlobalTensor<f32>, weight: &GlobalTensor<f32>, grad: &GlobalTensor<f32>, stride: (usize, usize), padding: (usize, usize)) -> MlResult<(GlobalTensor<f32>, GlobalTensor<f32>, GlobalTensor<f32>)> {
    let bias_shape = [weight.shape[0]];
    let (oh, ow) = conv2d_spec(&input.shape, &weight.shape, &bias_shape, stride, padding)?;
    let expected = vec![input.shape[0], weight.shape[0], oh, ow];
    if grad.shape != expected { return Err(AutogradError::GradientShapeMismatch { expected, got: grad.shape.clone() }.into()); }
    let (n, ci, h, w, co, kh, kw) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3], weight.shape[0], weight.shape[2], weight.shape[3]);
    let mut dx = vec![0.0; input.data.len()];
    let mut dw = vec![0.0; weight.data.len()];
    let mut db = vec![0.0; co];
    for b in 0..n { for oc in 0..co { for y in 0..oh { for x in 0..ow {
        let upstream = grad.data[((b * co + oc) * oh + y) * ow + x];
        db[oc] += upstream;
        for ic in 0..ci { for ky in 0..kh { for kx in 0..kw {
            let iy = y * stride.0 + ky;
            let ix = x * stride.1 + kx;
            if iy >= padding.0 && ix >= padding.1 {
                let sy = iy - padding.0;
                let sx = ix - padding.1;
                if sy < h && sx < w {
                    let input_index = ((b * ci + ic) * h + sy) * w + sx;
                    let weight_index = ((oc * ci + ic) * kh + ky) * kw + kx;
                    dx[input_index] += upstream * weight.data[weight_index];
                    dw[weight_index] += upstream * input.data[input_index];
                }
            }
        }}}
    }}}}
    Ok((GlobalTensor::from_vec(dx, &input.shape)?, GlobalTensor::from_vec(dw, &weight.shape)?, GlobalTensor::from_vec(db, &bias_shape)?))
}

struct MatmulSpec {
    left_batch: Vec<usize>,
    right_batch: Vec<usize>,
    batch_shape: Vec<usize>,
    output_shape: Vec<usize>,
    m: usize,
    k: usize,
    n: usize,
}

impl MatmulSpec {
    fn new(left: &[usize], right: &[usize]) -> MlResult<Self> {
        if left.is_empty() || right.is_empty() {
            return Err(TensorError::MatrixMultiplicationError {
                left_shape: left.to_vec(), right_shape: right.to_vec(),
            }.into());
        }
        let left_vector = left.len() == 1;
        let right_vector = right.len() == 1;
        let (m, k) = if left_vector { (1, left[0]) } else { (left[left.len() - 2], left[left.len() - 1]) };
        let (right_k, n) = if right_vector { (right[0], 1) } else { (right[right.len() - 2], right[right.len() - 1]) };
        let left_batch = if left_vector { vec![] } else { left[..left.len() - 2].to_vec() };
        let right_batch = if right_vector { vec![] } else { right[..right.len() - 2].to_vec() };
        let batch_shape = broadcast_shape(&left_batch, &right_batch).ok_or_else(|| {
            TensorError::MatrixMultiplicationError { left_shape: left.to_vec(), right_shape: right.to_vec() }
        })?;
        if k != right_k {
            return Err(TensorError::MatrixMultiplicationError {
                left_shape: left.to_vec(), right_shape: right.to_vec(),
            }.into());
        }
        let mut output_shape = batch_shape.clone();
        if !left_vector { output_shape.push(m); }
        if !right_vector { output_shape.push(n); }
        Ok(Self { left_batch, right_batch, batch_shape, output_shape, m, k, n })
    }
}

fn validate_permutation(shape: &[usize], axes: &[usize]) -> MlResult<()> {
    if axes.len() != shape.len() {
        return Err(TensorError::InvalidOperation { op: "transpose", reason: "axis count must equal rank".into() }.into());
    }
    let mut seen = vec![false; shape.len()];
    for &axis in axes {
        if axis >= shape.len() || seen[axis] {
            return Err(TensorError::InvalidAxis { axis, shape: shape.to_vec() }.into());
        }
        seen[axis] = true;
    }
    Ok(())
}

fn permute_data(data: &[f32], input_shape: &[usize], axes: &[usize]) -> Vec<f32> {
    let output_shape: Vec<_> = axes.iter().map(|&axis| input_shape[axis]).collect();
    let mut output = vec![0.0; data.len()];
    for output_flat in 0..output.len() {
        let mut remainder = output_flat;
        let mut input_coordinates = vec![0; input_shape.len()];
        for output_axis in (0..output_shape.len()).rev() {
            let coordinate = remainder % output_shape[output_axis];
            remainder /= output_shape[output_axis];
            input_coordinates[axes[output_axis]] = coordinate;
        }
        let input_flat = input_coordinates.iter().zip(input_shape).fold(0, |flat, (&coordinate, &dim)| flat * dim + coordinate);
        output[output_flat] = data[input_flat];
    }
    output
}

fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Option<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    let mut output = vec![1; rank];
    for offset in 0..rank {
        let left = lhs.len().checked_sub(offset + 1).map(|i| lhs[i]).unwrap_or(1);
        let right = rhs.len().checked_sub(offset + 1).map(|i| rhs[i]).unwrap_or(1);
        if left != right && left != 1 && right != 1 { return None; }
        output[rank - offset - 1] = left.max(right);
    }
    Some(output)
}

fn broadcast_offset(flat: usize, output_shape: &[usize], input_shape: &[usize]) -> usize {
    let rank_delta = output_shape.len() - input_shape.len();
    let mut remainder = flat;
    let mut coordinates = vec![0; output_shape.len()];
    for axis in (0..output_shape.len()).rev() {
        coordinates[axis] = remainder % output_shape[axis];
        remainder /= output_shape[axis];
    }
    let mut input_offset = 0;
    for (axis, &dim) in input_shape.iter().enumerate() {
        let coordinate = if dim == 1 { 0 } else { coordinates[axis + rank_delta] };
        input_offset = input_offset * dim + coordinate;
    }
    input_offset
}

fn broadcast_data(input: &GlobalTensor<f32>, output_shape: &[usize]) -> MlResult<Vec<f32>> {
    if broadcast_shape(&input.shape, output_shape).as_deref() != Some(output_shape) {
        return Err(TensorError::InvalidOperation {
            op: "broadcast", reason: format!("cannot broadcast {:?} to {:?}", input.shape, output_shape),
        }.into());
    }
    let length: usize = output_shape.iter().product();
    Ok((0..length).map(|flat| input.data[broadcast_offset(flat, output_shape, &input.shape)]).collect())
}

fn reduce_to_shape(input: &GlobalTensor<f32>, target_shape: &[usize]) -> MlResult<GlobalTensor<f32>> {
    if broadcast_shape(target_shape, &input.shape).as_deref() != Some(input.shape.as_slice()) {
        return Err(AutogradError::GradientShapeMismatch {
            expected: target_shape.to_vec(), got: input.shape.clone(),
        }.into());
    }
    let target_length: usize = target_shape.iter().product();
    let mut data = vec![0.0; target_length];
    for (flat, value) in input.data.iter().copied().enumerate() {
        data[broadcast_offset(flat, &input.shape, target_shape)] += value;
    }
    GlobalTensor::from_vec(data, target_shape)
}

fn tensor_map(tensor: &GlobalTensor<f32>, f: impl Fn(f32) -> f32) -> MlResult<GlobalTensor<f32>> {
    GlobalTensor::from_vec(tensor.data.iter().copied().map(f).collect(), &tensor.shape)
}

fn tensor_zip(
    lhs: &GlobalTensor<f32>,
    rhs: &GlobalTensor<f32>,
    f: impl Fn(f32, f32) -> f32,
) -> MlResult<GlobalTensor<f32>> {
    if lhs.shape != rhs.shape {
        return Err(AutogradError::GradientShapeMismatch {
            expected: rhs.shape.clone(),
            got: lhs.shape.clone(),
        }
        .into());
    }
    GlobalTensor::from_vec(
        lhs.data
            .iter()
            .copied()
            .zip(rhs.data.iter().copied())
            .map(|(a, b)| f(a, b))
            .collect(),
        &lhs.shape,
    )
}

impl ContextTensor {
    pub fn context_id(&self) -> ContextId {
        self.0.context_id
    }
    pub fn node_id(&self) -> NodeId {
        self.0.node_id
    }

    fn state(&self) -> MlResult<Rc<RefCell<ContextState>>> {
        self.0
            .context
            .upgrade()
            .ok_or_else(|| ContextError::Dropped.into())
    }

    fn snapshot(&self) -> MlResult<GlobalTensor<f32>> {
        let state = self.state()?;
        let state = state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?;
        state
            .tensors
            .get(&self.node_id())
            .cloned()
            .ok_or_else(|| ContextError::UnknownTensor(self.node_id()).into())
    }

    pub fn to_vec(&self) -> MlResult<Vec<f32>> {
        Ok(self.snapshot()?.data)
    }
    pub fn shape(&self) -> MlResult<Vec<usize>> {
        Ok(self.snapshot()?.shape)
    }

    pub fn item(&self) -> MlResult<f32> {
        let value = self.snapshot()?;
        if value.data.len() != 1 {
            return Err(TensorError::NotScalar { shape: value.shape }.into());
        }
        Ok(value.data[0])
    }

    pub fn get(&self, indices: &[usize]) -> MlResult<Option<f32>> {
        let value = self.snapshot()?;
        if indices.len() != value.shape.len() {
            return Ok(None);
        }
        let mut flat = 0usize;
        for (index, dim) in indices.iter().zip(&value.shape) {
            if index >= dim {
                return Ok(None);
            }
            flat = flat * dim + index;
        }
        Ok(value.data.get(flat).copied())
    }
}

impl ContextVariable {
    pub fn tensor(&self) -> &ContextTensor {
        &self.tensor
    }
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
    pub fn detach(&self) -> MlResult<Self> {
        let state = self.tensor.state()?;
        let ctx = ExecutionContext {
            id: self.tensor.context_id(),
            state,
            _not_sync: Rc::new(Cell::new(())),
        };
        let value = self.tensor.snapshot()?;
        ctx.variable(value.data, &value.shape, RequiresGrad::No)
    }

    pub fn retain_grad(&self) -> MlResult<()> {
        let state = self.tensor.state()?;
        let mut state = state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        if !state.tracked.contains(&self.tensor.node_id()) {
            return Err(AutogradError::NodeNotFound(self.tensor.node_id()).into());
        }
        state.retained_gradients.insert(self.tensor.node_id());
        Ok(())
    }

    pub fn grad(&self) -> MlResult<Option<GlobalTensor<f32>>> {
        let state = self.tensor.state()?;
        let state = state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?;
        Ok(state.gradients.get(&self.tensor.node_id()).cloned())
    }

    pub fn backward(&self) -> MlResult<()> {
        let state = self.tensor.state()?;
        let ctx = ExecutionContext {
            id: self.tensor.context_id(),
            state,
            _not_sync: Rc::new(Cell::new(())),
        };
        ctx.backward(self, BackwardOptions::default())
    }

    pub fn backward_with_grad(&self, gradient: &ContextTensor) -> MlResult<()> {
        let state = self.tensor.state()?;
        let ctx = ExecutionContext {
            id: self.tensor.context_id(),
            state,
            _not_sync: Rc::new(Cell::new(())),
        };
        ctx.backward(
            self,
            BackwardOptions {
                gradient: Some(gradient),
                retain_graph: false,
            },
        )
    }
}

fn context_for(tensor: &ContextTensor) -> MlResult<ExecutionContext> {
    Ok(ExecutionContext {
        id: tensor.context_id(),
        state: tensor.state()?,
        _not_sync: Rc::new(Cell::new(())),
    })
}

impl std::ops::Add<&ContextTensor> for &ContextTensor {
    type Output = MlResult<ContextTensor>;
    fn add(self, rhs: &ContextTensor) -> Self::Output {
        context_for(self)?.add(self, rhs)
    }
}

impl std::ops::Mul<&ContextTensor> for &ContextTensor {
    type Output = MlResult<ContextTensor>;
    fn mul(self, rhs: &ContextTensor) -> Self::Output {
        context_for(self)?.mul(self, rhs)
    }
}

impl std::ops::Sub<&ContextTensor> for &ContextTensor {
    type Output = MlResult<ContextTensor>;
    fn sub(self, rhs: &ContextTensor) -> Self::Output {
        context_for(self)?.sub(self, rhs)
    }
}

impl std::ops::Div<&ContextTensor> for &ContextTensor {
    type Output = MlResult<ContextTensor>;
    fn div(self, rhs: &ContextTensor) -> Self::Output {
        context_for(self)?.div(self, rhs)
    }
}

impl std::ops::Neg for &ContextTensor {
    type Output = MlResult<ContextTensor>;
    fn neg(self) -> Self::Output {
        context_for(self)?.neg(self)
    }
}

impl std::ops::Add<&ContextVariable> for &ContextVariable {
    type Output = MlResult<ContextVariable>;
    fn add(self, rhs: &ContextVariable) -> Self::Output {
        context_for(self.tensor())?.add_variable(self, rhs)
    }
}

impl std::ops::Mul<&ContextVariable> for &ContextVariable {
    type Output = MlResult<ContextVariable>;
    fn mul(self, rhs: &ContextVariable) -> Self::Output {
        context_for(self.tensor())?.mul_variable(self, rhs)
    }
}

impl std::ops::Sub<&ContextVariable> for &ContextVariable {
    type Output = MlResult<ContextVariable>;
    fn sub(self, rhs: &ContextVariable) -> Self::Output {
        context_for(self.tensor())?.sub_variable(self, rhs)
    }
}

impl std::ops::Div<&ContextVariable> for &ContextVariable {
    type Output = MlResult<ContextVariable>;
    fn div(self, rhs: &ContextVariable) -> Self::Output {
        context_for(self.tensor())?.div_variable(self, rhs)
    }
}

impl std::ops::Neg for &ContextVariable {
    type Output = MlResult<ContextVariable>;
    fn neg(self) -> Self::Output {
        context_for(self.tensor())?.neg_variable(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contexts_are_isolated_and_mismatch_is_rejected() -> MlResult<()> {
        let a = ExecutionContext::new();
        let b = ExecutionContext::new();
        let x = a.tensor(vec![1.0], &[1])?;
        let y = b.tensor(vec![2.0], &[1])?;
        assert!(matches!(
            a.add(&x, &y),
            Err(crate::MlError::ContextError(ContextError::Mismatch))
        ));
        Ok(())
    }

    #[test]
    fn dropped_context_is_reported() -> MlResult<()> {
        let tensor = {
            let ctx = ExecutionContext::new();
            ctx.tensor(vec![1.0], &[1])?
        };
        assert!(matches!(
            tensor.to_vec(),
            Err(crate::MlError::ContextError(ContextError::Dropped))
        ));
        Ok(())
    }

    #[test]
    fn no_grad_depth_is_restored_on_error() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let _: Result<(), _> = ctx.no_grad(|| Err(crate::MlError::StringError("stop".into())));
        assert_eq!(ctx.graph_stats()?.no_grad_depth, 0);
        Ok(())
    }

    #[test]
    fn backward_accumulates_fan_in_and_consumes_graph() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![3.0], &[])?;
        let xx = ctx.mul_variable(&x, &x)?;
        let y = ctx.add_variable(&xx, &x)?;
        y.backward()?;
        assert_eq!(x.grad()?.expect("leaf gradient").data, vec![7.0]);
        assert!(matches!(
            y.backward(),
            Err(crate::MlError::AutogradError(
                AutogradError::GraphAlreadyFreed(_)
            ))
        ));
        Ok(())
    }

    #[test]
    fn vector_output_requires_explicit_cotangent() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![2.0, 3.0], &[2])?;
        let y = ctx.mul_variable(&x, &x)?;
        assert!(matches!(
            y.backward(),
            Err(crate::MlError::AutogradError(
                AutogradError::OutputNotScalar(_)
            ))
        ));
        let seed = ctx.tensor(vec![1.0, 2.0], &[2])?;
        y.backward_with_grad(&seed)?;
        assert_eq!(x.grad()?.expect("leaf gradient").data, vec![4.0, 12.0]);
        Ok(())
    }

    #[test]
    fn fallible_operator_overloads_use_the_owning_context() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![2.0], &[])?;
        let y = (&x * &x)?;
        let z = (&y + &x)?;
        z.backward()?;
        assert_eq!(x.grad()?.expect("leaf gradient").data, vec![5.0]);
        Ok(())
    }

    #[test]
    fn no_grad_skips_graph_registration() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![2.0], &[])?;
        let y = ctx.no_grad(|| ctx.mul_variable(&x, &x))?;
        assert!(!y.requires_grad());
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
        Ok(())
    }

    #[test]
    fn unary_sum_chain_has_correct_gradient() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![0.5, 1.5], &[2])?;
        let squared = ctx.square_variable(&x)?;
        let exponentiated = ctx.variable_from(ctx.exp(squared.tensor())?)?;
        let loss = ctx.sum_variable(&exponentiated)?;
        loss.backward()?;
        let gradient = x.grad()?.expect("leaf gradient").data;
        assert!((gradient[0] - 2.0 * 0.5 * (0.25f32).exp()).abs() < 1e-5);
        assert!((gradient[1] - 2.0 * 1.5 * (2.25f32).exp()).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn matmul_gradient_matches_known_result() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let w = ctx.parameter(vec![2.0, 0.0, 1.0, 3.0], &[2, 2])?;
        let y = ctx.matmul_variable(&x, &w)?;
        let loss = ctx.sum_variable(&y)?;
        loss.backward()?;
        assert_eq!(
            x.grad()?.expect("x gradient").data,
            vec![2.0, 4.0, 2.0, 4.0]
        );
        assert_eq!(
            w.grad()?.expect("w gradient").data,
            vec![4.0, 4.0, 6.0, 6.0]
        );
        Ok(())
    }

    fn finite_difference_check(
        x0: f32,
        expected: impl Fn(f32) -> f32,
        build: impl Fn(&ExecutionContext, &ContextVariable) -> MlResult<ContextVariable>,
    ) -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![x0], &[])?;
        build(&ctx, &x)?.backward()?;
        let analytic = x.grad()?.expect("leaf gradient").data[0];
        let epsilon = 1e-3;
        let numeric = (expected(x0 + epsilon) - expected(x0 - epsilon)) / (2.0 * epsilon);
        let absolute = (analytic - numeric).abs();
        let relative = absolute / numeric.abs().max(1e-6);
        assert!(absolute <= 1e-3 || relative <= 1e-3,
            "analytic={analytic}, numeric={numeric}, abs={absolute}, rel={relative}");
        Ok(())
    }

    #[test]
    fn unary_gradients_match_central_finite_difference() -> MlResult<()> {
        finite_difference_check(1.3, f32::sqrt, |ctx, x| ctx.sqrt_variable(x))?;
        finite_difference_check(1.3, |x| x.powf(2.7), |ctx, x| ctx.powf_variable(x, 2.7))?;
        finite_difference_check(0.7, f32::sin, |ctx, x| ctx.sin_variable(x))?;
        finite_difference_check(0.7, f32::cos, |ctx, x| ctx.cos_variable(x))?;
        finite_difference_check(0.7, f32::tanh, |ctx, x| ctx.tanh_variable(x))?;
        finite_difference_check(0.7, |x| 1.0 / (1.0 + (-x).exp()), |ctx, x| ctx.sigmoid_variable(x))?;
        finite_difference_check(0.7, |x| x / (1.0 + (-x).exp()), |ctx, x| ctx.silu_variable(x))?;
        finite_difference_check(0.7, |x| x.max(0.0), |ctx, x| ctx.relu_variable(x))?;
        Ok(())
    }

    #[test]
    fn multidimensional_broadcast_reduces_gradients() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let rows = ctx.parameter(vec![1.0, 2.0], &[2, 1])?;
        let columns = ctx.parameter(vec![10.0, 20.0, 30.0], &[3])?;
        let product = ctx.mul_variable(&rows, &columns)?;
        assert_eq!(product.tensor().shape()?, vec![2, 3]);
        assert_eq!(product.tensor().to_vec()?, vec![10.0, 20.0, 30.0, 20.0, 40.0, 60.0]);
        ctx.sum_variable(&product)?.backward()?;
        assert_eq!(rows.grad()?.expect("row gradient").data, vec![60.0, 60.0]);
        assert_eq!(columns.grad()?.expect("column gradient").data, vec![3.0, 3.0, 3.0]);
        Ok(())
    }

    #[test]
    fn scalar_broadcast_and_incompatible_shapes_are_handled() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let values = ctx.parameter(vec![1.0, 2.0, 3.0], &[3])?;
        let scalar = ctx.parameter(vec![2.0], &[])?;
        let quotient = ctx.div_variable(&values, &scalar)?;
        ctx.sum_variable(&quotient)?.backward()?;
        assert_eq!(values.grad()?.expect("value gradient").data, vec![0.5; 3]);
        assert_eq!(scalar.grad()?.expect("scalar gradient").data, vec![-1.5]);

        let incompatible = ctx.tensor(vec![1.0; 4], &[2, 2])?;
        assert!(matches!(
            ctx.add(values.tensor(), &incompatible),
            Err(crate::MlError::TensorError(TensorError::InvalidOperation { op: "add", .. }))
        ));
        Ok(())
    }

    #[test]
    fn transpose_and_reshape_reverse_the_layout_in_backward() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input = ctx.parameter(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let transposed = ctx.transpose_variable(&input, &[1, 0])?;
        assert_eq!(transposed.tensor().to_vec()?, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let flattened = ctx.reshape_variable(&transposed, &[6])?;
        let cotangent = ctx.tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6])?;
        flattened.backward_with_grad(&cotangent)?;
        assert_eq!(input.grad()?.expect("input gradient").data, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
        Ok(())
    }

    #[test]
    fn concat_backward_splits_each_outer_block() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let left = ctx.parameter(vec![10.0, 20.0], &[2, 1])?;
        let right = ctx.parameter(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let joined = ctx.concat_variables(&[&left, &right], 1)?;
        assert_eq!(ctx.graph_stats()?.dynamic_backward_nodes, 1);
        assert_eq!(joined.tensor().to_vec()?, vec![10.0, 1.0, 2.0, 20.0, 3.0, 4.0]);
        let cotangent = ctx.tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        joined.backward_with_grad(&cotangent)?;
        assert_eq!(left.grad()?.expect("left gradient").data, vec![1.0, 4.0]);
        assert_eq!(right.grad()?.expect("right gradient").data, vec![2.0, 3.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn matmul_supports_vector_contracts() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![1.0, 2.0, 3.0], &[3])?;
        let w = ctx.parameter(vec![4.0, 5.0, 6.0], &[3])?;
        let dot = ctx.matmul_variable(&x, &w)?;
        assert_eq!(dot.tensor().shape()?, Vec::<usize>::new());
        assert_eq!(dot.tensor().item()?, 32.0);
        dot.backward()?;
        assert_eq!(x.grad()?.expect("x gradient").data, vec![4.0, 5.0, 6.0]);
        assert_eq!(w.grad()?.expect("w gradient").data, vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn batched_matmul_broadcasts_and_reduces_batch_gradient() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let lhs = ctx.parameter(vec![1.0; 4].into_iter().chain(vec![2.0; 4]).collect(), &[2, 2, 2])?;
        let rhs = ctx.parameter(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let output = ctx.matmul_variable(&lhs, &rhs)?;
        assert_eq!(ctx.graph_stats()?.dynamic_backward_nodes, 1);
        assert_eq!(output.tensor().shape()?, vec![2, 2, 2]);
        ctx.sum_variable(&output)?.backward()?;
        assert_eq!(lhs.grad()?.expect("lhs gradient").data,
            vec![3.0, 7.0, 3.0, 7.0, 3.0, 7.0, 3.0, 7.0]);
        assert_eq!(rhs.grad()?.expect("rhs gradient").data, vec![6.0, 6.0, 6.0, 6.0]);
        Ok(())
    }

    #[test]
    fn matrix_vector_matmul_has_expected_gradient() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let matrix = ctx.parameter(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let vector = ctx.parameter(vec![2.0, 3.0, 4.0], &[3])?;
        let output = ctx.matmul_variable(&matrix, &vector)?;
        assert_eq!(output.tensor().to_vec()?, vec![20.0, 47.0]);
        ctx.sum_variable(&output)?.backward()?;
        assert_eq!(matrix.grad()?.expect("matrix gradient").data, vec![2.0, 3.0, 4.0, 2.0, 3.0, 4.0]);
        assert_eq!(vector.grad()?.expect("vector gradient").data, vec![5.0, 7.0, 9.0]);
        Ok(())
    }

    #[test]
    fn non_leaf_gradients_require_explicit_retention() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![2.0], &[])?;
        let hidden = ctx.square_variable(&x)?;
        let output = ctx.square_variable(&hidden)?;
        output.backward()?;
        assert!(hidden.grad()?.is_none());
        assert_eq!(x.grad()?.expect("leaf gradient").data, vec![32.0]);

        let hidden = ctx.square_variable(&x)?;
        hidden.retain_grad()?;
        let output = ctx.square_variable(&hidden)?;
        output.backward()?;
        assert_eq!(hidden.grad()?.expect("retained gradient").data, vec![8.0]);
        Ok(())
    }

    #[test]
    fn detach_creates_an_untracked_leaf_in_the_same_context() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![3.0], &[])?;
        let connected = ctx.square_variable(&x)?;
        let detached = connected.detach()?;
        assert_eq!(detached.tensor().context_id(), x.tensor().context_id());
        assert_ne!(detached.tensor().node_id(), connected.tensor().node_id());
        assert!(!detached.requires_grad());
        assert_eq!(detached.tensor().item()?, 9.0);
        assert_eq!(ctx.graph_stats()?.graph_nodes, 1);
        Ok(())
    }

    #[test]
    fn graph_nodes_own_dynamic_backward_ops_and_separate_saved_tensors() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![1.0], &[])?;
        let sum = ctx.add_variable(&x, &x)?;
        let product = ctx.mul_variable(&sum, &x)?;
        let _exponential = ctx.exp_variable(&product)?;
        let stats = ctx.graph_stats()?;
        assert_eq!(stats.graph_nodes, 3);
        assert_eq!(stats.dynamic_backward_nodes, 3);
        assert_eq!(stats.saved_tensor_references, 1);
        Ok(())
    }

    #[test]
    fn abs_gradient_matches_finite_difference_away_from_zero() -> MlResult<()> {
        finite_difference_check(-0.7, f32::abs, |ctx, x| ctx.abs_variable(x))
    }

    #[test]
    fn softmax_axis_vjp_matches_finite_difference() -> MlResult<()> {
        let input_data = vec![0.2, -0.4, 1.1, 2.0, 0.3, -0.5];
        let cotangent_data = vec![1.0, -2.0, 0.5, 0.3, 0.7, -1.0];
        let ctx = ExecutionContext::new();
        let input = ctx.parameter(input_data.clone(), &[2, 3])?;
        let output = ctx.softmax_variable(&input, 1)?;
        let probabilities = output.tensor().to_vec()?;
        assert!((probabilities[..3].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((probabilities[3..].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let cotangent = ctx.tensor(cotangent_data.clone(), &[2, 3])?;
        output.backward_with_grad(&cotangent)?;
        let analytic = input.grad()?.expect("softmax input gradient").data;

        let objective = |values: &[f32]| -> f32 {
            values.chunks_exact(3).zip(cotangent_data.chunks_exact(3)).map(|(row, weights)| {
                let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<_> = row.iter().map(|x| (x - maximum).exp()).collect();
                let normalizer: f32 = exps.iter().sum();
                exps.iter().zip(weights).map(|(value, weight)| value / normalizer * weight).sum::<f32>()
            }).sum()
        };
        let epsilon = 1e-3;
        for index in 0..input_data.len() {
            let mut plus = input_data.clone();
            let mut minus = input_data.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numeric = (objective(&plus) - objective(&minus)) / (2.0 * epsilon);
            let error = (analytic[index] - numeric).abs();
            assert!(error <= 1e-3, "index={index}, analytic={}, numeric={numeric}", analytic[index]);
        }
        Ok(())
    }

    #[test]
    fn conv2d_forward_and_all_gradients_match_finite_difference() -> MlResult<()> {
        let input_data = vec![0.2, -0.4, 0.7, 1.1, -0.3, 0.5, 0.9, -0.8, 0.6];
        let weight_data = vec![0.4, -0.2, 0.3, 0.8];
        let bias_data = vec![0.15];
        let ctx = ExecutionContext::new();
        let input = ctx.parameter(input_data.clone(), &[1, 1, 3, 3])?;
        let weight = ctx.parameter(weight_data.clone(), &[1, 1, 2, 2])?;
        let bias = ctx.parameter(bias_data.clone(), &[1])?;
        let output = ctx.conv2d_variable(&input, &weight, &bias, (1, 1), (0, 0))?;
        assert_eq!(output.tensor().shape()?, vec![1, 1, 2, 2]);
        ctx.sum_variable(&output)?.backward()?;
        let analytic_input = input.grad()?.expect("input gradient").data;
        let analytic_weight = weight.grad()?.expect("weight gradient").data;
        let analytic_bias = bias.grad()?.expect("bias gradient").data;
        let objective = |x: &[f32], w: &[f32], b: &[f32]| -> MlResult<f32> {
            Ok(conv2d_forward_data(
                &GlobalTensor::from_vec(x.to_vec(), &[1, 1, 3, 3])?,
                &GlobalTensor::from_vec(w.to_vec(), &[1, 1, 2, 2])?,
                &GlobalTensor::from_vec(b.to_vec(), &[1])?,
                (1, 1), (0, 0),
            )?.data.iter().sum())
        };
        let epsilon = 1e-3;
        for index in 0..input_data.len() {
            let (mut plus, mut minus) = (input_data.clone(), input_data.clone());
            plus[index] += epsilon; minus[index] -= epsilon;
            let numeric = (objective(&plus, &weight_data, &bias_data)? - objective(&minus, &weight_data, &bias_data)?) / (2.0 * epsilon);
            assert!((analytic_input[index] - numeric).abs() <= 1e-3);
        }
        for index in 0..weight_data.len() {
            let (mut plus, mut minus) = (weight_data.clone(), weight_data.clone());
            plus[index] += epsilon; minus[index] -= epsilon;
            let numeric = (objective(&input_data, &plus, &bias_data)? - objective(&input_data, &minus, &bias_data)?) / (2.0 * epsilon);
            assert!((analytic_weight[index] - numeric).abs() <= 1e-3);
        }
        let numeric_bias = (objective(&input_data, &weight_data, &[bias_data[0] + epsilon])?
            - objective(&input_data, &weight_data, &[bias_data[0] - epsilon])?) / (2.0 * epsilon);
        assert!((analytic_bias[0] - numeric_bias).abs() <= 1e-3);
        Ok(())
    }

    #[test]
    fn conv2d_supports_batch_channels_stride_and_padding() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input = ctx.tensor(vec![1.0; 2 * 2 * 4 * 4], &[2, 2, 4, 4])?;
        let weight = ctx.tensor(vec![1.0; 3 * 2 * 3 * 3], &[3, 2, 3, 3])?;
        let bias = ctx.tensor(vec![1.0, 2.0, 3.0], &[3])?;
        let output = ctx.conv2d(&input, &weight, &bias, (2, 2), (1, 1))?;
        assert_eq!(output.shape()?, vec![2, 3, 2, 2]);
        let data = output.to_vec()?;
        assert_eq!(&data[..4], &[9.0, 13.0, 13.0, 19.0]);
        Ok(())
    }

    #[test]
    fn max_pool2d_uses_saved_mask_and_releases_it_with_the_graph() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input_data = vec![
            1.0, 4.0, 2.0,
            3.0, 8.0, 5.0,
            0.0, 6.0, 7.0,
        ];
        let input = ctx.parameter(input_data.clone(), &[1, 1, 3, 3])?;
        let output = ctx.max_pool2d_variable(&input, (2, 2), (1, 1))?;
        assert_eq!(output.tensor().to_vec()?, vec![8.0, 8.0, 8.0, 8.0]);
        let stats = ctx.graph_stats()?;
        assert_eq!(stats.saved_tensor_references, 1);
        assert_eq!(stats.tensors, 3);
        ctx.sum_variable(&output)?.backward()?;
        assert_eq!(input.grad()?.expect("max pool input gradient").data,
            vec![0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(ctx.graph_stats()?.saved_tensor_references, 0);
        assert_eq!(ctx.graph_stats()?.tensors, 3);

        let objective = |values: &[f32]| -> MlResult<f32> {
            Ok(max_pool2d_forward_data(
                &GlobalTensor::from_vec(values.to_vec(), &[1, 1, 3, 3])?,
                (2, 2),
                (1, 1),
            )?.0.data.iter().sum())
        };
        let epsilon = 1e-3;
        for index in 0..input_data.len() {
            let (mut plus, mut minus) = (input_data.clone(), input_data.clone());
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numeric = (objective(&plus)? - objective(&minus)?) / (2.0 * epsilon);
            let analytic = if index == 4 { 4.0 } else { 0.0 };
            assert!((analytic - numeric).abs() <= 1e-3);
        }
        Ok(())
    }

    #[test]
    fn avg_pool2d_accumulates_overlapping_window_gradients() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input = ctx.parameter((1..=9).map(|value| value as f32).collect(), &[1, 1, 3, 3])?;
        let output = ctx.avg_pool2d_variable(&input, (2, 2), (1, 1))?;
        assert_eq!(output.tensor().to_vec()?, vec![3.0, 4.0, 6.0, 7.0]);
        ctx.sum_variable(&output)?.backward()?;
        assert_eq!(input.grad()?.expect("average pool input gradient").data,
            vec![0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25]);
        Ok(())
    }

    #[test]
    fn clear_graph_releases_owned_max_pool_mask() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input = ctx.parameter(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
        let output = ctx.max_pool2d_variable(&input, (2, 2), (2, 2))?;
        assert_eq!(ctx.graph_stats()?.tensors, 3);
        ctx.clear_graph()?;
        let stats = ctx.graph_stats()?;
        assert_eq!(stats.tensors, 2);
        assert_eq!(stats.graph_nodes, 0);
        assert_eq!(stats.saved_tensor_references, 0);
        assert_eq!(output.tensor().item()?, 4.0);
        Ok(())
    }

    #[test]
    fn nearest_upsample2d_supports_asymmetric_scale_and_vjp() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = ctx.parameter(input_data.clone(), &[1, 1, 2, 2])?;
        let output = ctx.nearest_upsample2d_variable(&input, (2, 3))?;
        assert_eq!(output.tensor().shape()?, vec![1, 1, 4, 6]);
        assert_eq!(output.tensor().to_vec()?, vec![
            1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
            1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
            3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
            3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
        ]);
        let cotangent = ctx.tensor((1..=24).map(|value| value as f32).collect(), &[1, 1, 4, 6])?;
        output.backward_with_grad(&cotangent)?;
        let analytic = input.grad()?.expect("upsample input gradient").data;
        assert_eq!(analytic, vec![30.0, 48.0, 102.0, 120.0]);
        let cotangent_data: Vec<_> = (1..=24).map(|value| value as f32).collect();
        let objective = |values: &[f32]| -> MlResult<f32> {
            let output = nearest_upsample2d_forward_data(
                &GlobalTensor::from_vec(values.to_vec(), &[1, 1, 2, 2])?,
                (2, 3),
            )?;
            Ok(output.data.iter().zip(&cotangent_data).map(|(x, g)| x * g).sum())
        };
        let epsilon = 1e-3;
        for index in 0..input_data.len() {
            let (mut plus, mut minus) = (input_data.clone(), input_data.clone());
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numeric = (objective(&plus)? - objective(&minus)?) / (2.0 * epsilon);
            let absolute_error = (analytic[index] - numeric).abs();
            let relative_error = absolute_error / analytic[index].abs().max(numeric.abs()).max(1e-12);
            assert!(absolute_error <= 1e-3 || relative_error <= 1e-3);
        }
        Ok(())
    }

    #[test]
    fn nearest_upsample2d_rejects_zero_scale() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input = ctx.tensor(vec![1.0], &[1, 1, 1, 1])?;
        assert!(ctx.nearest_upsample2d(&input, (0, 2)).is_err());
        Ok(())
    }
}
