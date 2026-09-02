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
    pub no_grad_depth: usize,
}

#[derive(Debug)]
struct ContextState {
    tensors: HashMap<NodeId, GlobalTensor<f32>>,
    next_node: u64,
    graph: HashMap<NodeId, GraphNode>,
    tracked: HashSet<NodeId>,
    gradients: HashMap<NodeId, GlobalTensor<f32>>,
    consumed: HashSet<NodeId>,
    no_grad_depth: usize,
}

#[derive(Debug, Clone)]
struct GraphNode {
    inputs: Vec<NodeId>,
    backward: BuiltinBackward,
}

#[derive(Debug, Clone, Copy)]
enum BuiltinBackward {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Square,
    Exp,
    Log,
    Sum,
    Matmul2d,
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
            self.state
                .try_borrow_mut()
                .map_err(|_| ContextError::BorrowConflict)?
                .tracked
                .insert(tensor.node_id());
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
        if left.shape != right.shape {
            return Err(TensorError::InvalidShape {
                expected: left.shape,
                got: right.shape,
            }
            .into());
        }
        if left.data.len() != right.data.len() {
            return Err(TensorError::InvalidOperation {
                op,
                reason: "data length mismatch".into(),
            }
            .into());
        }
        let shape = left.shape.clone();
        let output = self.tensor(
            left.data
                .into_iter()
                .zip(right.data)
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
                    backward,
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
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        if state.no_grad_depth == 0 && inputs.iter().any(|id| state.tracked.contains(id)) {
            state.tracked.insert(output);
            state.graph.insert(output, GraphNode { inputs, backward });
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

    pub fn matmul(&self, lhs: &ContextTensor, rhs: &ContextTensor) -> MlResult<ContextTensor> {
        self.validate(lhs)?;
        self.validate(rhs)?;
        let (left, right) = (lhs.snapshot()?, rhs.snapshot()?);
        if left.shape.len() != 2 || right.shape.len() != 2 || left.shape[1] != right.shape[0] {
            return Err(TensorError::InvalidOperation {
                op: "matmul",
                reason: format!(
                    "expected [m,k] x [k,n], got {:?} x {:?}",
                    left.shape, right.shape
                ),
            }
            .into());
        }
        let (m, k, n) = (left.shape[0], left.shape[1], right.shape[1]);
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    data[i * n + j] += left.data[i * k + p] * right.data[p * n + j];
                }
            }
        }
        let output = self.tensor(data, &[m, n])?;
        self.record(
            output.node_id(),
            vec![lhs.node_id(), rhs.node_id()],
            BuiltinBackward::Matmul2d,
        )?;
        Ok(output)
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

    pub fn sum_variable(&self, input: &ContextVariable) -> MlResult<ContextVariable> {
        self.variable_from(self.sum(input.tensor())?)
    }

    pub fn matmul_variable(
        &self,
        lhs: &ContextVariable,
        rhs: &ContextVariable,
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.matmul(lhs.tensor(), rhs.tensor())?)
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
        state.graph.clear();
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
                .cloned()
                .ok_or(AutogradError::NodeNotFound(id))?;
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
            let grads = match node.backward {
                BuiltinBackward::Add => vec![grad.clone(), grad],
                BuiltinBackward::Sub => vec![grad.clone(), tensor_map(&grad, |g| -g)?],
                BuiltinBackward::Mul => {
                    require_arity(&values, 2)?;
                    let left = tensor_zip(&grad, &values[1], |g, r| g * r)?;
                    let right = tensor_zip(&grad, &values[0], |g, l| g * l)?;
                    vec![left, right]
                }
                BuiltinBackward::Div => {
                    require_arity(&values, 2)?;
                    let left = tensor_zip(&grad, &values[1], |g, r| g / r)?;
                    let right_data = grad
                        .data
                        .iter()
                        .zip(&values[0].data)
                        .zip(&values[1].data)
                        .map(|((g, l), r)| -g * l / (r * r))
                        .collect();
                    let right = GlobalTensor::from_vec(right_data, &values[1].shape)?;
                    vec![left, right]
                }
                BuiltinBackward::Neg => vec![tensor_map(&grad, |g| -g)?],
                BuiltinBackward::Square => {
                    require_arity(&values, 1)?;
                    vec![tensor_zip(&grad, &values[0], |g, x| 2.0 * g * x)?]
                }
                BuiltinBackward::Exp => {
                    let output = state
                        .tensors
                        .get(&id)
                        .ok_or(ContextError::UnknownTensor(id))?;
                    vec![tensor_zip(&grad, output, |g, y| g * y)?]
                }
                BuiltinBackward::Log => {
                    require_arity(&values, 1)?;
                    vec![tensor_zip(&grad, &values[0], |g, x| g / x)?]
                }
                BuiltinBackward::Sum => {
                    require_arity(&values, 1)?;
                    let scalar =
                        grad.data
                            .first()
                            .copied()
                            .ok_or(TensorError::InvalidOperation {
                                op: "sum backward",
                                reason: "empty output gradient".into(),
                            })?;
                    vec![GlobalTensor::from_vec(
                        vec![scalar; values[0].data.len()],
                        &values[0].shape,
                    )?]
                }
                BuiltinBackward::Matmul2d => {
                    require_arity(&values, 2)?;
                    let (lhs, rhs) = (&values[0], &values[1]);
                    let (m, k, n) = (lhs.shape[0], lhs.shape[1], rhs.shape[1]);
                    let mut dl = vec![0.0; m * k];
                    let mut dr = vec![0.0; k * n];
                    for i in 0..m {
                        for p in 0..k {
                            for j in 0..n {
                                dl[i * k + p] += grad.data[i * n + j] * rhs.data[p * n + j];
                                dr[p * n + j] += lhs.data[i * k + p] * grad.data[i * n + j];
                            }
                        }
                    }
                    vec![
                        GlobalTensor::from_vec(dl, &lhs.shape)?,
                        GlobalTensor::from_vec(dr, &rhs.shape)?,
                    ]
                }
            };
            if grads.len() != node.inputs.len() {
                return Err(AutogradError::BackwardArityMismatch {
                    expected: node.inputs.len(),
                    got: grads.len(),
                }
                .into());
            }
            for (input, incoming) in node.inputs.into_iter().zip(grads) {
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
                state.graph.remove(&id);
            }
            state.consumed.insert(output.tensor.node_id());
        }
        Ok(())
    }
}

fn require_arity(values: &[GlobalTensor<f32>], expected: usize) -> MlResult<()> {
    if values.len() != expected {
        return Err(AutogradError::BackwardArityMismatch {
            expected,
            got: values.len(),
        }
        .into());
    }
    Ok(())
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
    pub fn detach(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
            requires_grad: false,
        }
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
}
