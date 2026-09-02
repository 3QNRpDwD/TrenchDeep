//! Explicit, single-threaded execution context.
//!
//! This module is the migration target for the legacy thread-local tensor and
//! graph stores.  It deliberately exposes only fallible, borrow-safe access.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{ContextError, MlResult, TensorError};

use super::{GlobalTensor, NodeId, TensorBase};

static NEXT_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextId(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequiresGrad { No, Yes }

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
    graph_nodes: usize,
    no_grad_depth: usize,
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
    pub fn data(&self) -> &'a [f32] { self.data }
    pub fn shape(&self) -> &'a [usize] { self.shape }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
}

impl Default for ExecutionContext {
    fn default() -> Self { Self::new() }
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self {
            id: ContextId(NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed)),
            state: Rc::new(RefCell::new(ContextState {
                tensors: HashMap::new(),
                next_node: 0,
                graph_nodes: 0,
                no_grad_depth: 0,
            })),
            _not_sync: Rc::new(Cell::new(())),
        }
    }

    pub fn id(&self) -> ContextId { self.id }

    pub fn tensor(&self, data: Vec<f32>, shape: &[usize]) -> MlResult<ContextTensor> {
        let tensor = GlobalTensor::from_vec(data, shape)?;
        self.insert(tensor)
    }

    pub fn scalar(&self, value: f32) -> MlResult<ContextTensor> {
        self.tensor(vec![value], &[])
    }

    pub fn variable(
        &self, data: Vec<f32>, shape: &[usize], requires_grad: RequiresGrad,
    ) -> MlResult<ContextVariable> {
        Ok(ContextVariable {
            tensor: self.tensor(data, shape)?,
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
        let mut state = self.state.try_borrow_mut().map_err(|_| ContextError::BorrowConflict)?;
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
        if tensor.context_id() != self.id { return Err(ContextError::Mismatch.into()); }
        if !self.state.try_borrow().map_err(|_| ContextError::BorrowConflict)?
            .tensors.contains_key(&tensor.node_id()) {
            return Err(ContextError::UnknownTensor(tensor.node_id()).into());
        }
        Ok(())
    }

    pub fn with_tensor<R>(
        &self, tensor: &ContextTensor, f: impl FnOnce(TensorView<'_>) -> R,
    ) -> MlResult<R> {
        self.validate(tensor)?;
        let state = self.state.try_borrow().map_err(|_| ContextError::BorrowConflict)?;
        let value = state.tensors.get(&tensor.node_id())
            .ok_or(ContextError::UnknownTensor(tensor.node_id()))?;
        Ok(f(TensorView { data: &value.data, shape: &value.shape }))
    }

    fn binary(
        &self, lhs: &ContextTensor, rhs: &ContextTensor,
        op: &'static str, f: impl Fn(f32, f32) -> f32,
    ) -> MlResult<ContextTensor> {
        self.validate(lhs)?;
        self.validate(rhs)?;
        let (left, right) = (lhs.snapshot()?, rhs.snapshot()?);
        if left.shape != right.shape {
            return Err(TensorError::InvalidShape { expected: left.shape, got: right.shape }.into());
        }
        if left.data.len() != right.data.len() {
            return Err(TensorError::InvalidOperation { op, reason: "data length mismatch".into() }.into());
        }
        let shape = left.shape.clone();
        self.tensor(left.data.into_iter().zip(right.data).map(|(a, b)| f(a, b)).collect(), &shape)
    }

    pub fn add(&self, lhs: &ContextTensor, rhs: &ContextTensor) -> MlResult<ContextTensor> {
        self.binary(lhs, rhs, "add", |a, b| a + b)
    }

    pub fn mul(&self, lhs: &ContextTensor, rhs: &ContextTensor) -> MlResult<ContextTensor> {
        self.binary(lhs, rhs, "mul", |a, b| a * b)
    }

    pub fn clear_graph(&self) -> MlResult<()> {
        self.state.try_borrow_mut().map_err(|_| ContextError::BorrowConflict)?.graph_nodes = 0;
        Ok(())
    }

    pub fn clear_all(&self) -> MlResult<()> {
        let mut state = self.state.try_borrow_mut().map_err(|_| ContextError::BorrowConflict)?;
        state.tensors.clear();
        state.graph_nodes = 0;
        Ok(())
    }

    pub fn no_grad<T>(&self, f: impl FnOnce() -> MlResult<T>) -> MlResult<T> {
        {
            let mut state = self.state.try_borrow_mut().map_err(|_| ContextError::BorrowConflict)?;
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
        let state = self.state.try_borrow().map_err(|_| ContextError::BorrowConflict)?;
        Ok(GraphStats {
            tensors: state.tensors.len(),
            graph_nodes: state.graph_nodes,
            no_grad_depth: state.no_grad_depth,
        })
    }
}

impl ContextTensor {
    pub fn context_id(&self) -> ContextId { self.0.context_id }
    pub fn node_id(&self) -> NodeId { self.0.node_id }

    fn state(&self) -> MlResult<Rc<RefCell<ContextState>>> {
        self.0.context.upgrade().ok_or_else(|| ContextError::Dropped.into())
    }

    fn snapshot(&self) -> MlResult<GlobalTensor<f32>> {
        let state = self.state()?;
        let state = state.try_borrow().map_err(|_| ContextError::BorrowConflict)?;
        state.tensors.get(&self.node_id()).cloned()
            .ok_or_else(|| ContextError::UnknownTensor(self.node_id()).into())
    }

    pub fn to_vec(&self) -> MlResult<Vec<f32>> { Ok(self.snapshot()?.data) }
    pub fn shape(&self) -> MlResult<Vec<usize>> { Ok(self.snapshot()?.shape) }

    pub fn item(&self) -> MlResult<f32> {
        let value = self.snapshot()?;
        if value.data.len() != 1 { return Err(TensorError::NotScalar { shape: value.shape }.into()); }
        Ok(value.data[0])
    }

    pub fn get(&self, indices: &[usize]) -> MlResult<Option<f32>> {
        let value = self.snapshot()?;
        if indices.len() != value.shape.len() { return Ok(None); }
        let mut flat = 0usize;
        for (index, dim) in indices.iter().zip(&value.shape) {
            if index >= dim { return Ok(None); }
            flat = flat * dim + index;
        }
        Ok(value.data.get(flat).copied())
    }
}

impl ContextVariable {
    pub fn tensor(&self) -> &ContextTensor { &self.tensor }
    pub fn requires_grad(&self) -> bool { self.requires_grad }
    pub fn detach(&self) -> Self { Self { tensor: self.tensor.clone(), requires_grad: false } }
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
        assert!(matches!(a.add(&x, &y), Err(crate::MlError::ContextError(ContextError::Mismatch))));
        Ok(())
    }

    #[test]
    fn dropped_context_is_reported() -> MlResult<()> {
        let tensor = {
            let ctx = ExecutionContext::new();
            ctx.tensor(vec![1.0], &[1])?
        };
        assert!(matches!(tensor.to_vec(), Err(crate::MlError::ContextError(ContextError::Dropped))));
        Ok(())
    }

    #[test]
    fn no_grad_depth_is_restored_on_error() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let _: Result<(), _> = ctx.no_grad(|| Err(crate::MlError::StringError("stop".into())));
        assert_eq!(ctx.graph_stats()?.no_grad_depth, 0);
        Ok(())
    }
}
