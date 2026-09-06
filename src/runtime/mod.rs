//! Explicit runtime facade. Provider implementations are replaceable per context.
use crate::contracts::*;
use crate::{AutogradError, ContextError, MlError, MlResult, TensorError};
use std::{
    cell::{Cell, RefCell},
    collections::{HashMap, HashSet},
    rc::{Rc, Weak},
    sync::atomic::{AtomicU64, Ordering},
};

#[cfg(feature = "builtinKernels")]
use crate::backend::CpuBackend;
#[cfg(feature = "builtinStorage")]
use crate::backend::CpuTensorStore;
#[cfg(feature = "enableBackward")]
mod builtin_autograd;
#[cfg(feature = "enableBackward")]
pub use builtin_autograd::ReverseMode;

static CONTEXT_IDS: AtomicU64 = AtomicU64::new(1);
static PARAMETER_IDS: AtomicU64 = AtomicU64::new(1);

pub(crate) fn missing(
    module: &'static str,
    capability: &'static str,
    operation: &'static str,
) -> MlError {
    MlError::DependencyUnavailable {
        module,
        capability,
        operation,
    }
}

#[derive(Debug, Default)]
pub struct ExecutionContextBuilder {
    initialization_seed: Option<u64>,
    model_seed: Option<u64>,
    storage: Option<Box<dyn TensorStore>>,
    autograd: Option<Box<dyn AutogradEngine>>,
    operations: Option<Box<dyn OperationProvider>>,
}
impl ExecutionContextBuilder {
    /// An empty composition; nothing is silently substituted for missing providers.
    pub fn empty() -> Self {
        Self::default()
    }
    pub fn storage(mut self, provider: impl TensorStore + 'static) -> Self {
        self.storage = Some(Box::new(provider));
        self
    }
    pub fn autograd(mut self, provider: impl AutogradEngine + 'static) -> Self {
        self.autograd = Some(Box::new(provider));
        self
    }
    pub fn operations(mut self, provider: impl OperationProvider + 'static) -> Self {
        self.operations = Some(Box::new(provider));
        self
    }
    pub fn without_storage(mut self) -> Self {
        self.storage = None;
        self
    }
    pub fn without_autograd(mut self) -> Self {
        self.autograd = None;
        self
    }
    pub fn without_operations(mut self) -> Self {
        self.operations = None;
        self
    }
    pub fn initialization_seed(mut self, seed: u64) -> Self {
        self.initialization_seed = Some(seed);
        self
    }
    pub fn model_seed(mut self, seed: u64) -> Self {
        self.model_seed = Some(seed);
        self
    }
    pub fn build(self) -> ExecutionContext {
        use rand::SeedableRng;
        ExecutionContext {
            id: ContextId(CONTEXT_IDS.fetch_add(1, Ordering::Relaxed)),
            inner: Rc::new(Runtime {
                state: RefCell::new(State {
                    storage: self.storage,
                    autograd: self.autograd,
                    handles: HashMap::new(),
                    gradients: HashMap::new(),
                    parameters: HashSet::new(),
                    tracked: HashSet::new(),
                    leaves: HashSet::new(),
                    retained: HashSet::new(),
                    consumed: HashSet::new(),
                    next: 0,
                }),
                initialization_rng: RefCell::new(rand::rngs::StdRng::seed_from_u64(
                    self.initialization_seed.unwrap_or_else(rand::random),
                )),
                model_rng: RefCell::new(rand::rngs::StdRng::seed_from_u64(
                    self.model_seed.unwrap_or_else(rand::random),
                )),
                operations: self.operations,
                gc_pending: Cell::new(false),
                no_grad: Cell::new(0),
                training_active: Cell::new(false),
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionContext {
    id: ContextId,
    inner: Rc<Runtime>,
}
#[derive(Debug)]
struct Runtime {
    initialization_rng: RefCell<rand::rngs::StdRng>,
    model_rng: RefCell<rand::rngs::StdRng>,
    state: RefCell<State>,
    operations: Option<Box<dyn OperationProvider>>,
    gc_pending: Cell<bool>,
    no_grad: Cell<usize>,
    training_active: Cell<bool>,
}
#[derive(Debug)]
struct State {
    storage: Option<Box<dyn TensorStore>>,
    autograd: Option<Box<dyn AutogradEngine>>,
    handles: HashMap<TensorId, Entry>,
    gradients: HashMap<TensorId, TensorBuffer>,
    parameters: HashSet<TensorId>,
    tracked: HashSet<TensorId>,
    leaves: HashSet<TensorId>,
    retained: HashSet<TensorId>,
    consumed: HashSet<TensorId>,
    next: u64,
}
#[derive(Debug)]
struct Entry {
    external: Weak<Handle>,
    pins: usize,
}
#[derive(Debug)]
struct Handle {
    id: TensorId,
    context: ContextId,
    inner: Weak<Runtime>,
}
#[derive(Debug, Clone)]
pub struct Tensor(Rc<Handle>);
#[derive(Debug, Clone)]
pub struct Variable {
    tensor: Tensor,
}
#[derive(Debug, Clone)]
pub struct Parameter {
    id: ParameterId,
    value: Variable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequiresGrad {
    No,
    Yes,
}
#[derive(Debug, Default, Clone, Copy)]
pub struct BackwardOptions<'a> {
    pub gradient: Option<&'a Tensor>,
    pub retain_graph: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphStats {
    pub tensors: usize,
    pub graph_nodes: usize,
    pub dynamic_backward_nodes: usize,
    pub saved_tensor_references: usize,
    pub no_grad_depth: usize,
}
#[derive(Debug, Clone)]
pub struct TopKResult {
    pub values: Tensor,
    pub indices: Tensor,
}
#[derive(Debug, Clone)]
pub struct MaxResult {
    pub values: Tensor,
    pub indices: Tensor,
}

impl State {
    fn store(&self) -> MlResult<&dyn TensorStore> {
        self.storage
            .as_deref()
            .ok_or_else(|| missing("storage", "tensor buffers", "tensor access"))
    }
    fn store_mut(&mut self) -> MlResult<&mut (dyn TensorStore + 'static)> {
        self.storage
            .as_deref_mut()
            .ok_or_else(|| missing("storage", "tensor buffers", "tensor access"))
    }
    fn engine(&self) -> MlResult<&dyn AutogradEngine> {
        self.autograd
            .as_deref()
            .ok_or_else(|| missing("autograd", "reverse mode", "backward"))
    }
    fn valid(&self, id: TensorId) -> MlResult<()> {
        if self.handles.contains_key(&id) {
            Ok(())
        } else {
            Err(ContextError::UnknownTensor(id).into())
        }
    }
    fn snapshot(&self, id: TensorId) -> MlResult<TensorBuffer> {
        self.valid(id)?;
        snapshot(self.store()?, id)
    }
    fn collect(&mut self) -> MlResult<()> {
        let dead = self
            .handles
            .iter()
            .filter_map(|(&id, e)| (e.pins == 0 && e.external.upgrade().is_none()).then_some(id))
            .collect::<Vec<_>>();
        for id in dead {
            self.store_mut()?.remove(id)?;
            self.handles.remove(&id);
            self.gradients.remove(&id);
            self.tracked.remove(&id);
            self.parameters.remove(&id);
            self.leaves.remove(&id);
            self.retained.remove(&id);
            self.consumed.remove(&id);
        }
        Ok(())
    }
    fn remove_graph(&mut self, id: TensorId) -> MlResult<()> {
        if let Some(engine) = self.autograd.as_mut() {
            if let Some(record) = engine.remove(id)? {
                for id in std::iter::once(record.output)
                    .chain(record.inputs)
                    .chain(record.saved)
                {
                    if let Some(e) = self.handles.get_mut(&id) {
                        e.pins = e.pins.saturating_sub(1);
                    }
                }
            }
        }
        Ok(())
    }
    fn clear_graph(&mut self) -> MlResult<()> {
        let ids = self
            .autograd
            .as_ref()
            .map(|e| e.nodes())
            .unwrap_or_default();
        for id in ids {
            self.remove_graph(id)?;
            self.consumed.insert(id);
        }
        self.collect()
    }
}
impl Drop for Handle {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.upgrade() {
            inner.gc_pending.set(true);
            if let Ok(mut state) = inner.state.try_borrow_mut() {
                if state.collect().is_ok() {
                    inner.gc_pending.set(false);
                }
            }
        }
    }
}
impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}
impl ExecutionContext {
    pub fn builder() -> ExecutionContextBuilder {
        let builder = ExecutionContextBuilder::empty();
        #[cfg(feature = "builtinStorage")]
        let builder = builder.storage(CpuTensorStore::default());
        #[cfg(feature = "builtinKernels")]
        let builder = builder.operations(CpuBackend::default());
        #[cfg(feature = "enableBackward")]
        let builder = builder.autograd(ReverseMode::default());
        builder
    }
    pub fn new() -> Self {
        Self::builder().build()
    }
    pub fn id(&self) -> ContextId {
        self.id
    }
    fn collect(&self) -> MlResult<()> {
        if self.inner.gc_pending.get() {
            self.inner
                .state
                .try_borrow_mut()
                .map_err(|_| ContextError::BorrowConflict)?
                .collect()?;
            self.inner.gc_pending.set(false);
        }
        Ok(())
    }
    pub fn validate(&self, tensor: &Tensor) -> MlResult<()> {
        self.collect()?;
        if tensor.context_id() != self.id {
            return Err(ContextError::Mismatch.into());
        }
        self.inner
            .state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?
            .valid(tensor.id())
    }
    fn allocate(&self, buffer: Option<TensorBuffer>, alias: Option<TensorId>) -> MlResult<Tensor> {
        self.collect()?;
        let mut state = self
            .inner
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        let id = TensorId(state.next);
        state.next = state
            .next
            .checked_add(1)
            .ok_or_else(|| TensorError::InvalidOperation {
                op: "tensor",
                reason: "tensor IDs exhausted".into(),
            })?;
        if let Some(source) = alias {
            state.valid(source)?;
            state.store_mut()?.alias(id, source)?;
        } else if let Some(buffer) = buffer {
            state.store_mut()?.insert(id, buffer)?;
        }
        let handle = Rc::new(Handle {
            id,
            context: self.id,
            inner: Rc::downgrade(&self.inner),
        });
        state.handles.insert(
            id,
            Entry {
                external: Rc::downgrade(&handle),
                pins: 0,
            },
        );
        Ok(Tensor(handle))
    }
    pub fn tensor(&self, data: Vec<f32>, shape: &[usize]) -> MlResult<Tensor> {
        self.allocate(Some(TensorBuffer::from_vec(data, shape)?), None)
    }
    pub fn scalar(&self, value: f32) -> MlResult<Tensor> {
        self.tensor(vec![value], &[])
    }
    pub fn input(&self, data: Vec<f32>, shape: &[usize]) -> MlResult<Variable> {
        self.variable(data, shape, RequiresGrad::No)
    }
    pub fn variable(
        &self,
        data: Vec<f32>,
        shape: &[usize],
        requires_grad: RequiresGrad,
    ) -> MlResult<Variable> {
        if requires_grad == RequiresGrad::Yes {
            self.inner
                .state
                .try_borrow()
                .map_err(|_| ContextError::BorrowConflict)?
                .engine()?;
        }
        let tensor = self.tensor(data, shape)?;
        if requires_grad == RequiresGrad::Yes {
            let mut state = self
                .inner
                .state
                .try_borrow_mut()
                .map_err(|_| ContextError::BorrowConflict)?;
            state.tracked.insert(tensor.id());
            state.leaves.insert(tensor.id());
        }
        Ok(Variable { tensor })
    }
    pub fn parameter(&self, data: Vec<f32>, shape: &[usize]) -> MlResult<Parameter> {
        // Parameter data is also useful for inference without an autograd provider.
        let tensor = self.tensor(data, shape)?;
        let mut state = self
            .inner
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        state.tracked.insert(tensor.id());
        state.leaves.insert(tensor.id());
        state.parameters.insert(tensor.id());
        Ok(Parameter {
            id: ParameterId(PARAMETER_IDS.fetch_add(1, Ordering::Relaxed)),
            value: Variable { tensor },
        })
    }
    pub fn no_grad<T>(&self, f: impl FnOnce() -> MlResult<T>) -> MlResult<T> {
        self.collect()?;
        self.inner.no_grad.set(self.inner.no_grad.get() + 1);
        struct Guard<'a>(&'a Cell<usize>);
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get().saturating_sub(1));
            }
        }
        let _guard = Guard(&self.inner.no_grad);
        f()
    }
    pub fn with_tensor<T>(
        &self,
        tensor: &Tensor,
        f: impl FnOnce(TensorView<'_>) -> T,
    ) -> MlResult<T> {
        self.validate(tensor)?;
        let state = self
            .inner
            .state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?;
        let mut f = Some(f);
        let mut result = None;
        state.store()?.with_view(tensor.id(), &mut |view| {
            let callback = f.take().ok_or_else(|| TensorError::InvalidOperation {
                op: "view",
                reason: "store invoked view callback more than once".into(),
            })?;
            result = Some(callback(view));
            Ok(())
        })?;
        result.ok_or_else(|| ContextError::UnknownTensor(tensor.id()).into())
    }
    pub fn grad(&self, variable: &Variable) -> MlResult<Option<TensorBuffer>> {
        self.validate(variable.tensor())?;
        variable.grad()
    }
    pub fn clear_grad(&self, variable: &Variable) -> MlResult<()> {
        self.validate(variable.tensor())?;
        self.inner
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?
            .gradients
            .remove(&variable.tensor.id());
        Ok(())
    }
    pub fn scale_grad(&self, variable: &Variable, factor: f32) -> MlResult<()> {
        self.validate(variable.tensor())?;
        if !factor.is_finite() {
            return Err(TensorError::InvalidOperation {
                op: "scale_grad",
                reason: "factor must be finite".into(),
            }
            .into());
        }
        if let Some(grad) = self
            .inner
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?
            .gradients
            .get_mut(&variable.tensor.id())
        {
            for x in &mut grad.data {
                *x *= factor;
            }
        }
        Ok(())
    }
    pub fn replace_parameter(&self, variable: &Variable, buffer: TensorBuffer) -> MlResult<()> {
        self.validate(variable.tensor())?;
        if variable.tensor.shape()? != buffer.shape {
            return Err(TensorError::InvalidShape {
                expected: variable.tensor.shape()?,
                got: buffer.shape,
            }
            .into());
        }
        self.inner
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?
            .store_mut()?
            .replace(variable.tensor.id(), buffer)
    }
    fn update(&self, variable: &Variable, delta: &TensorBuffer, sign: f32) -> MlResult<()> {
        self.validate(variable.tensor())?;
        let mut value = variable.tensor.snapshot()?;
        if value.shape != delta.shape {
            return Err(TensorError::InvalidShape {
                expected: value.shape,
                got: delta.shape.clone(),
            }
            .into());
        }
        for (v, d) in value.data.iter_mut().zip(&delta.data) {
            *v += sign * d;
        }
        self.replace_parameter(variable, value)
    }
    pub fn add_assign(&self, v: &Variable, d: &TensorBuffer) -> MlResult<()> {
        self.update(v, d, 1.0)
    }
    pub fn sub_assign(&self, v: &Variable, d: &TensorBuffer) -> MlResult<()> {
        self.update(v, d, -1.0)
    }
    pub fn clear_graph(&self) -> MlResult<()> {
        self.collect()?;
        self.inner
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?
            .clear_graph()
    }
    pub fn clear_all(&self) -> MlResult<()> {
        self.clear_graph()?;
        let mut state = self
            .inner
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        for id in state.handles.keys().copied().collect::<Vec<_>>() {
            state.store_mut()?.remove(id)?;
            state.handles.remove(&id);
        }
        state.gradients.clear();
        state.tracked.clear();
        state.leaves.clear();
        state.retained.clear();
        state.consumed.clear();
        Ok(())
    }
    pub fn graph_stats(&self) -> MlResult<GraphStats> {
        self.collect()?;
        let state = self
            .inner
            .state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?;
        let nodes = state
            .autograd
            .as_ref()
            .map(|e| e.nodes())
            .unwrap_or_default();
        let saved = state
            .autograd
            .as_ref()
            .map(|e| {
                nodes
                    .iter()
                    .filter_map(|&id| e.get(id))
                    .map(|n| n.saved.len())
                    .sum()
            })
            .unwrap_or(0);
        Ok(GraphStats {
            tensors: state.handles.len(),
            graph_nodes: nodes.len(),
            dynamic_backward_nodes: nodes.len(),
            saved_tensor_references: saved,
            no_grad_depth: self.inner.no_grad.get(),
        })
    }
}

impl Tensor {
    pub fn id(&self) -> TensorId {
        self.0.id
    }
    pub fn context_id(&self) -> ContextId {
        self.0.context
    }
    pub fn execution_context(&self) -> MlResult<ExecutionContext> {
        Ok(ExecutionContext {
            id: self.context_id(),
            inner: self.0.inner.upgrade().ok_or(ContextError::Dropped)?,
        })
    }
    pub fn snapshot(&self) -> MlResult<TensorBuffer> {
        let context = self.execution_context()?;
        context.validate(self)?;
        context
            .inner
            .state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?
            .snapshot(self.id())
    }
    pub fn to_vec(&self) -> MlResult<Vec<f32>> {
        Ok(self.snapshot()?.into_vec())
    }
    pub fn with_view<T>(&self, f: impl FnOnce(TensorView<'_>) -> T) -> MlResult<T> {
        self.execution_context()?.with_tensor(self, f)
    }
    pub fn shape(&self) -> MlResult<Vec<usize>> {
        self.with_view(|v| v.shape().to_vec())
    }
    pub fn numel(&self) -> MlResult<usize> {
        self.with_view(|v| v.len())
    }
    pub fn item(&self) -> MlResult<f32> {
        self.with_view(|v| {
            if v.len() == 1 {
                Ok(v.data()[0])
            } else {
                Err(TensorError::NotScalar {
                    shape: v.shape().to_vec(),
                }
                .into())
            }
        })?
    }
    pub fn get(&self, index: &[usize]) -> MlResult<Option<f32>> {
        self.with_view(|v| {
            if index.len() != v.shape.len() {
                return None;
            }
            let mut flat = 0;
            for (&i, &d) in index.iter().zip(v.shape) {
                if i >= d {
                    return None;
                }
                flat = flat * d + i;
            }
            v.data.get(flat).copied()
        })
    }
    pub fn as_variable(&self) -> MlResult<Variable> {
        self.execution_context()?.validate(self)?;
        Ok(Variable {
            tensor: self.clone(),
        })
    }
    pub fn detach(&self) -> MlResult<Tensor> {
        self.execution_context()?.allocate(None, Some(self.id()))
    }
}
impl Variable {
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }
    pub fn requires_grad(&self) -> MlResult<bool> {
        let ctx = self.tensor.execution_context()?;
        ctx.validate(&self.tensor)?;
        Ok(ctx
            .inner
            .state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?
            .tracked
            .contains(&self.tensor.id()))
    }
    pub fn detach(&self) -> MlResult<Self> {
        self.tensor.detach()?.as_variable()
    }
    pub fn grad(&self) -> MlResult<Option<TensorBuffer>> {
        let ctx = self.tensor.execution_context()?;
        ctx.validate(&self.tensor)?;
        Ok(ctx
            .inner
            .state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?
            .gradients
            .get(&self.tensor.id())
            .cloned())
    }
    pub fn retain_grad(&self) -> MlResult<()> {
        let ctx = self.tensor.execution_context()?;
        ctx.validate(&self.tensor)?;
        if !self.requires_grad()? {
            return Err(AutogradError::NodeNotFound(self.tensor.id()).into());
        }
        ctx.inner
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?
            .retained
            .insert(self.tensor.id());
        Ok(())
    }
    pub fn backward(&self) -> MlResult<()> {
        self.tensor
            .execution_context()?
            .backward(self, BackwardOptions::default())
    }
    pub fn backward_with_grad(&self, gradient: &Tensor) -> MlResult<()> {
        self.tensor.execution_context()?.backward(
            self,
            BackwardOptions {
                gradient: Some(gradient),
                retain_graph: false,
            },
        )
    }
}
impl Parameter {
    pub fn id(&self) -> ParameterId {
        self.id
    }
    pub fn context_id(&self) -> ContextId {
        self.value.tensor.context_id()
    }
    pub fn tensor(&self) -> &Tensor {
        self.value.tensor()
    }
    pub fn variable(&self) -> &Variable {
        &self.value
    }
    pub fn grad(&self) -> MlResult<Option<TensorBuffer>> {
        self.value.grad()
    }
    pub fn clear_grad(&self, context: &ExecutionContext) -> MlResult<()> {
        context.clear_grad(&self.value)
    }
    pub fn retain_grad(&self) -> MlResult<()> {
        self.value.retain_grad()
    }
}
// A parameter behaves as a learning handle without creating a second identity.
impl std::ops::Deref for Parameter {
    type Target = Variable;
    fn deref(&self) -> &Variable {
        &self.value
    }
}

mod backward;
mod dispatch;
mod receiver;
mod training;

#[cfg(all(
    test,
    feature = "builtinStorage",
    feature = "builtinKernels",
    feature = "enableBackward"
))]
mod tests;

#[cfg(feature = "enableVisualization")]
mod visualization;

#[cfg(not(feature = "enableVisualization"))]
mod visualization_disabled;

impl ExecutionContext {
    fn uniform(rng: &RefCell<rand::rngs::StdRng>, count: usize, bound: f32) -> MlResult<Vec<f32>> {
        use rand::Rng;
        if !bound.is_finite() || bound < 0.0 {
            return Err(TensorError::InvalidOperation {
                op: "uniform",
                reason: "bound must be finite and nonnegative".into(),
            }
            .into());
        }
        let mut rng = rng
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        Ok((0..count)
            .map(|_| rng.random::<f32>() * 2.0 * bound - bound)
            .collect())
    }
    pub fn initialization_uniform(&self, count: usize, bound: f32) -> MlResult<Vec<f32>> {
        Self::uniform(&self.inner.initialization_rng, count, bound)
    }
    pub fn model_uniform(&self, count: usize, bound: f32) -> MlResult<Vec<f32>> {
        Self::uniform(&self.inner.model_rng, count, bound)
    }
}
