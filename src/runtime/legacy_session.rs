//! Native forwarding boundary for the opt-in Legacy route.
//! The owning thread must not call the raw legacy API during a session.
#![allow(dead_code)]
use crate::legacy as old;
use crate::{ContextError, MlError, MlResult, TensorBuffer, TensorError};
use old::nn::{Parameter, Variable};
use old::tensor::{
    AutogradFunction, TensorBase,
    operators::{
        Abs, Add, ApproxCos, ApproxSin, AvgPool2d, Concat, Conv2dOp, Cos, Div, Exp, Function,
        GroupNormOp, Log, Matmul, MaxPool2d, Mul, NearestUpsample2d, Neg, Pow, ReshapeOp, Sin,
        Sqrt, Square, Sub, Sum, Transpose,
    },
};
use std::{cell::Cell, collections::HashMap, marker::PhantomData, rc::Rc};

thread_local! { static ACTIVE: Cell<bool> = const { Cell::new(false) }; }

use crate::contracts::{AutogradEngine, GradientRecord, TensorId, TensorStore, TensorView};
use std::{cell::RefCell, collections::HashSet};

#[derive(Debug)]
pub(super) struct Bridge {
    session: NativeSession,
    handles: HashMap<TensorId, NativeHandle>,
    graph: HashSet<TensorId>,
}
impl Bridge {
    pub(super) fn clear_graph(&mut self) {
        self.session.clear_graph();
        self.graph.clear();
    }
    fn handle(&self, id: TensorId) -> MlResult<NativeHandle> {
        self.handles
            .get(&id)
            .copied()
            .ok_or_else(|| ContextError::UnknownTensor(id).into())
    }
}
#[derive(Debug)]
struct Store(Rc<RefCell<Bridge>>);
impl TensorStore for Store {
    fn insert(&mut self, id: TensorId, buffer: TensorBuffer) -> MlResult<()> {
        let mut bridge = self
            .0
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        let handle = bridge.session.tensor(buffer)?;
        bridge.handles.insert(id, handle);
        Ok(())
    }
    fn alias(&mut self, _id: TensorId, _source: TensorId) -> MlResult<()> {
        Err(unsupported("detach alias"))
    }
    fn with_view(
        &self,
        id: TensorId,
        visitor: &mut dyn FnMut(TensorView<'_>) -> MlResult<()>,
    ) -> MlResult<()> {
        let bridge = self
            .0
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?;
        let value = bridge.session.get(bridge.handle(id)?)?.tensor();
        visitor(TensorView::new(value.data(), value.shape())?)
    }
    fn replace(&mut self, id: TensorId, buffer: TensorBuffer) -> MlResult<()> {
        let mut bridge = self
            .0
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        let handle = bridge.handle(id)?;
        bridge.session.replace(handle, buffer)
    }
    fn remove(&mut self, id: TensorId) -> MlResult<()> {
        let mut bridge = self
            .0
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        if let Some(handle) = bridge.handles.remove(&id) {
            if !bridge.handles.values().any(|other| *other == handle) {
                bridge.session.values.remove(&handle.index);
            }
        }
        Ok(())
    }
}
#[derive(Debug)]
struct Graph(Rc<RefCell<Bridge>>);
impl AutogradEngine for Graph {
    fn record(&mut self, _: GradientRecord) -> MlResult<()> {
        Err(unsupported("P1 graph recording"))
    }
    fn get(&self, _: TensorId) -> Option<GradientRecord> {
        None
    }
    fn remove(&mut self, _: TensorId) -> MlResult<Option<GradientRecord>> {
        Err(unsupported("partial graph removal"))
    }
    fn nodes(&self) -> Vec<TensorId> {
        self.0.borrow().graph.iter().copied().collect()
    }
    fn order(&self, _: TensorId) -> MlResult<Vec<TensorId>> {
        Err(unsupported("P1 backward traversal"))
    }
}
fn unsupported(capability: &'static str) -> MlError {
    MlError::UnsupportedCapability {
        module: "legacy execution",
        capability,
        operation: "execute",
    }
}
pub(super) fn build_context(
    mut builder: super::ExecutionContextBuilder,
) -> MlResult<super::ExecutionContext> {
    if builder.providers_configured {
        return Err(unsupported("custom P1 providers on Legacy route"));
    }
    let bridge = Rc::new(RefCell::new(Bridge {
        session: NativeSession::new()?,
        handles: HashMap::new(),
        graph: HashSet::new(),
    }));
    builder.storage = Some(Box::new(Store(bridge.clone())));
    builder.autograd = Some(Box::new(Graph(bridge.clone())));
    builder.operations = None;
    let context = builder.build();
    context.inner.route.set(super::ExecutionRoute::Legacy);
    context.inner.state.borrow_mut().legacy = Some(bridge);
    Ok(context)
}
impl super::ExecutionContext {
    pub(super) fn execute_legacy(
        &self,
        operation: &crate::contracts::Operation,
        inputs: &[&super::Tensor],
        tracked: bool,
    ) -> MlResult<Vec<super::Tensor>> {
        let bridge = self
            .inner
            .state
            .borrow()
            .legacy
            .clone()
            .ok_or_else(|| unsupported("native session"))?;
        let native = {
            let mut bridge = bridge
                .try_borrow_mut()
                .map_err(|_| ContextError::BorrowConflict)?;
            let handles = inputs
                .iter()
                .map(|input| bridge.handle(input.id()))
                .collect::<MlResult<Vec<_>>>()?;
            if tracked {
                bridge.session.execute_many(operation, &handles)?
            } else {
                bridge
                    .session
                    .with_no_grad(|session| session.execute_many(operation, &handles))?
            }
        };
        let mut outputs = Vec::with_capacity(native.len());
        for handle in native {
            let tensor = self.allocate(None, None)?;
            bridge.borrow_mut().handles.insert(tensor.id(), handle);
            if tracked {
                bridge.borrow_mut().graph.insert(tensor.id());
                self.inner.state.borrow_mut().tracked.insert(tensor.id());
            }
            outputs.push(tensor);
        }
        Ok(outputs)
    }
    pub(super) fn backward_legacy(
        &self,
        output: &super::Variable,
        options: super::BackwardOptions<'_>,
        observe: impl FnOnce(&super::State) -> MlResult<()>,
    ) -> MlResult<()> {
        if options.gradient.is_some() {
            return Err(unsupported("explicit backward seed"));
        }
        let mut state = self
            .inner
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        let root = output.tensor().id();
        if state.consumed.contains(&root) {
            return Err(crate::AutogradError::GraphAlreadyFreed(root).into());
        }
        if !state.tracked.contains(&root) {
            return Err(crate::AutogradError::NodeNotFound(root).into());
        }
        let bridge = state
            .legacy
            .clone()
            .ok_or_else(|| unsupported("native session"))?;
        {
            let mut bridge = bridge
                .try_borrow_mut()
                .map_err(|_| ContextError::BorrowConflict)?;
            let handle = bridge.handle(root)?;
            bridge.session.backward(handle)?;
            state.gradients.clear();
            for (&id, &handle) in &bridge.handles {
                if state.tracked.contains(&id) {
                    let value = bridge.session.get(handle)?;
                    if value.is_grad_dirty() {
                        state.gradients.insert(id, bridge.session.gradient(handle)?);
                    }
                }
            }
        }
        observe(&state)?;
        if !options.retain_graph {
            state.consumed.extend(bridge.borrow().graph.iter().copied());
            bridge.borrow_mut().clear_graph();
            state.collect()?;
        }
        let keep = state
            .leaves
            .union(&state.retained)
            .copied()
            .collect::<HashSet<_>>();
        state.gradients.retain(|id, _| keep.contains(id));
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct NativeHandle {
    session: u64,
    index: usize,
}

#[derive(Debug)]
struct NativeSession {
    id: u64,
    values: HashMap<usize, Variable>,
    next: usize,
    forwards: usize,
    backwards: usize,
    no_grad: bool,
    training: bool,
    _single_thread: PhantomData<Rc<()>>,
}

fn translate(error: old::MlError) -> MlError {
    TensorError::InvalidOperation {
        op: "legacy execution",
        reason: error.to_string(),
    }
    .into()
}

impl NativeSession {
    fn new() -> MlResult<Self> {
        if ACTIVE.with(Cell::get) || old::comparison::statistics().map_err(translate)?.1 != 0 {
            return Err(ContextError::ActiveGraphConflict.into());
        }
        ACTIVE.with(|active| active.set(true));
        Ok(Self {
            id: super::CONTEXT_IDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            values: HashMap::new(),
            next: 0,
            forwards: 0,
            backwards: 0,
            no_grad: false,
            training: false,
            _single_thread: PhantomData,
        })
    }
    fn insert(&mut self, value: Variable) -> NativeHandle {
        let handle = NativeHandle {
            session: self.id,
            index: self.next,
        };
        self.next += 1;
        self.values.insert(handle.index, value);
        handle
    }
    fn get(&self, handle: NativeHandle) -> MlResult<&Variable> {
        if handle.session != self.id {
            return Err(ContextError::Mismatch.into());
        }
        self.values.get(&handle.index).ok_or_else(|| {
            TensorError::InvalidOperation {
                op: "legacy handle",
                reason: "released native tensor".into(),
            }
            .into()
        })
    }
    fn tensor(&mut self, buffer: TensorBuffer) -> MlResult<NativeHandle> {
        let tensor = old::tensor::Tensor::from_vec(buffer.data().to_vec(), buffer.shape())
            .map_err(translate)?;
        let value = Variable::new(tensor);
        value.retain_grad();
        Ok(self.insert(value))
    }
    fn execute(
        &mut self,
        operation: &crate::contracts::Operation,
        handles: &[NativeHandle],
    ) -> MlResult<NativeHandle> {
        Ok(self.execute_many(operation, handles)?.remove(0))
    }
    fn execute_many(
        &mut self,
        operation: &crate::contracts::Operation,
        handles: &[NativeHandle],
    ) -> MlResult<Vec<NativeHandle>> {
        use crate::contracts::{LossKind, Operation, Reduction};
        if handles.is_empty() || operation.input_count().is_some_and(|n| n != handles.len()) {
            return Err(TensorError::InvalidOperation {
                op: "legacy execution",
                reason: "invalid input count".into(),
            }
            .into());
        }
        let mut inputs = handles
            .iter()
            .map(|&h| self.get(h))
            .collect::<MlResult<Vec<_>>>()?;
        // Public loss targets are constants, including when pred/target are the
        // same tracked handle. A separate native leaf severs that graph edge.
        let loss_target = if matches!(operation, Operation::Loss { .. }) {
            let target = inputs[1].tensor();
            Some(Variable::new(
                old::tensor::Tensor::from_vec(target.data().to_vec(), target.shape())
                    .map_err(translate)?,
            ))
        } else {
            None
        };
        if let Some(target) = loss_target.as_ref() {
            inputs[1] = target;
        }
        // Legacy operators encode shape/axis attributes as extra tensor inputs.
        // Keep them local: original graph registration owns them when tracked.
        let mut attributes = Vec::new();
        let invalid = || TensorError::InvalidOperation {
            op: operation.name(),
            reason: "invalid legacy operation attributes or shapes".into(),
        };
        let shape = inputs[0].tensor().shape();
        let scalars: Vec<f32> = match operation {
            Operation::ApproxSin { threshold } | Operation::ApproxCos { threshold } => {
                // Both implementations use a fixed-order polynomial. The
                // legacy threshold field is never read by forward/backward.
                if !threshold.is_finite() || *threshold <= 0.0 {
                    return Err(invalid().into());
                }
                vec![]
            }
            Operation::Loss { kind, reduction } => {
                if *reduction != Reduction::Mean {
                    return Err(unsupported("legacy loss reduction other than mean"));
                }
                if shape != inputs[1].tensor().shape() || inputs[0].tensor().data().is_empty() {
                    return Err(invalid().into());
                }
                if matches!(kind, LossKind::CrossEntropy | LossKind::SoftmaxCrossEntropy)
                    && !(1..=2).contains(&shape.len())
                {
                    return Err(unsupported("high-rank legacy cross entropy reduction"));
                }
                if matches!(kind, LossKind::Huber { delta } if *delta != 1.0) {
                    return Err(unsupported("non-default legacy huber delta"));
                }
                vec![]
            }
            Operation::TopK { k, sorted } => {
                if shape.is_empty() || shape.contains(&0) || *k == 0 || *k > shape[shape.len() - 1]
                {
                    return Err(invalid().into());
                }
                vec![*k as f32, u8::from(*sorted) as f32]
            }
            Operation::Matmax { axis, keepdim } => {
                if shape.contains(&0)
                    || axis
                        .is_some_and(|a| a < -(shape.len() as isize) || a >= shape.len() as isize)
                {
                    return Err(invalid().into());
                }
                if axis.is_none() {
                    return Err(unsupported("global matmax index contract"));
                }
                vec![axis.unwrap() as f32, u8::from(*keepdim) as f32]
            }
            Operation::Pow(power) => vec![*power],
            Operation::Concat { axis } | Operation::Softmax { axis } => {
                if *axis >= shape.len() || shape.contains(&0) {
                    return Err(invalid().into());
                }
                vec![*axis as f32]
            }
            Operation::Conv2d { stride, padding } => {
                let weight = inputs[1].tensor().shape();
                if shape.len() != 4
                    || weight.len() != 4
                    || shape.contains(&0)
                    || weight.contains(&0)
                    || shape[1] != weight[1]
                    || inputs[2].tensor().shape() != [weight[0]]
                    || stride.0 == 0
                    || stride.1 == 0
                {
                    return Err(invalid().into());
                }
                for (size, pad, kernel) in [
                    (shape[2], padding.0, weight[2]),
                    (shape[3], padding.1, weight[3]),
                ] {
                    if pad
                        .checked_mul(2)
                        .and_then(|p| size.checked_add(p))
                        .is_none_or(|s| s < kernel)
                    {
                        return Err(invalid().into());
                    }
                }
                vec![
                    stride.0 as f32,
                    stride.1 as f32,
                    padding.0 as f32,
                    padding.1 as f32,
                ]
            }
            Operation::MaxPool2d { kernel, stride } | Operation::AvgPool2d { kernel, stride } => {
                if shape.len() != 4
                    || shape.contains(&0)
                    || kernel.0 == 0
                    || kernel.1 == 0
                    || stride.0 == 0
                    || stride.1 == 0
                    || kernel.0 > shape[2]
                    || kernel.1 > shape[3]
                {
                    return Err(invalid().into());
                }
                vec![
                    kernel.0 as f32,
                    kernel.1 as f32,
                    stride.0 as f32,
                    stride.1 as f32,
                ]
            }
            Operation::NearestUpsample2d { scale } => {
                if shape.len() != 4
                    || shape.contains(&0)
                    || scale.0 == 0
                    || scale.1 == 0
                    || inputs[0]
                        .tensor()
                        .data()
                        .len()
                        .checked_mul(scale.0)
                        .and_then(|n| n.checked_mul(scale.1))
                        .is_none()
                {
                    return Err(invalid().into());
                }
                vec![scale.0 as f32, scale.1 as f32]
            }
            Operation::GroupNorm { groups, epsilon } => {
                if shape.len() != 4
                    || shape.contains(&0)
                    || *groups == 0
                    || shape[1] % groups != 0
                    || !epsilon.is_finite()
                    || *epsilon <= 0.0
                    || inputs[1].tensor().shape() != [shape[1]]
                    || inputs[2].tensor().shape() != [shape[1]]
                {
                    return Err(invalid().into());
                }
                vec![*groups as f32, *epsilon]
            }
            _ => vec![],
        };
        for scalar in scalars {
            attributes.push(Variable::new(
                old::tensor::Tensor::from_vec(vec![scalar], &[]).map_err(translate)?,
            ));
        }
        match operation {
            Operation::Reshape(shape) => {
                let buffer =
                    TensorBuffer::from_vec(vec![0.0; inputs[0].tensor().data().len()], shape)?;
                attributes.push(Variable::new(
                    old::tensor::Tensor::from_vec(buffer.data().to_vec(), shape)
                        .map_err(translate)?,
                ));
            }
            Operation::Transpose(axes) => {
                let rank = inputs[0].tensor().shape().len();
                if rank < 2
                    || axes.len() != rank
                    || axes.iter().copied().collect::<HashSet<_>>().len() != rank
                    || axes.iter().any(|&axis| axis >= rank)
                {
                    return Err(TensorError::InvalidOperation {
                        op: "transpose",
                        reason: "invalid axis permutation".into(),
                    }
                    .into());
                }
                let changed = axes
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &axis)| (axis != i).then_some(i))
                    .collect::<Vec<_>>();
                let (a, b) = match changed.as_slice() {
                    [] => (0, 0),
                    [a, b] if axes[*a] == *b && axes[*b] == *a => (*a, *b),
                    _ => return Err(unsupported("transpose permutations beyond one axis swap")),
                };
                for axis in [a, b] {
                    attributes.push(Variable::new(
                        old::tensor::Tensor::from_vec(vec![axis as f32], &[]).map_err(translate)?,
                    ));
                }
            }
            Operation::Matmul => {
                let a = inputs[0].tensor().shape();
                let b = inputs[1].tensor().shape();
                if a.is_empty() || b.is_empty() {
                    return Err(invalid().into());
                }
                if a.len() > 2 || b.len() > 2 {
                    if a.len() < 2 || b.len() < 2 {
                        return Err(unsupported("batched vector matmul"));
                    }
                    let ab = &a[..a.len() - 2];
                    let bb = &b[..b.len() - 2];
                    // Original flattens batch dimensions. Only accept layouts
                    // for which this is equivalent to the public batch contract.
                    if ab != bb && !ab.is_empty() && !bb.is_empty() {
                        return Err(unsupported("multi-axis batch broadcasting"));
                    }
                    if !ab.is_empty() && bb.is_empty() && ab.iter().product::<usize>() == 1 {
                        return Err(unsupported("legacy singleton batch output shape"));
                    }
                }
                if a[a.len() - 1] != b[b.len().saturating_sub(2)]
                    || a.contains(&0)
                    || b.contains(&0)
                {
                    return Err(TensorError::InvalidOperation {
                        op: "matmul",
                        reason: "incompatible matrix dimensions".into(),
                    }
                    .into());
                }
            }
            _ => {}
        }
        inputs.extend(attributes.iter());
        let mut operator = match operation {
            Operation::Loss { kind, .. } => match kind {
                LossKind::Mse => old::loss::MeanSquaredError::new(),
                LossKind::Mae => old::loss::MeanAbsoluteError::new(),
                LossKind::Huber { .. } => old::loss::HuberLoss::new(),
                LossKind::BinaryCrossEntropy => old::loss::BinaryCrossEntropyLoss::new(),
                LossKind::CrossEntropy => old::loss::CrossEntropyLoss::new(),
                LossKind::SoftmaxCrossEntropy => old::loss::SoftmaxCrossEntropyLoss::new(),
            }
            .map_err(translate)?,
            Operation::TopK { .. } if self.no_grad => {
                old::tensor::operators::Topk::new().map_err(translate)?
            }
            Operation::Matmax { .. } if self.no_grad => {
                old::tensor::operators::Matmax::new().map_err(translate)?
            }
            Operation::Pow(_) => Pow::new().map_err(translate)?,
            Operation::Concat { .. } => Concat::new().map_err(translate)?,
            Operation::Conv2d { .. } => Conv2dOp::new().map_err(translate)?,
            Operation::GroupNorm { .. } => GroupNormOp::new().map_err(translate)?,
            Operation::MaxPool2d { .. } => MaxPool2d::new().map_err(translate)?,
            Operation::AvgPool2d { .. } => AvgPool2d::new().map_err(translate)?,
            Operation::NearestUpsample2d { .. } => NearestUpsample2d::new().map_err(translate)?,
            Operation::Div => {
                if inputs[0].tensor().shape() != inputs[1].tensor().shape() {
                    return Err(unsupported("broadcast division"));
                }
                Div::new().map_err(translate)?
            }
            Operation::Sum => Sum::new().map_err(translate)?,
            Operation::Tanh => {
                old::nn::activation::TanhOp::new().map_err(translate)?
            }
            Operation::Sigmoid => {
                old::nn::activation::SigmoidOp::new().map_err(translate)?
            }
            Operation::Softmax { .. } => {
                old::nn::activation::SoftmaxOp::new().map_err(translate)?
            }
            Operation::ApproxSin { .. } => ApproxSin::new().map_err(translate)?,
            Operation::ApproxCos { .. } => ApproxCos::new().map_err(translate)?,
            Operation::Add => Add::new().map_err(translate)?,
            Operation::Mul => Mul::new().map_err(translate)?,
            Operation::Reshape(_) => ReshapeOp::new().map_err(translate)?,
            Operation::Transpose(_) => Transpose::new().map_err(translate)?,
            Operation::Matmul => Matmul::new().map_err(translate)?,
            Operation::Sub => Sub::new().map_err(translate)?,
            Operation::Neg => Neg::new().map_err(translate)?,
            Operation::Square => Square::new().map_err(translate)?,
            Operation::Exp => Exp::new().map_err(translate)?,
            Operation::Sin => Sin::new().map_err(translate)?,
            Operation::Cos => Cos::new().map_err(translate)?,
            Operation::Relu => old::nn::activation::ReLUOp::new().map_err(translate)?,
            Operation::Silu => old::nn::activation::SiLUOp::new().map_err(translate)?,
            Operation::Abs => Abs::new().map_err(translate)?,
            Operation::Log => Log::new().map_err(translate)?,
            Operation::Sqrt => Sqrt::new().map_err(translate)?,
            _ => {
                return Err(MlError::UnsupportedCapability {
                    module: "legacy execution",
                    capability: "native operation",
                    operation: operation.name(),
                });
            }
        };
        let mut value = if self.no_grad {
            let tensors = inputs
                .iter()
                .map(|v| v.tensor() as &dyn TensorBase)
                .collect::<Vec<_>>();
            let mut outputs = operator.forward(&tensors).map_err(translate)?;
            if matches!(operation, Operation::TopK { .. } | Operation::Matmax { .. }) {
                let values = outputs
                    .into_iter()
                    .map(|output| output.to_id().map(Variable::new).map_err(translate))
                    .collect::<MlResult<Vec<_>>>()?;
                self.forwards += 1;
                return Ok(values.into_iter().map(|value| self.insert(value)).collect());
            }
            if outputs.is_empty() {
                return Err(TensorError::InvalidOperation {
                    op: "legacy execution",
                    reason: "expected a primary output".into(),
                }
                .into());
            }
            Variable::new(outputs.remove(0).to_id().map_err(translate)?)
        } else if matches!(
            operation,
            Operation::GroupNorm { .. } | Operation::MaxPool2d { .. }
        ) {
            operator.apply_with_saved(&inputs).map_err(translate)?
        } else {
            operator.apply(&inputs).map_err(translate)?
        };
        // Original reductions return [1,1]; public reductions are rank-zero.
        // Express the metadata conversion through the original reshape op too.
        if matches!(operation, Operation::Sum | Operation::Loss { .. }) {
            let scalar_shape =
                Variable::new(old::tensor::Tensor::from_vec(vec![0.0], &[]).map_err(translate)?);
            let mut reshape = ReshapeOp::new().map_err(translate)?;
            value = if self.no_grad {
                Variable::new(
                    reshape
                        .forward(&[value.tensor(), scalar_shape.tensor()])
                        .map_err(translate)?
                        .remove(0)
                        .to_id()
                        .map_err(translate)?,
                )
            } else {
                reshape.apply(&[&value, &scalar_shape]).map_err(translate)?
            };
        }
        self.forwards += 1;
        Ok(vec![self.insert(value)])
    }
    fn backward(&mut self, output: NativeHandle) -> MlResult<()> {
        if self.no_grad {
            return Err(MlError::UnsupportedCapability {
                module: "legacy execution",
                capability: "backward inside no-grad",
                operation: "backward",
            });
        }
        let value = self.get(output)?;
        if value.tensor().data().len() != 1 {
            return Err(
                crate::AutogradError::OutputNotScalar(value.tensor().shape().to_vec()).into(),
            );
        }
        value.backward().map_err(translate)?;
        self.backwards += 1;
        Ok(())
    }
    fn snapshot(&self, handle: NativeHandle) -> MlResult<TensorBuffer> {
        let value = self.get(handle)?.tensor();
        TensorBuffer::from_vec(value.data().to_vec(), value.shape())
    }
    fn gradient(&self, handle: NativeHandle) -> MlResult<TensorBuffer> {
        let value = self.get(handle)?.grad();
        TensorBuffer::from_vec(value.data().to_vec(), value.shape())
    }
    fn clear_graph(&mut self) {
        old::comparison::clear_graph();
        for value in self.values.values() {
            value.clear_grad();
        }
    }
    fn replace(&mut self, handle: NativeHandle, buffer: TensorBuffer) -> MlResult<()> {
        let value = self.get(handle)?;
        if value.tensor().shape() != buffer.shape() {
            return Err(TensorError::InvalidShape {
                expected: value.tensor().shape().to_vec(),
                got: buffer.shape().to_vec(),
            }
            .into());
        }
        let replacement =
            old::tensor::GlobalTensor::from_vec(buffer.data().to_vec(), buffer.shape())
                .map_err(translate)?;
        value.tensor().replace(replacement);
        Ok(())
    }
    fn with_no_grad<T>(&mut self, operation: impl FnOnce(&mut Self) -> MlResult<T>) -> MlResult<T> {
        struct Guard<'a> {
            session: &'a mut NativeSession,
            previous: bool,
        }
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                self.session.no_grad = self.previous;
            }
        }
        let previous = self.no_grad;
        self.no_grad = true;
        let guard = Guard {
            session: self,
            previous,
        };
        operation(guard.session)
    }
    /// Prototype scope: all newly created native handles are temporary. Only an
    /// owned result escapes; pre-existing parameter handles survive unchanged.
    fn training_step(
        &mut self,
        operation: impl FnOnce(&mut Self) -> MlResult<TensorBuffer>,
    ) -> MlResult<TensorBuffer> {
        if self.training || self.no_grad || old::comparison::statistics().map_err(translate)?.1 != 0
        {
            return Err(ContextError::ActiveGraphConflict.into());
        }
        struct Guard<'a> {
            session: &'a mut NativeSession,
            first_temporary: usize,
        }
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                self.session.clear_graph();
                self.session
                    .values
                    .retain(|index, _| *index < self.first_temporary);
                self.session.training = false;
            }
        }
        self.clear_graph();
        self.training = true;
        let first_temporary = self.next;
        let guard = Guard {
            session: self,
            first_temporary,
        };
        operation(guard.session)
    }
}
impl Drop for NativeSession {
    fn drop(&mut self) {
        self.clear_graph();
        self.values.clear();
        ACTIVE.with(|active| active.set(false));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::Operation;
    #[test]
    fn corrected_native_backward_contracts() -> MlResult<()> {
        let mut session = NativeSession::new()?;
        let x = Variable::new(old::tensor::Tensor::from_vec(vec![0.5], &[]).map_err(translate)?);
        x.retain_grad();
        let y = old::nn::activation::TanhOp::new()
            .map_err(translate)?
            .apply(&[&x])
            .map_err(translate)?;
        y.backward().map_err(translate)?;
        // Corrected Tanh recomputes its output from the original graph input.
        assert!((x.grad().data()[0] - (1.0 - 0.5f32.tanh().powi(2))).abs() < 1e-6);
        session.clear_graph();
        x.clear_grad();
        let sum = Sum::new()
            .map_err(translate)?
            .apply(&[&x])
            .map_err(translate)?;
        sum.backward().map_err(translate)?;
        assert_eq!(x.grad().shape(), &[] as &[usize]);
        assert_eq!(x.grad().data(), &[1.0]);
        session.clear_graph();
        Ok(())
    }
    #[test]
    fn native_training_updates_and_no_grad_preserve_lifetimes() -> MlResult<()> {
        let mut session = NativeSession::new()?;
        let parameter = session.tensor(TensorBuffer::from_vec(vec![2.0], &[])?)?;
        let baseline = old::comparison::statistics().map_err(translate)?;
        let mut expected = 2.0f32;
        for _ in 0..16 {
            session.training_step(|session| {
                assert!(
                    session
                        .training_step(|_| TensorBuffer::from_vec(vec![0.0], &[]))
                        .is_err()
                );
                let loss = session.execute(&Operation::Mul, &[parameter, parameter])?;
                let graph = old::comparison::statistics().map_err(translate)?.1;
                let prediction = session.with_no_grad(|session| {
                    session.with_no_grad(|session| {
                        session.execute(&Operation::Add, &[parameter, parameter])
                    })
                })?;
                assert_eq!(old::comparison::statistics().map_err(translate)?.1, graph);
                assert!((session.snapshot(prediction)?.data()[0] - expected * 2.0).abs() < 1e-6);
                session.backward(loss)?;
                let gradient = session.gradient(parameter)?.data()[0];
                assert!((gradient - 2.0 * expected).abs() < 1e-6);
                expected -= 0.1 * gradient;
                session.replace(parameter, TensorBuffer::from_vec(vec![expected], &[])?)?;
                session.snapshot(loss)
            })?;
            assert_eq!(old::comparison::statistics().map_err(translate)?, baseline);
            assert_eq!(session.values.len(), 1);
        }
        let before = session.snapshot(parameter)?;
        assert!(
            session
                .replace(parameter, TensorBuffer::from_vec(vec![1.0, 2.0], &[2])?)
                .is_err()
        );
        assert_eq!(session.snapshot(parameter)?, before);
        let failure = session.training_step(|session| {
            session.execute(&Operation::Mul, &[parameter, parameter])?;
            session.execute(&Operation::Matmax { axis: None, keepdim: false }, &[parameter])?;
            unreachable!()
        });
        assert!(failure.is_err());
        assert_eq!(old::comparison::statistics().map_err(translate)?, baseline);
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = session.training_step(|session| {
                session.with_no_grad(|session| {
                    session.execute(&Operation::Add, &[parameter, parameter])?;
                    panic!("injected step panic");
                })
            });
        }));
        assert!(unwind.is_err());
        assert!(!session.no_grad && !session.training);
        assert_eq!(old::comparison::statistics().map_err(translate)?, baseline);
        session.training_step(|session| session.snapshot(parameter))?;
        Ok(())
    }
    #[test]
    fn native_session_rejects_existing_graph_and_cleans_unwind() -> MlResult<()> {
        let x = Variable::new(old::tensor::Tensor::from_vec(vec![2.0], &[]).map_err(translate)?);
        let output = Mul::new()
            .map_err(translate)?
            .apply(&[&x, &x])
            .map_err(translate)?;
        let before = old::comparison::statistics().map_err(translate)?;
        assert!(NativeSession::new().is_err());
        assert_eq!(old::comparison::statistics().map_err(translate)?, before);
        old::comparison::clear_graph();
        drop(output);
        drop(x);
        let baseline = old::comparison::statistics().map_err(translate)?;
        let result = std::panic::catch_unwind(|| {
            let mut session = NativeSession::new().unwrap();
            let x = session
                .tensor(TensorBuffer::from_vec(vec![2.0], &[]).unwrap())
                .unwrap();
            session.execute(&Operation::Mul, &[x, x]).unwrap();
            panic!("injected native caller failure");
        });
        assert!(result.is_err());
        assert_eq!(old::comparison::statistics().map_err(translate)?, baseline);
        let _next = NativeSession::new()?;
        Ok(())
    }
    #[test]
    fn native_graph_backward_and_exclusive_lifetime() -> MlResult<()> {
        let baseline = old::comparison::statistics().map_err(translate)?;
        let stale;
        {
            let mut session = NativeSession::new()?;
            assert!(matches!(
                NativeSession::new(),
                Err(MlError::ContextError(ContextError::ActiveGraphConflict))
            ));
            let x = session.tensor(TensorBuffer::from_vec(vec![2.0], &[])?)?;
            stale = x;
            let squared = session.execute(&Operation::Mul, &[x, x])?;
            let loss = session.execute(&Operation::Add, &[squared, x])?;
            assert_eq!(session.snapshot(loss)?.data(), &[6.0]);
            assert!(old::comparison::statistics().map_err(translate)?.1 > 0);
            session.backward(loss)?;
            assert_eq!(session.gradient(x)?.data(), &[5.0]);
            assert_eq!((session.forwards, session.backwards), (2, 1));
            assert!(session.execute(&Operation::Matmax { axis: None, keepdim: false }, &[x]).is_err());
            session.clear_graph();
            assert_eq!(old::comparison::statistics().map_err(translate)?.1, 0);
            assert_eq!(session.snapshot(x)?.data(), &[2.0]);
        }
        assert_eq!(old::comparison::statistics().map_err(translate)?, baseline);
        let next = NativeSession::new()?;
        assert!(matches!(
            next.snapshot(stale),
            Err(MlError::ContextError(ContextError::Mismatch))
        ));
        Ok(())
    }
}
