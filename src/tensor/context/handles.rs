use super::*;

impl ContextTensor {
    pub fn context_id(&self) -> ContextId {
        self.0.context_id
    }
    pub fn node_id(&self) -> NodeId {
        self.0.node_id
    }

    fn runtime(&self) -> MlResult<Rc<ContextRuntime>> {
        self.0
            .runtime
            .upgrade()
            .ok_or_else(|| ContextError::Dropped.into())
    }

    pub(super) fn snapshot(&self) -> MlResult<GlobalTensor<f32>> {
        let runtime = self.runtime()?;
        let state = runtime
            .state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?;
        state
            .tensors
            .get(&self.node_id())
            .ok_or(ContextError::UnknownTensor(self.node_id()))?
            .snapshot()
    }

    pub fn to_vec(&self) -> MlResult<Vec<f32>> {
        Ok(self.snapshot()?.data)
    }
    pub fn shape(&self) -> MlResult<Vec<usize>> {
        self.with_view(|view| view.shape.to_vec())
    }

    /// Inspect borrowed data without copying the tensor buffer.
    pub fn with_view<R>(&self, f: impl FnOnce(TensorView<'_>) -> R) -> MlResult<R> {
        context_for(self)?.with_tensor(self, f)
    }

    pub fn numel(&self) -> MlResult<usize> {
        self.with_view(|view| view.data.len())
    }

    pub fn item(&self) -> MlResult<f32> {
        self.with_view(|value| {
        if value.data.len() != 1 {
            return Err(TensorError::NotScalar { shape: value.shape.to_vec() }.into());
        }
        Ok(value.data[0])
        })?
    }

    pub fn get(&self, indices: &[usize]) -> MlResult<Option<f32>> {
        self.with_view(|value| {
        if indices.len() != value.shape.len() {
            return Ok(None);
        }
        let mut flat = 0usize;
        for (index, dim) in indices.iter().zip(value.shape) {
            if index >= dim {
                return Ok(None);
            }
            flat = flat * dim + index;
        }
        Ok(value.data.get(flat).copied())
        })?
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
        let runtime = self.tensor.runtime()?;
        let ctx = ExecutionContext {
            id: self.tensor.context_id(),
            runtime: runtime.clone(),
            _not_sync: Rc::new(Cell::new(())),
        };
        let buffer = {
            let state = runtime
                .state
                .try_borrow()
                .map_err(|_| ContextError::BorrowConflict)?;
            state
                .tensors
                .get(&self.tensor.node_id())
                .ok_or(ContextError::UnknownTensor(self.tensor.node_id()))?
                .buffer
                .clone()
        };
        Ok(Self {
            tensor: ctx.insert_buffer(buffer)?,
            requires_grad: false,
        })
    }

    pub fn retain_grad(&self) -> MlResult<()> {
        let runtime = self.tensor.runtime()?;
        let mut state = runtime
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        if !state.tracked.contains(&self.tensor.node_id()) {
            return Err(AutogradError::NodeNotFound(self.tensor.node_id()).into());
        }
        state.retained_gradients.insert(self.tensor.node_id());
        Ok(())
    }

    pub fn grad(&self) -> MlResult<Option<GlobalTensor<f32>>> {
        let runtime = self.tensor.runtime()?;
        let state = runtime
            .state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?;
        if !state.tensors.contains_key(&self.tensor.node_id()) {
            return Err(ContextError::UnknownTensor(self.tensor.node_id()).into());
        }
        Ok(state.gradients.get(&self.tensor.node_id()).cloned())
    }

    pub fn backward(&self) -> MlResult<()> {
        let runtime = self.tensor.runtime()?;
        let ctx = ExecutionContext {
            id: self.tensor.context_id(),
            runtime,
            _not_sync: Rc::new(Cell::new(())),
        };
        ctx.backward(self, BackwardOptions::default())
    }

    pub fn backward_with_grad(&self, gradient: &ContextTensor) -> MlResult<()> {
        let runtime = self.tensor.runtime()?;
        let ctx = ExecutionContext {
            id: self.tensor.context_id(),
            runtime,
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
        runtime: tensor.runtime()?,
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

