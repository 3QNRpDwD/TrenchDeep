use super::*;

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self {
            id: ContextId(NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed)),
            runtime: Rc::new(ContextRuntime {
                state: RefCell::new(ContextState {
                    tensors: HashMap::new(),
                    next_node: 0,
                    graph: HashMap::new(),
                    tracked: HashSet::new(),
                    leaves: HashSet::new(),
                    retained_gradients: HashSet::new(),
                    gradients: HashMap::new(),
                    consumed: HashSet::new(),
                    no_grad_depth: 0,
                }),
                gc_pending: Cell::new(false),
            }),
            _not_sync: Rc::new(Cell::new(())),
        }
    }

    fn collect_pending_garbage(&self) -> MlResult<()> {
        if self.runtime.gc_pending.get() {
            let mut state = self
                .runtime
                .state
                .try_borrow_mut()
                .map_err(|_| ContextError::BorrowConflict)?;
            state.collect_garbage();
            self.runtime.gc_pending.set(false);
        }
        Ok(())
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
            let mut state = self
                .runtime
                .state
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

    pub fn grad(&self, parameter: &ContextVariable) -> MlResult<Option<GlobalTensor<f32>>> {
        self.validate(parameter.tensor())?;
        parameter.grad()
    }

    pub fn clear_grad(&self, parameter: &ContextVariable) -> MlResult<()> {
        self.validate(parameter.tensor())?;
        self.runtime.state.try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?
            .gradients.remove(&parameter.tensor().node_id());
        Ok(())
    }

    pub fn scale_grad(&self, parameter: &ContextVariable, factor: f32) -> MlResult<()> {
        self.validate(parameter.tensor())?;
        if !factor.is_finite() {
            return Err(TensorError::InvalidOperation {
                op: "scale_grad",
                reason: "factor must be finite".into(),
            }.into());
        }
        let mut state = self.runtime.state.try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        if let Some(gradient) = state.gradients.get_mut(&parameter.tensor().node_id()) {
            for value in &mut gradient.data {
                *value *= factor;
            }
        }
        Ok(())
    }

    fn update_parameter(
        &self,
        parameter: &ContextVariable,
        delta: &GlobalTensor<f32>,
        operation: &'static str,
        sign: f32,
    ) -> MlResult<()> {
        self.validate(parameter.tensor())?;
        let state = self.runtime.state.try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?;
        let entry = state.tensors.get(&parameter.tensor().node_id())
            .ok_or(ContextError::UnknownTensor(parameter.tensor().node_id()))?;
        let mut value = entry.buffer.try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        if value.shape != delta.shape {
            return Err(TensorError::InvalidShape {
                expected: value.shape.clone(),
                got: delta.shape.clone(),
            }.into());
        }
        if value.data.len() != delta.data.len() {
            return Err(TensorError::InvalidOperation {
                op: operation,
                reason: "shape-compatible tensors had different storage lengths".into(),
            }.into());
        }
        for (destination, change) in value.data.iter_mut().zip(&delta.data) {
            *destination += sign * change;
        }
        Ok(())
    }

    pub fn add_assign(
        &self,
        parameter: &ContextVariable,
        delta: &GlobalTensor<f32>,
    ) -> MlResult<()> {
        self.update_parameter(parameter, delta, "add_assign", 1.0)
    }

    pub fn sub_assign(
        &self,
        parameter: &ContextVariable,
        delta: &GlobalTensor<f32>,
    ) -> MlResult<()> {
        self.update_parameter(parameter, delta, "sub_assign", -1.0)
    }

    pub fn replace_parameter(
        &self,
        parameter: &ContextVariable,
        replacement: GlobalTensor<f32>,
    ) -> MlResult<()> {
        self.validate(parameter.tensor())?;
        let state = self.runtime.state.try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?;
        let entry = state.tensors.get(&parameter.tensor().node_id())
            .ok_or(ContextError::UnknownTensor(parameter.tensor().node_id()))?;
        let mut value = entry.buffer.try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        if value.shape != replacement.shape {
            return Err(TensorError::InvalidShape {
                expected: value.shape.clone(),
                got: replacement.shape,
            }.into());
        }
        *value = replacement;
        Ok(())
    }

    fn insert(&self, tensor: GlobalTensor<f32>) -> MlResult<ContextTensor> {
        self.insert_buffer(Rc::new(RefCell::new(tensor)))
    }

    pub(super) fn insert_buffer(&self, buffer: Rc<RefCell<GlobalTensor<f32>>>) -> MlResult<ContextTensor> {
        self.collect_pending_garbage()?;
        let mut state = self
            .runtime
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        let id = NodeId::from_raw(state.next_node);
        state.next_node += 1;
        let handle = Rc::new(ContextTensorHandle {
            context_id: self.id,
            node_id: id,
            runtime: Rc::downgrade(&self.runtime),
        });
        state.tensors.insert(
            id,
            TensorStorageEntry {
                buffer,
                external_handle: Rc::downgrade(&handle),
                graph_pins: 0,
                internal_saved: false,
            },
        );
        Ok(ContextTensor(handle))
    }

    fn validate(&self, tensor: &ContextTensor) -> MlResult<()> {
        self.collect_pending_garbage()?;
        if tensor.context_id() != self.id {
            return Err(ContextError::Mismatch.into());
        }
        if !self
            .runtime
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
            .runtime
            .state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?;
        let value = state
            .tensors
            .get(&tensor.node_id())
            .ok_or(ContextError::UnknownTensor(tensor.node_id()))?;
        let value = value
            .buffer
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?;
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
                reason: format!(
                    "shapes {:?} and {:?} cannot be broadcast",
                    left.shape, right.shape
                ),
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
        self.record(
            output.node_id(),
            vec![lhs.node_id(), rhs.node_id()],
            backward,
        )?;
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
            .runtime
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
                state.tensors.insert(
                    id,
                    TensorStorageEntry {
                        buffer: Rc::new(RefCell::new(value)),
                        external_handle: Weak::new(),
                        graph_pins: 0,
                        internal_saved: true,
                    },
                );
                owned_saved.push(id);
            }
            saved.extend(owned_saved.iter().copied());
            state.pin(output)?;
            for &input in &inputs {
                state.pin(input)?;
            }
            for &saved_id in &saved {
                state.pin(saved_id)?;
            }
            state.graph.insert(
                output,
                GraphNode {
                    inputs,
                    saved,
                    backward: into_node_backward(backward),
                },
            );
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

    pub fn approx_sin(&self, input: &ContextTensor, threshold: f32) -> MlResult<ContextTensor> {
        validate_approx_threshold("approx_sin", threshold)?;
        self.unary(input, BuiltinBackward::ApproxSin { threshold }, |x| {
            approx_sin_value(x)
        })
    }

    pub fn approx_cos(&self, input: &ContextTensor, threshold: f32) -> MlResult<ContextTensor> {
        validate_approx_threshold("approx_cos", threshold)?;
        self.unary(input, BuiltinBackward::ApproxCos { threshold }, |x| {
            approx_cos_value(x)
        })
    }

    pub fn tanh(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.unary(input, BuiltinBackward::Tanh, f32::tanh)
    }

    pub fn sigmoid(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.unary(input, BuiltinBackward::Sigmoid, |x| {
            1.0 / (1.0 + (-x).exp())
        })
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
            return Err(TensorError::InvalidAxis {
                axis,
                shape: value.shape,
            }
            .into());
        }
        let outer: usize = value.shape[..axis].iter().product();
        let width = value.shape[axis];
        let inner: usize = value.shape[axis + 1..].iter().product();
        let mut data = vec![0.0; value.data.len()];
        for outer_index in 0..outer {
            for inner_index in 0..inner {
                let maximum = (0..width)
                    .map(|i| value.data[(outer_index * width + i) * inner + inner_index])
                    .fold(f32::NEG_INFINITY, f32::max);
                let normalizer: f32 = (0..width)
                    .map(|i| {
                        (value.data[(outer_index * width + i) * inner + inner_index] - maximum)
                            .exp()
                    })
                    .sum();
                for i in 0..width {
                    let index = (outer_index * width + i) * inner + inner_index;
                    data[index] = (value.data[index] - maximum).exp() / normalizer;
                }
            }
        }
        let output = self.tensor(data, &value.shape)?;
        self.record(
            output.node_id(),
            vec![input.node_id()],
            BuiltinBackward::Softmax { axis },
        )?;
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

    fn loss(
        &self,
        prediction: &ContextTensor,
        target: &ContextTensor,
        reduction: Reduction,
        kind: ContextLossKind,
    ) -> MlResult<ContextTensor> {
        self.validate(prediction)?;
        self.validate(target)?;
        let tracked = {
            let state = self.runtime.state.try_borrow()
                .map_err(|_| ContextError::BorrowConflict)?;
            state.no_grad_depth == 0 && state.tracked.contains(&prediction.node_id())
        };
        let (value, saved) = {
            let state = self.runtime.state.try_borrow()
                .map_err(|_| ContextError::BorrowConflict)?;
            let prediction_entry = state.tensors.get(&prediction.node_id())
                .ok_or(ContextError::UnknownTensor(prediction.node_id()))?;
            let target_entry = state.tensors.get(&target.node_id())
                .ok_or(ContextError::UnknownTensor(target.node_id()))?;
            let prediction_value = prediction_entry.buffer.try_borrow()
                .map_err(|_| ContextError::BorrowConflict)?;
            let target_value = target_entry.buffer.try_borrow()
                .map_err(|_| ContextError::BorrowConflict)?;
            loss_forward(
                kind,
                reduction,
                TensorView { data: &prediction_value.data, shape: &prediction_value.shape },
                TensorView { data: &target_value.data, shape: &target_value.shape },
                tracked,
            )?
        };
        let output = self.insert(value)?;
        if tracked {
            self.record_with_saved(
                output.node_id(),
                vec![prediction.node_id(), target.node_id()],
                BuiltinBackward::Loss { kind, reduction },
                saved.into_iter().collect(),
            )?;
        }
        Ok(output)
    }

    pub fn mse_loss(
        &self,
        prediction: &ContextTensor,
        target: &ContextTensor,
        reduction: Reduction,
    ) -> MlResult<ContextTensor> {
        self.loss(prediction, target, reduction, ContextLossKind::Mse)
    }

    pub fn mae_loss(
        &self,
        prediction: &ContextTensor,
        target: &ContextTensor,
        reduction: Reduction,
    ) -> MlResult<ContextTensor> {
        self.loss(prediction, target, reduction, ContextLossKind::Mae)
    }

    pub fn huber_loss(
        &self,
        prediction: &ContextTensor,
        target: &ContextTensor,
        delta: f32,
        reduction: Reduction,
    ) -> MlResult<ContextTensor> {
        if !delta.is_finite() || delta <= 0.0 {
            return Err(LossError::InvalidOperation {
                op: "huber_loss",
                reason: format!("delta must be finite and positive, got {delta}"),
            }
            .into());
        }
        self.loss(
            prediction,
            target,
            reduction,
            ContextLossKind::Huber { delta },
        )
    }

    pub fn binary_cross_entropy(
        &self,
        prediction: &ContextTensor,
        target: &ContextTensor,
        reduction: Reduction,
    ) -> MlResult<ContextTensor> {
        self.loss(
            prediction,
            target,
            reduction,
            ContextLossKind::BinaryCrossEntropy,
        )
    }

    pub fn cross_entropy(
        &self,
        probabilities: &ContextTensor,
        one_hot_target: &ContextTensor,
        reduction: Reduction,
    ) -> MlResult<ContextTensor> {
        self.loss(
            probabilities,
            one_hot_target,
            reduction,
            ContextLossKind::CrossEntropy,
        )
    }

    pub fn softmax_cross_entropy(
        &self,
        logits: &ContextTensor,
        one_hot_target: &ContextTensor,
        reduction: Reduction,
    ) -> MlResult<ContextTensor> {
        self.loss(
            logits,
            one_hot_target,
            reduction,
            ContextLossKind::SoftmaxCrossEntropy,
        )
    }

    pub fn reshape(&self, input: &ContextTensor, shape: &[usize]) -> MlResult<ContextTensor> {
        self.validate(input)?;
        let value = input.snapshot()?;
        let requested_len = shape.iter().product::<usize>();
        if value.data.len() != requested_len {
            return Err(TensorError::InvalidDataLength {
                expected: value.data.len(),
                got: requested_len,
            }
            .into());
        }
        let output = self.tensor(value.data, shape)?;
        self.record(
            output.node_id(),
            vec![input.node_id()],
            BuiltinBackward::Reshape,
        )?;
        Ok(output)
    }

    pub fn transpose(&self, input: &ContextTensor, axes: &[usize]) -> MlResult<ContextTensor> {
        self.validate(input)?;
        let value = input.snapshot()?;
        validate_permutation(&value.shape, axes)?;
        let output_shape: Vec<_> = axes.iter().map(|&axis| value.shape[axis]).collect();
        let data = permute_data(&value.data, &value.shape, axes);
        let output = self.tensor(data, &output_shape)?;
        self.record(
            output.node_id(),
            vec![input.node_id()],
            BuiltinBackward::Transpose(axes.to_vec()),
        )?;
        Ok(output)
    }

    pub fn concat(&self, inputs: &[&ContextTensor], axis: usize) -> MlResult<ContextTensor> {
        if inputs.is_empty() {
            return Err(TensorError::InvalidInputCount {
                expected: 1,
                got: 0,
            }
            .into());
        }
        let values = inputs
            .iter()
            .map(|input| {
                self.validate(input)?;
                input.snapshot()
            })
            .collect::<MlResult<Vec<_>>>()?;
        let rank = values[0].shape.len();
        if axis >= rank {
            return Err(TensorError::InvalidAxis {
                axis,
                shape: values[0].shape.clone(),
            }
            .into());
        }
        for value in &values[1..] {
            if value.shape.len() != rank
                || value
                    .shape
                    .iter()
                    .enumerate()
                    .any(|(i, dim)| i != axis && *dim != values[0].shape[i])
            {
                return Err(TensorError::InvalidOperation {
                    op: "concat",
                    reason: "non-concatenated dimensions must match".into(),
                }
                .into());
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
            output.node_id(),
            inputs.iter().map(|input| input.node_id()).collect(),
            BuiltinBackward::Concat {
                axis,
                sizes: values.iter().map(|value| value.shape[axis]).collect(),
            },
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
                        data[(batch * spec.m + i) * spec.n + j] += left.data
                            [(left_batch * spec.m + i) * spec.k + p]
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
        let output =
            conv2d_forward_data(&input_value, &weight_value, &bias_value, stride, padding)?;
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

    pub fn group_norm(
        &self,
        input: &ContextTensor,
        gamma: &ContextTensor,
        beta: &ContextTensor,
        groups: usize,
        epsilon: f32,
    ) -> MlResult<ContextTensor> {
        self.validate(input)?;
        self.validate(gamma)?;
        self.validate(beta)?;
        let (input_value, gamma_value, beta_value) =
            (input.snapshot()?, gamma.snapshot()?, beta.snapshot()?);
        let (output, saved) =
            group_norm_forward_data(&input_value, &gamma_value, &beta_value, groups, epsilon)?;
        let result = self.tensor(output.data, &output.shape)?;
        self.record_with_saved(
            result.node_id(),
            vec![input.node_id(), gamma.node_id(), beta.node_id()],
            BuiltinBackward::GroupNorm { groups, epsilon },
            saved,
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

    pub(super) fn variable_from(&self, tensor: ContextTensor) -> MlResult<ContextVariable> {
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

    pub fn powf_variable(
        &self,
        input: &ContextVariable,
        exponent: f32,
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.powf(input.tensor(), exponent)?)
    }

    pub fn sin_variable(&self, input: &ContextVariable) -> MlResult<ContextVariable> {
        self.variable_from(self.sin(input.tensor())?)
    }

    pub fn cos_variable(&self, input: &ContextVariable) -> MlResult<ContextVariable> {
        self.variable_from(self.cos(input.tensor())?)
    }

    pub fn approx_sin_variable(
        &self,
        input: &ContextVariable,
        threshold: f32,
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.approx_sin(input.tensor(), threshold)?)
    }

    pub fn approx_cos_variable(
        &self,
        input: &ContextVariable,
        threshold: f32,
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.approx_cos(input.tensor(), threshold)?)
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

    pub fn softmax_variable(
        &self,
        input: &ContextVariable,
        axis: usize,
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.softmax(input.tensor(), axis)?)
    }

    pub fn sum_variable(&self, input: &ContextVariable) -> MlResult<ContextVariable> {
        self.variable_from(self.sum(input.tensor())?)
    }

    pub fn mse_loss_variable(
        &self,
        prediction: &ContextVariable,
        target: &ContextTensor,
        reduction: Reduction,
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.mse_loss(prediction.tensor(), target, reduction)?)
    }

    pub fn mae_loss_variable(
        &self,
        prediction: &ContextVariable,
        target: &ContextTensor,
        reduction: Reduction,
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.mae_loss(prediction.tensor(), target, reduction)?)
    }

    pub fn huber_loss_variable(
        &self,
        prediction: &ContextVariable,
        target: &ContextTensor,
        delta: f32,
        reduction: Reduction,
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.huber_loss(prediction.tensor(), target, delta, reduction)?)
    }

    pub fn binary_cross_entropy_variable(
        &self,
        prediction: &ContextVariable,
        target: &ContextTensor,
        reduction: Reduction,
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.binary_cross_entropy(prediction.tensor(), target, reduction)?)
    }

    pub fn cross_entropy_variable(
        &self,
        probabilities: &ContextVariable,
        one_hot_target: &ContextTensor,
        reduction: Reduction,
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.cross_entropy(probabilities.tensor(), one_hot_target, reduction)?)
    }

    pub fn softmax_cross_entropy_variable(
        &self,
        logits: &ContextVariable,
        one_hot_target: &ContextTensor,
        reduction: Reduction,
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.softmax_cross_entropy(
            logits.tensor(),
            one_hot_target,
            reduction,
        )?)
    }

    pub fn reshape_variable(
        &self,
        input: &ContextVariable,
        shape: &[usize],
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.reshape(input.tensor(), shape)?)
    }

    pub fn transpose_variable(
        &self,
        input: &ContextVariable,
        axes: &[usize],
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.transpose(input.tensor(), axes)?)
    }

    pub fn concat_variables(
        &self,
        inputs: &[&ContextVariable],
        axis: usize,
    ) -> MlResult<ContextVariable> {
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
        self.variable_from(self.conv2d(
            input.tensor(),
            weight.tensor(),
            bias.tensor(),
            stride,
            padding,
        )?)
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

    pub fn group_norm_variable(
        &self,
        input: &ContextVariable,
        gamma: &ContextVariable,
        beta: &ContextVariable,
        groups: usize,
        epsilon: f32,
    ) -> MlResult<ContextVariable> {
        self.variable_from(self.group_norm(
            input.tensor(),
            gamma.tensor(),
            beta.tensor(),
            groups,
            epsilon,
        )?)
    }

    fn is_tracked(&self, tensor: &ContextTensor) -> MlResult<bool> {
        self.validate(tensor)?;
        Ok(self
            .runtime
            .state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?
            .tracked
            .contains(&tensor.node_id()))
    }

    fn reject_tracked_forward_only(
        &self,
        tensor: &ContextTensor,
        operation: &'static str,
    ) -> MlResult<()> {
        self.validate(tensor)?;
        let state = self
            .runtime
            .state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?;
        if state.no_grad_depth == 0 && state.tracked.contains(&tensor.node_id()) {
            return Err(AutogradError::BackwardNotSupported(operation.to_string()).into());
        }
        Ok(())
    }

    pub fn topk(&self, input: &ContextTensor, k: usize, sorted: bool) -> MlResult<TopKResult> {
        self.reject_tracked_forward_only(input, "Topk")?;
        let (values, indices, shape) = self.with_tensor(input, |view| {
            topk_forward_data(view, k, sorted)
        })??;
        Ok(TopKResult {
            values: self.tensor(values, &shape)?,
            indices: self.tensor(indices, &shape)?,
        })
    }

    pub fn matmax(
        &self,
        input: &ContextTensor,
        axis: Option<isize>,
        keepdim: bool,
    ) -> MlResult<MaxResult> {
        self.reject_tracked_forward_only(input, "Matmax")?;
        let (values, indices, shape) = self.with_tensor(input, |view| {
            matmax_forward_data(view, axis, keepdim)
        })??;
        Ok(MaxResult {
            values: self.tensor(values, &shape)?,
            indices: self.tensor(indices, &shape)?,
        })
    }

    pub fn clear_graph(&self) -> MlResult<()> {
        self.collect_pending_garbage()?;
        let mut state = self
            .runtime
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        let outputs: Vec<_> = state.graph.keys().copied().collect();
        for output in outputs {
            state.remove_graph_node(output);
        }
        state.consumed.clear();
        state.collect_garbage();
        self.runtime.gc_pending.set(false);
        Ok(())
    }

    pub fn clear_all(&self) -> MlResult<()> {
        self.collect_pending_garbage()?;
        let mut state = self
            .runtime
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
        self.runtime.gc_pending.set(false);
        Ok(())
    }

    pub fn no_grad<T>(&self, f: impl FnOnce() -> MlResult<T>) -> MlResult<T> {
        self.collect_pending_garbage()?;
        {
            let mut state = self
                .runtime
                .state
                .try_borrow_mut()
                .map_err(|_| ContextError::BorrowConflict)?;
            state.no_grad_depth += 1;
        }
        struct Reset<'a>(&'a ExecutionContext);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                if let Ok(mut state) = self.0.runtime.state.try_borrow_mut() {
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
        self.collect_pending_garbage()?;
        let state = self
            .runtime
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
            .runtime
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
            let Some(grad) = state.gradients.get(&id).cloned() else {
                continue;
            };
            let values = node
                .inputs
                .iter()
                .map(|input| {
                    state
                        .tensors
                        .get(input)
                        .ok_or(ContextError::UnknownTensor(*input))?
                        .snapshot()
                })
                .collect::<MlResult<Vec<_>>>()?;
            let op = &node.backward;
            let saved = node
                .saved
                .iter()
                .map(|saved| {
                    state
                        .tensors
                        .get(saved)
                        .ok_or(ContextError::UnknownTensor(*saved))?
                        .snapshot()
                })
                .collect::<MlResult<Vec<_>>>()?;
            let input_views: Vec<_> = values
                .iter()
                .map(|value| TensorView {
                    data: &value.data,
                    shape: &value.shape,
                })
                .collect();
            let saved_views: Vec<_> = saved
                .iter()
                .map(|value| TensorView {
                    data: &value.data,
                    shape: &value.shape,
                })
                .collect();
            let results = op.backward(
                &input_views,
                &saved_views,
                TensorView {
                    data: &grad.data,
                    shape: &grad.shape,
                },
            )?;
            if results.len() != op.input_count() {
                return Err(AutogradError::BackwardArityMismatch {
                    expected: op.input_count(),
                    got: results.len(),
                }
                .into());
            }
            for ((input, input_value), incoming) in
                input_ids.into_iter().zip(values.iter()).zip(results)
            {
                let Some(incoming) = incoming else {
                    continue;
                };
                if incoming.shape != input_value.shape {
                    return Err(AutogradError::GradientShapeMismatch {
                        expected: input_value.shape.clone(),
                        got: incoming.shape,
                    }
                    .into());
                }
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
                state.remove_graph_node(id);
            }
            state.consumed.insert(output.tensor.node_id());
            state.collect_garbage();
            self.runtime.gc_pending.set(false);
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

