use super::*;
impl ExecutionContext {
    fn tracking(&self, inputs: &[&Tensor]) -> MlResult<bool> {
        for input in inputs {
            self.validate(input)?;
        }
        let state = self
            .inner
            .state
            .try_borrow()
            .map_err(|_| ContextError::BorrowConflict)?;
        let tracked =
            self.inner.no_grad.get() == 0 && inputs.iter().any(|x| state.tracked.contains(&x.id()));
        if tracked {
            state.engine()?;
        }
        Ok(tracked)
    }
    fn with_inputs<R>(
        &self,
        inputs: &[&Tensor],
        compute: impl FnOnce(&[TensorView<'_>]) -> MlResult<R>,
    ) -> MlResult<R> {
        // Owned input snapshots avoid holding provider borrows across arbitrary
        // user callbacks. Metadata/view APIs themselves remain copy-free.
        let values = inputs
            .iter()
            .map(|x| x.snapshot())
            .collect::<MlResult<Vec<_>>>()?;
        let views = values.iter().map(TensorBuffer::view).collect::<Vec<_>>();
        compute(&views)
    }
    pub fn execute(&self, operation: &Operation, inputs: &[&Tensor]) -> MlResult<Vec<Tensor>> {
        if inputs.is_empty() || operation.input_count().is_some_and(|n| n != inputs.len()) {
            return Err(TensorError::InvalidOperation {
                op: operation.name(),
                reason: "invalid input count".into(),
            }
            .into());
        }
        // Loss targets are constants even when their handles happen to be tracked.
        for input in inputs {
            self.validate(input)?;
        }
        let tracked = self.tracking(if matches!(operation, Operation::Loss { .. }) {
            &inputs[..1]
        } else {
            inputs
        })?;
        let provider = self
            .inner
            .operations
            .as_deref()
            .ok_or_else(|| missing("operations", "forward", operation.name()))?;
        let result = self.with_inputs(inputs, |views| provider.execute(operation, views))?;
        let expected = if matches!(operation, Operation::TopK { .. } | Operation::Matmax { .. }) {
            2
        } else {
            1
        };
        if result.outputs.len() != expected {
            return Err(TensorError::InvalidOperation {
                op: operation.name(),
                reason: "provider returned an invalid output count".into(),
            }
            .into());
        }
        self.commit(operation.name(), inputs, result, tracked)
    }
    pub fn apply_custom(&self, op: &dyn CustomOp, inputs: &[&Tensor]) -> MlResult<Tensor> {
        if inputs.len() != op.input_count() {
            return Err(TensorError::InvalidOperation {
                op: op.name(),
                reason: "invalid input count".into(),
            }
            .into());
        }
        let tracked = self.tracking(inputs)?;
        let result = self.with_inputs(inputs, |views| op.forward(views))?;
        let result = OperationOutput {
            outputs: vec![result.output],
            saved: result.saved,
            backward: result.backward,
        };
        self.commit(op.name(), inputs, result, tracked)?
            .pop()
            .ok_or_else(|| TensorError::EmptyTensor.into())
    }
    fn commit(
        &self,
        name: &'static str,
        inputs: &[&Tensor],
        result: OperationOutput,
        tracked: bool,
    ) -> MlResult<Vec<Tensor>> {
        if tracked && (result.backward.is_none() || result.outputs.len() != 1) {
            return Err(AutogradError::BackwardNotSupported(name.into()).into());
        }
        if let Some(op) = &result.backward {
            if op.input_count() != inputs.len() {
                return Err(AutogradError::BackwardArityMismatch {
                    expected: inputs.len(),
                    got: op.input_count(),
                }
                .into());
            }
        }
        let outputs = result
            .outputs
            .into_iter()
            .map(|v| self.allocate(Some(v), None))
            .collect::<MlResult<Vec<_>>>()?;
        if tracked {
            let saved = result
                .saved
                .into_iter()
                .map(|v| self.allocate(Some(v), None))
                .collect::<MlResult<Vec<_>>>()?;
            let output = outputs.first().ok_or(TensorError::EmptyTensor)?.id();
            let record = GradientRecord {
                output,
                inputs: inputs.iter().map(|x| x.id()).collect(),
                saved: saved.iter().map(|x| x.id()).collect(),
                backward: Rc::from(
                    result
                        .backward
                        .ok_or_else(|| AutogradError::BackwardNotSupported(name.into()))?,
                ),
            };
            let mut state = self
                .inner
                .state
                .try_borrow_mut()
                .map_err(|_| ContextError::BorrowConflict)?;
            let mut counts = HashMap::<TensorId, usize>::new();
            for id in std::iter::once(output)
                .chain(record.inputs.iter().copied())
                .chain(record.saved.iter().copied())
            {
                *counts.entry(id).or_default() += 1;
            }
            for (&id, &count) in &counts {
                let entry = state
                    .handles
                    .get(&id)
                    .ok_or(ContextError::UnknownTensor(id))?;
                entry
                    .pins
                    .checked_add(count)
                    .ok_or_else(|| TensorError::InvalidOperation {
                        op: "record",
                        reason: "pin overflow".into(),
                    })?;
            }
            state
                .autograd
                .as_mut()
                .ok_or_else(|| missing("autograd", "record", "forward"))?
                .record(record)?;
            for (id, count) in counts {
                if let Some(e) = state.handles.get_mut(&id) {
                    e.pins += count;
                }
            }
            state.tracked.insert(output);
        }
        Ok(outputs)
    }
    pub(super) fn single(&self, op: Operation, inputs: &[&Tensor]) -> MlResult<Tensor> {
        self.execute(&op, inputs)?
            .pop()
            .ok_or_else(|| TensorError::EmptyTensor.into())
    }
}
