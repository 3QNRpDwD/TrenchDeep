//! Shape-recording implementation behind the common prepare_forward API. No kernels run.
//! Placeholder buffers bridge the current concrete Tensor API; no placeholder
//! values enter the plan. This is not a sandbox for arbitrary Rust side effects.
use super::{invalid, plan::*};
use crate::{
    ExecutionContext, ExecutionRoute, MlResult, Parameter, Tensor, TensorBuffer, TensorId,
    contracts::Operation,
};
use std::collections::HashMap;

#[derive(Debug, Default)]
pub(crate) struct Recording {
    program: PreparedProgram,
    slots: HashMap<TensorId, TensorSlotId>,
    failed: bool,
}

#[cfg(all(
    test,
    feature = "builtinStorage",
    feature = "builtinKernels",
    feature = "enableBackward"
))]
mod tests {
    use super::*;
    use crate::contracts::{OperationOutput, OperationProvider, TensorView};
    #[derive(Debug)]
    struct NeverExecute;
    impl OperationProvider for NeverExecute {
        fn supports_prepared_replay(&self) -> bool {
            true
        }
        fn execute(&self, _: &Operation, _: &[TensorView<'_>]) -> MlResult<OperationOutput> {
            panic!("preparation must not run kernels")
        }
    }
    #[test]
    fn shape_recording_never_calls_kernels_or_captures_implicit_values() -> MlResult<()> {
        let ctx = ExecutionContext::builder().operations(NeverExecute).build();
        let input = ctx.tensor(vec![2.0], &[])?;
        let baseline = ctx.graph_stats()?;
        let plan = ctx.prepare_recorded(&[&input], &[], PreparedMode::Inference, || {
            let scale = ctx.constant_tensor(vec![3.0], &[])?;
            Ok(vec![input.mul(&scale)?])
        })?;
        assert_eq!(plan.node_count(), 1);
        assert_eq!(ctx.graph_stats()?, baseline);
        assert!(
            ctx.prepare_recorded(&[&input], &[], PreparedMode::Inference, || {
                let unbound = ctx.scalar(3.0)?;
                Ok(vec![input.mul(&unbound)?])
            })
            .is_err()
        );
        assert_eq!(ctx.graph_stats()?, baseline);
        Ok(())
    }
    #[test]
    fn host_reads_are_sticky_errors_and_failures_restore_context() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let input = ctx.tensor(vec![2.0], &[])?;
        let baseline = ctx.graph_stats()?;
        for read in 0..5 {
            assert!(
                ctx.prepare_recorded(&[&input], &[], PreparedMode::Inference, || {
                    match read {
                        0 => {
                            assert!(input.item().is_err());
                        }
                        1 => {
                            assert!(input.to_vec().is_err());
                        }
                        2 => {
                            assert!(input.with_view(|v| v.data()[0]).is_err());
                        }
                        3 => {
                            assert!(ctx.model_uniform(1, 1.0).is_err());
                        }
                        _ => {
                            assert!(ctx.parameter(vec![1.0], &[]).is_err());
                        }
                    }
                    // Even if the callback swallows a forbidden read, no plan installs.
                    Ok(vec![input.square()?])
                })
                .is_err()
            );
            assert_eq!(ctx.graph_stats()?, baseline);
            assert_eq!(input.item()?, 2.0);
        }
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = ctx.prepare_recorded(&[&input], &[], PreparedMode::Inference, || {
                panic!("adapter unwind")
            });
        }));
        assert!(panic.is_err());
        assert_eq!(input.item()?, 2.0);
        ctx.prepare_recorded(&[&input], &[], PreparedMode::Inference, || {
            Ok(vec![input.square()?])
        })?;
        assert_eq!(ctx.graph_stats()?, baseline);
        Ok(())
    }
    #[test]
    fn preparation_preserves_existing_parameter_gradient() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let parameter = ctx.parameter(vec![2.0], &[])?;
        let loss = parameter.variable().square()?;
        loss.backward()?;
        let before = parameter.grad()?;
        let baseline = ctx.graph_stats()?;
        for operation in 0..4 {
            assert!(
                ctx.prepare_recorded(&[], &[&parameter], PreparedMode::Training, || {
                    let attempt = match operation {
                        0 => ctx.scale_grad(parameter.variable(), 2.0),
                        1 => ctx.clear_all(),
                        2 => ctx.with_training_scope(|| Ok(())),
                        _ => ctx.replace_parameter(
                            parameter.variable(),
                            TensorBuffer::from_vec(vec![3.0], &[])?,
                        ),
                    };
                    assert!(attempt.is_err());
                    Ok(vec![parameter.tensor().square()?])
                })
                .is_err()
            );
            assert_eq!(parameter.grad()?, before);
            assert_eq!(parameter.tensor().item()?, 2.0);
            assert_eq!(ctx.graph_stats()?, baseline);
        }
        ctx.prepare_recorded(&[], &[&parameter], PreparedMode::Training, || {
            Ok(vec![parameter.tensor().square()?])
        })?;
        assert_eq!(parameter.grad()?, before);
        assert_eq!(ctx.graph_stats()?, baseline);
        Ok(())
    }
}
impl ExecutionContext {
    pub(crate) fn is_preparing(&self) -> bool {
        self.inner.preparation.borrow().is_some()
    }
    pub(crate) fn deny_preparation(&self, operation: &str) -> MlResult<()> {
        if let Some(recording) = self.inner.preparation.borrow_mut().as_mut() {
            recording.failed = true;
            return Err(invalid(format!(
                "{operation} is not allowed during shape preparation"
            )));
        }
        Ok(())
    }
    /// A caller assertion that values depend only on model configuration, not
    /// runtime data. Ordinary tensor() values are never implicitly captured.
    pub fn constant_tensor(&self, data: Vec<f32>, shape: &[usize]) -> MlResult<Tensor> {
        if !self.is_preparing() {
            return self.tensor(data, shape);
        }
        let value = TensorBuffer::from_vec(data, shape)?;
        let tensor = self.allocate(Some(value.clone()), None)?;
        if let Some(recording) = self.inner.preparation.borrow_mut().as_mut() {
            let slot = recording.program.constant(value)?;
            recording.slots.insert(tensor.id(), slot);
        }
        Ok(tensor)
    }
    pub(crate) fn record_operation(
        &self,
        operation: &Operation,
        inputs: &[&Tensor],
    ) -> MlResult<Vec<Tensor>> {
        let result = (|| {
            for input in inputs {
                self.validate(input)?;
            }
            let mut state = self.inner.preparation.borrow_mut();
            let recording = state.as_mut().expect("active recording");
            let ids = inputs.iter().map(|t| recording.slots.get(&t.id()).copied()
                .ok_or_else(|| invalid("undeclared tensor: bind a feed/parameter or declare an explicit constant")))
                .collect::<MlResult<Vec<_>>>()?;
            let slot = recording.program.operation(operation.clone(), &ids)?;
            let shape = recording.program.slots[slot.0].shape.clone();
            let tensor = self.allocate(
                Some(TensorBuffer::from_vec(
                    vec![0.0; super::prepare::numel(&shape)?],
                    &shape,
                )?),
                None,
            )?;
            recording.slots.insert(tensor.id(), slot);
            Ok(vec![tensor])
        })();
        if result.is_err() {
            self.inner.preparation.borrow_mut().as_mut().unwrap().failed = true;
        }
        result
    }
    pub(crate) fn prepare_recorded(
        &self,
        feeds: &[&Tensor],
        parameters: &[&Parameter],
        mode: PreparedMode,
        describe: impl FnOnce() -> MlResult<Vec<Tensor>>,
    ) -> MlResult<PreparedPlan> {
        if self.is_preparing() {
            return Err(invalid("nested preparation"));
        }
        {
            let state = self
                .inner
                .state
                .try_borrow()
                .map_err(|_| crate::ContextError::BorrowConflict)?;
            if self.inner.training_active.get()
                || state
                    .autograd
                    .as_ref()
                    .is_some_and(|engine| !engine.nodes().is_empty())
            {
                return Err(invalid("preparation requires an idle graph"));
            }
        }
        if self.route() != ExecutionRoute::P1
            || !self
                .inner
                .operations
                .as_ref()
                .is_some_and(|provider| provider.supports_prepared_replay())
        {
            return Err(invalid(
                "shape preparation requires an opted-in P1 provider",
            ));
        }
        let mut recording = Recording::default();
        for feed in feeds {
            self.validate(feed)?;
            if recording.slots.contains_key(&feed.id()) {
                return Err(invalid("duplicate recording feed handle"));
            }
            let slot = recording
                .program
                .input(&feed.shape()?, feed.as_variable()?.requires_grad()?)?;
            recording.slots.insert(feed.id(), slot);
        }
        for parameter in parameters {
            self.validate(parameter.tensor())?;
            if feeds
                .iter()
                .any(|feed| feed.id() == parameter.tensor().id())
            {
                return Err(invalid(
                    "parameter must not also be bound as a runtime feed",
                ));
            }
            let slot = recording.program.parameter(&parameter.tensor().shape()?)?;
            // Repeated parameters map to the first slot; all bindings stay checked.
            recording
                .slots
                .entry(parameter.tensor().id())
                .or_insert(slot);
        }
        // No eager graph is created by the adapter. Avoid TrainingScope here:
        // preparing must also preserve gradients from a completed prior run.
        let (recording, outputs) = (|| {
            *self.inner.preparation.borrow_mut() = Some(recording);
            struct Guard<'a>(&'a ExecutionContext);
            impl Drop for Guard<'_> {
                fn drop(&mut self) {
                    self.0.inner.preparation.borrow_mut().take();
                }
            }
            let _guard = Guard(self);
            let tensors = describe()?;
            let recording = self.inner.preparation.borrow_mut().take().unwrap();
            if recording.failed {
                return Err(invalid("preparation encountered a forbidden operation"));
            }
            let outputs = tensors
                .iter()
                .map(|t| {
                    recording
                        .slots
                        .get(&t.id())
                        .copied()
                        .ok_or_else(|| invalid("undeclared prepared output"))
                })
                .collect::<MlResult<Vec<_>>>()?;
            Ok((recording, outputs))
        })()?;
        self.prepare(&recording.program, parameters, &outputs, mode)
    }
}
