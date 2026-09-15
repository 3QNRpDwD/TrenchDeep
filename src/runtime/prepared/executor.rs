use super::{invalid, plan::*};
use crate::{ExecutionContext, MlResult, Parameter, Tensor};

impl PreparedPlan {
    /// Bind current values and execute allocating replay. The callback may run
    /// backward and update an optimizer. Scope cleanup also runs after errors.
    /// Returned tensor handles own independent results across later executions.
    /// Training callbacks may differentiate declared outputs once (optionally
    /// with an explicit seed), but cannot append operations or capture an eager
    /// graph. Parameter updates require completed backward.
    pub fn with_run<T>(
        &self,
        ctx: &ExecutionContext,
        feeds: &[&Tensor],
        parameters: &[&Parameter],
        callback: impl FnOnce(&[Tensor]) -> MlResult<T>,
    ) -> MlResult<T> {
        if self.input_signature.is_some() {
            return Err(invalid("named plans require with_inputs"));
        }
        self.run_bound(ctx, feeds, parameters, callback)
    }
    pub(super) fn run_bound<T>(
        &self,
        ctx: &ExecutionContext,
        feeds: &[&Tensor],
        parameters: &[&Parameter],
        callback: impl FnOnce(&[Tensor]) -> MlResult<T>,
    ) -> MlResult<T> {
        self.validate_bindings(ctx, feeds, parameters)?;
        if self.mode == PreparedMode::Training {
            return ctx.with_training_scope(|| self.run_training(ctx, feeds, parameters, callback));
        }
        let execute = || {
            let mut slots: Vec<Option<Tensor>> = vec![None; self.program.slots.len()];
            for (index, slot) in self.program.slots.iter().enumerate() {
                if let Source::Constant(value) = &slot.source {
                    slots[index] = Some(ctx.tensor(value.data().to_vec(), value.shape())?);
                }
            }
            for (feed, slot) in feeds.iter().zip(&self.program.feeds) {
                slots[slot.0] = Some((*feed).clone());
            }
            for (parameter, slot) in parameters.iter().zip(&self.program.parameters) {
                slots[slot.0] = Some(parameter.tensor().clone());
            }
            for node in &self.program.instructions {
                let inputs = node
                    .inputs
                    .iter()
                    .map(|id| slots[id.0].as_ref().expect("validated DAG"))
                    .collect::<Vec<_>>();
                let mut outputs = ctx.execute(&node.operation, &inputs)?;
                if outputs.len() != 1
                    || outputs[0].shape()? != self.program.slots[node.output.0].shape
                {
                    return Err(invalid("provider output differs from prepared signature"));
                }
                slots[node.output.0] = Some(outputs.remove(0));
            }
            let outputs = self
                .outputs
                .iter()
                .map(|id| slots[id.0].as_ref().expect("validated output").clone())
                .collect::<Vec<_>>();
            callback(&outputs)
        };
        ctx.no_grad(execute)
    }

    pub(super) fn validate_bindings(
        &self,
        ctx: &ExecutionContext,
        feeds: &[&Tensor],
        parameters: &[&Parameter],
    ) -> MlResult<()> {
        // Validate the complete binding before writing any tensor or graph state.
        if ctx.id() != self.context {
            return Err(crate::ContextError::Mismatch.into());
        }
        if feeds.len() != self.program.feeds.len() || parameters.len() != self.parameter_ids.len() {
            return Err(invalid("binding count mismatch"));
        }
        for (feed, slot) in feeds.iter().zip(&self.program.feeds) {
            ctx.validate(feed)?;
            let spec = &self.program.slots[slot.0];
            if feed.shape()? != spec.shape {
                return Err(invalid("feed shape mismatch; prepare a separate plan"));
            }
            if let Source::Feed { requires_grad } = spec.source {
                if feed.as_variable()?.requires_grad()? != requires_grad {
                    return Err(invalid("feed gradient signature mismatch"));
                }
            }
        }
        for ((parameter, id), slot) in parameters
            .iter()
            .zip(&self.parameter_ids)
            .zip(&self.program.parameters)
        {
            ctx.validate(parameter.tensor())?;
            if parameter.id() != *id
                || parameter.tensor().shape()? != self.program.slots[slot.0].shape
            {
                return Err(invalid(
                    "parameter binding or shape changed; prepare a separate plan",
                ));
            }
        }
        Ok(())
    }

    fn run_training<T>(
        &self,
        ctx: &ExecutionContext,
        feeds: &[&Tensor],
        parameters: &[&Parameter],
        callback: impl FnOnce(&[Tensor]) -> MlResult<T>,
    ) -> MlResult<T> {
        use super::backward::PreparedRun;
        let mut values = vec![None; self.program.slots.len()];
        let mut ids = vec![None; values.len()];
        for (index, slot) in self.program.slots.iter().enumerate() {
            if let Source::Constant(value) = &slot.source {
                values[index] = Some(value.clone());
            }
        }
        for (tensor, slot) in feeds.iter().copied().zip(&self.program.feeds).chain(
            parameters
                .iter()
                .map(|p| p.tensor())
                .zip(&self.program.parameters),
        ) {
            values[slot.0] = Some(tensor.snapshot()?);
            ids[slot.0] = Some(tensor.id());
        }
        let mut saved = Vec::new();
        for (index, node) in self.program.instructions.iter().enumerate() {
            let inputs = node
                .inputs
                .iter()
                .map(|s| values[s.0].as_ref().expect("validated input").view())
                .collect::<Vec<_>>();
            let mut result = ctx
                .inner
                .operations
                .as_ref()
                .expect("validated provider")
                .execute(&node.operation, &inputs)?;
            if result.outputs.len() != 1
                || result.outputs[0].shape() != self.program.slots[node.output.0].shape
            {
                return Err(invalid("provider output differs from prepared signature"));
            }
            if let Some(vjp) = &self.backward.nodes[index].vjp {
                if result.saved.len() != vjp.saved_shapes.len()
                    || result
                        .saved
                        .iter()
                        .zip(&vjp.saved_shapes)
                        .any(|(value, shape)| value.shape() != shape)
                {
                    return Err(invalid(
                        "provider saved values differ from prepared contract",
                    ));
                }
                saved.push(result.saved);
            } else {
                saved.push(Vec::new());
            }
            values[node.output.0] = Some(result.outputs.remove(0));
        }
        let mut handles: Vec<Option<Tensor>> = vec![None; values.len()];
        let mut exports = Vec::new();
        let mut outputs = Vec::new();
        for slot in &self.outputs {
            if handles[slot.0].is_none() {
                let tensor =
                    ctx.allocate(Some(values[slot.0].as_ref().expect("output").clone()), None)?;
                if self.backward.tracked[slot.0] {
                    ctx.inner.state.borrow_mut().tracked.insert(tensor.id());
                }
                ids[slot.0] = Some(tensor.id());
                exports.push((tensor.id(), slot.0));
                handles[slot.0] = Some(tensor);
            }
            outputs.push(handles[slot.0].as_ref().unwrap().clone());
        }
        *ctx.inner.prepared_run.borrow_mut() = Some(PreparedRun {
            plan: self.backward.clone(),
            values: values.into_iter().map(Option::unwrap).collect(),
            saved,
            into: None,
            ids,
            exports,
            consumed: false,
            completed: false,
        });
        let _cleanup = Cleanup(ctx);
        callback(&outputs)
    }
}

pub(super) struct Cleanup<'a>(pub &'a ExecutionContext);
impl Drop for Cleanup<'_> {
    fn drop(&mut self) {
        if let Some(run) = self.0.inner.prepared_run.borrow_mut().take() {
            self.0.inner.state.borrow_mut().consumed.extend(
                run.exports
                    .into_iter()
                    .filter(|(_, s)| run.plan.tracked[*s])
                    .map(|(id, _)| id),
            );
        }
    }
}
