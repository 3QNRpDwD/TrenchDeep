use super::{invalid, plan::*};
use crate::{ExecutionContext, MlResult, Parameter, Tensor};

impl PreparedPlan {
    /// Bind current values and execute allocating replay. The callback may run
    /// backward and update an optimizer. Scope cleanup also runs after errors.
    /// Returned tensor handles own independent results across later executions.
    pub fn with_run<T>(
        &self,
        ctx: &ExecutionContext,
        feeds: &[&Tensor],
        parameters: &[&Parameter],
        callback: impl FnOnce(&[Tensor]) -> MlResult<T>,
    ) -> MlResult<T> {
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
        let execute = || {
            let mut slots: Vec<Option<Tensor>> = vec![None; self.program.slots.len()];
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
        match self.mode {
            PreparedMode::Training => ctx.with_training_scope(execute),
            PreparedMode::Inference => ctx.no_grad(execute),
        }
    }
}
