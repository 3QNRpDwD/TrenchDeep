//! Named runtime inputs and an explicit host topology variant, shared by models.
use super::{PreparedMode, PreparedPlan, invalid};
use crate::{ExecutionContext, MlResult, Parameter, Tensor};

#[derive(Debug, Clone)]
pub struct ExecutionInputs {
    variant: String,
    tensors: Vec<(String, Tensor)>,
}
impl ExecutionInputs {
    pub fn new(variant: impl Into<String>) -> Self {
        Self {
            variant: variant.into(),
            tensors: Vec::new(),
        }
    }
    pub fn with(mut self, name: impl Into<String>, tensor: Tensor) -> MlResult<Self> {
        let name = name.into();
        if self.tensors.iter().any(|(n, _)| n == &name) {
            return Err(invalid("duplicate input name"));
        }
        self.tensors.push((name, tensor));
        Ok(self)
    }
    pub fn get(&self, name: &str) -> MlResult<&Tensor> {
        self.tensors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t)
            .ok_or_else(|| invalid(format!("missing input {name}")))
    }
    pub fn variant(&self) -> &str {
        &self.variant
    }
    pub(super) fn signature(&self) -> (String, Vec<String>) {
        (
            self.variant.clone(),
            self.tensors.iter().map(|(n, _)| n.clone()).collect(),
        )
    }
    // Names are independent feed slots even when the example tensors alias.
    // Only preparation needs unique handles; replay binds the original tensors.
    fn recording_inputs(&self, ctx: &ExecutionContext) -> MlResult<Self> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Self::new(self.variant.clone());
        for (name, tensor) in &self.tensors {
            ctx.validate(tensor)?;
            let tensor = if seen.insert(tensor.id()) {
                tensor.clone()
            } else {
                let requires_grad = if tensor.as_variable()?.requires_grad()? {
                    crate::RequiresGrad::Yes
                } else {
                    crate::RequiresGrad::No
                };
                ctx.variable(vec![0.0; tensor.numel()?], &tensor.shape()?, requires_grad)?
                    .tensor()
                    .clone()
            };
            result.tensors.push((name.clone(), tensor));
        }
        Ok(result)
    }
    pub(super) fn bindings(&self) -> Vec<&Tensor> {
        self.tensors.iter().map(|(_, t)| t).collect()
    }
}
impl ExecutionContext {
    /// Select backward roots by index in the exported tensor list. Exported
    /// values and retained gradients remain available independently of roots.
    pub fn prepare_forward_with_roots(
        &self,
        inputs: &ExecutionInputs,
        parameters: &[&Parameter],
        mode: PreparedMode,
        backward_outputs: &[usize],
        describe: impl FnOnce(&ExecutionInputs) -> MlResult<Vec<Tensor>>,
    ) -> MlResult<PreparedPlan> {
        let recording = inputs.recording_inputs(self)?;
        let mut plan = self.prepare_recorded_with_roots(
            &recording.bindings(),
            parameters,
            mode,
            Some(backward_outputs),
            || describe(&recording),
        )?;
        plan.input_signature = Some(inputs.signature());
        Ok(plan)
    }
    /// Describe any fixed-topology forward with named runtime inputs. Shape-only
    /// execution blocks tensor data reads and runtime mutations through this API.
    /// The caller must keep host branches/configuration fixed for the variant and
    /// avoid external side effects: arbitrary Rust closures are not sandboxed.
    pub fn prepare_forward(
        &self,
        inputs: &ExecutionInputs,
        parameters: &[&Parameter],
        mode: PreparedMode,
        describe: impl FnOnce(&ExecutionInputs) -> MlResult<Vec<Tensor>>,
    ) -> MlResult<PreparedPlan> {
        let recording = inputs.recording_inputs(self)?;
        let mut plan = self.prepare_recorded(&recording.bindings(), parameters, mode, || {
            describe(&recording)
        })?;
        plan.input_signature = Some(inputs.signature());
        Ok(plan)
    }
}
impl PreparedPlan {
    pub fn with_inputs<T>(
        &self,
        ctx: &ExecutionContext,
        inputs: &ExecutionInputs,
        parameters: &[&Parameter],
        callback: impl FnOnce(&[Tensor]) -> MlResult<T>,
    ) -> MlResult<T> {
        if self.input_signature.as_ref() != Some(&inputs.signature()) {
            return Err(invalid(
                "input names/order or topology variant changed; prepare a separate plan",
            ));
        }
        self.run_bound(ctx, &inputs.bindings(), parameters, callback)
    }
}

impl ExecutionInputs {
    pub(crate) fn tensor_signature(&self) -> MlResult<Vec<(String, Vec<usize>, bool)>> {
        self.tensors
            .iter()
            .map(|(name, t)| Ok((name.clone(), t.shape()?, t.as_variable()?.requires_grad()?)))
            .collect()
    }
}
