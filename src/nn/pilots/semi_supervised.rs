//! Explicit-context Pi-model style semi-supervised pilot.

use crate::nn::{Layer, Linear, Parameter};
use crate::trainer::{ForwardModel, TrainableModel};
use crate::{ContextId, ExecutionContext, MlResult, Tensor, Variable};

#[derive(Debug)]
pub struct PiClassifier {
    context: ExecutionContext,
    linear: Linear,
}

impl PiClassifier {
    pub fn new(
        context: &ExecutionContext,
        inputs: usize,
        outputs: usize,
        ) -> MlResult<Self> {

        Ok(Self {
            context: context.clone(),
            linear: Linear::new(context, inputs, outputs, "pi_linear")?,
        })
    }



    pub fn predict(&self, input: &Tensor) -> MlResult<Tensor> {
        self.linear.predict(input)
    }
}

impl TrainableModel for PiClassifier {
    fn context_id(&self) -> ContextId {
        self.context.id()
    }
    fn parameters(&self) -> Vec<&Parameter> {
        self.linear.parameters()
    }
}

impl ForwardModel for PiClassifier {
    fn forward(&self, input: &Variable) -> MlResult<Variable> { self.linear.apply(input) }
}
