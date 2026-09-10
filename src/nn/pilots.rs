//! Small explicit-context models used as P1 migration and E2E benchmark pilots.

mod autoregressive;
mod reinforcement;
mod semi_supervised;

pub use autoregressive::BigramLm;
pub use reinforcement::{LinearPolicy, TwoArmedBandit};
pub use semi_supervised::PiClassifier;

use crate::loss::Reduction;
use crate::trainer::{SupervisedModel, TrainableModel};
use crate::{ContextId, ExecutionContext, MlResult, Tensor, Variable};

use super::{Activation, ActivationKind, Layer, Linear, Parameter, Sequential};

#[derive(Debug)]
pub struct LinearRegression {
    context: ExecutionContext,
    layer: Linear,
}

impl LinearRegression {
    pub fn new(context: &ExecutionContext, inputs: usize, outputs: usize) -> MlResult<Self> {
        Ok(Self {
            context: context.clone(),
            layer: Linear::new(context, inputs, outputs, "linear")?,
        })
    }

    pub fn predict(&self, input: &Tensor) -> MlResult<Tensor> {
        self.layer.predict(input)
    }

    pub fn layer(&self) -> &Linear {
        &self.layer
    }
}

impl TrainableModel for LinearRegression {
    fn context_id(&self) -> ContextId {
        self.context.id()
    }
    fn parameters(&self) -> Vec<&Parameter> {
        self.layer.parameters()
    }
}

impl SupervisedModel for LinearRegression {
    fn forward_loss(
        &mut self,
        input: &Variable,
        target: &Tensor,
    ) -> MlResult<(Variable, Variable)> {
        let prediction = self.layer.apply(input)?;
        let loss = prediction.mse_loss(target, Reduction::Mean)?;
        Ok((prediction, loss))
    }
}

#[derive(Debug)]
pub struct Mlp {
    context: ExecutionContext,
    network: Sequential,
}

impl Mlp {
    pub fn new(
        context: &ExecutionContext,
        inputs: usize,
        hidden: usize,
        outputs: usize,
    ) -> MlResult<Self> {
        let mut network = Sequential::new(context, "MLP");
        network.push(Box::new(Linear::new(context, inputs, hidden, "linear1")?))?;
        network.push(Box::new(Activation::new(
            context,
            ActivationKind::Sigmoid,
            "hidden_act",
        )))?;
        network.push(Box::new(Linear::new(context, hidden, outputs, "linear2")?))?;
        Ok(Self {
            context: context.clone(),
            network,
        })
    }

    pub fn logits(&self, input: &Tensor) -> MlResult<Tensor> {
        self.network.predict(input)
    }

    pub fn predict(&self, input: &Tensor) -> MlResult<Tensor> {
        self.context.no_grad(|| {
            let logits = self.network.predict(input)?;
            self.context
                .softmax(&logits, logits.shape()?.len().saturating_sub(1))
        })
    }

    pub fn network(&self) -> &Sequential {
        &self.network
    }
    pub fn network_mut(&mut self) -> &mut Sequential {
        &mut self.network
    }
}

impl TrainableModel for Mlp {
    fn context_id(&self) -> ContextId {
        self.context.id()
    }
    fn parameters(&self) -> Vec<&Parameter> {
        self.network.parameters()
    }
}

impl SupervisedModel for Mlp {
    fn forward_loss(
        &mut self,
        input: &Variable,
        target: &Tensor,
    ) -> MlResult<(Variable, Variable)> {
        let logits = self.network.apply(input)?;
        let loss = logits.softmax_cross_entropy(target, Reduction::Mean)?;
        Ok((logits, loss))
    }
}
