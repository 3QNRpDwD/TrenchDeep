//! Small explicit-context models used as P1 migration and E2E benchmark pilots.

mod autoregressive;
mod diffusion;
mod reinforcement;
mod semi_supervised;

pub use autoregressive::BigramLm;
pub use diffusion::DiffusionPilot;
pub use reinforcement::{LinearPolicy, TwoArmedBandit};
pub use semi_supervised::PiClassifier;

use crate::loss::Reduction;
use crate::trainer::{SupervisedModel, TrainableModel};
use crate::{ContextId, Tensor, Variable, ExecutionContext, MlResult};

use super::{
    Activation, ActivationKind, Layer, Linear, Parameter,
    Sequential,
};

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

    pub fn layer(&self) -> &Linear { &self.layer }
}

impl TrainableModel for LinearRegression {
    fn context_id(&self) -> ContextId { self.context.id() }
    fn parameters(&self) -> Vec<&Parameter> { self.layer.parameters() }
}

impl SupervisedModel for LinearRegression {
    fn forward_loss(&mut self, input: &Variable, target: &Tensor) -> MlResult<(Variable, Variable)> {
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
            context, ActivationKind::Sigmoid, "hidden_act",
        )))?;
        network.push(Box::new(Linear::new(context, hidden, outputs, "linear2")?))?;
        Ok(Self { context: context.clone(), network })
    }

    pub fn logits(&self, input: &Tensor) -> MlResult<Tensor> {
        self.network.predict(input)
    }

    pub fn predict(&self, input: &Tensor) -> MlResult<Tensor> {
        self.context.no_grad(|| {
            let logits = self.network.predict(input)?;
            self.context.softmax(&logits, logits.shape()?.len().saturating_sub(1))
        })
    }

    pub fn network(&self) -> &Sequential { &self.network }
    pub fn network_mut(&mut self) -> &mut Sequential { &mut self.network }
}

impl TrainableModel for Mlp {
    fn context_id(&self) -> ContextId { self.context.id() }
    fn parameters(&self) -> Vec<&Parameter> { self.network.parameters() }
}

impl SupervisedModel for Mlp {
    fn forward_loss(&mut self, input: &Variable, target: &Tensor) -> MlResult<(Variable, Variable)> {
        let logits = self.network.apply(input)?;
        let loss = logits.softmax_cross_entropy(target, Reduction::Mean)?;
        Ok((logits, loss))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{Adam, Optimizer};
    use crate::trainer::{SupervisedDataset, SupervisedTrainer, EpochSchedule};

    #[test]
    fn mlp_pilot_trains_end_to_end_and_predicts_probabilities() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = Mlp::new(&context, 2, 4, 2)?;
        let inputs = [
            context.input(vec![0.0, 0.0], &[1, 2])?,
            context.input(vec![1.0, 1.0], &[1, 2])?,
        ];
        let targets = [
            context.tensor(vec![1.0, 0.0], &[1, 2])?,
            context.tensor(vec![0.0, 1.0], &[1, 2])?,
        ];
        let input_refs: Vec<_> = inputs.iter().collect();
        let target_refs: Vec<_> = targets.iter().collect();
        let dataset = SupervisedDataset::new(&context, &input_refs, &target_refs)?;
        let mut optimizer = Adam::new(&context, 0.05, 0.9, 0.999, 1e-8)?;
        optimizer.register_all(&model.parameters())?;

        let result = SupervisedTrainer::new(&context).fit(
            &mut model, &mut optimizer, &dataset, EpochSchedule::new(20)?,
        )?;
        assert!(result.final_loss.is_finite());
        assert_eq!(result.units_completed, 20);
        assert_eq!(context.graph_stats()?.graph_nodes, 0);

        let probabilities = model.predict(inputs[0].tensor())?.to_vec()?;
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        Ok(())
    }
}
