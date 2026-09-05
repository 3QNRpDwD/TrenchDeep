//! Small explicit-context models used as P1 migration and E2E benchmark pilots.

mod autoregressive;
mod diffusion;
mod reinforcement;
mod semi_supervised;

pub use autoregressive::ContextBigramLm;
pub use diffusion::ContextDiffusionPilot;
pub use reinforcement::{ContextLinearPolicy, ContextTwoArmedBandit};
pub use semi_supervised::ContextPiClassifier;

use crate::loss::Reduction;
use crate::trainer::{ContextSupervisedModel, ContextTrainableModel};
use crate::{ContextId, ContextTensor, ContextVariable, ExecutionContext, MlResult};

use super::{
    ContextActivation, ContextActivationKind, ContextLayer, ContextLinear, ContextParameter,
    ContextSequential,
};

#[derive(Debug)]
pub struct ContextLinearRegression {
    context: ExecutionContext,
    layer: ContextLinear,
}

impl ContextLinearRegression {
    pub fn new(context: &ExecutionContext, inputs: usize, outputs: usize) -> MlResult<Self> {
        Ok(Self {
            context: context.clone(),
            layer: ContextLinear::new(context, inputs, outputs, "linear")?,
        })
    }

    pub fn predict(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.layer.predict(input)
    }

    pub fn layer(&self) -> &ContextLinear { &self.layer }
}

impl ContextTrainableModel for ContextLinearRegression {
    fn context_id(&self) -> ContextId { self.context.id() }
    fn parameters(&self) -> Vec<&ContextParameter> { self.layer.parameters() }
}

impl ContextSupervisedModel for ContextLinearRegression {
    fn forward_loss(&mut self, input: &ContextVariable, target: &ContextTensor) -> MlResult<(ContextVariable, ContextVariable)> {
        let prediction = self.layer.apply(input)?;
        let loss = prediction.mse_loss(target, Reduction::Mean)?;
        Ok((prediction, loss))
    }
}

#[derive(Debug)]
pub struct ContextMlp {
    context: ExecutionContext,
    network: ContextSequential,
}

impl ContextMlp {
    pub fn new(
        context: &ExecutionContext,
        inputs: usize,
        hidden: usize,
        outputs: usize,
    ) -> MlResult<Self> {
        let mut network = ContextSequential::new(context, "MLP");
        network.push(Box::new(ContextLinear::new(context, inputs, hidden, "linear1")?))?;
        network.push(Box::new(ContextActivation::new(
            context, ContextActivationKind::Sigmoid, "hidden_act",
        )))?;
        network.push(Box::new(ContextLinear::new(context, hidden, outputs, "linear2")?))?;
        Ok(Self { context: context.clone(), network })
    }

    pub fn logits(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.network.predict(input)
    }

    pub fn predict(&self, input: &ContextTensor) -> MlResult<ContextTensor> {
        self.context.no_grad(|| {
            let logits = self.network.predict(input)?;
            self.context.softmax(&logits, logits.shape()?.len().saturating_sub(1))
        })
    }

    pub fn network(&self) -> &ContextSequential { &self.network }
    pub fn network_mut(&mut self) -> &mut ContextSequential { &mut self.network }
}

impl ContextTrainableModel for ContextMlp {
    fn context_id(&self) -> ContextId { self.context.id() }
    fn parameters(&self) -> Vec<&ContextParameter> { self.network.parameters() }
}

impl ContextSupervisedModel for ContextMlp {
    fn forward_loss(&mut self, input: &ContextVariable, target: &ContextTensor) -> MlResult<(ContextVariable, ContextVariable)> {
        let logits = self.network.apply(input)?;
        let loss = logits.softmax_cross_entropy(target, Reduction::Mean)?;
        Ok((logits, loss))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{ContextAdam, ContextOptimizer};
    use crate::trainer::{ContextSupervisedDataset, ContextSupervisedTrainer, EpochSchedule};

    #[test]
    fn mlp_pilot_trains_end_to_end_and_predicts_probabilities() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = ContextMlp::new(&context, 2, 4, 2)?;
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
        let dataset = ContextSupervisedDataset::new(&context, &input_refs, &target_refs)?;
        let mut optimizer = ContextAdam::new(&context, 0.05, 0.9, 0.999, 1e-8)?;
        optimizer.register_all(&model.parameters())?;

        let result = ContextSupervisedTrainer::new(&context).fit(
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
