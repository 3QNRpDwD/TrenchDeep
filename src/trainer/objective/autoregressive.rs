use super::*;
use crate::nn::BigramLm;
#[derive(Debug, Clone, Copy)]
pub struct Autoregressive<L> { loss: L }
impl<L> Autoregressive<L> { pub fn new(loss: L) -> Self { Self { loss } } }
fn prepare_sequence(model: &BigramLm, sequence: &Variable) -> MlResult<(Variable, Tensor, usize)> {
    let ctx = sequence.tensor().execution_context()?;

        let shape = sequence.tensor().shape()?;
        let (batch, length) = match shape.as_slice() {
            [length, vocab] if *vocab == model.vocab() => (1, *length),
            [batch, length, vocab] if *vocab == model.vocab() => (*batch, *length),
            _ => {
                return Err(MlError::StringError(
                    "bigram input must have shape [sequence, vocab] or [batch, sequence, vocab]"
                        .into(),
                ));
            }
        };
        if batch == 0 || length < 2 {
            return Err(MlError::StringError(
                "bigram input requires a non-empty batch and sequence length >= 2".into(),
            ));
        }
        let data = sequence.tensor().to_vec()?;
        let positions = length - 1;
        let tokens = batch * positions;
        let mut inputs = Vec::with_capacity(tokens * model.vocab());
        let mut targets = Vec::with_capacity(tokens * model.vocab());
        for batch_index in 0..batch {
            let sequence_start = batch_index * length * model.vocab();
            for position in 0..positions {
                let input_start = sequence_start + position * model.vocab();
                let target_start = input_start + model.vocab();
                inputs.extend_from_slice(&data[input_start..input_start + model.vocab()]);
                targets.extend_from_slice(&data[target_start..target_start + model.vocab()]);
            }
        }
        let input = ctx.input(inputs, &[tokens, model.vocab()])?;
        let target = ctx.tensor(targets, &[tokens, model.vocab()])?;
        Ok((input, target, tokens))
}
impl<L: Loss> Objective<BigramLm> for Autoregressive<L> {
    type Batch = AutoregressiveBatch;
    const PARADIGM: &'static str = "autoregressive";
    fn forward_batch(&self, model: &mut BigramLm, batch: &Self::Batch, _: &TrainingStepContext) -> MlResult<TrainingOutput> {
        let (input, target, tokens) = prepare_sequence(model, &batch.sequences)?;
        let prediction = model.forward(&input)?;
        let loss = self.loss.compute(&input.tensor().execution_context()?, &prediction, &target)?;
        Ok(TrainingOutput { loss, prediction: Some(prediction), target: None, weight: tokens, tokens: Some(tokens), lambda: None })
    }
}
impl<L: Loss> PreparedObjective<BigramLm> for Autoregressive<L> {
    fn execution_batch(&self, model: &mut BigramLm, batch: &Self::Batch, _: &TrainingStepContext) -> MlResult<PreparedBatch> {
        let (input, target, tokens) = prepare_sequence(model, &batch.sequences)?;
        let inputs = ExecutionInputs::new("autoregressive").with("x", input.tensor().clone())?.with("target", target)?;
        let mut batch = PreparedBatch::new(inputs, tokens);
        batch.tokens = Some(tokens);
        Ok(batch)
    }
    fn forward_inputs(&self, model: &BigramLm, inputs: &ExecutionInputs) -> MlResult<ModelOutput> {
        let input = inputs.get("x")?;
        let prediction = model.forward(&input.as_variable()?)?;
        let loss = self.loss.compute(&input.execution_context()?, &prediction, inputs.get("target")?)?;
        Ok(ModelOutput::new(loss, Some(prediction)))
    }
}
