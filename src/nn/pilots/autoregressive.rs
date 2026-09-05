//! Explicit-context bigram language-model pilot.

use crate::loss::Reduction;
use crate::nn::ContextParameter;
use crate::trainer::{ContextAutoregressiveModel, ContextTrainableModel};
use crate::{ContextId, ContextVariable, ExecutionContext, MlError, MlResult};

#[derive(Debug)]
pub struct ContextBigramLm {
    context: ExecutionContext,
    weight: ContextParameter,
    vocab: usize,
}

impl ContextBigramLm {
    pub fn new(context: &ExecutionContext, vocab: usize) -> MlResult<Self> {
        if vocab == 0 {
            return Err(MlError::StringError("vocabulary must not be empty".into()));
        }
        let values = (0..vocab * vocab)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.2)
            .collect();
        Ok(Self {
            context: context.clone(),
            weight: ContextParameter::new(context.parameter(values, &[vocab, vocab])?),
            vocab,
        })
    }

    pub fn weight(&self) -> &ContextParameter { &self.weight }
}

impl ContextTrainableModel for ContextBigramLm {
    fn context_id(&self) -> ContextId { self.context.id() }
    fn parameters(&self) -> Vec<&ContextParameter> { vec![&self.weight] }
}

impl ContextAutoregressiveModel for ContextBigramLm {
    fn forward_loss(
        &mut self,
        sequence: &ContextVariable,
    ) -> MlResult<(ContextVariable, ContextVariable, usize)> {
        let shape = sequence.tensor().shape()?;
        let (batch, length) = match shape.as_slice() {
            [length, vocab] if *vocab == self.vocab => (1, *length),
            [batch, length, vocab] if *vocab == self.vocab => (*batch, *length),
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
        let mut inputs = Vec::with_capacity(tokens * self.vocab);
        let mut targets = Vec::with_capacity(tokens * self.vocab);
        for batch_index in 0..batch {
            let sequence_start = batch_index * length * self.vocab;
            for position in 0..positions {
                let input_start = sequence_start + position * self.vocab;
                let target_start = input_start + self.vocab;
                inputs.extend_from_slice(&data[input_start..input_start + self.vocab]);
                targets.extend_from_slice(&data[target_start..target_start + self.vocab]);
            }
        }
        let input = self.context.input(inputs, &[tokens, self.vocab])?;
        let target = self.context.tensor(targets, &[tokens, self.vocab])?;
        let logits = input.matmul(self.weight.tensor())?;
        let loss = logits.softmax_cross_entropy(&target, Reduction::Mean)?;
        Ok((logits, loss, tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{ContextAdam, ContextOptimizer};
    use crate::trainer::{
        ContextAutoregressiveDataLoader, ContextAutoregressiveDataset,
        ContextAutoregressiveTrainer, EpochSchedule,
    };

    fn sequence(context: &ExecutionContext, tokens: &[usize], vocab: usize) -> MlResult<ContextVariable> {
        let mut data = vec![0.0; tokens.len() * vocab];
        for (row, token) in tokens.iter().copied().enumerate() {
            data[row * vocab + token] = 1.0;
        }
        context.input(data, &[tokens.len(), vocab])
    }

    #[test]
    fn bigram_pilot_trains_end_to_end() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = ContextBigramLm::new(&context, 4)?;
        let samples = [
            sequence(&context, &[0, 1, 2, 3, 0], 4)?,
            sequence(&context, &[1, 2, 3, 0, 1], 4)?,
        ];
        let refs = samples.iter().collect::<Vec<_>>();
        let dataset = ContextAutoregressiveDataset::new(&context, &refs)?;
        let mut optimizer = ContextAdam::new(&context, 0.05, 0.9, 0.999, 1e-8)?;
        optimizer.register_all(&model.parameters())?;
        let result = ContextAutoregressiveTrainer::silent(&context).fit(
            &mut model,
            &mut optimizer,
            &dataset,
            EpochSchedule::new(5)?.with_tolerance(0.0),
        )?;
        assert!(result.final_loss.is_finite());
        assert_eq!(context.graph_stats()?.graph_nodes, 0);
        Ok(())
    }

    #[test]
    fn bigram_pilot_accepts_stacked_loader_batches() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = ContextBigramLm::new(&context, 4)?;
        let samples = [
            sequence(&context, &[0, 1, 2, 3, 0], 4)?,
            sequence(&context, &[1, 2, 3, 0, 1], 4)?,
        ];
        let refs = samples.iter().collect::<Vec<_>>();
        let dataset = ContextAutoregressiveDataset::new(&context, &refs)?;
        let mut loader = ContextAutoregressiveDataLoader::new(&context, dataset)?
            .batch_size(2)?
            .shuffle(false);
        let mut optimizer = ContextAdam::new(&context, 0.02, 0.9, 0.999, 1e-8)?;
        optimizer.register_all(&model.parameters())?;
        let result = ContextAutoregressiveTrainer::silent(&context).fit_loader(
            &mut model,
            &mut optimizer,
            &mut loader,
            EpochSchedule::new(2)?.with_tolerance(0.0),
        )?;
        assert!(result.final_loss.is_finite());
        assert_eq!(context.graph_stats()?.graph_nodes, 0);
        Ok(())
    }

    #[test]
    fn autoregressive_padding_is_rejected_until_it_has_loss_semantics() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = ContextBigramLm::new(&context, 4)?;
        let sample = sequence(&context, &[0, 1, 2], 4)?;
        let refs = [&sample];
        let dataset = ContextAutoregressiveDataset::new(&context, &refs)?.with_pad_token_id(0);
        let mut optimizer = ContextAdam::new(&context, 0.02, 0.9, 0.999, 1e-8)?;
        optimizer.register_all(&model.parameters())?;
        assert!(ContextAutoregressiveTrainer::silent(&context)
            .fit(&mut model, &mut optimizer, &dataset, EpochSchedule::new(1)?)
            .is_err());
        Ok(())
    }
}
