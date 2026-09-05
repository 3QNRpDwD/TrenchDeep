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
        if shape.len() != 2 || shape[1] != self.vocab || shape[0] < 2 {
            return Err(MlError::StringError(
                "bigram input must have shape [sequence>=2, vocab]".into(),
            ));
        }
        let data = sequence.tensor().to_vec()?;
        let tokens = shape[0] - 1;
        let mut last_logits = None;
        let mut total_loss = None;
        for index in 0..tokens {
            let start = index * self.vocab;
            let next = start + self.vocab;
            let input = self.context.input(data[start..next].to_vec(), &[1, self.vocab])?;
            let target = self.context.tensor(
                data[next..next + self.vocab].to_vec(),
                &[1, self.vocab],
            )?;
            let logits = self.context.matmul_variable(&input, self.weight.variable())?;
            let loss = self.context.softmax_cross_entropy_variable(
                &logits,
                &target,
                Reduction::Mean,
            )?;
            total_loss = Some(match total_loss {
                Some(accumulated) => self.context.add_variable(&accumulated, &loss)?,
                None => loss,
            });
            last_logits = Some(logits);
        }
        let scale = self.context.input(vec![1.0 / tokens as f32], &[])?;
        let mean_loss = self.context.mul_variable(
            &total_loss.ok_or_else(|| MlError::StringError("empty bigram loss".into()))?,
            &scale,
        )?;
        Ok((
            last_logits.ok_or_else(|| MlError::StringError("empty bigram logits".into()))?,
            mean_loss,
            tokens,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{ContextAdam, ContextOptimizer};
    use crate::trainer::{ContextAutoregressiveDataset, ContextAutoregressiveTrainer, EpochSchedule};

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
}
