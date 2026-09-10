//! Explicit-context bigram language-model pilot.

use crate::loss::Reduction;
use crate::nn::Parameter;
use crate::trainer::{AutoregressiveModel, TrainableModel};
use crate::{ContextId, ExecutionContext, MlError, MlResult, Variable};

#[derive(Debug)]
pub struct BigramLm {
    context: ExecutionContext,
    weight: Parameter,
    vocab: usize,
}

impl BigramLm {
    pub fn new(context: &ExecutionContext, vocab: usize) -> MlResult<Self> {
        if vocab == 0 {
            return Err(MlError::StringError("vocabulary must not be empty".into()));
        }
        let values = context.initialization_uniform(
            vocab
                .checked_mul(vocab)
                .ok_or_else(|| MlError::StringError("vocabulary dimension overflow".into()))?,
            0.1,
        )?;
        Ok(Self {
            context: context.clone(),
            weight: context.parameter(values, &[vocab, vocab])?,
            vocab,
        })
    }

    pub fn weight(&self) -> &Parameter {
        &self.weight
    }
}

impl TrainableModel for BigramLm {
    fn context_id(&self) -> ContextId {
        self.context.id()
    }
    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight]
    }
}

impl AutoregressiveModel for BigramLm {
    fn forward_loss(&mut self, sequence: &Variable) -> MlResult<(Variable, Variable, usize)> {
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
