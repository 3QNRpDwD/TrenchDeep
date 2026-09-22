//! Explicit-context bigram language-model pilot.

use crate::nn::Parameter;
use crate::trainer::{ForwardModel, TrainableModel};
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

impl BigramLm { pub fn vocab(&self) -> usize { self.vocab } }
impl ForwardModel for BigramLm {
    fn forward(&self, input: &Variable) -> MlResult<Variable> { input.matmul(self.weight.tensor()) }
}
