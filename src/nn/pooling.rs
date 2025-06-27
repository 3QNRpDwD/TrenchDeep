use super::*;

impl Layer for Pooling {
    #[cfg(feature = "enableBackpropagation")]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        todo!()
    }
    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        todo!()
    }
    fn params(&self) -> Vec<&dyn Parameter> {
        todo!()
    }
    fn label(&self) -> &str {
        &self.label
    }
}