use crate::tensor::operators::Reshape;
use super::*;

impl Debug for Reshape {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        println!("Reshape");
        Ok(())
    }
}

impl Layer for Reshape {
    #[cfg(all(feature = "enableBackward"))]
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
        todo!()
    }
}