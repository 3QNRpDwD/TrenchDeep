pub mod sigmoid;
pub mod tanh;
pub mod relu;
mod softmax;

use super::*;

pub trait Activation<Type: Debug + Clone>: Layer {
    fn new() -> MlResult<Self> where Self: Sized {
        unimplemented!()
    }
    fn apply(&self, input: &Arc<Variable<Type>>) -> Arc<Variable<Type>> {
        unimplemented!()
    }
}


impl<T: Activation<f32>> Layer for T {
    fn forward(&self, input: &Tensor<f32>) -> MlResult<Tensor<f32>> {
        unimplemented!()
    }

    fn backward(
        &mut self,
        input: &Tensor<f32>,
        grad_output: &Tensor<f32>,
        _learning_rate: f32,
    ) -> MlResult<Tensor<f32>> {
        unimplemented!()
    }
}


pub struct Sigmoid {
    backend: Arc<dyn Backend>
}
pub struct Tanh {
    backend: Arc<dyn Backend>
}
pub struct Relu {
    backend: Arc<dyn Backend>
}

pub struct Softmax {
    backend: Arc<dyn Backend>
}