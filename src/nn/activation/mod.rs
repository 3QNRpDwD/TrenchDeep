mod functions;

use std::fmt::Debug;
use std::sync::Arc;
use crate::MlResult;
use crate::nn::Layer;
use crate::tensor::operators::Function;
use crate::tensor::{Tensor, TensorBase};

pub trait Activation<T: Debug + Clone>: Layer {
    fn act_forward(&self, input: &Tensor<f32>) -> MlResult<Tensor<f32>>;
    fn act_backward(&self, input: &Tensor<f32>, grad_output: &Tensor<f32>) -> MlResult<Tensor<f32>>;
}


impl<T: Activation<f32>> Layer for T {
    fn forward_pass(&self, input: &Tensor<f32>) -> MlResult<Tensor<f32>> {
        Self::act_forward(self, input)
    }

    fn backward_pass(
        &mut self,
        input: &Tensor<f32>,
        grad_output: &Tensor<f32>,
        _learning_rate: f32,
    ) -> MlResult<Tensor<f32>> {
        Self::act_backward(self, input, grad_output)
    }
}


pub struct Sigmoid {
    exp: Arc<dyn Function<f32>>,

    #[cfg(all(feature = "enableBackpropagation"))]
    mul: Arc<dyn Function<f32>>,
    #[cfg(all(feature = "enableBackpropagation"))]
    sub: Arc<dyn Function<f32>>,
}
pub struct Tanh {
    exp: Arc<dyn Function<f32>>,
    sub: Arc<dyn Function<f32>>,
    div: Arc<dyn Function<f32>>,
    add: Arc<dyn Function<f32>>,

    #[cfg(all(feature = "enableBackpropagation"))]
    mul: Arc<dyn Function<f32>>,
}
pub struct Relu {
    #[cfg(all(feature = "enableBackpropagation"))]
    mul: Arc<dyn Function<f32>>
}