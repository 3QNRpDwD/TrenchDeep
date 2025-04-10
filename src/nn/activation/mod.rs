pub mod sigmoid;
pub mod tanh;
pub mod relu;
mod softmax;

use std::fmt::Debug;
use std::sync::Arc;
use crate::MlResult;
use crate::nn::Layer;
use crate::tensor::{Tensor, TensorBase, Variable};
use crate::tensor::operators::{Add, Div, Exp, Function, Mul, Sub};

pub trait Activation<Type: Debug + Clone>: Layer {
    fn new() -> MlResult<Self> where Self: Sized {
        unimplemented!()
    }
    fn activation(&self, input: &Arc<Variable<Type>>) -> Arc<Variable<Type>> {
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