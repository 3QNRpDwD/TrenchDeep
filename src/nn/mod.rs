pub mod activation;
pub mod conv;
pub mod pooling;
pub mod linear;

use crate::backend::Backend;
use crate::backend::CpuBackend;
use crate::backend::Device;
use crate::tensor::operators::Function;
use crate::tensor::AutogradFunction;
use crate::tensor::{Tensor, TensorBase, Variable};
use crate::MlResult;
use std::fmt::Debug;
use std::sync::Arc;

pub trait Layer {
    fn forward(&self, input: &Tensor<f32>) -> MlResult<Tensor<f32>>;
    fn backward(
        &mut self,
        input: &Tensor<f32>,
        grad_output: &Tensor<f32>,
        learning_rate: f32,
    ) -> MlResult<Tensor<f32>>;
}

pub struct Linear<Type>    { operators: Arc<dyn Function<Type>> }
pub struct Conv<Type>      { operators: Arc<dyn Function<Type>> }
pub struct Pooling<Type>   { operators: Arc<dyn Function<Type>> }

