pub mod activation;
pub mod conv;
pub mod pooling;
pub mod linear;

use std::fmt::Debug;
use std::sync::Arc;
use crate::backend::Backend;
use crate::backend::CpuBackend;
use crate::backend::Device;
use crate::MlResult;
use crate::tensor::{Tensor, TensorBase, Variable};
use crate::tensor::operators::Function;

pub trait Layer {
    fn forward(&self, input: &Tensor<f32>) -> MlResult<Tensor<f32>>;
    fn backward(
        &mut self,
        input: &Tensor<f32>,
        grad_output: &Tensor<f32>,
        learning_rate: f32,
    ) -> MlResult<Tensor<f32>>;
}