use crate::MlResult;
use crate::tensor::Tensor;

pub mod activation;
pub mod conv;
pub mod pooling;
pub mod linear;

pub trait Layer {
    fn forward(&self, input: &Tensor<f32>) -> MlResult<Tensor<f32>>;
    fn backward(
        &mut self,
        input: &Tensor<f32>,
        grad_output: &Tensor<f32>,
        learning_rate: f32,
    ) -> MlResult<Tensor<f32>>;
}