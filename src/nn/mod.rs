use crate::MlResult;
use crate::tensor::Tensor;

pub mod activation;
mod Convolutional;
mod Recurrent;



pub trait Layer {
    fn forward_pass(&self, input: &[f32]) -> MlResult<Tensor<f32>>;
    fn backward_pass(
        &mut self,
        input: &Tensor<f32>,
        grad_output: &Tensor<f32>,
        learning_rate: f32,
    ) -> MlResult<Tensor<f32>>;
}