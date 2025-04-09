use crate::MlResult;
use crate::tensor::Tensor;

pub mod activation;

pub trait Layer {
    fn forward(&self, input: &[f32]) -> MlResult<Tensor<f32>>;
    fn backward(
        &mut self,
        input: &Tensor<f32>,
        grad_output: &Tensor<f32>,
        learning_rate: f32,
    ) -> MlResult<Tensor<f32>>;
}