use super::*;

impl<T> Layer for Linear<T> {
    fn forward(&self, input: &Tensor<f32>) -> MlResult<Tensor<f32>> {
        todo!()
    }

    fn backward(&mut self, input: &Tensor<f32>, grad_output: &Tensor<f32>, learning_rate: f32) -> MlResult<Tensor<f32>> {
        todo!()
    }
}