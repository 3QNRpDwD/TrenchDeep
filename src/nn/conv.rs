use super::*;

impl<T> Layer for Conv<T> {
    fn forward(&self, _input: &Tensor<f32>) -> MlResult<Tensor<f32>> {
        todo!()
    }

    fn backward(&mut self, _input: &Tensor<f32>, _grad_output: &Tensor<f32>, _learning_rate: f32) -> MlResult<Tensor<f32>> {
        todo!()
    }
}