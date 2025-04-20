use super::*;

impl Function<f32> for Sum {
    fn new() -> MlResult<Self> {
        todo!()
    }

    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        todo!()
    }

    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        todo!()
    }
}