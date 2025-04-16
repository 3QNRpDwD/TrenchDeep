use super::*;

impl Activation<f32> for Softmax {
    fn new() -> MlResult<Self> where Self: Sized {
        unimplemented!()
    }
    fn apply(&mut self, input: &Arc<Variable<f32>>) -> MlResult<Variable<f32>> {
        unimplemented!()
    }
}

impl Function<f32> for Softmax {
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