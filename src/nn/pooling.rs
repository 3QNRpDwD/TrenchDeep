use super::*;

impl Layer<f32> for Pooling<f32> {
    fn new() -> MlResult<Self> {
        todo!()
    }

    fn apply(&self, input: &Arc<Variable<f32>>) -> MlResult<Arc<Variable<f32>>> {
        todo!()
    }

    fn forward(&self, _input: &Tensor<f32>) -> MlResult<Tensor<f32>> {
        todo!()
    }
}