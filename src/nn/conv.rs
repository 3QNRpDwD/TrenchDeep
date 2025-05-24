use super::*;

impl Layer<f32> for Conv<f32> {
    fn new() -> MlResult<Self> {
        todo!()
    }

    fn parms(&self) -> MlResult<&[String]> {
        todo!()
    }

    fn set_parms(&self, name: String, parm: &Arc<dyn Parameter>) -> MlResult<&HashSet<Arc<dyn Parameter>>> {
        todo!()
    }

    fn get_parms(&self, name: String) -> MlResult<Arc<dyn Parameter>> {
        todo!()
    }

    fn apply(&self, input: &Arc<dyn Parameter>) -> MlResult<Arc<dyn Parameter>> {
        todo!()
    }

    fn forward(&self, _input: &Tensor<f32>) -> MlResult<Tensor<f32>> {
        todo!()
    }
}