use super::*;

impl Function<f32> for Neg {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Negates each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the negation of tensor_element
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(targets[0].data().iter().map(|&x| -x).collect(), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(grad.data().iter().map(|&x| -x).collect(), grad.shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl std::ops::Neg for Tensor<f32> {
    type Output = Tensor<f32>;

    fn neg(self) -> Self::Output {
        Neg::new().unwrap().forward(&[&self]).unwrap().remove(0)
    }
}

impl std::ops::Neg for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn neg(self) -> Self::Output {
        Neg::new().unwrap().forward(&[self]).unwrap().remove(0)
    }
}