use super::*;

impl Function for Neg {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Negates each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the negation of tensor_element
    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        let result = targets[0].with_data(|data| {
            data.iter().map(|&x| -x).collect::<Vec<f32>>()
        }).ok_or(MlError::from(TensorError::TensorNotFound))?;

        targets[0].with_shape(|shape| {
            Ok(vec![Tensor::from_vec(result, shape)?])
        }).ok_or(MlError::from(TensorError::TensorNotFound))?
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _: &[Tensor], grad: Tensor) -> MlResult<Vec<Tensor>> {
        let result = grad.with_data(|data| {
            data.iter().map(|&x| -x).collect::<Vec<f32>>()
        }).ok_or(MlError::from(TensorError::TensorNotFound))?;

        grad.with_shape(|shape| {
            Ok(vec![Tensor::from_vec(result, shape)?])
        }).ok_or(MlError::from(TensorError::TensorNotFound))?
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl std::ops::Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        Neg::new().unwrap().forward(&[self]).unwrap().remove(0)
    }
}

impl std::ops::Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        Neg::new().unwrap().forward(&[*self]).unwrap().remove(0)
    }
}
