use super::*;

impl Function for Mul {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Multiplies two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise multiplication
    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        match targets[0].chk_shape(&targets[1]) {
            Err(e) => Err(e),
            _ => Ok(vec![Tensor::from_vec(self.backend().multiply(targets[0].data().as_slice(), targets[1].data().as_slice()), targets[0].shape().as_slice())?])
        }
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[Tensor], grad: Tensor) -> MlResult<Vec<Tensor>> {
        Ok(vec![
            self.forward(&[grad, targets[1]])?.remove(0),
            self.forward(&[grad, targets[0]])?.remove(0)
        ])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}


/// Multiply trait implementation for owned tensors
///
/// # Arguments
/// * `other` - The tensor to multiply with self
///
/// # Returns
/// A new tensor containing the element-wise product (Hadamard product)
///
/// # Note
/// * This performs element-wise multiplication, not matrix multiplication
/// * For matrix multiplication, use `matmul()` instead
impl std::ops::Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Self::Output {
        Mul::new().unwrap().forward(&[self, other]).unwrap().remove(0)
    }
}

impl std::ops::Mul<&Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Self::Output {
        Mul::new().unwrap().forward(&[self, *other]).unwrap().remove(0)
    }
}

impl std::ops::Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Self::Output {
        Mul::new().unwrap().forward(&[*self, *other]).unwrap().remove(0)
    }
}

impl std::ops::Mul<Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Self::Output {
        Mul::new().unwrap().forward(&[*self, other]).unwrap().remove(0)
    }
}