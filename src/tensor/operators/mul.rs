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
            _ => {
                let result = targets[0].with_data(|data1| {
                    targets[1].with_data(|data2| {
                        self.backend().multiply(data1, data2)
                    }).ok_or(MlError::from(TensorError::TensorNotFound))
                }).ok_or(MlError::from(TensorError::TensorNotFound))??;

                targets[0].with_shape(|shape| {
                    Ok(vec![Tensor::from_vec(result, shape)?])
                }).ok_or(MlError::from(TensorError::TensorNotFound))?
            }
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