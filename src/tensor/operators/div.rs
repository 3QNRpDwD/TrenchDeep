use super::*;

impl Function<f32> for Div {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Divides two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to divide the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise division
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        match targets[0].chk_shape(targets[1]) {
            Err(e) => Err(e),
            _ => Ok(vec![Tensor::<f32>::from_vec(self.backend().div(targets[0].data(), targets[1].data()), targets[0].shape())?])
        }
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let x1 = targets[1];

        Ok(vec![
            self.forward(&[grad, x1])?.remove(0), // grad / x2
            grad * self.forward(&[&-targets[0], &(x1 * x1)])?.remove(0) // grad * (-x0 / x1^2)
        ])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

/// Divide trait implementation for owned tensors
///
/// # Arguments
/// * `other` - The tensor to divide self by
///
/// # Returns
/// A new tensor containing the element-wise quotient
impl std::ops::Div<Tensor<f32>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn div(self, other: Tensor<f32>) -> Self::Output {
        Div::new().unwrap().forward(&[&self, &other]).unwrap().remove(0)
    }
}

impl std::ops::Div<&Tensor<f32>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn div(self, other: &Tensor<f32>) -> Self::Output {
        Div::new().unwrap().forward(&[&self, other]).unwrap().remove(0)
    }
}

impl std::ops::Div<&Tensor<f32>> for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn div(self, other: &Tensor<f32>) -> Self::Output {
        Div::new().unwrap().forward(&[self, other]).unwrap().remove(0)
    }
}

impl std::ops::Div<Tensor<f32>> for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn div(self, other: Tensor<f32>) -> Self::Output {
        Div::new().unwrap().forward(&[self, &other]).unwrap().remove(0)
    }
}