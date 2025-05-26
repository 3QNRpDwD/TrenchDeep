use super::*;

impl Function for Div {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Divides two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to divide the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise division
    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        targets[0].chk_shape(&targets[1])?;
        let result = targets[0].with_data(|data0| {
            targets[1].with_data(|data1| {
                self.backend().div(data0, data1)
            })
        });
        let shape = targets[0].with_shape(|s| s.to_vec());
        Ok(vec![Tensor::from_vec(result, &shape)?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[Tensor], grad: Tensor) -> MlResult<Vec<Tensor>> {
        let x0 = targets[0];
        let x1 = targets[1];
        let grad_x0 = grad / x1;
        let grad_x1 = grad * (-(x0) / (x1 * x1));
        Ok(vec![grad_x0, grad_x1])
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
impl std::ops::Div<Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Self::Output {
        Div::new().unwrap().forward(&[self, other]).unwrap().remove(0)
    }
}

impl std::ops::Div<&Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Self::Output {
        Div::new().unwrap().forward(&[self, *other]).unwrap().remove(0)
    }
}

impl std::ops::Div<&Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Self::Output {
        Div::new().unwrap().forward(&[*self, *other]).unwrap().remove(0)
    }
}

impl std::ops::Div<Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Self::Output {
        Div::new().unwrap().forward(&[*self, other]).unwrap().remove(0)
    }
}