use super::*;

impl Function for Sub {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Subtracts two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to subtract from_vec the current tensor
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise subtraction
    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        let first_shape = targets[0].shape();
        let second_shape = targets[1].shape();
        let first_data = targets[0].data();
        let second_data = targets[1].data();


        if first_shape.len() == 2 && second_shape.len() == 1 && first_shape[1] == second_shape[0] {
            let (batch_size, features) = (first_shape[0], first_shape[1]);
            let mut data = vec![0.0; first_data.len()];

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = first_data[i * features + j] - second_data[j];
                }
            }
            return Ok(vec![Tensor::from_vec(data, &first_shape)?])
        }

        match targets[0].chk_shape(&targets[1]) {
            Err(e) => Err(e),
            _ => Ok(vec![Tensor::from_vec(self.backend().sub(first_data.as_slice(), second_data.as_slice()), first_shape.as_slice())?])
        }
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _: &[Tensor], grad: Tensor) -> MlResult<Vec<Tensor>> {
        Ok(vec![grad.clone(), -grad.clone()])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}


/// Subtract trait implementation for owned tensors
///
/// # Arguments
/// * `other` - The tensor to subtract from self
///
/// # Returns
/// A new tensor containing the element-wise difference
///
/// # Broadcasting
/// * Supports broadcasting when subtracting a 1D tensor from each row of a 2D tensor
impl std::ops::Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Self::Output {
        Sub::new().unwrap().forward(&[self, other]).unwrap().remove(0)
    }
}

impl std::ops::Sub<&Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Self::Output {
        Sub::new().unwrap().forward(&[self, *other]).unwrap().remove(0)
    }
}

impl std::ops::Sub<Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Self::Output {
        Sub::new().unwrap().forward(&[*self, other]).unwrap().remove(0)
    }
}

impl std::ops::Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Self::Output {
        Sub::new().unwrap().forward(&[*self, *other]).unwrap().remove(0)
    }
}