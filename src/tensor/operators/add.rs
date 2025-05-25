use super::*;
impl Function for Add {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Adds two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to add to the current tensor
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise addition
    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        let first_target = targets[0];
        let second_target = targets[1];
        let binding = first_target.shape();
        let first_shape = binding.as_slice();
        let binding = second_target.shape();
        let second_shape = binding.as_slice();
        let binding = first_target.data();
        let first_data = binding.as_slice();
        let binding = second_target.data();
        let second_data = binding.as_slice();

        if first_shape.len() == 2 && second_shape.len() == 1 && first_shape[1] == second_shape[0] {
            // Special case for matrix + vector broadcasting
            let (batch_size, features) = (first_shape[0], first_shape[1]);
            let mut data = vec![0.0; first_data.len()];

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = first_data[i * features + j] + second_data[j];
                }
            }
            return Ok(vec![Tensor::from_vec(data, first_shape)?])
        }

        match first_target.chk_shape(&second_target) {
            Err(e) => Err(e),
            _ => Ok(vec![Tensor::from_vec(self.backend().add(first_data, second_data), first_shape)?])
        }
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _: &[Tensor], grad: Tensor) -> MlResult<Vec<Tensor>> {
        Ok(vec![grad.clone(), grad.clone()])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

/// Add trait implementation for owned tensors
///
/// # Arguments
/// * `other` - The tensor to add to self
///
/// # Returns
/// A new tensor containing the element-wise sum
///
/// # Broadcasting
/// * Supports broadcasting when adding a 1D tensor to each row of a 2D tensor
impl std::ops::Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Self::Output {
        Add::new().unwrap().forward(&[self, other]).unwrap().remove(0)
    }
}

impl std::ops::Add<&Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Self::Output {
        Add::new().unwrap().forward(&[self, *other]).unwrap().remove(0)
    }
}

impl std::ops::Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Self::Output {
        Add::new().unwrap().forward(&[*self, *other]).unwrap().remove(0)
    }
}

impl std::ops::Add<Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Self::Output {
        Add::new().unwrap().forward(&[*self, other]).unwrap().remove(0)
    }
}