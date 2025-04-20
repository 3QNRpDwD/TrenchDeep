use super::*;
impl Function<f32> for Add {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Adds two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to add to the current tensor
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise addition
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        let first_target = targets[0];
        let second_target = targets[1];
        let first_shape = first_target.shape();
        let second_shape = second_target.shape();

        if first_shape.len() == 2 && second_shape.len() == 1 && first_shape[1] == second_shape[0] {
            // Special case for matrix + vector broadcasting
            let (batch_size, features) = (first_shape[0], first_shape[1]);
            let mut data = vec![0.0; first_target.data().len()];

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = first_target.data()[i * features + j] + second_target.data()[j];
                }
            }
            return Ok(vec![Tensor::<f32>::from_vec(data, first_shape)?])
        }

        match first_target.chk_shape(second_target) {
            Err(e) => Err(e),
            _ => Ok(vec![Tensor::<f32>::from_vec(self.backend().add(first_target.data(), second_target.data()), first_target.shape())?])
        }
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
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
impl std::ops::Add<Tensor<f32>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn add(self, other: Tensor<f32>) -> Self::Output {
        Add::new().unwrap().forward(&[&self, &other]).unwrap().remove(0)
    }
}

impl std::ops::Add<&Tensor<f32>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn add(self, other: &Tensor<f32>) -> Self::Output {
        Add::new().unwrap().forward(&[&self, other]).unwrap().remove(0)
    }
}

impl std::ops::Add<&Tensor<f32>> for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn add(self, other: &Tensor<f32>) -> Self::Output {
        Add::new().unwrap().forward(&[self, other]).unwrap().remove(0)
    }
}

impl std::ops::Add<Tensor<f32>> for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn add(self, other: Tensor<f32>) -> Self::Output {
        Add::new().unwrap().forward(&[self, &other]).unwrap().remove(0)
    }
}