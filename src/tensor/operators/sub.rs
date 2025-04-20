use super::*;

impl Function<f32> for Sub {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Subtracts two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to subtract from_vec the current tensor
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise subtraction
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        if targets[0].shape().len() == 2 && targets[1].shape().len() == 1 && targets[0].shape()[1] == targets[1].shape()[0] {
            let (batch_size, features) = (targets[0].shape()[0], targets[0].shape()[1]);
            let mut data = vec![0.0; targets[0].data().len()];

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = targets[0].data()[i * features + j] - targets[1].data()[j];
                }
            }
            return Ok(vec![Tensor::<f32>::from_vec(data, &targets[0].shape())?])
        }

        match targets[0].chk_shape(targets[1]) {
            Err(e) => Err(e),
            _ => Ok(vec![Tensor::<f32>::from_vec(self.backend().sub(targets[0].data(), targets[1].data()), targets[0].shape())?])
        }
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
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
impl std::ops::Sub<Tensor<f32>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn sub(self, other: Tensor<f32>) -> Self::Output {
        Sub::new().unwrap().forward(&[&self, &other]).unwrap().remove(0)
    }
}

impl std::ops::Sub<&Tensor<f32>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn sub(self, other: &Tensor<f32>) -> Self::Output {
        Sub::new().unwrap().forward(&[&self, other]).unwrap().remove(0)
    }
}

impl std::ops::Sub<Tensor<f32>> for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn sub(self, other: Tensor<f32>) -> Self::Output {
        Sub::new().unwrap().forward(&[self, &other]).unwrap().remove(0)
    }
}

impl std::ops::Sub<&Tensor<f32>> for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn sub(self, other: &Tensor<f32>) -> Self::Output {
        Sub::new().unwrap().forward(&[self, other]).unwrap().remove(0)
    }
}