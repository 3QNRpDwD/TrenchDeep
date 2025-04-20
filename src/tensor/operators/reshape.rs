use super::*;

impl Function<f32> for Reshape {
    fn new() -> MlResult<Self> {
        Ok(Self { backend: Arc::new(CpuBackend::new()?) })
    }

    /// Reshapes the tensor to the specified shape.
    ///
    /// # Arguments
    /// * `targets` - A slice of tensors to reshape.
    /// * `shape` - The new shape for the tensor.
    ///
    /// # Returns
    /// A new tensor with the specified shape.
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        let target = targets[0];
        let target_shape = target.shape();
        let target_size: usize = target_shape.iter().product();
        let new_shape = targets[1].shape();
        let new_size: usize = new_shape.iter().product();


        if target_size != new_size {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: new_shape.to_vec(),
                got: target_shape.to_vec(),
            }));
        }

        Ok(vec![Tensor::<f32>::from_vec(target.data().to_vec(), new_shape)?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let target = targets[0];
        let target_shape = target.shape();
        let target_size: usize = target_shape.iter().product();
        let new_shape = targets[1].shape();
        let new_size: usize = new_shape.iter().product();

        if target_size != new_size {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: new_shape.to_vec(),
                got: target_shape.to_vec(),
            }));
        }

        Ok(vec![Tensor::<f32>::from_vec(grad.data().to_vec(), target_shape)?])
    }
}