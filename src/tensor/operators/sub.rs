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
        let result = targets[0].with_shape(|first_shape| {
            targets[1].with_shape(|second_shape| {
                targets[0].with_data(|first_data| {
                    targets[1].with_data(|second_data| {
                        if first_shape.len() == 2 && second_shape.len() == 1 && first_shape[1] == second_shape[0] {
                            let (batch_size, features) = (first_shape[0], first_shape[1]);
                            let mut data = vec![0.0; first_data.len()];
                            for i in 0..batch_size {
                                for j in 0..features {
                                    data[i * features + j] = first_data[i * features + j] - second_data[j];
                                }
                            }
                            Tensor::from_vec(data, first_shape)
                        } else if first_shape == second_shape {
                            let data: Vec<f32> = first_data.iter().zip(second_data.iter())
                                .map(|(a, b)| a - b).collect();
                            Tensor::from_vec(data, first_shape)
                        } else {
                            Err(crate::MlError::TensorError(crate::TensorError::InvalidShape {
                                expected: first_shape.to_vec(),
                                got: second_shape.to_vec(),
                            }))
                        }
                    })
                })
            })
        })?;
        Ok(vec![result])
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