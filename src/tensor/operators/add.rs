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

        let first_shape_opt = first_target.with_shape(|shape| shape.to_vec());
        let second_shape_opt = second_target.with_shape(|shape| shape.to_vec());

        if first_shape_opt.is_none() || second_shape_opt.is_none() {
            return Err(MlError::from(TensorError::TensorNotFound));
        }

        let first_shape = first_shape_opt.unwrap();
        let second_shape = second_shape_opt.unwrap();

        // 행렬 + 벡터 특수 케이스 처리
        if first_shape.len() == 2 && second_shape.len() == 1 && first_shape[1] == second_shape[0] {
            let (batch_size, features) = (first_shape[0], first_shape[1]);

            let result = first_target.with_data(|first_data| {
                second_target.with_data(|second_data| {
                    let mut data = vec![0.0; first_data.len()];
                    for i in 0..batch_size {
                        for j in 0..features {
                            data[i * features + j] = first_data[i * features + j] + second_data[j];
                        }
                    }
                    data
                }).ok_or(MlError::from(TensorError::TensorNotFound))
            }).ok_or(MlError::from(TensorError::TensorNotFound))??;

            return Ok(vec![Tensor::from_vec(result, &first_shape)?]);
        }

        match first_target.chk_shape(&second_target) {
            Err(e) => Err(e),
            _ => {
                let result = first_target.with_data(|first_data| {
                    second_target.with_data(|second_data| {
                        self.backend().add(first_data, second_data)
                    }).ok_or(MlError::from(TensorError::TensorNotFound))
                }).ok_or(MlError::from(TensorError::TensorNotFound))??;

                Ok(vec![Tensor::from_vec(result, &first_shape)?])
            }
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