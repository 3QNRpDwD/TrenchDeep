use super::*;

impl Function for Sub {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Sub)
    }
    /// Subtracts two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to subtract from_vec the current tensor
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise subtraction
    fn forward(&self, targets: &[&Tensor]) -> MlResult<Vec<Tensor>> {
        if targets[0].shape().len() == 2 && targets[1].shape().len() == 1 && targets[0].shape()[1] == targets[1].shape()[0] {
            let (batch_size, features) = (targets[0].shape()[0], targets[0].shape()[1]);
            let mut data = vec![0.0; targets[0].data().len()];

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = targets[0].data()[i * features + j] - targets[1].data()[j];
                }
            }
            return Ok(vec![Tensor::from_vec(data, &targets[0].shape())?])
        }

        match targets[0].chk_shape(targets[1]) {
            Err(e) => Err(e),
            _ => Ok(vec![Tensor::from_vec(self.backend().sub(targets[0].data(), targets[1].data()), targets[0].shape())?])
        }
    }

    fn assign_forward(&self, targets: &[&Tensor]) -> MlResult<Vec<Tensor>> {
        if targets[0].shape().len() == 2 && targets[1].shape().len() == 1 && targets[0].shape()[1] == targets[1].shape()[0] {
            let (batch_size, features) = (targets[0].shape()[0], targets[0].shape()[1]);
            let mut data = vec![0.0; targets[0].data().len()];

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = targets[0].data()[i * features + j] - targets[1].data()[j];
                }
            }
            return Ok(vec![Tensor::with_id(data, &targets[0].shape(), targets[0].0)?])
        }

        match targets[0].chk_shape(targets[1]) {
            Err(e) => Err(e),
            _ => Ok(vec![Tensor::with_id(self.backend().sub(targets[0].data(), targets[1].data()), targets[0].shape(), targets[0].0)?])
        }
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _: &[&Tensor], grad: &Tensor) -> MlResult<Vec<Tensor>> {
        Ok(vec![grad.clone(), Tensor::from_vec(grad.data().iter().map(|&x| -x).collect(), grad.shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
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
        Sub::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Sub").unwrap().forward(&[&self, &other]).unwrap().remove(0))
    }
}

impl std::ops::Sub<&Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Self::Output {
        Sub::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Sub").unwrap().forward(&[&self, other]).unwrap().remove(0))
    }
}

impl std::ops::Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Self::Output {
        Sub::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Sub").unwrap().forward(&[self, other]).unwrap().remove(0))
    }
}

impl std::ops::Sub<Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Self::Output {
        Sub::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Sub").unwrap().forward(&[self, &other]).unwrap().remove(0))
    }
}

/// SubAssign trait implementation for Tensor
impl std::ops::SubAssign<Tensor> for Tensor {
    fn sub_assign(&mut self, other: Tensor) {
        Sub::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Sub").unwrap().assign_forward(&[self, &other]).unwrap().remove(0));
    }
}

impl std::ops::SubAssign<&Tensor> for Tensor {
    fn sub_assign(&mut self, other: &Tensor) {
        Sub::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Sub").unwrap().assign_forward(&[self, other]).unwrap().remove(0));
    }
}
