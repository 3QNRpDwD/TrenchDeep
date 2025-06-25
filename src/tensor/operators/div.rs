use super::*;

impl Function for Div {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Div)
    }
    
    /// Divides two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to divide the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise division
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        match targets[0].chk_shape(targets[1]) {
            Err(e) => Err(e),
            _ => Ok(vec![GlobalTensor::from_vec(self.backend().div(targets[0].data(), targets[1].data()), targets[0].shape())?])
        }
    }

    fn assign_forward(&self, targets: &[&dyn TensorBase], node_id: HandleId) -> MlResult<Vec<Tensor>> {
        match targets[0].chk_shape(targets[1]) {
            Err(e) => Err(e),
            _ => Ok(vec![Tensor::with_id(self.backend().div(targets[0].data(), targets[1].data()), targets[0].shape(), node_id)?])
        }
    }
    
    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let x1 = targets[1];

        Ok(vec![
            self.forward(&[grad, x1])?.remove(0), // grad / x2
            grad * &self.forward(&[&-targets[0], &(x1 * x1)])?.remove(0) // grad * (-x0 / x1^2)
        ])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &HandleId { &self.node_id }
}

/// Divide trait implementation for owned tensors
///
/// # Arguments
/// * `other` - The tensor to divide self by
///
/// # Returns
/// A new tensor containing the element-wise quotient
impl std::ops::Div<&Tensor> for Tensor {
    type Output = GlobalTensor<f32>;

    fn div(self, other: &Tensor) -> Self::Output {
        Div::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow_mut().get_mut("Div").unwrap().forward(&[&self, other]).unwrap().remove(0))
    }
}

impl std::ops::Div<Tensor> for Tensor {
    type Output = GlobalTensor<f32>;

    fn div(self, other: Tensor) -> Self::Output {
        Div::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow_mut().get_mut("Div").unwrap().forward(&[&self, &other]).unwrap().remove(0))
    }
}


impl std::ops::Div<&dyn TensorBase> for &dyn TensorBase {
    type Output = GlobalTensor<f32>;

    fn div(self, other: &dyn TensorBase) -> Self::Output {
        Div::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow_mut().get_mut("Div").unwrap().forward(&[self, other]).unwrap().remove(0))
    }
}

impl std::ops::Div<Tensor> for &Tensor {
    type Output = GlobalTensor<f32>;

    fn div(self, other: Tensor) -> Self::Output {
        Div::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow_mut().get_mut("Div").unwrap().forward(&[self, &other]).unwrap().remove(0))
    }
}

/// DivAssign trait implementation for Tensor
impl std::ops::DivAssign<Tensor> for Tensor {
    fn div_assign(&mut self, other: Tensor) {
        Div::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow_mut().get_mut("Div").unwrap().assign_forward(&[self, &other], self.0).unwrap().remove(0));
    }
}

impl std::ops::DivAssign<&Tensor> for Tensor {
    fn div_assign(&mut self, other: &Tensor) {
        Div::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow_mut().get_mut("Div").unwrap().assign_forward(&[self, other], self.0).unwrap().remove(0));
    }
}
