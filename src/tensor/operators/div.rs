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
    fn forward(&self, targets: &[&Tensor]) -> MlResult<Vec<Tensor>> {
        match targets[0].chk_shape(targets[1]) {
            Err(e) => Err(e),
            _ => Ok(vec![Tensor::from_vec(self.backend().div(targets[0].data(), targets[1].data()), targets[0].shape())?])
        }
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor], grad: &Tensor) -> MlResult<Vec<Tensor>> {
        let x1 = targets[1];

        Ok(vec![
            self.forward(&[grad, x1])?.remove(0), // grad / x2
            grad * self.forward(&[&-targets[0], &(x1 * x1)])?.remove(0) // grad * (-x0 / x1^2)
        ])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
}

/// Divide trait implementation for owned tensors
///
/// # Arguments
/// * `other` - The tensor to divide self by
///
/// # Returns
/// A new tensor containing the element-wise quotient
impl std::ops::Div<&Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Self::Output {
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Div").unwrap().forward(&[&self, other]).unwrap().remove(0))
    }
}

impl std::ops::Div<Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Self::Output {
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Div").unwrap().forward(&[&self, &other]).unwrap().remove(0))
    }
}


impl std::ops::Div<&Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Self::Output {
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Div").unwrap().forward(&[self, other]).unwrap().remove(0))
    }
}

impl std::ops::Div<Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Self::Output {
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Div").unwrap().forward(&[self, &other]).unwrap().remove(0))
    }
}

/// DivAssign trait implementation for Tensor
impl std::ops::DivAssign<Tensor> for Tensor {
    fn div_assign(&mut self, other: Tensor) {
        *self =  OPERATOR_STORAGE.with(|ops| ops.borrow().get("Div").unwrap().forward(&[self, &other]).unwrap().remove(0));
    }
}

impl std::ops::DivAssign<&Tensor> for Tensor {
    fn div_assign(&mut self, other: &Tensor) {
        *self =  OPERATOR_STORAGE.with(|ops| ops.borrow().get("Div").unwrap().forward(&[self, other]).unwrap().remove(0));
    }
}
