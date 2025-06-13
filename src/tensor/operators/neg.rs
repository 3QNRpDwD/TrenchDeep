use super::*;

impl Function for Neg {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Neg)
    }
    /// Negates each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the negation of tensor_element
    fn forward(&self, targets: &[&Tensor]) -> MlResult<Vec<Tensor>> {
        Ok(vec![Tensor::from_vec(targets[0].data().iter().map(|&x| -x).collect(), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _: &[&Tensor], grad: &Tensor) -> MlResult<Vec<Tensor>> {
        Ok(vec![Tensor::from_vec(grad.data().iter().map(|&x| -x).collect(), grad.shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
}

impl std::ops::Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Neg").unwrap().forward(&[&self]).unwrap().remove(0))
    }
}

impl std::ops::Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Neg").unwrap().forward(&[self]).unwrap().remove(0))
    }
}
