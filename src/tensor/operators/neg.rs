use super::*;

impl Function for Neg {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Neg)
    }
    /// Negates each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the negation of tensor_element
    fn forward(&mut self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        Ok(vec![GlobalTensor::from_vec(targets[0].data().iter().map(|&x| -x).collect(), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&mut self, _: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        Ok(vec![GlobalTensor::from_vec(grad.data().iter().map(|&x| -x).collect(), grad.shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
}

impl std::ops::Neg for Tensor {
    type Output = GlobalTensor<f32>;

    fn neg(self) -> Self::Output {
        OPERATOR_STORAGE.with(|ops| ops.borrow_mut().get_mut("Neg").unwrap().forward(&[&self]).unwrap().remove(0))
    }
}

impl std::ops::Neg for &dyn TensorBase {
    type Output = GlobalTensor<f32>;

    fn neg(self) -> Self::Output {
        OPERATOR_STORAGE.with(|ops| ops.borrow_mut().get_mut("Neg").unwrap().forward(&[self]).unwrap().remove(0))
    }
}
