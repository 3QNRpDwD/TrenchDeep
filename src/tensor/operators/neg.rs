use super::*;
use crate::nn::Variable;
use crate::tensor::AutogradFunction;

impl Function for Neg {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Neg)
    }
    /// Negates each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the negation of tensor_element
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        Ok(vec![GlobalTensor::from_vec(targets[0].data().iter().map(|&x| -x).collect(), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, _: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        Ok(vec![GlobalTensor::from_vec(grad.data().iter().map(|&x| -x).collect(), grad.shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
}

impl std::ops::Neg for Tensor {
    type Output = GlobalTensor<f32>;

    fn neg(self) -> Self::Output {
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Neg").unwrap().forward(&[&self]).unwrap().remove(0))
    }
}

impl std::ops::Neg for &dyn TensorBase {
    type Output = GlobalTensor<f32>;

    fn neg(self) -> Self::Output {
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Neg").unwrap().forward(&[self]).unwrap().remove(0))
    }
}

// Variable operator overloading (graph-tracked)
impl std::ops::Neg for &Variable {
    type Output = Variable;
    fn neg(self) -> Variable {
        Neg::new().unwrap().apply(&[self]).unwrap()
    }
}

impl std::ops::Neg for Variable {
    type Output = Variable;
    fn neg(self) -> Variable {
        Neg::new().unwrap().apply(&[&self]).unwrap()
    }
}
