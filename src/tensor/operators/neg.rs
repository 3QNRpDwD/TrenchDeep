use super::*;
use crate::nn::Variable;
use crate::tensor::AutogradFunction;

impl_function!(Neg,
    forward(self, targets) {
        Ok(vec![GlobalTensor::from_vec(targets[0].data().iter().map(|&x| -x).collect(), targets[0].shape())?])
    },
    backward(self, _targets, grad) {
        Ok(vec![GlobalTensor::from_vec(grad.data().iter().map(|&x| -x).collect(), grad.shape())?])
    }
);

impl std::ops::Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Neg").unwrap().forward(&[&self]).unwrap().remove(0))
            .to_id().unwrap()
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
