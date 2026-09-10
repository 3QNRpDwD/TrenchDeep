use super::*;
use crate::legacy::nn::Variable;
use crate::legacy::tensor::AutogradFunction;
use crate::legacy::tensor::broadcast::{broadcast_shape, broadcast_offsets, reduce_to_shape};

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
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        #[cfg(feature = "debugging")]
        tracing::debug!(
            "[Sub::forward] {} - {}",
            crate::legacy::tensor::operators::debug::summary("lhs", targets[0]),
            crate::legacy::tensor::operators::debug::summary("rhs", targets[1])
        );

        // Shared broadcast_shape currently treats max(0, 1) as 1. Avoid
        // indexing empty inputs until that shared contract is corrected.
        if targets[0].shape() != targets[1].shape() && targets.iter().any(|t| t.data().is_empty()) {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "sub", reason: "broadcast subtraction requires nonempty inputs".into(),
            }));
        }

        if targets[0].shape().len() == 2 && targets[1].shape().len() == 1 && targets[0].shape()[1] == targets[1].shape()[0] {
            let (batch_size, features) = (targets[0].shape()[0], targets[0].shape()[1]);
            let mut data = vec![0.0; targets[0].data().len()];

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = targets[0].data()[i * features + j] - targets[1].data()[j];
                }
            }
            return Ok(vec![GlobalTensor::from_vec(data, &targets[0].shape())?])
        }

        if targets[0].shape() == targets[1].shape() {
            return Ok(vec![GlobalTensor::from_vec(self.backend().sub(targets[0].data(), targets[1].data()), targets[0].shape())?]);
        }
        let shape = broadcast_shape(targets[0].shape(), targets[1].shape())?;
        let offsets = broadcast_offsets(targets[0].shape(), targets[1].shape(), &shape);
        let data = offsets.iter().map(|&(a, b)| targets[0].data()[a] - targets[1].data()[b]).collect();
        Ok(vec![GlobalTensor::from_vec(data, &shape)?])
    }

    fn assign_forward(&self, targets: &[&dyn TensorBase], node_id: NodeId) -> MlResult<Vec<Tensor>> {
        let result = self.forward(targets)?.remove(0);
        Ok(vec![Tensor::with_id(result.data().to_vec(), result.shape(), node_id)?])
    }

    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        #[cfg(feature = "debugging")]
        tracing::debug!("[Sub::backward] {}", crate::legacy::tensor::operators::debug::summary("grad", grad));

        let expected = broadcast_shape(targets[0].shape(), targets[1].shape())?;
        if grad.shape() != expected {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected, got: grad.shape().to_vec(),
            }));
        }
        let gt = GlobalTensor::from_vec(reduce_to_shape(grad.data(), grad.shape(), targets[0].shape()), targets[0].shape())?;
        let rhs = reduce_to_shape(grad.data(), grad.shape(), targets[1].shape());
        let neg = GlobalTensor::from_vec(rhs.iter().map(|&x| -x).collect(), targets[1].shape())?;

        #[cfg(feature = "debugging")]
        {
            crate::legacy::tensor::operators::debug::stats_raw("  └─ dlhs", &gt.data, &gt.shape);
            crate::legacy::tensor::operators::debug::stats_raw("  └─ drhs", &neg.data, &neg.shape);
        }

        Ok(vec![gt, neg])
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
/// * Supports trailing-axis broadcasting for nonempty tensors
impl std::ops::Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Self::Output {
        Sub::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Sub").unwrap().forward(&[&self, &other]).unwrap().remove(0))
            .to_id().unwrap()
    }
}

impl std::ops::Sub<&Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Self::Output {
        Sub::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Sub").unwrap().forward(&[&self, other]).unwrap().remove(0))
            .to_id().unwrap()
    }
}

impl std::ops::Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Self::Output {
        Sub::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Sub").unwrap().forward(&[self, other]).unwrap().remove(0))
            .to_id().unwrap()
    }
}

impl std::ops::Sub<Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Self::Output {
        Sub::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Sub").unwrap().forward(&[self, &other]).unwrap().remove(0))
            .to_id().unwrap()
    }
}

impl std::ops::Sub<&dyn TensorBase> for &dyn TensorBase {
    type Output = GlobalTensor<f32>;

    fn sub(self, other: &dyn TensorBase) -> Self::Output {
        Sub::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Sub").unwrap().forward(&[self, other]).unwrap().remove(0))
    }
}

/// SubAssign trait implementation for Tensor
impl std::ops::SubAssign<Tensor> for Tensor {
    fn sub_assign(&mut self, other: Tensor) {
        Sub::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Sub").unwrap().assign_forward(&[self, &other], self.id()).unwrap().remove(0));
    }
}

impl std::ops::SubAssign<&Tensor> for Tensor {
    fn sub_assign(&mut self, other: &Tensor) {
        Sub::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Sub").unwrap().assign_forward(&[self, other], self.id()).unwrap().remove(0));
    }
}

// Variable operator overloading (graph-tracked)
impl std::ops::Sub<&Variable> for &Variable {
    type Output = Variable;
    fn sub(self, other: &Variable) -> Variable {
        Sub::new().unwrap().apply(&[self, other]).unwrap()
    }
}

impl std::ops::Sub<&Variable> for Variable {
    type Output = Variable;
    fn sub(self, other: &Variable) -> Variable {
        Sub::new().unwrap().apply(&[&self, other]).unwrap()
    }
}

impl std::ops::Sub<Variable> for &Variable {
    type Output = Variable;
    fn sub(self, other: Variable) -> Variable {
        Sub::new().unwrap().apply(&[self, &other]).unwrap()
    }
}

impl std::ops::Sub<Variable> for Variable {
    type Output = Variable;
    fn sub(self, other: Variable) -> Variable {
        Sub::new().unwrap().apply(&[&self, &other]).unwrap()
    }
}

impl std::ops::SubAssign<&Variable> for Variable {
    fn sub_assign(&mut self, other: &Variable) {
        *self = Sub::new().unwrap().apply(&[self, other]).unwrap();
    }
}

impl std::ops::SubAssign<Variable> for Variable {
    fn sub_assign(&mut self, other: Variable) {
        *self = Sub::new().unwrap().apply(&[self, &other]).unwrap();
    }
}
