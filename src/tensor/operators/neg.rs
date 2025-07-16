use super::*;

impl Function for Neg {
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        Ok(vec![PooledTensor::from_vec(targets[0].data().iter().map(|&x| -x).collect(), targets[0].shape()).unwrap()])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        Ok(vec![PooledTensor::from_vec(grad.data().iter().map(|&x| -x).collect(), grad.shape()).unwrap()])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &HandleId { &self.node_id }
}

impl std::ops::Neg for Tensor {
    type Output = PooledTensor;

    fn neg(self) -> Self::Output {
        let op = Neg::new();
        op.forward(&[&self]).unwrap().remove(0)
    }
}

impl std::ops::Neg for &dyn TensorBase {
    type Output = PooledTensor;

    fn neg(self) -> Self::Output {
        let op = Neg::new();
        op.forward(&[self]).unwrap().remove(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MlResult, variable,tensor::{TensorBase, Tensor}};
    use crate::tensor::operators::tests::assert_tensor_eq;

    #[test]
    fn test_neg_backward() -> MlResult<()> {
        let a = variable!(vec![vec![1.0, -2.0], vec![3.0, -4.0]]);
        let op = Neg::new();
        let output = op.apply(&[&a])?;

        output.backward()?;

        let grad_a = a.grad();

        let expected_grad = Tensor::from_vec(vec![-1.0, -1.0, -1.0, -1.0], &[2, 2])?;
        assert_tensor_eq(grad_a, &expected_grad)?;

        Ok(())
    }
}