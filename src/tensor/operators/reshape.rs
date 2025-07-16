use super::*;

impl Function for Reshape {
    /// Reshapes the tensor to the specified shape.
    ///
    /// # Arguments
    /// * `targets` - A slice of tensors to reshape.
    /// * `shape` - The new shape for the tensor.
    ///
    /// # Returns
    /// A new tensor with the specified shape.
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        let target = targets[0];
        let target_shape = target.shape();
        let target_size: usize = target_shape.iter().product();
        let new_shape = targets[1].shape();
        let new_size: usize = new_shape.iter().product();


        if target_size != new_size {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: new_shape.to_vec(),
                got: target_shape.to_vec(),
            }));
        }

        Ok(vec![PooledTensor::from_vec(target.data().to_vec(), new_shape).unwrap()])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        let target = targets[0];
        let target_shape = target.shape();
        let target_size: usize = target_shape.iter().product();
        let new_shape = targets[1].shape();
        let new_size: usize = new_shape.iter().product();

        if target_size != new_size {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: new_shape.to_vec(),
                got: target_shape.to_vec(),
            }));
        }

        Ok(vec![PooledTensor::from_vec(grad.data().to_vec(), target_shape).unwrap()])
    }
    
    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &HandleId { &self.node_id }
}

#[cfg(test)]
mod tests {
    use crate::nn::Parameter;
use crate::{MlResult, tensor::{TensorBase, Tensor}, variable};
    use crate::tensor::AutogradFunction;
    use crate::tensor::operators::{Reshape, Function};
    use crate::tensor::operators::tests::assert_tensor_eq;

    #[test]
    fn tensor_reshape_operator() -> MlResult<()> {
        let tensor = Tensor::new(vec![vec![1.0, 2.0, 3.0, 4.0]]);
        let shape_tensor = Tensor::new(vec![vec![2.0, 2.0]]);
        let op = Reshape::new();
        let result = op.forward(&[&tensor, &shape_tensor])?.remove(0);
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.data(), vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn test_reshape_backward() -> MlResult<()> {
        let a = variable!(vec![vec![1.0, 2.0, 3.0, 4.0]]);
        let shape_tensor = variable!(vec![vec![2.0, 2.0]]);
        let op = Reshape::new();
        let output = op.apply(&[&a, &shape_tensor])?;

        output.backward()?;

        let grad_a = a.grad();
        let expected_grad = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 4])?;
        assert_tensor_eq(grad_a, &expected_grad)?;

        Ok(())
    }
}