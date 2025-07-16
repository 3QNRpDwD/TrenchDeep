use super::*;

impl Function for Transpose {
    fn forward(&self, input: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        let input = input[0];
        let rank = input.shape().len();
        if rank < 2 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "transpose",
                reason: "Tensor must have at least 2 dimensions".to_string(),
            }));
        }

        // Convert negative dimensions to positive
        let d0 = if self.dims.0 < 0 { rank as i32 + self.dims.0 } else { self.dims.0 } as usize;
        let d1 = if self.dims.1 < 0 { rank as i32 + self.dims.1 } else { self.dims.1 } as usize;

        if d0 >= rank || d1 >= rank {
            return Err(MlError::TensorError(TensorError::InvalidAxis {
                axis: d0.max(d1),
                shape: input.shape().to_vec(),
            }));
        }

        // Create new shape with dimensions swapped
        let mut new_shape = input.shape().to_vec();
        new_shape.swap(d0, d1);

        // Calculate strides for the original shape
        let mut strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * input.shape()[i + 1];
        }

        // Create transposed data
        let mut result = vec![0.0; input.data().len()];
        let mut coords = vec![0usize; rank];

        for i in 0..input.data().len() {
            // Calculate source coordinates
            let mut idx = i;
            for j in 0..rank {
                coords[j] = idx / strides[j];
                idx %= strides[j];
            }

            // Swap the specified dimensions
            coords.swap(d0, d1);

            // Calculate target index
            let mut target_idx = 0;
            let mut stride = 1;
            for j in (0..rank).rev() {
                target_idx += coords[j] * stride;
                stride *= new_shape[j];
            }

            result[target_idx] = input.data()[i];
        }

        Ok(vec![PooledTensor::from_vec(result, &new_shape)?])
    }

    #[cfg(feature = "enableBackpropagation")]
    fn backward(&self, _: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        let input = grad;
        let rank = input.shape().len();
        if rank < 2 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "transpose",
                reason: "Tensor must have at least 2 dimensions".to_string(),
            }));
        }

        // Convert negative dimensions to positive
        let d0 = if self.dims.0 < 0 { rank as i32 + self.dims.0 } else { self.dims.0 } as usize;
        let d1 = if self.dims.1 < 0 { rank as i32 + self.dims.1 } else { self.dims.1 } as usize;

        if d0 >= rank || d1 >= rank {
            return Err(MlError::TensorError(TensorError::InvalidAxis {
                axis: d0.max(d1),
                shape: input.shape().to_vec(),
            }));
        }

        // Create new shape with dimensions swapped
        let mut new_shape = input.shape().to_vec();
        new_shape.swap(d0, d1);

        // Calculate strides for the original shape
        let mut strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * input.shape()[i + 1];
        }

        // Create transposed data
        let mut result = vec![0.0; input.data().len()];
        let mut coords = vec![0usize; rank];

        for i in 0..input.data().len() {
            // Calculate source coordinates
            let mut idx = i;
            for j in 0..rank {
                coords[j] = idx / strides[j];
                idx %= strides[j];
            }

            // Swap the specified dimensions
            coords.swap(d0, d1);

            // Calculate target index
            let mut target_idx = 0;
            let mut stride = 1;
            for j in (0..rank).rev() {
                target_idx += coords[j] * stride;
                stride *= new_shape[j];
            }

            result[target_idx] = input.data()[i];
        }

        Ok(vec![PooledTensor::from_vec(result, &new_shape)?])
    }
    
    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &HandleId { &self.node_id }
}

#[cfg(test)]
mod tests {
    use crate::{MlResult, tensor::{TensorBase, Tensor}, variable};
    use crate::nn::Parameter;
    use crate::tensor::AutogradFunction;
    use crate::tensor::operators::{Transpose, Function};
    use crate::tensor::operators::tests::assert_tensor_eq;

    #[test]
    fn test_transpose_backward() -> MlResult<()> {
        let a = variable!(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let op = Transpose::new((0, 1));
        let output = op.apply(&[&a])?;

        output.backward()?;

        let grad_a = a.grad();
        let expected_grad = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[2, 2])?;
        assert_tensor_eq(grad_a, &expected_grad)?;

        Ok(())
    }
}
