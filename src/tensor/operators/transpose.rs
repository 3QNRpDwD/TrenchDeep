use super::*;


impl Function for Transpose {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Transpose)
    }
    
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let input = targets[0];
        let rank = input.shape().len();
        if rank < 2 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "transpose",
                reason: "Tensor must have at least 2 dimensions".to_string(),
            }));
        }

        // Parse d0
        let d0_val = if targets.len() > 1 {
            targets[1].data()[0]
        } else {
             return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "transpose",
                reason: "d0 must be provided".to_string(),
            }));
        };

        // Parse d1
        let d1_val = if targets.len() > 2 {
            targets[2].data()[0]
        } else {
             return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "transpose",
                reason: "d1 must be provided".to_string(),
            }));
        };

        let d0 = if d0_val < 0.0 { (rank as i32 + d0_val as i32) as usize } else { d0_val as usize };
        let d1 = if d1_val < 0.0 { (rank as i32 + d1_val as i32) as usize } else { d1_val as usize };

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

        #[cfg(feature = "debugging")]
        tracing::debug!(
            "[Transpose::forward] {:?} d{}↔d{} → {:?}",
            input.shape(), d0, d1, new_shape
        );

        Ok(vec![GlobalTensor::from_vec(result, &new_shape)?])
    }

    #[cfg(feature = "enableBackward")]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let input = grad; // Gradient of the output is the input to backward
        // The targets array contains [input_tensor, d0_tensor, d1_tensor] from the forward pass
        // We need d0 and d1 to reverse the transpose (which is just transposing again with same dims)
        
        let rank = input.shape().len();
        if rank < 2 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "transpose",
                reason: "Tensor must have at least 2 dimensions".to_string(),
            }));
        }

         // Parse d0
        let d0_val = if targets.len() > 1 {
            targets[1].data()[0]
        } else {
             // Should not happen if forward succeeded and graph saved inputs
             return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "transpose",
                reason: "d0 must be provided in backward".to_string(),
            }));
        };

        // Parse d1
        let d1_val = if targets.len() > 2 {
            targets[2].data()[0]
        } else {
             return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "transpose",
                reason: "d1 must be provided in backward".to_string(),
            }));
        };

        let d0 = if d0_val < 0.0 { (rank as i32 + d0_val as i32) as usize } else { d0_val as usize };
        let d1 = if d1_val < 0.0 { (rank as i32 + d1_val as i32) as usize } else { d1_val as usize };

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

        Ok(vec![GlobalTensor::from_vec(result, &new_shape)?])
    }
    
    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
}

#[cfg(test)]
mod tests {
    use crate::tensor::operators::{Function, Transpose};
    use crate::tensor::{Tensor, TensorBase};
    use crate::MlResult;

    #[test]
    fn test_transpose() -> MlResult<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let d0 = Tensor::scalar(0.0);
        let d1 = Tensor::scalar(1.0);
        
        let op = Transpose::new().unwrap();
        let result = op.forward(&[&input, &d0, &d1])?;
        let result_tensor = &result[0];

        assert_eq!(result_tensor.shape(), &[2, 2]);
        assert_eq!(result_tensor.data(), &[1.0, 3.0, 2.0, 4.0]);
        Ok(())
    }
}