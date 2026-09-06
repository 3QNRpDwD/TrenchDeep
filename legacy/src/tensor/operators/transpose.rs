use super::*;

impl_function!(Transpose,
    forward(self, targets) {
        let input = targets[0];
        let rank = input.shape().len();
        if rank < 2 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "transpose",
                reason: "Tensor must have at least 2 dimensions".to_string(),
            }));
        }

        let d0_val = if targets.len() > 1 {
            targets[1].data()[0]
        } else {
             return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "transpose",
                reason: "d0 must be provided".to_string(),
            }));
        };

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

        let mut new_shape = input.shape().to_vec();
        new_shape.swap(d0, d1);

        let mut strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * input.shape()[i + 1];
        }

        let mut result = vec![0.0; input.data().len()];
        let mut coords = vec![0usize; rank];

        for i in 0..input.data().len() {
            let mut idx = i;
            for j in 0..rank {
                coords[j] = idx / strides[j];
                idx %= strides[j];
            }

            coords.swap(d0, d1);

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
    },
    backward(self, targets, grad) {
        let input = grad;
        let rank = input.shape().len();
        if rank < 2 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "transpose",
                reason: "Tensor must have at least 2 dimensions".to_string(),
            }));
        }

        let d0_val = if targets.len() > 1 {
            targets[1].data()[0]
        } else {
             return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "transpose",
                reason: "d0 must be provided in backward".to_string(),
            }));
        };

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

        let mut new_shape = input.shape().to_vec();
        new_shape.swap(d0, d1);

        let mut strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * input.shape()[i + 1];
        }

        let mut result = vec![0.0; input.data().len()];
        let mut coords = vec![0usize; rank];

        for i in 0..input.data().len() {
            let mut idx = i;
            for j in 0..rank {
                coords[j] = idx / strides[j];
                idx %= strides[j];
            }

            coords.swap(d0, d1);

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
);

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
