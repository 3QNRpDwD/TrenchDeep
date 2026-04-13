use super::*;

impl_function!(ReshapeOp,
    forward(self, targets) {
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

        #[cfg(feature = "debugging")]
        tracing::debug!("[Reshape::forward] {:?} → {:?}", target_shape, new_shape);

        Ok(vec![GlobalTensor::from_vec(target.data().to_vec(), new_shape)?])
    },
    backward(self, targets, grad) {
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

        #[cfg(feature = "debugging")]
        tracing::debug!("[Reshape::backward] grad {:?} → restore {:?}", grad.shape(), target_shape);

        Ok(vec![GlobalTensor::from_vec(grad.data().to_vec(), target_shape)?])
    }
);
