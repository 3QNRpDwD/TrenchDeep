use super::*;

impl_function!(Sum,
    forward(self, inputs) {
        if inputs.len() != 1 {
            return Err(MlError::StringError(format!(
                "Sum operation expects 1 input tensor, but got {}",
                inputs.len()
            )));
        }
        let target = inputs[0];
        let total_sum: f32 = target.data().iter().sum();

        #[cfg(feature = "debugging")]
        tracing::debug!(
            "[Sum::forward] {} → scalar={:.6}",
            crate::legacy::tensor::operators::debug::summary("in", target),
            total_sum
        );

        Ok(vec![GlobalTensor::from_vec(vec![total_sum], &[1,1])?])
    },
    backward(self, targets, grad) {
        if targets.is_empty() {
            return Err(MlError::TensorError(TensorError::EmptyTensor));
        }

        #[cfg(feature = "debugging")]
        tracing::debug!(
            "[Sum::backward] {} → broadcast×{}",
            crate::legacy::tensor::operators::debug::summary("grad", grad),
            targets.len()
        );

        if targets.len() != 1 || grad.data().len() != 1 {
            return Err(MlError::StringError(
                "Sum backward expects one input and a scalar upstream gradient".into()
            ));
        }
        Ok(vec![GlobalTensor::from_vec(
            vec![grad.data()[0]; targets[0].data().len()],
            targets[0].shape(),
        )?])
    }
);
