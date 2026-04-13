use super::*;

impl_function!(Abs,
    forward(self, targets) {
        Ok(vec![GlobalTensor::from_vec(targets[0].data().iter().map(|&x| x.abs()).collect(), targets[0].shape())?])
    }
);

impl_function!(Exp,
    forward(self, targets) {
        #[cfg(feature = "debugging")]
        tracing::debug!(
            "[Exp::forward] {}",
            crate::tensor::operators::debug::summary("in", targets[0])
        );

        Ok(vec![GlobalTensor::from_vec(self.backend().exp(targets[0].data()), targets[0].shape())?])
    },
    backward(self, targets, grad) {
        let gradient: Vec<f32> = grad.data().iter()
            .zip(targets[0].data().iter())
            .map(|(grad_data, target_data)| target_data.exp() * grad_data)
            .collect();

        #[cfg(feature = "debugging")]
        crate::tensor::operators::debug::stats_raw("  └─ dExp", &gradient, targets[0].shape());

        Ok(vec![GlobalTensor::from_vec(gradient, targets[0].shape())?])
    }
);

impl_function!(Log,
    forward(self, targets) {
        Ok(vec![GlobalTensor::from_vec(targets[0].data().iter().map(|&x| x.ln()).collect(), targets[0].shape())?])
    }
);

impl_function!(Pow,
    forward(self, targets) {
        let power = if targets.len() > 1 {
            targets[1].data()[0]
        } else {
             return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "pow",
                reason: "exponent must be provided".to_string(),
            }));
        };
        Ok(vec![GlobalTensor::from_vec(self.backend().pow(targets[0].data(), power), targets[0].shape())?])
    },
    backward(self, targets, grad) {
        let power = targets[1].data()[0];
        let target = targets[0];
        let forwarded = GlobalTensor::from_vec(self.backend().pow(target.data(), power - 1.0), target.shape())?;

        let result_data: Vec<f32> = forwarded
                .data()
                .iter()
                .zip(grad.data().iter())
                .map(|(&x, &g)| power * x * g)
                .collect();

        Ok(vec![GlobalTensor::from_vec(result_data, target.shape())?])
    }
);

impl_function!(Square,
    forward(self, targets) {
        Ok(vec![GlobalTensor::from_vec(targets[0].data().iter().map(|x| x * x).collect(), targets[0].shape())?])
    },
    backward(self, targets, grad) {
        let grad_broadcasted = if grad.data().len() == 1 {
            vec![grad.data()[0]; targets[0].data().len()]
        } else {
            grad.data().to_vec()
        };

        let gradient: Vec<f32> = grad_broadcasted.iter()
            .zip(targets[0].data().iter())
            .map(|(grad_data, target_data)| grad_data * 2.0 * target_data)
            .collect();

        Ok(vec![GlobalTensor::from_vec(gradient, targets[0].shape())?])
    }
);

impl_function!(Sqrt,
    forward(self, targets) {
        Ok(vec![GlobalTensor::from_vec(self.backend().sqrt(targets[0].data()), targets[0].shape())?])
    }
);
