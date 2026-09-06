use super::*;

impl_function!(Sin,
    forward(self, targets) {
        Ok(vec![GlobalTensor::from_vec(targets[0].data().iter().map(|x| x.sin()).collect(), targets[0].shape())?])
    },
    backward(self, targets, grad) {
        let gradient = grad.data().iter()
            .zip(targets[0].data().iter())
            .map(|(grad_data, target)| target.cos() * grad_data)
            .collect();

        Ok(vec![GlobalTensor::from_vec(gradient, targets[0].shape())?])
    }
);

impl_function!(Cos,
    forward(self, targets) {
        Ok(vec![GlobalTensor::from_vec(targets[0].data().iter().map(|x| x.cos()).collect(), targets[0].shape())?])
    },
    backward(self, targets, grad) {
        let gradient = grad.data().iter()
            .zip(targets[0].data().iter())
            .map(|(grad_data, target)| -target.sin() * grad_data)
            .collect();

        Ok(vec![GlobalTensor::from_vec(gradient, targets[0].shape())?])
    }
);

/// 표준 sin/cos forward에 디버깅 로그 추가
#[cfg(feature = "debugging")]
fn log_trig_forward(name: &str, input: &dyn crate::tensor::TensorBase, output: &[f32], shape: &[usize]) {
    tracing::debug!(
        "[{}::forward] {} → {}",
        name,
        crate::tensor::operators::debug::summary("in", input),
        crate::tensor::operators::debug::summary_raw("out", output, shape)
    );
}

impl Function for ApproxSin {
    fn new() -> MlResult<GlobalFunction> {
        OPERATOR_STORAGE.with(|ops| {
            let my = "ApproxSin";
            let mut ops = ops.borrow_mut();
            match ops.contains_key(my) {
                true => Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id())),
                false => {
                    ops.insert(
                        String::from(my),
                        Box::new(ApproxSin { backend: Arc::new(CpuBackend::new()?), node_id: NODE_ID_GEN.next(), threshold: 0.0001 })
                    );
                    Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id()))
                }
            }
        })
    }

    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let x = targets[0];
        let x_data = x.data();
        let mut result = x_data.to_vec();

        let mut term_sign = -1.0;
        let mut current_power = 3;
        let mut x_power = self.backend.multiply(x_data, x_data);
        x_power = self.backend.multiply(&x_power, x_data);
        let mut factorial = 6.0;

        while current_power <= 15 {
            let term_value = self.backend.div(&x_power, &vec![factorial; x_power.len()]);
            let term = self.backend.multiply(&term_value, &vec![term_sign; term_value.len()]);
            result = self.backend.add(&result, &term);

            term_sign *= -1.0;
            x_power = self.backend.multiply(&x_power, x_data);
            x_power = self.backend.multiply(&x_power, x_data);
            factorial *= (current_power + 1) as f32 * (current_power + 2) as f32;
            current_power += 2;
        }

        #[cfg(feature = "debugging")]
        log_trig_forward("ApproxSin", x, &result, x.shape());

        Ok(vec![GlobalTensor::from_vec(result, x.shape())?])
    }

    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let mut cos = ApproxCos {
            backend: Arc::clone(&self.backend),
            threshold: self.threshold,
            node_id: NODE_ID_GEN.next()
        };

        let cos_output = cos.forward(targets)?;
        let x = targets[0];
        cos_output[0].chk_shape(grad)?;

        let grad_data = grad.data();
        let cos_data = cos_output[0].data();
        let result = self.backend.multiply(cos_data, grad_data);
        Ok(vec![GlobalTensor::from_vec(result, x.shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
    fn node_id(&self) -> &NodeId { &self.node_id }
}

impl Function for ApproxCos {
    fn new() -> MlResult<GlobalFunction> {
        OPERATOR_STORAGE.with(|ops| {
            let my = "ApproxCos";
            let mut ops = ops.borrow_mut();
            match ops.contains_key(my) {
                true => Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id())),
                false => {
                    ops.insert(
                        String::from(my),
                        Box::new(ApproxCos { backend: Arc::new(CpuBackend::new()?), node_id: NODE_ID_GEN.next(), threshold: 0.0001 })
                    );
                    Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id()))
                }
            }
        })
    }

    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let x = targets[0];
        let x_data = x.data();
        let mut result = vec![1.0; x_data.len()];

        let x_squared = self.backend.multiply(x_data, x_data);
        let mut term_sign = -1.0;
        let mut current_power = 2;
        let mut x_power = x_squared.clone();
        let mut factorial = 2.0;

        while current_power <= 14 {
            let term_value = self.backend.div(&x_power, &vec![factorial; x_power.len()]);
            let term = self.backend.multiply(&term_value, &vec![term_sign; term_value.len()]);
            result = self.backend.add(&result, &term);

            term_sign *= -1.0;
            x_power = self.backend.multiply(&x_power, x_squared.as_slice());
            factorial *= (current_power + 1) as f32 * (current_power + 2) as f32;
            current_power += 2;
        }

        #[cfg(feature = "debugging")]
        log_trig_forward("ApproxCos", x, &result, x.shape());

        Ok(vec![GlobalTensor::from_vec(result, x.shape())?])
    }

    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let mut sin = ApproxSin {
            backend: Arc::clone(&self.backend),
            node_id: NODE_ID_GEN.next(),
            threshold: self.threshold,
        };

        let sin_output = sin.forward(targets)?;
        let x = targets[0];
        sin_output[0].chk_shape(grad)?;

        let grad_data = grad.data();
        let sin_data = sin_output[0].data();
        let neg_sin = self.backend.multiply(sin_data, &vec![-1.0; sin_data.len()]);
        let result = self.backend.multiply(&neg_sin, grad_data);

        Ok(vec![GlobalTensor::from_vec(result, x.shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
    fn node_id(&self) -> &NodeId { &self.node_id }
}
