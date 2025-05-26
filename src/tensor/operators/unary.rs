use std::sync::Arc; // Assuming this is used by super::* for backend.

use super::*; // Assuming this brings in Tensor, Function, MlResult, Backend types, etc.

// Abs
impl Function for Abs {
    fn new() -> MlResult<Self> {
        Ok(Self {
            backend: Arc::new(CpuBackend::new()?),
        })
    }

    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        let target_tensor = targets[0];
        target_tensor.with_shape(|shape_slice| {
            target_tensor.with_data(|data_slice| {
                let result_data = data_slice.iter().map(|&x| x.abs()).collect();
                Tensor::from_vec(result_data, shape_slice)
            })
        })
            .map(|t| vec![t]) // Wrap the resulting Tensor in Ok(vec![...])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _targets: &[Tensor], _grad: Tensor) -> MlResult<Vec<Tensor>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

// Exp
impl Function for Exp {
    fn new() -> MlResult<Self> {
        Ok(Self {
            backend: Arc::new(CpuBackend::new()?),
        })
    }

    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        let target_tensor = targets[0];
        target_tensor.with_shape(|shape_slice| {
            target_tensor.with_data(|data_slice| {
                // Assuming self.backend().exp() takes &[f32] and returns Vec<f32>
                let result_data = self.backend().exp(data_slice);
                Tensor::from_vec(result_data, shape_slice)
            })
        })
            .map(|t| vec![t])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[Tensor], grad: Tensor) -> MlResult<Vec<Tensor>> {
        let target_tensor = targets[0];
        let result_tensor = target_tensor.with_shape(|target_shape_slice| {
            target_tensor.with_data(|target_data_slice| {
                grad.with_data(|grad_data_slice| {
                    let gradient_data: Vec<f32> = grad_data_slice
                        .iter()
                        .zip(target_data_slice.iter())
                        .map(|(&grad_val, &target_val)| target_val.exp() * grad_val)
                        .collect();
                    Tensor::from_vec(gradient_data, target_shape_slice)
                })
            })
        })?;
        Ok(vec![result_tensor])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

// Log
impl Function for Log {
    fn new() -> MlResult<Self> {
        Ok(Self {
            backend: Arc::new(CpuBackend::new()?),
        })
    }

    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        let target_tensor = targets[0];
        target_tensor.with_shape(|shape_slice| {
            target_tensor.with_data(|data_slice| {
                let result_data = data_slice.iter().map(|&x| x.ln()).collect();
                Tensor::from_vec(result_data, shape_slice)
            })
        })
            .map(|t| vec![t])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _targets: &[Tensor], _grad: Tensor) -> MlResult<Vec<Tensor>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

// Pow
impl Function for Pow {
    fn new() -> MlResult<Self> {
        Ok(Self {
            backend: Arc::new(CpuBackend::new()?),
            power: None,
        })
    }

    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        let target_tensor = targets[0];
        let power_val = self.power.ok_or_else(|| MlError::StringError("Power not set for Pow op".to_string()))?;
        target_tensor.with_shape(|shape_slice| {
            target_tensor.with_data(|data_slice| {
                // Assuming self.backend().pow() takes &[f32] and f32, returns Vec<f32>
                let result_data = self.backend().pow(data_slice, power_val);
                Tensor::from_vec(result_data, shape_slice)
            })
        })
            .map(|t| vec![t])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[Tensor], grad: Tensor) -> MlResult<Vec<Tensor>> {
        let target_tensor = targets[0];
        let power_val = self.power.ok_or_else(|| MlError::StringError("Power not set for Pow op".to_string()))?;

        let term_tensor = target_tensor.with_shape(|shape_slice| {
            target_tensor.with_data(|data_slice| {
                // Calculate data for c * x**(c-1)
                // Step 1: x_data_pow_c_minus_1 = x**(c-1)
                let x_data_pow_c_minus_1 = self.backend().pow(data_slice, power_val - 1.0);

                // Step 2: term_data = c * x**(c-1)
                let term_data: Vec<f32> = x_data_pow_c_minus_1.iter().map(|&x| power_val * x).collect();
                Tensor::from_vec(term_data, shape_slice)
            })
        })?;

        // Assuming Tensor * Tensor (term_tensor * grad) is a defined operation
        // that handles its own data access correctly.
        Ok(vec![term_tensor * grad])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

// Square
impl Function for Square {
    fn new() -> MlResult<Self> {
        Ok(Self {
            backend: Arc::new(CpuBackend::new()?),
        })
    }

    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        let target_tensor = targets[0];
        target_tensor.with_shape(|shape_slice| {
            target_tensor.with_data(|data_slice| {
                let result_data = data_slice.iter().map(|x| x * x).collect();
                Tensor::from_vec(result_data, shape_slice)
            })
        })
            .map(|t| vec![t])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[Tensor], grad: Tensor) -> MlResult<Vec<Tensor>> {
        let target_tensor = targets[0];
        let result_tensor = target_tensor.with_shape(|target_shape_slice| {
            target_tensor.with_data(|target_data_slice| {
                grad.with_data(|grad_data_slice| {
                    let gradient_data: Vec<f32> = grad_data_slice
                        .iter()
                        .zip(target_data_slice.iter())
                        .map(|(&grad_val, &target_val)| 2.0 * target_val * grad_val)
                        .collect();
                    Tensor::from_vec(gradient_data, target_shape_slice)
                })
            })
        })?;
        Ok(vec![result_tensor])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

// Sqrt
impl Function for Sqrt {
    fn new() -> MlResult<Self> {
        Ok(Self {
            backend: Arc::new(CpuBackend::new()?),
        })
    }

    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        let target_tensor = targets[0];
        target_tensor.with_shape(|shape_slice| {
            target_tensor.with_data(|data_slice| {
                // Assuming self.backend().sqrt() takes &[f32] and returns Vec<f32>
                let result_data = self.backend().sqrt(data_slice);
                Tensor::from_vec(result_data, shape_slice)
            })
        })
            .map(|t| vec![t])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _targets: &[Tensor], _grad: Tensor) -> MlResult<Vec<Tensor>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}