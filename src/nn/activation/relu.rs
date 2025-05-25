use super::*;

impl Function for Relu {
    fn new() -> MlResult<Self> { Ok(Relu { backend: Arc::new(CpuBackend::new()?) }) }

    fn forward(&self, x: &[Tensor]) -> MlResult<Vec<Tensor>> {
        // ReLU(x) = max(0, x)
        let result = x[0].data().iter()
            .map(|&val| if val > 0.0 { val } else { 0.0 })
            .collect::<Vec<f32>>();

        Ok(vec![Tensor::from_vec(result, x[0].shape().as_slice())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, target: &[Tensor], grad: Tensor) -> MlResult<Vec<Tensor>> {
        let relu_output = target[0];

        // ∂L/∂x = ∂L/∂y * ∂y/∂x = grad * mask
        Ok(vec![
            Tensor::from_vec(
                self.backend.multiply(
                    &grad.data(),
                    &relu_output.data().iter()
                        .map(|&val| if val > 0.0 { 1.0 } else { 0.0 })
                        .collect::<Vec<f32>>()
                ),
                grad.shape().as_slice()
            )?
        ])
    }
}