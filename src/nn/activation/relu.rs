
use super::*;

impl Function for Relu {
    fn new() -> MlResult<Self> { Ok(Relu { backend: Arc::new(CpuBackend::new()?) }) }

    fn forward(&self, x: &[Tensor]) -> MlResult<Vec<Tensor>> {
        // ReLU(x) = max(0, x)
        let result = x[0].with_data(|data| {
            data.iter().map(|&val| if val > 0.0 { val } else { 0.0 }).collect::<Vec<f32>>()
        });

        x[0].with_shape(|shape| {
            Ok(vec![Tensor::from_vec(result, shape)?])
        })
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, target: &[Tensor], grad: Tensor) -> MlResult<Vec<Tensor>> {
        let relu_output = target[0];

        // ∂L/∂x = ∂L/∂y * ∂y/∂x = grad * mask
        let result = grad.with_data(|grad_data| {
            relu_output.with_data(|output_data| {
                self.backend.multiply(
                    grad_data,
                    &output_data.iter()
                        .map(|&val| if val > 0.0 { 1.0 } else { 0.0 })
                        .collect::<Vec<f32>>()
                )
            })
        });

        grad.with_shape(|shape| {
            Ok(vec![Tensor::from_vec(result, shape)?])
        })
    }
}