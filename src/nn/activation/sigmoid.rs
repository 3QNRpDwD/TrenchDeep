
use super::*;

impl Function for Sigmoid {
    fn new() -> MlResult<Self> { Ok(Sigmoid { backend: Arc::new(CpuBackend::new()?) }) }

    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        let x = targets[0];

        let result = x.with_data(|data| {
            let ones = vec![1.0f32; data.len()];
            self.backend.div(&ones, &self.backend.add(&ones, &self.backend.exp(data)))
        });

        x.with_shape(|shape| {
            Ok(vec![Tensor::from_vec(result, shape)?])
        })
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[Tensor], grad: Tensor) -> MlResult<Vec<Tensor>> {
        let sigmoid_output = targets[0];
        // σ'(x) = σ(x) * (1 - σ(x))
        // ∂L/∂x = ∂L/∂y * ∂y/∂x = grad * σ'(x)

        let result = grad.with_data(|grad_data| {
            sigmoid_output.with_data(|sigmoid_data| {
                let ones = vec![1.0f32; sigmoid_data.len()];
                self.backend.multiply(
                    grad_data,
                    &self.backend.multiply(
                        sigmoid_data,
                        &self.backend.sub(
                            &ones,
                            sigmoid_data
                        )
                    )
                )
            })
        });

        grad.with_shape(|shape| {
            Ok(vec![Tensor::from_vec(result, shape)?])
        })
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}
