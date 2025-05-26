use super::*;

impl Function for Tanh {
    fn new() -> MlResult<Self> { Ok(Tanh { backend: Arc::new(CpuBackend::new()?) }) }

    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        let x = targets[0];

        let result = x.with_data(|data| {
            let pos_exp = self.backend.exp(data);
            let neg_exp = self.backend.exp(&data.iter().map(|&val| -val).collect::<Vec<f32>>());

            // tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
            self.backend.div(
                &self.backend.sub(
                    &pos_exp,
                    &neg_exp
                ),
                &self.backend.add(
                    &pos_exp,
                    &neg_exp
                )
            )
        });

        x.with_shape(|shape| {
            Ok(vec![Tensor::from_vec(result, shape)?])
        })
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[Tensor], grad: Tensor) -> MlResult<Vec<Tensor>> {
        let tanh_output = targets[0];

        let result = grad.with_data(|grad_data| {
            tanh_output.with_data(|tanh_data| {
                let ones = vec![1.0f32; tanh_data.len()];

                // ∂L/∂x = ∂L/∂y * ∂y/∂x = grad * (1 - tanh^2(x))
                self.backend.multiply(
                    grad_data,
                    &self.backend.sub(
                        &ones,
                        &self.backend.multiply(
                            tanh_data,
                            tanh_data
                        )
                    )
                )
            })
        });

        grad.with_shape(|shape| {
            Ok(vec![Tensor::from_vec(result, shape)?])
        })
    }
}