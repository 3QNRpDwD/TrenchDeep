use super::*;

impl Function for Tanh {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Tanh)
    }

    fn forward(&self, targets: &[&Tensor]) -> MlResult<Vec<Tensor>> {
        let x = targets[0];
        let pos_exp = self.backend.exp(&x.data());
        let neg_exp = self.backend.exp(&x.data().iter().map(|&val| -val).collect::<Vec<f32>>());

        // tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        Ok(vec![
            Tensor::from_vec(
                self.backend.div(
                    &self.backend.sub(
                        &pos_exp,
                        &neg_exp
                    ),
                    &self.backend.add(
                        &pos_exp,
                        &neg_exp
                    )
                ),
                x.shape()
            )?
        ])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor], grad: &Tensor) -> MlResult<Vec<Tensor>> {
        let tanh_output = targets[0];
        let ones = vec![1.0f32; tanh_output.data().len()];

        // ∂L/∂x = ∂L/∂y * ∂y/∂x = grad * (1 - tanh^2(x))
        Ok(vec![
            Tensor::from_vec(
                self.backend.multiply(
                    &grad.data(),
                    &self.backend.sub(
                        &ones,
                        &self.backend.multiply(
                            &tanh_output.data(),
                            &tanh_output.data()
                        )
                    )
                ),
                grad.shape()
            )?
        ])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn node_id(&self) -> &NodeId { &self.node_id }
}