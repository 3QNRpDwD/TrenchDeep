use super::*;

impl Function for Tanh {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Tanh)
    }

    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let x = targets[0];
        let pos_exp = self.backend.exp(&x.data());
        let neg_exp = self.backend.exp(&x.data().iter().map(|&val| -val).collect::<Vec<f32>>());

        // tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        Ok(vec![
            GlobalTensor::from_vec(
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
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let tanh_output = targets[0];
        let ones = vec![1.0f32; tanh_output.data().len()];

        // ∂L/∂x = ∂L/∂y * ∂y/∂x = grad * (1 - tanh^2(x))
        Ok(vec![
            GlobalTensor::from_vec(
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
    
    fn node_id(&self) -> &NodeId { &self.node_id }
}