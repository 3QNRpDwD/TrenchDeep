use super::*;

impl Function for TanhOp {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(TanhOp)
    }

    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let x = targets[0];

        Ok(vec![
            GlobalTensor::from_vec(
                x.data().iter().map(|&value| value.tanh()).collect(),
                x.shape()
            )?
        ])
    }

    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        // The graph supplies the original input, so recompute y = tanh(x).
        let outputs = self.forward(targets)?;
        let tanh_output = &outputs[0];
        let ones = vec![1.0f32; tanh_output.data().len()];

        Ok(vec![
            GlobalTensor::from_vec(
                self.backend.multiply(
                    &grad.data(),
                    &self.backend.sub(
                        &ones,
                        &self.backend.multiply(&tanh_output.data(), &tanh_output.data())
                    )
                ),
                grad.shape()
            )?
        ])
    }

    fn node_id(&self) -> &NodeId { &self.node_id }
}
