use super::*;

impl Function for ReLU {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(ReLU)
    }

    fn forward(&self, x: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let result = x[0].data().iter()
            .map(|&val| if val > 0.0 { val } else { 0.0 })
            .collect::<Vec<f32>>();

        Ok(vec![GlobalTensor::from_vec(result, x[0].shape())?])
    }

    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, target: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let relu_output = target[0];

        Ok(vec![
            GlobalTensor::from_vec(
                self.backend.multiply(
                    &grad.data(),
                    &relu_output.data().iter()
                        .map(|&val| if val > 0.0 { 1.0 } else { 0.0 })
                        .collect::<Vec<f32>>()
                ),
                grad.shape()
            )?
        ])
    }

    fn node_id(&self) -> &NodeId { &self.node_id }
}
