use super::*;

impl Function for Identity {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Identity)
    }

    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let x = targets[0];
        Ok(vec![GlobalTensor::from_vec(x.data().to_vec(), x.shape())?])
    }

    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, _targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        // dy/dx = 1, grad passes through unchanged
        Ok(vec![GlobalTensor::from_vec(grad.data().to_vec(), grad.shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
}
