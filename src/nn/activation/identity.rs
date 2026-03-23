use super::*;

impl IdentityLayer {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            operator: Identity::new().unwrap(),
        }
    }
}

impl Layer for IdentityLayer {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        self.operator.apply(&[input])
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        Ok(self.operator.forward(&[input])?.remove(0))
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        vec![]
    }

    fn label(&self) -> &str {
        &self.label
    }
}

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

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &NodeId {
        &self.node_id
    }
}
