use crate::tensor::TENSOR_ALLOCATOR;

use super::*;

impl ReLULayer {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            operator: ReLU::new().unwrap()
        }
    }
}

impl Layer for ReLULayer {
    #[cfg(feature = "enableBackpropagation")]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let output = self.operator.forward(&[input.tensor()])?.remove(0);
        let var_act = var_act!(output.to_id(true)?, self.label());
        var_act.with_grad_fn(self.operator.name(), &[&input]);
        Ok(var_act)
    }
    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        TENSOR_ALLOCATOR.with_borrow( |alloc| {
            let output = self.operator.forward(&[input])?.remove(0);
            Ok(alloc.get_tensor_ref(&output.id()).unwrap().clone())
        })
    }
    fn params(&self) -> Vec<&dyn Parameter> { vec![] }
    fn label(&self) -> &str { &self.label }
}

impl Function for ReLU {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(ReLU)
    }

    fn forward(&self, x: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        // ReLU(x) = max(0, x)
        let result = x[0].data().iter()
            .map(|&val| if val > 0.0 { val } else { 0.0 })
            .collect::<Vec<f32>>();

        Ok(vec![PooledTensor::from_vec(result, x[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, target: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        let relu_output = target[0];

        // ∂L/∂x = ∂L/∂y * ∂y/∂x = grad * mask
        Ok(vec![
            PooledTensor::from_vec(
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
    
    fn node_id(&self) -> &HandleId { &self.node_id }
}